import os
import wandb
import glob
import logging
import random
import numpy as np
import torch
import json

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import transformers
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

from lee_args import parse_args
from lee_train import trainer
from lee_sketch_utils import set_seed, create_logger, load_and_cache_examples, delete_model
from lee_evaluation import evaluate

SPECIAL_TOKENS = ["<premise>", "<raw>", "<cf>", "<none>"]
ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ['<premise>', '<raw>', '<cf>', '<none>']}
TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

def main(args):
    # Set seed 42
    set_seed(args.seed)
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else: # Initialize the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu=1
    args.device = device

    if args.do_train:
        filename = args.log_path + args.unique_flag + "_persona_sketch.log"
    elif args.do_predict_on_test:
        filename = args.log_path + args.unique_flag + "_persona_sketch_predict.log"
    else:
        filename = args.log_path + args.unique_flag + "_persona_eval.log"
    logger = create_logger(args, filename)
    args.output_dir = args.output_dir + args.unique_flag

    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome."
            .format(args.output_dir))

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        #args.fp16,
    )

    if not (args.do_predict_on_test or args.do_eval):
        if args.wandb:
            if args.name:
                wandb.init(project='persona', config=vars(args), name=args.name)
            else:
                wandb.init(project='persona', config=vars(args))

    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier() # Make sure only the first process in distributed training will download model&vocab

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained( args.config_name if args.config_name else args.model_name_or_path, )
    tokenizer_args = {
        k: v for k,v in vars(args).items() if v is not None and k in TOKENIZER_ARGS
    }
    logger.info("Tokenizer arguments: %s", tokenizer_args)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                              **tokenizer_args)
    model = AutoModelForTokenClassification.from_pretrained(args.model_name_or_path, config=config,)
    orig_num_tokens = tokenizer.vocab_size
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)

    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens = orig_num_tokens + num_added_tokens)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = trainer(args, model, tokenizer, pad_token_label_id, logger=logger)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            # Create output directory if needed
            if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(args.output_dir)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))


    # EVAL
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir,
                                                  **tokenizer_args)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(
                    glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME,
                              recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                "-")[-1] if len(checkpoints) > 1 else ""
            model = AutoModelForTokenClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result, _ = evaluate(args,
                                 model,
                                 tokenizer,
                                 pad_token_label_id,
                                 logger=logger,
                                 mode="exp_dev",
                                 prefix=global_step)
            if global_step:
                result = {
                    "{}_{}".format(global_step, k): v
                    for k, v in result.items()
                }
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(results.keys()):
                writer.write("{} = {}\n".format(key, str(results[key])))



    # Inference
    if args.do_predict_on_test and args.local_rank in [-1, 0]:
        if not os.path.exists(args.pred_dir_path):
            os.makedirs(args.pred_dir_path)

        if args.unique_flag == "8020":
            args.output_dir = "lee/sketch_model/8020/checkpoint-14000/"
        if args.unique_flag == "5050":
            args.output_dir = "lee/sketch_model/5050/checkpoint-16000/"
        if args.unique_flag == "2080":
            args.output_dir = "lee/sketch_model/2080/checkpoint-16000/"
        if args.unique_flag == "aug8020":
            args.output_dir = "lee/sketch_model/aug8020/checkpoint-3000/"
        if args.unique_flag == "350_8020":
            args.output_dir = "lee/sketch_model/350_8020/checkpoint-7000/"
        if args.unique_flag == "300_8020":
            args.output_dir = "lee/sketch_model/300_8020/checkpoint-7000/"
        if args.unique_flag == "exp_300_8020":
            args.output_dir = "lee/sketch_model/exp_300_8020/checkpoint-7000/"
        # '''
        # # weights for causal words and background words are (0.8,0.2)
        # python sketch_main.py --do_predict_on_test\
        #     --do_lower_case\
        #     --unique_flag 8020\
        #     --uni_w 0.8\
        #     --ske_w 0.2
        # '''

        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
        model = AutoModelForTokenClassification.from_pretrained(args.output_dir)
        model.to(args.device)

        result, predictions = evaluate(args,
                                       model,
                                       tokenizer,
                                       pad_token_label_id,
                                       logger=logger,
                                       mode="test_wo_exp")
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "test_wo_exp_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))

        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir,
                                                    "test_wo_exp_predictions.txt")
        output_test_predictions_json = os.path.join(
            args.pred_dir_path,
            args.unique_flag + "_test_wo_exp_skeletons_predictions.json")

        f = open("lee/data/test_skeletons_wo_exp.json", "r")
        all_data = json.load(f)

        all_pred = []
        g = open(output_test_predictions_json, "w")

        with open(output_test_predictions_file, "w") as writer:
            i = 0
            for item in all_data:
                pre = item['history']
                con = item['persona']
                #c_con = item['exp_persona']
                ske = item['raw_skeleton_gold'][0]
                end = item['gold']
                end_words = end.strip().split()
                words_raw = ske.strip().split()
                labels_raw = item['label_raw'][1][1:]
                #labels_cf = item['label_cf'][1][1:]

                assert len(labels_raw) == len(predictions[i])
                pred_ske_raw = ""
                for word, pred in zip(end_words, predictions[i]):
                    if pred == 1:
                        pred_ske_raw = (pred_ske_raw + " " + word)
                    else:
                        # append "__" and merge the consecutive blanks into one blank
                        if not pred_ske_raw.endswith(" __ "):
                            pred_ske_raw = pred_ske_raw + " __ "
                pred_ske_raw = pred_ske_raw.strip()

                words_raw = " ".join([w for w in words_raw])
                labels_raw = " ".join([str(w) for w in labels_raw])
                preds_raw = " ".join([str(w) for w in predictions[i]])
                i += 1 # 추가해줌. predictions[i]를 labels_raw에 맞춰서 해줘야 길이가 서로 맞음

                # assert len(labels_cf) == len(predictions[i])
                # pred_ske_cf = ""
                # for word, pred in zip(end_words_cf, predictions[i]):
                #     if pred == 1:
                #         pred_ske_cf = (pred_ske_cf + " " + word)
                #     else:
                #         # append "__" and merge the consecutive blanks into one blank
                #         if not pred_ske_cf.endswith(" __ "):
                #             pred_ske_cf = pred_ske_cf + " __ "
                # pred_ske_cf = pred_ske_cf.strip()

                # words_cf = " ".join([w for w in words_cf])
                # labels_cf = " ".join([str(w) for w in labels_cf])
                # preds_cf = " ".join([str(w) for w in predictions[i]])
                # i += 1

                writer.write(words_raw + "\n")
                writer.write(labels_raw + "\n")
                writer.write(preds_raw + "\n")

                # writer.write(words_cf + "\n")
                # writer.write(labels_cf + "\n")
                # writer.write(preds_cf + "\n")

                res = {}
                res['history'] = pre
                res['persona'] = con
                #res['exp_persona'] = c_con
                res['gold'] = end
                res['gt_sketch_gold'] = ske
                res['pred_sketch_gold'] = [pred_ske_raw]

                # pred for next customize gpt2 preprocess
                # res['raw_skeletons_endings'] = [pred_ske_cf]
                # res['counterfactual_condition'] = c_con
                # res['c_ending'] = c_end
                # res['gt_counterfactual_skeletons_ending'] = c_ske
                # pred for next customize gpt2 preprocess
                # res['counterfactual_skeletons_endings'] = [pred_ske_raw]
                all_pred.append(res)

        json.dump(all_pred, g)
        return

'''
--do_train
--evaluate_during_training
--do_lower_case
--unique_flag 300_8020
--uni_w 0.8
--ske_w 0.2
'''
'''
--do_predict_on_test
--do_lower_case
--unique_flag 300_8020
--uni_w 0.8
--ske_w 0.2
'''


# python sketch_main.py --do_predict_on_test\
#     --do_lower_case\
#     --unique_flag 8020\
#     --uni_w 0.8\
#     --ske_w 0.2

if __name__ == "__main__":
    args = parse_args(mode='train')
    main(args)