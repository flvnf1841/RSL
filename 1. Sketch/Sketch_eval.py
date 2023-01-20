import torch
import numpy as np

from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from sklearn.metrics import classification_report

from lee_sketch_utils import load_and_cache_examples
from lee_loss import cal_loss

import wandb


# def evaluate(args, model, tokenizer, pad_token_label_id, logger, mode, prefix=""):
#     eval_dataset = load_and_cache_examples(args,
#                                            tokenizer,
#                                            pad_token_label_id,
#                                            mode=mode,
#                                            logger=logger)
#
#     args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
#     # Note that DistributedSampler samples randomly
#     eval_sampler = SequentialSampler(
#         eval_dataset) if args.local_rank == -1 else DistributedSampler(
#             eval_dataset)
#     eval_dataloader = DataLoader(eval_dataset,
#                                  sampler=eval_sampler,
#                                  batch_size=args.eval_batch_size)
#
#     # multi-gpu evaluate
#     if args.n_gpu > 1:
#         model = torch.nn.DataParallel(model)
#
#     # Eval!
#     logger.info("***** Running evaluation %s *****", prefix)
#     logger.info("  Num examples = %d", len(eval_dataset))
#     logger.info("  Batch size = %d", args.eval_batch_size)
#     eval_loss = 0.0
#     nb_eval_steps = 0
#     preds = None
#     out_label_ids = None
#     model.eval()
#     for batch in tqdm(eval_dataloader, desc="Evaluating"):
#         batch = tuple(t.to(args.device) for t in batch)
#
#         with torch.no_grad():
#
#             inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
#             if args.model_type != "distilbert":
#                 inputs["token_type_ids"] = (
#                     batch[2] if args.model_type in ["bert", "xlnet"] else None
#                 )  # XLM and RoBERTa don"t use segment_ids
#             outputs = model(**inputs)
#             labels = batch[3]
#             logits = outputs[0]
#             tmp_eval_loss = cal_loss(labels=labels,
#                                      logits=logits,
#                                      attention_mask=inputs['attention_mask'],
#                                      uni_w=args.uni_w,
#                                      ske_w=args.ske_w)
#
#             if args.n_gpu > 1:
#                 tmp_eval_loss = tmp_eval_loss.mean(
#                 )  # mean() to average on multi-gpu parallel evaluating
#
#             eval_loss += tmp_eval_loss.item()
#         nb_eval_steps += 1
#         if preds is None:
#             preds = logits.detach().cpu().numpy()
#             out_label_ids = labels.detach().cpu().numpy()
#         else:
#             preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
#             out_label_ids = np.append(out_label_ids,
#                                       labels.detach().cpu().numpy(),
#                                       axis=0)
#
#     eval_loss = eval_loss / nb_eval_steps
#     preds = np.argmax(preds, axis=2)
#
#     out_label_list = [[] for _ in range(out_label_ids.shape[0])]
#     preds_list = [[] for _ in range(out_label_ids.shape[0])]
#
#     out_label_ = []
#     preds_ = []
#     for i in range(out_label_ids.shape[0]):
#         for j in range(out_label_ids.shape[1]):
#             if out_label_ids[i, j] != pad_token_label_id:
#                 preds_list[i].append(preds[i][j])
#                 out_label_.append(out_label_ids[i][j])
#                 preds_.append(preds[i][j])
#     report = classification_report(out_label_, preds_, output_dict=True)
#     scores_0 = report['0']
#     scores_1 = report['1']
#     results = {
#         "eval_loss": eval_loss,
#         "report": "\n" + classification_report(out_label_, preds_),
#         "precision_0": scores_0['precision'],
#         "recall_0": scores_0['recall'],
#         "f1_0": scores_0['f1-score']
#     }
#
#     logger.info("***** Eval results %s *****", prefix)
#     for key in sorted(results.keys()):
#         logger.info("  %s = %s", key, str(results[key]))
#
#     return results, preds_list

#
#
def evaluate(args, model, tokenizer, pad_token_label_id, logger, mode, prefix=""):
    #logger를 글로벌 변수로 쓰기 때문에 인자로 받아옴(원본 소스를 변경)
    eval_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode=mode, logger=logger)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():

            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = ( batch[2] if args.model_type in ["bert", "xlnet"] else None)
                # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            labels = batch[3]
            logits = outputs[0]
            tmp_eval_loss = cal_loss(labels=labels, logits=logits, attention_mask=inputs['attention_mask'],
                                     uni_w=args.uni_w, ske_w=args.ske_w)

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    out_label_ = []
    preds_ = []
    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                preds_list[i].append(preds[i][j])
                out_label_.append(out_label_ids[i][j])
                preds_.append(preds[i][j])
    report = classification_report(out_label_, preds_, output_dict=True)
    scores_0 = report['0']
    scores_1 = report['1']
    results = {
        "eval_loss": eval_loss,
        "report": "\n" + classification_report(out_label_, preds_),
        "precision_0": scores_0['precision'],
        "recall_0": scores_0['recall'],
        "f1_0": scores_0['f1-score']
    }

    if not (args.do_predict_on_test or args.do_eval):
        wandb.log({"eval_loss": results['eval_loss'], "precision_0": results['precision_0'], "recall_0": results['recall_0'], "f1_0": results['f1_0']})

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
        # wandb.log({f"eval_{key}": results[key]}, commit=False)

    return results, preds_list