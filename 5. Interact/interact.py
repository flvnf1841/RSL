# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import json
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
import torch
import torch.nn.functional as F
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                  GPT2DoubleHeadsModel, GPT2Tokenizer, GPT2LMHeadModel, WEIGHTS_NAME, CONFIG_NAME)
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model, make_logdir

logger = logging.getLogger(__file__)
def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="perturb3_final_optimized_final.json", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache_perturb3_final_optimized_final', help="Path or url of the dataset cache")
    parser.add_argument("--model_checkpoint", type=str, default="./runs/top3_gpt2/", help="Path, url or short name of the model")
    # parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path, url or short name of the model")
    parser.add_argument("--num_candidates", type=int, default=1, help="Number of candidates for training")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous exchanges to keep in history")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Accumulate gradients on several steps")
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--lm_coef", type=float, default=1.0, help="LM loss coefficient")
    parser.add_argument("--mc_coef", type=float, default=1.0, help="Multiple-choice loss coefficient")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=0, help="Number of training epochs")
    parser.add_argument("--personality_permutations", type=int, default=1, help="Number of permutations of personality sentences")
    parser.add_argument("--eval_before_start", action='store_true', help="If true start with a first evaluation before training")
    parser.add_argument("--fp16", type=str, default="", help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    
    # parser.add_argument("--model", type=str, default="gpt2", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=40, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process. logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if -1 in [-1, 0] else logging.WARN)
    logger.warning("Running process %d", -1)  # This is a logger.warning: it will be printed by all distributed processes
    logger.info("Arguments: %s", pformat(args))

    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)

    logger.info("Prepare tokenizer, pretrained model and optimizer.")
    tokenizer_class = GPT2Tokenizer if "gpt2" in args.model_checkpoint else OpenAIGPTTokenizer # cant use Autotokenizer because checkpoint could be a Path
    tokenizer = tokenizer_class.from_pretrained('gpt2')


    model_class = GPT2LMHeadModel if "gpt2" in args.model_checkpoint else OpenAIGPTDoubleHeadsModel
    model = model_class.from_pretrained(args.model_checkpoint)

    # if args.model_checkpoint == "":
    #     if args.model == 'gpt2':
    #         raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
    #     else:
    #         args.model_checkpoint = download_pretrained_model()
	

    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    logger.info("Sample a personality")
    dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
    # pc_valid = dataset["valid"]
    
    # counter = 0
    # with open('personachat_self_original_final2.json') as f:
    #     pc = json.load(f)
    #     pc_valid2 = pc["valid"]

    # tempjson = {'samples':[]}

    # for idx1, item1 in enumerate(pc_valid):

    #     for idx2 in range(len(item1['utterances'])):
    #         if pc_valid2[idx1]['utterances'][idx2]['skeleton'] != '<none>':
    #             personality = item1['personality']
                
                
    #             # print(pc_valid2[idx1]['utterances'][idx2])
    #             # print(item1['utterances'][idx2])
    #             # exp_temp = item1['coment_annotation'][0]['comet']
    #             # exp_temp = item1['coment_annotation']

    #             exp_personality = []
    #             # exp_temp = item1['exp_personality']
    #             # print(exp_temp)
    #             for i in range(0, len(personality)):
    #             #     # exp_temp = item1['coment_annotation'][i]['comet']
    #             #     # exp_personality+=exp_temp['oEffect']['beams']
    #             #     # exp_personality+=exp_temp['oReact']['beams']
    #             #     # exp_personality+=exp_temp['oWant']['beams']
    #             #     # exp_personality+=exp_temp['xAttr']['beams']
    #             #     # exp_personality+=exp_temp['xEffect']['beams']
    #             #     # exp_personality+=exp_temp['xIntent']['beams']
    #             #     # exp_personality+=exp_temp['xNeed']['beams']
    #             #     # exp_personality+=exp_temp['xReact']['beams']
    #             #     # exp_personality+=exp_temp['xWant']['beams']
                
    #                 exp_temp = item1['exp_personality'][i]
                
                    

    #                 for j in range(0,len(exp_temp['xattr'])):
    #                     if 23108 not in exp_temp['xattr'][j]:
    #                         exp_personality.append(exp_temp['xattr'][j])
                            
    #                 for j in range(0,len(exp_temp['xeffect'])):
    #                     if 23108 not in exp_temp['xeffect'][j]:
    #                         exp_personality.append(exp_temp['xeffect'][j])
                            
    #                 for j in range(0,len(exp_temp['xintent'])):
    #                     if 23108 not in exp_temp['xintent'][j]:
    #                         exp_personality.append(exp_temp['xintent'][j])
                            
    #                 for j in range(0,len(exp_temp['xneed'])):
    #                     if 23108 not in exp_temp['xneed'][j]:
    #                         exp_personality.append(exp_temp['xneed'][j])
                            
    #                 for j in range(0,len(exp_temp['xwant'])):
    #                     if 23108 not in exp_temp['xwant'][j]:
    #                         exp_personality.append(exp_temp['xwant'][j])
                            
    #                 for j in range(0,len(exp_temp['xreact'])):
    #                     if 23108 not in exp_temp['xreact'][j]:
    #                         exp_personality.append(exp_temp['xreact'][j])


    #             temp_h = item1['utterances'][idx2]['history'][-3:]
    #             gold = item1['utterances'][idx2]['candidates'][-1]
    #             new_gold = item1['utterances'][idx2]['aug'][0]['new_responses']
    #             exp_personality = [item1['utterances'][idx2]['aug'][0]['exp_persona']]
    #             temp={}
    #             temp['sample_idx'] = counter
    #             temp['history'] = []
    #             temp['origin_persona'] = []
    #             temp['exp_persona'] = []
    #             for ii in temp_h:
    #                 temp['history'].append(tokenizer.decode(ii))
    #             # for ii in personality:
    #             #     temp['origin_persona'].append(tokenizer.decode(ii))
    #             for ii in exp_personality:
    #                 temp['exp_persona'].append(tokenizer.decode(ii))
                
    #             with torch.no_grad():
    #                 # print(history)
    #                 # print(personality)
    #                 out_ids = sample_sequence(personality+exp_personality, temp_h, tokenizer, model, args)
    #             out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    #             # out_text = tokenizer.decode(gold)
    #             out_text = tokenizer.decode(new_gold)
    #             temp['generated_response'] = out_text
    #             counter+=1
    #             tempjson['samples'].append(temp)

                



    #             if counter == 50:
    #                 break
    #     if counter == 50:
    #                 break  

    # with open('human_rating_new_gold.json', 'w') as f:
    #     json.dump(tempjson,f)    

    # personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
    # personality = random.choice(personalities)
    # logger.info("Selected personality: %s", tokenizer.decode(chain(*personality)))
    pc_valid = dataset["valid"]
    for idx1, item1 in enumerate(pc_valid):
        # print(item1['personality'])
        # print(item1['exp_personality'])
        # print(item1['coment_annotation'])
        if idx1 ==0:
            personality = item1['personality']
            exp_personality1 = []
            exp_personality2 = []
            for i in range(0, len(personality)):
                exp_temp = item1['coment_annotation'][i]['comet']
                exp_personality1+=exp_temp['oEffect']['beams']
                exp_personality1+=exp_temp['oReact']['beams']
                exp_personality1+=exp_temp['oWant']['beams']
                exp_personality1+=exp_temp['xAttr']['beams']
                exp_personality1+=exp_temp['xEffect']['beams']
                exp_personality1+=exp_temp['xIntent']['beams']
                exp_personality1+=exp_temp['xNeed']['beams']
                exp_personality1+=exp_temp['xReact']['beams']
                exp_personality1+=exp_temp['xWant']['beams']
            for i in range(0, len(personality)):
                exp_temp = item1['exp_personality'][i]
                for j in range(0,len(exp_temp['xattr'])):
                    if 23108 not in exp_temp['xattr'][j]:
                        exp_personality2.append(exp_temp['xattr'][j])
                                
                for j in range(0,len(exp_temp['xeffect'])):
                    if 23108 not in exp_temp['xeffect'][j]:
                        exp_personality2.append(exp_temp['xeffect'][j])
                                
                for j in range(0,len(exp_temp['xintent'])):
                    if 23108 not in exp_temp['xintent'][j]:
                        exp_personality2.append(exp_temp['xintent'][j])
                                
                for j in range(0,len(exp_temp['xneed'])):
                    if 23108 not in exp_temp['xneed'][j]:
                        exp_personality2.append(exp_temp['xneed'][j])
                                
                for j in range(0,len(exp_temp['xwant'])):
                    if 23108 not in exp_temp['xwant'][j]:
                        exp_personality2.append(exp_temp['xwant'][j])
                                
                for j in range(0,len(exp_temp['xreact'])):
                    if 23108 not in exp_temp['xreact'][j]:
                        exp_personality2.append(exp_temp['xreact'][j])
# ['the holidays make me depressed .', "i've rainbow hair .", 'my age is too old to say .', "i'm an animal activist .", 'i spend my time bird watching with my cats .']
            # personality = [tokenizer.encode('i spend my time bird watching with my cats .'),tokenizer.encode('i am an animal activist .'),tokenizer.encode("i've rainbow hair ."), tokenizer.encode('my age is too old to say .'),
            # tokenizer.encode('the holidays make me depressed .')]
            personality=[tokenizer.encode('i already have a step children .')]
            personality=[tokenizer.encode('i have a children and a dogs .'), tokenizer.encode('i work in it and have been at the same company for 15 years .'), tokenizer.encode('i am a male .'), 
            tokenizer.encode('i enjoy american sports .')]
            personality=[]
            for i1 in personality:
                print(tokenizer.decode(i1))
            print()
            for i1 in exp_personality2:
                print(tokenizer.decode(i1))
            print(len(personality))
            print(len(exp_personality2))
            break


    history = []
    while True:
        raw_text = input(">>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input(">>> ")
        history.append(tokenizer.encode(raw_text))
        history = history[-1:]
        with torch.no_grad():
            # print(history)
            # print(personality)
            out_ids = sample_sequence(personality, history, tokenizer, model, args)
        # history.append(out_ids)
        # history = history[-(2*args.max_history+1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print(out_text)


if __name__ == "__main__":
    run()