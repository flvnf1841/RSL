#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 논문 읽기
# BERT 입히기 (입력이 안들어감)
# BERT 재학습(토크나이저 때문에)

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95
Example command with discriminator:
python examples/run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel, GPT2DoubleHeadsModel
import tempfile
import tarfile
from transformers import RobertaModel, RobertaTokenizer

from pplm_classification_head import ClassificationHead

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

BAG_OF_WORDS_ARCHIVE_MAP = {
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

DISCRIMINATOR_MODELS_PARAMS = {
    # 낚시 기사 / 그렇지 않은 기사
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-large")
        self.pre_classifier = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(1024, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # hidden_state = output_1[0]
        # pooler = hidden_state[:, 0]
        pooler = input_ids
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_past(
        past,
        model,
        last,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        one_hot_bows_vectors=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR
):
    # Generate inital perturbed past
    # gradient 구하기 위한 초기화 과정
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]


    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            # print("Iteration ", i + 1)
            pass
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        all_logits, _, all_hidden = model(last, past=perturbed_past)
        hidden = all_hidden[-1]

        # 결국 여기
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss = -torch.log(torch.sum(bow_logits))
                loss += bow_loss
                loss_list.append(bow_loss)
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_bow_loss:", loss.data.cpu().numpy())

        if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
            ce_loss = torch.nn.CrossEntropyLoss()
            # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings()


            for _ in range(horizon_length):

                # 결국 여기
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                _, curr_unpert_past, curr_all_hidden = model(
                    past=curr_unpert_past,
                    inputs_embeds=inputs_embeds
                )
                curr_hidden = curr_all_hidden[-1]


                new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                    curr_hidden, dim=1)

            # 결국 이전 히든 스테이드들 존나 넣는 거 -> 임베딩 레이어 지날 수 가 없음 애초에
            # 임베딩 레이어 다음부터 적용되면 몰라도 -> 그거 뜯는게 일단 말이 안됨
            # 기존 = 1,1024
            # pc-gpt2 = 1,768
            # print((new_accumulated_hidden / curr_length + 1 + horizon_length).shape)
            # print((new_accumulated_hidden / curr_length + 1 + horizon_length))
            # ss

            prediction = classifier(new_accumulated_hidden /
                                    (curr_length + 1 + horizon_length))
            mask_bert = torch.ones(1,1024).to(device, dtype = torch.long)
            token_type_ids_bert = torch.zeros(1,1024).to(device, dtype = torch.long)


            # 원래 BERT 인풋은 정수만..
            # prediction = classifier(new_accumulated_hidden /
            #                         (curr_length + 1 + horizon_length), mask_bert, token_type_ids_bert)

            label = torch.tensor(prediction.shape[0] * [class_label],
                                 device=device,
                                 dtype=torch.long)
            discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERY_VERBOSE:
                # print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
                pass
            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                # print(' kl_loss', kl_loss.data.cpu().numpy())
                pass
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            # print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())
            pass

        # compute gradients
        loss.backward()

        # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients
        grad = [
            -stepsize *
            (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
        name: Optional[str],
        class_label: Union[str, int],
        device: str,
        verbosity_level: int = REGULAR
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer) -> \
        List[List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
        else:
            filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        bow_indices.append(
            [tokenizer.encode(word.strip(),
                              add_prefix_space=True,
                              add_special_tokens=False)
             for word in words])
    return bow_indices


def build_bows_one_hot_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors


def full_text_generation(
        model,
        tokenizer,
        context=None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        discrim=None,
        class_label=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        skeleton=None,
        **kwargs
):
    # classifier, class_id = get_classifier(
    #     discrim,
    #     class_label,
    #     device
    # )

    classifier = ClassificationHead(
        class_size=3,
        embed_size=1024
    ).to(device)
    classifier.load_state_dict(
        torch.load('./goodnew50.pt', map_location=device))
    classifier.eval()
    class_id = 0

    # classifier = RobertaClass()
    # classifier = torch.load('./pytorch_roberta_empatheticdialogues_sentiment.pt')
    # class_id = class_label

    bow_indices = []
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer)

    # 둘 다 쓰는 것도 가능
    # word list에 있는 것들이 명시적 언급 가능
    # 관련 있는 + 없는 것들도 언급 가능
    # 페르소나 문장을 conceptnet-based COMET을 이용해서 topk 몇개 등 listup해도 될듯
    # ex) I've rainbow hair to look free -> fashion, design, designer, freedom 등등
    # 여기서는 왜 원본 문장도 함꼐? 어차피 결국 집중하게 되는것은 여기서 퍼져 나온 새로운 단어들이라 같이 넣어도됨.
    # 여기서 목표하는 것은 look free에 직접 언급 보다는 look free에 관련된 단어들을 언급하는 것 그렇게 생각하니 I want to look free만 넣는게 맞을 수도
    # ex) I want to look free -> freedom .. 등등
    if bag_of_words and classifier:
        loss_type = PPLM_BOW_DISCRIM
        if verbosity_level >= REGULAR:
            print("Both PPLM-BoW and PPLM-Discrim are on. "
                  "This is not optimized.")

    elif bag_of_words:
        loss_type = PPLM_BOW
        if verbosity_level >= REGULAR:
            print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        if verbosity_level >= REGULAR:
            print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        verbosity_level=verbosity_level,
        skeleton=skeleton
    )


    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=True,
            bow_indices=bow_indices,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            verbosity_level=verbosity_level,
            skeleton=skeleton
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def generate_text_pplm(
        model,
        tokenizer,
        context=None,
        past=None,
        device="cuda",
        perturb=True,
        bow_indices=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        skeleton=None
):

    if skeleton != None:
        skeleton = tokenizer.encode(skeleton)

    output_so_far = None
    persona_length = 0
    context_length = len(context)

    for item in context:
        persona_length += 1
        if item == 50261:
            # persona_length += 1
            break

    for item in context:
        persona_length += 1
        if item == 50262:
            # persona_length += 1
            break

    if context:
        # print(context)

        token_type_ids = []
        current_push = 50260
        for item in context:
            if item == 50260 or item == 50261:
                if item == 50261:
                    current_push = 50261
                    token_type_ids.append(current_push)
                else:
                    current_push = 50260
                    token_type_ids.append(current_push)
            else:
                token_type_ids.append(current_push)

        # print(token_type_ids)
        # print(tokenizer.decode(context))
        # print(tokenizer.decode(token_type_ids))
        # print(len(context))
        # print(len(token_type_ids))

        context_t = torch.tensor(context, device=device, dtype=torch.long)
        token_type_ids_t = torch.tensor(token_type_ids, device=device, dtype=torch.long)

        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
            token_type_ids_t = token_type_ids_t.unsqueeze(0)

        output_so_far = context_t
        output_so_far_tti = token_type_ids_t


    # collect one hot vectors for bags of words
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer,
                                                      device)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    if verbosity_level >= VERBOSE:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)


    skeleton_index = 0
    for i in range_func:

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:

            last = output_so_far[:, -1:]
            last_tti = output_so_far_tti[:, -1:]

            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1], token_type_ids=output_so_far_tti[:, :-1])




        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far, token_type_ids=output_so_far_tti)


        unpert_last_hidden = unpert_all_hidden[-1]


        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            # 이전꺼(컨디션까지 다 가져감)
            # print(unpert_last_hidden)
            # print(unpert_last_hidden.shape)
            # print(unpert_last_hidden[:, :-1, :].shape)

            a1 = unpert_last_hidden[:,:persona_length,:]
            # print(a1.shape)
            a2 = unpert_last_hidden[:,context_length-1:-1,:]
            # print(a2.shape)

            # accumulated_hidden = unpert_last_hidden[:, :-1, :]

            accumulated_hidden = torch.cat([a1,a2],dim=1)
            # print(accumulated_hidden.shape)
            # ss

            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)


            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    classifier=classifier,
                    class_label=class_label,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                    verbosity_level=verbosity_level
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        pert_logits, past, pert_all_hidden = model(last, token_type_ids=last_tti, past=pert_past)
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)



        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()

            # mask_bert = torch.ones(1,1024).to(device, dtype = torch.long)
            # token_type_ids_bert = torch.zeros(1,1024).to(device, dtype = torch.long)

            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([class_label], device=device,
                                 dtype=torch.long)
            unpert_discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERBOSE:
                # print(
                #     "unperturbed discrim loss",
                #     unpert_discrim_loss.data.cpu().numpy()
                # )
                pass
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST

            pert_probs = top_k_filter(pert_probs, k=top_k,
                                      probs=True)  # + SMALL_CONST


            # rescale
            if torch.sum(pert_probs) <= 1:
                if torch.sum(pert_probs) == 0:
                    pert_probs = pert_probs
                else:

                    # print('ssssssssssssssssss')
                    pert_probs = pert_probs / torch.sum(pert_probs)




        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            # len 50263
            # 어디가 0이 아닐까 맞추는 거, last 토큰 맞추기

            last = torch.multinomial(pert_probs, num_samples=1)


        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token


        if skeleton == None:
            if last[0][0] == 50258:
                break

            output_so_far = (
                last if output_so_far is None
                else torch.cat((output_so_far, last), dim=1)
            )
            output_so_far_tti = (
                last_tti if output_so_far_tti is None
                else torch.cat((output_so_far_tti, last_tti), dim=1)
            )

        else:
            if skeleton_index == len(skeleton):
                break

            # print(last)
            if last[0][0] == 50258 and skeleton_index >= len(skeleton):
                break

            if skeleton[skeleton_index] != 4808 and skeleton[skeleton_index] != 62:
                aa = [[skeleton[skeleton_index]]]
                last = torch.tensor(aa, device=device, dtype=torch.long)
                skeleton_index += 1
            else:
                skeleton_index += 1

            output_so_far = (
                last if output_so_far is None
                else torch.cat((output_so_far, last), dim=1)
            )
            output_so_far_tti = (
                last_tti if output_so_far_tti is None
                else torch.cat((output_so_far_tti, last_tti), dim=1)
                )

        if verbosity_level >= REGULAR:
            # print(tokenizer.decode(output_so_far.tolist()[0]))
            pass

    return output_so_far, unpert_discrim_loss, loss_in_time


def set_generic_model_params(discrim_weights, discrim_meta):
    if discrim_weights is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_weights need to be specified')
    if discrim_meta is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_meta need to be specified')

    with open(discrim_meta, 'r') as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta['path'] = discrim_weights
    DISCRIMINATOR_MODELS_PARAMS['generic'] = meta

def download_pretrained_model():

    HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"

    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir

def add_special_tokens_(model, tokenizer):

    # # 여기
    # SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>", "<ending>"]
    # ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
    #                          'additional_special_tokens': ['<speaker1>', '<speaker2>', '<ending>']}


    # 여기
    SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>", "<skeleton>"]
    ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                             'additional_special_tokens': ['<speaker1>', '<speaker2>', '<skeleton>']}

    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer.encoder)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

def run_pplm_example(
        pretrained_model="gpt2-small",
        cond_text="",
        uncond=False,
        num_samples=1,
        bag_of_words=None,
        discrim=None,
        discrim_weights=None,
        discrim_meta=None,
        class_label=-1,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=0,
        no_cuda=False,
        colorama=False,
        verbosity='regular'
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    if discrim == 'generic':
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model
            if verbosity_level >= REGULAR:
                print("discrim = {}, pretrained_model set "
                "to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model

    # 학습할 때는 더블헤드 인터렉트할 때는 그냥 헤드? 뭔차이지, transfer transfo에서도 그럼
    # model = GPT2LMHeadModel.from_pretrained(
    #     pretrained_model,
    #     output_hidden_states=True
    # )
    # model.to(device)
    # model.eval()
    #
    # # load tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)


    # model_checkpoint = download_pretrained_model()
    # 여기
    # model_checkpoint = './Aug25_21-01-40_iclserver-desktop_gpt2-medium/'
    # model_checkpoint = 'Oct28_15-01-01_iclserver-desktop_gpt2-medium/'
    # model_checkpoint = 'Nov01_12-03-48_iclserver-desktop_gpt2-medium/'
    # model_checkpoint = 'Nov02_14-49-40_iclserver-desktop_gpt2-medium/'
    model_checkpoint = 'Nov03_18-57-32_iclserver-desktop_gpt2-medium/'

    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel)
    # tokenizer_class, model_class = (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    model = model_class.from_pretrained(model_checkpoint, output_hidden_states=True)
    model.to(device)
    model.eval()
    add_special_tokens_(model, tokenizer)

    # print(tokenizer.encode('k  _ the day of _ meeting, dan  _.'))
    # print(tokenizer.encode('k  __ the day of __ meeting, dan  __.'))
    # print(tokenizer.encode('  _ the day of the meeting, dan  _.'))
    # print(tokenizer.encode('  __ the day of the meeting, dan  __.'))
    # print(tokenizer.encode('dan decided to call his banker and  __. dan felt  __ about  __ a very important meeting with his banker.'))

    # print(tokenizer.encode("<bos> <eos> <speaker1> <speaker2> <pad>")) # [50257, 50258, 50260, 50261, 50259]
    # ss

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    if uncond:
        tokenized_cond_text = tokenizer.encode(
            [tokenizer.bos_token],
            add_special_tokens=False
        )
    else:
        raw_text = cond_text
        # raw_text = "the holidays make me depressed. I've rainbow hair. I am an animal activist. my age is too old to say. i spend my time bird watching with my cats. \
        # <speaker2> your kitties like birds? don't they try to snatch them? <speaker1>"


        raw_text = "I believe that mermaids are real. \
        <speaker2> what do you do for living? <speaker1> I am "
        skeleton=None

        # print(tokenizer.encode('. . . .  # 13 764
        # ss
        raw_text = "I think deeply about beauty of nature. \
        <speaker2> I agree, what do you do for living? <speaker1>"
        skeleton="I am a researcher I am researching the fact _ _ _ _ _ _ _ _ _ _"
        # skeleton=None

        raw_text = "I want to decorate the home. \
        <speaker2> I am ! For my hobby I like to do canning or some whittling. <speaker1>"
        skeleton="I also _ _ _ _ _ _ _ _ _ _ _ _ _ _ when I am not out bow hunting ."
        # skeleton=None

        # i read twenty books a year .
        # i rather read , i've read about 20 books this year .
        raw_text = "I learn new things. \
        <speaker2> I just got done watching a horror movie. <speaker1>"
        skeleton="I rather _ , I've _ _ _ _ _ _."
        skeleton=None

        # I take dance lessons once a week.
        # that is funny ! i take dance lessons so i can dance just like them .
        raw_text = "I want to learn a new skill. \
        <speaker2> not really . i think one of them has brown curly hair , though , like me ! <speaker1>"
        # skeleton="that is funny ! _ _ _ _ so i can dance just like them _ ."
        skeleton="that is funny ! I _ _ _ _ _ so i can _  _  _  _ just like them ."
        # skeleton=None

        # i love cooking but i also enjoy fishing .
        # not really into coffee . i love cooking but also enjoy fishing . do you like fishing ?
        # raw_text = "I want to enjoy the outdoors. \
        # <speaker2> no , i like to taste coffee , do you ? <speaker1>"
        # # skeleton="not really into coffee. _ _ _ _ _ _ _ _ do you like fishing?"
        # skeleton=None
        #
        # # i practice vegetarianism .
        # # how about , maintaining a good diet , try being a vegetarian , it helps me .
        # raw_text = "I become more healthy. \
        # <speaker2> i'm not reading this for fun . i'm trying to learn . <speaker1>"
        # skeleton="how about , maintaining a good diet , _ _ _ _ _ _ _ _, it helps me ."
        # skeleton=None

        # i have played since i was 4 years old .
        # never . i have been playing since i was four . it pays more than lifting weights .
        raw_text = "I want to find a new job. \
        <speaker2> no wonder you like watches! i'm commercial electrician. <speaker1>"
        skeleton="wonderful! yeah, however, _ _ _ _ _ _ _ _ _ _ _ _ _."
        # skeleton=None

        # i have played since i was 4 years old .
        # never . i have been playing since i was four . it pays more than lifting weights .
        # raw_text = "I want to play with friends. \
        # <speaker2> i lift . i lift heavy weights . violinist is for wussy ? <speaker1>"
        # skeleton="never . I _ _ _ _ _ _ _ . it pays more than lifting weights ."
        # # skeleton=None
        #
        # # i recently got an apartment with my best friend .
        # # hey ! ! so i have a house warming party for my new apartment with my buddy tomorrow !
        # raw_text = "I has to pay rent. \
        # <speaker2> hi there . how goes it ? <speaker1>"
        # skeleton="hey ! ! so _ have a house warming party for my new  _ _ _ buddy tomorrow ! "
        # # skeleton="hey ! ! so  __ have a house warming party for my new  __  __  __ buddy tomorrow ! "
        # # skeleton=None

        # i love to drink fancy tea .
        # i am doing good . just sipping tea . what do you do for work ?
        # raw_text = "I want to go to tea shop. \
        # <speaker2> hi how are you doing ? i am okay how about you ? <speaker1>"
        # # # skeleton="i am doing good . _ _ _ . what do you do for work ?"
        # skeleton=None
        #
        # # I love cats and have two cats.
        # # I am much more of a cat person actually.
        # raw_text = "I have a cat litter box. \
        # <speaker2> hi! do you like turtles? <speaker1>"
        # # # skeleton="i am doing good . _ _ _ . what do you do for work ?"
        # skeleton=None
        #
        # # i love to drink wine and dance in the moonlight .
        # # that sounds like fun . i like wine and dancing too !
        # raw_text = "I need to go to the bar. \
        # <speaker2> nice . i'm playing some card games with my family . <speaker1>"
        # skeleton = "that sounds like fun . i like _ _ _ _ _ _ _."
        # # skeleton=None
        #
        # # i feel like i might live forever .
        # # awesome ! walking like that you might live forever like me ! !
        # raw_text = "I want to be healthy. \
        # <speaker2> same . i try to get a small workout in a three mile walk for me is key.  <speaker1>"
        # # skeleton = "awesome ! walking like that you _ _ _ like me."
        # skeleton=None

        # 그냥 = consistent x, attr = consistent o
        # raw_text = "i spend my time bird watching with my cats. \
        # <speaker2> do you have pets? <speaker1> I just "
        # skeleton=None

        # 그냥 = consistent x, attr = consistent o
        # raw_text = "i spend my time bird watching with my cats. \
        # <speaker2> do you have pets? <speaker1> I just"
        # skeleton= None

        # 둘 다 consistent o
        # raw_text = "i take dance lessons once a week. \
        # <speaker2> what do you do for fun? <speaker1>"
        # skeleton=None
        #
        # # 그냥 = consistent o, attr = 평가 불가(일반적 대답)
        # raw_text = "i love cooking but i also enjoy fishing. \
        # <speaker2> i like to taste coffee, do you? <speaker1>"
        #
        # # 그냥 = 평가 불가, attr = consistent o
        # raw_text = "my dad taught me everything i know. \
        # <speaker2> i have a family. <speaker1> "


        #skeleton
        # skeleton = "I was _ _ _ _ _ and liked _ _ _ _ _ . <eos>"
        # # print(tokenizer.encode('_ _ _ _ _'))
        # # print(tokenizer.decode([62, 4808, 4808,62]))
        # # ss
        #
        # # 조작용
        # raw_text = "my dad taught me everything i know. \
        # <speaker2> i have a family. <speaker1> "

        # 조작용
        # raw_text = "i spend my time bird watching with my cats. \
        # <speaker2> do you have pets? <speaker1>"
        # I just _ _ _ _ with _ _ .

        # 조작용
        # raw_text = "i spend my time bird watching with my cats because i want to enjoy nature. \
        # <speaker2> do you have pets? <speaker1>"
        # I just _ _ _ _ with _ _ .

        # 조작용
        # raw_text = "i want to enjoy nature. \
        # <speaker2> do you have pets? <speaker1>"
        # I just _ _ _ _ with _ _ .

        # 조작용
        # raw_text = "i want to enjoy nature. \
        # <speaker2> i take dance lessons during break. <speaker1>"
        # I just _ _ _ _ with _ _ .


        # 거의 최종 실험 1006
        # raw_text = "I think deeply about beauty of nature. \
        # <speaker2> I agree, what do you do for living? <speaker1>"
        # skeleton = "I am a researcher I am researching the fact _ _ _ _ _"
        # skeleton=None

        # raw_text = "I become more healthy. \
        # <speaker2> i'm not reading this for fun. i'm trying to learn. <speaker1>"
        # skeleton = "how about, maintaining a good diet, try being a _ _ _ _ _ _, it helps me ."
        # skeleton=None

        # raw_text = "I need to go to the bar. \
        # <speaker2> nice. I'm playing some card games with my family . <speaker1>"
        # skeleton = "that sounds like fun . I like _ _ _ _ _ too !"

        # print(tokenizer.encode('I am a researcher _ am researching the fact _ _ _ _'))
        # print(tokenizer.encode('how about , maintaining a good diet , try being a vegetarian , it helps me _'))
        # print(tokenizer.encode('that sounds like fun . _ like _ _ dancing too !'))
        # print(tokenizer.encode('i rather read ,  _ ‘ve  _ about 20  _ this  _  _ '))
        # [40, 716, 257, 13453, 11593, 716, 24114, 262, 1109, 11593, 11593, 11593, 11593]
        # [4919, 546, 837, 10941, 257, 922, 5496, 837, 1949, 852, 257, 24053, 837, 340, 5419, 502, 4808]
        # [5562, 5238, 588, 1257, 764, 11593, 588, 11593, 11593, 15360, 1165, 5145]
        # [72, 2138, 1100, 837, 1849, 11593, 564, 246, 303, 1849, 11593, 546, 1160, 1849, 11593, 428, 1849, 11593, 1849, 11593]


        raw_text = "I learn new things. \
        <speaker2> I just got done watching a horror movie. <speaker1>"
        skeleton = "i rather _, I ‘ve _ _ _ _ _ _ ."

        raw_texts = [
        "I think deeply about beauty of nature. <speaker2> I agree, what do you do for living? <speaker1> <ending>",
        "I think deeply about beauty of nature. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher __ am researching the fact __ <ending>",
        "I think deeply about beauty of nature. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        "I think deeply about beauty of nature. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        "I think deeply about beauty of nature. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        "I am curious. <speaker2> I agree, what do you do for living? <speaker1> <ending>",
        "I am curious. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher __ am researching the fact __ <ending>",
        "I am curious. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        "I am curious. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        "I am curious. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        "I need to go to the beach. <speaker2> I agree, what do you do for living? <speaker1> <ending>",
        "I need to go to the beach. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher __ am researching the fact __ <ending>",
        "I need to go to the beach. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        "I need to go to the beach. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        "I need to go to the beach. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        "I become more healthy. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> <ending>",
        "I become more healthy. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a vegetarian, it helps me __ <ending>",
        "I become more healthy. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        "I become more healthy. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        "I become more healthy. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me . <ending>",
        "I eat less meat. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> <ending>",
        "I eat less meat. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a vegetarian, it helps me __ <ending>",
        "I eat less meat. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        "I eat less meat. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        "I eat less meat. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me . <ending>",
        "I am vegetarian. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> <ending>",
        "I am vegetarian. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a vegetarian, it helps me __ <ending>",
        "I am vegetarian. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        "I am vegetarian. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        "I am vegetarian. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me . <ending>"]
        skeletons = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None]
        # raw_texts = [
        # "I think deeply about beauty of nature. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher __ am researching the fact __ <ending>",
        # "I think deeply about beauty of nature. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        # "I think deeply about beauty of nature. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        # "I think deeply about beauty of nature. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        # "I am curious. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher __ am researching the fact __ <ending>",
        # "I am curious. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        # "I am curious. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        # "I am curious. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        # "I need to go to the beach. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher __ am researching the fact __ <ending>",
        # "I need to go to the beach. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        # "I need to go to the beach. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        # "I need to go to the beach. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        # "I become more healthy. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a vegetarian, it helps me __ <ending>",
        # "I become more healthy. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        # "I become more healthy. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        # "I become more healthy. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me . <ending>",
        # "I eat less meat. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a vegetarian, it helps me __ <ending>",
        # "I eat less meat. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        # "I eat less meat. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        # "I eat less meat. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me . <ending>",
        # "I am vegetarian. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a vegetarian, it helps me __ <ending>",
        # "I am vegetarian. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        # "I am vegetarian. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        # "I am vegetarian. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me . <ending>"]
        # skeletons = [
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None]

        # 순서 바꿔
        raw_texts = [
        # "I think deeply about beauty of nature. <skeleton>  I am a researcher __ am researching the fact __ <speaker2> I agree, what do you do for living? <speaker1>",
        # "I think deeply about beauty of nature. <skeleton> I am a researcher I am researching the fact __ <speaker2> I agree, what do you do for living? <speaker1>",
        # "I think deeply about beauty of nature. <skeleton> I am a researcher I am researching the fact __ . <speaker2> I agree, what do you do for living? <speaker1>",
        # "I am curious. <skeleton>  I am a researcher __ am researching the fact __ <speaker2> I agree, what do you do for living? <speaker1>",
        # "I am curious. <skeleton> I am a researcher I am researching the fact __ <speaker2> I agree, what do you do for living? <speaker1>",
        # "I am curious. <skeleton> I am a researcher I am researching the fact __ . <speaker2> I agree, what do you do for living? <speaker1>",
        # "I need to go to the beach. <skeleton> I am a researcher __ am researching the fact __ <speaker2> I agree, what do you do for living? <speaker1>",
        # "I need to go to the beach. <skeleton> I am a researcher I am researching the fact __ <speaker2> I agree, what do you do for living? <speaker1>",
        # "I need to go to the beach. <skeleton> I am a researcher I am researching the fact __ . <speaker2> I agree, what do you do for living? <speaker1>",
        # "I become more healthy. <skeleton> how about, maintaining a good diet, try being a vegetarian, it helps me __ <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1>",
        # "I become more healthy. <skeleton> how about, maintaining a good diet, try being a __, it helps me __ <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1>",
        # "I become more healthy. <skeleton> how about, maintaining a good diet, try being a __, it helps me . <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1>",
        # "I eat less meat. <skeleton> how about, maintaining a good diet, try being a vegetarian, it helps me __ <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1>",
        # "I eat less meat. <skeleton> how about, maintaining a good diet, try being a __, it helps me __ <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1>",
        # "I eat less meat. <skeleton> how about, maintaining a good diet, try being a __, it helps me . <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1>",
        "I am vegetarian. <skeleton> how about, maintaining a good diet, try being a vegetarian, it helps me __ <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1>",
        "I am vegetarian. <skeleton> how about, maintaining a good diet, try being a __, it helps me __ <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1>",
        "I am vegetarian. <skeleton> how about, maintaining a good diet, try being a __, it helps me . <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1>"]
        skeletons = [
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        None,
        None,
        None]

        # # 원본이랑 같이
        # raw_texts = [
        # "I think deeply about beauty of nature. I am a researcher I am researching the fact that mermaids are real. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher __ am researching the fact __ <ending>",
        # "I think deeply about beauty of nature. I am a researcher I am researching the fact that mermaids are real. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        # "I think deeply about beauty of nature. I am a researcher I am researching the fact that mermaids are real. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ . <ending>",
        # "I am curious. I am a researcher I am researching the fact that mermaids are real. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher __ am researching the fact __ <ending>",
        # "I am curious. I am a researcher I am researching the fact that mermaids are real. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        # "I am curious. I am a researcher I am researching the fact that mermaids are real. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ . <ending>",
        # "I need to go to the beach. I am a researcher I am researching the fact that mermaids are real. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher __ am researching the fact __ <ending>",
        # "I need to go to the beach. I am a researcher I am researching the fact that mermaids are real. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ <ending>",
        # "I need to go to the beach. I am a researcher I am researching the fact that mermaids are real. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact __ . <ending>",
        # "I become more healthy. how about , maintaining a good diet , try being a vegetarian , it helps me . <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a vegetarian, it helps me __ <ending>",
        # "I become more healthy. how about , maintaining a good diet , try being a vegetarian , it helps me . <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        # "I become more healthy. how about , maintaining a good diet , try being a vegetarian , it helps me . <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me . <ending>",
        # "I eat less meat. how about , maintaining a good diet , try being a vegetarian , it helps me . <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a vegetarian, it helps me __ <ending>",
        # "I eat less meat. how about , maintaining a good diet , try being a vegetarian , it helps me . <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        # "I eat less meat. how about , maintaining a good diet , try being a vegetarian , it helps me . <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me . <ending>",
        # "I am vegetarian. how about , maintaining a good diet , try being a vegetarian , it helps me . <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a vegetarian, it helps me __ <ending>",
        # "I am vegetarian. how about , maintaining a good diet , try being a vegetarian , it helps me . <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me __ <ending>",
        # "I am vegetarian. how about , maintaining a good diet , try being a vegetarian , it helps me . <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a __, it helps me . <ending>"]
        # skeletons = [
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None]

        # # 원본만
        # raw_texts = [
        # "I think deeply about beauty of nature. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact that mermaids are real. <ending>",
        # "I am curious. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact that mermaids are real. <ending>",
        # "I need to go to the beach. <speaker2> I agree, what do you do for living? <speaker1> I am a researcher I am researching the fact that mermaids are real. <ending>",
        # "I become more healthy. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a vegetarian, it helps me . <ending>",
        # "I eat less meat. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a vegetarian, it helps me . <ending>",
        # "I am vegetarian. <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1> how about, maintaining a good diet, try being a vegetarian, it helps me . <ending>"]
        # skeletons = [
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None,
        # None]


        # while not raw_text:
        #     print("Did you forget to add `--cond_text`? ")
        #     raw_text = input("Model prompt >>> ")
        # tokenized_cond_text = tokenizer.encode(
        #     tokenizer.bos_token + raw_text,
        #     add_special_tokens=False
        # )


    # blank 처리하는 방법
    # [62, 11593, 46444]
    # print(tokenizer.encode('_ __ ___'))
    # ss

    pc = None
    pc_train = None
    # pc_train_blank = None
    pc_valid = None
    # pc_valid_blank = None

    with open('./personachat_beta_self_original.json') as f:
        pc = json.load(f)
        pc_train = pc["train"]
        pc_valid = pc["valid"]
        # pc_train_blank = pc["train"]
        # pc_valid_blank = pc["valid"]

    counter = 0

    for i in range(0, len(pc_train)):

        for j in range(0, len(pc_train[i]["utterances"])):


            # I am vegetarian. <skeleton> how about, maintaining a good diet, try being a vegetarian, it helps me __ <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1>

            skeleton_real = pc_train[i]["utterances"][j]["skeleton"]
            gold = pc_train[i]["utterances"][j]["candidates"][-1]

            pc_train[i]["utterances"][j]["new_responses"] = []
            if gold == skeleton_real:
                continue

            if len(pc_train[i]["utterances"][j]["exp_persona"]) == 0:
                continue

            

            counter+=1
            for k in range(0,len(pc_train[i]["utterances"][j]["exp_persona"])):

                temp = []
                history = pc_train[i]["utterances"][j]["history"][-1]

                raw_text = pc_train[i]["utterances"][j]["exp_persona"][k] + " <skeleton>" + skeleton_real + " <speaker2> " + history + " <speaker1>"

                skeleton = None

                # while not raw_text:
                #     print("Did you forget to add `--cond_text`? ")
                #     raw_text = input("Model prompt >>> ")

                tokenized_cond_text = tokenizer.encode(
                    tokenizer.bos_token + raw_text,
                    add_special_tokens=False
                )

                # print("= Prefix of sentence =")
                # print(tokenizer.decode(tokenized_cond_text))
                # print()

                # generate unperturbed and perturbed texts

                # full_text_generation returns:
                # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
                unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
                    model=model,
                    tokenizer=tokenizer,
                    context=tokenized_cond_text,
                    device=device,
                    num_samples=num_samples,
                    bag_of_words=bag_of_words,
                    discrim=discrim,
                    class_label=class_label,
                    length=length,
                    stepsize=stepsize,
                    temperature=temperature,
                    top_k=top_k,
                    sample=sample,
                    num_iterations=num_iterations,
                    grad_length=grad_length,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    gm_scale=gm_scale,
                    kl_scale=kl_scale,
                    verbosity_level=verbosity_level,
                    skeleton=skeleton
                )

                # untokenize unperturbed text
                unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])

                if verbosity_level >= REGULAR:
                    # print("=" * 80)
                    pass
                # print("= Unperturbed generated text =")
                # print(unpert_gen_text)
                # print()

                generated_texts = []

                bow_word_ids = set()
                if bag_of_words and colorama:
                    bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                                           tokenizer)
                    for single_bow_list in bow_indices:
                        # filtering all words in the list composed of more than 1 token
                        filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
                        # w[0] because we are sure w has only 1 item because previous fitler
                        bow_word_ids.update(w[0] for w in filtered)

                # iterate through the perturbed texts
                for l, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
                    try:

                        # untokenize unperturbed text
                        if colorama:

                            import colorama

                            pert_gen_text = ''
                            for word_id in pert_gen_tok_text.tolist()[0]:
                                if word_id in bow_word_ids:
                                    pert_gen_text += '{}{}{}'.format(
                                        colorama.Fore.RED,
                                        tokenizer.decode([word_id]),
                                        colorama.Style.RESET_ALL
                                    )
                                    
                                else:
                                    pert_gen_text += tokenizer.decode([word_id])
                        else:
                            aa = pert_gen_tok_text.tolist()[0]
                            aa = aa[aa.index(50260)+1:]
                            pert_gen_text = tokenizer.decode(aa)

                        temp.append(pert_gen_text)

                    except:
                        pass

                    # keep the prefix, perturbed seq, original seq for each index
                    generated_texts.append(
                        (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
                    )
                pc_train[i]["utterances"][j]["new_responses"].append(temp)
                # print(pc_train[i]["utterances"][j])
                
            print(counter)

    counter=0

    for i in range(0, len(pc_valid)):

        for j in range(0, len(pc_valid[i]["utterances"])):


            # I am vegetarian. <skeleton> how about, maintaining a good diet, try being a vegetarian, it helps me __ <speaker2> I'm not reading this for fun. I'm trying to learn. <speaker1>

            skeleton_real = pc_valid[i]["utterances"][j]["skeleton"]
            gold = pc_valid[i]["utterances"][j]["candidates"][-1]

            pc_valid[i]["utterances"][j]["new_responses"] = []

            if gold == skeleton_real:
                continue

            if len(pc_valid[i]["utterances"][j]["exp_persona"]) == 0:
                continue


            counter+=1
            for k in range(0, len(pc_valid[i]["utterances"][j]["exp_persona"])):
                temp = []
                history = pc_valid[i]["utterances"][j]["history"][-1]

                raw_text = pc_valid[i]["utterances"][j]["exp_persona"][k] + " <skeleton> " + skeleton_real + " <speaker2> " + history + " <speaker1>"

                skeleton = None

                # while not raw_text:
                #     print("Did you forget to add `--cond_text`? ")
                #     raw_text = input("Model prompt >>> ")

                tokenized_cond_text = tokenizer.encode(
                    tokenizer.bos_token + raw_text,
                    add_special_tokens=False
                )


                # print("= Prefix of sentence =")
                # print(tokenizer.decode(tokenized_cond_text))
                # print()

                # generate unperturbed and perturbed texts

                # full_text_generation returns:
                # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
                unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
                    model=model,
                    tokenizer=tokenizer,
                    context=tokenized_cond_text,
                    device=device,
                    num_samples=num_samples,
                    bag_of_words=bag_of_words,
                    discrim=discrim,
                    class_label=class_label,
                    length=length,
                    stepsize=stepsize,
                    temperature=temperature,
                    top_k=top_k,
                    sample=sample,
                    num_iterations=num_iterations,
                    grad_length=grad_length,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    gm_scale=gm_scale,
                    kl_scale=kl_scale,
                    verbosity_level=verbosity_level,
                    skeleton=skeleton
                )

                # untokenize unperturbed text
                unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])

                if verbosity_level >= REGULAR:
                    # print("=" * 80)
                    pass
                # print("= Unperturbed generated text =")
                # print(unpert_gen_text)
                # print()

                generated_texts = []

                bow_word_ids = set()
                if bag_of_words and colorama:
                    bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                                           tokenizer)
                    for single_bow_list in bow_indices:
                        # filtering all words in the list composed of more than 1 token
                        filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
                        # w[0] because we are sure w has only 1 item because previous fitler
                        bow_word_ids.update(w[0] for w in filtered)

                # iterate through the perturbed texts
                for l, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
                    try:

                        # untokenize unperturbed text
                        if colorama:
                            import colorama

                            pert_gen_text = ''
                            for word_id in pert_gen_tok_text.tolist()[0]:
                                if word_id in bow_word_ids:
                                    pert_gen_text += '{}{}{}'.format(
                                        colorama.Fore.RED,
                                        tokenizer.decode([word_id]),
                                        colorama.Style.RESET_ALL
                                    )
                                else:
                                    pert_gen_text += tokenizer.decode([word_id])
                        else:
                            aa = pert_gen_tok_text.tolist()[0]
                            aa = aa[aa.index(50260)+1:]
                            pert_gen_text = tokenizer.decode(aa)

                        # print("= Perturbed generated text {} =".format(l + 1))
                        # print(pert_gen_text)
                        temp.append(pert_gen_text)

                    except:
                        pass

                    # keep the prefix, perturbed seq, original seq for each index
                    generated_texts.append(
                        (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
                    )
                pc_valid[i]["utterances"][j]["new_responses"].append(temp)
                # print(pc_valid[i]["utterances"][j])
            print(counter)


    with open('personachat_beta_new_self_original.json', 'w') as f:
        json.dump(pc,f)


    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--cond_text", type=str, default="The lake",
        help="Prefix texts to condition on"
    )
    parser.add_argument(
        "--uncond", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument('--discrim_weights', type=str, default=None,
                        help='Weights for the generic discriminator')
    parser.add_argument('--discrim_meta', type=str, default=None,
                        help='Meta information for the generic discriminator')
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true",
                        help="colors keywords")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")

    args = parser.parse_args()
    run_pplm_example(**vars(args))
