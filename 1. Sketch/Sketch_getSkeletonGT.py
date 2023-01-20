import os
import json
import jsonlines
from tqdm import tqdm
import random
import numpy as np

from transformers.tokenization_bert import BasicTokenizer

PAD_LABEL_ID = -100


def basic_tokenize(string):
    return BasicTokenizer().tokenize(string)


def get_vocab(paths):
    if os.path.isfile(paths['vocab_path']):
        vocab_path = paths['vocab_path']
        g = open(vocab_path, 'r', encoding='utf-8')
        voc = [s.strip() for s in g.readlines()]
    else:
        vocab = {}
        g = open(paths['vocab_path'], 'w', encoding='utf-8')
        with open(paths['gold_pair_train_path'], 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            # json_data = jsonlines.Reader(f)
            train_data = []
            for item in json_data['train']:
                train_data.append(item)
            for story_index, story in enumerate(tqdm(train_data)):
                history = basic_tokenize(story['beforegold'])
                persona = basic_tokenize(story['persona'])
                gold = basic_tokenize(story['gold'])
                ori_beforegold = basic_tokenize(story['ori_beforegold'])
                ori_persona = basic_tokenize(story['ori_persona'])
                # ori_gold = basic_tokenize(story['ori_gold'])

                all = history + persona + gold + ori_beforegold + ori_persona
                for word in all:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

            voc = [v[0] for v in sorted(vocab.items(), key=lambda item: item[1], reverse=True)]

        for x in voc:
            g.write(x + "\n")

    return voc


def get_exp_vocab(paths):
    if os.path.isfile(paths['vocab_path']):
        g = open(paths['vocab_path'], 'r', encoding='utf-8')
        voc = [s.strip() for s in g.readlines()]
    else:
        vocab = {}
        g = open(paths['vocab_path'], 'w', encoding='utf-8')
        with open(paths['gold_pair_train_path'], 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            # json_data = jsonlines.Reader(f)
            train_data = []
            for item in json_data['train']:
                train_data.append(item)
            for story_index, story in enumerate(tqdm(train_data)):
                history = basic_tokenize(story['beforegold'])
                persona = basic_tokenize(story['persona'])
                gold = basic_tokenize(story['gold'])

                temp_persona = []
                temp_persona.append(story['exp_personality']['xattr'])
                temp_persona.append(story['exp_personality']['xeffect'])
                temp_persona.append(story['exp_personality']['xintent'])
                temp_persona.append(story['exp_personality']['xneed'])
                temp_persona.append(story['exp_personality']['xreact'])
                temp_persona.append(story['exp_personality']['xwant'])
                temp_persona = list(filter('<none>'.__ne__, temp_persona))

                exp_persona = basic_tokenize(" ".join(temp_persona))
                all = history + persona + gold + exp_persona


                for word in all:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

            voc = [v[0] for v in sorted(vocab.items(), key=lambda item: item[1], reverse=True)]

        for x in voc:
            g.write(x + "\n")

    return voc


def get_aug_skeletons(s, mask_rate, replace_rate, vocab):
    """ This function is used for generating augmented skeletons

    Args:
        s: list. The raw skeleton
        mask_rate: The rate of background words to be masked
        replace_rate: The rate of background words to be replaced with random words.
        vocab: The vocab for getting random words.
    """
    # a for mask, b for replace, c for shuffle
    # ending string을 3가지 카테고리(랜덤 마스킹, 랜덤 단어 대치, 랜덤 순서 변경)로 augment 시켜서 리스트를 리턴
    a = s.copy()
    b = s.copy()
    c = s.copy()

    # 랜덤으로 토큰 __ 처리
    for i in range(len(s)):
        if s[i] != "__":
            rand = random.random()
            if rand < mask_rate:
                a[i] = "__"

    # 랜덤으로 토큰에 다른 토큰으로 대채함(보카 사전에 있는 단어 중에서 골라옴)
    for i in range(len(s)):
        if s[i] != "__":
            rand = random.random()
            if rand < replace_rate:
                rand_word_id = np.random.randint(0, len(vocab)-1)
                b[i] = vocab[rand_word_id]

    # 블랭크 뺀 토큰들만 다시 담아서 셔플해버림
    no_blank_word_id_in_sen = []
    no_blank_word_in_sen = []
    for i in range(len(s)):
        if s[i] != "__":
            no_blank_word_id_in_sen.append(i)
            no_blank_word_in_sen.append(s[i])
    random.shuffle(no_blank_word_in_sen)
    j = 0
    for i in range(len(s)):
        if s[i] != "__":
            c[i] = no_blank_word_in_sen[j]
            j += 1

    str_mask = " ".join([s for s in a]).strip()
    str_replace = " ".join([s for s in b]).strip()
    str_shuffle = " ".join([s for s in c]).strip()

    return [str_mask, str_replace, str_shuffle]


def bottom_up_dp_lcs(str_a, str_b, do_merge, mask_rate, replace_rate, vocab, mode):
    """Get LCS skeletons using the bottom up DP algorithm

    Args:
        str_a, : String
        do_merge: bool. Whether merge the consecutive blanks into one blank
        mask_rate: float. The rate of background words to be masked
        replace_rate: float. The rate of background words to be replaced with the random word
        vocab: list. The vocab for getting random words.
        mode: string. "train", "dev" or "test"
    """

    """ LCS 로직 수정 
    <기존> : str_a와 str_b. 두 개의 스트링의 토큰들을 각각 끝 단어부터 서로 비교하면서
    공통된 단어(background word)는 살리고, 서로 다른 단어(Unique word)는 블랭크 처리

    example.
    str_a ( persona ) : i have played since i was 4 years old .
    str_b ( gold )    : never . i have been playing since i was four. it pays more than lifiting weights .

    lcs_a ( persona after process) : i have __ since i was __ __ __ . 
    lcs_b ( gold after process)    : __ __ i have __ __ since i was __ __ __ __ __ __ __ .

    * merge = True --> 이어지는 / consecutive blank는 하나의 blank로 합침.
      데이터 증강 전에 이 작업을 해주고, 증강을 시킴

    lcs_a ( persona after process) : i have __ since i was __ . 
    lcs_b ( gold after process)    : __ i have __ since i was __ .

    <수정> : str_a와 str_b. 두 개의 스트링의 토큰들을 각각 끝 단어부터 서로 비교하면서
    서로 다른 단어(Unique word)는 살리고, 서로 같은 단어(background words)는 personal words로 취급하여 블랭크 처리 

    lcs_a ( persona after process ) :  __ __ played __ __ __ 4 years old __
    lcs_b ( gold after process )    : never . __ __ been playing __ __ __ four . it pays more than lifting weights __

    * merge = True
    lcs_a ( persona after process ) :  __ played __ 4 years old __
    lcs_b ( gold after process )    : never . __ been playing __ four . it pays more than lifting weights __

    * Aug 

    lcs_a : [__ played __ 4 years old __],
            [__ played __ 4 __ old __ ] : 랜덤으로 blank 추가된 데이터
            [__ played __ 4 months old __] : 단어 사전에서 뽑아온 신규 단어를 랜덤으로 교체
            [__ 4 __ played years old __] : 랜덤으로 단어 순서를 바꿈 

    이렇게 기존 데이터에 3가지 타입의 데이터를 더 붙여서 어그맨팅함
    우리는 lcs_b만을 훈련 과정에 사용한다 
    """
    str_a = basic_tokenize(str_a)
    str_b = basic_tokenize(str_b)
    str_a.insert(0, "_str_")
    str_b.insert(0, "_str_")
    if len(str_a) == 0 or len(str_b) == 0:
        return 0
    # Persona sentence 길이만큼 0으로 리스트를 채우고, 그 리스트를 Gold sentence 길이만큼 반복해서 복사한 리스트를 채운다
    dp = [[0 for _ in range(len(str_b) + 1)] for _ in range(len(str_a) + 1)]
    for i in range(1, len(str_a) + 1):
        for j in range(1, len(str_b) + 1):
            if str_a[i - 1] == str_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max([dp[i - 1][j], dp[i][j - 1]])

    i, j = len(str_a), len(str_b)
    LCS_a = ""
    LCS_b = ""
    a_unique = []
    b_unique = []

    # str_a와 str_b의 문장들의 가장 끝부분부터 차례대로 검사하면서, 공통 단어를 체크한다. 공통 단어가 아닌 경우를 unique 리스트에 추가한다
    while i > 0 and j > 0:
        # str_a와 str_b의 마지막 토큰들을 서로 비교하여 같을 경우에 아래 조건문 진입
        if str_a[i - 1] == str_b[j - 1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            # LCS_a = str_a[i - 1] + " " + LCS_a
            # LCS_b = str_a[i - 1] + " " + LCS_b
            # merge가 True일 경우 __이 맨 앞에 있을 경우는 중복 블랭크를 없애야 하기 때문에 if 문으로 구별
            if LCS_a.startswith(" __ ") and do_merge:
                LCS_a = LCS_a
            else:
                LCS_a = " __ " + LCS_a
            if LCS_b.startswith(" __ ") and do_merge:
                LCS_b = LCS_b
            else:
                LCS_b = " __ " + LCS_b

            # LCS_a = " __ " + LCS_a
            # LCS_b = " __ " + LCS_b
            i, j = i - 1, j - 1
            continue
        # str_a와 str_b의 마지막 토큰들을 서로 비교하여 다를 경우에 아래 조건문 진입
        if dp[i][j] == dp[i - 1][j]:  # lcs_a를 채워나가는 조건문
            i, j = i - 1, j
            if LCS_a.startswith(" __ ") and do_merge:
                # LCS_a = LCS_a
                LCS_a = str_a[i] + " " + LCS_a
            else:
                # LCS_a = " __ " + LCS_a
                LCS_a = str_a[i] + " " + LCS_a
            a_unique.append((i, str_a[i]))
            continue
        if dp[i][j] == dp[i][j - 1]:  # lcs_b를 채워나가는 조건문
            i, j = i, j - 1
            if LCS_b.startswith(" __ ") and do_merge:
                # LCS_b = LCS_b
                LCS_b = str_b[j] + " " + LCS_b
            else:
                # LCS_b = " __ " + LCS_b
                LCS_b = str_b[j] + " " + LCS_b
            b_unique.append((j, str_b[j]))
            continue
    a_unique = a_unique[::-1]
    b_unique = b_unique[::-1]

    LCS_a = LCS_a[4:]  # _str_ 이후의 글자들
    LCS_b = LCS_b[4:]

    if not do_merge:
        lcsas = [LCS_a]
        lcsbs = [LCS_b]
    else:
        if mode == "train_aug":
            split_LCS_a = LCS_a.split()
            split_LCS_b = LCS_b.split()
            lcsas = [LCS_a] + get_aug_skeletons(split_LCS_a, mask_rate, replace_rate, vocab)
            lcsbs = [LCS_b] + get_aug_skeletons(split_LCS_b, mask_rate, replace_rate, vocab)
        else:
            lcsas = [LCS_a]
            lcsbs = [LCS_b]

    return lcsas, lcsbs, a_unique, b_unique, str_a[1:], str_b[1:]


def get_train_skeletons(paths, mode, do_merge, mask_rate, replace_rate, exp):

    if exp == True:
        vocab = get_exp_vocab(paths)
    else:
        vocab = get_vocab(paths)

    with open(paths['gold_pair_train_path'], 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        train_data = []
        for item in json_data['train']:
            train_data.append(item)

    skeletons = []

    with open(paths['train_skeletons_path'], 'w', encoding="utf-8") as f:
        for story_index, story in enumerate(tqdm(train_data)):
            history = story['ori_beforegold']
            upg_persona = story['upg_persona']
            gold = story['gold']
            if exp:
                # atomic의 9가지 관계 중 others 관계 3가지를 제외한 6가지 관계만 관리
                # 6가지 관계 중 NLI 필터를 거쳐서 남은 1등 페르소나(빔서치 내림차순)를 사용
                # None은 NLI 필터를 거쳐서 살아남은 녀석이 없는 것임.
                temp_persona = []
                temp_persona.append(story['exp_personality']['xattr'])
                temp_persona.append(story['exp_personality']['xeffect'])
                temp_persona.append(story['exp_personality']['xintent'])
                temp_persona.append(story['exp_personality']['xneed'])
                temp_persona.append(story['exp_personality']['xreact'])
                temp_persona.append(story['exp_personality']['xwant'])

                exp_persona = list(filter('<none>'.__ne__, temp_persona))


                #"xattr": "<none>", "xeffect": "<none>", "xintent": "<none>", "xneed": "i need to drive to the diner . .", "xreact": "<none>", "xwant": "<none>"

            skeleton_persona, skeleton_gold, \
            persona_unique_word, gold_unique_word, \
            basic_toked_persona, basic_toked_gold = bottom_up_dp_lcs(upg_persona, gold, do_merge, mask_rate, replace_rate,
                                                                     vocab, mode)

            skeleton = {}
            skeleton['history'] = story['ori_beforegold']
            skeleton['gold'] = gold
            skeleton['ori_gold'] = story['ori_gold']
            skeleton['persona'] = upg_persona
            skeleton['ori_persona'] = story['ori_persona']
            skeleton['exp_personality'] = story['exp_personality']
            skeleton['used_exp'] = story['used_exp']
            skeleton['raw_skeleton_gold'] = skeleton_gold
            # skeleton['raw_skeleton_gold_unique'] = gold_unique_word
            # skeleton['raw_skeleton_persona_unique'] = persona_unique_word
            basic_toked_history = basic_tokenize(history)

            if exp:
                if not exp_persona:
                    skeleton['exp_persona'] = '[PAD]'
                    basic_toked_exp_persona = basic_tokenize(exp_persona)
                else:
                    skeleton['exp_persona'] = exp_persona
                    basic_toked_exp_persona = basic_tokenize(" ".join(exp_persona))

                sen_pre_1 = ["<premise>"] + basic_toked_history + ["<raw>"] + basic_toked_persona + [
                    "<cf>"] + basic_toked_exp_persona
                sen_after_1 = ["<raw>"] + basic_toked_gold

            else:
                sen_pre_1 = ["<premise>"] + basic_toked_history + ["<raw>"] + basic_toked_persona
                sen_after_1 = ["<raw>"] + basic_toked_gold

            # sen_pre_1은 sketch 레이블에는 관여되지 않는 컨택스트들이라서 -100으로 패딩 처리
            # gold 자리의 레이블은 토큰 길이만큼 0으로 우선 채우고, 정답이 아닌 위치의 레이블은 1로 변경
            len_pre = len(sen_pre_1)
            label_pre = [PAD_LABEL_ID for i in range(len(sen_pre_1))]
            label_gold = [0 for i in range(len(basic_toked_gold))]

            # 우리가 맞춰야 할 자리는 페르소나 문장과 골드 문장의 공통된 단어 토큰
            # 그래서 unique_words의 레이블은 1로 변환. 정답 위치의 레이블은 0으로 두고
            # for x in persona_unique_word:
            #     label_gold[x[0]] = 1
            for x in gold_unique_word:
                label_gold[x[0] - 1] = 1

            label_gold = [PAD_LABEL_ID] + label_gold
            label_raw = [label_pre, label_gold]
            ex_raw = [sen_pre_1, sen_after_1]

            assert len(label_raw[1]) == len(ex_raw[1])

            # skeleton['label_raw'] = label_raw
            # skeleton['ex_raw'] = ex_raw
            skeletons.append(skeleton)

        json.dump(skeletons, f)

def get_train_skeletons_edit(paths):
    '''
    triple_5_ForLCS 데이터를 가지고 스켈레톤을 우선 만든다 ( skeleton = LCS(전처리반영된페르소나, 골드)
    '''
    skeleton_path = paths['train_skeletons_path']
    triple_path = paths['gold_pair_train_path']
    result_path = paths['train_skeletons_edit_path']

    with open(skeleton_path, 'r', encoding='utf-8') as f:
        skeleton_item = json.load(f)
    with open(triple_path, 'r', encoding='utf-8') as t:
        triple_item = json.load(t)

    with open(result_path, 'w', encoding='utf-8') as w:
        result = []
        # triple_persona = []
        # for i, data in enumerate(triple_item):
        #     triple_persona.append(data['originP'])

        for i, data in enumerate(skeleton_item):
            edit_triple = {}
            edit_triple['history'] = data['history']
            edit_triple['persona'] = data['ori_persona']
            edit_triple['gold'] = data['gold']
            edit_triple['raw_skeleton_gold'] = data['raw_skeleton_gold']
            result.append(edit_triple)
            #print("check")
        json.dump(result, w, separators=(',', ': '))


def get_dev_skeletons(paths, mode, do_merge, exp):
    if mode == "dev":
        pair_path = paths['gold_pair_dev_path']
        skeletons_path = paths['dev_skeletons_path']
    elif mode == "test":
        pair_path = paths['gold_pair_test_path']
        skeletons_path = paths['test_skeletons_path']

    with open(pair_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        train_data = []
        for item in json_data['valid']:
            train_data.append(item)

    skeletons = []

    with open(skeletons_path, 'w', encoding="utf-8") as f:
        for story_index, story in enumerate(tqdm(train_data)):
            history = story['beforegold']
            upg_persona = story['upg_persona']
            gold = story['gold']
            if exp:
                # atomic의 9가지 관계 중 others 관계 3가지를 제외한 6가지 관계만 관리
                # 6가지 관계 중 NLI 필터를 거쳐서 남은 1등 페르소나(빔서치 내림차순)를 사용
                # None은 NLI 필터를 거쳐서 살아남은 녀석이 없는 것임.
                temp_persona = []
                temp_persona.append(story['exp_personality']['xattr'])
                temp_persona.append(story['exp_personality']['xeffect'])
                temp_persona.append(story['exp_personality']['xintent'])
                temp_persona.append(story['exp_personality']['xneed'])
                temp_persona.append(story['exp_personality']['xreact'])
                temp_persona.append(story['exp_personality']['xwant'])

                exp_persona = list(filter('<none>'.__ne__, temp_persona))


            skeleton_persona, skeleton_gold, \
            persona_unique_word, gold_unique_word, \
            basic_toked_persona, basic_toked_gold = bottom_up_dp_lcs(upg_persona, gold, do_merge, 0, 0, [], mode)

            skeleton = {}
            skeleton['history'] = story['ori_beforegold']
            skeleton['gold'] = gold
            skeleton['ori_gold'] = story['ori_gold']
            skeleton['persona'] = upg_persona
            skeleton['ori_persona'] = story['ori_persona']
            skeleton['exp_personality'] = story['exp_personality']
            skeleton['used_exp'] = story['used_exp']
            skeleton['raw_skeleton_gold'] = skeleton_gold
            # skeleton['raw_skeleton_gold_unique'] = gold_unique_word
            # skeleton['raw_skeleton_persona_unique'] = persona_unique_word
            basic_toked_history = basic_tokenize(history)

            if exp:
                if not exp_persona:
                    skeleton['exp_persona'] = '[PAD]'
                    basic_toked_exp_persona = basic_tokenize(exp_persona)
                else:
                    skeleton['exp_persona'] = exp_persona
                    basic_toked_exp_persona = basic_tokenize(" ".join(exp_persona))

                sen_pre_1 = ["<premise>"] + basic_toked_history + ["<raw>"] + basic_toked_persona + [
                    "<cf>"] + basic_toked_exp_persona
                sen_after_1 = ["<raw>"] + basic_toked_gold

            else:
                sen_pre_1 = ["<premise>"] + basic_toked_history + ["<raw>"] + basic_toked_persona
                sen_after_1 = ["<raw>"] + basic_toked_gold


            #
            #
            # if exp:
            #     skeleton['exp_persona'] = exp_persona
            #     basic_toked_exp_persona = basic_tokenize(exp_persona)
            #
            # skeleton['raw_skeleton_gold'] = skeleton_gold
            # skeleton['raw_skeleton_gold_unique'] = gold_unique_word
            # # skeleton['raw_skeleton_persona_unique'] = persona_unique_word
            #
            # basic_toked_history = basic_tokenize(history)
            #
            # if exp:
            #     sen_pre_1 = ["<premise>"] + basic_toked_history + ["<raw>"] + basic_toked_persona + [
            #         "<cf>"] + basic_toked_exp_persona
            #     sen_after_1 = ["<raw>"] + basic_toked_gold
            # else:
            #     sen_pre_1 = ["<premise>"] + basic_toked_history + ["<raw>"] + basic_toked_persona
            #     sen_after_1 = ["<raw>"] + basic_toked_gold

            # sen_pre_1은 sketch 레이블에는 관여되지 않는 컨택스트들이라서 -100으로 패딩 처리
            # gold 자리의 레이블은 토큰 길이만큼 0으로 우선 채우고, 정답이 아닌 위치의 레이블은 1로 변경
            len_pre = len(sen_pre_1)
            label_pre = [PAD_LABEL_ID for i in range(len(sen_pre_1))]
            label_gold = [0 for i in range(len(basic_toked_gold))]

            # 우리가 맞춰야 할 자리는 페르소나 문장과 골드 문장의 공통된 단어 토큰
            # 그래서 unique_words의 레이블은 1로 변환. 정답 위치의 레이블은 0으로 두고
            # for x in persona_unique_word:
            #     label_gold[x[0]] = 1
            for x in gold_unique_word:
                label_gold[x[0] - 1] = 1

            label_gold = [PAD_LABEL_ID] + label_gold
            label_raw = [label_pre, label_gold]
            ex_raw = [sen_pre_1, sen_after_1]

            assert len(label_raw[1]) == len(ex_raw[1])

            # skeleton['label_raw'] = label_raw
            # skeleton['ex_raw'] = ex_raw
            skeletons.append(skeleton)

        json.dump(skeletons, f)


if __name__ == '__main__':
    paths = {}
    # 최종 데이터 #

    do_merge = True
    exp = False
    paths['vocab_path'] = "dataForFinal/vocab.txt"
    paths['gold_pair_train_path'] = "trimdata/edited_triple_train.json"
    paths['train_skeletons_path'] = "dataForFinal/train_skeletons_aug.json"
    paths['train_skeletons_edit_path'] = "dataForFinal/train_skeletons_aug_edit.json"
    get_train_skeletons(paths, "train_aug", do_merge, 0.2, 0.2, exp)
    # get_train_skeletons_edit(paths)

    #dev
    do_merge = False
    exp = False
    paths['gold_pair_dev_path'] = "trimdata/edited_triple_train.json"
    paths['dev_skeletons_path'] = "dataForFinal/dev_skeletons.json"
    get_dev_skeletons(paths, "dev", do_merge, exp)



    # # --- 원규 데이터  ----#

    # train
    # do_merge = True
    # exp = True
    # paths['vocab_path'] = "dataForWG/vocab_exp.txt"
    # paths['gold_pair_train_path'] = "dataForWG/gold_pair_train_exp_atomic_filter.json"  # read
    # paths['train_skeletons_path'] = "dataForWG/train_skeletons_atomic_filter_aug.json"  # write
    # get_train_skeletons(paths, "train_aug", do_merge, 0.2, 0.2, exp)

    #dev
    # do_merge = False
    # exp = True
    # paths['gold_pair_dev_path'] = "dataForWG/gold_pair_dev_exp_atomic_filter.json"
    # paths['dev_skeletons_path'] = "dataForWG/dev_skeletons_atomic_filter_TTTT.json"
    # get_dev_skeletons(paths, "dev", do_merge, exp)
    # ----- 원규 데이터 용도 끝 -------

    # do_merge = False
    # exp = True
    # paths['vocab_path'] = "data/vocab_exp_none.txt"
    # paths['gold_pair_dev_path'] = "dataForWG/gold_pair_dev.json"
    # paths['dev_skeletons_path'] = "dataForWG/dev_skeletons.json"
    # get_dev_skeletons(paths, "dev", do_merge, exp)

    # do_merge = False
    # exp = True
    # paths['vocab_path'] = "data/vocab_exp_none.txt"
    # paths['gold_pair_dev_path'] = "data/gold_pair_dev_exp_none.json"
    # paths['dev_skeletons_path'] = "data/dev_skeletons_exp_none.json"
    # get_dev_skeletons(paths, "dev", do_merge, exp)

    # #test_exp
    # do_merge = False
    # exp = True
    # paths['gold_pair_test_path'] = "data/gold_pair_test_exp_none.json"
    # paths['test_skeletons_path'] = "data/test_skeletons_exp_none.json"
    # get_dev_skeletons(paths, "test", do_merge, exp)

    # test
    # do_merge = False
    # exp = False
    # paths['gold_pair_test_path'] = "data/gold_pair_test_wo_exp2.json"
    # paths['test_skeletons_path'] = "data/test_skeletons_wo_exp2.json"
    #
    # get_dev_skeletons(paths, "test", do_merge, exp)

    # # --- YES merge  YES Aug  TRAINING ----#
    # do_merge = True
    # paths['vocab_path'] = "data/vocab.txt"
    # paths['gold_pair_train_path'] = "data/gold_pair_train.json"
    # paths['train_skeletons_path'] = "data/train_skeletons.json"
    # #
    # get_train_skeletons(paths, "train", do_merge, 0.2, 0.2)

