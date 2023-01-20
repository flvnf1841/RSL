import os
import json
import jsonlines
from tqdm import tqdm
import string
import re

from num2words import num2words
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk
# nltk.download("punkt")

'''데이터 셋 생성 함수 모음 '''
#(1)
def make_dnli_trimPeriod(paths, mode):
    if mode == "train":
        dnli_data_path = paths['dnli_train_0']
        result_path = paths['dnli_train_1']
    else:
        dnli_data_path = paths['dnli_dev_0']
        result_path = paths['dnli_dev_1']

    with open(dnli_data_path, 'r', encoding='utf-8') as f:
        dnli_data = json.load(f)
        entail_group = []
        for idx, item in enumerate(dnli_data):
            entail_pair = {}
            if item['dtype'] == 'matchingtriple_up' and item['label'] == 'positive':
                entail_pair['gold'] = item['sentence1'].strip(string.punctuation).strip()
                entail_pair['persona'] = item['sentence2'].strip(string.punctuation).strip()
                entail_group.append(entail_pair)
    with open(result_path, 'w', encoding='utf-8') as w:
        json.dump(entail_group, w, separators=(',', ': '))
#(2)
def edit_pc(paths):

    pc_data_path = paths['personachat_train_0']
    result_path = paths['personachat_train_1']
    quote_dict = {  # 조동사 모음
        " cant ": " cannot ", "can t ": "cannot ", "couldn t ": "could not ",
        "shouldn t ": "should not ", "i ll ": "i will ", "won t ": "will not ", "you ll ": "you will ",
        "i d ": "i would ", "i dn t ": "i would not ", "wouldn t ": "would not ", "you d ": "you would ",
        "don t ": "do not ", "doesn t ": "does not ", "didn t ": "did not ",
        # Be동사 모음
        "i m ": "i am ", "isn t ": "is not ", "aren t ": "are not ", "wasn t ": "was not ",
        "she s ": "she is ", "she sn t": "she is not ", "he s ": "he is ", "he sn t": "he is not ",
        "they re ": "they are ", "they ren t ": "they are not ",
        "that s ": "that is ", "that sn t ": "that is not ",
        "there s ": "there is ", "there sn t ": "there is not ",
        "isn thing ": "is nothing ", " sn thing ": " is nothing",
        "weren t ": "were not ",
        # 일반동사 모음
        "i ve ": "i have ", "i ven t ": "i have not ", "haven t ": "have not ", "hasn t ": "has not ",
        "haven thing ": "have nothing "
    }

    with open(pc_data_path, 'r', encoding='utf-8') as f:
        pc_data = json.load(f)
        trim_total_pc = {}

    with open(result_path, 'w', encoding='utf-8') as w:
        for i, type in enumerate(pc_data):
            trim_pc_data = []
            for j, pc_item in enumerate(pc_data[type]):
                ori_persona_group = []
                ori_history_group = []
                trim_persona_group = []
                trim_gold_group = []
                trim_cand_group = []
                exp_group = pc_item['exp_personality']
                ori_gold_list = []
                ori_gold_result = []
                result = {}

                for q in range (len(pc_item['utterances'])):
                    ori_gold_list.append(pc_item['utterances'][q]['candidates'][-1])

                for ori_gold in ori_gold_list:
                    ori_gold_dict = {}
                    trim_gold = ori_gold.strip(string.punctuation).strip()
                    if re.findall(r"\'+", trim_gold):
                        trim_gold = trimQuote(trim_gold)
                    matchingG = {s:quote_dict[s] for s in quote_dict.keys() if s in trim_gold}
                    if matchingG:
                        for key in matchingG.keys():
                            trim_gold = trim_gold.replace(key, matchingG[key]) # 축약형 -> 완전형
                    ori_gold_dict[trim_gold] = ori_gold
                    ori_gold_result.append(ori_gold_dict)


                for persona in pc_item['personality']:
                    ori_persona = persona
                    trim_persona = persona.strip(string.punctuation).strip() # 구두점 제거
                    if re.findall(r"\'+", trim_persona):
                        trim_persona = trimQuote(trim_persona)               # quote 제거
                    matchingP = {s:quote_dict[s] for s in quote_dict.keys() if s in trim_persona}
                    if matchingP:
                        for key in matchingP.keys():
                            trim_persona = trim_persona.replace(key, matchingP[key]) # 축약형 -> 완전형
                    trim_persona_group.append(trim_persona)
                    ori_persona_group.append(ori_persona)


                for history in pc_item['utterances'][-1]['history']:
                    ori_history = history
                    trim_gold = history.strip(string.punctuation).strip() # 구두점 제거
                    if re.findall(r"\'+", trim_gold):
                        trim_gold = trimQuote(trim_gold)               # quote 제거
                    matchingG = {s:quote_dict[s] for s in quote_dict.keys() if s in trim_gold}
                    if matchingG:
                        for key in matchingG.keys():
                            trim_gold = trim_gold.replace(key, matchingG[key]) # 축약형 -> 완전형

                    trim_gold_group.append(trim_gold)
                    ori_history_group.append(ori_history)

                cand = pc_item['utterances'][-1]['candidates'][-1]
                trim_cand = cand.strip(string.punctuation).strip() # 구두점 제거
                if re.findall(r"\'+", trim_cand):
                    trim_cand = trimQuote(trim_cand)               # quote 제거
                matchingC = {s:quote_dict[s] for s in quote_dict.keys() if s in trim_cand}
                if matchingC:
                    for key in matchingC.keys():
                        trim_cand = trim_cand.replace(key, matchingC[key]) # 축약형 -> 완전형

                result['ori_persona'] = ori_persona_group
                result['persona'] = trim_persona_group
                result['history'] = trim_gold_group
                result['ori_history'] = ori_history_group
                result['candidates'] = [trim_cand]
                result['exp_personality'] = exp_group
                result['ori_gold_dict'] = ori_gold_result

                trim_pc_data.append(result)
            trim_total_pc[type] = trim_pc_data
            print("check")
        json.dump(trim_total_pc, w, separators=(',', ': '))

def edit_dnli(paths, mode):
    if mode == "train":
        dnli_path = paths['dnli_train_0']
        result_path = paths['dnli_train_1']
    else:
        dnli_path = paths['dnli_dev_0']
        result_path = paths['dnli_dev_1']

    quote_dict = {  # 조동사 모음
        " cant ": " cannot ", "can t ": "cannot ", "couldn t ": "could not ",
        "shouldn t ": "should not ", "i ll ": "i will ", "won t ": "will not ", "you ll ": "you will ",
        "i d ": "i would ", "i dn t ": "i would not ", "wouldn t ": "would not ", "you d ": "you would ",
        "don t ": "do not ", "doesn t ": "does not ", "didn t ": "did not ",
        # Be동사 모음
        "i m ": "i am ", "isn t ": "is not ", "aren t ": "are not ", "wasn t ": "was not ",
        "she s ": "she is ", "she sn t": "she is not ", "he s ": "he is ", "he sn t": "he is not ",
        "they re ": "they are ", "they ren t ": "they are not ",
        "that s ": "that is ", "that sn t ": "that is not ",
        "there s ": "there is ", "there sn t ": "there is not ",
        "isn thing ": "is nothing ", " sn thing ": " is nothing",
        "weren t ": "were not ",
        # 일반동사 모음
        "i ve ": "i have ", "i ven t ": "i have not ", "haven t ": "have not ", "hasn t ": "has not ",
        "haven thing ": "have nothing "
    }

    with open(dnli_path, 'r', encoding='utf-8') as f:
        dnli_data = json.load(f)
        entail_group = []
        trim_dnli_data = []
    with open(result_path, 'w', encoding='utf-8') as w:
        for i, dnli_item in enumerate(dnli_data):
            result = {}
            if dnli_item['dtype'] == "matchingtriple_up" and dnli_item['label'] == 'positive':
                trim_persona = dnli_item['sentence2'].strip(string.punctuation).strip()
                trim_gold = dnli_item['sentence1'].strip(string.punctuation).strip()
                matchingP = {s: quote_dict[s] for s in quote_dict.keys() if s in trim_persona}
                matchingG = {s: quote_dict[s] for s in quote_dict.keys() if s in trim_gold}
                if matchingP:
                    for key in matchingP.keys():
                        trim_persona = trim_persona.replace(key, matchingP[key])
                if matchingG:
                    for key in matchingG.keys():
                        trim_gold = trim_gold.replace(key, matchingG[key])
                result['persona'] = trim_persona
                result['gold'] = trim_gold
                result['ori_persona'] = dnli_item['sentence2']
                result['ori_gold'] = dnli_item['sentence1']
                trim_dnli_data.append(result)
        json.dump(trim_dnli_data, w, separators=(',', ': '))

def upgrade(persona, gold):
    stopwords = ['i']
    # upgrade skeleton을 위해 페르소나를 한번 더 전처리 한다
    '''(1) 트리플 갯수를 늘리기 위해 전처리한 페르소나를 가지고 전처리 수행
        - LCS의 약점을 보완하고자 페르소나에 i 토큰 제거 AND 골드와의 어간 체크를 통하여 토큰 변경'''
    p_sentence = [w for w in persona.split() if w not in stopwords]
    g_sentence = gold.split()
    p_stemmed_sentence = ' '.join(p_sentence)
    # 포터 알고리즘을 사용하여 어간 추출
    s = PorterStemmer()
    p_stem = {s.stem(w): w for w in p_sentence}
    g_stem = {s.stem(w): w for w in g_sentence}
    for j, p_token in enumerate(p_stem.keys()):
        for k, g_token in enumerate(g_stem.keys()):
            if p_token == g_token:
                p_stemmed_sentence = p_stemmed_sentence.replace(p_stem[p_token], g_stem[g_token])
    return p_stemmed_sentence

def transferdigit(sentence):
    hasNumber = lambda stringVal: any(elem.isdigit() for elem in stringVal)
    if hasNumber(sentence):
        digit_list = re.findall(r'\d+', sentence)
        for i, digit in enumerate(digit_list):
            a = num2words(digit)
            if '-' in a:
                a = a.replace('-', ' ')
            sentence = sentence.replace(digit, a)
        return sentence
    else:
        return sentence

def makeEditTriple(paths):
    edit_dnli_path = paths['dnli_train_1']
    edit_dnli_dev_path = paths['dnli_dev_1']
    edit_pc_path = paths['personachat_train_1']
    edit_triple_path = paths['triple_train_1']
    stopwords = ['i']


    with open(edit_dnli_path, 'r', encoding='utf-8') as d:
        dnli_data = json.load(d)
    with open(edit_dnli_dev_path, 'r', encoding='utf-8') as d:
        dnli_dev_data = json.load(d)
    with open(edit_pc_path, 'r', encoding='utf-8') as p:
        pc_data = json.load(p)
    with open(edit_triple_path, 'w', encoding='utf-8') as w:
        total_result = {}
        result = []
        for i, dnli_item in enumerate(tqdm(dnli_data)):
            for j, pc_item in enumerate(pc_data['train']):
                if (dnli_item['persona'] in pc_item['persona']) and (dnli_item['gold'] in pc_item['history']):
                    triple = {}

                    gold_index = pc_item['history'].index(dnli_item['gold'])
                    triple['ori_beforegold'] = pc_item['ori_history'][gold_index -1]
                    triple['beforegold'] = pc_item['history'][gold_index - 1]
                    ori_gold_keys = [y for x in pc_item['ori_gold_dict'] for y in x.keys()]
                    ori_gold_values = [y for x in pc_item['ori_gold_dict'] for y in x.values()]
                    if dnli_item['gold'] in ori_gold_keys:
                        index = ori_gold_keys.index(dnli_item['gold'])
                        triple['ori_gold'] = ori_gold_values[index]
                    else:
                        triple['ori_gold'] = ''
                    triple['gold'] = transferdigit(dnli_item['gold'])

                    try:
                        persona_index = pc_item['persona'].index(dnli_item['persona'])
                        triple['ori_persona'] = pc_item['ori_persona'][persona_index]
                    except:
                        triple['ori_persona'] = ''

                    # triple['ori_persona'] = dnli_item['ori_persona']
                    triple['persona'] = transferdigit(dnli_item['persona'])
                    triple['upg_persona'] = transferdigit(upgrade(dnli_item['persona'], dnli_item['gold']))
                    triple['exp_personality'] = pc_item['exp_personality']
                    triple['used_exp'] = pc_item['exp_personality'][pc_item['persona'].index(dnli_item['persona'])]

                    result.append(triple)
                elif (dnli_item['persona'] in pc_item['persona']) and (pc_item['candidates'][0] == dnli_item['gold']):
                    triple = {}

                    # gold_index = pc_item['history'].index(dnli_item['gold'])
                    triple['ori_beforegold'] = pc_item['ori_history'][-1]
                    triple['beforegold'] = pc_item['history'][-1]
                    ori_gold_keys = [y for x in pc_item['ori_gold_dict'] for y in x.keys()]
                    ori_gold_values = [y for x in pc_item['ori_gold_dict'] for y in x.values()]
                    if dnli_item['gold'] in ori_gold_keys:
                        index = ori_gold_keys.index(dnli_item['gold'])
                        triple['ori_gold'] = ori_gold_values[index]
                    else:
                        triple['ori_gold'] = ''
                    triple['gold'] = transferdigit(dnli_item['gold'])
                    try:
                        persona_index = pc_item['persona'].index(dnli_item['persona'])
                        triple['ori_persona'] = pc_item['ori_persona'][persona_index]
                    except:
                        triple['ori_persona'] = ''
                    triple['persona'] = transferdigit(dnli_item['persona'])
                    triple['upg_persona'] = transferdigit(upgrade(dnli_item['persona'], dnli_item['gold']))
                    triple['exp_personality'] = pc_item['exp_personality']
                    triple['used_exp'] = pc_item['exp_personality'][pc_item['persona'].index(dnli_item['persona'])]

                    result.append(triple)
                    if triple['ori_persona'] == '':
                        print("null ori persona")
        total_result["train"] = result
        print(f"train number : {len(total_result['train'])}")

        result = []
        for i, dnli_item in enumerate(tqdm(dnli_dev_data)):
            for j, pc_item in enumerate(pc_data['valid']):
                if (dnli_item['persona'] in pc_item['persona']) and (dnli_item['gold'] in pc_item['history']):
                    triple = {}

                    gold_index = pc_item['history'].index(dnli_item['gold'])
                    triple['ori_beforegold'] = pc_item['ori_history'][gold_index - 1]
                    triple['beforegold'] = pc_item['history'][gold_index - 1]
                    ori_gold_keys = [y for x in pc_item['ori_gold_dict'] for y in x.keys()]
                    ori_gold_values = [y for x in pc_item['ori_gold_dict'] for y in x.values()]
                    if dnli_item['gold'] in ori_gold_keys:
                        index = ori_gold_keys.index(dnli_item['gold'])
                        triple['ori_gold'] = ori_gold_values[index]
                    else:
                        triple['ori_gold'] = ''
                    triple['gold'] = transferdigit(dnli_item['gold'])

                    try:
                        persona_index = pc_item['persona'].index(dnli_item['persona'])
                        triple['ori_persona'] = pc_item['ori_persona'][persona_index]
                    except:
                        triple['ori_persona'] = ''

                    # triple['ori_persona'] = dnli_item['ori_persona']
                    triple['persona'] = transferdigit(dnli_item['persona'])
                    triple['upg_persona'] = transferdigit(upgrade(dnli_item['persona'], dnli_item['gold']))
                    triple['exp_personality'] = pc_item['exp_personality']
                    triple['used_exp'] = pc_item['exp_personality'][pc_item['persona'].index(dnli_item['persona'])]

                    result.append(triple)
                elif (dnli_item['persona'] in pc_item['persona']) and (pc_item['candidates'][0] == dnli_item['gold']):
                    triple = {}

                    # gold_index = pc_item['history'].index(dnli_item['gold'])
                    triple['ori_beforegold'] = pc_item['ori_history'][-1]
                    triple['beforegold'] = pc_item['history'][-1]
                    ori_gold_keys = [y for x in pc_item['ori_gold_dict'] for y in x.keys()]
                    ori_gold_values = [y for x in pc_item['ori_gold_dict'] for y in x.values()]
                    if dnli_item['gold'] in ori_gold_keys:
                        index = ori_gold_keys.index(dnli_item['gold'])
                        triple['ori_gold'] = ori_gold_values[index]
                    else:
                        triple['ori_gold'] = ''
                    triple['gold'] = transferdigit(dnli_item['gold'])
                    try:
                        persona_index = pc_item['persona'].index(dnli_item['persona'])
                        triple['ori_persona'] = pc_item['ori_persona'][persona_index]
                    except:
                        triple['ori_persona'] = ''
                    triple['persona'] = transferdigit(dnli_item['persona'])
                    triple['upg_persona'] = transferdigit(upgrade(dnli_item['persona'], dnli_item['gold']))
                    triple['exp_personality'] = pc_item['exp_personality']
                    triple['used_exp'] = pc_item['exp_personality'][pc_item['persona'].index(dnli_item['persona'])]

                    result.append(triple)
                    if triple['ori_persona'] == '':
                        print("null ori persona")
        total_result["valid"] = result
        print(f"valid number : {len(total_result['valid'])}")


        json.dump(total_result, w, separators=(',', ': '))



#(3)
def make_pc_NoQuote(paths, mode):
    if mode == "train":
        pc1_data_path = paths['personachat_train_1']
        result_path = paths['personachat_train_2']
    else:
        pass

    with open(pc1_data_path, 'r', encoding='utf-8') as f:
        pc1_data = json.load(f)
        results = []
        for i, data in enumerate(tqdm(pc1_data)):
            noQuote_persona_group = []
            noQuote_gold_group = []
            noQuote_groups = {}

            for j, persona in enumerate(data['personality']):
                if re.findall(r"\'+", persona):
                    persona = trimQuote(persona)
                noQuote_persona_group.append(persona)

            for k, gold in enumerate(data['history']):
                if re.findall(r"\'+", gold):
                    gold = trimQuote(gold)
                noQuote_gold_group.append(gold)

            noQuote_groups['personality'] = noQuote_persona_group
            noQuote_groups['history'] = noQuote_gold_group
            results.append(noQuote_groups)

    with open(result_path, 'w', encoding='utf-8') as w:
        json.dump(results, w, separators=(',', ': '))
#(4)
def editForMatchingDNLI2(paths, mode):
    if mode == "train":
        dnli_path = paths['dnli_train_1']
        edit_dnli_path = paths['dnli_train_3']
    else:
        pass

    quote_dict = {  # 조동사 모음
        " cant ": " cannot ", "can t ": "cannot ", "couldn t ": "could not ",
        "shouldn t ": "should not ", "i ll ": "i will ", "won t ": "will not ", "you ll ": "you will ",
        "i d ": "i would ", "i dn t ": "i would not ", "wouldn t ": "would not ", "you d ": "you would ",
        "don t ": "do not ", "doesn t ": "does not ", "didn t ": "did not ",
        # Be동사 모음
        "i m ": "i am ", "isn t ": "is not ", "aren t ": "are not ", "wasn t ": "was not ",
        "she s ": "she is ", "she sn t": "she is not ", "he s ": "he is ", "he sn t": "he is not ",
        "they re ": "they are ", "they ren t ": "they are not ",
        "that s ": "that is ", "that sn t ": "that is not ",
        "there s ": "there is ", "there sn t ": "there is not ",
        "isn thing ": "is nothing ", " sn thing ": " is nothing",
        "weren t ": "were not ",
        # 일반동사 모음
        "i ve ": "i have ", "i ven t ": "i have not ", "haven t ": "have not ", "hasn t ": "has not ",
        "haven thing ": "have nothing "
    }

    with open(dnli_path, 'r', encoding='utf-8') as f:
        dnli_data = json.load(f)

    with open(edit_dnli_path, 'w', encoding='utf-8') as w:
        edited_list = []
        for i, dnli_item in enumerate(tqdm(dnli_data)):
            edited_result = {}
            tempP = dnli_item['persona']
            tempG = dnli_item['gold']

            matchingP = {s: quote_dict[s] for s in quote_dict.keys() if s in dnli_item['persona']}
            matchingG = {s: quote_dict[s] for s in quote_dict.keys() if s in dnli_item['gold']}

            if matchingP:
                for key in matchingP.keys():
                    dnli_item['persona'] = dnli_item['persona'].replace(key, matchingP[key])
            if dnli_item['persona'] != tempP:
                edited_result['persona'] = dnli_item['persona']
            else:
                edited_result['persona'] = tempP

            if matchingG:
                for key in matchingG.keys():
                    dnli_item['gold'] = dnli_item['gold'].replace(key, matchingG[key])
            if dnli_item['gold'] != tempG:
                edited_result['gold'] = dnli_item['gold']
            else:
                edited_result['gold'] = tempG
            edited_list.append(edited_result)

        json.dump(edited_list, w, separators=(',', ': '))

def editForMatchingPC2(paths, mode):
    if mode == "train":
        pc_path = paths['personachat_train_2']
        edit_pc_path = paths['personachat_train_3']
    else:
        pass

    with open(pc_path, 'r', encoding='utf-8') as f:
        pc_data = json.load(f)
    with open(edit_pc_path, 'w', encoding='utf-8') as w:
        edited_list = []
        quote_dict = {  # 조동사 모음
            " cant ": " cannot ", "can t ": "cannot ", "couldn t ": "could not ",
            "shouldn t ": "should not ", "i ll ": "i will ", "won t ": "will not ", "you ll ": "you will ",
            "i d ": "i would ", "i dn t ": "i would not ", "wouldn t ": "would not ", "you d ": "you would ",
            "don t ": "do not ", "doesn t ": "does not ", "didn t ": "did not ",
            # Be동사 모음
            "i m ": "i am ", "isn t ": "is not ", "aren t ": "are not ", "wasn t ": "was not ",
            "she s ": "she is ", "she sn t": "she is not ", "he s ": "he is ", "he sn t": "he is not ",
            "they re ": "they are ", "they ren t ": "they are not ",
            "that s ": "that is ", "that sn t ": "that is not ",
            "there s ": "there is ", "there sn t ": "there is not ",
            "isn thing ": "is nothing ", " sn thing ": " is nothing",
            "weren t ": "were not ",
            # 일반동사 모음
            "i ve ": "i have ", "i ven t ": "i have not ", "haven t ": "have not ", "hasn t ": "has not ",
            "haven thing ": "have nothing "
        }

        for i, pc_item in enumerate(tqdm(pc_data)):
            edited_result = {}
            temp_p = []
            temp_g = []
            for j, persona in enumerate(pc_item['personality']):
                temp1 = persona
                #persona = 'i m tennis player. i cant hard because i can t play any more'
                matching = {s:quote_dict[s] for s in quote_dict.keys() if s in persona}
                if matching:
                    for key in matching.keys():
                        persona = persona.replace(key, matching[key])
                if persona != temp1:
                    temp_p.append(persona)
                else:
                    temp_p.append(temp1)

            for k, gold in enumerate(pc_item['history']):
                temp2 = gold
                matchingG = {s:quote_dict[s] for s in quote_dict.keys() if s in gold}
                if matchingG:
                    for key in matchingG.keys():
                        gold = gold.replace(key, matchingG[key])
                if gold != temp2:
                    temp_g.append(gold)
                else:
                    temp_g.append(temp2)

            edited_result['personality'] = temp_p
            edited_result['history'] = temp_g
            edited_list.append(edited_result)
        json.dump(edited_list, w, separators=(',', ': '))


#(5)
def makeTriple(paths, mode):
    if mode == "train":
        triple_dnli_path = paths['dnli_train_3']
        triple_pc_path = paths['personachat_train_3']
        triple_path = paths['triple_result_3']
    else:
        pass

    with open(triple_dnli_path, 'r', encoding='utf-8') as f:
        dnli_data = json.load(f)

    with open(triple_pc_path, 'r', encoding='utf-8') as g:
        pc_data = json.load(g)

    with open(triple_path, 'w', encoding='utf-8') as w:
        result = []
        for i, dnli_item in enumerate(tqdm(dnli_data)):
            for j, pc_item in enumerate(pc_data):
              triple = {}
              if (dnli_item['persona'] in pc_item['personality']) and (dnli_item['gold'] in pc_item['history']):
                  gold_index = pc_item['history'].index(dnli_item['gold'])
                  triple['beforegold'] = pc_item['history'][gold_index-1]
                  triple['gold'] = dnli_item['gold']
                  triple['persona'] = dnli_item['persona']
                  result.append(triple)
        print(f"The number of triple({triple_path.split('/')[1]}) is {len(result)}")
        json.dump(result, w, separators=(',', ': '))
#(6)
def editForTriple(paths, mode):
    if mode == "train":
        triple_path = paths['triple_result_3']
        result = paths['triple_result_4']
    else:
        pass

    digit_dict = countDigit(paths, mode)
    ''' 0: zero, 1: one '''

    with open(triple_path, 'r', encoding='utf-8') as f:
        triple_data = json.load(f)

    with open(result, 'w', encoding='utf-8') as w:
        results = []
        for i, triple_item in enumerate(tqdm(triple_data)):
            digit_result = {}
            tempP = triple_item['persona']
            tempG = triple_item['gold']

            matchingP = {s:digit_dict[s] for s in digit_dict.keys() if triple_item['persona'].find(s) != -1}
            matchingG = {s:digit_dict[s] for s in digit_dict.keys() if triple_item['gold'].find(s) != -1}

            if matchingP:
                triple_item['persona'] = triple_item['persona'].replace(list(matchingP.keys())[-1], list(matchingP.values())[-1])
            if triple_item['persona'] != tempP:
                digit_result['persona'] = triple_item['persona']
            else:
                digit_result['persona'] = tempP

            if matchingG:
                triple_item['gold'] = triple_item['gold'].replace(list(matchingG.keys())[-1], list(matchingG.values())[-1])
            if triple_item['gold'] != tempG:
                digit_result['gold'] = triple_item['gold']
            else:
                digit_result['gold'] = tempG
            digit_result['beforegold'] = triple_item['beforegold']
            results.append(digit_result)

        json.dump(results, w, separators=(',', ': '))
#(7)
def editForLCS(paths, mode):
    '''
    골드가 담고 있는 페르소나 색깔(Content)을 빼는 목적을 두고 있다.
    이 목적을 위해 LCS(Longest Common Subsequece)알고리즘을 사용하여
    source : persona sentence
    target : gold sentence
    LCS(source, target) => skeleton이 된다
    LCS만 단독으로 사용하면 다음과 같은 한계점이 있다.
    1. Gold sentence에서 문장의 틀을 지켜주는 토큰. 특히 페르소나는 1인칭의 문장이 주됨.
    그래서 I 라는 토큰은 틀을 지켜주는 토큰이라 fix templete token이라 볼 수 있다. 이 토큰은 블랭크 처리가 안되는 게 좋다
    ==> Persona 문장에서 Stopword = I 토큰을 제거하여 데이터 셋 구성
    2. Gold sentence와 Persona sentence에는 어간이 같은 토큰들이 자주 사용된다
    Persona : i practice vegetarianism
    gold : how about , maintaining a good diet , try being a vegetarian , it helps me
    ==> stemming 라이브러리를 사용하여 서로가 공통된 어간이 있으면, 페르소나의 해당 토큰을 골드의 해당 토큰으로 대체
    3. Gold sentence와 Persona Sentence에는 같은 의미를 지닌 숫자와 영어 표현을 혼용해서 사용한다.
    Persona : i have played since i was 4 years old
    Gold : never . i have been playing since i was four . it pays more than lifting weights
    ==> 페르소나와 골드에서 사용된 숫자를 다 추출 조사하여, 영어 표현으로 전부 통일 시킴
    '''


    if mode == "train":
        triple_path = paths['triple_result_4']
        result = paths['triple_result_5']
    else:
        pass

    # 페르소나 문장에서 i 토큰 지우기
    with open(triple_path, 'r', encoding='utf-8') as f:
        triple_data = json.load(f)
    with open(result, 'w', encoding='utf-8') as w:
        stopwords = ['i']
        processed_persona = []
        for i, triple_item in enumerate(tqdm(triple_data)):
            results = {}
            # triple_item['persona'] = 'i am learning vegetarianism and enjoy diet'
            # triple_item['gold'] = 'maintaining a good diet , try being a vegetarian , i learn'
            #페르소나 문장에서 I 토큰 제거
            p_sentence = [w for w in triple_item['persona'].split() if w not in stopwords]
            g_sentence = triple_item['gold'].split()
            p_stemmed_sentence = ' '.join(p_sentence)
            # 포터 알고리즘을 사용하여 어간 추출
            s = PorterStemmer()
            p_stem = {s.stem(w):w for w in p_sentence}
            g_stem = {s.stem(w):w for w in g_sentence}

            for j, p_token in enumerate(p_stem.keys()):
                for k, g_token in enumerate(g_stem.keys()):
                    if p_token == g_token:
                        p_stemmed_sentence = p_stemmed_sentence.replace(p_stem[p_token], g_stem[g_token])

            results['beforegold'] = triple_item['beforegold']
            results['gold'] = triple_item['gold']
            results['persona'] = p_stemmed_sentence
            results['originP'] = triple_item['persona']
            processed_persona.append(results)
        json.dump(processed_persona, w, separators=(',', ': '))



    # 페르소나 토큰과 골드 토큰의 표제어=기본 사전형 단어 비교를 통해 겹치는 게 있는지 확인

# Utility function
def trimQuote(sentence):
    unquote_token = []
    #unquote_result = []
    temp = sentence.split()
    for token in temp:
        if re.findall(r"\'+", token):
            token = re.sub(r"\'+", " ", token)
        unquote_token.append(token)

    return (' '.join(unquote_token))

def countQuoteWord(paths):

    count_path_persona = paths['personachat_train_1']
    quote_persona_list = []
    quote_gold_list = []
    with open(count_path_persona, 'r', encoding='utf-8') as g:
        pc_data = json.load(g)

    for i, data in enumerate(pc_data):
        for j, persona in enumerate(data['personality']):
            if re.findall(r"\'+", persona):
                #quote_persona.append(persona)
                temp = persona.split()
                for token in temp:
                    if re.findall(r"\'+", token):
                        quote_persona_list.append(token)

        for k, gold in enumerate(data['history']):
            if re.findall(r"\'+", gold):
                #quote_gold.append(gold)
                temp = gold.split()
                for token in temp:
                    if re.findall(r"\'+", token):
                        quote_gold_list.append(token)

    persona_wordCount = {}
    for word in quote_persona_list:
        persona_wordCount[word] = persona_wordCount.get(word, 0) + 1

    persona_wordCount = sorted(persona_wordCount.items(), reverse=True, key=lambda item:item[1])

    gold_wordCount = {}
    for word in quote_gold_list:
        gold_wordCount[word] = gold_wordCount.get(word, 0) + 1

    gold_wordCount = sorted(gold_wordCount.items(), reverse=True, key=lambda item:item[1])
    print(gold_wordCount)
    print("Complete")

def countDigit(paths, mode):
    if mode == "train":
        resource_path = paths['triple_result_3']
    else:
        pass

    with open(resource_path, 'r', encoding='utf-8') as f:
        triple_data = json.load(f)

        persona_digit = []
        persona_List = []
        gold_digit = []
        gold_List = []

        for idx, item in enumerate(triple_data):
            digitP = re.findall(r'\d+', item['persona'])
            if digitP:
                persona_digit.extend(digitP)
                persona_List.append(item['persona'])

            digitG = re.findall(r'\d+', item['gold'])
            if digitG:
                gold_digit.extend(digitG)
                gold_List.append(item['gold'])

        p_set = set(persona_digit)
        g_set = set(gold_digit)
        total_digit = p_set | g_set
        total_digit = sorted(list(total_digit), key=int)
        print("digit count complete")

    # make num2word dictionary
    num2word = {}
    for i, digit in enumerate(total_digit):
        temp = num2words(digit)
        if '-' in temp:
            temp = temp.replace('-', ' ')
            num2word[str(digit)] = temp
        else:
            num2word[str(digit)] = temp

    return num2word

def check(paths, mode):
    if mode == "train":
        skeleton_path = paths['raw_skeleton']
        result = paths['result_skeleton']
    else:
        pass

    with open(skeleton_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(result, 'w', encoding='utf-8') as w:
        results = []
        for i, data_item in enumerate(data):
            result_list = {}
            result_list['beforegold'] = data_item['history']
            result_list['gold'] = data_item['gold']
            result_list['persona'] = data_item['persona']
            result_list['skeleton'] = data_item['raw_skeleton_gold']
            results.append(result_list)
        json.dump(results, w, separators=(',', ': '))


if __name__ == '__main__':
    ''' 0차 _ 전처리 없음 : Triple 11,722'''
    ''' 1차 전처리 : 온점 + 공백 제거''' # 21049
    ''' 2 전처리 : single quote 제거 ''' #21050
    ''' i have rainbow hair  / i ve rainbow hair  /    i've rainbow hair  '''
    ''' DNLI 데이터 셋에는 애초에 single quote를 가진 문장이 없음'''
    ''' 그래서 PC 데이터에서 single quote를 제거하여 기표 스타일을 일치 시켜줌'''
    ''' single quote 제거 형태를 맞춰줬는데 달랑 트리플 1개가 증가했다? 축약형과 완성형 혼재가 많은거라 가정하고 실험을 추가 진행'''
    ''' 3차 전처리 : 축약형 표현을 완전체 표현을 바꾸기 ''' #31112
    ''' 앞에 2차에서 가정했던 대로, PC데이터와 DNLI 사이에는 동일한 문장에 표현(축약/완성)을 달리한 경우가 많음을 확인할 수 있었음 '''

    paths = {}
    paths['dnli_train_0'] = "data/dialogue_nli_train.json"
    paths['dnli_dev_0'] = "data/dialogue_nli_dev.jsonl"
    paths['dnli_train_1'] = "trimdata/edited_dnli_train.json"
    paths['dnli_dev_1'] = "trimdata/edited_dnli_dev.json"
    paths['personachat_train_0'] = "trimdata/personachat_self_original_exp3.json"
    paths['personachat_train_1'] = "trimdata/edited_pc_exp3.json"

    paths['triple_train_1'] = "trimdata/edited_triple_train.json"
    # edit_pc(paths)
    # edit_dnli(paths, "train")
    # # edit_dnli(paths, "dev")

    # makeEditTriple(paths)


    paths['personachat_train_2'] = "trimdata/pc_2_Quote.json"



    paths['personachat_train_3'] = 'trimdata/pc_3_Expression.json'
    paths['dnli_train_3'] = 'trimdata/dnli_3_Expression.json'
    # editForMatchingDNLI(paths, "train")
    # editForMatchingPC(paths, "train")
    # editForMatchingDNLI2(paths, "train")
    # editForMatchingPC2(paths, "train")

    # '''--------------------------'''
    paths['triple_result_0'] = 'trimdata/triple_0.json'
    paths['triple_result_1'] = 'trimdata/triple_1_PCDNLI_Dot.json'
    paths['triple_result_2'] = 'trimdata/triple_2_PC_Quote.json'
    paths['triple_result_3'] = 'trimdata/triple_3_PCDNLI_Expression.json'
    paths['triple_result_4'] = 'trimdata/triple_4_PCDNLI_Digit.json'
    #makeTriple(paths, "train")
    #editForTriple(paths, "train")

    ''' LCS를 위한 전처리'''
    # paths['triple_result_5'] = 'trimdata/triple_5_ForLCS.json'
    # editForLCS(paths, "train")


    ''' Count 함수 '''
    # countQuoteWord(paths)
    #countDigit(paths, "train")

    ''' Data Check'''
    paths['raw_skeleton'] = 'dataForFinal/train_skeletons.json'
    paths['result_skeleton'] = 'dataForFinal/train_skeleton_show.json'
    # check(paths, "train")

    paths['origin_pair_train'] = 'data/gold_pair_train.json'
    paths['origin_pair_dev'] = 'data/gold_pair_dev.json'

    with open(paths['origin_pair_train'], 'r', encoding='utf-8') as ot:
        data1 = json.load(ot) # 11722개

    with open(paths['origin_pair_dev'], 'r', encoding='utf-8') as od:
        data2 = json.load(od) # 796

    with open('trimdata/edited_triple_train.json', 'r', encoding='utf-8') as ot:
        data3 = json.load(ot) # 33578개 / 2071 개

    with open('trimdata/edited_triple_train.json', 'r', encoding='utf-8') as ot:
        data3 = json.load(ot) # 33578개 / 2071 개

    print("check")