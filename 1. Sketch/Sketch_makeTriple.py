import os
import json
import jsonlines

from tqdm import tqdm


import os
import json
import jsonlines

from tqdm import tqdm


def get_dnli(paths, mode):

    if mode == "train":
        dnli_path = paths['dnli_raw_path_train']
        combi_path = paths['dnli_combi_path_train']
    elif mode == "valid":
        dnli_path = paths['dnli_raw_path_dev']
        combi_path = paths['dnli_combi_path_dev']
    else:
        dnli_path = paths['dnli_raw_path_test']
        combi_path = paths['dnli_combi_path_test']

    with open(dnli_path, 'r', encoding='utf-8') as f:
        #json_data = jsonlines.Reader(f)
        json_data = json.load(f)
        train_data = []
        for index, item in enumerate(json_data):
            train_data.append(item)


    positive_combies = {}
    results = []

    with open(combi_path, 'w', encoding="utf-8") as f:
        pair = {}
        for index, data in enumerate(tqdm(train_data)):
            if data['label'] == 'positive':
                positive_combies['gold'] = data['sentence1']
                positive_combies['persona'] = data['sentence2']
                copy = positive_combies.copy()
                results.append(copy)
            else:
                continue

        json.dump(results, f, separators=(',', ': '))

def get_goldpair(paths, mode):

    if mode == "train":
        persona_path = paths['persona_raw_path']
        dnli_combi_path = paths['dnli_combi_path_train']
        pair_path = paths['gold_pair_path_train']
    elif mode == "valid":
        persona_path = paths['persona_raw_path']
        dnli_combi_path = paths['dnli_combi_path_dev']
        pair_path = paths['gold_pair_path_dev']

    with open(persona_path, 'r', encoding='utf-8') as p:
        json_personachat = json.load(p)

    with open(dnli_combi_path, 'r', encoding='utf-8') as d:
        json_combi = json.load(d)

    with open(pair_path, 'w', encoding='utf-8') as g:

        results = []
        gold_pair = {}

        for i, combi_item in enumerate(tqdm(json_combi)):
            for j, chat_item in enumerate(json_personachat[mode]):

                # 특정 페르소나 및 골드가 있는지 확인하는 프로세스
                combi_item['persona'] = "i love iced tea ."
                combi_item['gold'] = 'that is a good show i watch that while drinking iced tea .'
                if combi_item['gold'] in chat_item['utterances'][-1]['history']:
                    print("catch")

        for i, chat_item in enumerate(json_personachat[mode]):
            if chat_item['utterances'][-1]['history']:

                # if combi_item['persona'] in chat_item['personality']:
                #     if combi_item['gold'] in chat_item['utterances'][-1]['history']:
                #         history_index = chat_item['utterances'][-1]['history'].index(combi_item['gold'].strip())
                #         gold_pair['persona'] = combi_item['persona'].strip()
                #         gold_pair['gold'] = combi_item['gold'].strip()
                #         gold_pair['beforegold'] = chat_item['utterances'][-1]['history'][history_index-1]
                #         copy = gold_pair.copy()
                #         results.append(copy)
                #     else:
                #         continue

        #json.dump(results, g, separators=(',', ': '))

def get_count(paths, mode):

    yes = 0
    no = 0
    if mode == "train":
        persona_path = paths['persona_raw_path']
        dnli_combi_path = paths['dnli_combi_path_train']
        pair_path = paths['gold_pair_path_train']

    with open(persona_path, 'r', encoding='utf-8') as p:
        json_personachat = json.load(p)

    with open(dnli_combi_path, 'r', encoding='utf-8') as d:
        json_combi = json.load(d)

    # for i, combi_item in enumerate(tqdm(json_combi)):
    #     for j, chat_item in enumerate(json_personachat[mode]):
    #         if combi_item['persona'] in chat_item['personality']:
    #             if combi_item['gold'] in chat_item['utterances'][-1]['history']:
    #                 yes += 1
    #             else:
    #                 no += 1

    for j, chat_item in enumerate(tqdm(json_personachat[mode])):
        if 'my favorite color is blue .' in chat_item['personality']:
            if 'blue is mine too ! i actually just bought a blue car !' in chat_item['utterances'][-1]['history']:
                print("count")


    #
    # print(yes)
    # print(no)


if __name__ == '__main__':
    paths = {}

    paths['dnli_raw_path_train'] = "data/dialogue_nli_train.json"
    paths['dnli_raw_path_dev'] = "data/dialogue_nli_dev.jsonl"
    paths['dnli_raw_path_test'] = "data/dialogue_nli_test.jsonl"

    paths['dnli_combi_path_train'] = "dataForWG/positive_combi_train.json"
    paths['dnli_combi_path_dev'] = "dataForWG/positive_combi_dev.json"
    paths['dnli_combi_path_test'] = "data/positive_combi_test.json"

    paths['persona_raw_path'] = "data/personachat_self_original.json"


    paths['gold_pair_path_train'] = "dataForWG/gold_pair_train_test.json"
    paths['gold_pair_path_dev'] = "dataForWG/gold_pair_dev.json"
    #paths['gold_pair_path_test'] = "data/gold_pair_test2.json"


    #get_dnli(paths, "train")
    get_goldpair(paths, "train")
    #get_count(paths, "train")
    #
    # get_dnli(paths, "valid")
    # get_goldpair(paths, "valid")

    #get_dnli(paths, "test")
    #get_goldpair(paths, "test")