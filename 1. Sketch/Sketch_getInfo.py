import os
import json
import jsonlines

from tqdm import tqdm


import os
import json
import jsonlines

from tqdm import tqdm

def get_dnli_info(paths, mode):

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
            if item['dtype'] == 'matchingtriple_up':
                train_data.append(item)
            else:
                continue
        print(len(train_data))

def get_info(paths, mode):

    if mode == "train":
        persona_path = paths['persona_raw_path']
        pair_path = paths['gold_pair_path_train']
        info_path = paths['info_path_train']

    with open(pair_path, 'r', encoding='utf-8') as f:
        json_pair = json.load(f)

    with open(persona_path, 'r', encoding='utf-8') as p:
        json_personachat = json.load(p)

    with open(info_path, 'w', encoding='utf-8') as k:
        result = [['none', 'none', 'none', 'none']]
        count = 0

        # for i, pair_item in enumerate(tqdm(json_pair)):
        #     for j, chat_item in enumerate(json_personachat[mode]):
        #         if pair_item['persona'] in chat_item['personality']:
        #             if pair_item['gold'] in chat_item['utterances'][-1]['history']:
        #                 if chat_item['personality'][0] != result[-1][0] or chat_item['personality'][1] != result[-1][1]\
        #                     or chat_item['personality'][2] != result[-1][2] or chat_item['personality'][3] != result[-1][3]:
        #                     result.append(chat_item['personality'])
        #                 else:
        #                     print("same dialogue")

        for i, chat_item in enumerate(tqdm(json_personachat[mode])):
            for j, pair_item in enumerate(json_pair):
                if pair_item['persona'] in chat_item['personality']:
                    if pair_item['gold'] in chat_item['utterances'][-1]['history']:
                        try:
                            # if chat_item['personality'][0] != result[-1][0] or chat_item['personality'][1] != result[-1][1] \
                            #     or chat_item['personality'][2] != result[-1][2] or chat_item['personality'][3] != result[-1][3]:
                            #     result.append(chat_item['personality'])
                            # else:
                            #     count += 1
                            if chat_item['personality'][:] == result[-1][:]:
                                count += 1
                            else:
                                result.append(chat_item['personality'])
                        except:
                            continue

        print(count)
        print("complete")
        print(len(result))
        json.dump(result, k, separators=(',', ': '))



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
                # combi_item['persona'] == "i believe that mermaids are real .":
                # combi_item['gold'] = 'i am a researcher i am researching the fact that mermaids are real .'
                # if combi_item['gold'] in chat_item['utterances'][-1]['history']:
                #     print("catch")
                if combi_item['persona'] in chat_item['personality']:
                    if combi_item['gold'] in chat_item['utterances'][-1]['history']:
                        history_index = chat_item['utterances'][-1]['history'].index(combi_item['gold'].strip())
                        gold_pair['persona'] = combi_item['persona'].strip()
                        gold_pair['gold'] = combi_item['gold'].strip()
                        gold_pair['beforegold'] = chat_item['utterances'][-1]['history'][history_index-1]
                        copy = gold_pair.copy()
                        results.append(copy)
                    else:
                        continue

        json.dump(results, g, separators=(',', ': '))

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


    paths['dnli_combi_path_train'] = "data/positive_combi_train.json"
    paths['dnli_combi_path_dev'] = "data/positive_combi_dev.json"
    paths['dnli_combi_path_test'] = "data/positive_combi_test.json"

    paths['persona_raw_path'] = "data/personachat_self_original.json"

    paths['gold_pair_path_train'] = "data/gold_pair_train_original.json"
    paths['gold_pair_path_dev'] = "data/gold_pair_test2.json"
    #paths['gold_pair_path_test'] = "data/gold_pair_test2.json"

    paths['info_path_train'] = "data/info.json"

    #get_dnli(paths, "train")
    # get_goldpair(paths, "train")
    #get_count(paths, "train")

    get_info(paths, "train")
    #
    # get_dnli(paths, "valid")
    #get_goldpair(paths, "valid")

    #get_dnli(paths, "test")
    #get_goldpair(paths, "test")

    #get_dnli_info(paths, "train")