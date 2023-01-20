import json
import pickle
import time
import pandas as pd
from tqdm import tqdm


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def forRank2():
    with open(rank2_read_file_path, "rb") as f:
        dataset = json.load(f)
        rank2_context = []
        rank2_persona = []
        context_info = []
        c = 0
        print("train set: ", len(dataset['train']))
        for i in range(0, len(dataset['train'])):
            dialogue_i = dataset['train'][i]
            num_of_context = len(dialogue_i['utterances'])
            user_id_A = "conv" + str(c) + "A"
            user_id_B = "conv" + str(c) + "B"

            # persona
            flag = False
            persona = []
            persona.extend(dialogue_i['personality'])
            rank2_persona.append([user_id_B, persona])

            # context
            for j in range(0, num_of_context):
                exp_persona = dialogue_i['utterances'][j]['used_exp_persona']
                num_of_exp_persona = len(exp_persona)
                new_responses = dialogue_i['utterances'][j]['new_responses']
                num_of_new_response = len(new_responses)
                if num_of_exp_persona == 1 or num_of_new_response == 0:
                    continue
                for p in exp_persona:
                    if flag is False:
                        persona.append(p)
                flag = True
                context = dialogue_i['utterances'][j]['history']
                for idx, r in enumerate(new_responses):
                    temp = context.copy()
                    temp.append(r[0])
                    new_context = []
                    for k in range(0, len(temp), 2):
                        new_context.append([user_id_A, [temp[k]]])
                        new_context.append([user_id_B, [temp[k + 1]]])
                    context_info.append([i, j, num_of_new_response, idx, 't'])
                    rank2_context.append(new_context)
            c += 1

        print("valid set: ", len(dataset['valid']))
        for i in range(0, len(dataset['valid'])):
            dialogue_i = dataset['valid'][i]
            num_of_context = len(dialogue_i['utterances'])
            user_id_A = "conv" + str(c) + "A"
            user_id_B = "conv" + str(c) + "B"

            # persona
            flag = False
            persona = []
            persona.extend(dialogue_i['personality'])
            rank2_persona.append([user_id_B, persona])

            # context
            for j in range(0, num_of_context):
                exp_persona = dialogue_i['utterances'][j]['used_exp_persona']
                num_of_exp_persona = len(exp_persona)
                new_responses = dialogue_i['utterances'][j]['new_responses']
                num_of_new_response = len(new_responses)
                if num_of_exp_persona == 1 or num_of_new_response == 0:
                    continue
                for p in exp_persona:
                    if flag is False:
                        persona.append(p)
                flag = True
                context = dialogue_i['utterances'][j]['history']
                for idx, r in enumerate(new_responses):
                    temp = []
                    temp.extend(context)
                    temp.append(r[0])
                    new_context = []
                    for k in range(0, len(temp), 2):
                        new_context.append([user_id_A, [temp[k]]])
                        new_context.append([user_id_B, [temp[k + 1]]])
                    context_info.append([i, j, num_of_exp_persona, idx, 'v'])
                    rank2_context.append(new_context)
            c += 1

        # response = []
        # for i in rank2_context:
        #     response.append(i[-1])
        # rank2_dict = {'context': rank2_context, 'response': response, 'info': context_info}
        # rank2_dict = pd.DataFrame(rank2_dict)
        # rank2_dict.to_csv('./ranking_dataset_context.tsv', sep='\t', na_rep='NaN')
        #
        # rank2_dict = {'persona': rank2_persona}
        # rank2_dict = pd.DataFrame(rank2_dict)
        # rank2_dict.to_csv('./ranking_dataset_persona.tsv', sep='\t', na_rep='NaN')
    print("new_response num: ", len(rank2_context))

    with open(rank2_context_path, 'wb') as write:
        pickle.dump([rank2_context, context_info], write)

    with open(rank2_context_path_txt, 'w') as write_txt:
        write_txt.write(json.dumps(load_pickle(rank2_context_path)))

    with open(rank2_persona_path, 'wb') as write:
        pickle.dump(rank2_persona, write)

    with open(rank2_persona_path_txt, 'w') as write_txt:
        write_txt.write(json.dumps(load_pickle(rank2_persona_path)))


def main():
    forRank2()


if __name__ == "__main__":
    rank2_read_file_path = "./perturb3/perturb3_final.json"
    rank2_context_path = "./perturb3/perturb3_context.pkl"
    rank2_context_path_txt = "./perturb3/perturb3_context.txt"
    rank2_persona_path = "./perturb3/perturb3_persona.pkl"
    rank2_persona_path_txt = "./perturb3/perturb3_persona.txt"

    start = time.time()
    main()
    print("code time elapse: ", str(time.time() - start))
