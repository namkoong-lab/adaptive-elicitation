import pickle as pkl
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
from dataset.common import train_split_entities, make_proportion_dict
from pathlib import Path
import json

def load_opinion_qa(data_dir):
    PEW_SURVEY_LIST = [26, 27, 29, 32, 34, 36, 41, 42, 43, 45, 49, 50, 54, 82, 92]
    data_dir = Path(data_dir)
    response_matrix = np.load(data_dir / "response_matrix.npy")
    with open(data_dir / "mappings.pkl", "rb") as f:
        mapping_dicts = pkl.load(f)

    RESULTS_DIR = data_dir / "human_resp"
    info_df = []
    for wave in PEW_SURVEY_LIST:

        SURVEY_NAME = f'American_Trends_Panel_W{wave}'

        idf = pd.read_csv(RESULTS_DIR / f'{SURVEY_NAME}/info.csv')
        idf['survey'] = f'ATP {wave}'
        info_df.append(idf)

    info_df = pd.concat(info_df)
    return response_matrix, info_df, mapping_dicts


def get_opinion_qa_prompt(row, qidx, qid_to_name, info_df, max_num_answers, all_answers):
    prompt = ""
    answer_list = []
    for qid in qidx:
        qname = qid_to_name[qid]
        qinfo = info_df[info_df["key"]==qname]
            
        if len(qinfo) != 1:
            continue
        else:
            qinfo = qinfo.iloc[0].to_dict()

            question = qinfo["question"]
        
        answers_dict = eval(qinfo["option_mapping"])
        options = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        if len(answers_dict) <= max_num_answers+1:
            prompt += f"Question: {question}\n"
            answer = options[int(row[qid])-1]
            if all_answers:
                count = 0
                for key, value in answers_dict.items():
                    if key != 99.0 and value != 'Refused':
                        prompt += f"{options[count]}: {value}\n"
                        count+=1
                prompt+=f"<Answer>{answer}\n"
            else:
                prompt+=f"<Answer>{answer}: {eval(qinfo['option_mapping'])[row[qid]]}\n"
            prompt+="<EOS>"
            answer_list.append(answer)
    if prompt != "":
        prompt+="<EOP>"
    return prompt, answer_list


def process_data(
    root_data_dir,
    cutoff,
    max_num_answers,
    all_answers
):
    data_dir = Path(root_data_dir) / "raw" / "opinion_qa"
    response_matrix, info_df, mapping_dicts = load_opinion_qa(data_dir)
    pid_to_name = mapping_dicts["pid_to_name"]
    qid_to_name = mapping_dicts["qid_to_name"]
    pname_to_race = mapping_dicts["pname_to_race"]

    bin_mat = np.maximum(np.minimum(response_matrix, 1), 0)
    q_means = np.sum(bin_mat, 0)

    if cutoff is None:
        filtered = response_matrix
        all_idx = np.arange(len(pid_to_name))
    else:
        filtered = response_matrix[:, q_means>cutoff]
        all_idx = np.where(np.all(filtered > 0, axis=1) >0)[0]

    text_data = ""
    proportion_list = []
    for i in tqdm(all_idx):
        row = response_matrix[i]
        race = pname_to_race[pid_to_name[i]]
        if cutoff is None:
            answered = np.where(row > 0)[0]
        else:
            answered = np.where(q_means > cutoff)[0]
        prompt, answer_list = get_opinion_qa_prompt(row, answered, qid_to_name, info_df, max_num_answers, all_answers)
        text_data+=f"{race}<race>"
        text_data += prompt
        proportion_list.append(answer_list)
    proportion_dict = make_proportion_dict(proportion_list)
    return text_data, proportion_dict


def get_race_list(text_data):
    race_list = []
    new_text_data = ""
    person_data = text_data.strip("<EOP>").split("<EOP>")
    for person in person_data:
        race, prompt = person.split("<race>")
        race_list.append(race)
        new_text_data+=prompt
        new_text_data += "<EOP>"
    return new_text_data, race_list


if __name__ == "__main__":
    seed=0
    text_data, proportion_dict = process_data("data", cutoff=7000, max_num_answers=4, all_answers=False)
    train, test = train_split_entities(text_data, split_ratio=0.85, seed=seed)
    train, val = train_split_entities(train, split_ratio=0.85, seed=seed)
    train, train_race_list = get_race_list(train)
    val, val_race_list = get_race_list(val)
    test, test_race_list = get_race_list(test)
    data_dict = {'train': train, 'val': val, 'test': test, 'proportion_dict': proportion_dict, 'train_race_list': train_race_list, 'val_race_list': val_race_list, 'test_race_list': test_race_list}
    save_dir = Path("data") / "processed" / "opinion_qa"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "split_entities.json", 'w') as f:
        json.dump(data_dict, f)




        








    
