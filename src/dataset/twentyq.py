import os 
import pandas as pd 
import torch 
import random
import json
from dataset.common import train_split_entities, train_split_designs, make_proportion_dict
from pathlib import Path 
import argparse
from tqdm import trange

def construct_prompt(questions, answers, possible_answers):
    prompt = ""
    answer_list = []
    for i in range(len(questions)):
        prompt += questions[i]
        prompt += "\n"
        prompt += "<Answer>"
        prompt += possible_answers[answers[i]]
        answer_list.append(possible_answers[answers[i]])
        prompt += "\n<EOS>"
    prompt += "<EOP>"
    return prompt, answer_list


def construct_text(questions, answers_df, possible_answers):
    text_data = ""
    proportion_list = []
    for i in trange(answers_df.shape[0]):
        object = answers_df.index[i]
        answers = answers_df.loc[object]
        prompt, answer_list = construct_prompt(questions, answers, possible_answers)
        text_data += prompt
        proportion_list.append(answer_list)
    proportion_dict = make_proportion_dict(proportion_list)
    return text_data, proportion_dict

def process_data(
    root_data_dir="data", 
    noobj_qs="general_questions.txt",
    noobj_ans="general_answers.csv",
    obj_qs="object_questions.txt",
    obj_ans="object_answers.csv", 
    ):
    possible_answers = ["No", "Sometimes", "Often", "Yes"]
    
    #get questions that do not involve objects
    data_dir = Path(root_data_dir) / "raw" / "twentyq"
    with open(data_dir / noobj_qs, 'r') as f:
        questions = f.read().split('\n')

    answers_df = pd.read_csv(data_dir / noobj_ans, index_col=0)
    noobj_text_data, noobj_proportion_dict = construct_text(questions, answers_df, possible_answers)

    with open(data_dir / obj_qs, 'r') as f:
        obj_questions = f.read().split('\n')
    obj_answers_df = pd.read_csv(data_dir / obj_ans, index_col=0)
    questions = questions + obj_questions
    answers_df = pd.concat((answers_df, obj_answers_df), axis=1)
    obj_text_data, obj_proportion_dict = construct_text(questions, answers_df, possible_answers)
    return (noobj_text_data, noobj_proportion_dict), (obj_text_data, obj_proportion_dict)


def save_train_test(text_data, proportion_dict, save_dir, split_ratio=0.85, seed=0):
    train, test = train_split_entities(text_data, split_ratio, seed)
    #split by designs 
    train_qs, _ = train_split_designs(train, split_ratio, seed)
    _, test_qs = train_split_designs(test, split_ratio, seed)
    train_qs, val_qs = train_split_designs(train_qs, split_ratio, seed)
    data_dict = {'train': train_qs, 'val': val_qs, 'test': test_qs}
    with open(save_dir / "split_designs.json", 'w') as f:
        json.dump(data_dict, f)
    #split by entities
    train, val = train_split_entities(train, split_ratio=0.85, seed=0)
    data_dict = {'train': train, 'val': val, 'test': test, 'proportion_dict': proportion_dict}
    
    with open(save_dir / "split_entities.json", 'w') as f:
        json.dump(data_dict, f)
    
    
    

if __name__ == "__main__":
    print("Processing data")
    noobj_data, obj_data = process_data(
        root_data_dir = "data", 
        noobj_qs = "general_questions.txt", 
        noobj_ans = "general_answers.csv",
        obj_qs = "object_questions.txt",
        obj_ans = "object_answers.csv")
    print("Data processed")
    
    noobj_text_data, noobj_proportion_dict = noobj_data
    noobj_save_dir = Path("data") / "processed" / "twentyq_noobj"
    noobj_save_dir.mkdir(parents=True, exist_ok=True)
    save_train_test(noobj_text_data, noobj_proportion_dict, noobj_save_dir)

    obj_text_data, obj_proportion_dict = obj_data
    obj_save_dir = Path("data") / "processed" / "twentyq_obj"
    obj_save_dir.mkdir(parents=True, exist_ok=True)
    save_train_test(obj_text_data, obj_proportion_dict, obj_save_dir)

    




