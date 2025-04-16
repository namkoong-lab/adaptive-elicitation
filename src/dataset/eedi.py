import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import ast
from dataset.common import train_split_entities, make_proportion_dict
from pathlib import Path
import json 
from tqdm import tqdm

def load_eedi(data_dir):
    data_dir = Path(data_dir)
    question_metadata = pd.read_csv(data_dir / "question_metadata.csv")   
    subject_metadata = pd.read_csv(data_dir / "subject_metadata.csv")
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test_public.csv")
    with open(data_dir / "questions.txt") as f:
        question_text = f.readlines()

    return question_metadata, subject_metadata, train, test, question_text
 
def create_question_prompts(i, questions, question_metadata, subject_metadata):
    subject_ids = question_metadata[question_metadata['QuestionId'] == i]['SubjectId'].iloc[0]
    prompt = ""
    prompt+= "Subjects:"
    for subject_id in ast.literal_eval(subject_ids):
        subject = subject_metadata[subject_metadata['SubjectId'] == subject_id]['Name'].iloc[0]
        prompt+= f" {subject},"
    prompt+= " question: "
    prompt+= questions[i]
    return prompt

def create_student_prompts(student_data, question_prompt_list, student_idx, question_idx):
    unique_students = student_data['UserId'].unique()
    student_prompt = ""
    proportion_list = []
    for i, student in enumerate(unique_students):
        if student_idx[i]:
            answer_list = []
            temp_student_data = student_data[student_data["UserId"] == student]
            for qid in question_idx:
                row = temp_student_data[temp_student_data["QuestionId"] == qid]
                student_prompt+=question_prompt_list[qid]
                student_prompt+="Correct Answer?"
                student_prompt+="<Answer>"
                if row['IsCorrect'].iloc[0] == 0:
                    student_prompt+="No\n"
                    answer_list.append("No")
                else:
                    student_prompt+="Yes\n"
                    answer_list.append("Yes")
                student_prompt+="<EOS>"
            student_prompt+="<EOP>"
            proportion_list.append(answer_list)
    proportion_dict = make_proportion_dict(proportion_list)
    return student_prompt, proportion_dict

def filter_students(student_data, total_qs, num_filtered_qs):
    unique_students = student_data['UserId'].unique()
    answered_matrix = np.zeros((len(unique_students), total_qs))
    for i, student in tqdm(enumerate(unique_students), total=len(unique_students)):
        temp_student_data = student_data[student_data["UserId"] == student]
        for row in temp_student_data.iterrows():
            qid = int(row[1]['QuestionId'])
            if qid < total_qs:
                answered_matrix[i, qid] = 1
    indices = np.argsort(np.sum(answered_matrix, axis=0))[::-1][:num_filtered_qs]
    subset = answered_matrix[:, indices]
    students_answered_all = np.all(subset==1, axis=1)
    return students_answered_all, indices
    


def process_data(root_data_dir, total_qs, num_filtered_qs):
    data_dir = Path(root_data_dir) / "raw" / "eedi"
    question_metadata, subject_metadata, train, test, question_text = load_eedi(data_dir)
    student_data = pd.concat([train, test], axis=0)
    student_idx, question_idx = filter_students(student_data, total_qs, num_filtered_qs)
    question_prompt_list = [create_question_prompts(i, question_text, question_metadata, subject_metadata) for i in range(total_qs)]
    text_data, proportion_dict = create_student_prompts(student_data, question_prompt_list, student_idx, question_idx)
    return text_data, proportion_dict

if __name__ == "__main__":
    text_data, proportion_dict = process_data(root_data_dir = "data", total_qs = 250, num_filtered_qs=112)
    train, test = train_split_entities(text_data, split_ratio=0.85, seed=0)
    train, val = train_split_entities(train, split_ratio=0.85, seed=0)
    data_dict = {'train': train, 'val': val, 'test': test, 'proportion_dict': proportion_dict}
    save_dir = Path("data") / "processed" / "eedi"
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "split_entities.json", 'w') as f:
        json.dump(data_dict, f)