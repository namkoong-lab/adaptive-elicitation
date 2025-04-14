import torch 
import random
import torch.nn.functional as F
import os 
import json 
from torch.nn.utils.rnn import pad_sequence

def shuffle_row(row, block_size):
    random.shuffle(row)
    row = torch.tensor([item for sublist in row for item in sublist])
    return row[:block_size]

def get_batch(data, batch_size, block_size, answer_token):
    batch_data = []
    print(len(data))
    print(len(data[0]))
    while len(batch_data) < batch_size:
        row_idx = torch.randint(0, len(data), (1,))
        row = shuffle_row(data[row_idx], block_size+1)
        if len(row) == block_size+1:
            batch_data.append(row)
    batch_data = torch.stack(batch_data)   
    X = batch_data[:, :-1].contiguous()
    Y = batch_data[:, 1:].contiguous()
    gradient_mask = 1*(X == answer_token)
    print(X.shape, Y.shape)
    return X, Y, gradient_mask

def train_split_entities(text_data, split_ratio=0.85, seed=0):
    random.seed(seed)
    text_data = text_data.strip('<EOP>').split('<EOP>')
    n_entities = len(text_data)
    indices = random.sample(range(n_entities), n_entities)
    split_idx = int(n_entities * split_ratio)
    train_data, test_data = "", ""
    for idx in indices:
        if idx < split_idx:
            train_data += text_data[idx] + '<EOP>'
        else:
            test_data += text_data[idx] + '<EOP>'
    return train_data, test_data

def train_split_designs(text_data, split_ratio=0.85, seed=0):
    random.seed(seed)
    text_data = text_data.strip('<EOP>').split('<EOP>')
    n_designs = len(text_data[0].strip('<EOS>').split('<EOS>'))
    indices = random.sample(range(n_designs), n_designs)
    split_idx = int(n_designs * split_ratio)
    train_data, test_data = "", ""
    for entity in text_data:
        designs = entity.strip('<EOS>').split('<EOS>')
        train_data += '<EOS>'.join([designs[idx] for idx in indices[:split_idx]]) + '<EOS>' + '<EOP>'
        test_data += '<EOS>'.join([designs[idx] for idx in indices[split_idx:]]) + '<EOS>' + '<EOP>'
    return train_data, test_data

def make_proportion_dict(proportion_list):
    proportion_dict = {i: {} for i in range(len(proportion_list[0]))} 
    for answer_list in proportion_list:
        for i, answer in enumerate(answer_list):
            if answer not in proportion_dict[i]:
                proportion_dict[i][answer] = 1
            else:
                proportion_dict[i][answer] += 1
    for key, answer_dict in proportion_dict.items():
        total = sum(answer_dict.values())
        for answer, count in answer_dict.items():
            proportion_dict[key][answer] = count / total
    return proportion_dict


class TextDataClass():
    def __init__(self, text_data_dict, tokenizer):
        self.tokenizer = tokenizer
        self.answer_token = tokenizer("<Answer>")['input_ids'][-1]
        self.data_dict = self.get_token_dict(text_data_dict)

    def get_batch(self, split, batch_size, block_size):
       return get_batch(self.data_dict[split], batch_size, block_size, self.answer_token)
    
    def turn_text_into_tokens(self, text):
        text_list = [sentence for sentence in text.strip("<EOP>").split("<EOP>") if sentence]
        text_list = [text_list[i].strip("<EOS>").split("<EOS>") for i in range(len(text_list))]
        tokens_list = [self.tokenizer(text_list[i])['input_ids'] for i in range(len(text_list))]
        return tokens_list
    
    def get_token_dict(self, text_data_dict):
        tokens_dict = {}
        for key, value in text_data_dict.items():
            if key in ['train', 'val', 'test']:
                tokens_dict[key] = self.turn_text_into_tokens(value)
            elif key in ['proportion_dict']:
                new_proportion_dict = {}
                for q_id, answer_dict in value.items():
                    new_dict = {}
                    for answer, value in answer_dict.items():
                        text = "<Answer>" + answer
                        converted = self.tokenizer(text, return_tensors='pt')['input_ids'][:, -1]
                        new_dict[converted.item()] = value
                    new_proportion_dict[int(q_id)] = new_dict
                tokens_dict[key] = new_proportion_dict
            else:
                tokens_dict[key] = value         
        return tokens_dict

        