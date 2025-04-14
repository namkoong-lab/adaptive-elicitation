from torch.nn.utils.rnn import pad_sequence
import torch 
import random
import torch 
import random 
from torch.nn.utils.rnn import pad_sequence
import torch 
import random 
from torch.nn.utils.rnn import pad_sequence

class ActiveEnv():

    def __init__(self, dataset, num_targets, num_designs, split, race_list=None, proportion=None):
        self.num_targets = num_targets
        self.num_designs = num_designs
        self.data = dataset.data_dict[split]
        self.pad_token = dataset.tokenizer('[PAD]')['input_ids'][-1]
        self.answer_token = dataset.answer_token
        self.race_list = race_list
        self.proportion = None
        if proportion is not None:
            self.proportion_dict = dataset.data_dict['proportion_dict']
            self.proportion = proportion
        
    def step(self, context, action):
        self.designs, chosen_context, self.design_idx, qid = self.get_chosen_context(self.designs, action, self.design_idx)
        if self.designs.shape[0] == 0:
            done = True
        else:
            done = False
        context = torch.cat((context, chosen_context), dim=1)  
        return context, self.designs, qid, done
     
    def reset(self):
        while True:
            idx = torch.randint(0, len(self.data), (1,))
            avg_prob=0
            # make sure there are enough designs and targets
            if len(self.data[idx]) > self.num_targets + self.num_designs + 1:
                self.curr_entity = [torch.tensor(seq, dtype=torch.int64) for seq in self.data[idx]]
                random_idx = torch.randperm(len(self.curr_entity)).tolist()
                target_idx = random_idx[:self.num_targets]
                design_idx = random_idx[self.num_targets:self.num_targets+self.num_designs+1]
                targets = [self.curr_entity[i] for i in target_idx]
                designs = [self.curr_entity[i] for i in design_idx]
                if self.proportion is not None:
                    avg_prob = 0  
                    for target, qid  in zip(targets, target_idx):
                        answer = target[torch.where(target == self.answer_token)[0]+1]
                        answer_prob = self.proportion_dict[qid][answer.item()]
                        avg_prob += answer_prob
                    avg_prob /= len(targets)
                    if avg_prob < self.proportion:
                        break
                else: 
                    break
        self.target_idx = target_idx
        self.design_idx = design_idx
        self.targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_token)
        self.designs = pad_sequence(designs, batch_first=True, padding_value=self.pad_token)
        
        randn = torch.randint(0, len(designs), (1,))
        self.designs, context, self.design_idx, qid = self.get_chosen_context(self.designs, randn, self.design_idx)
        return context, self.designs, self.targets, avg_prob
    
    def get_chosen_context(self, designs, action, design_idx = None):
        context = designs[action:action+1]
        if (context==self.pad_token).sum() > 0:
            pad_idx = torch.nonzero(context == self.pad_token)[0][-1]
            context = context[:, :pad_idx]
        designs = torch.cat([designs[:action], designs[action+1:]])
        if design_idx is None:
            return designs, context, None, None
        elif design_idx is not None:
            qid = design_idx.pop(action)
            return designs, context, design_idx, qid

    
