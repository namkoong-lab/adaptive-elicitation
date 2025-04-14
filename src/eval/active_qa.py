import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import random
import os 
import pandas as pd 
import yaml
from peft import PeftModel
from eval.info_gain import calc_info_gain, MCTS
from eval.args import parse_eval_arguments
from eval.active_qa_env import ActiveEnv
from eval.utils import get_target_condition_probs
import json
from tqdm import trange
from dataset.common import TextDataClass
from finetune.training_utils import load_dataset
from pathlib import Path

def active_qa(args):
    if "Llama" in args.model_name:
        model_name = os.path.join("meta-llama", args.model_name)
    else:
        model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ["<EOS>", "<EOP>", "[PAD]", "<Answer>"]})
    if model_name == 'meta-llama/Llama-3.1-8B' and args.peft == True:
        model = AutoModelForCausalLM.from_pretrained("/scratch/jw4209/hfmodels/Llama-3.1-8B")
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, args.model_path)
    else:    
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        model.resize_token_embeddings(len(tokenizer))
        model_name = args.model_name
    model.to(args.device)
    print("loaded model")
    if args.dataset == 'twentyq':
        example_text = ["<Answer>Yes", "<Answer>No", "<Answer>Sometimes", "<Answer>Often"]
    elif args.dataset == 'opinion_qa': 
        example_text = ["<Answer>A", "<Answer>B", "<Answer>C", "<Answer>D"]
    elif args.dataset == 'eedi':
        example_text = ["<Answer>Yes", "<Answer>No"]
    possible_answer_tokens = tokenizer(example_text, return_tensors='pt')['input_ids'][:, -1]
    results_dict = {sampling_method: [] for sampling_method in args.sampling_list}
    print("loading data")
    text_data_dict = load_dataset(args.data_dir)
    dataset = TextDataClass(text_data_dict, tokenizer)
    active_env = ActiveEnv(dataset, args.n_targets, args.n_designs, args.split, proportion=args.proportion)
    print("starting active learning")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    for _ in trange(args.n_epochs):
        base_context, base_designs, targets, avg_prob = active_env.reset()
        base_designs_idx = active_env.design_idx.copy()
        qids = []
        for sampling_method in args.sampling_list: 
            temp_dict = {}
            active_env.design_idx, active_env.designs = base_designs_idx.copy(), base_designs.clone()
            temp_context, temp_designs = base_context.clone(), base_designs.clone()
            for trial in range(args.trial_length):
                probs, next_token = get_target_condition_probs(model, temp_context.repeat(targets.shape[0], 1), targets, dataset.answer_token, possible_answer_tokens)
                correct_list = (next_token.to(args.device).unsqueeze(1) == possible_answer_tokens.to(args.device)).nonzero(as_tuple=True)[1]
                temp_dict[trial] = probs.cpu().tolist()
                if trial == args.trial_length - 1:
                    break
                if sampling_method == 'random':
                    action = torch.randint(0, temp_designs.shape[0], (1,))
                elif sampling_method == 'mcts':
                    info_gain = MCTS(model, active_env, temp_context, temp_designs, targets, dataset.answer_token, possible_answer_tokens, depth=args.trial_length-trial-2)
                    action = torch.argmax(info_gain)
                elif sampling_method == 'info_gain':
                    info_gain = calc_info_gain(model, temp_context, temp_designs, targets, dataset.answer_token, possible_answer_tokens)
                    action = torch.argmax(info_gain)
                #print(tokenizer.decode(temp_context[0]))
                temp_context, temp_designs, qid, _ = active_env.step(temp_context, action)
                qids.append(qid)
            temp_dict['Correct Answers'] = correct_list.cpu().tolist()
            temp_dict['QID'] = qids
            temp_dict['TargetID'] = active_env.target_idx
            temp_dict['Trial Length'] = args.trial_length
            results_dict[sampling_method].append(temp_dict)

        out_dir = Path(args.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_dir = out_dir / f"{args.dataset}_{args.proportion}_{args.trial_length}_{args.n_targets}_{args.n_designs}_{args.model_name}_results.json"
        with open(out_dir, 'w') as f:
            json.dump(results_dict, f)
