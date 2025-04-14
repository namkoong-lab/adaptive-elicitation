# Code based off Karpathy nanoGPT

from finetune.training_utils import *
import torch 
import os 
import pandas as pd 
import math 
import yaml 
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import wandb
from dataset.common import TextDataClass
from peft import get_peft_model, LoraConfig
# Dataset settings 


def train(args):
# load model 
    print("loading model") 
    print(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.add_special_tokens({'additional_special_tokens': ["<EOS>", "<EOP>", "[PAD]", "<Answer>"]})
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)
    model.resize_token_embeddings(len(tokenizer))
    if args.peft: 
        lora_config = LoraConfig(r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=["q_proj", "v_proj"])
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    print("loading data")
    text_data_dict = load_dataset(args.data_dir)
    dataset = TextDataClass(text_data_dict, tokenizer)
    #wandb 
    if args.wandb:
        import wandb # 'run' + str(time.time())
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=args.betas, weight_decay=args.weight_decay)

    best_val_loss = 1e9
    print(dataset.data_dict.keys())
    for iter_num in range(args.epochs):
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % args.eval_interval == 1:
            losses = {}
            for split in ['train', 'val', 'test']:
                losses[split] = estimate_loss(model, dataset, split, args.device, args.batch_size, args.block_size)
            print(f'iter: {iter_num}', losses)
            if args.wandb:
                wandb.log({"iter": iter_num, "lr": lr})
                for key, item in losses.items():
                    wandb.log({key: item})
            if losses['val'] < best_val_loss and args.save_dir is not None:
                best_val_loss = losses['val']
                if iter_num > 0 and args.save_dir is not None:
                    model.save_pretrained(args.save_dir)
                    print(f"saving checkpoint to {args.save_dir}")
        
        # evaluate the loss on train/val sets and write checkpoints
        X,Y, gradient_mask = dataset.get_batch('train', args.batch_size, args.block_size)
        X,Y, gradient_mask = X.to(args.device), Y.to(args.device), gradient_mask.to(args.device)
        loss = get_loss(model, X, Y, gradient_mask, return_scalar=True, mask=True)
        scaler.scale(loss).backward()
        # clip the gradient
        if args.grad_clip != 0.0:
            scaler.unscale_(optimizer) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)




