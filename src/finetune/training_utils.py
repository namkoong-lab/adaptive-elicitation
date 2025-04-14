import math
import torch
import torch.nn.functional as F
import json 

def load_dataset(data_dir):
    with open(data_dir) as f:
        data_dict = json.load(f)
    return data_dict


def get_lr(it, warmup_iters = 2000, lr_decay_iters = 600000, min_lr = 6e-5, learning_rate = 6e-4):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters: 
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def get_loss(model, X, Y, gradient_mask, mask=True, return_scalar=True):
    outputs = model(input_ids=X)
    logits = outputs.logits
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1, reduction='none')
    if mask:
        loss = loss.reshape(-1, X.shape[1]) * gradient_mask
        if not return_scalar:
            return loss    
        return torch.mean(loss[loss != 0])
    return torch.mean(loss)

@torch.no_grad()
def estimate_loss(model, dataset, split, device, batch_size, block_size, eval_iters=100):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y, gradient_mask = dataset.get_batch(split, batch_size, block_size)
        X, Y, gradient_mask = X.to(device), Y.to(device), gradient_mask.to(device)                          
        losses[k] = get_loss(model, X, Y, gradient_mask, mask=True, return_scalar=True)
    out = losses.mean().item()
    model.train()
    return out