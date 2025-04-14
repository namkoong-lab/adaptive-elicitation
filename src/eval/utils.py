import torch


def get_corresponding_logits(model, context, answer_token):
    with torch.no_grad():
        logits = model(input_ids=context).logits
    last_indices = (context == answer_token).cumsum(dim=1).argmax(dim=1)
    logits_at_indices = logits[torch.arange(logits.size(0), device=model.device), last_indices]
    next_tokens = context[torch.arange(context.size(0), device=model.device), last_indices + 1]
    return logits_at_indices, next_tokens

def get_target_condition_probs(model, context, targets=None, answer_token=None, possible_answers_tokens=None):
    softmax = torch.nn.Softmax(dim=-1)
    if targets is not None:
        context = torch.cat((context, targets), dim=1)
    context = context.to(model.device)
    logits_at_indices, next_tokens = get_corresponding_logits(model, context, answer_token)
    probs = softmax(logits_at_indices[:, possible_answers_tokens.to(model.device)])
    return probs, next_tokens

def eval_metrics(model, context, targets, answer_token):
    softmax = torch.nn.Softmax(dim=-1)
    metric_dict = {}
    if context is None:
        context = targets.to(model.device)
    else:
        context = torch.cat((context, targets), dim=1).to(model.device)
    logits_at_indices, next_tokens = get_corresponding_logits(model, context, answer_token)
    logits_at_indices = softmax(logits_at_indices)
    accuracy = (torch.argmax(logits_at_indices, dim=-1) == next_tokens).float()
    nll = - torch.mean(torch.log(logits_at_indices[torch.arange(logits_at_indices.size(0)), next_tokens])).item()
    return accuracy.mean().item(), nll

