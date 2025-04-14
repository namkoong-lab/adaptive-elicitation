import torch 
from .utils import get_target_condition_probs, get_corresponding_logits

def calc_info_gain(model, context, designs, targets, answer_token, possible_answers_tokens = [3363, 1400, 8975, 18023]):
    #get entropy without adding any designs 
    no_design_probs, _ = get_target_condition_probs(model, context.repeat(targets.shape[0], 1), targets, answer_token, possible_answers_tokens)
    entropy_without_designs = - torch.sum(no_design_probs * torch.log(no_design_probs), dim=-1)

    #get probabilities for each design and answer
    conditional_design_probs, _ = get_target_condition_probs(model, context.repeat(designs.shape[0], 1), designs, answer_token, possible_answers_tokens)
    
    design_entropies = []
    for i in range(targets.shape[0]):
        # get entropy without adding context 
        entropies = []
        for j, token in enumerate(possible_answers_tokens):
            repeated_target = targets[i].repeat(designs.shape[0], 1)
            temp = designs.clone()
            mask = torch.roll(designs==answer_token, 1, 1)
            temp[mask] = token

            context_designs = torch.cat((context.repeat(designs.shape[0], 1), temp), dim=1)
            probs, _ = get_target_condition_probs(model, context_designs, repeated_target, answer_token, possible_answers_tokens)
            entropy = -torch.sum(probs * torch.log(probs), dim=-1)
            entropies.append(conditional_design_probs[:, j]*entropy)
        design_entropies.append(torch.sum(torch.stack(entropies), dim=0))
    return torch.mean(entropy_without_designs) - torch.mean(torch.stack(design_entropies), dim=0)

def MCTS(model, env, context, designs, targets, answer_token, possible_answers_tokens = [3363, 1400, 8975, 18023], depth = 2, n_iter = 1):
    info_gain = calc_info_gain(model, context, designs, targets, answer_token, possible_answers_tokens)
    cumul_info_gain = torch.zeros_like(info_gain)
    for i in range(designs.shape[0]):
        action = i
        for _ in range(n_iter):
            temp_context = context.clone()
            temp_designs = designs.clone()
            for _ in range(depth):
                temp_designs, design, _, _ = env.get_chosen_context(temp_designs, action)
                answer_idx = torch.where(design == answer_token)[1] + 1
                temp_context = torch.cat([temp_context, design], dim=1)
                probs, _ = get_target_condition_probs(model, temp_context, answer_token=answer_token, possible_answers_tokens=possible_answers_tokens)
                sample = torch.multinomial(probs, 1)
                sample_token = possible_answers_tokens[sample.to(possible_answers_tokens.device)]
                temp_context[:, torch.where(temp_context == answer_token)[1][-1] + 1] = sample_token
                temp_info_gain = calc_info_gain(model, temp_context, temp_designs, targets, answer_token, possible_answers_tokens)
                cumul_info_gain[i] += torch.max(temp_info_gain)
                action = torch.argmax(temp_info_gain)

    cumul_info_gain /= n_iter
    
    return info_gain + cumul_info_gain