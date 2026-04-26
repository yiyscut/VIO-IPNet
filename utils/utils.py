import torch
import numpy as np

def save_state(out_states:dict, in_state:dict):
    for k, v in in_state.items():
        if v is None:
            continue
        elif isinstance(v, dict):
            save_state(out_states=out_states, in_state=v)
        elif k in out_states.keys():
            out_states[k].append(v)
        else:
            out_states[k] = [v]
        
def cat_state(in_state:dict):
    pop_list = []
    for k, v in in_state.items():
        if len(v[0].shape) > 2:
            in_state[k] = torch.cat(v,  dim=-2)
        else:
            pop_list.append(k)
    for k in pop_list:
        in_state.pop(k)

def move_to(obj, device):
    if torch.is_tensor(obj):return obj.to(device)
    elif obj is None:
        return None
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    elif isinstance(obj, np.ndarray):
        return torch.tensor(obj).to(device)
    else:
        raise TypeError("Invalid type for move_to", type(obj))
