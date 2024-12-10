import numpy as np
import random
import torch
from typing import Dict, List
from copy import deepcopy
from collections import OrderedDict
from flwr.server.strategy.aggregate import aggregate
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes

FILTER_PARAMS = ['conv1.weight', 'conv2.weight', 'fc.weight']
OTHER_PARAMS = ['bn1.num_batches_tracked', 'bn2.num_batches_tracked']
NUM_CHANNELS = 1
CLASSES = 62
Learnable_Params = ['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias', 
                    'conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias',
                    'fc.weight', 'fc.bias', 'bn1.running_mean', 'bn1.running_var',
                    'bn2.running_mean', 'bn2.running_var']

def get_parameters(net:torch.nn.Module) -> List[np.ndarray]: # Access the parameters of a neural network 
  return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]): # modify the parameters of a neural network
  params_dict = zip(net.state_dict().keys(), parameters)
  state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
  net.load_state_dict(state_dict, strict=False)

def get_filters(net:torch.nn.Module) -> List[np.ndarray]:
    params_list = []
    for k, v in net.state_dict().items():
        if k not in OTHER_PARAMS:
            params_list.append(v.cpu().numpy())       
    return params_list

def set_filters(net:torch.nn.Module, parameters: List[np.ndarray]): # modify the parameters of a neural network
    param_set_index = 0
    all_names = []
    all_params = []
    old_param_dict = net.state_dict()
    for k, _ in old_param_dict.items():
        if k not in OTHER_PARAMS:
            all_params.append(parameters[param_set_index])
            all_names.append(k)
            param_set_index += 1
    params_dict = zip(all_names, all_params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=False)

def generate_filters_random(global_model:torch.nn.Module, rate):
    drop_information = {}
    if rate >= 0.99:
        return drop_information, get_filters(global_model)
    param_dict = global_model.state_dict()
    old_indices = None
    subparams = []
    for name in param_dict.keys():
        if name not in OTHER_PARAMS:
            w = param_dict[name]
            num_filters = w.shape[0]
            num_selected_filters = max(1, int(num_filters * rate))
            if name == 'fc.weight':
                num_filters = w.shape[1]
            non_masked_filter_ids = sorted(random.sample(list(range(num_filters)), num_selected_filters))
            if name == "conv1.weight":
                non_masked_filter_ids = sorted(random.sample(list(range(num_filters)), num_selected_filters))
                sub_param = torch.index_select(w,0,torch.tensor(non_masked_filter_ids))
            elif name == 'conv2.weight':
                non_masked_filter_ids = sorted(random.sample(list(range(num_filters)), num_selected_filters))
                sub_param_1 = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
                sub_param = torch.index_select(sub_param_1, 1, torch.tensor(old_indices))
            elif name == 'fc.bias':
                sub_param = torch.index_select(w,0,torch.tensor(list(range(CLASSES))))
            elif name != 'fc.weight':
                non_masked_filter_ids = old_indices
                sub_param = torch.index_select(w,0,torch.tensor(old_indices))
            else:
                fcindices = []
                for f in old_indices:
                    for i in range(f*7*7, (f+1)*7*7):
                        fcindices.append(i)
                sub_param = torch.index_select(w, 1, torch.tensor(fcindices))
                non_masked_filter_ids = fcindices
            drop_information[name] = non_masked_filter_ids
            subparams.append(sub_param.numpy())
            old_indices = non_masked_filter_ids
    drop_information['fc.bias'] = list(range(CLASSES))
    return drop_information, subparams

def generate_subnet_ordered(global_model:torch.nn.Module, rate):
    drop_information = {}
    if rate >= 0.99:
        return drop_information, get_filters(global_model)
    param_dict = global_model.state_dict()
    subparams = []
    for name in param_dict.keys():
        if name not in OTHER_PARAMS:
            w = param_dict[name]
            non_masked_filter_ids = []
            l = int(20*rate)*7*7 if name == 'fc.weight' else int(w.shape[0] * rate)
            for i in range(l):
                if len(non_masked_filter_ids) < l: # this filter is not getting dropped
                    non_masked_filter_ids.append(i)
                else:
                    break
            drop_information[name] = non_masked_filter_ids
            if name == "conv1.weight":
                sub_param = torch.index_select(w,0,torch.tensor(non_masked_filter_ids))
            elif name == 'conv2.weight':
                sub_param_1 = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
                sub_param = torch.index_select(sub_param_1, 1, torch.tensor(non_masked_filter_ids))
            elif name == 'fc.bias':
                sub_param = torch.index_select(w,0,torch.tensor(list(range(CLASSES))))
            elif name == 'fc.weight':
                sub_param = torch.index_select(w,1,torch.tensor(non_masked_filter_ids))
            else:
                sub_param = torch.index_select(w,0,torch.tensor(non_masked_filter_ids))
            subparams.append(sub_param.numpy())
            drop_information[name] = non_masked_filter_ids
    drop_information['fc.bias'] = list(range(CLASSES))
    return drop_information, subparams

def get_rate(cid):
    drop_rate = 1.0
    if int(cid) < 20:
      drop_rate = 0.2
    elif int(cid) < 40:
      drop_rate = 0.4
    elif int(cid) < 60:
      drop_rate = 0.6
    elif int(cid) < 80:
      drop_rate = 0.8
    else:
      drop_rate = 1.0
    return drop_rate

def merge_subnet(sub_params, full_params, drop_info) -> List[np.ndarray]:
        if len(drop_info) == 0:
            return sub_params
        layer_count = 0
        result = []
        last_layer_indices = list(range(NUM_CHANNELS))
        for k in drop_info.keys():
            selected_filters = drop_info[k]
            full_layer = deepcopy(full_params[layer_count])
            sub_layer = sub_params[layer_count]
            i1 = 0
            if k == "conv1.weight" or k == "conv2.weight":
                for f in selected_filters:
                    j1 = 0
                    for j in last_layer_indices:
                        full_layer[f][j] = sub_layer[i1][j1]
                        j1 += 1
                    i1 += 1
            elif k == "fc.bias":
                for f in range(CLASSES):
                    full_layer[f] = sub_layer[f]
            elif k != "fc.weight":
                j1 = 0
                for f in selected_filters:
                    full_layer[f] = sub_layer[j1]
                    j1 += 1
            else:
                for f in range(CLASSES):
                    j1 = 0
                    for j in last_layer_indices:
                        full_layer[f][j] = sub_layer[f][j1]
                        j1 += 1
            result.append(full_layer)
            layer_count += 1
            last_layer_indices = selected_filters
        return result

def get_subnet(model:torch.nn.Module, rate, num_filters=20):
        drop_info = {}
        if rate >= 0.99:
            return drop_info, get_filters(model)
        num_selected_filters = max(int(rate*20), 1)
        mask1 = sorted(random.sample(list(range(num_filters)), num_selected_filters))
        mask2 = sorted(random.sample(list(range(num_filters)), num_selected_filters))
        param_dict = model.state_dict()
        model_params = get_filters(model)
        layer_count = 0
        for k in param_dict.keys():
            if k not in OTHER_PARAMS and (k in ['conv1.weight', 'conv1.bias'] or 'bn1' in k):
                drop_info[k] = mask1
                params = model_params[layer_count]
                for w in range(num_filters):
                    if w not in mask1:
                        params[w] = 0
                layer_count += 1
            elif k not in OTHER_PARAMS and (k in ['conv2.weight', 'conv2.bias'] or 'bn2' in k):
                drop_info[k] = mask2
                params = model_params[layer_count]
                for w in range(num_filters):
                    if w not in mask2:
                        params[w] = 0
                layer_count += 1
            elif k == 'fc.weight':
                drop_info[k] = []
                for m in mask2:
                    for m_ in range(m*7*7, (m+1)*7*7):
                        drop_info[k].append(m_)
                params = model_params[layer_count]
                for w in range(CLASSES):
                    for w_ in range(params.shape[1]):
                        if w_ not in drop_info[k]:
                            params[w][w_] = 0
            elif k == 'fc.bias':
                drop_info[k] = list(range(CLASSES))
        return drop_info, model_params

def spu_aggregation(Fit_res:List[FitRes], global_param:List[np.ndarray]):
    Aggregation_Dict = {}
    Aggregated_params = {}
    full_results = []
    for fit_res in Fit_res:
        param, num, merge_info = parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples, fit_res.metrics["drop_info"]
        if len(merge_info) == 0:
            full_results.append((param, num))
            for l1 in range(len(param)):
                layer = param[l1]
                for l2 in range(len(layer)):
                    filter = layer[l2]
                    if len(layer.shape) == 3:
                        for l3 in range(len(filter)):
                            if (l1,l2,l3) in Aggregation_Dict.keys():
                                Aggregation_Dict[(l1,l2,l3)].append(([filter[l3]], num))
                            else:
                                Aggregation_Dict[(l1,l2,l3)] = [([filter[l3]], num)]
                    else:
                        if (l1,l2) in Aggregation_Dict.keys():
                            Aggregation_Dict[(l1,l2)].append(([filter], num))
                        else:
                            Aggregation_Dict[(l1,l2)] = [([filter], num)]
        else:
            last_layer_indices = list(range(NUM_CHANNELS))
            layer_count = 0
            for k in merge_info.keys():
                #print(f"layer name == {k}")
                selected_filters = merge_info[k]
                layer = param[layer_count]
                i1 = 0
                if k in Learnable_Params and not (k in FILTER_PARAMS):
                    for f in selected_filters:
                        if (layer_count, f) in Aggregation_Dict.keys():
                            Aggregation_Dict[(layer_count, f)].append(([layer[i1]], num))
                        else:
                            Aggregation_Dict[(layer_count, f)] = [([layer[i1]], num)]
                elif k != "fc.weight":
                    for f in selected_filters:
                        j1 = 0
                        for j in last_layer_indices:
                            if (layer_count,f,j) in Aggregation_Dict.keys():
                                Aggregation_Dict[(layer_count,f,j)].append(([layer[i1][j1]], num))
                            else:
                                Aggregation_Dict[(layer_count,f,j)] = [([layer[i1][j1]], num)]
                            j1 += 1
                        i1 += 1
                else:
                    for f in range(CLASSES):
                        j1 = 0
                        for j in last_layer_indices:
                            if (layer_count,f,j) in Aggregation_Dict.keys():
                                Aggregation_Dict[(layer_count,f,j)].append(([layer[f][j1]], num))
                            else:
                                Aggregation_Dict[(layer_count,f,j)] = [([layer[f][j1]], num)]
                            j1 += 1
                layer_count += 1
                last_layer_indices = selected_filters
    for z, p in Aggregation_Dict.items():
        Aggregated_params[z] = aggregate(p)
    full_param = aggregate(full_results) if len(full_results) > 0 else deepcopy(global_param)
    for Key in Aggregated_params.keys():
        if len(Key) == 2:
            layer_idx, filter = Key
            full_param[layer_idx][filter] = Aggregated_params[Key][0]
        else:
            layer_idx, filter, last_filter = Key
            full_param[layer_idx][filter][last_filter] = Aggregated_params[Key][0]
    return full_param