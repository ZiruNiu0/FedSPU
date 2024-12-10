import numpy as np
import random
import torch
from typing import Dict, List
from copy import deepcopy
from collections import OrderedDict
from flwr.server.strategy.aggregate import aggregate
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitRes

FILTER_PARAMS = ['conv1.weight', 'conv2.weight', 'fc.weight', 'fc1.weight', 'fc2.weight']
OTHER_PARAMS = ['bn1.num_batches_tracked', 'bn2.num_batches_tracked']
NUM_CHANNELS = 3
CLASSES = 10

Learnable_Params = ['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias', 
                    'conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias',
                    'fc.weight', 'fc.bias', 'bn1.running_mean', 'bn1.running_var',
                    'bn2.running_mean', 'bn2.running_var', 'fc1.weight', 'fc2.weight',]

def get_parameters(net:torch.nn.Module) -> List[np.ndarray]: # Access the parameters of a neural network 
  return [val.cpu().numpy() for _, val in net.state_dict().items()]

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
    subparams = []
    for name in param_dict.keys():
        if name not in OTHER_PARAMS:
            if name == 'conv1.weight':
                w = param_dict[name]
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = sorted(random.sample(list(range(total_filters)), num_selected_filters))
                sub_param = torch.index_select(w,0,torch.tensor(non_masked_filter_ids))
            elif name in ['conv1.bias', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var']:
                w = param_dict[name] 
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = drop_information['conv1.weight']
                sub_param = torch.index_select(w,0,torch.tensor(non_masked_filter_ids))
            elif name == 'conv2.weight':
                w = param_dict[name]
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = sorted(random.sample(list(range(total_filters)), num_selected_filters))
                lastindices = drop_information['conv1.weight']
                sub_param_1 = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
                sub_param = torch.index_select(sub_param_1, 1, torch.tensor(lastindices))
            elif name in ['conv2.bias', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var']:
                w = param_dict[name] 
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = drop_information['conv2.weight']
                sub_param = torch.index_select(w,0,torch.tensor(non_masked_filter_ids))
            elif name == 'fc1.weight':
                w = param_dict[name] 
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = sorted(random.sample(list(range(total_filters)), num_selected_filters))
                lastindices = []
                for q in drop_information['conv2.weight']:
                    for q_ in range(q*8*8, (q+1)*8*8):
                        lastindices.append(q_)
                sub_param_1 = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
                sub_param = torch.index_select(sub_param_1, 1, torch.tensor(lastindices))
            elif name == 'fc2.weight':
                w = param_dict[name]
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = sorted(random.sample(list(range(total_filters)), num_selected_filters))
                lastindices = drop_information['fc1.weight']
                sub_param_1 = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
                sub_param = torch.index_select(sub_param_1, 1, torch.tensor(lastindices))
            elif name == 'fc.weight':
                w = param_dict[name] 
                non_masked_filter_ids = drop_information['fc2.weight']
                sub_param = torch.index_select(w,1,torch.tensor(non_masked_filter_ids))
            else: # fc.bias
                w = param_dict[name] 
                sub_param = w
                non_masked_filter_ids = list(range(CLASSES))
            drop_information[name] = non_masked_filter_ids
            subparams.append(sub_param.numpy())
    return drop_information, subparams

def generate_subnet_ordered(global_model:torch.nn.Module, rate):
    drop_information = {}
    if rate >= 0.99:
        return drop_information, get_filters(global_model)
    param_dict = global_model.state_dict()
    subparams = []
    for name in param_dict.keys():
        if name not in OTHER_PARAMS:
            if name == 'conv1.weight':
                w = param_dict[name]
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = list(range(num_selected_filters))
                sub_param = torch.index_select(w,0,torch.tensor(non_masked_filter_ids))
            elif name in ['conv1.bias', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var']:
                w = param_dict[name] 
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = drop_information['conv1.weight']
                sub_param = torch.index_select(w,0,torch.tensor(non_masked_filter_ids))
            elif name == 'conv2.weight':
                w = param_dict[name]
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = list(range(num_selected_filters))
                lastindices = drop_information['conv1.weight']
                sub_param_1 = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
                sub_param = torch.index_select(sub_param_1, 1, torch.tensor(lastindices))
            elif name in ['conv2.bias', 'bn2.weight', 'bn2.bias', 'bn2.running_mean', 'bn2.running_var']:
                w = param_dict[name] 
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = drop_information['conv2.weight']
                sub_param = torch.index_select(w,0,torch.tensor(non_masked_filter_ids))
            elif name == 'fc1.weight':
                w = param_dict[name] 
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = list(range(num_selected_filters))
                lastindices = []
                for q in drop_information['conv2.weight']:
                    for q_ in range(q*8*8, (q+1)*8*8):
                        lastindices.append(q_)
                sub_param_1 = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
                sub_param = torch.index_select(sub_param_1, 1, torch.tensor(lastindices))
            elif name == 'fc1.bias':
                w = param_dict[name] 
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = drop_information['fc1.weight']
                sub_param = torch.index_select(w,0,torch.tensor(non_masked_filter_ids))
            elif name == 'fc2.weight':
                w = param_dict[name]
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = list(range(num_selected_filters))
                lastindices = drop_information['fc1.weight']
                sub_param_1 = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
                sub_param = torch.index_select(sub_param_1, 1, torch.tensor(lastindices))
            elif name == 'fc2.bias':
                w = param_dict[name] 
                total_filters = w.shape[0]
                num_selected_filters = max(1, int(total_filters * rate))
                non_masked_filter_ids = drop_information['fc2.weight']
                sub_param = torch.index_select(w,0,torch.tensor(non_masked_filter_ids))
            elif name == 'fc.weight':
                w = param_dict[name] 
                non_masked_filter_ids = drop_information['fc2.weight']
                sub_param = torch.index_select(w,1,torch.tensor(non_masked_filter_ids))
            else: # fc.bias
                w = param_dict[name] 
                sub_param = w
                non_masked_filter_ids = list(range(CLASSES))
            drop_information[name] = non_masked_filter_ids
            subparams.append(sub_param.numpy())
    return drop_information, subparams

def compute_update(w1: List[np.ndarray], w2: List[np.ndarray])-> List[np.ndarray]:
    result = []
    for w1_, w2_ in zip(w1, w2):
        result.append(w1_ - w2_)
    return result

def compute_norm(w:List[np.ndarray]):
    v = []
    for w_ in w:
        v = np.append(v, w_.flatten())
    n = np.dot(v,v)
    return n

def merge_subnet(sub_params, full_params, drop_info) -> List[np.ndarray]:
        if len(drop_info) == 0:
            return sub_params
        layer_count = 0
        result = []
        for k in drop_info.keys():
            selected_filters = drop_info[k]
            full_layer = deepcopy(full_params[layer_count])
            sub_layer = sub_params[layer_count]
            if k == "conv1.weight":
                i1 = 0
                for f in selected_filters:
                    j1 = 0
                    for j in list(range(NUM_CHANNELS)):
                        full_layer[f][j] = sub_layer[i1][j1]
                        j1 += 1
                    i1 += 1
            elif k == 'conv1.bias' or 'bn1' in k:
                i1 = 0
                for f in selected_filters:
                    full_layer[f] = sub_layer[i1]
                    i1 += 1
            elif k == 'conv2.weight':
                i1 = 0
                for f in selected_filters:
                    j1 = 0
                    for j in drop_info['conv1.weight']:
                        full_layer[f][j] = sub_layer[i1][j1]
                        j1 += 1
                    i1 += 1
            elif k == 'conv2.bias' or 'bn2' in k:
                i1 = 0
                for f in selected_filters:
                    full_layer[f] = sub_layer[i1]
                    i1 += 1
            elif k == 'fc1.weight':
                i1 = 0
                for f in selected_filters:
                    j1 = 0
                    for j in drop_info['conv2.weight']:
                        for q in range(j*8*8, (j+1)*8*8):
                            full_layer[f][q] = sub_layer[i1][j1]
                            j1 += 1
                    i1 += 1
            elif k == 'fc2.weight':
                i1 = 0
                for f in selected_filters:
                    j1 = 0
                    for j in drop_info['fc1.weight']:
                        full_layer[f][j] = sub_layer[i1][j1]
                        j1 += 1
                    i1 += 1
            elif k == "fc.bias":
                full_layer = sub_layer
                #for f in range(CLASSES):
                #    full_layer[f] = sub_layer[f]
            else: # k == fc.weight
                for f in range(CLASSES):
                    j1 = 0
                    for j in drop_info['fc2.weight']:
                        full_layer[f][j] = sub_layer[f][j1]
                        j1 += 1
            result.append(full_layer)
            layer_count += 1
        return result

def get_subnet(model:torch.nn.Module, drop_info, channels=NUM_CHANNELS, classes=CLASSES):
        if len(drop_info) == 0:
            return get_filters(model)
        sub_params = []
        full_params = get_filters(model)
        last_layer_indices = list(range(channels))
        layer_count = 0
        for k in drop_info.keys():
            l = list(range(classes)) if (k == 'fc.bias' or k == 'fc.weight') else drop_info[k]
            filters = []
            for f in l:
                if k == 'fc.bias':
                    filters.append(full_params[layer_count][f])
                elif k == 'fc.weight':
                    weights = []
                    for number in drop_info[k]:
                        weights.append(full_params[layer_count][f][number])
                    filters.append(weights)
                elif k == 'fc1.weight':
                    weights = []
                    for number in last_layer_indices:
                        for q in range(number*8*8, (number+1)*8*8):
                            weights.append(full_params[layer_count][f][q])
                    filters.append(weights)
                elif k == 'fc2.weight':
                    weights = []
                    for number in last_layer_indices:
                        weights.append(full_params[layer_count][f][number])
                    filters.append(weights)
                elif k != "conv1.weight" and k != "conv2.weight":
                    for number in last_layer_indices:
                        filters.append(full_params[layer_count][number])
                else:
                    weights = []
                    for weight_count in last_layer_indices:
                        weights.append(full_params[layer_count][f][weight_count])
                    filters.append(weights)
            sub_params.append(np.array(filters))
            last_layer_indices = drop_info[k]
            layer_count += 1
        return sub_params    

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
                selected_filters = merge_info[k]
                layer = param[layer_count]
                i1 = 0
                if k in Learnable_Params and not (k in FILTER_PARAMS):
                    for f in selected_filters:
                        if (layer_count, f) in Aggregation_Dict.keys():
                            Aggregation_Dict[(layer_count, f)].append(([layer[i1]], num))
                        else:
                            Aggregation_Dict[(layer_count, f)] = [([layer[i1]], num)]
                elif k == 'fc1.weight':
                    for f in selected_filters:
                        j1 = 0
                        for j_ in last_layer_indices:
                            for j in range(j_*8*8, (j_+1)*8*8):
                                if (layer_count,f,j) in Aggregation_Dict.keys():
                                    Aggregation_Dict[(layer_count,f,j)].append(([layer[i1][j1]], num))
                                else:
                                    Aggregation_Dict[(layer_count,f,j)] = [([layer[i1][j1]], num)]
                                j1 += 1
                        i1 += 1
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

def jaccard_distance(set1:set, set2:set):
    #Symmetric difference of two sets
    Symmetric_difference = set1 & set2
    # Unions of two sets
    union = set1.union(set2)
    return len(Symmetric_difference)/len(union)

def get_neuron_sets(masks):
    mask1 = sorted(masks['conv1.weight'])
    mask2 = sorted(masks['conv2.weight'])
    mask3 = sorted(masks['fc1.weight'])
    mask4 = sorted(masks['fc2.weight'])
    neuron_index_set = set()
    for i in mask1:
        neuron_index_set.add(('conv1',i))
    for i in mask2:
        neuron_index_set.add(('conv2',i))
    for i in mask3:
        neuron_index_set.add(('fc1',i))
    for i in mask4:
        neuron_index_set.add(('fc2',i))
    return neuron_index_set