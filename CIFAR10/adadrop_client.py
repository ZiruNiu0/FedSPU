# FLower:
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import Code, EvaluateIns, EvaluateRes, FitRes, Status
# other dependecies:
from models import CNN
import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict
from util import set_filters, get_filters, merge_subnet, get_subnet
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Status
from typing import List
from copy import deepcopy
import numpy as np

DEVICE = torch.device('cpu')
CLASSES = 10
CHANNELS = 3
OTHER_PARAMS = ['bn1.num_batches_tracked', 'bn2.num_batches_tracked']

class adadrop_client(fl.client.Client):
    def __init__(self, cid, dataset, rate, epoch, batch):
        self.cid = cid
        self.model = CNN(in_channels=CHANNELS, outputs=CLASSES).to(DEVICE)
        self.testmodel = CNN(in_channels=CHANNELS, outputs=CLASSES).to(DEVICE)
        self.local_epoch = epoch
        self.local_batch_size= batch
        self.sub_model_rate = rate
        self.Masks = None
        len_train = int(len(dataset) * 0.7)
        len_test = len(dataset) - len_train
        ds_train, ds_val = random_split(dataset, [len_train, len_test], torch.Generator().manual_seed(25))
        self.trainloader = DataLoader(ds_train, self.local_batch_size, shuffle=True)
        self.testloader = DataLoader(ds_val, self.local_batch_size, shuffle=False)
    
    def fit(self, ins: FitIns) -> FitRes:
        # Deserialize parameters to NumPy ndarray's
        sub_params = ins.parameters
        drop_info = ins.config['drop_info']
        needmask = ins.config['needmask']
        personal_model = ins.config['personal model']
        set_filters(self.model, personal_model)
        # Update local model, train, get updated parameters
        merged_params = merge_subnet(parameters_to_ndarrays(sub_params), personal_model, drop_info)
        set_filters(self.model, merged_params)
        # masking channels:
        if needmask:
            masks = self.get_channel_mask()
            drop_info = self.mask_channels(masks)
            self.Masks = masks
        self.train()
        # Serialize ndarray's into a Parameters object
        parameters_updated = get_subnet(self.model, drop_info)
        status = Status(code=Code.OK, message="Success")
        return FitRes(status=status, parameters=ndarrays_to_parameters(parameters_updated), num_examples=len(self.trainloader), metrics={"drop_info":drop_info, "personal model": get_filters(self.model)},)
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        set_filters(self.testmodel, parameters_to_ndarrays(parameters_original))
        loss, accuracy = self.test() # return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.testloader),
            metrics={"accuracy": float(accuracy)},
        )

    def train(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005)
        self.model.train()
        for e in range(self.local_epoch):
            for samples, labels in self.trainloader:
                samples, labels = samples.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(samples)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def test(self):
        """Evaluate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.testmodel.eval()
        with torch.no_grad():
            for samples, labels in self.testloader:
                samples, labels = samples.to(DEVICE), labels.to(DEVICE)
                outputs = self.testmodel(samples)
                loss = criterion(outputs, labels).item() * labels.size(0)
                total += labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += predicted.eq(labels).sum()
        loss = loss / total
        accuracy = correct / total
        return loss, accuracy
    
    def get_channel_mask(self):
        if self.sub_model_rate >= 0.99:
            return None
        maskmodel = CNN(in_channels=CHANNELS, outputs=CLASSES).to(DEVICE)
        maskmodel.train()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(maskmodel.parameters(), lr=0.005)
        for e in range(1):
            for samples, labels in self.trainloader:
                samples, labels = samples.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = maskmodel(samples)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                break
        param_dict = maskmodel.state_dict()
        masks = {'conv1.weight':[], 'conv2.weight':[], 'fc1.weight':[], 'fc2.weight':[]}
        for k in ['conv1.weight','conv2.weight']:
            channel_norms = {}
            num_channel = 6 if k == 'conv1.weight' else 16
            num_masks = max(int(num_channel) * self.sub_model_rate, 1)
            for w in range(num_channel):
                channel = param_dict[k][w]
                norm = torch.norm(channel, p=2)
                channel_norms[w] = norm.item()
            sorted_channel_norms = sorted(channel_norms.items(), key=lambda x:x[1], reverse=True)
            i = 0
            while i < num_masks:
                masks[k].append(sorted_channel_norms[i][0])
                i += 1
        for k in ['fc1.weight', 'fc2.weight']:
            channel_norms = {}
            num_channel = 120 if k == 'fc1.weight' else 84
            num_masks = max(int(num_channel) * self.sub_model_rate, 1)
            for w in range(num_channel):
                channel = param_dict[k][w]
                norm = channel.pow(2).sum().pow(0.5)
                channel_norms[w] = norm
            sorted_channel_norms = sorted(channel_norms.items(), key=lambda x:x[1], reverse=True)
            i = 0
            while i < num_masks:
                masks[k].append(sorted_channel_norms[i][0])
                i += 1
        return masks
    
    def mask_channels(self, masks):
        if masks == None:
            return {}
        mask1 = sorted(masks['conv1.weight'])
        mask2 = sorted(masks['conv2.weight'])
        mask3 = sorted(masks['fc1.weight'])
        mask4 = sorted(masks['fc2.weight'])
        param_dict = self.model.state_dict()
        model_params = get_filters(self.model)
        layer_count = 0
        for k in param_dict.keys():
            if k not in OTHER_PARAMS and (k in ['conv1.weight', 'conv1.bias'] or 'bn1' in k):
                params = model_params[layer_count]
                for w in range(params.shape[0]):
                    if w not in mask1:
                        params[w] = 0.0
                layer_count += 1
            elif k not in OTHER_PARAMS and (k in ['conv2.weight', 'conv2.bias'] or 'bn2' in k):
                params = model_params[layer_count]
                for w in range(params.shape[0]):
                    if w not in mask2:
                        params[w] = 0.0
                    elif k == 'conv2.weight':
                        for w_ in range(params.shape[1]):
                            if w_ not in mask1:
                                params[w][w_] = 0.0
                layer_count += 1
            elif k == 'fc1.weight':
                params = model_params[layer_count]
                for w in range(params.shape[0]):
                    if w not in mask3:
                        params[w] = 0.0
                    else:
                        for w_ in range(params.shape[1]):
                            if int(w_/64) not in mask2:
                                params[w][w_] = 0.0
                layer_count += 1
            elif k == 'fc2.weight':
                params = model_params[layer_count]
                for w in range(params.shape[0]):
                    if w not in mask4:
                        params[w] = 0.0
                    else:
                        for w_ in range(params.shape[1]):
                            if w_ not in mask3:
                                params[w][w_] = 0.0
                layer_count += 1
            elif k == 'fc.weight':
                params = model_params[layer_count]
                for w in range(CLASSES):
                    for w_ in range(params.shape[1]):
                        if w_ not in mask4:
                            params[w][w_] = 0.0
                layer_count += 1
        set_filters(self.model, model_params)
        drop_info = {}
        if self.sub_model_rate >= 0.99:
            return drop_info
        for k in param_dict.keys():
            if k not in OTHER_PARAMS:
                if 'conv1' in k or 'bn1' in k:
                    drop_info[k] = mask1
                elif 'conv2' in k or 'bn2' in k:
                    drop_info[k] = mask2
                elif k == 'fc.bias':
                    drop_info[k] = list(range(CLASSES))
                elif k == 'fc1.weight':
                    drop_info[k] = mask3
                elif k == 'fc2.weight' or k == 'fc.weight':
                    drop_info[k] = mask4
        return drop_info