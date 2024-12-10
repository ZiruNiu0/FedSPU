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
import numpy as np
from typing import List

DEVICE = torch.device('cpu')
CLASSES = 10
CHANNELS = 3
OTHER_PARAMS = ['bn1.num_batches_tracked', 'bn2.num_batches_tracked']

class feddrop_client(fl.client.Client):
    def __init__(self, cid, dataset, rate, epoch, batch):
        self.cid = cid
        self.model = CNN(CHANNELS, outputs=CLASSES).to(DEVICE)
        self.testmodel = CNN(in_channels=CHANNELS, outputs=CLASSES).to(DEVICE)
        self.local_epoch = epoch
        self.local_batch_size= batch
        self.sub_model_rate = rate
        len_train = int(len(dataset) * 0.7)
        len_test = len(dataset) - len_train
        ds_train, ds_val = random_split(dataset, [len_train, len_test], torch.Generator().manual_seed(1622))
        self.trainloader = DataLoader(ds_train, self.local_batch_size, shuffle=True)
        self.testloader = DataLoader(ds_val, self.local_batch_size, shuffle=False)

    def fit(self, ins: FitIns) -> FitRes:
        # Deserialize parameters to NumPy ndarray's
        subnet = ins.parameters
        drop_info = ins.config['drop_info']
        sparam = parameters_to_ndarrays(subnet)
        merged_params = merge_subnet(sparam, get_filters(self.model), drop_info)
        masked_params =self.mask_channels(drop_info, model_params=merged_params)
        # Update local model, train, get updated parameters
        set_filters(self.model, masked_params)
        self.train()
        # Serialize ndarray's into a Parameters object
        parameters_updated = get_subnet(self.model, drop_info)
        status = Status(code=Code.OK, message="Success")
        return FitRes(status=status, parameters=ndarrays_to_parameters(parameters_updated), num_examples=len(self.trainloader), metrics={"drop_info":drop_info},)
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        set_filters(self.testmodel, ndarrays_original)
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
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                #correct += (outputs == labels).sum()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum()
        loss /= len(self.testloader.dataset)
        accuracy = correct / total
        return loss, accuracy
    
    def mask_channels(self, drop_info, model_params):
        if len(drop_info) == 0:
            return model_params
        mask1 = sorted(drop_info['conv1.weight'])
        mask2 = sorted(drop_info['conv2.weight'])
        mask3 = sorted(drop_info['fc1.weight'])
        mask4 = sorted(drop_info['fc2.weight'])
        param_dict = self.model.state_dict()
        layer_count = 0
        for k in param_dict.keys():
            if k not in OTHER_PARAMS and (k in ['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias']):
                params = model_params[layer_count]
                for w in range(params.shape[0]):
                    if w not in mask1:
                        params[w] = 0
            elif k not in OTHER_PARAMS and (k in ['conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias']):
                params = model_params[layer_count]
                if k == 'conv2.weight':
                    for w in range(params.shape[0]):
                        if w not in mask2:
                            params[w] = 0
                        else:
                            for w_ in range(params.shape[1]):
                                if not w_ in mask1:
                                    params[w][w_] = 0
                else:
                    for w in range(params.shape[0]):
                        if w not in mask2:
                            params[w] = 0
            elif k == 'fc1.weight':
                params = model_params[layer_count]
                for w in range(params.shape[0]):
                    if w not in mask3:
                        params[w] = 0
                    else:
                        last_indices = []
                        for q in mask2:
                            for q_ in range(q*8*8, (q+1)*8*8):
                                last_indices.append(q_)
                        for w_ in range(params.shape[1]):
                            if not w_ in last_indices:
                                params[w][w_] = 0
            elif k == 'fc2.weight':
                params = model_params[layer_count]
                for w in range(params.shape[0]):
                    if w not in mask4:
                        params[w] = 0
                    else:
                        for w_ in range(params.shape[1]):
                            if not w_ in mask3:
                                params[w][w_] = 0
            elif k == 'fc.weight':
                params = model_params[layer_count]
                for w in range(CLASSES):
                    for w_ in range(params.shape[1]):
                        if w_ not in mask4:
                            params[w][w_] = 0
            if not k in OTHER_PARAMS:
                layer_count += 1
        return model_params