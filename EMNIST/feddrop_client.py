# FLower:
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import Code, EvaluateIns, EvaluateRes, FitRes, Status
# other dependecies:
from models import CNN
import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict
from util import set_filters, get_filters, merge_subnet
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Status
import numpy as np
from typing import List

DEVICE = torch.device('cpu')
CLASSES = 62
CHANNELS = 1
Learnable_Params = ['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias', 
                    'conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias',
                    'fc.weight', 'fc.bias']
OTHER_PARAMS = ['bn1.num_batches_tracked', 'bn2.num_batches_tracked']

class feddrop_client(fl.client.Client):
    def __init__(self, cid, dataset, rate, epoch, batch):
        self.cid = cid
        self.model = CNN(outputs=CLASSES).to(DEVICE)
        self.testmodel = CNN(outputs=CLASSES).to(DEVICE)
        self.local_epoch = epoch
        self.local_batch_size= batch
        self.sub_model_rate = rate
        len_train = int(len(dataset) * 0.7)
        len_test = len(dataset) - len_train
        ds_train, ds_val = random_split(dataset, [len_train, len_test], torch.Generator().manual_seed(3))
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
        parameters_updated = self.get_updated_parameters(drop_info)
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
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001)
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
    
    def get_updated_parameters(self, drop_info, C=CHANNELS, classes=CLASSES) -> List[np.ndarray]:
        if len(drop_info) == 0:
            return get_filters(self.model)
        sub_params = []
        full_params = get_filters(self.model)
        layer_count = 0
        for k in drop_info.keys():
            filters = []
            if k == 'conv1.weight':
                for f in drop_info[k]:
                    weights = []
                    for weight_count in list(range(C)):
                        weights.append(full_params[layer_count][f][weight_count])
                    filters.append(weights) 
            elif k == 'conv2.weight':
                for f in drop_info[k]:
                    weights = []
                    for weight_count in drop_info['conv1.weight']:
                        weights.append(full_params[layer_count][f][weight_count])
                    filters.append(weights)
            elif k == 'fc.weight':
                for f in range(classes):
                    weights = []
                    for weight_count in drop_info['conv2.weight']:
                        for q in range(7*7*weight_count, 7*7*(weight_count+1)):
                            weights.append(full_params[layer_count][f][q])
                    filters.append(weights) 
            elif k == 'fc.bias':
                for f in range(classes):
                    filters.append(full_params[layer_count][f])
            elif 'bn1' in k or k == 'conv1.bias':
                for f in drop_info['conv1.weight']:
                    filters.append(full_params[layer_count][f])
            elif 'bn2' in k or k == 'conv2.bias':
                for f in drop_info['conv2.weight']:
                    filters.append(full_params[layer_count][f])     
            sub_params.append(np.array(filters))
            #last_layer_indices = drop_info[k]
            layer_count += 1
        return sub_params
    
    def mask_channels(self, drop_info, model_params):
        if len(drop_info) == 0:
            return model_params
        mask1 = sorted(drop_info['conv1.weight'])
        mask2 = sorted(drop_info['conv2.weight'])
        mask3 = sorted(drop_info['fc.weight'])
        param_dict = self.model.state_dict()
        layer_count = 0
        for k in param_dict.keys():
            if k not in OTHER_PARAMS and (k in ['conv1.weight', 'conv1.bias'] or 'bn1' in k):
                params = model_params[layer_count]
                for w in range(params.shape[0]):
                    if w not in mask1:
                        params[w] = 0
                layer_count += 1
            elif k not in OTHER_PARAMS and (k in ['conv2.weight', 'conv2.bias'] or 'bn2' in k):
                params = model_params[layer_count]
                for w in range(params.shape[0]):
                    if w not in mask2:
                        params[w] = 0
                layer_count += 1
            elif k == 'fc.weight':
                params = model_params[layer_count]
                for w in range(CLASSES):
                    for w_ in range(params.shape[1]):
                        if w_ not in mask3:
                            params[w][w_] = 0
        return model_params