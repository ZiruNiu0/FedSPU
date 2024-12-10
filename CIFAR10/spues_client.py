# FLower:
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import Code, EvaluateIns, EvaluateRes, FitRes, Status
# other dependecies:
from models import CNN
import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict
from util import set_filters, get_filters, compute_normalized_norm, compute_update, get_subnet, merge_subnet
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Status
from flwr.server.client_manager import SimpleClientManager
from typing import Dict, Optional, List
from logging import INFO
from flwr.common.logger import log
from flwr.server.criterion import Criterion
from copy import deepcopy
import numpy as np

DEVICE = torch.device("cpu") # Try "cuda" to train on GPU
CLASSES = 10
CHANNELS = 3
Learnable_Params = ['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias', 
                    'conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias',
                    'fc.weight', 'fc.bias', 'fc1.weight', 'fc2.weight']
SPLIT = 0.7

OTHER_PARAMS = ['bn1.num_batches_tracked', 'bn2.num_batches_tracked']

class spues_client(fl.client.Client):
    def __init__(self, cid, dataset, rate, epoch, batch):
        self.cid = cid
        self.model = CNN(in_channels=CHANNELS, outputs=CLASSES).to(DEVICE)
        self.testmodel = CNN(in_channels=CHANNELS, outputs=CLASSES).to(DEVICE)
        self.local_epoch = epoch
        self.local_batch_size= batch
        self.sub_model_rate = rate
        len_train = int(len(dataset) * 0.7)
        len_test = len(dataset) - len_train
        ds_train, ds_val = random_split(dataset, [len_train, len_test], torch.Generator().manual_seed(25))
        self.trainloader = DataLoader(ds_train, self.local_batch_size, shuffle=True)
        self.testloader = DataLoader(ds_val, self.local_batch_size, shuffle=False)
        self.last_validation_loss = None
        self.last_training_loss = None
        self.last_loss = None
    
    def fit(self, ins: FitIns) -> FitRes:
         # Deserialize parameters to NumPy ndarray's
        sub_params = ins.parameters
        drop_info = ins.config['drop_info']
        personal_model = ins.config['personal model']
        last_loss = ins.config["last loss"]
        self.last_loss = last_loss
        set_filters(self.model, personal_model)
        # Update local model, train, get updated parameters
        merged_parameters = merge_subnet(parameters_to_ndarrays(sub_params), personal_model, drop_info)
        set_filters(self.model, merged_parameters)
        # freeze neurons:
        masks = mask_gradients(self.model, drop_info)
        self.freeze_filters(masks)
        self.train()
        ifes = self.check_early_stopping()
        # Serialize ndarray's into a Parameters object
        parameters_updated = get_subnet(self.model, drop_info)
        status = Status(code=Code.OK, message="Success")
        return FitRes(status=status, parameters=ndarrays_to_parameters(parameters_updated), num_examples=len(self.trainloader), metrics={"drop_info":drop_info, "ifes": ifes, "personal model": get_filters(self.model), "last loss": self.last_loss},)
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)
        set_filters(self.testmodel, ndarrays_original)
        loss, accuracy = self.test() 
        global_model = ins.config["global model"]
        set_filters(self.testmodel, global_model)
        _, inference_acc = self.test()
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.testloader),
            metrics={"accuracy": float(accuracy), "inference": float(inference_acc)},
        )
    
    def train(self):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005)
        self.model.train()
        total, training_loss = 0, 0.0
        for e in range(self.local_epoch):
            for samples, labels in self.trainloader:
                samples, labels = samples.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(samples)
                loss = criterion(outputs, labels)
                loss.backward()
                training_loss += loss.item() * labels.size(0)
                total += labels.size(0)
                optimizer.step()
        self.last_training_loss = training_loss / total
        with torch.no_grad():
            total, validation_loss = 0, 0.0
            for samples, labels in self.testloader:
                samples, labels = samples.to(DEVICE), labels.to(DEVICE)
                outputs = self.model(samples)
                loss = criterion(outputs, labels).item() * labels.size(0)
                total += labels.size(0)
                validation_loss += loss
            self.last_validation_loss = validation_loss / total

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

    def freeze_filters(self, masks):
        self.model.conv1.weight.register_hook(lambda grad: grad * masks[0])
        self.model.conv1.bias.register_hook(lambda grad: grad * masks[1])
        self.model.bn1.weight.register_hook(lambda grad: grad * masks[2])
        self.model.bn1.bias.register_hook(lambda grad: grad * masks[3])
        self.model.conv2.weight.register_hook(lambda grad: grad * masks[4])
        self.model.conv2.bias.register_hook(lambda grad: grad * masks[5])
        self.model.bn2.weight.register_hook(lambda grad: grad * masks[6])
        self.model.bn2.bias.register_hook(lambda grad: grad * masks[7])
        self.model.fc1.weight.register_hook(lambda grad: grad * masks[8])
        self.model.fc2.weight.register_hook(lambda grad: grad * masks[9])
        self.model.fc.weight.register_hook(lambda grad: grad * masks[10])
        self.model.fc.bias.register_hook(lambda grad: grad * masks[11])

    def check_early_stopping(self, Lambda=SPLIT) -> bool:
        new_loss = Lambda * self.last_training_loss + (1-Lambda) * self.last_validation_loss
        if self.last_loss == None:
            self.last_loss = new_loss
            return False
        ifes = (self.last_loss - new_loss <= 0.0)
        self.last_loss = new_loss
        return ifes

def mask_gradients(model:CNN, dropout_info:Dict, C=CHANNELS):
    weights = []
    params = model.state_dict()
    for k, v in params.items():
        if k in Learnable_Params:
            weights.append(v)
    if len(dropout_info) == 0:
        return [torch.ones(w.shape) for w in weights]
    last_layer_indices = list(range(C))
    Masks = []
    l = 0
    for k in dropout_info.keys():
        if k == 'fc.bias':
            Masks.append(torch.ones(weights[l].shape))
        elif k in Learnable_Params:
            non_mask_filters = dropout_info[k]
            gradient_mask = torch.ones(weights[l].shape)
            for i in range(gradient_mask.shape[0]):
                if i in non_mask_filters or k == 'fc.weight':
                    if k == 'conv1.weight' or k == 'conv2.weight' or k == 'fc2.weight': 
                        for j in range(gradient_mask.shape[1]):
                            if not (j in last_layer_indices):
                                gradient_mask[i, j] = 0.0
                    elif k == 'fc1.weight':
                        old_indices = []
                        for j_ in last_layer_indices:
                            for q in range(j_*8*8, (j_+1)*8*8):
                                old_indices.append(q)
                        for j in range(gradient_mask.shape[1]):
                            if not (j in last_layer_indices):
                                gradient_mask[i, j] = 0.0
                    elif k == 'fc.weight':
                        for j in range(gradient_mask.shape[1]):
                            if not (j in last_layer_indices):
                                gradient_mask[i, j] = 0.0
                else:
                    gradient_mask[i] = 0.0
            Masks.append(gradient_mask)
            last_layer_indices = non_mask_filters
            l += 1
    return Masks

class spu_es_client_manager(SimpleClientManager):
    def sample(self, num_clients, exploit_factor:Optional[float]=None, client_map:List[str]=None, min_num_clients:Optional[int]=None, criterion:Optional[Criterion]=None):
        # Block until at least num_clients are connected.
        if exploit_factor == None or client_map == None or len(client_map) < num_clients:
            return super().sample(num_clients, min_num_clients, criterion)
        if min_num_clients is None:
            min_num_clients = num_clients
        super().wait_for(min_num_clients)
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        # select clients with the highest utility score (exploit):
        if exploit_factor:
            return super().sample(num_clients, min_num_clients, criterion)
        else:
            import random
            selected_clients = random.sample(client_map, k=num_clients)
            return [self.clients[selected_cid] for selected_cid in selected_clients]