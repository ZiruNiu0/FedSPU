# FLower:
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import Code, EvaluateIns, EvaluateRes, FitRes, Status
# other dependecies:
from models import CNN
import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict
from util import set_filters, get_filters
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, Status
import numpy as np
from typing import List

DEVICE = torch.device("cpu") # Try "cuda" to train on GPU
CLASSES = 62
CHANNELS = 1
Learnable_Params = ['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias', 
                    'conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias',
                    'fc.weight', 'fc.bias']

class fjord_client(fl.client.Client):
    def __init__(self, cid, dataset, rate, epoch, batch):
        self.cid = cid
        self.model = CNN(outputs=CLASSES, rate=rate).to(DEVICE)
        self.testmodel = CNN(outputs=CLASSES, rate=rate).to(DEVICE)
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
        # Update local model, train, get updated parameters
        set_filters(self.model, parameters_to_ndarrays(subnet))
        self.train()
        # Serialize ndarray's into a Parameters object
        parameters_updated = get_filters(self.model)
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