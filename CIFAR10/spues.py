from typing import List, Tuple, Union, Dict
from models import CNN
from spues_client import spues_client
import random
import math
import torch
import flwr as fl
from flwr.common import Metrics
from flwr.common import FitIns, FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
from logging import WARNING
from cifardataset import cifar10Dataset
import numpy as np
from util import get_filters, get_parameters, generate_filters_random, set_filters, merge_subnet, spu_aggregation
from torchvision import transforms

CHANNEL = 3
Batch = 128
DEVICE = torch.device("cpu")
CLASSES = 10

class spu_es_strategy(fl.server.strategy.FedAvg):
    def __init__(self, ff, fe, mfc, mec, mac, accuracies=[], ClientsSelection=[], ESCriteria=[], Inf = []):
        super().__init__(fraction_fit=ff, fraction_evaluate=fe, min_fit_clients=mfc, min_evaluate_clients=mec, min_available_clients=mac, evaluate_metrics_aggregation_fn=weighted_average)
        self.fraction_fit_=ff,
        self.fraction_evaluate_=fe,
        self.min_fit_clients_=mfc,
        self.min_evaluate_clients_=mec,
        self.min_available_clients_=mac
        self.global_model = CNN(CHANNEL, outputs=CLASSES)
        self.accuracy_record = accuracies
        self.inference_acc = Inf
        self.selected_clients_records = ClientsSelection
        self.stopped = False
        self.earlystopping_round = 0
        self.earlystopping_acc = 0.0
        self.ES_map = {}
        self.last_losses = {}
        for i in range(mac):
            self.ES_map[str(i)] = False
            self.last_losses[str(i)] = None
        self.personal_models = {}
        self.early_stopping_criteria = ESCriteria
        initial_parameters = get_filters(self.global_model)
        for i in range(mac):
            self.personal_models[i] = initial_parameters

    def record_selected_clients(self, clients:List[str]):
        self.selected_clients_records.append(" ".join(clients))

    def record_criteria_acc_round(self):
        self.early_stopping_criteria.append(" ".join([str(self.earlystopping_round), str(self.earlystopping_acc)]))
    
    def get_ES_map(self):
        return self.ES_map
    
    def update_ES_status(self, cid:str, status:bool):
        self.ES_map[cid] = status

    def get_nones_clients(self):
        clients = []
        for cid, ifes in self.ES_map.items():
            if not ifes:
                clients.append(cid)
        return clients

    def record_test_accuracy(self, acc):
        self.accuracy_record.append(acc) 

    def record_global_inference_accuracy(self, acc):
        self.inference_acc.append(acc)

    """override"""
    def initialize_parameters(self, client_manager: ClientManager):
        return ndarrays_to_parameters(get_parameters(self.global_model))

    def configure_fit(
        self, server_round: int, parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        random.seed(server_round)
        config_fit_list = []
        non_stopped_clients = self.get_nones_clients()
        isstopped = self.stopped
        clients = client_manager.sample(num_clients=sample_size, exploit_factor=isstopped, client_map=non_stopped_clients, min_num_clients=min_num_clients)
        for client in clients:
            cid = int(client.cid)
            config = {}
            base_rate = get_rate(cid)
            total_rate = base_rate
            drop_info, sub_parameters = generate_filters_random(self.global_model, total_rate)
            personal_model = self.personal_models[cid]
            config['personal model'] = personal_model
            config['drop_info'] = drop_info
            config['last loss'] = self.last_losses[str(cid)]
            fit_ins = FitIns(ndarrays_to_parameters(sub_parameters), config)
            config_fit_list.append((client, fit_ins))
        return config_fit_list

    def configure_evaluate(self, server_round: int, parameters, client_manager: ClientManager):
        """override"""
        if self.fraction_evaluate_ == 0.0:
            return []
        # Parameters and config
        config = {'global model':get_filters(self.global_model)}
        # Sample clients
        sample_size, min_num_clients = super().num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config_evaluate_list = []
        for client in clients:
            cid = int(client.cid)
            parameters = self.personal_models[cid]
            fit_ins = FitIns(ndarrays_to_parameters(parameters), config)
            config_evaluate_list.append((client, fit_ins))
        return config_evaluate_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}
        selected_clients = []
        current_parameter = get_filters(self.global_model)
        Fitres = []
        for client, fit_res in results:
            Fitres.append(fit_res)
            cid = client.cid
            self.personal_models[int(cid)] = fit_res.metrics["personal model"]
            self.last_losses[cid] = fit_res.metrics["last loss"]
            selected_clients.append(cid)
            ifes = fit_res.metrics["ifes"]
            self.update_ES_status(cid, ifes)
        self.record_selected_clients(selected_clients)
        if len(self.get_nones_clients()) < self.min_fit_clients:
            self.earlystopping_round = server_round
            self.stopped = True
            self.record_criteria_acc_round()
        parameters_aggregated = spu_aggregation(Fitres, current_parameter)
        #parameters_aggregated = aggregate(Fitres)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        set_filters(self.global_model, parameters_aggregated)
        return parameters_aggregated, metrics_aggregated
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(1, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
            self.record_test_accuracy(metrics_aggregated['accuracy'])
            inference_acc = sum([res.metrics['inference'] for _, res in results]) / self.min_available_clients_
            self.record_global_inference_accuracy(inference_acc)
            print(f"Round {server_round}, early stopping = {self.stopped}, test accuracy = {metrics_aggregated['accuracy']}, inference accuracy = {inference_acc}\n")
            if self.stopped:
                if server_round == self.earlystopping_round:
                    self.earlystopping_acc = metrics_aggregated['accuracy']
                    self.record_criteria_acc_round()
                else:
                    print(f"stopped at {self.earlystopping_round} with an test accuracy of {self.earlystopping_acc}")
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        return loss_aggregated, metrics_aggregated 

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
  # Multiply accuracy of each client by number of examples used
  accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
  examples = [num_examples for num_examples, _ in metrics]
  # Aggregate and return custom metric (weighted average)
  return {"accuracy": sum(accuracies) / sum(examples)}

def spues_client_fn(cid) -> spues_client:
  Epoch = 5
  drop_rate = get_rate(cid)
  dataset = cifar10Dataset("clientdata/cifar10_client_"+ str(cid) + "_ALPHA_0.1.csv")
  return spues_client(cid, dataset, drop_rate, Epoch, Batch)

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
