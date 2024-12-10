from typing import List, Tuple, Union, Dict
from models import CNN
from fedmp_client import fedmp_client
import torch
import flwr as fl
import random
from flwr.common import Metrics
from flwr.common import FitIns, FitRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.common.logger import log
from logging import WARNING
from dataset import EmnistDataset
import numpy as np
from util import get_filters, get_parameters, set_filters, spu_aggregation, parameters_to_ndarrays

CHANNEL = 1
Batch = 16
DEVICE = torch.device("cpu")
CLASSES = 62

class fedmp_strategy(fl.server.strategy.FedAvg):
    def __init__(self, ff, fe, mfc, mec, mac, ACC=[], ClientsSelection=[]):
        super().__init__(fraction_fit=ff, fraction_evaluate=fe, min_fit_clients=mfc, min_evaluate_clients=mec, min_available_clients=mac, evaluate_metrics_aggregation_fn=weighted_average)
        self.fraction_fit_=ff,
        self.fraction_evaluate_=fe,
        self.min_fit_clients_=mfc,
        self.min_evaluate_clients_=mec,
        self.min_available_clients_=mac
        self.global_model = CNN(CHANNEL, outputs=CLASSES)
        self.accuracy_record = ACC
        self.ClientMasks = {}
        self.personal_models = {}
        initial_parameters = get_filters(CNN(CHANNEL, outputs=CLASSES))
        for i in range(mac):
            self.personal_models[i] = initial_parameters

    def record_test_accuracy(self, acc):
        self.accuracy_record.append(acc)

    """override"""
    def initialize_parameters(self, client_manager: ClientManager):
        return ndarrays_to_parameters(get_parameters(self.global_model))

    """override"""
    def configure_fit(self, server_round: int, parameters, client_manager: ClientManager):
        random.seed(random.randint(0, server_round))
        sample_size, min_num_clients = super().num_fit_clients(client_manager.num_available()) 
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config_fit_list = []
        for client in clients:
            cid = int(client.cid)
            config = {}
            drop_info = None
            needMask = True
            if not cid in self.ClientMasks.keys():
               drop_info = {}
               sub_parameters = self.personal_models[cid]
            else:
               drop_info = self.ClientMasks[cid]
               needMask = False
               sub_parameters = self.get_subnet(drop_info)
            config['needmask'] = needMask
            config['drop_info'] = drop_info
            personal_model = self.personal_models[cid]
            config['personal model'] = personal_model
            fit_ins = FitIns(ndarrays_to_parameters(sub_parameters), config)
            config_fit_list.append((client, fit_ins))
        return config_fit_list
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]):
      """override"""
      """Aggregate fit results using weighted average."""
      if not results:
        return None, {}
      # Do not aggregate if there are failures and failures are not accepted
      if not self.accept_failures and failures:
        return None, {}
      # Convert results
      Fit_res = []
      for client, fit_res in results:
        Fit_res.append(fit_res)
        cid = client.cid
        dropinfo = fit_res.metrics["drop_info"]
        self.ClientMasks[int(cid)] = dropinfo
        self.personal_models[int(cid)] = fit_res.metrics["personal model"]
      #for params, size, rate in weights_results:
      aggregated_parameters = spu_aggregation(Fit_res, get_filters(self.global_model))
      # Aggregate custom metrics if aggregation fn was provided
      metrics_aggregated = {}
      if self.fit_metrics_aggregation_fn:
          fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
          metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
      elif server_round == 1:  # Only log this warning once
          log(WARNING, "No fit_metrics_aggregation_fn provided")
      set_filters(self.global_model, aggregated_parameters)
      return ndarrays_to_parameters(aggregated_parameters), metrics_aggregated

    def configure_evaluate(self, server_round: int, parameters, client_manager: ClientManager):
        """override"""
        if self.fraction_evaluate_ == 0.0:
            return []
        # Sample clients
        sample_size, min_num_clients = super().num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        config_evaluate_list = []
        for client in clients:
            config = {}
            parameters = self.personal_models[int(client.cid)]
            #parameters = get_filters(self.global_model)
            fit_ins = FitIns(ndarrays_to_parameters(parameters), config)
            config_evaluate_list.append((client, fit_ins))
        return config_evaluate_list
    
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
            print(f"FedMP: Round {server_round}, test accuracy = {metrics_aggregated['accuracy']}")
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        return loss_aggregated, metrics_aggregated

    def get_subnet(self, drop_info, C=CHANNEL, classes=CLASSES) -> List[np.ndarray]:
        if len(drop_info) == 0:
            return get_filters(self.global_model)
        sub_params = []
        full_params = get_filters(self.global_model)
        last_layer_indices = list(range(C))
        layer_count = 0
        for k in drop_info.keys():
            if k == 'fc.bias' or k == 'fc.weight':
                l = list(range(classes))
            else:
                l = drop_info[k]
            filters = []
            for f in l:
                if k == 'fc.bias':
                    filters.append(full_params[layer_count][f])
                elif k == 'fc.weight':
                    weights = []
                    for number in drop_info[k]:
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

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
  # Multiply accuracy of each client by number of examples used
  accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
  examples = [num_examples for num_examples, _ in metrics]
  # Aggregate and return custom metric (weighted average)
  return {"accuracy": sum(accuracies) / sum(examples)}

def fedmp_client_fn(cid) -> fedmp_client:
  Epoch = 5
  drop_rate = get_rate(cid)
  dataset = EmnistDataset("clientdata/femnist_client_"+str(cid)+"_ALPHA_1.0.csv")
  return fedmp_client(cid, dataset, drop_rate, Epoch, Batch)

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