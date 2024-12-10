from datetime import datetime
from feddrop import dropout_strategy, feddrop_client_fn
from fjord import fjord_strategy, fjord_client_fn
from spues import spu_es_strategy, spues_client_fn
from spues_client import spu_es_client_manager
from adadrop import adadrop_strategy, adadrop_client_fn
from fedmp import fedmp_strategy, fedmp_client_fn
from pruneFL import pruneFL_strategy, pruneFL_client_fn
import flwr as fl
import random


NUM_SIMS = 1
ROUNDS = 500
FF = 0.1
FE = 1.0
MFC = 10
MEC = 100
MAC = 100


for i in range(NUM_SIMS):
    randseed = random.randint(0, 99999)
    random.seed(randseed)
    test_acc = []
    selected_clients = []
    strategy = adadrop_strategy(FF, FE, MFC, MEC, MAC, ACC=test_acc, ClientsSelection=selected_clients)
    fl.simulation.start_simulation(
        client_fn=adadrop_client_fn,
        num_clients=MAC,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
    
    now = datetime.now()
    with open('results/adadrop0.75_accuracies_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
            for item in test_acc:
                # write each item on a new line
                fp.write("%f\n" % item)

for i in range(NUM_SIMS):
    randseed = random.randint(0, 99999)
    random.seed(randseed)
    test_acc = []
    selected_clients = []
    consensus = []
    cu = []
    hcp = []
    Inf = []
    earlystopping_records = []
    strategy = spu_es_strategy(FF, FE, MFC, MEC, MAC, accuracies=test_acc, ClientsSelection=selected_clients, ESCriteria=earlystopping_records, Inf=Inf)
    fl.simulation.start_simulation(
        client_fn=spues_client_fn,
        num_clients=MAC,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy,
        client_manager=spu_es_client_manager()
    )
    now = datetime.now()
    with open('results/spuES_accuracies_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in test_acc:
            # write each item on a new line
            fp.write("%f\n" % item)
    with open('results/spuES_clients_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in selected_clients:
            # write each item on a new line
            fp.write("%s\n" % item)
    with open('results/spuES_inference_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in Inf:
            # write each item on a new line
            fp.write("%f\n" % item)
    with open('results/spuES_earlystopping_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in earlystopping_records:
            # write each item on a new line
            fp.write("%s\n" % item)

for i in range(NUM_SIMS):
    randseed = random.randint(0, 99999)
    random.seed(randseed)
    test_acc = []
    selected_clients = []
    strategy = dropout_strategy(FF, FE, MFC, MEC, MAC, ACC=test_acc, ClientsSelection=selected_clients)
    fl.simulation.start_simulation(
        client_fn=feddrop_client_fn,
        num_clients=MAC,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
    now = datetime.now()
    with open('results/feddrop_accuracies_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in test_acc:
            # write each item on a new line
            fp.write("%f\n" % item)

for i in range(NUM_SIMS):
    randseed = random.randint(0, 99999)
    random.seed(randseed)
    test_acc = []
    selected_clients = []
    strategy = fjord_strategy(FF, FE, MFC, MEC, MAC, ACC=test_acc, ClientsSelection=selected_clients)
    fl.simulation.start_simulation(
        client_fn=fjord_client_fn,
        num_clients=MAC,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
    now = datetime.now()
    with open('results/fjord_accuracies_alpha0.1_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
        for item in test_acc:
            # write each item on a new line
            fp.write("%f\n" % item)

for i in range(NUM_SIMS):
    randseed = random.randint(0, 99999)
    random.seed(randseed)
    test_acc = []
    selected_clients = []
    strategy = fedmp_strategy(FF, FE, MFC, MEC, MAC, ACC=test_acc, ClientsSelection=selected_clients)
    fl.simulation.start_simulation(
        client_fn=fedmp_client_fn,
        num_clients=MAC,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
    
    now = datetime.now()
    with open('results/fedmp_accuracies_alpha0.5_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
            for item in test_acc:
                # write each item on a new line
                fp.write("%f\n" % item)

for i in range(NUM_SIMS):
    randseed = random.randint(0, 99999)
    random.seed(randseed)
    test_acc = []
    selected_clients = []
    strategy = pruneFL_strategy(FF, FE, MFC, MEC, MAC, ACC=test_acc, ClientsSelection=selected_clients)
    fl.simulation.start_simulation(
        client_fn=pruneFL_client_fn,
        num_clients=MAC,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
    
    now = datetime.now()
    with open('results/pruneFL_accuracies_alpha0.5_' + now.strftime("%Y%m%d%H%M") + '.txt', 'w') as fp:
            for item in test_acc:
                # write each item on a new line
                fp.write("%f\n" % item)