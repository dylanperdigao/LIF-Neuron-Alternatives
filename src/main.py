import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import optuna
import time
import sys
import argparse
import random
PATH = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(PATH, os.path.pardir)))

from torch import manual_seed, tensor
from torch.cuda.random import manual_seed_all
from torch import backends
from modules.utils import load_dataset
from modules.model import ModelSpikingNeuron
from pyJoules.device import DeviceFactory
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.energy_meter import EnergyMeter

####################### HYPERPARAMETERS
BATCH = 1024
TRIALS = 100
EPOCHS = 10
MEASURE_ENERGY = True
NEURONS = ['leaky', 'rleaky', 'synaptic', 'lapicque', 'alpha', 'rsynaptic']

####################### REPRODUCIBILITY
BASE_SEED = 42
np.random.seed(BASE_SEED)
SEEDS = np.random.randint(0, 1_000_000, size=TRIALS)
print("SEEDS:", SEEDS)


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    manual_seed_all(seed)
    backends.cudnn.deterministic=True

def objective(trial: optuna.Trial, dataset: str, neuron: str, steps: int, seed: int, gpu: int|str):
    beta = trial.suggest_float('beta', 0.0, 1.0, step=1e-6)
    threshold = trial.suggest_float('threshold', 0.0, 1.0, step=1e-6)
    recurrent = trial.suggest_float('recurrent', 0.0, 1.0, step=1e-6) if neuron in ['rleaky', 'rsynaptic'] else None
    alpha = trial.suggest_float('alpha', beta, 1.0, step=1e-6) if neuron in ['alpha', 'synaptic', 'rsynaptic'] else None
    trial.suggest_int("seed", int(seed), int(seed))
    trial.suggest_int("batch", BATCH, BATCH)
    trial.suggest_int("epochs", EPOCHS, EPOCHS)
    trial.suggest_int("steps", steps, steps)
    train_loader, test_loader = load_dataset(dataset, BATCH, seed=int(seed))
    classes = len(train_loader.dataset.classes)
    model = ModelSpikingNeuron(
        neuron=neuron,
        neuron_alpha=alpha,
        neuron_beta=beta,
        neuron_threshold=threshold,
        neuron_recurrent=recurrent,
        classes=classes,
        dataset=dataset,
        num_steps=steps, 
        epochs=EPOCHS,
        class_weights= tensor([0.002, 0.998]) if 'BAF' in dataset else None,
        gpu_number=gpu
    )
    try:
        if MEASURE_ENERGY and gpu != "cpu":
            domains = [NvidiaGPUDomain(0)]
            devices = DeviceFactory.create_devices(domains)
            meter = EnergyMeter(devices)
            meter.start()
            model.fit(train_loader)
            meter.stop()
            trace_train = meter.get_trace()
            meter.start()
            preds, targets = model.predict(test_loader)
            meter.stop()
            trace_test = meter.get_trace()
            energy_metrics = {
                "time_train": trace_train[0].duration,
                "time_test": trace_test[0].duration,
                "energy_train": sum(trace_train[0].energy.values()) / 1000,
                "energy_test": sum(trace_test[0].energy.values()) / 1000
            }
        else:
            t = time.time()
            model.fit(train_loader)
            time_train = time.time() - t
            t = time.time()
            preds, targets = model.predict(test_loader)
            time_test = time.time() - t
            energy_metrics = {
                "time_train": time_train,
                "time_test": time_test,
                "energy_train": 0,
                "energy_test": 0
            }
        results = model.evaluate(targets, preds)
        print(f"Trial {trial.number} {neuron} {results['accuracy']} {results['precision']} {results['recall']} {results['fpr']} {results['f1_score']} {results['auc']}")
        trial.set_user_attr("time_train", energy_metrics["time_train"])
        trial.set_user_attr("time_test", energy_metrics["time_test"])
        trial.set_user_attr("energy_train", energy_metrics["energy_train"])
        trial.set_user_attr("energy_test", energy_metrics["energy_test"])
        trial.set_user_attr("power_train", energy_metrics["energy_train"] / energy_metrics["time_train"])
        trial.set_user_attr("power_test", energy_metrics["energy_test"] / energy_metrics["time_test"])
        trial.set_user_attr("metric_accuracy", results["accuracy"])
        trial.set_user_attr("metric_precision", results["precision"])
        trial.set_user_attr("metric_recall", results["recall"])
        trial.set_user_attr("metric_fpr", results["fpr"])
        trial.set_user_attr("metric_f1", results["f1_score"])
        trial.set_user_attr("metric_auc", results["auc"])
        return results['accuracy']
    except RuntimeError as e:
        print(e)
        return float('nan')

def main(dataset, steps, neuron_name=None, neuron_start=0, run_start=0, gpu=None):
    neurons = [neuron_name] if neuron_name is not None else NEURONS
    for neuron in neurons[neuron_start:]:
        for i in range(run_start, TRIALS):
            seed = SEEDS[i]
            fix_seed(int(seed))
            sampler = optuna.samplers.TPESampler(seed=seed)
            machine_name = os.uname().nodename
            study = optuna.create_study(
                study_name=f"{neuron}_{steps}", 
                storage=f"sqlite:///{PATH}/results/{'test-'}{dataset.lower()}-{machine_name}.db", 
                direction="maximize",
                sampler=sampler,
                load_if_exists=True
            )
            study.optimize(lambda trial: objective(trial, dataset, neuron, steps, seed, gpu), n_trials=1)
            print(f"Trial {i} {neuron} {study.best_trial.number} {study.best_trial.values}")
      

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--neuron', type=str, required=True)
    parser.add_argument('--gpu', type=int, required=False)
    args = parser.parse_args()
    dataset = args.dataset
    steps = args.steps
    neuron_name = args.neuron
    gpu = args.gpu if args.gpu is not None else 'cpu'
    main(dataset, steps, neuron_name, gpu=gpu)