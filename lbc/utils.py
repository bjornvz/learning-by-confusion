import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import itertools
import inspect
import random
from tqdm import tqdm
from torcheval.metrics.functional import r2_score
from sklearn.linear_model import LinearRegression
import os

from sklearn.metrics import r2_score as sk_r2_score
import pprint
from pathlib import Path

sns.set_theme()
sns.color_palette("Paired")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class AttrDict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__




def set_plotting_logging_strings(cf):
    """
    Sets strings for plotting and logging based on task and goodness function.
    """  
    if cf.task != "regression":
        cf.goodness_str = "acc"
    cf.phase_indicator_str = r"$\bar{y}$" if cf.task == "classification" else r"${\partial\hat{\gamma}}/{\partial\gamma}$"
    cf.loss_str = "CEL" if cf.task != "regression" else "MSE"



def create_path(folder_path, overwrite=False):
    """
    Creates a path to save a file in a folder.
    If the *parent experiment folder* exists, adds a suffix to it.
    """
    base_folder = Path(folder_path)
    experiment_folder = base_folder.parent / base_folder.name  # initial path

    if not overwrite:
        counter = 1
        while experiment_folder.exists():
            print(f"Folder {experiment_folder} exists. Adding suffix.")
            experiment_folder = base_folder.parent.with_name(
                f"{base_folder.parent.name}_{counter}"
            ) / base_folder.name
            counter += 1

    experiment_folder.mkdir(parents=True, exist_ok=True)
    return experiment_folder


def func_name():
    return inspect.stack()[1].function


def save_json(obj, folder_path, file_name):
    """
    Saves obj to path as a json file.
    """
    obj_to_save = {
        k: (str(v) if not isinstance(v, (str, int, float, bool, type(None), list, dict)) else v)
        for k, v in obj.items()
    }
    folder = Path(folder_path)
    load_path = folder.joinpath(file_name)
    with open(load_path, "w") as file:
        json.dump(obj_to_save, file)

def load_json(folder_path, file_name):
    """
    Loads a json file from path.
    """
    folder = Path(folder_path)
    load_path = folder.joinpath(file_name)
    with open(load_path) as file:
        return json.load(file)


def set_seeds(seed_no):
    """"
    Sets the seed for reproducibility.
    NB: seed is accessible globally.
    """
    global seed
    seed = seed_no
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class EarlyStopper:
    """
    Standard early stopping class.
    """
    def __init__(self, patience=1, min_delta=0, verbose=True, relative=False, pbar=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_obj = float('inf')
        self.verbose = verbose
        self.relative = relative

    def early_stop(self, validation_obj):
        if np.isnan(validation_obj):
            print("Validation objective is NaN. Stopping early.")
            return True
        difference = validation_obj - self.min_validation_obj
        if self.relative:
            difference /= self.min_validation_obj
        if validation_obj < self.min_validation_obj:
            # if self.verbose:
                # tqdm.write(f"Validation objective decreased ({self.min_validation_obj:.2e} --> {validation_obj:.2e}).")
            self.min_validation_obj = validation_obj
            self.counter = 0
        elif difference >= self.min_delta:
            self.counter += 1
            if self.verbose:
                tqdm.write(f"Validation objective increased ({self.min_validation_obj:.2e} --> {validation_obj:.2e}). Counter: {self.counter} out of {self.patience}.")
            if self.counter >= self.patience:
                return True
        return False
 

def l1_regularization(z, penalties):
    """"
    Returns $\sum_{k=1}^K |z_k| \lambda_k$, where $z_k$ is the $k-$th bottleneck activation,
    and $\lambda_k$ is the corresponding penalty for this branch. l1 regularization encourages
    sparsity in the bottleneck, and penalties encourage simplicity.
    """
    assert z.shape[1] == len(penalties), "Number of penalties must match the number of branches."
    branches = torch.abs(z) 
    branches *= penalties # TODO double check
    return torch.sum( branches, dim=1 ) # sum over the branches


def normalize01(y, y_min, y_max):
    """
    Normalize to [0, 1] range.
    """
    return (y - y_min) / (y_max - y_min)

def denormalize01(y, y_min, y_max):
    """
    Denormalize from [0, 1] range to original range.
    """
    return y * (y_max - y_min) + y_min


def r2_agg(y_pred, y_true):
    """
    y_true: tensor [N, 1]
    y_pred: tensor [N, 1]
    """
    # TODO: gpt-generated; need to check
    # Flatten to [N]
    y_true, y_pred = y_true.view(-1), y_pred.view(-1)

    unique_targets = torch.unique(y_true)
    avg_preds = torch.stack([
        y_pred[y_true == t].mean() for t in unique_targets
    ])

    ss_res = torch.sum((unique_targets - avg_preds) ** 2)
    ss_tot = torch.sum((unique_targets - unique_targets.mean()) ** 2)
    return 1 - ss_res / ss_tot



def update_metrics_per_seed(cf, metrics, metrics_per_seed):
    """
    Updates the metrics_per_seed dictionary with the latest metrics.
    """
    if cf.model != "tetriscnn": return

    index = cf.patience if cf.epochs > cf.patience else 1
    
    if cf.task == "regression":                  
        MVUL_out = np.array(metrics["MVUL_out"])[-index]
        metrics_per_seed["MVUL_out"] = MVUL_out.tolist()

        if cf.label_param == "deltaomega":
            metrics_per_seed["pt"].append( metrics["pt"][-index] )
            metrics_per_seed["pt2"].append( metrics["pt2"][-index] )
        else:
            metrics_per_seed["pt"].append( metrics["pt"][-index] )


    if cf.save_histories:
        assert len(np.array(metrics[f'z_0'])) > 1 , "Saved data does not have right dimension; history not saved?"
        z = np.array([ metrics[f'z_{i}'][-index] for i in range(len(cf.kernels)) ]) 
    else:
        assert len(np.array(metrics[f'z_0'])) == 1, "Saved data does not have right dimension; history was saved."
        z = np.array([ metrics[f'z_{i}'] for i in range(len(cf.kernels)) ]) # if we did not save histories, there is only one value
    assert z.ndim == 1 and z.shape[0] == len(cf.kernels), f"z has wrong shape: {z.shape}, expected ({len(cf.kernels)}, )"
    
    metrics_per_seed["z"].append(z.tolist()) # one value per kernel

    metrics_per_seed["net2_norm"].append(metrics["net2_norm"])
    metrics_per_seed[f"val_{cf.loss_str}"].append(metrics[f"val_{cf.loss_str}"][-index])
    metrics_per_seed[f"val_{cf.goodness_str}"].append(metrics[f"val_{cf.goodness_str}"][-index])
    metrics_per_seed["epochs"].append(len(metrics[f"val_{cf.loss_str}"]))


def update_metrics_per_partition(cf, metrics, metrics_per_partition):
    index = cf.patience if cf.epochs > cf.patience else 1
    if cf.model == "tetriscnn":
        if cf.save_histories:
            assert len(np.array(metrics[f'z_0'])) > 1 , "Saved data does not have right dimension; history not saved?"
            z = np.array([ metrics[f'z_{i}'][-index] for i in range(len(cf.kernels)) ]) 
        else:
            assert len(np.array(metrics[f'z_0'])) == 1, "Saved data does not have right dimension; history was saved."
            z = np.array([ metrics[f'z_{i}'] for i in range(len(cf.kernels)) ]) # if we did not save histories, there is only one value
        assert z.ndim == 1 and z.shape[0] == len(cf.kernels), f"z has wrong shape: {z.shape}, expected ({len(cf.kernels)}, )"
        metrics_per_partition["z"].append(z.tolist()) # one value per kernel
    metrics_per_partition[f"val_{cf.loss_str}"].append(metrics[f"val_{cf.loss_str}"][-index]) 
    metrics_per_partition[f"train_{cf.loss_str}"].append(metrics[f"train_{cf.loss_str}"][-index]) 
    metrics_per_partition[f"val_{cf.goodness_str}"].append(metrics[f"val_{cf.goodness_str}"][-index])
    metrics_per_partition[f"train_{cf.goodness_str}"].append(metrics[f"train_{cf.goodness_str}"][-index])

