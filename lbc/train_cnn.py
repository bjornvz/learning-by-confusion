import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import pprint
from torcheval.metrics.functional import r2_score

from lbc.models import *
from lbc.utils import *
from lbc.datasets import *
from lbc.plots import *

def train(cf, filename):
    cf.logdir = create_path(cf.logdir, overwrite=False)
    print(f"\nTRAINING {cf.logdir}: --seed {cf.seed} --device {DEVICE}\n")
    pprint.pprint(cf)

    set_seeds(cf.seed)
    start_time = datetime.now()

    # LOAD DATASET
    train_loader = DataLoader(cf.train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(cf.val_dataset, batch_size=len(cf.val_dataset), shuffle=False) # batch_size = len(val_dataset) is required to compute goodness_metric correclty

    criterion = nn.CrossEntropyLoss()

    goodness_function = lambda out, y: torch.mean((torch.argmax(out.data, dim=1) == torch.argmax(y, dim=1)).float())

    net1 = Custom1DCNN().to(DEVICE) 

    cf.logdir = str(cf.logdir)
    save_json(cf, cf.logdir, "config.json")

    # OPTIMIZER
    optimizer = optim.Adam( list(net1.parameters()) , 
                           lr=cf.learning_rate , weight_decay=cf.weight_decay)
    
    early_stopper = EarlyStopper(patience=cf.patience, min_delta=0)

    metrics = {f"train_{cf.loss_str}": [], "train_l1": [], f"train_{cf.goodness_str}": [],
               f"val_{cf.loss_str}": [], "val_l1": [], f"val_{cf.goodness_str}": [], 
               "unique_labels" : [], "MVUL_out" : [], "pt":[], "pt2":[], "train_time": [], "net2_norm": []}

    # TRAINING LOOP
    print(f"\nTraining for {cf.epochs} epochs with {len(train_loader)} training batches and {len(val_loader)} validation batches.\n")
    pbar = tqdm(range(cf.epochs))
    for epoch in pbar:
        net1.train()
        # net2.train()

        total_loss = 0
        train_outputs, train_targets = [], [] # for goodness metric calculation

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE).view(-1, cf.train_dataset.output_dim)
            optimizer.zero_grad()

            out = net1(x)

            loss = criterion( out, y )  
            
            total_loss += loss.item() 

            train_outputs.append(out.cpu())
            train_targets.append(y.cpu())

            loss.backward()
            optimizer.step()

        # Concatenate all outputs and targets for the whole epoch
        train_outputs = torch.cat(train_outputs, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        metrics[f"train_{cf.loss_str}"].append( total_loss/len(train_loader) )      # loss is mean by default, so we need to divide by len(train_loader)
        # metrics["train_l1"].append( total_l1/len(train_loader) )  # l1 is not a mean, so we divide by tot. number of samples in the dataset
        metrics[f"train_{cf.goodness_str}"].append( goodness_function( train_outputs, train_targets ).item() )

        # VALIDATION
        net1.eval()
        total_loss = 0
        with torch.no_grad(): # this enables larger batch size for validation; less memory
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(DEVICE), y.to(DEVICE).view(-1, cf.train_dataset.output_dim)

                out = net1(x)
                
                total_loss += criterion( out, y ).item() 

            
            # METRICS FOR EACH EPOCH
            # Loss, l1, goodness metric
            metrics[f"val_{cf.loss_str}"].append( total_loss/len(val_loader) ) 
            # metrics["val_l1"].append( total_l1/len(val_loader) )
            metrics[f"val_{cf.goodness_str}"].append( goodness_function( out, y ).item() )

            # PROGRESS BAR
            pbar.set_description(f"EPOCH {epoch+1}/{cf.epochs}")
            pbar.set_postfix({
                f'TRAIN {cf.loss_str}':     f"{metrics[f'train_{cf.loss_str}'][-1]:.2e}",
                f'TRAIN {cf.goodness_str}': f"{metrics[f'train_{cf.goodness_str}'][-1]:.2f}",
                f'VAL {cf.loss_str}':       f"{metrics[f'val_{cf.loss_str}'][-1]:.2e}",
                f'VAL {cf.goodness_str}':   f"{metrics[f'val_{cf.goodness_str}'][-1]:.2f}"
            }) 
            pbar.update(1)

            if early_stopper.early_stop(metrics[f"val_{cf.loss_str}"][-1]): # NOTE: consider whether one should use the complete loss or just the MSE/CEL
                tqdm.write(f"\nEarly stopping at epoch {epoch+1}")                                 
                break

    # METRICS FOR ONLY LAST EPOCH
    out = out.cpu()
    y = y.cpu()   

    metrics["out0"] = out[:,0].numpy().tolist()
    metrics["labels"] = y.numpy().tolist()
    # metrics["unique_labels"] = cf.val_dataset.unique_labels_unnormalized.tolist()
    metrics["train_time"].append(str(datetime.now() - start_time))

    print(f"\nTraining time: {metrics['train_time']}")

    save_json(metrics, cf.logdir, filename)