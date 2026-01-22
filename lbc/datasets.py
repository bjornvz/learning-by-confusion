import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from torch.utils.data import Dataset

from lbc.utils import *
from lbc.dataprocessing import XXZDataProcessor



def get_no_partitions(cf):
    cf.partition_index = 0
    _, temp_dataset = create_datasets(cf)   # deterministic; train-val split uses a separate seed.
    return getattr(temp_dataset.processor, "no_partitions", None)


def create_datasets(cf):
    """
    Returns train and validation datasets based on the configuration.
    Args:
        cf (object): Config object with following attributes: 
            `dataset`: str, name of the dataset to load (e.g., "ILGT", "Paris_Ising", "Paris_XY", "1D_TFIM")

    Returns:
        tuple: (train_dataset, val_dataset)
    """

    if cf.task in ("lbc", "partition"): 
        assert cf.partition_index is not None, "Partition index required for learning by confusion."
        learning_by_confusion = True
        partition_index = cf.partition_index
    else:
        learning_by_confusion = False
        partition_index = None
        

    # the processor loads the data, and returns samples.
    # samples is a list of (data, label) tuples.
    # the train-test split is then done here.
    # from the train and test samples, we create PhaseDataset instances.
    if cf.dataset == "XXZ":
        processor = XXZDataProcessor(learning_by_confusion=learning_by_confusion, partition_index=partition_index)
    else:
        raise ValueError(f"Unknown dataset: {cf.dataset}")
    
    cf.unique_labels = getattr(processor, "unique_labels", None)

    samples = getattr(processor, 'samples')
    g = torch.Generator().manual_seed(42) # use an isolated seed; do not want to average over different train-test splits
    
    n_subset = int(cf.data_fraction * len(samples))
    idx = torch.randperm(len(samples), generator=g)[:n_subset]

    split = int(0.7 * n_subset)
    train_samples = [samples[i] for i in idx[:split]]
    val_samples   = [samples[i] for i in idx[split:]]
        
    train_dataset = PhaseDataset(train_samples, processor=processor)
    val_dataset = PhaseDataset(val_samples, processor=processor)

    return train_dataset, val_dataset


           

class PhaseDataset(Dataset):
    """
    Main dataset class for phase discovery/classification. 
    """
    def __init__(self, samples, processor):
        self.processor = processor

        data, labels = zip(*samples)
        labels = torch.stack(labels).to(torch.float32)
        data = torch.stack(data)

        self.output_dim = 2 
        self.labels = F.one_hot(labels.long(), num_classes=self.output_dim).float()  # One-hot encode
        self.data = data

        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]