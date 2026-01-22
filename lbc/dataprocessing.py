
import numpy as np
import torch
import os
import re
import numpy as np



class XXZDataProcessor():
    """
    Processor for 1d, XXZ dataset from Nitya.
    - (300, 1) grid size
    - supports only continuous labels.
    """
    def __init__(self, learning_by_confusion=False, partition_index=None):
        # if discrete_labels:
            # raise NotImplementedError("Classification not yet implemented for XXZ dataset.")

        self.grid_size = (300,1) 
        directory = './data/simulated/XXZ/'
        file_list = [os.path.join(directory, f) for f in os.listdir(directory)]

        # Loads files that start with 't=' and end with '.dat' in the specified directory
        self.samples = []

        for filepath in file_list:
            filename = os.path.basename(filepath)
            if filename.startswith('Sz') and filename.endswith('.txt'):
                match = re.search(r'Jz([-+]?[0-9]*\.?[0-9]+)', filename)
                if match:
                    label = float(match.group(1))
                    data = np.loadtxt(filepath)
                    reshaped = data.reshape(self.grid_size[0], self.grid_size[1])
                    self.samples.append((
                                        torch.tensor(reshaped, dtype=torch.float32).unsqueeze(0),
                                        torch.tensor(label, dtype=torch.float32)
                                    ))

        labels = torch.stack([lbl for _, lbl in self.samples]).cpu().numpy()
        unique_labels = np.unique(labels)
        self.unique_labels = unique_labels

        if learning_by_confusion:
            assert partition_index is not None, "Must provide partition_index when learning_by_confusion=True"
            self.no_partitions = len(unique_labels) - 1  # total partitions between label values
            if partition_index < 0 or partition_index >= self.no_partitions:
                raise ValueError(f"partition_index must be in [0, {self.no_partitions-1}]")

            # threshold-based partition 
            label_threshold = 0.5 * (unique_labels[partition_index] + unique_labels[partition_index + 1])
            low_labels = unique_labels[unique_labels < label_threshold]
            high_labels = unique_labels[unique_labels >= label_threshold]

            new_samples = []
            for data, lbl in self.samples:
                if lbl.item() in low_labels:
                    new_samples.append((data, torch.tensor(0, dtype=torch.float32)))
                elif lbl.item() in high_labels:
                    new_samples.append((data, torch.tensor(1, dtype=torch.float32)))

            self.samples = new_samples
            print(f"\n [LbC] CREATING DATASET: partition {partition_index+1}/{self.no_partitions}, threshold={label_threshold:.4f}, total {len(self.samples)} samples.")
