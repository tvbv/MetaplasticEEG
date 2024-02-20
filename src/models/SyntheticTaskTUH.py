# %% [markdown]
# # Training models on fft TUH Dataset with permuted frequency axis
#
# the data is composed of ffts of the channels on a time frame. To produce different tasks, we permute the frequency axis of the ffts.

# %% [markdown]
# ## Imports

# %%
import sys
import os
import gc
import importlib

sys.path.append(os.path.join("src"))

import wandb

os.environ["WANDB_SILENT"] = "false"

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset

from collections import OrderedDict


from utils.metautils import (
    Adam_meta,
    test,
    train,
    BNNAxel,
    NpyDataset,
    NpyDatasetFFTPermutation,
    get_weight_histogram,
)

from utils.experimentutils import (
    create_saving_architecture,
    get_random_permuted_subset_loader,
    evaluate_model,
)






# HYPERPARAMETERS
PROJECT_NAME = "PermutedTaskTUH"
GROUP_NAME = "5tasks_2x4096"

config = {
    "NUMBER_OF_TASKS": 5,
    "NUMBER_OF_EPOCH_PER_TASK": 20,
    "META": 0.0,
    "META_LIST":[0.0, 15.0, 30.0, 45.0, 60.0],
    "WEIGHT_DECAY": 1e-7,
    "WIDTH_INIT": 1,
    "DATA_DIR": "/.../Data/TUH_preprocessed",
    "NET_SHAPE": (2898, 4 * 1024, 4 * 1024, 2),
    "LR": 0.0001,
    "BATCH_SIZE": 512,
    "STFT_scramble": "random",
}
metrics_to_save = ["Accuracy", "AUC"]
device = torch.device("cuda:" + str(1) if torch.cuda.is_available() else "cpu")





class NpyDatasetSTFT(Dataset):
    def __init__(
        self,
        x_filename,
        y_filename,
        frequency_scrambling=False,
        frequency_index_scrambler=None,
        frequency_scrambling_type="permuted",
    ):
        super(NpyDatasetSTFT, self).__init__()

        self._x = np.load(x_filename, mmap_mode="r")
        self.y = torch.from_numpy(np.load(y_filename)).long()
        self.frequency_index_scrambler = frequency_index_scrambler
        self.frequency_scrambling_type = frequency_scrambling_type

        self.window = torch.hann_window(300)
        lower_freq = 55  # 60 - 5
        upper_freq = 65  # 60 + 5
        self.lower_bin = int(lower_freq * 150 / 125)
        self.upper_bin = int(upper_freq * 150 / 125)

        self.x = [None] * len(self._x)
        for i in range(len(self.x)):
            if i in [int(len(self.x) * j / 100) for j in range(0, 101, 10)]:
                print(int(i / len(self.x) * 100))

            x_temp = torch.tensor(self._x[i]).float().T
            self.x[i] = torch.log(
                torch.abs(
                    torch.stft(
                        input=x_temp,
                        n_fft=300,
                        hop_length=150,
                        win_length=300,
                        window=self.window,
                        center=True,
                        normalized=True,
                        onesided=True,
                        return_complex=True,
                    )
                )
                + 1e-8
            ).half()
            self.x[i] = torch.mean(self.x[i], dim=0)
            self.x[i] = torch.cat(
                (self.x[i][1 : self.lower_bin, :], self.x[i][self.upper_bin :, :]),
                dim=0,
            )  # x[i] is of dimension [ n_freq , n_time]

        del self._x

        self.x = torch.stack(self.x, dim=0).float()

        if frequency_scrambling:
            if self.frequency_index_scrambler is None:
                if self.frequency_scrambling_type == "permuted":
                    self.frequency_index_scrambler = [
                        None,
                        torch.randperm(self.x.shape[1]),
                        torch.randperm(self.x.shape[2]),
                    ]
                elif self.frequency_scrambling_type == "random":
                    print("random")
                    self.frequency_index_scrambler = torch.randperm(
                        self.x.shape[1] * self.x.shape[2]
                    )

            self.update_frequency_scrambler(self.frequency_index_scrambler)
        gc.collect()

    def update_frequency_scrambler(self, frequency_index_scrambler):
        if self.x is None:
            raise ValueError("self.x is None")
        self.frequency_index_scrambler = frequency_index_scrambler
        if self.frequency_scrambling_type == "permuted":
            self.x = self.x[:, :, self.frequency_index_scrambler[2]][
                :, self.frequency_index_scrambler[1], :
            ]
        elif self.frequency_scrambling_type == "random":
            self.x = self.x.flatten(start_dim=1)[
                :, self.frequency_index_scrambler
            ].reshape((self.x.shape[0], self.x.shape[1], self.x.shape[2]))
        gc.collect()

    def __getitem__(self, index):
        # Load data and target
        x_data = self.x[index]
        y_data = self.y[index]

        return x_data, y_data

    def __len__(self):
        return len(self.x)


# %% [markdown]
# ## Task creation
def main():
    task_datasets = {}

    # Datasets definition for each task.
    for task in range(config["NUMBER_OF_TASKS"]):
        if task == 0:
            task_datasets[f"test_{task}"] = NpyDatasetSTFT(
                os.path.join(config["DATA_DIR"], "dev_x_not_mmap.npy"),
                os.path.join(config["DATA_DIR"], "dev_y_not_mmap.npy"),
                frequency_scrambling=False,
            )
            task_datasets[f"train_{task}"] = NpyDatasetSTFT(
                os.path.join(config["DATA_DIR"], "train_x_not_mmap.npy"),
                os.path.join(config["DATA_DIR"], "train_y_not_mmap.npy"),
                frequency_scrambling=False,
            )

        else:
            task_datasets[f"train_{task}"] = NpyDatasetSTFT(
                os.path.join(config["DATA_DIR"], "dev_x_not_mmap.npy"),
                os.path.join(config["DATA_DIR"], "dev_y_not_mmap.npy"),
                frequency_scrambling=True,
                frequency_scrambling_type="random",
            )
            permuted_indexes = task_datasets[f"train_{task}"].frequency_index_scrambler
            task_datasets[f"test_{task}"] = NpyDatasetSTFT(
                os.path.join(config["DATA_DIR"], "train_x_not_mmap.npy"),
                os.path.join(config["DATA_DIR"], "train_y_not_mmap.npy"),
                frequency_scrambling=True,
                frequency_index_scrambler=permuted_indexes,
                frequency_scrambling_type="random",
            )

    for run in range(5):
        # %%
        # New frequency scrambling at every run
        for task in range(config["NUMBER_OF_TASKS"]):
            if task > 0:
                dim_1, dim_2 = task_datasets[f"train_{task}"].x.shape[1:]
                frequency_index_scrambler = torch.randperm(dim_1 * dim_2)
                task_datasets[f"train_{task}"].update_frequency_scrambler(
                    frequency_index_scrambler
                )
                task_datasets[f"test_{task}"].update_frequency_scrambler(
                    frequency_index_scrambler
                )

        task_loader = {}
        for dataset_name, dataset in task_datasets.items():
            task_loader[dataset_name] = DataLoader(
                dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=8
            )

        # %% [markdown]
        # ## Training

        # %%
        def log_model_results_and_weights(
            model,
            task_loader,
            device,
            metrics_to_save,
            results_run,
            task_idx_in_training,
            bn_states,
        ):
            results_task = {}
            weights = get_weight_histogram(model)
            for weight_name, weight_value in weights.items():
                results_task[weight_name] = wandb.Histogram(
                    np_histogram=weight_value, num_bins=500
                )
            for dataset_name, _ in task_loader.items():
                current_task_id = dataset_name.split("_")[1]
                if f"task_{current_task_id}" in bn_states.keys():
                    model.load_bn_states(bn_states[f"task_{current_task_id}"])
                else:
                    model.load_bn_states(bn_states[f"task_{task_idx_in_training}"])
                for metric_name, metric_value in evaluate_model(
                    model,
                    task_loader[dataset_name],
                    device=device,
                    metrics=metrics_to_save,
                ).items():
                    results_run[f"{metric_name}_{dataset_name}"].append(metric_value)
                    results_task[f"{metric_name}_{dataset_name}"] = metric_value
            wandb.log(results_task)
            return results_task
        META_LIST = config["META_LIST"]
        for meta in META_LIST:
            # %%

            config["META"] = meta
            train_loader_list = [
                DataLoader(
                    task_datasets[f"train_{task}"],
                    batch_size=config["BATCH_SIZE"],
                    shuffle=True,
                    num_workers=16,
                )
                for task in range(config["NUMBER_OF_TASKS"])
            ]
            wandb.init(config=config, project=PROJECT_NAME, group=GROUP_NAME)

            LR = config["LR"]
            NUMBER_OF_EPOCH_PER_TASK = config["NUMBER_OF_EPOCH_PER_TASK"]
            WEIGHT_DECAY = config["WEIGHT_DECAY"]
            NET_SHAPE = config["NET_SHAPE"]
            BATCH_SIZE = config["BATCH_SIZE"]
            META = config["META"]
            NUMBER_OF_TASKS = config["NUMBER_OF_TASKS"]
            DATA_DIR = config["DATA_DIR"]

            metrics_to_save = ["Accuracy", "AUC"]

            results_run = {
                f"{metric}_{task}": []
                for metric in metrics_to_save
                for task in task_datasets.keys()
            }

            model = BNNAxel(NET_SHAPE, init="uniform", width=config["WIDTH_INIT"]).to(
                device
            )
            criterion = torch.nn.CrossEntropyLoss()
            wandb.watch(
                model, criterion=criterion, log="all", log_freq=5, log_graph=True
            )

            bn_states = {"task_0": model.save_bn_states()}

            log_model_results_and_weights(
                model, task_loader, device, metrics_to_save, results_run, 0, bn_states
            )

            for task_idx, task in enumerate(train_loader_list):
                optimizer = Adam_meta(
                    model.parameters(),
                    lr=LR,
                    meta=META,
                    weight_decay=WEIGHT_DECAY,
                )

                print(
                    "task : ",
                    task_idx + 1,
                    "/",
                    NUMBER_OF_TASKS,
                    "; lr : ",
                    LR,
                    "; meta : ",
                    META,
                )

                for epoch in range(1, NUMBER_OF_EPOCH_PER_TASK + 1):
                    print(
                        "task: ",
                        task_idx,
                        "; epoch : ",
                        epoch,
                        "/",
                        NUMBER_OF_EPOCH_PER_TASK,
                    )
                    train(model, task, task_idx, optimizer, device, criterion=criterion)

                    bn_states[f"task_{task_idx}"] = model.save_bn_states()

                    log_model_results_and_weights(
                        model,
                        task_loader,
                        device,
                        metrics_to_save,
                        results_run,
                        task_idx,
                        bn_states,
                    )

                    model.load_bn_states(bn_states[f"task_{task_idx}"])

            wandb.finish()


if __name__ == "__main__":
    main()
