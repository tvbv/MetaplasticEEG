"""
this script is used to stream the TUH dataset and train a metaplastic 
binarized network on this stream of data.

"""

import sys
import os
import gc
import random

sys.path.append(os.path.join("src"))

import wandb


import numpy as np

import torch
from torch.utils.data.dataset import Subset, Dataset


from utils.metautils import (
    Adam_meta,
    train,
    BNNAxel,
    get_weight_histogram,
)
from utils.experimentutils import (
    evaluate_model,
)


# hyperparameters

PROJECT_NAME = "StreamTUH"
GROUP_NAME = "2Lx4096_ref"

NUMBER_OF_RUNS = 2
META_LIST = [0.0, 30.0, 60.0, 90.0]

config = {
    "META": 0.0,
    "WEIGHT_DECAY": 0,
    "WIDTH_INIT": 1,
    "DATA_DIR": "/.../Data/TUH_preprocessed",
    "NET_SHAPE": (2898, 4 * 1024, 4 * 1024, 2),
    "LR": 0.0001,
    "BATCH_SIZE_SUBSET": 128,
    "NUMBER_SUBSET": 1,
    "NUMBER_OF_EPOCH_PER_SUBSET": 20,
}

device = torch.device("cuda:" + str(1) if torch.cuda.is_available() else "cpu")


# Helper function for logging the model weights and the metrics.


def log_model_results_and_weights(
    model,
    task_loader,
    device,
    metrics_to_save,
    results_run,
    task_idx_in_training,
    bn_states,
):
    """
    Logs the model results and weights.

    This function logs the model results and weights for a given task and dataloader.
    It logs the weights as histograms and the results as defined by the `metrics_to_save` parameter.

    Args:
        model (nn.Module): The model to evaluate.
        task_loader (dict): A dictionary containing the dataloader for each task.
        device (torch.device): The device to use for computation.
        metrics_to_save (list): A list of strings representing the names of the metrics to save.
        results_run (dict): A dictionary to store the results of the model evaluation.
        task_idx_in_training (int): The index of the current task in the training process.
        bn_states (dict): A dictionary containing the batch normalization states for each task.

    Returns:
        dict: A dictionary containing the results of the model evaluation and the weights as histograms.

    """

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


# Definition of the dataset class for preproccesed TUH data.


class NpyDatasetSTFT(Dataset):
    def __init__(
        self,
        x_filename,
        y_filename,
        frequency_scrambling=False,
        frequency_index_scrambler=None,
        frequency_scrambling_type="permuted",
        device=torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu"),
    ):
        super(NpyDatasetSTFT, self).__init__()

        self.device = device
        self._x = np.load(x_filename, mmap_mode="r")
        self.y = torch.from_numpy(np.load(y_filename)).long()
        self.y = self.y.to(self.device)
        self.frequency_index_scrambler = frequency_index_scrambler
        self.frequency_scrambling_type = frequency_scrambling_type

        self.window = torch.hann_window(300)
        lower_freq = 55  # 60 - 5
        upper_freq = 65  # 60 + 5
        self.lower_bin = int(lower_freq * 300 / 250)
        self.upper_bin = int(upper_freq * 300 / 250)

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

        gc.collect()

        self.x = self.x.to(self.device)

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

    def to(self, device):
        self.device = device
        self.x = self.x.to(device)
        self.y = self.y.to(device)
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


# Learning.


def main():
    LR = config["LR"]
    NUMBER_OF_EPOCH_PER_SUBSET = config["NUMBER_OF_EPOCH_PER_SUBSET"]
    WEIGHT_DECAY = config["WEIGHT_DECAY"]
    NET_SHAPE = config["NET_SHAPE"]
    BATCH_SIZE_SUBSET = config["BATCH_SIZE_SUBSET"]
    META = config["META"]
    NUMBER_SUBSET = config["NUMBER_SUBSET"]
    DATA_DIR = config["DATA_DIR"]

    # Datasets definition.
    #   creates dataset object and dataloader
    #   dataloaders test and train will be used to assess the metrics, but won't be used for training

    train_dataset = NpyDatasetSTFT(
        os.path.join(config["DATA_DIR"], "train_x_not_mmap.npy"),
        os.path.join(config["DATA_DIR"], "train_y_not_mmap.npy"),
        device=device,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=True  # , num_workers=16
    )

    test_dataset = NpyDatasetSTFT(
        os.path.join(config["DATA_DIR"], "dev_x_not_mmap.npy"),
        os.path.join(config["DATA_DIR"], "dev_y_not_mmap.npy"),
        device=device,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=True  # , num_workers=16
    )

    dataset_train_length = len(train_dataset)

    print("len train dataset: ", dataset_train_length)
    print("len test dataset: ", len(test_dataset))

    # Training

    train_loader_list = []

    for i in range(NUMBER_SUBSET):
        train_loader_list.append(
            torch.utils.data.DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE_SUBSET,
                sampler=torch.utils.data.SubsetRandomSampler(
                    range(
                        int(i * (dataset_train_length / NUMBER_SUBSET)),
                        int((i + 1) * (dataset_train_length / NUMBER_SUBSET)),
                    )
                ),
                shuffle=False,
                # num_workers=16,
            )
        )

    random.shuffle(train_loader_list)

    metrics_to_save = ["Accuracy", "AUC", "recall", "f1"]
    dataset_to_evaluate = {
        "test": test_loader,
        "train": train_loader,
    }
    results_run = {
        f"{metric}_{dataset}": []
        for metric in metrics_to_save
        for dataset in dataset_to_evaluate
    }
    for META in META_LIST:
        config["META"] = META
        wandb.init(config=config, project=PROJECT_NAME, group=GROUP_NAME)
        model = BNNAxel(NET_SHAPE, init="uniform", width=config["WIDTH_INIT"]).to(
            device
        )
        criterion = torch.nn.CrossEntropyLoss()
        wandb.watch(model, criterion=criterion, log="all", log_freq=5, log_graph=True)

        optimizer = Adam_meta(
            model.parameters(),
            lr=LR,
            meta=META,
            weight_decay=WEIGHT_DECAY,
        )
        results_task = {}
        weights = get_weight_histogram(model)
        for weight_name, weight_value in weights.items():
            results_task[weight_name] = wandb.Histogram(
                np_histogram=weight_value, num_bins=100
            )
        for dataset_name, _ in dataset_to_evaluate.items():
            for metric_name, metric_value in evaluate_model(
                model,
                dataset_to_evaluate[dataset_name],
                device=device,
                metrics=metrics_to_save,
            ).items():
                results_run[f"{metric_name}_{dataset_name}"].append(metric_value)
                results_task[f"{metric_name}_{dataset_name}"] = metric_value
        wandb.log(results_task)
        for task_idx, task in enumerate(train_loader_list):
            print(
                "task : ",
                task_idx + 1,
                "/",
                NUMBER_SUBSET,
                "; lr : ",
                LR,
                "; meta : ",
                META,
            )
            for epoch in range(1, NUMBER_OF_EPOCH_PER_SUBSET + 1):
                train(model, task, task_idx, optimizer, device, criterion=criterion)
            if task_idx % 5 == 0:
                results_task = {}
                weights = get_weight_histogram(model)
                for weight_name, weight_value in weights.items():
                    results_task[weight_name] = wandb.Histogram(
                        np_histogram=weight_value, num_bins=100
                    )
                for dataset_name, _ in dataset_to_evaluate.items():
                    for metric_name, metric_value in evaluate_model(
                        model,
                        dataset_to_evaluate[dataset_name],
                        device=device,
                        metrics=metrics_to_save,
                    ).items():
                        results_run[f"{metric_name}_{dataset_name}"].append(
                            metric_value
                        )
                        results_task[f"{metric_name}_{dataset_name}"] = metric_value
                wandb.log(results_task)
        wandb.finish()


if __name__ == "__main__":
    for run in range(NUMBER_OF_RUNS):
        main()
