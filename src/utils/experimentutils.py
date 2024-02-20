import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def create_saving_architecture(
    folders_list: list, str_to_show: list, args_to_show: list
) -> dict:
    """
    Creates a directory structure for a given architecture. Each directory is named according to a list of
    strings (`str_to_show`) and corresponding arguments (`args_to_show`). If a directory doesn't exist, it is created.

    Args:
        folders_list (list): A list of base directories where the new directories should be created.
        str_to_show (list): A list of strings used as a part of the name for each new directory.
        args_to_show (list): A list of arguments corresponding to `str_to_show`. Each argument is also used as a part of the name for each new directory.

    Returns:
        dict: A dictionary where keys are the input base directories from `folders_list` and the values are the paths to the newly created directories.

    Raises:
        ValueError: If the lengths of `str_to_show` and `args_to_show` don't match.

    Example:
        >>> BNN_shape_str = "64x64"
        >>> number_of_epoch_per_mini_batch = 10
        >>> NUMBER_OF_RUN = 5
        >>> create_saving_architecture(
        >>>     folders_list=["results", "figures", "models"],
        >>>     str_to_show=["BNN", "epoch_per_mini_batch", "averaged_over"],
        >>>     args_to_show=[BNN_shape_str, number_of_epoch_per_mini_batch, NUMBER_OF_RUN],
        >>> )
        {'results': 'results/BNN_64x64_epoch_per_mini_batch_10_averaged_over_5_',
         'figures': 'figures/BNN_64x64_epoch_per_mini_batch_10_averaged_over_5_',
         'models': 'models/BNN_64x64_epoch_per_mini_batch_10_averaged_over_5_'}
    """
    if len(str_to_show) != len(args_to_show):
        raise ValueError("`str_to_show` and `args_to_show` must have the same length")

    folder_names_to_return = {}
    experience_name = "".join(
        [f"{str_to_show[i]}_{args_to_show[i]}_" for i in range(len(str_to_show))]
    )
    for folder in folders_list:
        new_folder_path = os.path.join(
            folder,
            experience_name,
        )
        folder_names_to_return[folder] = new_folder_path
        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
    return folder_names_to_return


def get_random_permuted_subset_loader(
    dataset: torch.utils.data.Dataset,
    len_divided_by: int,
    batch_size: int = 128,
    num_workers: int = 8,
) -> torch.utils.data.DataLoader:
    """
    Creates a DataLoader from a randomly permuted subset of the input dataset.
    The size of the subset is determined by dividing the size of the original dataset by `len_divided_by`.

    Args:
        dataset (torch.utils.data.Dataset): The input dataset to be sampled.
        len_divided_by (int): The denominator to determine the subset size by `len(dataset) // len_divided_by`.
        batch_size (int, optional): The number of samples per batch to load. Default is 128.
        num_workers (int, optional): How many subprocesses to use for data loading. Default is 8.

    Returns:
        torch.utils.data.DataLoader: A DataLoader instance representing the random subset of the original dataset.

    Example:
        >>> test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
        >>> subset_test_loader = get_random_permuted_subset_loader(
        >>>     dataset=test_dataset, len_divided_by=50, batch_size=128
        >>> )
    """
    indices_test = torch.randperm(len(dataset))[: len(dataset) // len_divided_by]
    subset_test_loader = DataLoader(
        Subset(dataset, indices_test),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    return subset_test_loader


def get_results_plot(
    results: list,
    metrics_to_plot: list,
    name_to_save: str,
    folders_to_save: dict,
    dataset_to_show: list = ["train", "test"],
):
    """
    Creates a plot of specified metrics from the given results. The function calculates the mean and standard deviation
    of the metrics and plots them for both training and testing datasets.

    Args:
        results (list): A list of dictionaries, where each dictionary contains metrics (e.g., "Accuracy", "Recall", "F1", "AUC", "Confusion Matrix")
                        for the datasets specified in `dataset_to_show`.
        metrics_to_plot (list): A list of metrics to be plotted.
        name_to_save (str): The name of the plot to be saved.
        folders_to_save (dict): A dictionary where the key is the type of directory (e.g., "figures", "results") and the value is the directory path.
        dataset_to_show (list, optional): A list of datasets for which to plot metrics. Default is ["train", "test"].

    Example:
        >>> results = [
        >>>     {"train_metrics_list": {"Accuracy": 0.9, "Recall": 0.8, "F1": 0.85, "AUC": 0.88, "Confusion Matrix": np.array([[80, 20], [15, 85]])},
        >>>     "test_metrics_list": {"Accuracy": 0.87, "Recall": 0.76, "F1": 0.82, "AUC": 0.86, "Confusion Matrix": np.array([[78, 22], [18, 82]])}},
        >>>     {"train_metrics_list": {"Accuracy": 0.91, "Recall": 0.81, "F1": 0.86, "AUC": 0.89, "Confusion Matrix": np.array([[81, 19], [14, 86]])},
        >>>     "test_metrics_list": {"Accuracy": 0.88, "Recall": 0.77, "F1": 0.83, "AUC": 0.87, "Confusion Matrix": np.array([[79, 21], [17, 83]])}},
        >>>     {"train_metrics_list": {"Accuracy": 0.92, "Recall": 0.82, "F1": 0.87, "AUC": 0.90, "Confusion Matrix": np.array([[82, 18], [13, 87]])},
        >>>     "test_metrics_list": {"Accuracy": 0.89, "Recall": 0.78, "F1": 0.84, "AUC": 0.88, "Confusion Matrix": np.array([[80, 20], [16, 84]])}}
        >>> ]
        >>> folders = {"figures": "./figures", "results": "./results"}
        >>> get_results_plot(
        >>>     results=results,
        >>>     metrics_to_plot=["Accuracy", "AUC"],
        >>>     name_to_save=f"BNNMetaparam.png",
        >>>     folders_to_save=folders,
        >>>     dataset_to_show=["train", "test"],
        >>> )
    """

    for metric in metrics_to_plot:
        plt.plot(
            np.mean(results_experiment[metric], axis=0),
            label=f"{metric}",
        )
        plt.fill_between(
            range(len(results_experiment[metric][0])),
            np.mean(results_experiment[metric], axis=0)
            - np.std(results_experiment[metric], axis=0),
            np.mean(results_experiment[metric], axis=0)
            + np.std(results_experiment[metric], axis=0),
            alpha=0.2,
        )

    # Create the figure.
    _, ax = plt.subplots(
        len(metrics_to_plot), 1, figsize=(12, 4 * len(metrics_to_plot))
    )

    for dataset in dataset_to_show:
        for metric in metrics_to_plot:
            ax_index = metrics_to_plot.index(metric)
            ax[ax_index].plot(
                curves[f"{dataset}_{metric}_mean"], label=f"{dataset}_{metric}"
            )

            ax[ax_index].fill_between(
                range(number_of_datapoint),
                curves[f"{dataset}_{metric}_mean"] - curves[f"{dataset}_{metric}_std"],
                curves[f"{dataset}_{metric}_mean"] + curves[f"{dataset}_{metric}_std"],
                alpha=0.2,
            )

            ax[ax_index].legend()
            ax[ax_index].set_title(metric)
            ax[ax_index].set_xlabel("Epoch")
            ax[ax_index].set_ylabel(metric)
            ax[ax_index].set_ylim(0, 1)
            ax[ax_index].grid("on", alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(folders_to_save["figures"], name_to_save))


def evaluate_model(
    net,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    metrics: list,
    verbose=False,
):
    """
    Evaluates the model on a given test set and calculates the specified metrics.

    Args:
        net (nn.Module): The model to be evaluated.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        metrics (list): List of the metrics to compute.
            Supported metrics are 'accuracy', 'recall', 'f1', 'auc', and 'confusion_matrix'.
        device (torch.device): The device on which the model and data are placed.
        verbose (bool, optional): If True, prints the values of the metrics during computation. Defaults to False.

    Returns:
        dict: The same dictionary passed in 'metrics' argument, but updated with new values calculated from the test set.
    """
    model = net
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(data)
            predictions = torch.argmax(output, dim=1)
            all_predictions.append(predictions)
            all_labels.append(labels)

    all_predictions = torch.cat(all_predictions).to("cpu").numpy()
    all_labels = torch.cat(all_labels).to("cpu").numpy()

    metrics_to_return = {}

    for metric in metrics:
        if metric.lower() == "accuracy":
            metrics_to_return[metric] = accuracy_score(all_labels, all_predictions)
        elif metric.lower() == "recall":
            metrics_to_return[metric] = recall_score(
                all_labels, all_predictions, average="macro"
            )
        elif metric.lower() == "f1":
            metrics_to_return[metric] = f1_score(
                all_labels, all_predictions, average="macro"
            )
        elif metric.lower() == "auc":
            metrics_to_return[metric] = roc_auc_score(
                all_labels, all_predictions, multi_class="ovr"
            )
        elif metric.lower() == "confusion_matrix":
            metrics_to_return[metric] = confusion_matrix(all_labels, all_predictions)
        else:
            print(f"Metric {metric} is not supported")

    if verbose:
        for keys in metrics_to_return:
            print(f"{keys} : {metrics[keys]:.4f}")

    return metrics_to_return


if __name__ == "__main__":
    X, y = np.random.rand(100, 10), np.random.randint(0, 2, 100)
    dataset_train = torch.utils.data.TensorDataset(
        torch.tensor(X[:80]).float(), torch.tensor(y[:80]).float()
    )
    dataset_test = torch.utils.data.TensorDataset(
        torch.tensor(X[80:]).float(), torch.tensor(y[80:]).float()
    )

    n_runs = 5
    metrics_to_save = ["Accuracy", "AUC", "f1", "Recall"]
    dataset_to_evaluate = {"train": dataset_train, "test": dataset_test}
    results_experiment = {
        f"{metric}_{dataset}": []
        for metric in metrics_to_save
        for dataset in dataset_to_evaluate
    }

    for run in range(n_runs):
        results_run = {
            f"{metric}_{dataset}": []
            for metric in metrics_to_save
            for dataset in dataset_to_evaluate
        }
        model = torch.nn.Linear(10, 2)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for dataset_name, _ in dataset_to_evaluate.items():
            for metric_name, metric_value in evaluate_model(
                model,
                dataset_to_evaluate[dataset_name],
                device=torch.device("cpu"),
                metrics=metrics_to_save,
            ).items():
                results_run[f"{metric_name}_{dataset_name}"].append(metric_value)

        n_epochs = 10
        for epoch in range(n_epochs):
            for data, labels in dataset_train:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, labels.long())
                loss.backward()
                optimizer.step()

            for dataset_name, _ in dataset_to_evaluate.items():
                for metric_name, metric_value in evaluate_model(
                    model,
                    dataset_to_evaluate[dataset_name],
                    device=torch.device("cpu"),
                    metrics=metrics_to_save,
                ).items():
                    results_run[f"{metric_name}_{dataset_name}"].append(metric_value)

        for dataset_name in dataset_to_evaluate:
            for metric_name in metrics_to_save:
                results_experiment[f"{metric_name}_{dataset_name}"].append(
                    results_run[f"{metric_name}_{dataset_name}"]
                )

    metrics_to_plot = ["Accuracy", "AUC"]
    for dataset_name, _ in dataset_to_evaluate.items():
        for metric_name in metrics_to_plot:
            plt.plot(
                np.mean(results_experiment[f"{metric_name}_{dataset_name}"], axis=0),
                label=f"{metric_name}",
            )
            plt.fill_between(
                range(len(results_experiment[f"{metric_name}_{dataset_name}"][0])),
                np.mean(results_experiment[f"{metric_name}_{dataset_name}"], axis=0)
                - np.std(results_experiment[f"{metric_name}_{dataset_name}"], axis=0),
                np.mean(results_experiment[f"{metric_name}_{dataset_name}"], axis=0)
                + np.std(results_experiment[f"{metric_name}_{dataset_name}"], axis=0),
                alpha=0.2,
            )
            plt.show()
