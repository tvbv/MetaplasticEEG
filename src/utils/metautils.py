import os
import copy
import math
from datetime import datetime
from collections import OrderedDict
import pickle

import numpy as np
import matplotlib.pyplot as plt


import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset


# BNN utils


class Adam_meta(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        meta=0.75,
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            meta=meta,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
        super(Adam_meta, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam_meta, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                    if len(p.size()) != 1:
                        state["followed_weight"] = np.random.randint(
                            p.size(0)
                        ), np.random.randint(p.size(1))
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad.add_(p.data, alpha=group["weight_decay"])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                binary_weight_before_update = torch.sign(p.data)
                condition_consolidation = (
                    torch.mul(binary_weight_before_update, exp_avg) > 0.0
                )

                decay = torch.max(0.3 * torch.abs(p.data), torch.ones_like(p.data))
                decayed_exp_avg = torch.mul(
                    torch.ones_like(p.data)
                    - torch.pow(torch.tanh(group["meta"] * torch.abs(p.data)), 2),
                    exp_avg,
                )
                exp_avg_2 = torch.div(exp_avg, decay)

                if len(p.size()) == 1:  # True if p is bias, false if p is weight
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                else:
                    # p.data.addcdiv_(-step_size, exp_avg , denom)  #normal update
                    p.data.addcdiv_(
                        torch.where(condition_consolidation, decayed_exp_avg, exp_avg),
                        denom,
                        value=-step_size,
                    )  # assymetric lr for metaplasticity

        return loss


class SignActivation(
    torch.autograd.Function
):  # We define a sign activation with derivative equal to clip
    @staticmethod
    def forward(ctx, i):
        result = i.sign()
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (i,) = ctx.saved_tensors
        grad_i = grad_output.clone()
        grad_i[i.abs() > 1.0] = 0
        return grad_i


def Binarize(tensor):
    return tensor.sign()


class BinarizeLinear(torch.nn.Linear):
    def __init__(self, *kargs, binarizing_inputs=True, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)
        self.binarizing_inputs = binarizing_inputs

    def forward(self, input):
        if self.binarizing_inputs:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, "org"):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)
        out = torch.nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out


class BNN(torch.nn.Module):
    """
    MyNet can consist either of fc layers followed by batchnorm, fc weights being either float kind="classical_bn"
    or binarized kind="binary", or fc layers with biases kind="classical_bias". When BatchNorm is used the adtication function is
    the sign function and when biases are used the activation function is Tanh
    weights can be initialized to gaussian with init="gauss" or uniform distribution with init="uniform"
    The width of the distribution is tuned with width
    the only non specified argument is the list of neurons [input, hidden ... , output]
    """

    def __init__(self, layers_dims, init="uniform", width=0.01):
        super(BNN, self).__init__()

        self.hidden_layers = len(layers_dims) - 2
        self.layers_dims = layers_dims

        layer_list = []

        for layer in range(self.hidden_layers + 1):
            if layer == 0:
                binarize_input = False
            else:
                binarize_input = True
            layer_list = layer_list + [
                (
                    ("fc" + str(layer + 1)),
                    BinarizeLinear(
                        layers_dims[layer],
                        layers_dims[layer + 1],
                        binarizing_inputs=binarize_input,
                        bias=False,
                    ),
                )
            ]
            layer_list = layer_list + [
                (
                    ("bn" + str(layer + 1)),
                    torch.nn.BatchNorm1d(
                        layers_dims[layer + 1],
                        affine=False,
                        track_running_stats=True,
                    ),
                )
            ]

            layer_list = layer_list + [
                (
                    ("PReLU" + str(layer + 1)),
                    torch.nn.PReLU(),
                )
            ]

            layer_list = layer_list + [
                (
                    ("do" + str(layer + 1)),
                    torch.nn.Dropout(),
                )
            ]

        self.layers = torch.nn.ModuleDict(OrderedDict(layer_list))

        # weight init
        for layer in range(self.hidden_layers + 1):
            if init == "gauss":
                torch.nn.init.normal_(
                    self.layers["fc" + str(layer + 1)].weight, mean=0, std=width
                )
            if init == "uniform":
                torch.nn.init.uniform_(
                    self.layers["fc" + str(layer + 1)].weight,
                    a=-width / 2,
                    b=width / 2,
                )

    def forward(self, x):
        size = self.layers_dims[0]
        x = x.reshape(-1, size)

        for layer in range(self.hidden_layers + 1):
            x = self.layers["fc" + str(layer + 1)](x)
            x = self.layers["bn" + str(layer + 1)](x)
            x = self.layers["PReLU" + str(layer + 1)](x)
            if layer != self.hidden_layers:
                x = SignActivation.apply(x)
                x = self.layers["do" + str(layer + 1)](x)

        return x


class BNNAxel(torch.nn.Module):
    """
    MyNet can consist either of fc layers followed by batchnorm, fc weights being either float kind="classical_bn"
    or binarized kind="binary", or fc layers with biases kind="classical_bias". When BatchNorm is used the adtication function is
    the sign function and when biases are used the activation function is Tanh
    weights can be initialized to gaussian with init="gauss" or uniform distribution with init="uniform"
    The width of the distribution is tuned with width
    the only non specified argument is the list of neurons [input, hidden ... , output]
    """

    def __init__(self, layers_dims, init="uniform", width=0.01):
        super(BNNAxel, self).__init__()

        self.hidden_layers = len(layers_dims) - 2
        self.layers_dims = layers_dims

        layer_list = []

        for layer in range(self.hidden_layers + 1):
            if layer == 0:
                binarize_input = False
            else:
                binarize_input = True
            layer_list = layer_list + [
                (
                    ("fc" + str(layer + 1)),
                    BinarizeLinear(
                        layers_dims[layer],
                        layers_dims[layer + 1],
                        binarizing_inputs=binarize_input,
                        bias=False,
                    ),
                )
            ]
            layer_list = layer_list + [
                (
                    ("bn" + str(layer + 1)),
                    torch.nn.BatchNorm1d(
                        layers_dims[layer + 1],
                        affine=False,
                        track_running_stats=True,
                    ),
                )
            ]

        self.layers = torch.nn.ModuleDict(OrderedDict(layer_list))

        # weight init
        for layer in range(self.hidden_layers + 1):
            if init == "gauss":
                torch.nn.init.normal_(
                    self.layers["fc" + str(layer + 1)].weight, mean=0, std=width
                )
            if init == "uniform":
                torch.nn.init.uniform_(
                    self.layers["fc" + str(layer + 1)].weight,
                    a=-width / 2,
                    b=width / 2,
                )

    def forward(self, x):
        size = self.layers_dims[0]
        x = x.reshape(-1, size)

        for layer in range(self.hidden_layers + 1):
            x = self.layers["fc" + str(layer + 1)](x)
            x = self.layers["bn" + str(layer + 1)](x)
            if layer != self.hidden_layers:
                x = SignActivation.apply(x)

        return x

    def save_bn_states(self):
        bn_states = []
        if "bn1" in self.layers.keys():
            for l in range(self.hidden_layers + 1):
                bn = copy.deepcopy(self.layers["bn" + str(l + 1)].state_dict())
                bn_states.append(bn)
        return bn_states

    def load_bn_states(self, bn_states):
        if "bn1" in self.layers.keys():
            for l in range(self.hidden_layers + 1):
                self.layers["bn" + str(l + 1)].load_state_dict(bn_states[l])


# CNN utils
class BinarizeConv2d(torch.nn.Conv2d):
    def __init__(self, *kargs, binarizing_inputs=True, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)
        self.binarizing_inputs = binarizing_inputs

    def forward(self, input):
        if self.binarizing_inputs:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, "org"):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)

        out = torch.nn.functional.conv2d(
            input,
            self.weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out


def normal_init(m):
    if m.__class__.__name__.find("Binarize") != -1:
        torch.nn.init.xavier_normal_(m.weight)
    # elif m.__class__.__name__.find('BatchNorm') !=-1:
    #    torch.nn.init.ones_(m.weight)


class ConvBNN(torch.nn.Module):
    def __init__(self, init="gauss", width=0.01, channels=[16, 32, 64, 128, 256]):
        super(ConvBNN, self).__init__()

        # input: (mb x 1 x 600 x 19)
        self.features = torch.nn.Sequential(
            BinarizeConv2d(
                1,
                channels[0],
                binarizing_inputs=False,
                kernel_size=(5, 1),
                padding=(2, 0),
                stride=(1, 1),
                bias=False,
            ),
            BinarizeConv2d(
                channels[0],
                channels[0],
                binarizing_inputs=False,
                kernel_size=(5, 1),
                padding=(0, 0),
                stride=(5, 1),
                bias=False,
            ),  # out: (mb x channels[0] x 600 x 19)
            torch.nn.BatchNorm2d(channels[0], affine=True, track_running_stats=True),
            torch.nn.PReLU(),
            BinarizeConv2d(
                channels[0],
                channels[1],
                binarizing_inputs=True,
                kernel_size=(5, 1),
                padding=(2, 0),
                stride=(1, 1),
                bias=False,
            ),
            BinarizeConv2d(
                channels[1],
                channels[1],
                binarizing_inputs=True,
                kernel_size=(5, 1),
                padding=(0, 0),
                stride=(5, 1),
                bias=False,
            ),  # out: (mb x channels[1] x 600 x 1)
            torch.nn.BatchNorm2d(channels[1], affine=True, track_running_stats=True),
            torch.nn.PReLU(),
            BinarizeConv2d(
                channels[1],
                channels[2],
                binarizing_inputs=True,
                kernel_size=(11, 1),
                padding=(5, 0),
                stride=(1, 1),
                bias=False,
            ),
            BinarizeConv2d(
                channels[2],
                channels[2],
                binarizing_inputs=True,
                kernel_size=(11, 1),
                padding=(0, 0),
                stride=(11, 1),
                bias=False,
            ),
            torch.nn.BatchNorm2d(channels[2], affine=True, track_running_stats=True),
            torch.nn.PReLU(),
            BinarizeConv2d(
                channels[2],
                channels[3],
                binarizing_inputs=True,
                kernel_size=(1, 3),
                padding=(0, 1),
                stride=(1, 1),
                bias=False,
            ),
            BinarizeConv2d(
                channels[3],
                channels[3],
                binarizing_inputs=True,
                kernel_size=(1, 3),
                padding=(0, 0),
                stride=(1, 3),
                bias=False,
            ),
            torch.nn.BatchNorm2d(channels[3], affine=True, track_running_stats=True),
            torch.nn.PReLU(),
            BinarizeConv2d(
                channels[3],
                channels[4],
                binarizing_inputs=True,
                kernel_size=(1, 3),
                padding=(0, 1),
                stride=(1, 1),
                bias=False,
            ),
            BinarizeConv2d(
                channels[4],
                channels[4],
                binarizing_inputs=True,
                kernel_size=(1, 3),
                padding=(0, 0),
                stride=(1, 3),
                bias=False,
            ),
            torch.nn.BatchNorm2d(channels[4], affine=True, track_running_stats=True),
            torch.nn.PReLU(),
        )

        self.classifier = torch.nn.Sequential(
            BinarizeLinear(1024, 256, bias=False, binarizing_inputs=False),
            torch.nn.BatchNorm1d(256, affine=True, track_running_stats=True),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.5),
            BinarizeLinear(256, 64, bias=False),
            torch.nn.BatchNorm1d(64, affine=True, track_running_stats=True),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.5),
            BinarizeLinear(64, 2, bias=False),
        )

        # initialization
        # for layer in self.features:
        #     # if conv2D then init
        #     if isinstance(layer, BinarizeConv2d):
        #         torch.nn.init.uniform_(layer.weight, a=-width / 2, b=width / 2)
        for layer in self.classifier:
            if isinstance(layer, BinarizeLinear):
                torch.nn.init.uniform_(layer.weight, a=-width / 2, b=width / 2)

    def forward(self, x):
        x = self.features(x.unsqueeze(1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


class ConvBNNBogdan(torch.nn.Module):
    def __init__(self, init="gauss", width=0.01, channels=[40, 40]):
        super(ConvBNNBogdan, self).__init__()

        self.features = torch.nn.Sequential(
            BinarizeConv2d(
                1,
                channels[0],
                binarizing_inputs=False,
                kernel_size=(29, 1),
                padding=(14, 0),
                stride=(1, 1),
                bias=False,
            ),  # out: (mb x channels[0] x 600 x 19)
            torch.nn.BatchNorm2d(channels[0], affine=True, track_running_stats=True),
            torch.nn.PReLU(),
            BinarizeConv2d(
                channels[0], channels[1], kernel_size=(1, 19), padding=0, bias=False
            ),  # out: (mb x channels[1] x 600 x 1)
            torch.nn.BatchNorm2d(channels[1], affine=True, track_running_stats=True),
            torch.nn.PReLU(),
            torch.nn.AvgPool2d(
                kernel_size=(30, 1), stride=(15, 1)
            ),  # out: (mb x channels[1] x 40x 1)
        )

        self.classifier = torch.nn.Sequential(
            BinarizeLinear(1560, 80, bias=False),
            torch.nn.BatchNorm1d(80, affine=True, track_running_stats=True),
            torch.nn.PReLU(),
            torch.nn.Dropout(0.5),
            BinarizeLinear(80, 2, bias=False),
        )

        # initialization
        # for layer in self.features:
        #     # if conv2D then init
        #     if isinstance(layer, BinarizeConv2d):
        #         torch.nn.init.uniform_(layer.weight, a=-width / 2, b=width / 2)
        for layer in self.classifier:
            if isinstance(layer, BinarizeLinear):
                torch.nn.init.uniform_(layer.weight, a=-width / 2, b=width / 2)

    def forward(self, x):
        x = self.features(x.unsqueeze(1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def train(
    model,
    train_loader,
    current_task_index,
    optimizer,
    device,
    criterion=torch.nn.CrossEntropyLoss(),
    clamp=10,
    verbose=False,
):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for data, target in train_loader:
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()

        # This loop is for BNN parameters having 'org' attribute
        for p in list(
            model.parameters()
        ):  # blocking weights with org value greater than a threshold by setting grad to 0
            if hasattr(p, "org"):
                p.data.copy_(p.org)

        optimizer.step()

        # This loop is only for BNN parameters as they have 'org' attribute
        for p in list(model.parameters()):  # updating the org attribute
            if hasattr(p, "org"):
                p.org.copy_(p.data)

        running_loss += loss.item()
        num_batches += 1

    epoch_loss = running_loss / num_batches
    return epoch_loss


def train_stream(
    model,
    data,
    target,
    optimizer,
    device,
    criterion=torch.nn.CrossEntropyLoss(),
    verbose=False,
):
    model.train()

    if torch.cuda.is_available():
        data, target = data.to(device), target.to(device)

    optimizer.zero_grad()

    output = model(data)
    loss = criterion(output, target)

    loss.backward()

    # This loop is for BNN parameters having 'org' attribute
    for p in list(
        model.parameters()
    ):  # blocking weights with org value greater than a threshold by setting grad to 0
        if hasattr(p, "org"):
            p.data.copy_(p.org)

    optimizer.step()

    # This loop is only for BNN parameters as they have 'org' attribute
    for p in list(model.parameters()):  # updating the org attribute
        if hasattr(p, "org"):
            p.org.copy_(p.data)

    return loss.item()


def test(
    model,
    data_loader,
    device,
    frac=1,
    criterion=torch.nn.CrossEntropyLoss(),
    verbose=False,
):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in data_loader:
        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[
            1
        ]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(data_loader.dataset)

    test_acc = round(100.0 * float(correct) * frac / len(data_loader.dataset), 2)

    if verbose:
        print(
            "Test accuracy: {}/{} ({:.2f}%)".format(
                correct, len(data_loader.dataset), test_acc
            )
        )

    return test_acc


def plot_parameters(model, path, epoch, save=True):
    hid_lay = model.hidden_layers
    num_hist = hid_lay + 1
    fig = plt.figure(figsize=(15, num_hist * 5))

    for hist_idx in range(num_hist):
        fig.add_subplot(num_hist, 1, hist_idx + 1)
        if not hasattr(model.layers["fc" + str(hist_idx + 1)].weight, "org"):
            weights = model.layers["fc" + str(hist_idx + 1)].weight.data.cpu().numpy()
        else:
            weights = np.copy(
                model.layers["fc" + str(hist_idx + 1)].weight.org.data.cpu().numpy()
            )
        plt.title("Poids de la couche " + str(hist_idx + 1))
        max_abs_weight = np.max(np.abs(weights))
        plt.hist(
            weights.flatten(),
            100,
            density=True,
            range=(-max_abs_weight, max_abs_weight),
        )
        plt.xlim(-max_abs_weight, max_abs_weight)

    if save:
        time = datetime.now().strftime("%H-%M-%S")
        fig.savefig(path + "/" + f"epoch_{epoch}_" + time + "_weight_distribution.png")
        plt.close()
    else:
        plt.show()


def save_histogram_data(model, path, epoch, save=True):
    hid_lay = model.hidden_layers
    num_hist = hid_lay + 1
    histogram_data = {}

    for hist_idx in range(num_hist):
        if not hasattr(model.layers["fc" + str(hist_idx + 1)].weight, "org"):
            weights = model.layers["fc" + str(hist_idx + 1)].weight.data.cpu().numpy()
        else:
            weights = np.copy(
                model.layers["fc" + str(hist_idx + 1)].weight.org.data.cpu().numpy()
            )
        histogram_data[f"layer_{hist_idx + 1}"] = np.histogram(
            weights.flatten(), 100, density=False
        )

    if save:
        with open(path + "/" + f"epoch_{epoch}_" + "histogram_data.pkl", "wb") as file:
            pickle.dump(histogram_data, file)
    else:
        print(histogram_data)


def get_weight_histogram(model):
    hid_lay = model.hidden_layers
    num_hist = hid_lay + 1
    histogram_data = {}

    for hist_idx in range(num_hist):
        if not hasattr(model.layers["fc" + str(hist_idx + 1)].weight, "org"):
            weights = model.layers["fc" + str(hist_idx + 1)].weight.data.cpu().numpy()
        else:
            weights = np.copy(
                model.layers["fc" + str(hist_idx + 1)].weight.org.data.cpu().numpy()
            )
        histogram_data[f"layer_{hist_idx + 1}"] = np.histogram(
            weights.flatten(), bins=100, density=False
        )

    return histogram_data


def plot_parameters_from_data(path, epoch):
    # Load the histogram data
    histogram_data_file = f"epoch_{epoch}_histogram_data.pkl"
    with open(path + "/" + histogram_data_file, "rb") as file:
        histogram_data = pickle.load(file)

    num_hist = len(histogram_data)
    fig = plt.figure(figsize=(15, num_hist * 5))

    for hist_idx, (layer_name, hist_data) in enumerate(histogram_data.items()):
        fig.add_subplot(num_hist, 1, hist_idx + 1)
        plt.title("Poids de la couche " + str(hist_idx + 1))
        hist, bin_edges = hist_data
        max_abs_weight = np.max(np.abs(bin_edges))
        plt.hist(
            bin_edges[:-1],
            bin_edges,
            weights=hist,
            density=True,
            range=(-max_abs_weight, max_abs_weight),
        )
        plt.xlim(-max_abs_weight, max_abs_weight)

    plt.show()


# Dataset Utils


class NpyDataset(Dataset):
    def __init__(self, x_filename, y_filename, stride_frequency=1, fft=True):
        super(NpyDataset, self).__init__()
        self.x_mmap = np.load(x_filename, mmap_mode="r")
        self.y_mmap = np.load(y_filename, mmap_mode="r")
        self.stride_frequency = stride_frequency
        self.fft = fft

    def __getitem__(self, index):
        # Load data and target
        x_data = self.x_mmap[index]
        y_data = self.y_mmap[index]

        # Convert float64 to float32
        x_data = torch.tensor(x_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.long)

        # Preprocess x_data with rfft and take absolute value
        # Here we're applying rfft along the last dimension (the "channel" dimension)
        # print(x_data.shape)
        if self.fft:
            x_data = torch.fft.rfft(x_data, dim=0)[:: self.stride_frequency, :]
            x_data = torch.abs(x_data)
        else:
            x_data = x_data[:: self.stride_frequency, :]

        return x_data, y_data

    def __len__(self):
        return len(self.x_mmap)


class NpyDataset2(Dataset):
    def __init__(
        self, x_filename, y_filename, stride_frequency=1, type: str = "normal"
    ):
        super(NpyDataset2, self).__init__()
        self.x_mmap = np.load(x_filename, mmap_mode="r")
        self.y_mmap = np.load(y_filename, mmap_mode="r")
        self.stride_frequency = stride_frequency
        self.type = type
        if self.type.lower() == "stft":
            self.window = torch.hann_window(300)
        self.random_starting_indice = np.random.randint(0, self.stride_frequency)

    def __getitem__(self, index):
        # Load data and target
        x_data = self.x_mmap[index]
        y_data = self.y_mmap[index]

        # Convert float64 to float32
        x_data = torch.tensor(x_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.long)

        # Preprocess x_data with rfft and take absolute value
        # Here we're applying rfft along the last dimension (the "channel" dimension)
        # print(x_data.shape)
        if self.type.lower() == "fft":
            x_data = torch.fft.rfft(x_data, dim=0)[1 :: self.stride_frequency, :][
                :5700, :
            ]
            x_data = torch.abs(x_data)
        elif self.type.lower() == "stft":
            x_data = torch.stft(
                input=x_data.T,
                n_fft=300,
                hop_length=150,
                win_length=300,
                window=self.window,
                center=True,
                normalized=False,
                onesided=True,
                return_complex=True,
            )

            # Compute the magnitude of the spectrogram
            x_data = torch.abs(x_data)

            # Calculate the frequency bins to exclude (line noise)
            lower_freq = 55  # 60 - 5
            upper_freq = 65  # 60 + 5

            lower_bin = int(lower_freq * 300 / 250)
            upper_bin = int(upper_freq * 300 / 250)

            # exclude the specified frequency bins
            x_data = torch.cat(
                [x_data[:, :lower_bin, :], x_data[:, upper_bin:, :]],
                dim=1,
            )

            x_data = torch.mean(x_data, dim=0)
        elif self.type.lower() == "normal":
            x_data = x_data[:: self.stride_frequency, :]
        else:
            raise ValueError("type must be one of fft, stft, or normal")

        return x_data, y_data

    def __len__(self):
        return len(self.x_mmap)


class NpyDatasetFFTPermutation(Dataset):
    def __init__(
        self, x_filename, y_filename, permuted_indexes=None, stride_frequency=1
    ):
        super(NpyDatasetFFTPermutation, self).__init__()

        self.x_mmap = np.load(x_filename, mmap_mode="r")
        self.y_mmap = np.load(y_filename, mmap_mode="r")

        self.stride_frequency = stride_frequency
        len_fft = torch.fft.rfft(
            torch.tensor(self.x_mmap[0, :, :], dtype=torch.float32), dim=0
        ).shape[
            0
        ]  # x_mmap[0,:,:] has shape : [1501, 19]

        if permuted_indexes is not None:
            self.permuted_indexes = permuted_indexes
        else:
            self.permuted_indexes = np.random.permutation(len_fft)[
                :: self.stride_frequency
            ]

    def __getitem__(self, index):
        # Load data and target
        x_data = self.x_mmap[index]
        y_data = self.y_mmap[index]

        # Convert float64 to float32
        x_data = torch.tensor(x_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.long)

        # Preprocess x_data with rfft and take absolute value
        # Here we're applying rfft along the last dimension (the "channel" dimension)
        # print(x_data.shape)
        x_data = torch.fft.rfft(x_data, dim=0)[self.permuted_indexes, :]
        x_data = torch.abs(x_data)

        return x_data, y_data

    def __len__(self):
        return len(self.x_mmap)
