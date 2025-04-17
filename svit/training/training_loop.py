import gc
import torch
import torch.nn as nn
import torch.optim as optim

from svit import config, device
from svit.training.energy import energy_function

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.notebook import tqdm

def get_criterion() -> nn.Module:
    """
    Provides a criterion for neural network training.

    This function returns an instance of a loss function suitable for use in
    training a neural network. The returned criterion calculates the
    cross-entropy loss, typically used for classification tasks.

    :return: A PyTorch `nn.Module` instance representing the loss function.
    :rtype: nn.Module
    """
    return nn.CrossEntropyLoss()

def get_optimizer(model: nn.Module,
                  lr: float=config["training"]["learning_rate"],
                  weight_decay: float=config["training"]["weight_decay"]) -> optim.Optimizer:
    """
    Get an optimizer for the given model using Adam optimization algorithm.

    This function initializes and returns an Adam optimizer with the specified
    learning rate and weight decay, which are typically used for training neural
    network models. The optimizer is configured to handle the model's parameters.

    :param model: The neural network model whose parameters the optimizer will update.
    :type model: nn.Module
    :param lr: The learning rate to use for the optimizer. If not provided,
        it defaults to a value specified in the configuration.
    :type lr: float
    :param weight_decay: The weight decay (L2 regularization) to apply during optimization.
        If not provided, it defaults to a value specified in the configuration.
    :type weight_decay: float
    :return: An Adam optimizer instance configured for the given model and parameters.
    :rtype: optim.Optimizer
    """
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer: optim.Optimizer,
                  t_max: int=config["training"]["T_max"]) -> CosineAnnealingLR:
    """
    Creates and returns a cosine annealing learning rate scheduler.

    This function generates a learning rate scheduler using PyTorch's
    CosineAnnealingLR. The scheduler varies the learning rate according
    to a cosine function, decreasing it over the number of epochs
    or iterations specified by the T_max parameter.

    :param optimizer: Optimizer whose learning rate is being scheduled.
    :param t_max: Maximum number of iterations or epochs for the cosine
        annealing schedule.
    :return: A CosineAnnealingLR instance configured to modify the learning
        rate of the provided optimizer according to the cosine schedule.
    """
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

def train(model:nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          epoch: int=config["training"]["epoch"]) -> None:
    """
    Trains a neural network model using the provided data loaders for a specified
    number of epochs. The function utilizes a criterion for loss calculation, an
    optimizer for parameter updates, and a scheduler for learning rate adjustments.
    It also tracks various metrics, including energy, cost, L1 norm, and accuracy,
    and logs the progress for each epoch.

    :param model: The neural network model to be trained.
    :type model: nn.Module
    :param train_dataloader: DataLoader for training data, providing batches of input
       features and corresponding labels.
    :type train_dataloader: DataLoader
    :param val_dataloader: DataLoader for validation data, which is used during the
       calculation of energy and other metrics.
    :type val_dataloader: DataLoader
    :param epoch: The number of epochs for which the model should be trained. Defaults
       to the value fetched from the configuration dictionary.
    :type epoch: int
    :return: Does not return a value. All computations and updates are performed in place.
    :rtype: None
    """
    criterion = get_criterion()
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)

    model.to(device)
    criterion.to(device)

    pbar = tqdm(range(epoch))

    F_list = []
    cost_list = []
    l1_list = []
    accuracy_list = []
    for epoch in pbar:
        F_list_e = []
        cost_list_e = []
        l1_list_e = []
        accuracy_list_e = []

        inner_pbar = tqdm(train_dataloader, leave=False)
        for data in inner_pbar:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            x, y = data
            x, y = x.to(device), y.to(device)

            output = model(x)
            F, cost, l1, a = energy_function(model, output, y, val_dataloader, criterion)

            F_list_e.append(F)
            cost_list_e.append(cost)
            l1_list_e.append(l1)
            accuracy_list_e.append(a)

            optimizer.zero_grad()
            F.backward()
            optimizer.step()
            scheduler.step()

            inner_pbar.set_postfix({"energy: ": F.item(), "cost: ": cost.item(), "l1: ": l1.item(), "accuracy: ": a})

        average_F = sum(F_list_e) / len(F_list_e)
        average_cost = sum(cost_list_e) / len(cost_list_e)
        average_l1 = sum(l1_list_e) / len(l1_list_e)
        average_accuracy = sum(accuracy_list_e) / len(accuracy_list_e)

        F_list.append(average_F)
        cost_list.append(average_cost)
        l1_list.append(average_l1)
        accuracy_list.append(average_accuracy)

        pbar.set_postfix({"energy: ": average_F, "cost: ": average_cost, "l1: ": average_l1, "accuracy: ": average_accuracy})