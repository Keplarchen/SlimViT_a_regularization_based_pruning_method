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

    :return:
    """
    return nn.CrossEntropyLoss()

def get_optimizer(model: nn.Module,
                  lr: float=config["training"]["learning_rate"],
                  weight_decay: float=config["training"]["weight_decay"]) -> optim.Optimizer:
    """

    :param model:
    :param lr:
    :param weight_decay:
    :return:
    """
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer: optim.Optimizer,
                  t_max: int=config["training"]["T_max"]) -> CosineAnnealingLR:
    """

    :param optimizer:
    :param t_max:
    :return:
    """
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

def train(model:nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          epoch: int=config["training"]["epoch"]) -> None:
    """

    :param model:
    :param train_dataloader:
    :param val_dataloader:
    :param epoch:
    :return:
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
        for data in tqdm(train_dataloader, leave=False):
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            x, y = data
            x, y = x.to(device), y.to(device)

            output = model(x)
            F, cost, l1, a = energy_function(model, output, y, val_dataloader, criterion)

            optimizer.zero_grad()
            F.backward()
            optimizer.step()
            scheduler.step()

        average_F = sum(F_list_e) / len(F_list_e)
        average_cost = sum(cost_list_e) / len(cost_list_e)
        average_l1 = sum(l1_list_e) / len(l1_list_e)
        average_accuracy = sum(accuracy_list_e) / len(accuracy_list_e)

        F_list.append(average_F)
        cost_list.append(average_cost)
        l1_list.append(average_l1)
        accuracy_list.append(average_accuracy)

        pbar.postfix({"energy: ": average_F, "cost: ": average_cost, "l1: ": average_l1, "accuracy: ": average_accuracy})