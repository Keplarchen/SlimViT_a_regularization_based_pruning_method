import gc
import torch
import torch.nn as nn
import torch.optim as optim

from svit import config, device
from svit.training.energy import energy_function

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.notebook import tqdm

from layer_evaluation import evaluate_and_log, plot_layer_distribution

def get_criterion() -> nn.Module:
    """

    :return:
    """
    return nn.CrossEntropyLoss()

def get_optimizer(model: nn.Module,
                  lr: float,
                  weight_decay: float) -> optim.Optimizer:
    """

    :param model:
    :param lr:
    :param weight_decay:
    :return:
    """
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def get_scheduler(optimizer: optim.Optimizer,
                  t_max: int) -> CosineAnnealingLR:
    """

    :param optimizer:
    :param t_max:
    :return:
    """
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

def train(model:nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          config_var: dict=config) -> None:
    """

    :param model:
    :param train_dataloader:
    :param val_dataloader:
    :param config_var:
    :return:
    """
    epoch = config_var["training"]["epoch"]
    accuracy_tradeoff = config_var["energy"]["accuracy_tradeoff"]

    criterion = get_criterion()

    lr = config_var["training"]["learning_rate"]
    weight_decay = config_var["training"]["weight_decay"]
    optimizer = get_optimizer(model, lr, weight_decay)

    t_max = config_var["training"]["T_max"]
    scheduler = get_scheduler(optimizer, t_max)

    model.to(device)
    criterion.to(device)

    pbar = tqdm(range(epoch))

    F_list = []
    cost_list = []
    l1_list = []
    accuracy_list = []
    best_accuracy = 0.0
    for epoch in pbar:
        F_list_e = []
        cost_list_e = []
        l1_list_e = []
        accuracy_list_e = []

        inner_pbar = tqdm(train_dataloader, leave=False)

        evaluate_and_log(model, val_dataloader, device=device, save_csv=True, epoch=epoch)
        plot_layer_distribution(model, val_dataloader, device=device, epoch=epoch)
        for data in inner_pbar:
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            x, y = data
            x, y = x.to(device), y.to(device)

            output = model(x)

            F, cost, l1, a, s = energy_function(model, output, y, val_dataloader, criterion)

            F_list_e.append(F)
            cost_list_e.append(cost)
            l1_list_e.append(l1)
            accuracy_list_e.append(a)

            optimizer.zero_grad()
            F.backward()
            optimizer.step()
            scheduler.step()

            if not model.is_base_model:
                inner_pbar.set_postfix({"energy: ": F.item(), "cost: ": cost.item(), "l1: ": l1.item(), "accuracy: ": a, "sparsity: ": s})
            else:
                inner_pbar.set_postfix({"energy: ": F.item(), "cost: ": cost.item(), "l1: ": l1, "accuracy: ": a, "sparsity: ": s})

        average_F = sum(F_list_e) / len(F_list_e)
        average_cost = sum(cost_list_e) / len(cost_list_e)
        average_l1 = sum(l1_list_e) / len(l1_list_e)
        average_accuracy = sum(accuracy_list_e) / len(accuracy_list_e)

        F_list.append(average_F)
        cost_list.append(average_cost)
        l1_list.append(average_l1)
        accuracy_list.append(average_accuracy)

        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            torch.save(model.state_dict(), "checkpoint.pt")

        pbar.set_postfix({"energy: ": average_F, "cost: ": average_cost, "l1: ": average_l1, "accuracy: ": average_accuracy})
    return
