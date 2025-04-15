import torch
import torch.nn as nn

from svit import config

from torch.utils.data import DataLoader

def energy_function(model: nn.Module,
                    output: torch.Tensor,
                    y: torch.Tensor,
                    val_dataloader: DataLoader,
                    criterion: nn.Module,
                    lambda_l1: float=config["energy"]["lambda_l1"],
                    target_sparsity: float=config["energy"]["target_sparsity"],
                    lambda_sparsity: float=config["energy"]["lambda_sparsity"],
                    target_accuracy: float=config["energy"]["target_accuracy"],
                    lambda_accuracy: float=config["energy"]["lambda_accuracy"],
                    multi_prune: bool=config["models"]["multi_prune"]) -> tuple[torch.Tensor, float, float, float, float]:
    """

    :param model:
    :param output:
    :param y:
    :param val_dataloader:
    :param criterion:
    :param lambda_l1:
    :param target_sparsity:
    :param lambda_sparsity:
    :param target_accuracy:
    :param lambda_accuracy:
    :param multi_prune:
    :return:
    """
    cost = criterion(output, y)

    if multi_prune:
        scaler_l1 = lambda_l1 * sum(torch.norm(s.scaler, p=1) for s in model.scaler)
    else:
        scaler_l1 = lambda_l1 * torch.norm(model.scaler.scaler, p=1)

    # TODO: sparsity
    sparsity = 0.0
    # TODO: accuracy
    accuracy = 0.0

    F = cost + lambda_l1 * scaler_l1 + lambda_sparsity * sparsity + lambda_accuracy * accuracy
    return F, cost, scaler_l1, sparsity, accuracy
