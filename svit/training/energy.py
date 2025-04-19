import torch
import torch.nn as nn

from svit import config, device

from tqdm.notebook import tqdm
from torch.utils.data import DataLoader

def energy_function(model: nn.Module,
                    output: torch.Tensor,
                    y: torch.Tensor,
                    val_dataloader: DataLoader,
                    criterion: nn.Module,
                    lambda_l1: float=config["energy"]["lambda_l1"],
                    target_accuracy: float=config["energy"]["target_accuracy"],
                    lambda_accuracy: float=config["energy"]["lambda_accuracy"],
                    multi_prune: bool=config["models"]["multi_prune"],
                    accuracy_tradeoff: bool=config["energy"]["accuracy_tradeoff"]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """

    :param model:
    :param output:
    :param y:
    :param val_dataloader:
    :param criterion:
    :param lambda_l1:
    :param target_accuracy:
    :param lambda_accuracy:
    :param multi_prune:
    :param accuracy_tradeoff:
    :return:
    """
    cost = criterion(output, y)
    scaler_l1 = 0.0
    if not model.is_base_model:
        if multi_prune:
            scaler_l1 = sum(torch.norm(s.scaler, p=1) for s in model.scaler) * lambda_l1
        else:
            scaler_l1 = torch.norm(model.scaler.scaler, p=1) * lambda_l1

    total_correct = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for x_val, y_val in tqdm(val_dataloader, leave=False):
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            logits = model(x_val)
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == y_val).sum().item()
            total_samples += y_val.size(0)
    model.train()
    accuracy = total_correct / total_samples
    accuracy_penalty = 0.0
    if not model.is_base_model:
        if accuracy_tradeoff and accuracy < target_accuracy:
            accuracy_penalty = (accuracy / target_accuracy) ** lambda_accuracy

    F = cost + scaler_l1 + accuracy_penalty
    return F, cost, scaler_l1, accuracy
