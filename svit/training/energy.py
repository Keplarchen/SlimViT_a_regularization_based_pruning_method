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
                    multi_prune: bool=config["models"]["multi_prune"]) -> tuple[torch.Tensor, float, float, float]:
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
        scaler_l1 = sum(torch.norm(s.scaler, p=1) for s in model.scaler) ** lambda_l1
    else:
        scaler_l1 = torch.norm(model.scaler.scaler, p=1) ** lambda_l1

    # # TODO: sparsity
    # sparsity = 0.0

    model.eval()
    accuracy_list = []
    with torch.no_grad():
      for data in tqdm(val_dataloader, leave=False):
        x, y = data
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        predict = torch.argmax(output, dim=-1)
        accuracy_list.append((predict == y).sum().item() / y.shape[0])
    model.train()
    accuracy = sum(accuracy_list) / len(accuracy_list)
    if accuracy < target_accuracy:
      accuracy_penalty = 0
    else:
      accuracy_penalty = (accuracy / target_accuracy) ** lambda_accuracy

    F = cost + scaler_l1 + lambda_accuracy * accuracy_penalty
    return F, cost, scaler_l1, accuracy
