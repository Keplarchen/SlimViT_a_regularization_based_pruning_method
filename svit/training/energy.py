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
    Computes the energy function for a model, combining the loss function, L1 regularization penalty,
    and an optional penalty based on model accuracy if constraints are not met. The function evaluates
    the tradeoff between achieving a target accuracy and minimizing other terms in the energy function.

    :param model: The machine learning model being evaluated. Assumes it's a PyTorch model.
    :param output: The output predictions of the model produced during forward pass.
    :param y: The ground truth labels corresponding to the input data.
    :param val_dataloader: A DataLoader instance used to assess model accuracy on validation data.
    :param criterion: The loss function used for computing the cost between `output` and `y`.
    :param lambda_l1: Hyperparameter controlling the impact of L1 regularization on the energy function.
    :param target_accuracy: Desired minimum accuracy value for the model during validation.
    :param lambda_accuracy: Hyperparameter adjusting the penalty factor when target accuracy is not achieved.
    :param multi_prune: If True, applies L1 regularization to multiple model scalers; otherwise, applies
                       regularization to a single scaler.
    :param accuracy_tradeoff: If True, incorporates accuracy penalty into the energy function based on
                              target accuracy constraints.
    :return: A tuple containing the energy function value, the cost value from the loss function, the
             L1 regularization penalty, and the validation accuracy of the model.
    """
    cost = criterion(output, y)

    if multi_prune:
        scaler_l1 = sum(torch.norm(s.scaler, p=1) for s in model.scaler) * lambda_l1
    else:
        scaler_l1 = torch.norm(model.scaler.scaler, p=1) * lambda_l1

    # # TODO: sparsity
    # sparsity = 0.0

    accuracy = 0.0
    accuracy_penalty = 0.0
    if accuracy_tradeoff:
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
            accuracy_penalty = (accuracy / target_accuracy) ** lambda_accuracy

    F = cost + scaler_l1 + accuracy_penalty
    return F, cost, scaler_l1, accuracy
