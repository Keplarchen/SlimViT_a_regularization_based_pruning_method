import torch
import matplotlib.pyplot as plt
import pandas as pd

layer_output_stats = {}
layer_distributions = []


def register_hooks(model: torch.nn.Module):
    """
    注册 forward hook 用于记录每一层 encoder 的输出统计。
    """
    for i, layer in enumerate(model.vit.encoder.layers):
        def hook_fn(module, input, output, idx=i):
            out = output.detach().cpu()
            stats = {
                "mean": out.mean().item(),
                "std": out.std().item(),
                "min": out.min().item(),
                "max": out.max().item(),
                "sparsity": (out.abs() < 1e-4).float().mean().item()
            }
            layer_output_stats[f"encoder_layer_{idx}"] = stats
        layer.register_forward_hook(hook_fn)


def evaluate_and_log(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = "cuda", save_csv: bool = False, epoch: int = 0):
    """
    在一个 batch 上运行模型，记录所有层的输出分布。
    """
    model.eval()
    x_sample, _ = next(iter(dataloader))
    x_sample = x_sample.to(device)

    register_hooks(model)
    with torch.no_grad():
        _ = model(x_sample)

    if save_csv:
        df = pd.DataFrame.from_dict(layer_output_stats, orient="index")
        df.to_csv(f"layer_output_statistics_epoch{epoch}.csv")

    for name, stats in layer_output_stats.items():
        print(f"{name}:")
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}")


def plot_layer_distribution(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = "cuda", epoch: int = 0):
    """
    画出所有 encoder 层输出的直方图。
    """
    model.eval()
    x_sample, _ = next(iter(dataloader))
    x_sample = x_sample.to(device)

    outputs = []
    hooks = []

    for layer in model.vit.encoder.layers:
        hooks.append(layer.register_forward_hook(lambda m, i, o: outputs.append(o.detach().cpu().flatten())))

    with torch.no_grad():
        _ = model(x_sample)

    for i, out in enumerate(outputs):
        plt.figure()
        plt.hist(out.numpy(), bins=100)
        plt.title(f'Encoder Layer {i} Output Distribution')
        plt.xlabel('Activation')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"layer_distribution_epoch{epoch}_layer{i}.png")
        plt.close()

    for h in hooks:
        h.remove()
