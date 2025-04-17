import os
import yaml
import torch

with open(os.path.join(os.path.dirname(__file__), "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'