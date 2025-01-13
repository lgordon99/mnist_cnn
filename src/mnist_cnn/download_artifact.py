import torch
import wandb
from model import MyAwesomeModel

run = wandb.init()
artifact = run.use_artifact('luciagordon-harvard-university-org/wandb-registry-model/corrupt-mnist-model:latest', type='model')
artifact_dir = artifact.download("corrupt-mnist-model")
model = MyAwesomeModel()
model.load_state_dict(torch.load("corrupt-mnist-model/model.pth"))

