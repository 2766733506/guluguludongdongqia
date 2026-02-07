import sys
sys.path.append('../../../')
import flwr as fl
from client import Client
from recovery import construct_model
from data import split_datasets

datasets = split_datasets(100)  # 100 份数据集


model, _ = construct_model('ResNet18', 10)
client = Client(model, datasets)   # 每个用户用不同数据

try:
    model, _ = construct_model('ResNet18', 10)
    client = Client(model, datasets)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client
    )
except Exception:
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client
    )