import torch

model_name = "ProsusAI/finbert"
dataset_id = "dair-ai/emotion"
output_dir = "./model_output"
dataset_dir = "./data"
wandb_project = "TextClassifier"
seed = 123
learning_rate = 2e-5
batch_size = 256
epochs = 10
warmup_ratio = 0.1
early_stopping_patience = 3
max_length = 128
num_workers = 2 
weight_decay = 0.01
device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_wandb = True
warmup_steps = 0.05
number_of_class = 6
hidden_size = 1024