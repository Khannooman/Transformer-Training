from train import Trainer
from dataloader import DataModule
from model import load_model
import config


def setup_training(data_dir : str , config) -> None:
    """Combine all the module and setup training"""
    ### load dataloader
    datamodule = DataModule(config)
    train_loader, val_loader, test_loader = datamodule.create_dataloaders(data_dir)
    ### load model
    model = load_model()

    ### load trainer
    trainer = Trainer(config)

    ### hit the trainer
    trainer.train(train_loader, val_loader)

data_dir = config.dataset_dir

 
### hit tthe training
if __name__ == "__main__":
    setup_training(data_dir, config)