from train import Trainer
from dataloader import create_dataloaders
import config


def setup_training(data_dir : str , config) -> None:
    """Combine all the module and setup training"""
    ### load dataloader
    train_loader, val_loader, test_loader = create_dataloaders(config)

    ### load trainer
    trainer = Trainer(config)

    ### hit the trainer
    trainer.train(train_loader, val_loader)

data_dir = config.dataset_dir

 
### hit tthe training
if __name__ == "__main__":
    setup_training(data_dir, config)