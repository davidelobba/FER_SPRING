
import torch
from argparse import ArgumentParser
from datetime import datetime
import wandb
import yaml

from models.MoCo import MoCo
from models.Encoder import EncoderResnet
from utils.SupCon import SupConLoss
from utils.LinearWarmupScheduler import LinearWarmupCosineAnnealingLR
from data.CAER import CAER
from train_model import train 


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    wandb.init(project="fer_CAER", config={
                "learning_rate": config["training"]["lr"],
                "backbone": config["model_params"]["backbone"],
                "dataset": args.dataset,
            })

    # data initialization
    dataset_train = CAER(config["dataset"]["train"]["path"],config["dataset"]["train"]["dim_image"], config["dataset"]["train"]["min_frames"])
    dataset_test = CAER(config["dataset"]["train"]["path"],config["dataset"]["train"]["dim_image"], config["dataset"]["train"]["min_frames"])
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["training"]["batch_size"],num_workers=config["dataset"]["num_workers"], drop_last= True)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config["training"]["batch_size"], num_workers=config["dataset"]["num_workers"], drop_last= True)

    # encoder + Moco
    base_encoder = EncoderResnet()
    moco_encoder = MoCo(base_encoder)
    linear = torch.nn.Linear(512, config["dataset"]["train"]["classes"]).to(device)
    
    # loss
    encoder_loss = SupConLoss(batch_size = config["training"]["batch_size"])
    classifier_loss = torch.nn.CrossEntropyLoss()
    
    # optimizer
    optimizer = torch.optim.SGD([{'params': moco.parameters()},
              {'params': linear.parameters(), 'lr': config["training"]["lr_linear"]}], lr=config["training"]["lr_encoder"], weight_decay=config["training"]["wd"], momentum=config["training"]["momentum"]) 
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs = 5, max_epochs = 150)

    # train
    train(moco_encoder, linear, loader_train, optimizer, scheduler, encoder_loss, classifier_loss, wandb,epoch=config["training"]["epoch"],device=args.device, test=True, loader_test=loader_test)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:2', type=str, help='device')
    parser.add_argument('--dataset', default='CAER', type=str, help='dataset| default CAER')
    parser.add_argument('--config', default=None, required=True , type=str, help='path to config file')
    args = parser.parse_args()

    main(args)