
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

    wandb.init(project="fer_CAER",  config={
                "learning_rate_encoder": config["training"]["lr_encoder"],
                "learning_rate_classif": config["training"]["lr_linear"],
                "backbone": config["model_params"]["backbone"],
                "img_dim": config["dataset"]["train"]["dim_image"],
                "frame_l": config["dataset"]["train"]["min_frames"],
                "dataset": args.dataset
            } )

    print(f"------ Initializing dataset...")
    # data initialization
    dataset_train = CAER(config["dataset"]["train"]["path"],dim_image=config["dataset"]["train"]["dim_image"], min_frames=config["dataset"]["train"]["min_frames"])
    dataset_test = CAER(config["dataset"]["train"]["path"],dim_image=config["dataset"]["train"]["dim_image"], min_frames=config["dataset"]["train"]["min_frames"])
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["training"]["batch_size"],num_workers=config["training"]["num_workers"], drop_last= True)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config["training"]["batch_size"], num_workers=config["training"]["num_workers"], drop_last= True)
    print(f"------ Dataset Initialized")
    
    # encoder + Moco
    print(f"------ Initializing networks")
    #base_encoder = EncoderResnet()
    moco_encoder = MoCo(EncoderResnet)
    moco_encoder = moco_encoder.to(args.device)
    linear = torch.nn.Linear(512, config["dataset"]["train"]["classes"]).to(args.device)
    linear = linear.to(args.device)
    print(f"------ Networks Initialized")

    # loss
    print(f"------ Creating the losses")
    encoder_loss = SupConLoss(batch_size = config["training"]["batch_size"])
    classifier_loss = torch.nn.CrossEntropyLoss()
    
    # optimizer
    print(f"------ Creating the optimizers")
    optimizer = torch.optim.SGD([{'params': moco_encoder.parameters()},
              {'params': linear.parameters(), 'lr': config["training"]["lr_linear"]}], lr=config["training"]["lr_encoder"], weight_decay=config["training"]["wd"], momentum=config["training"]["momentum"]) 
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs = 5, max_epochs = 150)

    # train
    print(f"------ Training Started")
    train(moco_encoder, linear, loader_train, optimizer, scheduler, encoder_loss, classifier_loss, wandb, epochs=config["training"]["epochs"], device=args.device, test=True, loader_test=loader_test, log_model=config["training"]["log_model"], output_dir=args.output)
    

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda:2', type=str, help='device')
    parser.add_argument('--dataset', default='CAER', type=str, help='dataset| default CAER')
    parser.add_argument('--offline', default=True,  help='set offline wandb')
    parser.add_argument('--output', default=None, required=True , help='folder where to store the ckp')
    parser.add_argument('--config', default=None, required=True , type=str, help='path to config file')
    args = parser.parse_args()

    main(args)