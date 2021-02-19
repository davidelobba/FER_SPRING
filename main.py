
import torch
from argparse import ArgumentParser
from datetime import datetime
import wandb
import yaml
import numpy as np

from models.MoCo import MoCo
from models.Encoder import EncoderResnet
from models.FER_GAT import FER_GAT
from models.STGCN import STGCN, get_normalized_adj
from models.SKELETON_STGCN import SK_STGCN

from utils.SupCon import SupConLoss
from utils.LinearWarmupScheduler import LinearWarmupCosineAnnealingLR
from utils.FocalLoss import FocalLoss
#from focal_loss.focal_loss import FocalLoss

from data.CAER import CAER
from data.RAVDESS_LD import RAVDESS_LANDMARK
from data.CAER_LD import CAER_LANDMARK


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if bool(args.wandb):
        import wandb
        wandb.init(project="fer",  config={
                    "learning_rate_encoder": config["training"]["lr_encoder"],
                    "learning_rate_classif": config["training"]["lr_linear"],
                    "backbone": config["model_params"]["backbone"],
                    "img_dim": config["dataset"]["train"]["dim_image"],
                    "frame_l": config["dataset"]["train"]["min_frames"],
                    "dataset": args.dataset
                } )
    else:
        wandb = None

    print(f"------ Initializing dataset...")
    # data initialization
    if args.dataset == "CAER":
        dataset_train = CAER(config["dataset"]["train"]["path"],dim_image=config["dataset"]["train"]["dim_image"], min_frames=config["dataset"]["train"]["min_frames"])
        dataset_test = CAER(config["dataset"]["test"]["path"],dim_image=config["dataset"]["train"]["dim_image"], min_frames=config["dataset"]["test"]["min_frames"])
    elif args.dataset== "RAVDESS":
        dataset_train = RAVDESS_LANDMARK(config["dataset"]["train"]["path"], min_frames=config["dataset"]["train"]["min_frames"], zero_start=config["dataset"]["train"]["zero_start"], contrastive=config["training"]["contrastive"],  mixmatch=config["training"]["augmented"], random_aug=config["training"]["random_aug"])
        dataset_test = RAVDESS_LANDMARK(config["dataset"]["test"]["path"], min_frames=config["dataset"]["test"]["min_frames"],  test=True, zero_start=config["dataset"]["train"]["zero_start"],  contrastive=config["training"]["contrastive"],  mixmatch=config["training"]["augmented"], random_aug=config["training"]["random_aug"])
    elif args.dataset== "CAER_LD":
        dataset_train = CAER_LANDMARK(config["dataset"]["train"]["path"], min_frames=config["dataset"]["train"]["min_frames"], zero_start=config["dataset"]["train"]["zero_start"],  contrastive=config["training"]["contrastive"])
        dataset_test = CAER_LANDMARK(config["dataset"]["test"]["path"], min_frames=config["dataset"]["test"]["min_frames"],  test=True, zero_start=config["dataset"]["train"]["zero_start"],  contrastive=config["training"]["contrastive"])
      

    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["training"]["batch_size"],shuffle=True,num_workers=config["training"]["num_workers"], drop_last= False)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config["training"]["batch_size"], shuffle=True,num_workers=config["training"]["num_workers"], drop_last= False)
    print(f"------ Dataset Initialized")
    
    # encoder + Moco
    print(f"------ Initializing networks")
    if config["training"]["contrastive"]:

        num_nodes = 51
        #enc_1 = STGCN(num_nodes,2,config["dataset"]["train"]["min_frames"],config["model_params"]["feat_out"], num_classes=config["dataset"]["train"]["classes"],edge_weight=config["model_params"]["edge_weight"], contrastive=config["training"]["contrastive"])
        #enc_2 = STGCN(num_nodes,2,config["dataset"]["train"]["min_frames"],config["model_params"]["feat_out"], num_classes=config["dataset"]["train"]["classes"],edge_weight=config["model_params"]["edge_weight"], contrastive=config["training"]["contrastive"])
        with open(config["model_params"]["adj_matr"], 'rb') as f:
            A = np.load(f)
        A_hat = torch.Tensor(get_normalized_adj(A)).to(args.device)
        #moco_encoder = MoCo(STGCN,A=A_hat, config=config)
        moco_encoder = STGCN(num_nodes,2,config["dataset"]["train"]["min_frames"],config["model_params"]["feat_out"], num_classes=config["dataset"]["train"]["classes"],edge_weight=config["model_params"]["edge_weight"], contrastive=config["training"]["contrastive"])
        moco_encoder = moco_encoder.to(args.device)
        linear = torch.nn.Linear(config["model_params"]["feat_out"]*num_nodes, config["dataset"]["train"]["classes"]).to(args.device)
        linear = linear.to(args.device)

    else:
        # model = FER_GAT(device=args.device, num_frames=config["dataset"]["train"]["min_frames"])
        num_nodes = 51
        if not args.skeleton:
            # num_nodes, num_features, num_timesteps_input, num_featout
            model = STGCN(num_nodes,2,config["dataset"]["train"]["min_frames"],config["model_params"]["feat_out"], num_classes=config["dataset"]["train"]["classes"],edge_weight=config["model_params"]["edge_weight"] )
            model = model.to(args.device)
        else:
            with open(config["model_params"]["adj_matr"], 'rb') as f:
                A = np.load(f)
            A_hat = torch.Tensor(get_normalized_adj(A)).to(args.device)
            model = SK_STGCN(2,config["dataset"]["train"]["classes"], A_hat,True)
            model = model.to(args.device)

    
    print(f"------ Networks Initialized")

    # optimizer
    print(f"------ Creating the optimizers")
    if config["training"]["contrastive"]:
        #optimizer = torch.optim.SGD(moco_encoder.parameters(),config["training"]["lr_encoder"], weight_decay=config["training"]["wd"], momentum=config["training"]["momentum"])
        optimizer = torch.optim.SGD([{'params': moco_encoder.parameters()},
               {'params': linear.parameters(), 'lr': config["training"]["lr_linear"]}], lr=config["training"]["lr_encoder"], weight_decay=config["training"]["wd"], momentum=config["training"]["momentum"]) 
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs = 5, max_epochs = 150)
    else:
        optimizer = torch.optim.SGD(model.parameters(),config["training"]["lr_encoder"], weight_decay=config["training"]["wd"], momentum=config["training"]["momentum"])

    if args.ckp is not None:
        checkpoint = torch.load(args.ckp)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"------ loaded from checkpoint {args.ckp}")


    # loss
    print(f"------ Creating the losses")
    if config["training"]["contrastive"]:
        #encoder_loss = SupConLoss(batch_size = config["training"]["batch_size"])
        encoder_loss = SupConLoss()
        classifier_loss = torch.nn.CrossEntropyLoss()
    else:
        if config["training"]["focal"]:
            classifier_loss = FocalLoss(gamma=2, alpha=0.25)
        else:
            classifier_loss = torch.nn.CrossEntropyLoss()


    # train
    print(f"------ Training Started")
    if config["training"]["contrastive"]:
        from train_model_contrastive import train 
        train(moco_encoder, linear, loader_train, optimizer, scheduler, encoder_loss, classifier_loss, wandb, epochs=config["training"]["epochs"], device=args.device, test=True, loader_test=loader_test, log_model=config["training"]["log_model"], output_dir=args.output,  adj=config["model_params"]["adj_matr"], config_file=args.config)
    else:
        from train_model_graph import train
        train(model,loader_train, optimizer, classifier_loss, wandb, epochs=config["training"]["epochs"], device=args.device, test=True, loader_test=loader_test, log_model=config["training"]["log_model"], output_dir=args.output, adj=config["model_params"]["adj_matr"], config_file=args.config)
 

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda:2', type=str, help='device')
    parser.add_argument('--dataset', default='CAER', type=str, help='dataset| default CAER')
    parser.add_argument('--output', default=None, required=True , help='folder where to store the ckp')
    parser.add_argument('--skeleton', default=False, type=bool, help='if true use skeleton implementation')
    parser.add_argument('--config', default=None, required=True , type=str, help='path to config file')
    parser.add_argument('--wandb', default=False, type=bool,  help='if false wandb doesnt log| defeault True')
    parser.add_argument('--ckp', default=None, help='Path to checkpoint file| default None')
    
    args = parser.parse_args()

    main(args)