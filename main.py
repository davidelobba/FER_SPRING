
import torch
from argparse import ArgumentParser
from datetime import datetime
import wandb
import yaml
import numpy as np

from models.MoCo import MoCo
from models.FER_GAT import FER_GAT
from models.STGCN import STGCN, get_normalized_adj
from models.SKELETON_STGCN import SK_STGCN
from models.Encoder import Encoder
from models.AudioEncoder import AudioEncoder
from models.TCN import TCN

from utils.SupCon import SupConLoss
from utils.LinearWarmupScheduler import LinearWarmupCosineAnnealingLR
from utils.FocalLoss import FocalLoss
from utils.utils import split_dataset
#from focal_loss.focal_loss import FocalLoss

from data.CAER import CAER
from data.RAVDESS_LD import RAVDESS_LANDMARK
from data.CAER_LD import CAER_LANDMARK
from data.AffWild2 import AffWild

from torch.utils.data import WeightedRandomSampler



def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if bool(args.wandb):
        import wandb
        wandb.init(project="fer",  config={
                    "learning_rate_encoder": config["training"]["lr_encoder"],
                    "learning_rate_classif": config["training"]["lr_linear"],
                    "backbone": config["model_params"]["backbone"],
                    "split_percentage": config["dataset"]["split_percentage"],
                    "frame_l": config["dataset"]["n_frames"],
                    "dataset": args.dataset
                } )
    else:
        wandb = None

    num_nodes = 51
    num_feat_in = 2

    print(f"------ Initializing dataset...")
    # data initialization
    sampler = None 
    if args.dataset == "CAER":
        dataset_train = CAER(config["dataset"]["train"]["path"],dim_image=config["dataset"]["train"]["dim_image"], min_frames=config["dataset"]["train"]["min_frames"])
        dataset_test = CAER(config["dataset"]["test"]["path"],dim_image=config["dataset"]["train"]["dim_image"], min_frames=config["dataset"]["test"]["min_frames"])
    elif args.dataset== "AFFWILD":
        dataset_train = AffWild(config["dataset"]["path_train"],n_frames=config["dataset"]["n_frames"], audio=config["dataset"]["audio"],audio_only=config["training"]["audio_only"],audio_separate=config["training"]["audio_separate"], contrastive=config["training"]["contrastive"], block_dimension=config["dataset"]["block_dimension"], twod=config["dataset"]["twod"], random_aug=config["training"]["random_aug"])
        dataset_test =  AffWild(config["dataset"]["path_test"],test=True, n_frames=config["dataset"]["n_frames"], audio=config["dataset"]["audio"],audio_only=config["training"]["audio_only"],audio_separate=config["training"]["audio_separate"], contrastive=config["training"]["contrastive"], block_dimension=config["dataset"]["n_frames"], twod=config["dataset"]["twod"])
        print(dataset_train.__len__())
        print(dataset_test.__len__())
        if not config["dataset"]["twod"]:
            num_feat_in = 3
        if config["dataset"]["audio"] and not config["training"]["audio_separate"] :
            num_feat_in += 1
    elif args.dataset== "RAVDESS":
        sample_test, sample_train = split_dataset(path=config["dataset"]["path"], perc=config["dataset"]["split_percentage"],path_audio=config["dataset"]["path_audio"], actor_split=config["dataset"]["actor_split"])
        audio =False 
        if config["dataset"]["path_audio"] is not None:
            audio =True 
        dataset_train = RAVDESS_LANDMARK(config["dataset"]["path"], samples=sample_train, min_frames=config["dataset"]["min_frames"],n_mels=config["dataset"]["n_mels"],audio=audio, audio_only=config["training"]["audio_only"],audio_separate=config["training"]["audio_separate"], zero_start=config["dataset"]["zero_start"], contrastive=config["training"]["contrastive"],  mixmatch=config["training"]["augmented"], random_aug=config["training"]["random_aug"], drop_kp=config["training"]["drop_kp"])
        dataset_test = RAVDESS_LANDMARK(config["dataset"]["path"], samples=sample_test, min_frames=config["dataset"]["min_frames"],n_mels=config["dataset"]["n_mels"],test=True, audio=audio,audio_only=config["training"]["audio_only"],audio_separate=config["training"]["audio_separate"], zero_start=config["dataset"]["zero_start"],  contrastive=config["training"]["contrastive"],  mixmatch=config["training"]["augmented"], random_aug=config["training"]["random_aug"])        
        #dataset_train = RAVDESS_LANDMARK(config["dataset"]["train"]["path"], min_frames=config["dataset"]["train"]["min_frames"], zero_start=config["dataset"]["train"]["zero_start"], contrastive=config["training"]["contrastive"],  mixmatch=config["training"]["augmented"], random_aug=config["training"]["random_aug"], drop_kp=config["training"]["drop_kp"])
        #dataset_test = RAVDESS_LANDMARK(config["dataset"]["test"]["path"], min_frames=config["dataset"]["test"]["min_frames"],  test=True, zero_start=config["dataset"]["train"]["zero_start"],  contrastive=config["training"]["contrastive"],  mixmatch=config["training"]["augmented"], random_aug=config["training"]["random_aug"])
        _ , samples_weight = dataset_train.get_class_sample_count()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        if config["dataset"]["path_audio"] is not None:
            num_feat_in = 3
        if config["training"]["audio_only"]:
            num_nodes = config["dataset"]["n_mels"]
            num_feat_in = 1

    elif args.dataset== "CAER_LD":
        dataset_train = CAER_LANDMARK(config["dataset"]["train"]["path"], min_frames=config["dataset"]["train"]["min_frames"], zero_start=config["dataset"]["train"]["zero_start"],  contrastive=config["training"]["contrastive"])
        dataset_test = CAER_LANDMARK(config["dataset"]["test"]["path"], min_frames=config["dataset"]["test"]["min_frames"],  test=True, zero_start=config["dataset"]["train"]["zero_start"],  contrastive=config["training"]["contrastive"])

    if sampler is None:
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["training"]["batch_size"],shuffle=True,num_workers=config["training"]["num_workers"], drop_last= False)
    else:
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["training"]["batch_size"],shuffle=True,num_workers=config["training"]["num_workers"], drop_last= False, sampler=sampler)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config["training"]["batch_size"], shuffle=True,num_workers=config["training"]["num_workers"], drop_last= False)
    print(f"------ Dataset Initialized")
    
    # encoder + Moco
    print(f"------ Initializing networks")


    if config["training"]["contrastive"]:
        with open(config["model_params"]["adj_matr"], 'rb') as f:
            A = np.load(f)
            
        A_hat = torch.Tensor(get_normalized_adj(A)).to(args.device)
        if config["training"]["audio_separate"]:
            encoder = Encoder(config_file=args.config, device=args.device, num_feat_video=num_feat_in)
            linear = torch.nn.Sequential(torch.nn.Linear(512, 256),torch.nn.ReLU(),torch.nn.Linear(256, config["dataset"]["classes"]))
            # test with audio preprocess
            #linear = torch.nn.Sequential(torch.nn.Linear(config["model_params"]["feat_out"]*num_nodes, 512),torch.nn.ReLU(),torch.nn.Linear(512, config["dataset"]["classes"]))
        else:
            encoder = STGCN(num_nodes,num_feat_in,config["dataset"]["n_frames"],config["model_params"]["feat_out"], num_classes=128,edge_weight=config["model_params"]["edge_weight"], contrastive=config["training"]["contrastive"], separate_graph=config["model_params"]["separate_graph"], attention=config["model_params"]["attention"]) #config["dataset"]["classes"]
            if config["model_params"]["separate_graph"]:
                linear = torch.nn.Sequential(torch.nn.Linear(config["model_params"]["feat_out"]*4, 512),torch.nn.ReLU(),torch.nn.Linear(512, config["dataset"]["classes"]))
            else:
                linear = torch.nn.Sequential(torch.nn.Linear(config["model_params"]["feat_out"]*num_nodes, 512),torch.nn.ReLU(),torch.nn.Linear(512, config["dataset"]["classes"]))
        
        #############
        #encoder = Encoder(config_file=args.config, device=args.device)
        #linear = torch.nn.Sequential(torch.nn.Linear(512, 256),torch.nn.ReLU(),torch.nn.Linear(256, config["dataset"]["classes"]))
        #linear = torch.nn.Sequential(torch.nn.Linear(256, 128),torch.nn.ReLU(),torch.nn.Linear(128, config["dataset"]["classes"]))
        #################
        
        linear = linear.to(args.device)
        encoder = encoder.to(args.device)

    else:
        if not args.skeleton:
            if config["training"]["audio_only"]:
                model = TCN(in_chan=128, n_blocks=5, n_repeats=2, out_chan=8) #AudioEncoder()
                model = model.to(args.device)
            else:
                # num_nodes, num_features, num_timesteps_input, num_featout
                model = STGCN(num_nodes,num_feat_in,config["dataset"]["min_frames"],config["model_params"]["feat_out"], num_classes=config["dataset"]["classes"],edge_weight=config["model_params"]["edge_weight"], separate_graph=config["model_params"]["separate_graph"], attention=config["model_params"]["attention"])
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
        optimizer_encoder = torch.optim.SGD(encoder.parameters(),config["training"]["lr_encoder"], weight_decay=config["training"]["wd"], momentum=config["training"]["momentum"])
        optimizer_decoder = torch.optim.SGD(linear.parameters(),config["training"]["lr_linear"], weight_decay=config["training"]["wd"], momentum=config["training"]["momentum"])
        scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=config["training"]["scheduler_step"], gamma=config["training"]["scheduler_gamma"])
        scheduler_decoder = torch.optim.lr_scheduler.StepLR(optimizer_decoder, step_size=config["training"]["scheduler_step"], gamma=config["training"]["scheduler_gamma"])

    else:
        optimizer = torch.optim.SGD(model.parameters(),config["training"]["lr_encoder"], weight_decay=config["training"]["wd"], momentum=config["training"]["momentum"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step"], gamma=config["training"]["scheduler_gamma"])
    
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
        train(encoder, linear, loader_train, [optimizer_encoder, optimizer_decoder], [scheduler_encoder, scheduler_decoder], encoder_loss, classifier_loss, wandb, epochs=config["training"]["epochs"], device=args.device, test=True, loader_test=loader_test, log_model=config["training"]["log_model"], output_dir=args.output,  adj=config["model_params"]["adj_matr"], config_file=args.config)
    else:
        from train_model_graph import train
        train(model,loader_train, optimizer, classifier_loss, wandb,scheduler, epochs=config["training"]["epochs"], device=args.device, test=True, loader_test=loader_test, log_model=config["training"]["log_model"], output_dir=args.output, adj=config["model_params"]["adj_matr"], config_file=args.config)
 

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