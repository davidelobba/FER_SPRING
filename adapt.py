
import torch
import torch.nn.functional as F


from argparse import ArgumentParser
from datetime import datetime
import wandb
import yaml
import numpy as np

from models.MoCo import MoCo
from models.Encoder import Encoder
from models.FER_GAT import FER_GAT
from models.STGCN import STGCN, get_normalized_adj
from matplotlib import pyplot as plt
from utils.SupCon import SupConLoss
from tqdm import tqdm


#from landmark.landmark_detection import extract_landmark
import cv2
import face_alignment
from utils.utils import split_dataset

from data.RAVDESS_LD import RAVDESS_LANDMARK




def drawLandmark_multiple(img,  landmark):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
    return img

def main(args):

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device =args.device
    adj = config["model_params"]["adj_matr"]

    sample_test, sample_train = split_dataset(path=config["dataset"]["path"], perc=config["dataset"]["split_percentage"],path_audio=config["dataset"]["path_audio"])
    tot_dataset =sample_test + sample_train
    print(f"{len(tot_dataset)} sample {len(sample_test)} train {len(sample_train)}")
    audio =False 
    if config["dataset"]["path_audio"] is not None:
        audio =True 
    dataset_test = RAVDESS_LANDMARK(config["dataset"]["path"], samples=sample_test, min_frames=config["dataset"]["min_frames"],n_mels=config["dataset"]["n_mels"],test=True, audio=audio,audio_only=config["training"]["audio_only"],zero_start=config["dataset"]["zero_start"],  contrastive=config["training"]["contrastive"],  mixmatch=config["training"]["augmented"], random_aug=config["training"]["random_aug"])        
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config["training"]["batch_size"], shuffle=True,num_workers=config["training"]["num_workers"], drop_last= False)

    #dataset_train = RAVDESS_LANDMARK(config["dataset"]["path"], samples=tot_dataset, min_frames=config["dataset"]["min_frames"],n_mels=config["dataset"]["n_mels"],test=False, audio=audio,audio_only=config["training"]["audio_only"],zero_start=config["dataset"]["zero_start"],  contrastive=config["training"]["contrastive"],  mixmatch=config["training"]["augmented"], random_aug=config["training"]["random_aug"])        
    #loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True,num_workers=config["training"]["num_workers"], drop_last= False)


    num_nodes = 51
    num_feat_in = 2

    if config["dataset"]["path_audio"] is not None:
        num_feat_in = 3

    #encoder = Encoder(config_file=args.config, device=args.device)
    #linear = torch.nn.Sequential(torch.nn.Linear(256, 128),torch.nn.ReLU(),torch.nn.Linear(128, config["dataset"]["classes"]))
    
    encoder = STGCN(num_nodes,num_feat_in,config["dataset"]["min_frames"],config["model_params"]["feat_out"], num_classes=128,edge_weight=config["model_params"]["edge_weight"], contrastive=config["training"]["contrastive"])#config["dataset"]["classes"]
    linear = torch.nn.Sequential(torch.nn.Linear(config["model_params"]["feat_out"]*num_nodes, 512),torch.nn.ReLU(),torch.nn.Linear(512, config["dataset"]["classes"]))
    linear = linear.to(args.device)
    encoder = encoder.to(args.device)   

    encoder.load_state_dict(torch.load(args.model_encoder,map_location=device)["model_state_dict"])
    linear.load_state_dict(torch.load(args.model_linear,map_location=device)["model_state_dict"])
    
    optimizer_encoder = torch.optim.SGD(encoder.parameters(),config["training"]["lr_encoder"], weight_decay=config["training"]["wd"], momentum=config["training"]["momentum"])
    optimizer_encoder.load_state_dict(torch.load(args.model_encoder,map_location=device)["optimizer_state_dict"])

    optimizer_decoder = torch.optim.SGD(linear.parameters(),config["training"]["lr_linear"], weight_decay=config["training"]["wd"], momentum=config["training"]["momentum"])
    optimizer_decoder.load_state_dict(torch.load(args.model_linear,map_location=device)["optimizer_state_dict"])

    encoder_loss = SupConLoss()
    linear_loss = torch.nn.CrossEntropyLoss()

    print("..................................")

    with open(adj, 'rb') as f:
        A = np.load(f)
    A_hat = torch.Tensor(get_normalized_adj(A)).to(device)
   
    epochs = 100
    unsupervised = True
    
    for e in range(epochs):
        samples = 0.
        batch_count =0
        cumulative_contr_loss = 0.
        cumulative_accuracy = 0.
        label_pred = [0,0,0,0,0,0,0,0]
        label_pred_count = [0,0,0,0,0,0,0,0]
        label_count = [0,0,0,0,0,0,0,0]

        encoder.eval()
        linear.eval()

        with torch.no_grad():                
            for _, batch in enumerate(tqdm(loader_test)) :

                if len(batch) ==3:
                    targets, ld_1, ld_2 =  batch[0].to(device),batch[1].to(device), batch[2].to(device)
                else:
                    targets, ld_1, ld_2, ad_1, ad_2 =  batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device),batch[4].to(device)

                if len(batch) ==3:  
                    q1, vf_q1 = encoder(A_hat, ld_1)
                    q2, vf_q2 = encoder(A_hat, ld_2)
                else:
                    q1, vf_q1 = encoder(ld_1, ad_1)
                    q2, vf_q2 = encoder(ld_2, ad_2)
                
                contr_feat = torch.cat((q1.unsqueeze(1),q2.unsqueeze(1)),1)

                if unsupervised:
                    contr_loss = encoder_loss(contr_feat)
                else:
                    contr_loss = encoder_loss(contr_feat, targets)

                contr_loss = encoder_loss(contr_feat, targets)
                
                video_feat = vf_q1.detach()
                logits = linear(video_feat)
                

                batch_size = ld_1.shape[0]
                samples+=logits.shape[0]
                batch_count +=1
                cumulative_contr_loss += contr_loss.item() # Note: the .item() is needed to extract scalars from tensors
                _, predicted = logits.max(1)
                cumulative_accuracy += predicted.eq(targets).sum().item()

                for i in range(predicted.shape[0]):
                    if predicted[i] == targets[i]:
                        label_pred[predicted[i]] +=1
                    label_count[targets[i]] +=1
        final_contr_loss = cumulative_contr_loss/batch_count
        accuracy = cumulative_accuracy/samples*100
        print(f"accuracy {accuracy} final_contr_loss {final_contr_loss}")


        encoder.train()
        linear.train()

        for _, batch in enumerate(tqdm(loader_test)) :

            if len(batch) ==3:
                targets, ld_1, ld_2 =  batch[0].to(device),batch[1].to(device), batch[2].to(device)
            else:
                targets, ld_1, ld_2, ad_1, ad_2 =  batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device),batch[4].to(device)

            if len(batch) ==3:  
                q1, vf_q1 = encoder(A_hat, ld_1)
                q2, vf_q2 = encoder(A_hat, ld_2)
            else:
                q1, vf_q1 = encoder(ld_1, ad_1)
                q2, vf_q2 = encoder(ld_2, ad_2)
            
            video_feat = vf_q1.detach()
            logits1 = linear(video_feat)

            video_feat = vf_q2.detach()
            logits2 = linear(video_feat)
            logits = F.sigmoid((logits1 + logits2)/2)

            #print(logits)
            tes, predicted = logits.max(1)
            #print(f"{torch.where(tes > 0.998)} , {predicted[torch.where(tes > 0.998)]}")
            #print(f"test {tes} predicted {predicted.shape}")
            print(torch.where(logits > 0.999) ) 

            loss = linear_loss(logits[torch.where(logits > 0.999)], targets[torch.where(logits > 0.999)])            
            optimizer_decoder.zero_grad()
            loss.backward()
            optimizer_decoder.step()

            #contr_feat = torch.cat((q1.unsqueeze(1),q2.unsqueeze(1)),1)

            # if unsupervised:
            #     contr_loss = encoder_loss(contr_feat)
            # else:
            #     contr_loss = encoder_loss(contr_feat, targets)

            #contr_loss = encoder_loss(contr_feat, targets)
            #optimizer_encoder.zero_grad()
            #contr_loss.backward()
            #optimizer_encoder.step()
            
            cumulative_contr_loss += 0#contr_loss.item() # Note: the .item() is needed to extract scalars from tensors

        final_contr_loss = cumulative_contr_loss/batch_count
        #accuracy = cumulative_accuracy/samples*100
        print(f"loss {final_contr_loss}" )
        #print(f"accuracy {accuracy} final_contr_loss {final_contr_loss}")


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda:2', type=str, help='device')
    parser.add_argument('--model_encoder', default=None, required=True , help='folder where to store the ckp')
    parser.add_argument('--model_linear', default=None, required=True , help='folder where to store the ckp')
    parser.add_argument('--config', default=None, required=True , type=str, help='path to config file')
    parser.add_argument('--plot', default=False,  type=bool , help='path to config file')

    args = parser.parse_args()

    main(args)