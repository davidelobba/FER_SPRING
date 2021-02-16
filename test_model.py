
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
from matplotlib import pyplot as plt

#from landmark.landmark_detection import extract_landmark
import cv2
import face_alignment


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
    with open(adj, 'rb') as f:
        A = np.load(f)
    A_hat = torch.Tensor(get_normalized_adj(A)).to(device)

    num_nodes = A.shape[0]
    #### for RAVDESS
    #label = ["neutral", "calm", "happy","sad", "angry", "fearful", "disgust", "surprised"]
    #### for CAER
    label =  ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    # {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}


    #num_nodes, num_features, num_timesteps_input, num_timesteps_output
    model = STGCN(num_nodes,2,config["dataset"]["train"]["min_frames"],8, config["dataset"]["train"]["classes"])
    model.load_state_dict(torch.load(args.model,map_location=device))
    model = model.to(device)

    plot = args.plot
    #cap = cv2.VideoCapture(2)
    
    
    #s = "/home/riccardo/Downloads/Video_Song_Actor_03/Actor_03/01-02-03-01-02-01-03.mp4" #01-02-06-02-02-01-03.mp4" #"/home/riccardo/Datasets/RAVDESS/Test_set/03/01-01-03-01-01-02-24.mp4" #"/home/riccardo/Datasets/CAER_crop/train/Happy/1222.avi"
    #s = "/home/riccardo/Datasets/CAER_crop/validation/Anger/0019.avi" #01-02-06-02-02-01-03.mp4" #"/home/riccardo/Datasets/RAVDESS/Test_set/03/01-01-03-01-01-02-24.mp4" #"/home/riccardo/Datasets/CAER_crop/train/Happy/1222.avi"
    s = "/home/riccardo/Datasets/CAER_crop/validation/Happy/0126.avi"
    cap = cv2.VideoCapture(s)
    ## tensor filled with landmarks up to the number of frames required
    lds = torch.Tensor([]) 
    if plot:
        fig, ax = plt.subplots()
        rects = ax.bar(label,[10,-10,0,0, 0,0,0], label=label)
    
    ## from https://github.com/1adrianb/face-alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    with  torch.no_grad():
        while True:
            
            ret, orig_image = cap.read()

            #print(orig_image)
            if orig_image is None:
                cap = cv2.VideoCapture(s)
                continue
            
            ld = fa.get_landmarks(orig_image)[0]
            image_annot = drawLandmark_multiple(orig_image,ld)
            cv2.imshow('annotated', image_annot)
            lds = torch.cat((lds, torch.Tensor(ld[17:]).unsqueeze(0)),0)
            if lds.shape[0] > 80:
                lds = lds[-80:]
                kx = lds[:,:,0].numpy()
                ky = lds[:,:,1].numpy()
                kx = (kx - np.min(kx))/np.ptp(kx)
                ky = (ky - np.min(ky))/np.ptp(ky)
                norm_ld = np.array([kx,ky]).T
                norm_ld = torch.Tensor(np.rollaxis(norm_ld,1,0))

                out  = model(A_hat,norm_ld.unsqueeze(0).to(device))
                _, predicted = out.max(1)
                cv2.imshow('annotated', image_annot)
                print(f"pred : {label[predicted]}")
                if plot:
                    data = out.squeeze(0).cpu().numpy()
                    for i in range(len(rects)):
                        rects[i].set_height(data[i])
                    fig.canvas.draw()
                    plt.pause(0.00001)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




      
    




if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda:2', type=str, help='device')
    parser.add_argument('--model', default=None, required=True , help='folder where to store the ckp')
    parser.add_argument('--config', default=None, required=True , type=str, help='path to config file')
    parser.add_argument('--plot', default=False,  type=bool , help='path to config file')

    args = parser.parse_args()

    main(args)