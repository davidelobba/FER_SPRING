
import torch
from argparse import ArgumentParser
from datetime import datetime
import wandb
import yaml
import math

import numpy as np

from models.MoCo import MoCo
from models.Encoder import Encoder
from models.FER_GAT import FER_GAT
from models.STGCN import STGCN, get_normalized_adj
from matplotlib import pyplot as plt

#from landmark.landmark_detection import extract_landmark
import cv2
import face_alignment


def drawLandmark_multiple(img,  landmark, color=(0,255,0)):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    for x, y in landmark:
        cv2.circle(img, (int(x), int(y)), 2, color, -1)
    return img

def drawgraph_connection(img,ld, from_ld,to_ld):
    '''
    Input:
    - img: gray or RGB
    - bbox: type of BBox
    - landmark: reproject landmark of (5L, 2L)
    Output:
    - img marked with landmark and bbox
    '''
    kx = ld[17:,0] #.numpy()
    ky = ld[17:,1] #.numpy()
    
    x_values = [kx[from_ld], kx[to_ld]]
    y_values = [ky[from_ld], ky[to_ld]]
    for fr, to in zip(from_ld,to_ld):
        cv2.line(img, (int(kx[fr]), int(ky[fr])), (int(kx[to]), int(ky[to]))  ,(0,255,0), 1)
    return img

def main(args):

    with open(args.config) as f:
        config = yaml.safe_load(f)
    device =args.device
    adj = config["model_params"]["adj_matr"]
    with open(adj, 'rb') as f:
        A = np.load(f)
    from_ld, to_ld = np.nonzero(A)
    A_hat = torch.Tensor(get_normalized_adj(A)).to(device)

    num_nodes = A.shape[0]
    #### for RAVDESS
    #label = ["neutral", "calm", "happy","sad", "angry", "fearful", "disgust", "surprised"]
    #### for CAER
    label =  ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    # {'Anger': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Neutral': 4, 'Sad': 5, 'Surprise': 6}


    #num_nodes, num_features, num_timesteps_input, num_timesteps_output
    #model = STGCN(num_nodes,2,config["dataset"]["train"]["min_frames"],8, config["dataset"]["train"]["classes"])
    #model.load_state_dict(torch.load(args.model,map_location=device))
    #model = model.to(device)

    plot = args.plot
    #cap = cv2.VideoCapture(2)
    
    s = "/home/riccardo/Datasets/AffWild2/phoebe/dk15/new_aff_wild/Aff-Wild2_ready/Expression_Set/videos/Train_Set/video49.mp4"
    s = "/home/riccardo/Datasets/AffWild2/phoebe/dk15/new_aff_wild/Aff-Wild2_ready/Expression_Set/videos/Train_Set/6-30-1920x1080.mp4"
    s = "/home/riccardo/Datasets/AffWild2/phoebe/dk15/new_aff_wild/Aff-Wild2_ready/Expression_Set/videos/Train_Set/46-30-484x360.mp4"
    
    cap = cv2.VideoCapture(s)
    # = "/home/riccardo/Downloads/Video_Song_Actor_03/Actor_03/01-02-03-01-02-01-03.mp4" #01-02-06-02-02-01-03.mp4" #"/home/riccardo/Datasets/RAVDESS/Test_set/03/01-01-03-01-01-02-24.mp4" #"/home/riccardo/Datasets/CAER_crop/train/Happy/1222.avi"
    #s = "/home/riccardo/Datasets/CAER_crop/validation/Anger/0019.avi" #01-02-06-02-02-01-03.mp4" #"/home/riccardo/Datasets/RAVDESS/Test_set/03/01-01-03-01-01-02-24.mp4" #"/home/riccardo/Datasets/CAER_crop/train/Happy/1222.avi"
    #s = "/home/riccardo/Datasets/CAER_crop/validation/Happy/0126.avi"
    #
    ## tensor filled with landmarks up to the number of frames required
    lds = torch.Tensor([]) 
    if plot:
        fig, ax = plt.subplots()
        rects = ax.bar(label,[10,-10,0,0, 0,0,0], label=label)
    
    ## from https://github.com/1adrianb/face-alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    faces_bucket  = None
    shift = 0.1

    with  torch.no_grad():
        while True:
            
            ret, orig_image = cap.read()

            scale_percent = 40 # percent of original size
            width = int(orig_image.shape[1] * scale_percent / 100)
            height = int(orig_image.shape[0] * scale_percent / 100)
            dim = (width, height)
            
            # resize image
            #print(orig_image.shape)
            orig_image = cv2.resize(orig_image, dim, interpolation = cv2.INTER_AREA)
            #print(orig_image.shape)

            shift_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * shift  # float `width`
            shift_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * shift # float `height`

            #print(orig_image)
            if orig_image is None:
                cap = cv2.VideoCapture(s)
                continue
            
            ld_l = fa.get_landmarks(orig_image)
            colors=  [(0,255,0), (0,255,255)]
            if faces_bucket is None:
                faces_bucket = [[k[:,0].mean(),k[:,1].mean()] for k in ld_l]
                print(faces_bucket)
            elif len(faces_bucket) < len(ld_l):
                print("aggiungimi!")
                
            for i in range(len(ld_l)):
                ld = ld_l[i]
                max_dist = max(shift_width, shift_height)
                c_index = 0 
                
                for k in range(len(faces_bucket)):
                    fb = np.array(faces_bucket[k])
                    dist = np.linalg.norm([ld[:,0].mean(),ld[:,1].mean()]-np.array(fb))
                    #dist = math.dist([ld[:,0].mean(),ld[:,1].mean()],fb )
                    if dist < max_dist:
                        max_dist = dist
                        c_index = k
                
                ## update of the center face works as expected
                faces_bucket[c_index] = [ld[:,0].mean(),ld[:,1].mean()]
                ant = 0
                
                if c_index ==0:
                    ant = 1
                
                # x works as expected in this way the right one is not considered 
                # the other way around the left one is not considered
                if  faces_bucket[c_index][0] > faces_bucket[ant][0]:
                    continue 
                image_annot = drawLandmark_multiple(orig_image,ld,colors[c_index])
                #image_annot = drawgraph_connection(image_annot, ld, from_ld, to_ld)
            

            #plt.plot(x_values, y_values, color="green");
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

                #out  = model(A_hat,norm_ld.unsqueeze(0).to(device))
                #_, predicted = out.max(1)
                # print(f"pred : {label[predicted]}")
                # if plot:
                #     data = out.squeeze(0).cpu().numpy()
                #     for i in range(len(rects)):
                #         rects[i].set_height(data[i])
                #     fig.canvas.draw()
                #     plt.pause(0.00001)
            
            #cv2.imshow('annotated', image_annot)
            

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