import torch
import numpy as np
import pandas as pd 
import face_alignment
from argparse import ArgumentParser
import os
import cv2
from matplotlib import pyplot as plt
#from tqdm.notebook import tqdm
from tqdm import tqdm
from moviepy.editor import AudioFileClip
import librosa
import librosa.display
import pickle

def find_faces(ld_l, faces_bucket, shift_width, shift_height, left_face, twod=True):
    # update the face bucket
    for ld in ld_l:
        max_dist = max(shift_width, shift_height)
        c_index = None 

        for k in range(len(faces_bucket)):
            fb = np.array(faces_bucket[k]["center"])
            if twod:
                dist = np.linalg.norm([ld[:,0].mean(),ld[:,1].mean()]-np.array(fb))
            else:
                dist = np.linalg.norm([ld[:,0].mean(),ld[:,1].mean(),ld[:,2].mean()]-np.array(fb))
            #dist = math.dist([ld[:,0].mean(),ld[:,1].mean()],fb )
            if dist < max_dist:
                max_dist = dist
                c_index = k
                
        # update the number of frames of that face
        if c_index is not None:
            if twod:
                faces_bucket[c_index]["center"] = [ld[:,0].mean(),ld[:,1].mean()]
            else:
                faces_bucket[c_index]["center"] = [ld[:,0].mean(),ld[:,1].mean(),ld[:,2].mean()]
                
            faces_bucket[c_index]["frames"] += 1
            faces_bucket[c_index]["ld"] = ld
    
        # means the face was not there so insert in the bucket
        if c_index is None:
            if twod:
                faces_bucket.append({"center" : [ld[:,0].mean(),ld[:,1].mean()], "ld":ld, "frames" :1})
            else:
                faces_bucket.append({"center" : [ld[:,0].mean(),ld[:,1].mean(), ld[:,2].mean()], "ld":ld, "frames" :1})
                
    # loop over the face bucket and extract the left or right more reliable aka more frames
    face_selected = None
    for face in faces_bucket:
        try:
            if face_selected is None:
                face_selected = face
            # if we are considering the left
            elif left_face and face["center"][0] < face_selected["center"][0] and face["frames"] >= face_selected["frames"]*0.6:
                face_selected = face
            # if we are considerign the right
            elif not left_face and face["center"][0] > face_selected["center"][0] and face["frames"] >= face_selected["frames"]*0.6:
                face_selected = face
        except:
            print(face)
            print("------------")
            print(face_selected)


    return face_selected["ld"], faces_bucket
def main(args):

    #path_to_train_data = os.path.join("/home/riccardo/Datasets/AffWild2/phoebe/dk15/new_aff_wild/Aff-Wild2_ready/Expression_Set/videos/", args.split)
    #path_to_train_annot = os.path.join("/home/riccardo/Datasets/AffWild2/phoebe/dk15/new_aff_wild/Aff-Wild2_ready/Expression_Set/annotations/", args.split)
    path_to_train_data = os.path.join("/data1/rfranceschini/AffWild2/phoebe/dk15/new_aff_wild/Aff-Wild2_ready/Expression_Set","videos",args.split)
    path_to_train_annot = os.path.join("/data1/rfranceschini/AffWild2/phoebe/dk15/new_aff_wild/Aff-Wild2_ready/Expression_Set","annotations",args.split)
    print(path_to_train_data)
    audio_files_tmp =  "/data1/rfranceschini/"
    out_folder = args.output
    device = args.device
    twod = True

    if twod:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    else:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device=device)

    ## annotation extraction
    ####{0,1,2,3,4,5,6}. {Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise}.
    annots = []

    for root, direct, files in tqdm(os.walk(path_to_train_annot, topdown=False)):
        for f in files:
            ann = {"file":f, "data":[]}
            data = open(os.path.join(root,f), "r")
            for idx, v in enumerate(data):
                if idx != 0:
                    ann["data"].append(int(v))
            annots.append(ann)


    shift = 0.1 # max percentage of face movement between two frame to be considered consistent
    ## loop over the files
    for root, direct, files in tqdm(os.walk(path_to_train_data, topdown=False)):

        for ann in annots:
            double_flag = False
            left_face = False
            v_file = None

            for f in files:
                if ann["file"].split(".")[0] == f.split(".")[0]:
                    v_file = f
                    break
                if ann["file"].split(".")[0].split("_")[0] == f.split(".")[0]:
                    v_file = f
                    # means there are two faces in the image
                    double_flag = True
                    if ann["file"].split(".")[0].split("_")[1] == "left":
                        left_face = True
                    print(f"Spastico {ann['file']} {v_file} left {left_face}")
            
            data_annot = {"file": ann['file'], "data": []}
            if  v_file is None:
                print(ann['file'])
                continue
            
            
            cap = cv2.VideoCapture(os.path.join(root,v_file))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            shift_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH) * shift  # float `width`
            shift_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * shift # float `height`
            

                
            ## extract audio and convert to mel spectogram
            my_audio_clip = AudioFileClip(os.path.join(root,v_file))
            my_audio_clip.write_audiofile(os.path.join(audio_files_tmp,"tm_audio.wav"), logger=None)
            data, sampling_rate = librosa.load(os.path.join(audio_files_tmp,"tm_audio.wav"))
            hop_length = int(len(data)/len(ann["data"]))
            mel_spect = np.array(librosa.feature.melspectrogram(y=data, sr=sampling_rate, hop_length=hop_length, n_mels=128))
            mel_spect = mel_spect[:,:len(ann["data"])].T

            ##### extract landmark
            f_count = 0
            i=1
            faces_bucket = None
            pbar = tqdm(total = frame_count)
            if os.path.isfile(os.path.join(out_folder,ann['file'].split('.')[0]+'.pkl')):
                print (f"exist : {os.path.join(out_folder,ann['file'].split('.')[0]+'.pkl')}")
            else:     
                while True:
                    pbar.update(i)
                    ret, frame = cap.read()
                    
                    if frame is None:
                        break
                    # resize the image if is too big
                    if cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 720:

                        scale_percent = 60 # percent of original size
                        if cap.get(cv2.CAP_PROP_FRAME_WIDTH) >= 1080:
                            scale_percent = 40 # percent of original size
                
                        width = int(frame.shape[1] * scale_percent / 100)
                        height = int(frame.shape[0] * scale_percent / 100)
                        dim = (width, height)
                        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                    f_count = min(f_count, len(ann["data"])-1 )
                    if ann["data"][f_count] != -1:
                        ld_l = fa.get_landmarks(frame)
                        # if no faces are detected put to everything to zero and but save the audio
                        if ld_l is None:
                            if twod:
                                ld_l = [np.zeros((68,2))] # if 3D 3 instead of two
                            else:
                                ld_l = [np.zeros((68,3))] # if 3D 3 instead of two
                                
                        if len(ld_l) == 1 and not double_flag:
                            data_annot["data"] = data_annot["data"] + [{"label":ann["data"][f_count], "ld":ld_l[0], "mel_spect": mel_spect[f_count]}]
                        if double_flag:
                            # means that there are two face so create the face bucket
                            if faces_bucket is None:
                                if twod:
                                    faces_bucket = [ {"center" : [ld[:,0].mean(),ld[:,1].mean()], "ld":ld, "frames" :1} for ld in ld_l ]
                                else:
                                    faces_bucket = [ {"center" : [ld[:,0].mean(),ld[:,1].mean(),ld[:,2].mean()], "ld":ld, "frames" :1} for ld in ld_l ]
                                    
                            # find the correct face checking the consistency in the bucket  
                            ld, faces_bucket = find_faces(ld_l,faces_bucket,shift_width,shift_height,left_face, twod=twod)
                            data_annot["data"] = data_annot["data"] + [{"label":ann["data"][f_count], "ld":ld, "mel_spect": mel_spect[f_count]}]
                            
                            
                    f_count +=1
                    # if f_count > 5:
                    #     break
                        
                #print(len(ld_l))
                with open(os.path.join(out_folder,ann["file"].split(".")[0]+".pkl"), 'wb') as fl:
                    print(f"save {os.path.join(out_folder,ann['file'].split('.')[0]+'.pkl')}")
                    pickle.dump(data_annot, fl)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--device', default='cuda:2', type=str, help='device')
    parser.add_argument('--split', default=None,  help='Train_set|Test_set')
    parser.add_argument('--output', default=None,  help='folder where to store the ckp')
    
    args = parser.parse_args()

    main(args)