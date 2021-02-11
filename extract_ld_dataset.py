import os 
import numpy as np 
from landmark.landmark_detection import extract_landmark
from tqdm import tqdm
import json
import cv2

source = "/home/riccardo/Datasets/CAER_crop/"
outputFolder = "/home/riccardo/Datasets/CAER_LD/"

for root, direct, files in os.walk(source, topdown=False):

    for name in direct:
        out_f = os.path.join(outputFolder,os.path.split(root)[1],name)
        if not os.path.exists(out_f):
            os.makedirs(out_f)
    
    for source in tqdm(files):
        
        if source:
            cap = cv2.VideoCapture(os.path.join(root,source))
            outputFile = os.path.basename(source)[:-4] + ".json"

        out_f = os.path.join(outputFolder,os.path.split(root)[0].split("/")[-1], os.path.split(root)[1], outputFile)

        features = {"video":source, "label":os.path.split(root)[-1],"landmark":[]}
        while True:
            ret, orig_image = cap.read()
            if orig_image is None:
                break
            try:
                _ , ld = extract_landmark(orig_image, image_back=False)
                if ld is not None:
                    features["landmark"].append(ld.tolist())
            except:
                print(f"failed : {features['video']}")
                pass

        if len(features["landmark"]) < 50 :
            print(f"short : {features['video']}")
            continue
        with open(out_f, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=0)
        