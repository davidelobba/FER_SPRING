"""
This code uses the onnx model to detect faces from live video or cameras.
Use a much faster face detector: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
Date: 3/26/2020 by Cunjian Chen (ccunjian@gmail.com)
"""
import time
import cv2
import numpy as np
import onnx
import vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend
import math

import onnxruntime as ort
import os 

# import libraries for landmark
from common.utils import BBox,drawLandmark,drawLandmark_multiple, rotate, rotate_by_nose
from PIL import Image
import torchvision.transforms as transforms

# setup the parameters
resize = transforms.Resize([56, 56])
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
# import the landmark detection models
import onnx
import onnxruntime
onnx_model_landmark = onnx.load("onnx/landmark_detection_56_se_external.onnx")
onnx.checker.check_model(onnx_model_landmark)
ort_session_landmark = onnxruntime.InferenceSession("onnx/landmark_detection_56_se_external.onnx")
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# face detection setting
def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


label_path = "models/voc-model-labels.txt"

onnx_path = "models/onnx/version-RFB-320.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

# perform face detection and alignment from camera
#cap = cv2.VideoCapture(2)  # capture from camera
#cap = cv2.VideoCapture("/home/riccardo/Downloads/CAER/test/Anger/0006.avi")
threshold = 0.7
sum = 0
###
source = "/home/riccardo/Datasets/RAVDESS/Test_set/01"
outputFolder = "/home/riccardo/CAER_clean/Test_set/01"
resize_size = 128
show = False
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)


for root, direct, files in os.walk(source, topdown=False):
    for name in direct:
        print(f"name {os.path.join(root,name)}")
        print(f"r {os.path.split(root)}")
        print(f" {os.path.split(root)[1]}")

        out_f = os.path.join(outputFolder,os.path.split(root)[1],name)
        if not os.path.exists(out_f):
            os.makedirs(out_f)
    
    for source in files:
        if source:
            cap = cv2.VideoCapture(os.path.join(root,source))
            outputFile = os.path.basename(source)[:-4] + ".avi"
        else:
            cap = cv2.VideoCapture(2, cv2.CAP_V4L)
            outputFile = "grabbed_from_camera.avi"

        out_f = os.path.join(outputFolder,os.path.split(root)[0].split("/")[-1], os.path.split(root)[1])

        faces = []
        diff_pix = 25
        while True:
            ret, orig_image = cap.read()
            if orig_image is None:
                break
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (320, 240))
            # image = cv2.resize(image, (640, 480))
            image_mean = np.array([127, 127, 127])
            image = (image - image_mean) / 128
            image = np.transpose(image, [2, 0, 1])
            image = np.expand_dims(image, axis=0)
            image = image.astype(np.float32)
            # confidences, boxes = predictor.run(image)
            time_time = time.time()
            confidences, boxes = ort_session.run(None, {input_name: image})
            #print("cost time:{}".format(time.time() - time_time))
            
            boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)
            distance = 0
            for i in range(boxes.shape[0]):
                box = boxes[i, :]
                label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
                #cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
                # perform landmark detection
                out_size = 56
                img=orig_image.copy()
                height,width,_=img.shape
                x1=box[0]
                y1=box[1]
                x2=box[2]
                y2=box[3]
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                size = int(max([w, h])*1.1)
                cx = x1 + w//2
                cy = y1 + h//2
                x1 = cx - size//2
                x2 = x1 + size
                y1 = cy - size//2
                y2 = y1 + size
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)   
                new_bbox = list(map(int, [x1, x2, y1, y2]))
                new_bbox = BBox(new_bbox)
                cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
                if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                    cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
                cropped_face = cv2.resize(cropped, (out_size, out_size))

                if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                    continue
                cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)    
                cropped_face = Image.fromarray(cropped_face)
                test_face = resize(cropped_face)
                test_face = to_tensor(test_face)
                test_face = normalize(test_face)
                test_face.unsqueeze_(0)

                start = time.time()             
                ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_face)}
                ort_outs = ort_session_landmark.run(None, ort_inputs)
                end = time.time()
                #print('Time: {:.6f}s.'.format(end - start))
                landmark = ort_outs[0]
                landmark = landmark.reshape(-1,2)

                # compute left and right eye center therefore face rotation angle
                r_eye =  np.average(landmark[36:42,:], axis=0)
                l_eye =  np.average(landmark[42:48,:], axis=0)
                c_nose = np.average(landmark[31:36,:], axis=0)
                alpha = 180.0 + math.degrees(math.atan2(r_eye[1]-l_eye[1], r_eye[0]-l_eye[0]))

                c_nose = new_bbox.reprojectLandmark(landmark[27:36,:])
                eyes = new_bbox.reprojectLandmark(np.array([r_eye,l_eye]))
                distance =np.linalg.norm(eyes[0]-eyes[1])
                gap_face = int(2.4*distance)

                face , _ = rotate_by_nose(img, new_bbox, c_nose, alpha, gap_face)

                ### check center and extract face
                find = False
                center = np.average(c_nose, axis=0)

                for f in faces:
                    if np.linalg.norm(f["center"]-center) < diff_pix:
                        find = True 
                        f["box_list"].append(face)
                        f["center"] = np.array([(f["center"][0]+center[0])/2, (f["center"][1]+center[1])/2])
                        break
                if not find:
                    faces.append({"center":center,"box_list":[face]})

                if show:
                    try:
                        cv2.imshow('crop', face)
                    except:
                        pass

                # # eye landmark
                # landmark = landmark[27:48,:]
                # #print(f"l_shape {landmark.shape} : r {r_eye} l {l_eye}") ##[37..42] left [43,...48] right

                landmark = new_bbox.reprojectLandmark(landmark)
                orig_image = drawLandmark_multiple(orig_image, new_bbox, landmark)
                orig_image = drawLandmark_multiple(orig_image, new_bbox, new_bbox.reprojectLandmark(np.array([r_eye,l_eye])))

                # vid_writer = cv2.VideoWriter(
                #     os.path.join(out_f, outputFile),
                #     cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                #     15,
                #     (128,128),
                #     )

            sum += boxes.shape[0]
            if show:
                orig_image = cv2.resize(orig_image, (0, 0), fx=0.7, fy=0.7)
                cv2.imshow('annotated', orig_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        max_len = 0
        max_index = 0 
        for i in range(len(faces)):
            if len(faces[i]["box_list"])> max_len:
                max_index = i
                max_len = len(faces[i]["box_list"]) 

        try:
            vid_writer = cv2.VideoWriter(
                    os.path.join(out_f, outputFile),
                    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                    15,
                    (resize_size,resize_size),
                    )
            print(os.path.join(out_f,outputFile))
            for roi in faces[max_index]["box_list"]:
                
                # if you want to crop but the result isn't better that simply resizing
                #x,y = np.array(roi).shape[:-1]
                #x,y = int(x/2), int(y/2)
                #roi = cv2.resize(roi[y-crop_min:y+crop_min,x-crop_min:x+crop_min], (128,128), interpolation = cv2.INTER_AREA)
                
                roi = cv2.resize(roi, (resize_size,resize_size), interpolation = cv2.INTER_AREA)
                if show:
                    cv2.imshow("crop", roi)
                    k = cv2.waitKey(5)
                    if k == 27:
                        break
                    time.sleep(0.05)
                vid_writer.write(roi)
        except:
            print("Failed")
            pass
        cap.release()
        cv2.destroyAllWindows()
        vid_writer.release()