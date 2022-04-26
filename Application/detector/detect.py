# ---------------------------------------------
# Detect.py - 
# File to handle the instruction of "perform inferencing passed from GUI element"
# variables in: folder [string] (the folder containing the images to be inferenced), 
#               show_confidence [bool] to show confidence on the resulting labels,
#               conf_thres [float] to limit the confidence thresholding of generated detection boxes
#
# @Author - Chris Cranston W18018468
# ---------------------------------------------


import os
import sys
from pathlib import Path
# import torch
from torch import no_grad, from_numpy
import torch.backends.cudnn as cudnn
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
from models.common import DetectMultiBackend
from utils.datasets import  LoadImages
from utils.general import (check_img_size, cv2, non_max_suppression,  scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device




@no_grad()
# Run detector function used in GUI to inference images and return image and labels
def run_detector (folder, show_confidence, conf_thresh):
    # set default values
    weights=ROOT / 'input/weights/best.pt'
    source=folder
    data=ROOT / 'data/coco128.yaml'
    imgsz=(512, 512)
    device=''
    result = []
    haemorrhage_count = 0
    exudate_count = 0
    source = str(source)
    show_conf = show_confidence

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # load images for inferencing
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    ## Run inference
    # warmup
    model.warmup(imgsz=(1, 3, *imgsz))  
    # set location to save images
    save_dir = folder+"/results/"
    try: 
        os.stat(save_dir)
    except:
        os.mkdir(save_dir)

    # link image to device for each image
    for path, img, im0s, vid_cap, s in dataset:
        img = from_numpy(img).to(device)
        img = img.half() if model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference on each image
        pred = model(img, augment=False, visualize=False)

        # Non_max_suppression
        pred = non_max_suppression(pred, conf_thresh, 0.45, None, False, max_det=50)

        # Process predictions per image
        for det in pred:  
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            annotator = Annotator(im0, line_width=1, example=str(names))
            if len(det):

                # Rescale boxes 
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    conf_label = f'{names[c]} {conf:.2f}'
                    label = names[c]
                    annotator.box_label(xyxy, label if show_conf == False else conf_label, color=colors(c, True))
                    # increase count of appropriate detection
                    if (label == "Haemorrhage"):
                        haemorrhage_count += 1
                    if (label == "Exudate"):
                        exudate_count += 1

            # Stream results
            im0 = annotator.result()
            # save image to file
            cv2.imwrite(save_dir+p.name, im0)
            # add image path to result to be used in GUI
            result.append(str(save_dir+p.name))   
    
    return result, haemorrhage_count, exudate_count




