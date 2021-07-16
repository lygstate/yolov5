import argparse
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    out, source, weights, view_img, save_txt, imgsz = opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    if not os.path.exists(out):
        os.makedirs(out)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    #****************************************************************
    four_pic=[]#存放4张图的结果
    #****************************************************************
    for path, img, im0s, vid_cap in dataset:  #dataset里保存四个图像
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            save_path = str(Path(out)/p.name)

            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                #****************************************************************
                # Write results
                #a-z 0-25; 0-9, 26-35; 36-37
                alphabet= []#保存检测到的字母
                num=[] #保存检测到的数字
                pillar = []#保存检测到的棱柱
                center_alphabet=[]#保存检测到的字母坐标中心
                center_num=[]#保存检测到的数字坐标中心
                center_pillar=[]#保存检测到的棱柱中心
                for *xyxy, conf, cls in reversed(det):                                  
                    xyxy = torch.tensor(xyxy).numpy()
                    center = [((xyxy[2]+xyxy[0])/2.0),((xyxy[3]+xyxy[1])/2.0)]
                    center =np.array(center)
                    if cls<26: 
                       alphabet.append(cls.cpu().numpy())
                       center_alphabet.append(center)
                       continue
                    if cls>25 and cls<36: 
                       num.append(cls.cpu().numpy())
                       center_num.append(center)
                       continue
                    if cls>35 and conf.cpu().numpy()>0.8: 
                       pillar.append(cls.cpu().numpy())
                       center_pillar.append(center)
                four_pic.append([num,alphabet,pillar,center_num,center_alphabet,center_pillar])
                #****************************************************************
    #将检测到的所有数字归属到自己最近的字母
    dict_= []
    for i in range(26):
        dict_.append([])

    for pic_num in range(4):
        pic =four_pic[pic_num]
        for num_i in range(len(pic[0])):
            temp = np.abs(pic[3][num_i]-pic[4])   
            temp =(temp.T)**2
            temp = temp[0,:]+temp[1,:]
            temp = temp.tolist()
            dict_[int(pic[1][temp.index(min(temp))])].append(pic[0][num_i])

    #判断每个字母属于三菱柱还是四棱柱
    dict_pil= []
    for i in range(26):
        dict_pil.append([])

    for pic_num in range(4):
        pic =four_pic[pic_num]
        for num_i in range(len(pic[2])):
            temp = np.abs(pic[5][num_i]-pic[4])   
            temp =(temp.T)**2
            temp = temp[0,:]+temp[1,:]
            temp = temp.tolist()
            dict_pil[int(pic[1][temp.index(min(temp))])].append(pic[2][num_i])

    #计算结果
    for i in range(len(dict_)):
        if len(dict_[i])>0:
            temp_num = np.unique(np.array(dict_[i]))
            temp_pil = np.unique(np.array(dict_pil[i]))
            if len(temp_pil)==1:
                if int(temp_pil) == 36 and len(temp_num) ==3:

                    result = np.sum(temp_num)
                    print('zimu {} : he :{}'.format(i,result))
                if int(temp_pil) == 37 and len(temp_num) ==4:
                    result = np.sum(temp_num)
                    print('zimu {} : he :{}'.format(i,result))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='work_dirs/1920_bs32_5s/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--output', type=str, default='./work_dirs/1920_bs32_5s-four-pic', help='output folder')
    parser.add_argument('--source', type=str, default='/share/caodong/code/yolov5-4.0/xiaoluo/all/four-pic/VID_20210712_094216', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', default=False,help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()

