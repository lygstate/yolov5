import argparse
from pathlib import Path
import os

import json
import cv2
import torch
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, \
    strip_optimizer, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device


def detect():
    out, source, weights, imgsz = opt.output, opt.source, opt.weights, opt.img_size
    if not os.path.exists(out):
        os.makedirs(out)

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # 加载四张图片
    dataset = LoadImages(source, img_size=imgsz)

    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    
    #****************************************************************
    four_pic=[]#存放4张图的结果
    pic_index = 0
    #****************************************************************
    for p, img, im0s, _ in dataset:  #dataset里保存四个图像
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        items = []
        # Process detections
        im0 = im0s
        for _, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):                                  
                    xyxy = torch.tensor(xyxy).numpy()
                    conf = conf.cpu().numpy().item()
                    cls = int(cls.cpu().numpy().item())
                    items.append({
                        'x0': int(xyxy[0]),
                        'y0': int(xyxy[1]),
                        'x1': int(xyxy[2]),
                        'y1': int(xyxy[3]),
                        'conf': conf,
                        'label': names[cls],
                        'cls': cls
                    })
        print('\n')
        for item in items:
            cls = item['cls']
            conf = item['conf']
            label = item['label']
            color=colors[cls]
            xyxy = [item['x0'], item['y0'], item['x1'], item['y1']]
            #if False:
            if cls > 35 and conf > 0.6:
                cropped_image = im0[item['y0']-2:item['y1']+2, item['x0']-2:item['x1']+2]
                crop_p = f"{out}/{pic_index}_{item['x0']}_{item['y0']}_{item['x1']}_{item['y1']}_{cls}_{label}.bmp"
                # cv2.imwrite(crop_p, cropped_image)
            plot_one_box(xyxy, im0, label=label, color=color, line_thickness=1)
        four_pic.append(items)
        pic_index +=1
        # cv2.imshow(p, im0)
        name = Path(p).name
        cv2.imwrite(f'{out}/{name}', im0)

    with open(f'{out}/detected.json', 'w') as f:
        json.dump(four_pic, f, indent=2)
    cv2.waitKey()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_name = 'VID_20210717_real'
    parser.add_argument('--weights', nargs='+', type=str, default='work_dirs/1280_bs32/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--output', type=str, default=f'./output-data/{data_name}', help='output folder')
    parser.add_argument('--source', type=str, default=f'./source-data/{data_name}', help='source')  # file/folder, 0 for webcam
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

