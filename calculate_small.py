import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler

import json
import socketserver
from typing import Tuple
import cv2
import torch
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device

class Detector(object):
    def __init__(self, opt):
        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        weights, imgsz = opt.weights, opt.img_size
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        # Run inference
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once    
        self.imgsz = imgsz
        self.model = model
        self.device = device
        self.half = half
        self.names = names
        self.colors = colors
        print("Init finished")

    def loadImage(self, buf):
        nparr = np.frombuffer(buf, np.uint8)
        img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Padded resize
        img = letterbox(img0, new_shape=self.imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img, img0

    def detect(self, buf):
        device = self.device
        half = self.half
        model = self.model

        img, img0 = self.loadImage(buf)
        # print(f'Handing {img0.shape}')
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
        for _, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
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
                        'label': self.names[cls],
                        'cls': cls
                    })
        return items

class Resquest(BaseHTTPRequestHandler):
    def __init__(self, request: bytes, client_address: Tuple[str, int], server: socketserver.BaseServer) -> None:
        super().__init__(request, client_address, server)

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length'))
        post_body = self.rfile.read(content_len)
        items = detector.detect(post_body)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(items).encode())

host = ('0.0.0.0', 8887)
detector = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    data_name = 'VID_20210717_real'
    parser.add_argument('--weights', nargs='+', type=str, default='drone-models/detection-small/best-small.pt', help='model.pt path(s)')
    parser.add_argument('--output', type=str, default=f'./output-data/{data_name}', help='output folder')
    parser.add_argument('--source', type=str, default=f'./source-data/{data_name}', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=224, help='inference size (pixels)')
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
    detector = Detector(opt)

    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()
