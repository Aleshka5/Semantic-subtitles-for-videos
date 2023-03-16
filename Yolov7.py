#              Imports
#         -----------------
import time
import torch

import numpy as np

from utils.datasets import letterbox
from experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
#         -----------------

class Yolov7():
    def __init__(self, width):
        self.weights = 'yolov7-tiny.pt'
        self.device = select_device('')
        print(f'device: {self.device}')
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(width, s=self.stride)  # check img_size

    def get_dataset_from_frames(self, frames, indexes,yolo_koef):
        dataset = []
        # print(images_to_predict)
        # Предобработка фото для работы YOLOv7
        for im0s in frames[indexes, 0, ::yolo_koef, ::yolo_koef]:
            my_img = letterbox(im0s, self.imgsz, stride=self.stride)[0]
            my_img = my_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            my_img = np.ascontiguousarray(my_img)
            dataset.append([None, my_img, im0s, None])
        return dataset

    def predict(self, dataset, view_img=False):
        save_img = False
        save_txt = False
        COCO_LABELS_RUS = """человек, велосипед, автомобиль, мотоцикл, самолет, автобус, поезд, грузовик, лодка, светофор, пожарный_гидрант, знак_остановки,
                        парковочный_счетчик, скамейка, птица, кошка, собака, лошадь, овца, корова, слон, медведь, зебра, жираф, рюкзак, зонтик, сумочка, галстук,
                        чемодан, фрисби, лыжи, сноуборд, спортивный_мяч, воздушный_змей, бейсбольная_бита, бейсбольная_перчатка, скейтборд, доска_для_серфинга, теннисная_ракетка,
                        бутылка, бокал_для_вина, чашка, вилка, нож, ложка, миска, банан, яблоко, сэндвич, апельсин, брокколи, морковь, хот-дог, пицца, пончик,
                        торт, стул, диван, растение_в_горшке, кровать, обеденный_стол, туалет, телевизор, ноутбук, мышь, пульт_дистанционного_управления, клавиатура, сотовый_телефон, микроволновая_печь, духовка,
                        тостер, раковина, холодильник, книга, часы, ваза, ножницы, плюшевый_мишка, фен, зубная_щетка"""
        COCO_LABELS_RUS = [ii.strip() for ii in (COCO_LABELS_RUS.split(","))]

        # Initialize
        set_logging()
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(self.imgsz, s=stride)  # check img_size

        if half:
            self.model.half()  # to FP16

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        output = []
        for _, img, im0s, _ in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != 'cpu' and (
                    old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=False)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=False)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=True)
            t3 = time_synchronized()
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                s, im0 = '', im0s

                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {COCO_LABELS_RUS[int(c)]},"  # add to string
                    output.append(s)
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if False else (cls, *xywh)  # label format

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0 / 255, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        # print(f'Done. ({time.time() - t0:.3f}s)')
        return output