'''
QGY041.mp4
QGY041.mp4参数设置，行人上下，和行人属性判断
'''
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import copy
import time
import torch
from torch.backends import cudnn
import argparse
from deep_sort_pytorch.deepsort import DEEPSORT
import random
from copy import deepcopy
from yolov5.yolo_det import YOLO_DET
from yolov5.utils.general import scale_coords
from image_utils import img_write, draw_point, compute_color_for_id
from opticalflow_utils import OpticalFlow
from utils import xyxy2xywh
from utils import get_xywh
from yolov5_trt.python_trt import Detector


class Detect:
    def __init__(self):
        super(Detect, self).__init__()
        # deepsort模型加载
        deepsort = DEEPSORT()
        self.deepsort = deepsort.load_deepsort()

        self.area_set = 350

    def track_img(self, img_0, img, pred):
        self.img0 = deepcopy(img_0)  # 原图
        self.output_deepsort = []
        track_id_list = []

        # 开始推理
        with torch.no_grad():
            my_im0 = deepcopy(self.img0)  # 跟踪用的帧
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    xyxys = scale_coords(img.shape[2:], det[:, :4], img_0.shape).round()
                    xywhs = xyxy2xywh(xyxys)
                    confs = det[:, 4]
                    clss = det[:, 5]
                    in_plot_idx = []
                    for j, box in enumerate(xywhs):  # 遍历每个跟踪到的行人
                        area = int(box[2] * box[3])
                        xcen = int(box[0])
                        ycen = int(box[1])
                        if clss[j] == 0:
                            in_plot_idx.append(j)
                    if len(in_plot_idx) > 0:
                        people_clss = torch.stack(
                            [clss[in_plot_idx[a]] for a in range(len(in_plot_idx))], 0)
                        people_xywhs = torch.stack(
                            [xywhs[in_plot_idx[a]] for a in range(len(in_plot_idx))], 0)
                        people_confs = torch.stack(
                            [confs[in_plot_idx[a]] for a in range(len(in_plot_idx))], 0)
                    else:
                        people_clss = torch.tensor([])
                        people_xywhs = torch.tensor([])
                        people_confs = torch.tensor([])
                        torch.tensor([])
                    if len(people_xywhs) > 0:
                        output_deepsort = self.deepsort.update(people_xywhs.cpu(), people_confs.cpu(), people_clss,
                                                               my_im0)  # 输入行人进行跟踪

                        for people in output_deepsort:
                            track_id_list.append(people[4])
                        if len(output_deepsort) > 0:  # 如果跟踪到行人
                            self.output_deepsort.append(output_deepsort)

            return track_id_list, self.output_deepsort  # [bboxes(xyxy), id, clss]


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-c', '--compound_coef', type=int, default=4, help='coefficients of efficientdet')
    parser.add_argument('-g', '--gpu_device', type=str, default='0', help="gpu id")
    parser.add_argument('-v', '--video', type=str, default='E:\study_file\project\\futi\lbt\\shang2.mp4',
                        help="video name(without suffix)")
    parser.add_argument('-s', '--save_name', type=str, default='shang2', help="video name(without suffix)")
    parser.add_argument('-w', '--weight', type=str, default='yolov5/yolov5s.pt', help="weight path")
    args = parser.parse_args()
    return args


def get_sort_result(sort_result_list, id):
    id_clss_list = [3] * 10
    for i, sort_result in enumerate(sort_result_list):
        for j, person_id in enumerate(sort_result[0]):
            clss = sort_result[1][0][j][5]
            if id == person_id:
                id_clss_list[i] = clss
    return id_clss_list


if __name__ == "__main__":
    opt = get_args()
    obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                'teddy bear',
                'hair drier', 'toothbrush']  # class names
    # Video's path
    video_dir = opt.video
    # Video capture
    cap = cv2.VideoCapture(video_dir)
    fgo = 0  # start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, fgo)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cap.read()
    video_name = opt.video.split('/')[-1].split('\\')[-1].split('.')[0].split('_')[0]

    len_sort = 2*fps
    len_sort_min = int(len_sort / 3 * 2)

    # attributes new video path
    pwd = os.getcwd()
    print(pwd)
    new_video_path = os.path.join(pwd, 'test/')
    new_video_name = opt.save_name + '.avi'
    new_video_dir = new_video_path + new_video_name
    print('new_video_dir{}'.format(new_video_dir))
    if not os.path.exists(new_video_path):
        print('create video_dir ')
        os.mkdir(new_video_path)
    else:
        print('video_dir exists')

    videoWriter = cv2.VideoWriter(new_video_dir, cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25,
                                  (frame_width, frame_height))
    count_frame = -1
    # ********************************************************************************************
    if os.path.exists('box_list.txt'):
        os.remove('box_list.txt')
    sort_result_list = []

    # ********************************************************************************************

    # optical initial
    person_label = ''

    # load model
    weight = opt.weight
    weight_type = weight.split('.')[1]
    dect_model = YOLO_DET(weight)

    deepsort = Detect()
    sort_output = None

    while (1):
        t_start = time.time()
        count_frame += 1
        ret, img = cap.read()

        if not ret:
            break

        # frame preprocessing
        frameperson = copy.deepcopy(img)
        frame_trans = dect_model.img_trans(frameperson)

        # model predict
        with torch.no_grad():
            pred = dect_model.detect_img(frameperson)
        sort_output = deepsort.track_img(frameperson, frame_trans, pred)  # trackid_list, output([xyxys], id, clss)

        if sort_output is not None:
            if len(sort_output[0]) > 0:
                sort_result_list.append(sort_output)
                if len(sort_result_list) > len_sort:
                    del sort_result_list[0]
                id_lists = []
                for p, sort_result in enumerate(sort_result_list):
                    bboxes = sort_result[1][0][:, :4]
                    id_list = sort_result[1][0][:, 4]
                    for id in id_list:
                        if id not in id_lists:
                            id_lists.append(id)
                            id_lists.sort()

                with open('box_list.txt', 'a') as f:
                    f.write('%s' % (count_frame) + '\n')
                    for id in id_lists:
                        box_list = []
                        area_list = []
                        l_y_list = []
                        gap_list = []
                        gap_percent_list = []
                        l_y, l_y_bf = 0, 0
                        flag = 0
                        for p, sort_result in enumerate(sort_result_list):
                            bboxes = sort_result[1][0][:, :4]
                            id_list = sort_result[1][0][:, 4]
                            if id in id_list:
                                num = np.where(id_list == id)[0][0]
                                h = (bboxes[num][1] + bboxes[num][3])/2
                                box_list.append(bboxes[num])
                                l_y = bboxes[num][1] if bboxes[num][1] != 0 else 1

                                if len(gap_list) > 1:
                                    gap = round((l_y - l_y_bf)/h, 3)
                                    if gap > 0.0025:
                                        gap_percent = 1
                                    elif gap < -0.0025:
                                        gap_percent = -1
                                    else:
                                        gap_percent = 0
                                else:
                                    gap = 0
                                    gap_percent = 0
                                l_y_bf = bboxes[num][1]
                                gap_list.append(gap)
                                area = (bboxes[num][2] - bboxes[num][0]) * (bboxes[num][3] - bboxes[num][1])
                                area_list.append(area)
                                l_y_list.append(bboxes[num][1])
                                gap_percent_list.append(gap_percent)
                        if len(box_list) > len_sort_min:
                            # 进入判断逻辑, l_y and area 共同判断
                            if gap_percent_list.count(1) > len(gap_percent_list) / 2 or gap_percent_list.count(
                                    -1) > len(gap_percent_list) / 2 or abs(sum(gap_percent_list)/len(gap_percent_list)) > 0.025:
                                flag = 0  # 正常行走
                            elif gap_percent_list.count(0) > 2 * len(gap_percent_list) / 3:
                                flag = 1  # 停留
                            elif gap_percent_list.count(1) > len(gap_percent_list) / 3 and gap_percent_list.count(
                                    -1) > len(gap_percent_list) / 3:
                                flag = 2  # 逆行
                        f.write(f"{id}" + ' ' + f"{box_list}" + ' ' + '\n')
                        f.write(f'{area_list}' + '\n')
                        f.write(f'{l_y_list}' + '\n')
                        f.write(f'{gap_list}' + '\n')
                        flag_list = ['normal', 'stay', 'retrograde']
                        f.write(f'{flag}' + f'{flag_list[flag]}' + '\n')
                        color = compute_color_for_id(id)
                        cv2.putText(frameperson, str(id), (10, 25+20*id),cv2.FONT_HERSHEY_PLAIN, 2.0, color, 2, 8)
                        cv2.putText(frameperson, flag_list[flag], (30, 25+20*id),cv2.FONT_HERSHEY_PLAIN, 2.0, color, 2, 8)

        if sort_output is not None:
            if len(sort_output[0]) > 0:
                bboxes = sort_output[1][0][:, :4]
                id = sort_output[1][0][:, 4]
                clss = sort_output[1][0][:, 5]
                for q, box in enumerate(bboxes):
                    clss_name = obj_list[clss[q]]
                    color = compute_color_for_id(id[q])
                    cv2.rectangle(frameperson, (box[0], box[1]), (box[2], box[3]), color, 2, 8)
                    cv2.putText(frameperson, str(id[q]) + ' ' + clss_name, ((box[0] + box[2]) // 2, box[1]),
                                cv2.FONT_HERSHEY_PLAIN, 2.0,
                                color, 2, 8)

        # 显示图像
        cv2.imshow("Airport pedestrian features analysis demo", frameperson)
        c = cv2.waitKey(1)
        if c == 27:
            break
        videoWriter.write(frameperson)
        t_end = time.time()
        print("It cost %s s" % (t_end - t_start))
cap.release()
cv2.destroyAllWindows()
