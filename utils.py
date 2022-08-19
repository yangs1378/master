import torch
import numpy as np
import os

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
def xlylwh2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] + x[:, 2] / 2  # x center
    y[:, 1] = x[:, 1] + x[:, 3] / 2  # y center
    y[:, 2] = x[:, 2]  # width
    y[:, 3] = x[:, 3]  # height
    return y
def file_get(file_path):
    if not os.path.exists(file_path):
        if not os.path.exists(file_path.split('/')[0]):
            os.mkdir(file_path.split('/')[0])

    del_file(file_path)


def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        file_data = path_data + "\\" + i  # 当前文件夹的下面的所有东西的绝对路径
        if os.path.isfile(file_data) == True:  # os.path.isfile判断是否为文件,如果是文件,就删除.如果是文件夹.递归给del_file.
            os.remove(file_data)
        else:
            del_file(file_data)

def get_xywh(video_name, frame_width, frame_height):
    try:
        if video_name == 'QGY012 20210704 143300 20210704 170640':
            xywh = [0.344531, 0.58741, 0.154687, 0.212963]
        elif video_name == 'QGY041 20210713 131537 20210713 152000':
            xywh = [0.816406, 0.474537, 0.145313, 0.212037]
        elif video_name == 'QGY012 20210704 143300 20210704 170640':
            xywh = [0.340365, 0.563889, 0.148438, 0.22037]
        elif video_name == 'QGY021':
            xywh = [0.805469, 0.471296, 0.123438, 0.181481]
        elif video_name == 'QGY041':
            xywh = [0.824479, 0.500926, 0.134375, 0.17963]
        elif video_name == 'QGY081':
            xywh = [0.814479, 0.433926, 0.134375, 0.17963]
        elif video_name == 'QGY091':
            xywh = [0.300365, 0.563889, 0.148438, 0.22037]
        elif video_name == 'QGY101':
            xywh = [0.300365, 0.613889, 0.148438, 0.22037]
        xyxy_p = [(xywh[0] - xywh[2] / 2) * frame_width - 30,
                  (xywh[1] - xywh[3] / 2) * frame_height - 30,
                  (xywh[0] + xywh[2] / 2) * frame_width + 250,
                  (xywh[1] + xywh[3] / 2) * frame_height + 30
                  ]
    except:
        print('error : video_name have not get area')
    return xyxy_p

def is_in_poly(p, poly):
    """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
    px, py = p
    is_in = False
    for i, corner in enumerate(poly):
        next_i = i + 1 if i + 1 < len(poly) else 0
        x1, y1 = corner
        x2, y2 = poly[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # if point is on edge
                is_in = True
                break
            elif x > px:  # if point is on left-side of line
                is_in = not is_in
    return