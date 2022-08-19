'''
QGY041.mp4
QGY041.mp4参数设置，行人上下，和行人属性判断
'''
import numpy as np
import copy
import torch
import sys

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
def get_seg(segment_list):
    fs_left_first, fs_left_end, fs_left_middle = [0, 0], [0, 0], [0, 0]
    fs_right_first, fs_right_end, fs_right_middle = [0, 0], [0, 0], [0, 0]
    roi = [0, 0, 0, 0]
    for j, segment in enumerate(segment_list):
        x_list = []
        y_list = []
        for i in range(int(len(segment) / 2)):
            x_list.append(int(segment[2 * i]))
            y_list.append(int(segment[2 * i + 1]))
        x_sort = copy.deepcopy(x_list)
        y_sort = copy.deepcopy(y_list)
        x_sort.sort(reverse=False)
        y_sort.sort(reverse=False)
        if j == 0:
            x_min = x_list[y_list.index(y_sort[0])]
            fs_left_end = [int(x_list[y_list.index(y_sort[0])]), int(y_sort[0])]
            fs_left_first = [int(x_list[y_list.index(y_sort[-1])]), int(y_sort[-1])]
            k_max = -100
            q_max = 0
            for q, px in enumerate(x_list):
                if (fs_left_end[0] - x_list[q]) != 0:
                    k = (fs_left_end[1] - y_list[q]) / (fs_left_end[0] - x_list[q])
                    if 0 > k > k_max:
                        k_max = k
                        q_max = q
            fs_left_middle = [int(x_list[q_max]), int(y_list[q_max])]
        elif j == 1:
            fs_right_end = [int(x_list[y_list.index(y_sort[0])]), int(y_sort[0])]
            fs_right_first = [int(x_list[y_list.index(y_sort[-1])]), int(y_sort[-1])]
            k_min = 100
            q_min = 0
            for q, px in enumerate(x_list):
                if (fs_right_end[0] - x_list[q]) != 0:
                    k = (fs_right_end[1] - y_list[q]) / (fs_right_end[0] - x_list[q])
                    if 0 < k < k_min:
                        k_min = k
                        q_min = q
            fs_right_middle = [int(x_list[q_min]), int(y_list[q_min])]
        elif j == 2:
            roi = [int((x_sort[0] + x_sort[-1]) / 2), int((y_sort[0] + y_sort[-1]) / 2), int(x_sort[-1]),
                   int(y_sort[-1])]
    return [fs_left_end, fs_left_middle, fs_left_first], [fs_right_end, fs_right_middle,fs_right_first], roi


class judgement:
    def __init__(self, segment_list):  # 输入实例分割结果，即扶梯分割结果
        self.sort_result_list = []
        self.dict_state = {}
        self.state = ['normal', 'stay', 'return', 'crowded']
        self.count = 0
        self.seg = get_seg(segment_list)

    def judge(self, sort_output, len_sort, direction=-1):
        # 输入跟踪结果，最大跟踪时间范围，跟踪的人所处扶梯方向
        # 输出跟踪id列表以及相应的状态列表
        seg = self.seg
        left_fs = seg[0]
        right_fs = seg[1]
        len_sort_min = int(len_sort / 3 * 2)
        # fs = [self.fs[0], self.fs[1]]
        self.count += 1
        state_list = []
        id_lists = []
        if sort_output is not None:
            if len(sort_output[0]) > 0:
                self.sort_result_list.append(sort_output)
                if len(self.sort_result_list) > len_sort:
                    del self.sort_result_list[0]

                for p, sort_result in enumerate(self.sort_result_list):
                    bboxes = sort_result[1][0][:, :4]
                    xywhs = xyxy2xywh(bboxes)
                    id_list = sort_result[1][0][:, 4]
                    for q, xywh in enumerate(xywhs):
                        if left_fs[-1][1] > xywh[1] > (left_fs[0][1]+50):
                            if (left_fs[-1][0] - left_fs[0][0]) != 0:
                                if ((left_fs[1][1] - left_fs[0][1]) / (left_fs[1][0] - left_fs[0][0])) * (
                                        (left_fs[1][1] - left_fs[0][1]) / (left_fs[1][0] - left_fs[0][0]) * xywh[0] + (
                                        left_fs[1][0] * left_fs[0][1] - left_fs[0][0] * left_fs[1][1]) / (
                                                left_fs[1][0] - left_fs[0][0]) - xywh[1]) > 0 \
                                        and ((right_fs[1][1] - right_fs[0][1]) / (right_fs[1][0] - right_fs[0][0])) * (
                                        (right_fs[1][1] - right_fs[0][1]) / (right_fs[1][0] - right_fs[0][0]) * xywh[
                                    0] + (right_fs[1][0] * right_fs[0][1] - right_fs[0][0] * right_fs[1][1]) / (
                                                right_fs[1][0] - right_fs[0][0]) - xywh[1]) < 0:
                                    if id_list[q] not in id_lists:
                                        id_lists.append(id_list[q])
                                        id_lists.sort()

                for x, id in enumerate(id_lists):
                    gap_list = []
                    gap_instant_1, gap_instant_2, gap_instant_3 = 0, 0, 0
                    l_y, l_y_bf = 0, 0
                    state = 0
                    for p, sort_result in enumerate(self.sort_result_list):
                        bboxes = sort_result[1][0][:, :4]
                        id_list = sort_result[1][0][:, 4]
                        if id in id_list:
                            num = np.where(id_list == id)[0][0]
                            h = (bboxes[num][1] + bboxes[num][3]) / 2
                            l_y = bboxes[num][1] if bboxes[num][1] != 0 else 1

                            if len(gap_list) > 1:
                                gap = round((l_y - l_y_bf) / h, 5)


                                if gap > 0.0025:
                                    gap_instant_1 += 1
                                elif gap < -0.0025:
                                    gap_instant_2 += 1
                                else:
                                    gap_instant_3 += 1
                            else:
                                gap = 0
                                gap_instant_3 += 1
                            gap_list.append(gap)
                            l_y_bf = l_y
                    # V1
                    if len(gap_list) > len_sort_min:
                        # 进入判断逻辑, l_y判断
                        if gap_instant_1 > len(gap_list) / 2 or gap_instant_2 > len(gap_list) / 2 or abs(
                                sum(gap_list) / len(gap_list)) > 0.00025:
                            state = 0  # 正常行走
                        elif gap_instant_3 > 3 * len(gap_list) / 4:
                            state = 1  # 停留

                        if direction == -1 and gap_instant_1 > len(gap_list) / 3:
                            state = 2  # 逆行
                        elif direction == 1 and gap_instant_2 > len(gap_list) / 3:
                            state = 2  # 逆行
                    state_list.append(state)
        return id_lists, state_list
