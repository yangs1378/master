import copy

import cv2
# name_list = ['left_baffle', 'right_baffle', 'step']
# segment_list = [[380.6753246753247,545.3506493506494,365.7142857142857,540.3636363636364,359.06493506493507,497.14285714285717,359.06493506493507,437.2987012987013,359.06493506493507,392.4155844155844,400.62337662337666,257.76623376623377,515.3246753246754,71.5844155844156,515.3246753246754,101.50649350649351,422.2337662337663,375.7922077922078,410.5974025974026,468.88311688311694,402.28571428571433,512.1038961038961],
#            [546.909090909091,69.92207792207793,616.7272727272727,257.76623376623377,638.3376623376623,364.1558441558442,646.6493506493507,438.961038961039,644.987012987013,532.0519480519481,628.3636363636364,510.44155844155847,613.4025974025974,468.88311688311694,545.2467532467533,98.18181818181819],
#            [413.92207792207796,488.83116883116884,605.0909090909091,492.1558441558442,545.2467532467533,96.51948051948052,516.987012987013,98.18181818181819]]
# image = cv2.imread('shang2-00_00_03-2022_07_13_16_57_39.jpg')
# # for j, name in enumerate(name_list):
# for j, segment in enumerate(segment_list):
#     # if name == 'left_baffle':
#     if j == 0:
#         x_list = []
#         y_list = []
#         for i in range(int(len(segment[j])/2)):
#             # cv2.circle(image, (int(segment[j][2*i]), int(segment[j][2*i+1])), 1, (0, 0, 255), 4)
#             x_list.append(int(segment[j][2*i]))
#             y_list.append(int(segment[j][2*i+1]))
#         x_sort = copy.deepcopy(x_list)
#         y_sort = copy.deepcopy(y_list)
#         x_sort.sort(reverse=False)
#         y_sort.sort(reverse=False)
#         x_min = x_list[y_list.index(y_sort[0])]
#         fs_left_end = [int(x_list[y_list.index(y_sort[0])]), int(y_sort[0])]
#         fs_left_first = [int(x_list[y_list.index(y_sort[-1])]), int(y_sort[-1])]
#         k_max = -100
#         q_max = 0
#         for q, px in enumerate(x_list):
#             if (fs_left_end[0] - x_list[q]) != 0:
#                 k = (fs_left_end[1] - y_list[q]) / (fs_left_end[0] - x_list[q])
#                 if 0>k>k_max:
#                     k_max = k
#                     q_max = q
#         fs_left_middle = [int(x_list[q_max]), int(y_list[q_max])]
#         # cv2.circle(image, (int(fs_left_middle[0]), int(fs_left_middle[1])), 1,(255,255,255), 4)
#         # cv2.circle(image, (int(fs_left_end[0]), int(fs_left_end[1])), 1, (255,255,255), 4)
#         # cv2.circle(image, (int(fs_left_first[0]), int(fs_left_first[1])), 1, (255,255,255), 4)
#
#     # elif name == 'right_baffle':
#     elif j == 1:
#         x_list = []
#         y_list = []
#         for i in range(int(len(segment[j])/2)):
#             # cv2.circle(image, (int(segment[j][2*i]), int(segment[j][2*i+1])), 1, (0, 0, 255), 4)
#             x_list.append(int(segment[j][2*i]))
#             y_list.append(int(segment[j][2*i+1]))
#         x_sort = copy.deepcopy(x_list)
#         y_sort = copy.deepcopy(y_list)
#         x_sort.sort(reverse=False)
#         y_sort.sort(reverse=False)
#         fs_right_end = [int(x_list[y_list.index(y_sort[0])]), int(y_sort[0])]
#         fs_right_first = [int(x_list[y_list.index(y_sort[-1])]), int(y_sort[-1])]
#         k_min = 100
#         q_min = 0
#         for q, px in enumerate(x_list):
#             if (fs_right_end[0] - x_list[q]) != 0:
#                 k = (fs_right_end[1] - y_list[q]) / (fs_right_end[0] - x_list[q])
#                 if 0 < k < k_min:
#                     k_min = k
#                     q_min = q
#         fs_right_middle = [int(x_list[q_min]), int(y_list[q_min])]
#         # cv2.circle(image, (int(fs_right_middle[0]), int(fs_right_middle[1])), 1, (255,255,255), 4)
#         # cv2.circle(image, (int(fs_right_first[0]), int(fs_right_first[1])), 1, (255,255,255), 4)
#         # cv2.circle(image, (int(fs_right_end[0]), int(fs_right_end[1])), 1, (255,255,255), 4)
#     # elif name == 'step':
#     elif j == 2:
#
#         x_list = []
#         y_list = []
#         for i in range(int(len(segment[j]) / 2)):
#             cv2.circle(image, (int(segment[j][2 * i]), int(segment[j][2 * i + 1])), 1, (0, 0, 255), 4)
#             x_list.append(int(segment[j][2 * i]))
#             y_list.append(int(segment[j][2 * i + 1]))
#         x_sort = copy.deepcopy(x_list)
#         y_sort = copy.deepcopy(y_list)
#         x_sort.sort(reverse=False)
#         y_sort.sort(reverse=False)
#         # y1 = int(y_list[x_list.index(x_sort[0])])
#         # y2 = int(y_list[x_list.index(x_sort[-1])])
#         # x1 = int(x_list[y_list.index(y_sort[0])])
#         # x2 = int(x_list[y_list.index(y_sort[-1])])
#         # roi = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
#         roi = [int((x_sort[0]+x_sort[-1])/2), int((y_sort[0]+y_sort[-1])/2), int(x_sort[-1]), int(y_sort[-1])]
#         cv2.rectangle(image, (roi[0], roi[1]), (roi[2], roi[3]), (255,255,255), 1, 8)
#
# cv2.imshow('f', image)
# cv2.waitKey(0)

v1 = [1,2,3,4]
v2 = v1.__iter__()
print(v2.__next__())
print(v2.__next__())
print(v2.__next__())
print(v2.__next__())

class IT:
    def __init__(self):
        self.num = 0
    def __iter__(self):
        return self
    def __next__(self):
        self.num += 1
        if self.num > 6:
            raise StopIteration
        return self.num
class FOO:
    def __iter__(self):
        return IT()
obj = IT()
obj_1 = FOO()
for item in obj_1:
    print(item)


class gener:
    def __iter__(self):
        yield 1
        yield 2

obj_2 = gener()
for item in obj_2:
    print(item)
