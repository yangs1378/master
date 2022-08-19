import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def compute_color_for_id(label):
    """
    Simple function that adds fixed color depending on the id
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_point(frame,all_node):
    width = frame.shape[1]
    height = frame.shape[0]
    position = [width - 100, 10]
    frame = cv2ImgAddText(frame, '清洁上下机：', position[0], position[1], (0, 245, 188), textSize=15)
    frame = cv2ImgAddText(frame, '机组上下机：', position[0], position[1]+15, (0, 245, 188), textSize=15)


    img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_PIL)
    position[0] = width - 30
    #对于每个节点状态，分别画个圆点
    count = 0
    for i in range(2):  # 上清洁，上机组，下清洁，下机组
        for j in range(2):
            draw.text((position[0]+10, position[1]), str(count), fill=(0, 255, 0))
            if all_node[count] == 1:
                draw.ellipse(((position[0]+10, position[1]), (position[0]+20, position[1]+10)), fill=(0, 255, 0), outline=(0, 255, 0), width=2)
            else:
                draw.ellipse(((position[0]+10, position[1]), (position[0]+20, position[1]+10)), fill=(255, 0, 0), outline=(255, 0, 0), width=2)
            position[1] += 15
            count += 1
        position[0] = width - 20
        position[1] = 10

    frame = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR)

    return frame

def cv2ImgAddText(img, text, x, y, textColor, textSize=2):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "./Fonts/HGY4_CNKI.TTF", textSize, encoding="utf-8")
    draw.text((x, y), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def img_write(frame, sort_output, person_label, obj_list, flag):
    if flag == 1:
        frame = cv2ImgAddText(frame, '行人向下', 160, 10, (0, 245, 188), textSize=50)
    elif flag == 2:
        frame = cv2ImgAddText(frame, '行人向上', 160, 10, (255, 64, 204), textSize=50)
    else:
        frame = cv2ImgAddText(frame, '没有行人上下', 160, 10, (255, 255, 255), textSize=50)
    roiperson_num = 0
    person_count = [0, 0]
    # 显示
    if sort_output is not None:
        if len(sort_output[0]) > 0:
            bboxes = sort_output[1][0][:, :4]
            id = sort_output[1][0][:, 4]
            clss = sort_output[1][0][:, 5]
            for q, box in enumerate(bboxes):
                roiperson_num += 1
                clss_name = obj_list[clss[q]]
                color = compute_color_for_id(clss[q])
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2, 8)
                cv2.putText(frame, str(id[q])+' '+clss_name, ((box[0] + box[2]) // 2, box[1]), cv2.FONT_HERSHEY_PLAIN, 2.0,
                            color, 2, 8)

        # 写入数据
        if len(sort_output[0]) > 0:
            for i, clss in enumerate(sort_output[1][0][:, 5], start=1):  # 清洁工
                if i > 10:
                    break
                if clss == 1:  # cleaner
                    person_count[0] += 1
                elif clss == 0:
                    person_count[1] += 1
    sum_2 = roiperson_num
    sum_1 = roiperson_num
    cv2.putText(frame,
                ' current roi num:  {}  now all num: {}  clean:{} crew:{} '.format(
                    sum_1, sum_2, person_count[0], person_count[1]),
                (0, 15), cv2.FONT_HERSHEY_PLAIN, 1.5, (220, 20, 60), 2, 4)
    frame = cv2ImgAddText(frame, person_label, 390, 50, (200, 245, 188), textSize=25)
    return frame