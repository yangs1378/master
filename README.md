# yolov7-deepsort
## 权重部分
### ckpt.t7(跟踪部分网络权重，放到master/deep_sort_pytorch\deep_sort\deep\checkpoint中)
链接：https://pan.baidu.com/s/1t9_ECRPYd-An_TpholPBqQ 
提取码：54pn
### traced_model.pt（yolov7部分权重，放到master/yolov7中）
链接：https://pan.baidu.com/s/1XOEc4gcBpDgNsV3fQb0sCg 
提取码：nxqf
### yolov7.pt（yolov7网络权重，放到master中）
链接：https://pan.baidu.com/s/1yk7w1vobYgQXXj433-BWfQ 
提取码：toqf
## 使用说明
### 1、跟踪逻辑部分
主要的跟踪逻辑在judge.py  
根据实例分割获取的扶梯范围筛选行人，统计3s内检测跟踪到的行人边界框每两帧间移动距离移动方向，运动距离未超过一定阈值判定为停留，运动方向与光流检测到的扶梯运动方向相反判断为逆行，输出每个跟踪目标的状态  

### 2、运行文件
python yolo_v7_deepsort_judge.py --video 6mm_20220526142813224.avi --weight yolov7.pt  
