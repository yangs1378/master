# -*- coding: utf-8 -*-
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

class DEEPSORT:
    def __init__(self):
        # deepsort模型加载
        self.cfg = get_config()
        self.config_deepsort = "deep_sort_pytorch/configs/deep_sort.yaml"
        self.cfg.merge_from_file(self.config_deepsort)

    def load_deepsort(self):
        self.deepsort = DeepSort(self.cfg.DEEPSORT.REID_CKPT,
                                 max_dist=self.cfg.DEEPSORT.MAX_DIST, min_confidence=self.cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=self.cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=self.cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=self.cfg.DEEPSORT.MAX_AGE, n_init=self.cfg.DEEPSORT.N_INIT,
                                 nn_budget=self.cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)
        return self.deepsort