'''
Descripttion: 动作识别部分的参数配置
Author: Wei Jiangning
version: v1.0
Date: 2022-09-16 10:22:42
LastEditors: Wei Jiangning
LastEditTime: 2023-01-08 11:15:22
'''
#  Copyright (c) 2023. IPCRC, Lab. Jiangnig Wei
#  All rights reserved

import argparse
import os


class action_config():

    def __init__(self, video_path, keypoints_dir) -> None:
        self.video_path = video_path
        self.video_infer_raw_name = os.path.basename(video_path).replace(".mp4", "")
        sub_keypoints_dir = os.path.join(keypoints_dir, self.video_infer_raw_name)
        self.keypoints_dir = sub_keypoints_dir
        os.makedirs(keypoints_dir, exist_ok=True)
        os.makedirs(sub_keypoints_dir, exist_ok=True)
