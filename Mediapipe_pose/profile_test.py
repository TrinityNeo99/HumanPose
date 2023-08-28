"""
@Project: 2023-HumanPose
@FileName: profile_test.py
@Description: 自动描述，请及时修改
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2023/8/27 21:01 at PyCharm
"""
import os
import time

from public.config import *
from mediapipe_keypoint import mideapipe_keypoints
from public.log import logger

class Extract():
    def __init__(self, video_dir, out_dir="./output"):
        self.data_dir = video_dir
        self.out_dir = out_dir
        self.files = os.listdir(self.data_dir)

    def run(self):
        start = time.time()
        for f in self.files:
            args = action_config(video_path=os.path.join(self.data_dir, f), keypoints_dir=self.out_dir)
            all_kps = mideapipe_keypoints(args, save_key_points_csv=True)
        end = time.time()

        logger.info(f"Total cost: {round((end - start), 2)} second")

if __name__ == '__main__':
    e = Extract(video_dir=r"/root/2DHPE/data/pingpong-109-coco-mini")
    e.run()
    print("woof woof I quit, bye!~")