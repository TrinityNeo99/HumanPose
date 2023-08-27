"""
@Project: 2023-HumanPose
@FileName: profile_test.py
@Description: 测试HRnet的性能，并保存各个帧关节点坐标
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2023/8/27 19:37 at PyCharm
"""
import os
import time

import inference as hrnet_api
import common.utils as u
from common.log import logger


class Extract():
    def __init__(self, video_dir, out_dir="./output"):
        self.data_dir = video_dir
        self.out_dir = out_dir
        self.files = os.listdir(self.data_dir)

    def run(self):
        logger.info("hrnet_api initializing...")
        args, box_model, pose_model, pose_transform = hrnet_api.initialize()
        args.outputDir = self.out_dir
        start = time.time()
        for f in self.files:
            logger.info(f"processing {f} ({self.files.index(f) + 1}/{len(self.files)})")
            frame_num, fps, duration, width, height = u.video_info(os.path.join(self.data_dir, f))
            hrnet_api.generate_kps(os.path.join(self.data_dir, f), args,
                                   box_model,
                                   pose_model, pose_transform)
        end = time.time()

        logger.info(f"Total cost: {round((end - start), 2)} second")

if __name__ == '__main__':
    test = Extract(video_dir=r"C:\Users\weiji\Downloads\diving")
    test.run()
