"""
@Project: 2023-HumanPose
@FileName: extract_human_pose.py
@Description: 提取人体骨骼关键点
@Author: Wei Jiangning
@version: 1.0.0a1.0
@Date: 2023/8/1 11:06 at PyCharm
"""
import os
import pickle

import inference as hrnet_api
import common.utils as u
from common.log import logger


class Extract():
    def __init__(self, video_dir, out_dir, label_list_path):
        self.data_dir = video_dir
        self.out_dir = out_dir
        self.files = os.listdir(self.data_dir)
        self.val_num = [1]
        with open(label_list_path, 'r', encoding='utf-8') as f:
            self.labels = f.read().splitlines()
            self.labels = {l.split(" ")[0]: l.split(" ")[1] for l in self.labels}

    def annotate_f_pyskl(self):
        logger.info("hrnet_api initializing...")
        args, box_model, pose_model, pose_transform = hrnet_api.initialize()
        annotations = []
        logger.info("annotating...")
        for f in self.files:
            logger.info(f"processing {f} ({self.files.index(f) + 1}/{len(self.files)})")
            frame_num, fps, duration, width, height = u.video_info(os.path.join(self.data_dir, f))
            pose_preds_frames, confidence_frames = hrnet_api.generate_kps(os.path.join(self.data_dir, f), args,
                                                                          box_model,
                                                                          pose_model, pose_transform)
            assert pose_preds_frames[:, 0, :, :].shape == (1, 17, 2)
            assert confidence_frames[:, 0, :, :].shape == (1, 17, 1)  # follow coco format 17 keypoints for human body
            assert pose_preds_frames.shape[1] == confidence_frames.shape[1]  # ensure time len is equal
            ann = {"frame_dir": f, "img_shape": (width, height), 'original_shape': (width, height),
                   'label': int(self.labels.get(f)), "keypoint": pose_preds_frames, "keypoint_score": confidence_frames,
                   'total_frames': pose_preds_frames.shape[1]}
            annotations.append(ann)

        return annotations

    def split_f_pyskl(self, annotations):
        logger.info("splitting...")
        train = []
        val = []
        for a in annotations:
            if int(a['frame_dir'].split("Sub_")[1].split(".")[0]) in self.val_num:
                val.append(a['frame_dir'])
            else:
                train.append(a['frame_dir'])
        split = {'train': train, 'val': val}
        return split

    def generate(self, format='pyskl', data_name='pinpong-109-coco'):
        logger.info('Generating...')
        annotate_func = eval(f"self.annotate_f_{format}")
        annotations = annotate_func()
        with open(os.path.join(self.out_dir, f'{data_name}_{format}_hrnet_annotations.pkl'), 'wb') as file:
            pickle.dump(annotations, file)
        spilt_func = eval(f"self.split_f_{format}")
        split = spilt_func(annotations)
        data = {'annotations': annotations, 'split': split}
        with open(os.path.join(self.out_dir, f'{data_name}_{format}_hrnet_annotations_split.pkl'), 'wb') as file:
            pickle.dump(data, file)
        logger.info("Done!")


if __name__ == '__main__':
    e = Extract(video_dir=r"F:\pingpong-all-data\2023-4-19_北体合作_动作示范视频_实验用小规模数据集",
                out_dir=r"F:\pingpong-all-data\2023-4-19_北体合作_动作示范视频_实验用小规模数据集_pyskl_output",
                label_list_path='./resource/pingpong-109-coco_label.list')
    e.generate()

    # python extract_human_pose --cfg ./config/inference-config.yaml --videoFile ../video/IMG_2411_153_327.mp4 --writeBoxFrames --outputDir ./output  TEST.MODEL_FILE ./resource/pose_hrnet_w32_256x192.pth
