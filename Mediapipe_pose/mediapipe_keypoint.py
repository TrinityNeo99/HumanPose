'''
Descripttion: MediaPipe 提取骨骼关键点
Author: Wei Jiangning
version: v 1.0
Date: 2022-12-03 11:25:31
LastEditors: Wei Jiangning
LastEditTime: 2023-06-16 11:00:31
'''
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
from tqdm import tqdm
import time
from public.utils import *
from fractions import Fraction
from public.myexception import *
from public.config import *
import math
import csv

# 导出关键点的参考资料 https://blog.csdn.net/qq_64605223/article/details/125606507


cap = cv2.VideoCapture()
pose_model = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)


COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

def mideapipe_keypoints(args, save_key_points_csv=False):
    revert = True
    cap.open(args.video_path)
    frame_num, fps, duration, width, height = video_info(args.video_path)
    processed_fps = 60
    if fps > processed_fps:
        a = round(fps / processed_fps)
        b = 1
        skip_frame = generate_skip_frame_sequence(frame_num, a, b)
    else:
        skip_frame = generate_skip_frame_sequence(frame_num, 1, 1)  # 所有帧都要
    logger.debug(f"fps: {fps}, {str(skip_frame)}")
    keypoints_video_out_path = os.path.join(args.keypoints_dir, args.video_infer_raw_name)
    # vout = get_vout(keypoints_video_out_path, w, h)
    vout = get_vout_H264_mp4(keypoints_video_out_path)  # 支持H264编码
    all_kps = []
    t1 = time.time()
    cnt = 0
    write_cnt = 0
    csv_output_rows = []
    if frame_num < 2:
        raise VideoRead("ERRORR: reading org video {}".format(args.video_path))
    with tqdm(total=frame_num) as pbar:
        while cap.isOpened():
            success, image = cap.read()
            if revert:
                image = cv2.flip(image, 1)
            pbar.update(1)
            cnt += 1
            if not success:
                break
            if skip_frame[cnt] == 0:
                continue  # skip this frame
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose_model.process(image)
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # debug_print("cur_results: ", results.pose_landmarks)
            if results.pose_landmarks == None:
                continue
            cur_kps = keyponts_convert(results.pose_landmarks.landmark, width, height, cnt)
            cur_kps_2d = keyponts_convert_2d(results.pose_landmarks.landmark, width, height, cnt)
            csv_output_rows.append(cur_kps_2d)
            cur_kps_pairs = kps_line2pairs(cur_kps_2d)
            all_kps.append(cur_kps)
            image_sk = draw_skeleton_kps_on_origin(cur_kps_pairs, image, ratio=width / 1920)
            image_sk = cv2.cvtColor(image_sk, cv2.COLOR_BGR2RGB)
            if revert:
                image_sk = cv2.flip(image_sk, 1)
            vout.append_data(image_sk)
            write_cnt += 1

    t2 = time.time()
    frame_num = cnt  # opencv 读取原视频帧数可能会出现轻微偏差，与原视频的文件名可能有关，frame_num 以infer视频为准
    avg_fps = frame_num / (t2 - t1 + 0.00001)
    logger.info("generating infer video takes {} seconds".format(str(round(t2 - t1, 2))))
    logger.info("average fps: {}".format(avg_fps))
    cap.release()
    vout.close()
    if len(all_kps) == 0:
        raise VideoNoPerson("ERROR: there is no person in the video")
    all_kps = np.array(all_kps).astype(np.float32)

    # save keypoint to csv
    csv_headers = ['frame']
    for keypoint in COCO_KEYPOINT_INDEXES.values():
        csv_headers.extend([keypoint + '_x', keypoint + '_y'])

    csv_output_filename = os.path.join(args.keypoints_dir, 'pose-data.csv')
    print(csv_output_filename)
    if save_key_points_csv:
        with open(csv_output_filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(csv_headers)
            csvwriter.writerows(csv_output_rows)

    infer_frame_num, infer_fps, infer_duration, infer_width, infer_height = video_info(
        keypoints_video_out_path + ".mp4")
    logger.info(
        f"width: {width}, height: {height}, frame_count: {frame_num} fps: {fps}, duration: {duration}, infer_frame_num: {infer_frame_num}, infer_fps: {infer_fps}, infer_duration: {infer_duration}")
    infer2org_map = infer2org_frame_map(skip_frame)
    return all_kps, width, height, frame_num, duration, infer_frame_num, infer_duration, infer2org_map


def keyponts_convert(frame_results, frame_w, frame_h, frame_cnt):
    nose = frame_results[mp_pose.PoseLandmark.NOSE]
    keypoints = []
    nose_x = nose.x * frame_w
    nose_y = nose.y * frame_h
    # keypoints.append(frame_cnt)
    keypoints.append([nose_x, nose_y])
    # keypoints.append(nose_y)
    left_eye = frame_results[mp_pose.PoseLandmark.LEFT_EYE]
    keypoints.append([left_eye.x * frame_w, left_eye.y * frame_h])

    right_eye = frame_results[mp_pose.PoseLandmark.RIGHT_EYE]
    keypoints.append([right_eye.x * frame_w, right_eye.y * frame_h])

    left_ear = frame_results[mp_pose.PoseLandmark.LEFT_EAR]
    keypoints.append([left_ear.x * frame_w, left_ear.y * frame_h])

    right_ear = frame_results[mp_pose.PoseLandmark.RIGHT_EAR]
    keypoints.append([right_ear.x * frame_w, right_ear.y * frame_h])

    left_shoulder = frame_results[mp_pose.PoseLandmark.LEFT_SHOULDER]
    keypoints.append([left_shoulder.x * frame_w, left_shoulder.y * frame_h])

    right_shoulder = frame_results[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    keypoints.append([right_shoulder.x * frame_w, right_shoulder.y * frame_h])

    left_elbow = frame_results[mp_pose.PoseLandmark.LEFT_ELBOW]
    keypoints.append([left_elbow.x * frame_w, left_elbow.y * frame_h])

    right_elbow = frame_results[mp_pose.PoseLandmark.RIGHT_ELBOW]
    keypoints.append([right_elbow.x * frame_w, right_elbow.y * frame_h])

    left_wrist = frame_results[mp_pose.PoseLandmark.LEFT_WRIST]
    keypoints.append([left_wrist.x * frame_w, left_wrist.y * frame_h])

    right_wrist = frame_results[mp_pose.PoseLandmark.RIGHT_WRIST]
    keypoints.append([right_wrist.x * frame_w, right_wrist.y * frame_h])

    left_hip = frame_results[mp_pose.PoseLandmark.LEFT_HIP]
    keypoints.append([left_hip.x * frame_w, left_hip.y * frame_h])

    right_hip = frame_results[mp_pose.PoseLandmark.RIGHT_HIP]
    keypoints.append([right_hip.x * frame_w, right_hip.y * frame_h])

    left_knee = frame_results[mp_pose.PoseLandmark.LEFT_KNEE]
    keypoints.append([left_knee.x * frame_w, left_knee.y * frame_h])

    right_knee = frame_results[mp_pose.PoseLandmark.RIGHT_KNEE]
    keypoints.append([right_knee.x * frame_w, right_knee.y * frame_h])

    left_ankle = frame_results[mp_pose.PoseLandmark.LEFT_ANKLE]
    keypoints.append([left_ankle.x * frame_w, left_ankle.y * frame_h])
    right_ankle = frame_results[mp_pose.PoseLandmark.RIGHT_ANKLE]
    keypoints.append([right_ankle.x * frame_w, right_ankle.y * frame_h])

    # print(keypoints)
    # print(len(keypoints))

    return keypoints


def keyponts_convert_2d(frame_results, frame_w, frame_h, frame_cnt):
    nose = frame_results[mp_pose.PoseLandmark.NOSE]
    keypoints = []
    nose_x = nose.x * frame_w
    nose_y = nose.y * frame_h
    keypoints.append(frame_cnt)  # 0
    keypoints.append(nose_x)  # 1
    keypoints.append(nose_y)  # 2
    keypoints.extend([-1] * 8)  # 3 4 5 6 7 8 9 10 # 面部的这几个3点我们不用，所以无需在意其坐标
    left_shoulder = frame_results[mp_pose.PoseLandmark.LEFT_SHOULDER]
    keypoints.append(left_shoulder.x * frame_w)  # 11
    keypoints.append(left_shoulder.y * frame_h)  # 12
    right_shoulder = frame_results[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    keypoints.append(right_shoulder.x * frame_w)  # 13
    keypoints.append(right_shoulder.y * frame_h)  # 14
    left_elbow = frame_results[mp_pose.PoseLandmark.LEFT_ELBOW]
    keypoints.append(left_elbow.x * frame_w)  # 15
    keypoints.append(left_elbow.y * frame_h)  # 16
    right_elbow = frame_results[mp_pose.PoseLandmark.RIGHT_ELBOW]
    keypoints.append(right_elbow.x * frame_w)  # 17
    keypoints.append(right_elbow.y * frame_h)  # 18
    left_wrist = frame_results[mp_pose.PoseLandmark.LEFT_WRIST]
    keypoints.append(left_wrist.x * frame_w)  # 19
    keypoints.append(left_wrist.y * frame_h)  # 20
    right_wrist = frame_results[mp_pose.PoseLandmark.RIGHT_WRIST]
    keypoints.append(right_wrist.x * frame_w)  # 21
    keypoints.append(right_wrist.y * frame_h)  # 22
    left_hip = frame_results[mp_pose.PoseLandmark.LEFT_HIP]
    keypoints.append(left_hip.x * frame_w)  # 23
    keypoints.append(left_hip.y * frame_h)  # 24
    right_hip = frame_results[mp_pose.PoseLandmark.RIGHT_HIP]
    keypoints.append(right_hip.x * frame_w)  # 25
    keypoints.append(right_hip.y * frame_h)  # 26
    left_knee = frame_results[mp_pose.PoseLandmark.LEFT_KNEE]
    keypoints.append(left_knee.x * frame_w)  # 27
    keypoints.append(left_knee.y * frame_h)  # 28
    right_knee = frame_results[mp_pose.PoseLandmark.RIGHT_KNEE]
    keypoints.append(right_knee.x * frame_w)  # 29
    keypoints.append(right_knee.y * frame_h)  # 30
    left_ankle = frame_results[mp_pose.PoseLandmark.LEFT_ANKLE]
    keypoints.append(left_ankle.x * frame_w)  # 31
    keypoints.append(left_ankle.y * frame_h)  # 32
    right_ankle = frame_results[mp_pose.PoseLandmark.RIGHT_ANKLE]
    keypoints.append(right_ankle.x * frame_w)  # 33
    keypoints.append(right_ankle.y * frame_h)  # 34
    # print(keypoints)
    # print(len(keypoints))

    return keypoints


def kps_line2pairs(kps):
    ret = []
    for i in range(1, 35, 2):
        ret.append([kps[i], kps[i + 1]])
    return ret


if __name__ == "__main__":
    all_kps = mideapipe_keypoints(r"C:\Users\weiji\Downloads\diving\80.mp4")
# print(all_kps)
