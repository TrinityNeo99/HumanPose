'''
Descripttion: 自动描述，请修改
Author: Wei Jiangning
version: 
Date: 2022-01-20 11:11:32
LastEditors: Wei Jiangning
LastEditTime: 2023-04-18 15:31:34
'''
import pandas as pd
import numpy as np
import cv2


KEYPOINT = {
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

SKELETON = [
    [1,3],[1,0],[2,4],[2,0],[0,5],[0,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16],[5,6]
]

COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [220,220,220], [220,220,220], [220,220,220], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [220,220,220]]

def test_absolute_model():
    df = pd.read_excel(r"C:\Users\neo\OneDrive\文档\WeChat Files\wxid_bddyspb8b3vt12\FileStorage\File\2022-01\正手模型1.3.xlsx")
    print(df.head())
    for j, r in df.iterrows():
        img = np.zeros((720,1280,3), np.uint8)
        r = r.to_dict()
        for i in range(len(SKELETON)):
            kpt_a, kpt_b = KEYPOINT[SKELETON[i][0]], KEYPOINT[SKELETON[i][1]]
            x_a, y_a = r[kpt_a + "_x"] + 330, r[kpt_a + "_y"] + 410
            x_b, y_b = r[kpt_b + "_x"] + 330, r[kpt_b + "_y"] + 410
            cv2.circle(img, (int(x_a), int(y_a)), 6, COLORS[i], -1)
            cv2.circle(img, (int(x_b), int(y_b)), 6, COLORS[i], -1)
            cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), COLORS[i], 2)
        cv2.imwrite("../output/img_{}.jpg".format(j+1), img)
        print("finish")

def draw_skeleton(file, frame_n, mask):
    df = pd.read_csv(file, error_bad_lines=False, index_col="frame")
    print(df.head())
    r = df.loc[frame_n]
    img = np.zeros((720,1280,3), np.uint8)
    r = r.to_dict()
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = KEYPOINT[SKELETON[i][0]], KEYPOINT[SKELETON[i][1]]
        if kpt_a in mask or (kpt_a + "_x") not in r.keys():
            continue
        x_a, y_a = r[kpt_a + "_x"], r[kpt_a + "_y"]
        x_b, y_b = r[kpt_b + "_x"], r[kpt_b + "_y"]
        cv2.circle(img, (int(x_a), int(y_a)), 6, COLORS[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, COLORS[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), COLORS[i], 2)
    cv2.imwrite("./output/05-high-gravity_{}.jpg".format(frame_n), img)
    print("finish")

def draw_skeleton_kps(kps, filename, mask=[1, 2], size=(1920, 1080)):
    img = np.zeros((size[1],size[0],3), np.uint8)
    for i in range(len(SKELETON)):
        if SKELETON[i][0] in mask:
            continue
        a_x, a_y = kps[SKELETON[i][0]][0], kps[SKELETON[i][0]][1]
        b_x, b_y = kps[SKELETON[i][1]][0], kps[SKELETON[i][1]][1]
        cv2.circle(img, (int(a_x), int(a_y)), 6, COLORS[i], -1)
        cv2.circle(img, (int(b_x), int(b_y)), 6, COLORS[i], -1)
        cv2.line(img, (int(a_x), int(a_y)), (int(b_x), int(b_y)), COLORS[i], 2)
    cv2.imencode('.jpg', img)[1].tofile(filename)
    return img


def draw_skeleton_kps_on_org(kps, org, mask=[1, 2]):
    for i in range(len(SKELETON)):
        if SKELETON[i][0] in mask:
            continue
        a_x, a_y = kps[SKELETON[i][0]][0], kps[SKELETON[i][0]][1]
        b_x, b_y = kps[SKELETON[i][1]][0], kps[SKELETON[i][1]][1]
        cv2.line(org, (int(a_x), int(a_y)), (int(b_x), int(b_y)), COLORS[i], 2)
    return org

    

if __name__ == "__main__":
    f = [13, 27, 42, 55, 70, 83] # 01-standard
    f = [1, 16, 32, 47, 62, 75, 91, 104, 119, 132, 147, 161] # 06-high-backswing
    f = [1, 12, 31, 41, 61, 71, 89, 99, 117, 129] # 07-low-backswing
    f = [2, 11, 27, 36, 54, 64, 78, 91, 108, 121, 136, 149, 165, 179, 192, 200]# 05-high-gravity

    file = r"F:\HRnet-test\01-pose-output\01-pose-output\pose-data.csv" # 01-stardard
    file = r"F:\HRnet-test\wrong_action\05-pose\pose-data.csv" # 05
    mask = ['right_eye', 'left_eye']
    for n in f:
        draw_skeleton(file, n, mask)

