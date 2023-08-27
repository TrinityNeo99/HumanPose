'''
Descripttion: 骨骼关节点绘图工具
Author: Wei Jiangning
version: v 1.2
Date: 2022-12-03 17:27:57
LastEditors: Wei Jiangning
LastEditTime: 2022-12-05 23:31:20
'''
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
    16: 'right_ankle',
    17: 'right_index'
}


SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16], [5, 6]
]

COLORS = [[255, 1, 1], [255, 85, 1], [255, 170, 1], [255, 255, 1], [170, 255, 1], [85, 255, 1], [1, 255, 1],
          [1, 255, 85], [1, 255, 170], [1, 255, 255], [220, 220, 220], [220, 220, 220], [220, 220, 220], [85, 1, 255],
          [170, 1, 255], [255, 1, 255], [255, 1, 170], [220, 220, 220]]


def draw_skeleton_kps_on_back_save_fig(kps, filename, mask=[1, 2], size=(1920, 1080)):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    for i in range(len(SKELETON)):
        if SKELETON[i][0] in mask:
            continue
        a_x, a_y = kps[SKELETON[i][0]][0], kps[SKELETON[i][0]][1]
        b_x, b_y = kps[SKELETON[i][1]][0], kps[SKELETON[i][1]][1]
        cv2.circle(img, (int(a_x), int(a_y)), 6, COLORS[i], -1)
        cv2.circle(img, (int(b_x), int(b_y)), 6, COLORS[i], -1)
        cv2.line(img, (int(a_x), int(a_y)), (int(b_x), int(b_y)), COLORS[i], 2)
        cv2.imencode('.jpg', img)[1].tofile(filename)


def draw_skeleton_kps_on_origin(kps, img, mask=[1, 2], ratio=1):
    for i in range(len(SKELETON)):
        if SKELETON[i][0] in mask:
            continue
        a_x, a_y = kps[SKELETON[i][0]][0], kps[SKELETON[i][0]][1]
        b_x, b_y = kps[SKELETON[i][1]][0], kps[SKELETON[i][1]][1]
        # cv2.circle(img, (int(a_x), int(a_y)), 6, COLORS[i], -1)
        # cv2.circle(img, (int(b_x), int(b_y)), 6, COLORS[i], -1)
        # cv2.line(img, (int(a_x), int(a_y)), (int(b_x), int(b_y)), COLORS[i], 2)
        # 同一绘图颜色
        color = (255, 255, 0)
        radius = int(10 * ratio)
        line_thickness = int(3 * ratio)
        cv2.circle(img, (int(a_x), int(a_y)), radius, color, -1)
        cv2.circle(img, (int(b_x), int(b_y)), radius, color, -1)
        cv2.line(img, (int(a_x), int(a_y)), (int(b_x), int(b_y)), color, thickness=line_thickness)
    return img


if __name__ == "__main__":
    f = [13, 27, 42, 55, 70, 83]  # 01-standard
    f = [1, 16, 32, 47, 62, 75, 91, 104, 119, 132, 147, 161]  # 06-high-backswing
    f = [1, 12, 31, 41, 61, 71, 89, 99, 117, 129]  # 07-low-backswing
    f = [2, 11, 27, 36, 54, 64, 78, 91, 108, 121, 136, 149, 165, 179, 192, 200]  # 05-high-gravity

    file = r"F:\HRnet-test\01-pose-output\01-pose-output\pose-data.csv"  # 01-stardard
    file = r"F:\HRnet-test\wrong_action\05-pose\pose-data.csv"  # 05
    mask = ['right_eye', 'left_eye']
    for n in f:
        draw_skeleton(file, n, mask)
