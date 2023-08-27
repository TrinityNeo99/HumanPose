'''
Descripttion: 项目的小工具，以后其他项目可能会用到
Author: Wei Jiangning
version: v1.0
Date: 2022-04-19 11:10:09
LastEditors: Wei Jiangning
LastEditTime: 2023-01-14 13:57:28
'''
from ast import For
from cmath import cos, pi, sqrt
import math
import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from colorama import Fore, Back, Style, init
from public.draw import draw_skeleton_kps_on_origin
from moviepy.editor import VideoFileClip, AudioFileClip

init(autoreset=True)
import shutil
import time
from scipy import signal
import scipy.interpolate
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import imageio as iio
from public.log import *
from public.myexception import *


def lenof(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)


def lenxof(A, B):
    lenx = abs(A[0] - B[0])
    if abs(A[0] - B[0]) == 0:
        lenx = 1
    return lenx


'''
descripttion: 
param {*} A
param {*} B
param {*} C
return {*} 返回以A为顶点的角的度数
'''


def angleof(A, B, C):
    c = lenof(A, B)
    a = lenof(B, C)
    b = lenof(A, C)
    if b == 0 or c == 0:
        debug_print("A", A)
        debug_print("B", B)
        debug_print("C", C)
        consinA = 1
    else:
        consinA = (c ** 2 + b ** 2 - a ** 2) / (2 * b * c)
    consinA = round(consinA, 4)
    return math.acos(consinA) * 180 / pi


'''
descripttion: 可以产生大于180的角度
param {*} A
param {*} B
param {*} C
return {*} 返回以A为顶点的角的度数
'''


def angle360of(A, B, C):
    d = lineOf(A, B)
    e = lineOf(A, C)
    if d[0] > 0:
        if e[0] < 0:
            return angleof(A, B, C)
        else:
            if d[0] > e[0]:
                return 180 + (180 - angleof(A, B, C))

            else:
                return angleof(A, B, C)
    else:
        if d[0] < e[0]:
            return angleof(A, B, C)
        else:
            return 180 + (180 - angleof(A, B, C))


def merge2pic(path1, path2):
    # p1 = cv2.imread(path1)
    # p2 = cv2.imread(path2)
    p1 = cv_imread(path1)
    p2 = cv_imread(path2)
    p1 = cut_pic(p1)
    p2 = cut_pic(p2)
    p = np.concatenate([p1, p2], axis=1)
    return p


def cut_pic(img):
    h = len(img)
    w = len(img[0])
    img = img[0: h, 0: int(w / 2)]
    return img


def generate_anotation_pic(p1, p2, name, text, outdir=None):
    p = merge2pic(p1, p2)
    p = cv2AddChinese(p, name, (600, 50), (0, 255, 0), 30)
    # cv2.putText(p, name, (800, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 255), 1)
    img_l = len(p[0])
    p = cv2AddChinese(p, text, (img_l - 780, 70), (0, 255, 255), 30)
    # cv2.imshow("Chinese", p)
    # cv2.waitKey(1000)
    if outdir == None:
        cv2.imwrite("output/" + name, p)
    else:
        cv2.imencode('.jpg', p)[1].tofile(os.path.join(outdir, name))


def cv2AddChinese(img, text, pos, color, size):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    sys_font = "C:/Windows/Fonts/simsun.ttc"
    # print("in cv2addVChinese func: the size: ", size)
    fontstyle = ImageFont.truetype(sys_font, size, encoding="utf-8")
    draw.text(pos, text, color, font=fontstyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)  # 转换为opencv 格式


def listdir(path):
    bad_filename = []
    bad_filename.append("~$模板动作标注(2410).xlsx")
    bad_filename.append("~$模板动作标注(2398).xlsx")
    bad_filename.append("~$模板动作标注(2411).xlsx")
    list_name = []
    for file in os.listdir(path):
        if file in bad_filename:
            continue
        # print(Fore.GREEN + file)
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name


def cut_video_from_file(cut_basis):
    # path = "F:\pingpang-all-data\Video_Iphone_0110\视频切分_v1\IMG_2400.txt"
    name = cut_basis.split("\\")[-1].replace('.txt', "")
    out_dir = os.path.join(r"F:\pingpang-all-data\Video_iPhone_0228\视频切分结果", name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        print("path: {} is already exist, do you want to overwirte it?, input yes/no".format(out_dir))
        ans = input()
        if ans == "yes":
            print("begin overwrite")
        else:
            print("bye")
            return -1

    video_name = name + ".MOV"
    input_video = os.path.join(r"F:\pingpang-all-data\Video_iPhone_0228\素材", video_name)
    f = open(cut_basis)
    basis = f.readlines()
    for ba in basis:
        s = int(ba.split(",")[0])
        e = int(ba.split(",")[1].replace("\n", ""))
        get_part_of_video(input_video, s, e, name, out_dir)


def cut_video_from_list_pairs(pairs, name, video_root_path, video_cut_path):
    out_dir = os.path.join(video_cut_path, name)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        print("path: {} is already exist, do you want to overwirte it?, input yes/no".format(out_dir))
        ans = input()
        if ans == "yes":
            print("begin overwrite")
        else:
            print("bye")
            return -1

    video_name = name + ".mp4"
    input_video = os.path.join(video_root_path, video_name)
    for st in pairs:
        get_part_of_video(input_video, st[0], st[1], name, out_dir)


def get_part_of_video(video, start, end, name, outdir, isGif=False):
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    all_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # print("fps: ", fps)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    if end >= all_frame_cnt:
        end = all_frame_cnt - 1

    if isGif:
        gif_images = []
        gif_name = name + "_" + str(start) + "_" + str(end) + ".gif"
    else:
        name = name + "_" + str(start) + "_" + str(end)
        vout = get_vout_H264_mp4(os.path.join(outdir, name))  # 使用浏览器可播放的格式
    before = cap.get(cv2.CAP_PROP_POS_FRAMES)
    # print(Fore.GREEN + "now video pointer: " + str(before))
    t1 = time.time()
    step = 1000
    for i in range(0, start + 1, step):
        if start - i < step:
            i = start
        if i > 20000:
            j = 20000
            while j < i + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                j = j + 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    t2 = time.time()
    after = cap.get(cv2.CAP_PROP_POS_FRAMES)
    # print(Fore.GREEN + "now video pointer: " + str(after))
    # print("During time: ", t2 - t1)
    length = end - start + 1
    i = 0
    while i < length:
        flag, frame = cap.read()
        if flag:
            # print(name, "now at: ", i + start)
            # print(Fore.BLUE + "all frame cnt: " + str(all_frame_cnt))
            # print(Fore.GREEN + "now video pointer: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            pass
        else:
            # print(Fore.RED + name + " get frame from video wrong")
            # print(Fore.BLUE + "all frame cnt: " + str(all_frame_cnt))
            # print(Fore.GREEN + "now video pointer: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            raise VideoRead("ERRORR: reading infer video {} to make sub-infers".format(video))
            pass
        if isGif:
            im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gif_images.append(im_rgb)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vout.append_data(frame)
        i = i + 1

    if isGif:
        gif_path = os.path.join(outdir, gif_name)
        # imageio.mimsave(gif_path, gif_images, fps=5)
        # print("the gif path is: ",gif_path)
    cap.release()
    return os.path.join(outdir, name)


'''
descripttion: get a segment from origin video and add text into the video
param {*} video video path
param {*} start segment frame start
param {*} end segmant frame end
param {*} name raw video name "0122"
param {*} outdir output dir path
param {*} text text context
param {*} pos text position
param {*} color text color
param {*} size text size
param {*} IsGif whether or not generate .gif with video
return {*} 
'''


def get_part_of_video_add_text(video, start, end, name, outdir, text, pos=(50, 50), color=(0, 255, 0), text_size=200,
                               IsGif=True):
    print(Fore.GREEN + "we are here!")
    print("the video path: ", video)
    # print("start: ", start)

    cap = cv2.VideoCapture(video)
    flag, frame = cap.read()
    if flag == False:
        print(Fore.RED + "in the get_part_of_video_add_test" + str(flag))
        print(Fore.RED + "video path: " + str(video))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) / 2
    all_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    black_border_size = 1000  # origin
    # black_border_size = 0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) + black_border_size, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("fps: ", fps)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    video_name = name + "_" + str(start) + "_" + str(end) + ".mp4"
    vout = cv2.VideoWriter(os.path.join(outdir, video_name), fourcc, fps, size)

    # print("the vout: ", os.path.join(outdir, name))

    before = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(Fore.GREEN + "now video pointer: " + str(before))
    t1 = time.time()
    step = 1000
    for i in range(0, start + 1, step):
        if start - i < step:
            i = start
        if i > 20000:
            j = 20000
            while j < i + 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, j)
                j = j + 1
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    t2 = time.time()
    after = cap.get(cv2.CAP_PROP_POS_FRAMES)
    print(Fore.GREEN + "now video pointer: " + str(after))
    print("During time: ", t2 - t1)
    length = end - start + 1
    i = 0
    gif_images = []
    while i < length:
        flag, frame = cap.read()
        if flag == False:
            print(Fore.RED + "can not read frame")
            return
        frame = cv2.copyMakeBorder(frame, 0, 0, black_border_size, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if flag:
            # print(name, "now at: ", i + start)
            # print(Fore.BLUE + "all frame cnt: " + str(all_frame_cnt))
            # print(Fore.GREEN + "now video pointer: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            pass
        else:
            # print(Fore.RED + name + " get frame from video wrong")
            # print(Fore.BLUE + "all frame cnt: " + str(all_frame_cnt))
            # print(Fore.GREEN + "now video pointer: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            pass
        # print("in another func the text_size: ", text_size)
        frame = cv2AddChinese(frame, text, pos, color, text_size)
        vout.write(frame)
        i = i + 1

        if IsGif:
            im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gif_images.append(im_rgb)

    if len(gif_images) > 0:
        gif_name = name + "_" + str(start) + "_" + str(end) + ".gif"
        gif_path = os.path.join(outdir, gif_name)
        # imageio.mimsave(gif_path, gif_images, fps=5)
        print("the gif path is: ", gif_path)


'''
descripttion: 中文路径的读写操作
param {*} filePath
return {*}
'''


def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    return cv_img


def cv_imwrite(filePathName, img):
    try:
        _, cv_img = cv2.imencode(".jpg", img)[1].tofile(filePathName)
        return True
    except:
        return False


''' 
anotation file: 24xx.excel
select = ["引拍过高"， "其实重心过高"]
    
'''


def generate_video_source(name, predir):
    name = name.replace(".mp4", "")
    if name.split("_")[0] == "IMG":
        video_number = name.split("_")[1]
        # print(name)
        # predir = r"F:\pingpang-all-data\Video_Iphone_0110\关键点提取结果"
        dir_name = "IMG_" + video_number
    # elif name.split("_")[0] == "1" or name.split("_")[0] == "2":
    #     dir_name = "0" + name.split("_")[0]
    else:
        video_number = name.split("_")[0]
        dir_name = video_number

    pose_csv = os.path.join(predir, dir_name, name, "pose-data.csv")
    pose_imgs = os.path.join(predir, dir_name, name, "pose")
    pose_skes = os.path.join(predir, dir_name, name, "ske")
    return pose_csv, pose_imgs, pose_skes


def make_new_dir_and_clear(out_dir):
    if os.path.exists(out_dir) and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
        print(Fore.RED + "delete " + out_dir)
    os.makedirs(out_dir, exist_ok=True)


def bin_class_score(TP, TN, FP, FN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = (2 * precision * recall) / (precision + recall)
    print(Fore.BLUE + "acuracy: ", str(accuracy))
    print(Fore.BLUE + "precision: ", str(precision))
    print(Fore.BLUE + "recall: ", str(recall))
    print(Fore.BLUE + "F1_score: ", str(F1_score))
    return accuracy, precision, recall, F1_score


def slope_line(A, B):
    ratio = (A[1] - B[1]) / (A[0] - B[0])
    return ratio


def ratio2angle(r):
    return r * 180 / pi


# point [x, y]
# line [k, b]
# return  distance,  pointOnLine(drop foot) [x, y]
def distance_point2line(point, line):
    k = - (1 / line[0])
    b = point[1] - k * point[0]
    drop_foot = get_joint_two_lines(line, [k, b])
    distance = lenof(drop_foot, point)
    return distance, drop_foot


def get_joint_two_lines(la, lb):
    x = (lb[1] - la[1]) / (la[0] - lb[0])
    y = la[0] * x + la[1]
    return [x, y]


def lineOf(A, B):
    if abs(A[0] - B[0]) < 1:
        k = -1000
    else:
        k = (A[1] - B[1]) / (A[0] - B[0])
    b = A[1] - k * A[0]
    return [k, b]


def fliter_low_pass(data):
    b, a = signal.butter(8, 0.99, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data)  # data为要过滤的信号
    return filtedData


''' 
descripttion: 查看变量占用内存情况
param unit: 显示的单位，可为`B`,`KB`,`MB`,`GB`
param threshold: 仅显示内存数值大于等于threshold的变量
'''


def show_memory(val_list, unit='KB', threshold=1):
    from sys import getsizeof
    scale = {'B': 1, 'KB': 1024, 'MB': 1048576, 'GB': 1073741824}[unit]
    print("show the memory")
    for i in val_list:
        memory = eval("getsizeof({})".format(i)) // scale
        if memory >= threshold:
            print(i, memory)


def get_shoulder_distances(kps):
    shoulder2s_distances = []
    for line in kps:
        right_shoulder = [line[13], line[14]]
        left_shoulder = [line[11], line[12]]
        dis_shoulder2 = lenof(right_shoulder, left_shoulder)
        if right_shoulder[0] > left_shoulder[0]:
            dis_shoulder2 = -1 * dis_shoulder2
        shoulder2s_distances.append(dis_shoulder2)
    # print("the len of sequence :", len(shoulder2s_distances))
    # draw_fig(shoulder2s_distances, name, outdir)
    # print(len(shoulder2s_distances))
    # if len(shoulder2s_distances) < 31:
    shoulder2s_distances = scale_sequence(shoulder2s_distances, 30, 'linear')
    return shoulder2s_distances


'''
descripttion: convert 1d sequence to setted length using interpolating method.
              The sequence can be shortened or extended to a specified size, like(10 -> 30, 10->5 )
param {*} seq 
param {*} length int
param {*} kind 'linear' or '...'
return {*}
'''


def scale_sequence(seq, length, kind):
    x = np.arange(0, len(seq))
    x = x * length / len(x)
    y = seq
    x_new = np.linspace(0, x.max(), length)
    f = scipy.interpolate.interp1d(x, y, kind=kind)
    y_new = f(x_new)
    # plt.plot(x, y, 'x')
    # plt.plot(x_new, y_new, 'o')
    # plt.show()
    return y_new


def dict2json_format(d):
    return json.dumps(d, sort_keys=True, indent=4, separators=(',', ':'), ensure_ascii=False)


def video_segment_add_padding(se: list, max_len, padding=5):
    s = se[0]
    e = se[1]
    s -= padding
    e += padding
    if s < 0:
        s = 0
    if e > max_len:
        e = max_len

    return [s, e]


'''
descripttion: save every frame to img from video
param {*} video_path 
param {*} output_dir
return {*}
'''


def video2frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    cnt = 0
    flag, frame = cap.read()
    frame_num = cap.get(7)
    with tqdm(total=frame_num) as pbar:
        pbar.set_description('Processing:')
        while flag:
            name = "frame_{:08d}.jpg".format(cnt)
            cv_imwrite(os.path.join(output_dir, name), frame)
            cnt += 1
            flag, frame = cap.read()
            pbar.update(1)


# 修改透明背景为白色背景图
def transparent2white(img):
    height, width, channel = img.shape
    for h in range(height):
        for w in range(width):
            color = img[h, w]
            if (color == np.array([0, 0, 0, 0])).all():
                img[h, w] = [255, 255, 255, 255]

    return img


# 修改纯白背景图为透明背景图
def white2transparent(img):
    height, width, channel = img.shape
    for h in range(height):
        for w in range(width):
            color = img[h, w]
            if (color == np.array([255, 255, 255, 255])).all():
                img[h, w] = [0, 0, 0, 0]
    return img


def set_image_alpha(img):
    b, g, r, a = cv2.split(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if not (b[i][j] == 255 and g[i][j] == 255 and r[i][j] == 255):
                a[i][j] = 255  # 255不透明，0全透明，有像素的地方设置不透明
    image = cv2.merge((b, g, r, a))
    return image


'''
descripttion: show the debug image for analysis
param {str} video_path raw video path
param {int} frame_th the i-th frame in video
param {list} frame_kps skeleton points in this frame
param {str} text the anotation you want to show in the image
return {*}
'''


def debug_analysis_image(video_path: str, frame_th: int, frame_kps: list, text: str):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_th)
    flag, frame = cap.read()
    if flag == False:
        print(Fore.RED + "error in cap.read")
        return
    h, w = frame.shape[0], frame.shape[1]
    black_img = np.zeros(frame.shape)
    skeleton_on_black = draw_skeleton_kps_on_origin(frame_kps, black_img)
    image_merged = merge2pic_on_half(frame, skeleton_on_black)
    cv2.imwrite("./test_out/merge.jpg", image_merged)
    image_merged = cv2.imread("./test_out/merge.jpg")
    image_ana = cv2AddChinese(image_merged, text, (0.5 * w, 0.1 * h), (0, 255, 255), 15)
    return image_ana


def merge2pic_on_half(p1, p2):
    if p1.shape != p2.shape:
        print(Fore.RED + "the shape of two images is not equal")
        return
    p1 = cut_pic(p1)
    p2 = cut_pic(p2)
    p = np.concatenate([p1, p2], axis=1)
    return p


def debug_print(name, val):
    print(Fore.YELLOW + str(name) + ": " + str(val))


def get_vout(video_raw_name, width, height):
    logger.debug("video_raw_name: {}".format(video_raw_name))
    outcap = cv2.VideoWriter(
        '{}.mp4'.format(video_raw_name),
        cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 30, (width, height))
    return outcap


def get_vout_H264(video_raw_name):
    video_writer = iio.get_writer(
        "{}.mp4".format(video_raw_name), format="ffmpeg", mode="I", fps=30, codec="libx264", pixelformat="yuv420p",
        macro_block_size=None, ffmpeg_log_level="quiet", quality=5)
    return video_writer


def get_vout_H264_mp4(video_raw_name):
    video_writer = iio.get_writer(
        "{}.mp4".format(video_raw_name), format="ffmpeg", mode="I", fps=30, codec="libx264", pixelformat="yuv420p",
        macro_block_size=None, ffmpeg_log_level="quiet", quality=5)
    return video_writer


'''
descripttion: 对视频进行跳帧处理，原始视频fps = 45, 生成视频 fps = 30  45/30 = 3 / 2 
                可以处理非整数比值视频
param {*} all_frame 视频总的帧数
param {*} a = 3 化简后的分子
param {*} b = 2 化简后的分母
return {*} 跳帧序列
'''


def generate_skip_frame_sequence(all_frame, a, b):
    ret = {}
    i = 0
    while i < all_frame + 1:
        j = i
        k_finish = False
        while j < i + a:
            if not k_finish:
                k = j
                while k < j + b:
                    ret[k] = 1
                    k += 1
                j = k
                k_finish = True
            if j == i + a:
                break
            ret[j] = 0
            j += 1
        i = j
    # for k, v in ret.items():
    #     print(k, v)
    return ret


def infer2org_frame_map(skip_index: dict):
    '''

    Args:
        skip_index: 原始的skip_frame_index, 从下标0开始，如果为1，表示这一帧保留，如果为0，则不保留

    Returns:
        infer 以后的frame_index到原始视频的映射关系

    '''
    save_index = [key for key in skip_index if skip_index[key] == 1]
    return {save_index.index(i): i for i in save_index}


def video_info(path):
    '''

    Args:
        path: video path

    Returns:
        frame_num, fps, duration(ms), width, height
    '''
    cap = cv2.VideoCapture()
    cap.open(path)
    frame_num = int(cap.get(7))
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = int(frame_num / fps * 1000)
    return frame_num, fps, duration, width, height


def save_key_frame_image(video_path: str, frame_th: int, out_image_path: str):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_th)
    flag, frame = cap.read()
    if flag == False:
        print(Fore.RED + "error in cap.read")
        return
    cv2.imwrite(out_image_path, frame)
    return


def save_key_frame_image_by_group(video_path: str, frame_ths: list, out_image_paths: list):
    """
    按组对关键帧进行抽取并保存
    Args:
        video_path:
        frame_ths:
        out_image_paths:

    Returns:

    """
    assert len(frame_ths) == len(out_image_paths)
    cap = cv2.VideoCapture(video_path)
    all_frame = int(cap.get(7))
    for i in range(len(frame_ths)):
        if frame_ths[i] >= all_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, all_frame)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ths[i])
        flag, frame = cap.read()
        if flag == False:
            print(Fore.RED + "error in cap.read")
            return
            cv2.imwrite(out_image_paths[i], frame)
    return


def video_add_audio(video_path, audio_path="./resource/silence15ms.mp3"):
    # logger.debug(os.getcwd())
    """
    视频添加音轨：为支持ios的视频倍速播放问题，需要对生成的视频添加无声ying'gui
    Args:
        video_path:
        audio_path:

    Returns:

    """
    # 读取视频和音频文件
    logger.debug(video_path)
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    # 给视频添加音轨
    video_with_audio = video.set_audio(audio)

    # 输出视频文件路径
    output_video_path = video_path

    # 保存视频文件
    video_with_audio.write_videofile(output_video_path, codec='libx264')


def analysis_debug_info_on_image(info: list, org_video_path, frame_th: list, skeletons: list, out_dir, out_name_predix):
    """
    将用于debug的信息绘制到debug图片上：多个信息对应多张图片对应多个骨骼，1-1-1 的关系
    Args:
        info: debug 信息，
        org_video_path: 原始视频地址
        frame_th: 指定帧
        skeletons: 指定骨骼
        out_dir: 输出目录
        out_name_predix: 输出名字特定前缀

    Returns:

    """
    assert len(info) == len(frame_th) == len(skeletons)
    for i in range(len(info)):
        debug_image = debug_analysis_image(org_video_path, frame_th[i], skeletons[i], info[i])
        cv2.imwrite(os.path.join(out_dir, out_name_predix + f"debug{i}.png"), debug_image)


if __name__ == "__main__":
    A = [0, 0]
    B = [2, 0]
    C = [1, 1]
    print("A: ", angleof(A, B, C))
    print("B: ", angleof(B, A, C))
    print("C: ", angleof(C, B, A))

    # path = r"F:\pingpang-all-data\Video_Iphone_0110\视频切分_v1"
    # path = "F:/pingpang-all-data/Video_iPhone_0228/视频切分v1/01_02_07_08"
    # files = listdir(path)
    # print(len(files))
    # for file in files:
    #     cut_video(file)

    # get_part_of_video("F:/pingpang-all-data/Video_iPhone_0228/素材/01.MOV", 18010, 18137, "01-10279-10389", "F:/pingpang-all-data/Video_iPhone_0228/dev-null")
    # get_part_of_video("F:/pingpang-all-data/Video_iPhone_0228/素材/02.MOV", 20480, 20597, "02-20480-20597", "F:/pingpang-all-data/Video_iPhone_0228/dev-null")

    # cut_video(files[0])
    # cut_video(files[1])
    # cut_video(files[2])
    # cut_video(files[4])
    # i = 5
    #  while i < len(files):
    #     cut_video(files[i])
    #     i = i + 1

    '''
    ex_files = listdir(r"F:\pingpang-all-data\Video_Iphone_0110\正手动作标注结果v1")
    # print(ex_files)
    for e in ex_files:
        videos = get_video_from_anotation(e, ["转腰不足"])
        # print(Fore.BLUE + e)
        for v in videos:
            csv, poses, skes = generate_video_source(v)
            # print(csv, "\n", poses, "\n", skes)
            print(csv)
    '''

    '''
    path1 = r"F:\pingpang-all-data\Video_Iphone_0110\正手动作标注结果v1\模板动作标注(2398).xlsx"
    path2 = r"F:\pingpang-all-data\Video_Iphone_0110\正手动作标注结果v1\模板动作标注(2408).xlsx"
    v1 = get_video_from_anotation(path1, 0.2, None)
    v2 = get_video_from_anotation(path2, 0.2, None)
    print(v1)
    print(v2)
    '''
    '''
    path = r"F:\pingpang-all-data\Video_Iphone_0110\视频切分_v1\IMG_2417.txt"
    cut_video(path)
    '''
    '''
    line = lineOf([0, 2], [2, 0])
    print("line: ", line)
    distance, drop_foot = distance_point2line([4, 0], line)
    print("distance is: ", distance, "drop_foot: ", drop_foot)
    '''

    # video_path = r"F:\pingpang-all-data\邱薪竹_动作示范（3个）-0904-2022\01-getKeyPoints.avi"
    # output_dir = r"C:\Users\weiji\OneDrive - bupt.edu.cn\文档\宽广实验室\2022-正手动作分析论文\论文素材\关键点视频图像"
    # video2frames(video_path, output_dir)

    generate_skip_frame_sequence(157, 5, 2)
