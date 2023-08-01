from cmath import cos, pi, sqrt
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from colorama import Fore, Back, Style, init
init(autoreset = True)
import pandas as pd
import shutil
import random
import time
from scipy import signal
# import h5py

def lenof(A, B):
    return math.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)


def angleof(A, B, C):
    c = lenof(A, B)
    a = lenof(B, C)
    b = lenof(A, C)
    consinA = (c**2 + b**2 - a**2) / (2*b*c)
    return math.acos(consinA) * 180 / pi


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
    img = img[0: h, 0: int(w/2)]
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
    fontstyle = ImageFont.truetype(sys_font, size, encoding="utf-8")
    draw.text(pos, text, color, font=fontstyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR) # 转换为opencv 格式

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

def cut_video(cut_basis):
    # path = "F:\pingpang-all-data\Video_Iphone_0110\视频切分_v1\IMG_2400.txt"
    name = cut_basis.split("\\")[-1].replace('.txt',"")
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
        
    pass

def get_part_of_video(video, start, end, name, outdir):
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    all_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("fps: ", fps)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    name = name + "_" + str(start) + "_" + str(end) + ".mp4"
    vout = cv2.VideoWriter(os.path.join(outdir, name), fourcc, fps, size)

   
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
            pass
        vout.write(frame)
        i = i + 1 


def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    return cv_img

''' 
anotation file: 24xx.excel
select = ["引拍过高"， "其实重心过高"]
    
'''
'''
descripttion: 
param {*} excel_file 标注的excel文件
param {*} nomal_ratio 旧参数，不用
param {*} select 需要挑选出的项目， 为None时挑出完全正确的动作
param {*} inverse 反转，即跳出非 select 的项目
return {*}
'''
def get_video_from_anotation(excel_file, nomal_ratio=0.2, select=None, inverse=False):
    # print(Fore.YELLOW + excel_file)
    df = pd.read_excel(excel_file)
    # print(df.head())

    # 去除不好的数据
    df = df[(df["备注"] != "废动作") & (df["备注"] != "反手") & (df["备注"] != "废动作（反手）")]
    
    # 选出错误动作数据
    if select is not None:
        if inverse == False:
            print("判决项目：", select[0])
            df = df[df[select[0]] == 1]
            video_name = list(df['name'])
        else:
            print("判决项目：非", select[0])
            df = df[df[select[0]] == 0]
            video_name = list(df['name'])
            
    # 全部正常动作
    else:
        video_name_all = []
        video_name = []
        for tup in df.itertuples():
            tup = list(tup)
            if 1.0 in tup:
                continue
            else:
                video_name_all.append(tup[1])
        
        video_name = video_name_all
        # num_all = len(video_name_all)
        # num = int(nomal_ratio * num_all)
        # for n in range(num):
        #     video_idx = random.randint(0, num_all - 1)
        #     video_name.append(video_name_all[video_idx])
        # print("all video is ", num_all)
        # print("ratio video is ", len(video_name))

    return video_name


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
    print("TN:", TN)
    print("FP:", FP)
    print("TP:", TP)
    print("FN:", FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = (2*precision*recall) / (precision + recall)
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
    k =  - (1 / line[0])
    b  = point[1] - k * point[0]
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

def read_kpss(file):
    df = pd.read_csv(file)
    # print(df.head())
    kpss = []
    for i, r in df.iterrows():
        kps = []
        for j in range(1, 35, 2):
            # print(j)
            kps.append([r[j], r[j+1]])
            if j == 33:
                break
        kpss.append(kps)
    return kpss


def fliter_low_pass(data):
    b, a = signal.butter(8, 0.99, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data)  #data为要过滤的信号
    return filtedData


def read_h5(path):
    f = h5py.File(path,'r')               #打开h5文件
    # f.keys()    
    print(f.keys())                        #可以查看所有的主键
    a = f['3D_positions'][:]                    #取出主键为data的所有的键值
    # print(a)
    print(len(a))
    for item in a:
        print("len: ", len(item))
        print(item)
        for k in item:
            print(k)
    f.close()


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


    path = r"C:\Users\neo\Downloads\Smoking.h5"
    print("hello")
    read_h5(path)