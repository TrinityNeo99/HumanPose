import sys
import time
import os
import argparse
import csv
import shutil
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
from colorama import Fore, Back, Style, init

init(autoreset=True)
sys.path.append("./lib")
from common.utils import *
from common.draw import *

from lib.inference1 import get_final_preds
from lib.transforms1 import get_affine_transform
from lib.default import _C as cfg
from lib.default import update_config
import pose_hrnet
import pose_resnet
from tqdm import tqdm

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_person_detection_boxes(model, img, threshold=0.5):
    pil_image = Image.fromarray(img)  # Load the image
    transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
    transformed_img = transform(pil_image)  # Apply the transform to the image
    pred = model([transformed_img.to(CTX)])  # Pass the image to the model
    # Use the first detected person
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
    pred_scores = list(pred[0]['scores'].cpu().detach().numpy())

    person_boxes = []
    # Select box has score larger than threshold and is person
    for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes, pred_scores):
        if (pred_score > threshold) and (pred_class == 'person'):
            person_boxes.append(pred_box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)  # .unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)

    # compute output heatmap
    output = pose_model(model_inputs.to(CTX))
    coords, confidence = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))

    return coords, confidence


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--videoFile', type=str, required=False)
    parser.add_argument('--outputDir', type=str, default='/output/')
    parser.add_argument('--inferenceFps', type=int, default=25)
    parser.add_argument('--writeBoxFrames', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def initialize():
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    print("inference fps: ", args.inferenceFps)
    update_config(cfg, args)
    pose_dir = prepare_output_dirs(args.outputDir)

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()
    pose_model = eval(cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model.to(CTX)
    pose_model.eval()
    return args, box_model, pose_model, pose_transform


def main():
    args, box_model, pose_model, pose_transform = initialize()
    # files = listdir(r"F:\pingpong-all-data\2023-4-19_北体合作_动作示范视频_实验用小规模数据集") 
    files = listdir_full_path(r"C:\Users\weiji\Downloads\fineGym_test")
    files = listdir_full_path(r"C:\Users\weiji\Downloads\diving")
    # files = listdir(r"../video")
    print(files)
    for f in files:
        generate_kps(f, args, box_model, pose_model, pose_transform)
    # python inference.py --cfg inference-config.yaml --videoFile ../video/IMG_2411_153_327.mp4 --writeBoxFrames --outputDir F:\pingpang-all-data\Video_iPhone_0311\关键点提取结果\IMG_3114  TEST.MODEL_FILE pose_hrnet_w32_256x192.pth 


def generate_kps(video_path, args, box_model, pose_model, pose_transform):
    csv_output_rows = []
    # Loading an video
    args.videoFile = video_path
    name = video_path.split("\\")[-1].replace(".mp4", "")
    print("the name is", name)

    out_dir = os.path.join(args.outputDir, name)
    if os.path.exists(out_dir) and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
        print(Fore.RED + "delete " + out_dir)
    os.makedirs(out_dir, exist_ok=True)

    pose_out_dir = os.path.join(out_dir, "pose")
    if os.path.exists(pose_out_dir) and os.path.isdir(pose_out_dir):
        shutil.rmtree(pose_out_dir)
        print(Fore.RED + "delete " + pose_out_dir)
    os.makedirs(pose_out_dir, exist_ok=True)

    ske_out_dir = os.path.join(out_dir, "ske")
    if os.path.exists(ske_out_dir) and os.path.isdir(ske_out_dir):
        shutil.rmtree(ske_out_dir)
        print(Fore.RED + "delete " + ske_out_dir)
    os.makedirs(ske_out_dir, exist_ok=True)

    print(Fore.CYAN + out_dir)
    print(Fore.CYAN + pose_out_dir)

    vidcap = cv2.VideoCapture(args.videoFile)
    print(args.videoFile)
    frame_num, fps, duration, width, height = video_info(args.videoFile)
    if fps < args.inferenceFps:
        print('desired inference fps is ' + str(args.inferenceFps) + ' but video fps is ' + str(fps))
        exit()
    print("the fps is ", fps, " the inferenceFps is: ", args.inferenceFps)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outcap = cv2.VideoWriter(
        '{}/{}_pose.avi'.format(out_dir, os.path.splitext(os.path.basename(args.videoFile))[0]),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(10), (frame_width, frame_height))

    count = 0
    ret, image_bgr = vidcap.read()
    pose_preds_frames = np.zeros((1, frame_num, 17, 2))
    confidence_frames = np.zeros((1, frame_num, 17, 1))
    with tqdm(total=frame_num) as pbar:
        while ret:
            total_now = time.time()
            ret, image_bgr = vidcap.read()
            count += 1

            if not ret:
                continue

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            # Clone 2 image for person detection and pose estimation
            if cfg.DATASET.COLOR_RGB:
                image_per = image_rgb.copy()
                image_pose = image_rgb.copy()
            else:
                image_per = image_bgr.copy()
                image_pose = image_bgr.copy()

            # Clone 1 image for debugging purpose
            image_debug = image_bgr.copy()

            # object detection box
            now = time.time()
            pred_boxes = get_person_detection_boxes(box_model, image_per, threshold=0.9)
            then = time.time()
            # print("Find person bbox in: {} sec".format(then - now))

            # Can not find people. Move to next frame
            if not pred_boxes:
                count += 1
                continue

            sig_box = []
            max_box = pred_boxes[0]
            for b in pred_boxes:
                if squareof(b[0], b[1]) > squareof(max_box[0], max_box[1]):
                    max_box = b
            sig_box.append(max_box)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

            if args.writeBoxFrames:
                i = 0
                for box in sig_box:
                    # print("box:", box, end=" ")
                    # print(lenof(box[0], box[1]))
                    box[0] = (int(box[0][0]), int(box[0][1]))
                    box[1] = (int(box[1][0]), int(box[1][1]))
                    cv2.rectangle(image_debug, box[0], box[1], color=colors[i % 4],
                                  thickness=3)  # Draw Rectangle with the coordinates
                    cv2.putText(image_debug, str(i), box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    i = i + 1

            # pose estimation : for multiple people
            centers = []
            scales = []
            for box in sig_box:
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                centers.append(center)
                scales.append(scale)

            now = time.time()
            pose_preds, confidence = get_pose_estimation_prediction(pose_model, image_pose, centers, scales,
                                                                    transform=pose_transform)
            then = time.time()
            # print("Find person pose in: {} sec".format(then - now))
            # print(pose_preds)
            pose_preds_frames[0, count - 1, :, :] = pose_preds[0, :, :]
            confidence_frames[0, count - 1, :, :] = confidence[0, :, :]

            new_csv_row = []

            # draw points on image
            for coords in pose_preds:
                # Draw each point on image
                new_csv_row.extend([count])
                for coord in coords:
                    x_coord, y_coord = int(coord[0]), int(coord[1])
                    cv2.circle(image_debug, (x_coord, y_coord), 4, (255, 0, 0), 2)
                    new_csv_row.extend([x_coord, y_coord])

            total_then = time.time()

            text = "{:03.2f} sec".format(total_then - total_now)
            cv2.putText(image_debug, text, (100, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)

            # cv2.imshow("pos", image_debug)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            csv_output_rows.append(new_csv_row)
            img_file = os.path.join(pose_out_dir, 'pose_{:08d}.jpg'.format(count))
            img_file_ske = os.path.join(ske_out_dir, 'ske_{:08d}.jpg'.format(count))
            # cv2.imwrite(img_file, image_debug)
            # cv2.imencode('.jpg', image_debug)[1].tofile(img_file)
            # draw_skeleton_kps(pose_preds[0], img_file_ske)
            image_debug = draw_skeleton_kps_on_org(pose_preds[0], image_debug)

            # print(Fore.CYAN +img_file)
            outcap.write(image_debug)
            pbar.update(1)

    # write csv
    csv_headers = ['frame']
    for keypoint in COCO_KEYPOINT_INDEXES.values():
        csv_headers.extend([keypoint + '_x', keypoint + '_y'])

    csv_output_filename = os.path.join(out_dir, 'pose-data.csv')
    print(Fore.YELLOW + "csv filename is " + csv_output_filename)
    print(csv_output_filename)
    with open(csv_output_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(csv_headers)
        csvwriter.writerows(csv_output_rows)

    vidcap.release()
    outcap.release()
    cv2.destroyAllWindows()


    return del_zero_array(pose_preds_frames), del_zero_array(confidence_frames)


def del_zero_array(array):
    zero_subarrays_mask = np.all(array == 0, axis=(2,3))
    filtered_array = array[~zero_subarrays_mask]
    return np.array([filtered_array])


if __name__ == '__main__':
    main()

    # python inference.py --cfg inference-config.yaml --videoFile ../video/IMG_2411_153_327.mp4 --writeBoxFrames --outputDir ./output  TEST.MODEL_FILE pose_hrnet_w32_256x192.pth
