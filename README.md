# HumanPose
- Extract human pose and better than mmcv or mmdet trash
- The extracted data are in [pyskl format](https://github.com/kennymckormick/pyskl/tree/main/tools/data)
- Including keypoint x, y and confidence
- Easy to use, free of mmcv, mmdet packages
## HRnet Usage
- Download HRnet pretrained model `pose_hrnet_w32_256*192.pth` in[Baidu Drive](https://pan.baidu.com/s/1RHJfjatYaZ2j4kVnhvlYPw?pwd=hxn8), and move it in `./resource`
- Modify the data source path in `extract_human_pose.py`: which only includes your raw video;
- Modify the output dir in `extract_human_pose.py`, where the output annotations.pkl is saved;
- Modify the label list in `extract_human_pose.py`, which is the filename to its ground label, as following;
  - name1.mp4 0
  - name2.mp4 1
  - name3.mp4 2
  - ...
  - nameN.mp4 n
- `python extract_human_pose.py --cfg ./config/inference-config.yaml --writeBoxFrames --outputDir ./output  TEST.MODEL_FILE ./resource/pose_hrnet_w32_256x192.pth`


## Mediapipe Usage
