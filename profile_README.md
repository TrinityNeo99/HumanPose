# 2D Human Pose Estimation Profile Test

## Test environment
- server: 余浩服务器
- GPU： 2080Ti
- Data: /root/2DHPE/data/pingpong-109-coco-mini
  - 5 videos: 856 frames
  - fps: 60

result

| Model      | Year  | Data | Accuracy | CPU(%)  | Graph Memory(G) | GPU util(%) | Time Cost(s) |FPS/file| Other |
|------------|-------|------|----------|---------|-----------------|-------------|--------------|-------|-------|
| HRnet      | 2019  |      |          | 904     | 2.52            | 35          | 190.01       |       |       |
| MediaPipe  | 2022  |      |          | 447     | 0               | 0           | 51.06        |       |       |
