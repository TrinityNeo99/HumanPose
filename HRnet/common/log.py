'''
Descripttion: 处理日志模块
Author: Wei Jiangning
version: v1.1
Date: 2022-12-09 23:28:52
LastEditors: Wei Jiangning
LastEditTime: 2023-06-13 15:24:51
'''
import logging
from datetime import datetime

# 记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 处理器
# 1.标准输出
sh = logging.StreamHandler()
sh.setLevel(logging.INFO)
# 2.文件输出
log_path = f"./{str(datetime.now()).replace(' ', '_')}_process.log"
fh = logging.FileHandler(filename="extract.log", mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
# 格式器
fmt = logging.Formatter(fmt="%(asctime)s - %(levelname)-2s - %(filename)-2s : %(lineno)s line - %(message)s")
# 给处理器设置格式
sh.setFormatter(fmt)
fh.setFormatter(fmt)
# 记录器设置处理器
logger.addHandler(sh)
logger.addHandler(fh)
