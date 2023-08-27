'''
Descripttion: 自己定义的异常类
Author: Wei Jiangning
version: v 1.0
Date: 2022-12-09 23:15:56
LastEditors: Wei Jiangning
LastEditTime: 2022-12-10 11:52:15
'''
class VideoRead(Exception):
    def __init__(self, info):
        self.info = info
        

class VideoNoPerson(Exception):
    def __init__(self, info):
        self.info = info