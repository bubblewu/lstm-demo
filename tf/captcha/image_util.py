#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : image_util.py
# @Author: wu gang
# @Date  : 2019/5/6
# @Desc  : 图像处理类
# @Contact: 752820344@qq.com

import numpy as np
import random
from PIL import Image

from tf.captcha.config import *


def get_all_file(paths=captcha_path):
    """
    获取路径下的子文件夹，并加载文件内容
    """
    all_target_file_list = []
    # root当前目录路径
    # dirs当前路径下所有子目录
    # files当前路径下所有非目录子文件
    for root, dirs, files in os.walk(paths):
        if len(files) > 1:
            print('处理{}路径下文件'.format(root))
            target_file_list = os.listdir(root)
            all_target_file_list.append(target_file_list)
    print('文件夹数量为: {}'.format(len(all_target_file_list)))


# def get_batch(data_path=captcha_path, is_training=True):
#     """
#     加载指定路径下的文件信息
#     """
#     # 读取路径下的所有文件名
#     target_file_list = os.listdir(data_path)
#     # 确认batch 大小
#     batch = batch_size if is_training else len(target_file_list)
#     # batch 数据
#     batch_x = np.zeros([batch, time_steps, n_input])
#     # batch 标签
#     batch_y = np.zeros([batch, captcha_num, n_classes])
#     for i in range(batch):
#         # 确认要打开的文件名
#         file_name = random.choice(target_file_list) if is_training else target_file_list[i]
#         img = Image.open(data_path + '/' + file_name)  # 打开图片
#         img = np.array(img)
#         if len(img.shape) > 2:
#             # 转换成灰度图像:(26,80,3) =>(26,80)
#             img = np.mean(img, -1)
#             # 标准化，为了防止训练集的方差过大而导致的收敛过慢问题。
#             img = img / 255
#             # img = np.reshape(img,[time_steps,n_input])  #转换格式：(2080,) => (26,80)
#         batch_x[i] = img
#
#         label = np.zeros(captcha_num * n_classes)
#         for num, char in enumerate(file_name.split('.')[0]):
#             index = num * n_classes + char2index(char)
#             try:
#                 label[index] = 1
#             except:
#                 print(file_name)
#
#         label = np.reshape(label, [captcha_num, n_classes])
#         batch_y[i] = label
#     return batch_x, batch_y


def get_batch(data_path=captcha_path, is_training=True):
    target_file_list = os.listdir(data_path)  # 读取路径下的所有文件名

    batch = batch_size if is_training else len(target_file_list)  # 确认batch 大小
    batch_x = np.zeros([batch, time_steps, n_input])  # batch 数据
    batch_y = np.zeros([batch, captcha_num, n_classes])  # batch 标签

    for i in range(batch):
        file_name = random.choice(target_file_list) if is_training else target_file_list[i]  # 确认要打开的文件名
        img = Image.open(data_path + '/' + file_name)  # 打开图片
        img = np.array(img)
        if len(img.shape) > 2:
            img = np.mean(img, -1)  # 转换成灰度图像:(26,80,3) =>(26,80)
            img = img / 255  # 标准化，为了防止训练集的方差过大而导致的收敛过慢问题。
            # img = np.reshape(img,[time_steps,n_input])  #转换格式：(2080,) => (26,80)
        batch_x[i] = img

        label = np.zeros(captcha_num * n_classes)
        dd = enumerate(file_name.split('.')[0])
        for num, char in dd:
            ind = char2index(char)
            c = num * n_classes
            index = n_classes + ind
            # label[index] = 1
            try:
                label[index] = 1
            except:
                print(file_name)
        label = np.reshape(label, [captcha_num, n_classes])
        batch_y[i] = label
    return batch_x, batch_y


def char2index(c):
    k = ord(c)
    index = -1
    # 数字索引
    if 48 <= k <= 57:
        index = k - 48
    # 大写字母索引
    if 65 <= k <= 90:
        index = k - 55
    # 小写字母索引
    if 97 <= k <= 122:
        index = k - 61
    if index == -1:
        raise ValueError('No Map')
    return index


def index2char(k):
    # k = chr(num)
    index = -1
    # 数字索引
    if 0 <= k < 10:
        index = k + 48
    # 大写字母索引
    if 10 <= k < 36:
        index = k + 55
    # 小写字母索引
    if 36 <= k < 62:
        index = k + 61
    if index == -1:
        raise ValueError('No Map')
    return chr(index)


if __name__ == '__main__':
    # get_batch()
    print(os.getcwd())
    paths = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    print(paths)
    print(os.path.split(os.path.abspath(__file__))[0])