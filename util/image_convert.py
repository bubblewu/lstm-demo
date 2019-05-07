#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : image_convert.py
# @Author: wu gang
# @Date  : 2019/5/6
# @Desc  : 调整图片尺寸
# @Contact: 752820344@qq.com
import os

from PIL import Image


def get_images(input_path):
    """
    获取指定路径下的图片
    :param input_path: 存储图片的路径
    :return:
    """
    target_file_list = os.listdir(input_path)
    list_count = len(target_file_list)
    print("开始加载[%s]路径下的图片, 数量为: %s" % (input_path, list_count))
    return target_file_list, list_count


def adjust_size(in_file, output_file, width, height):
    """
    根据指定大小调整图片尺寸
    :param in_file: 输入图片地址
    :param output_file: 输出图片地址
    :param width: 宽
    :param height: 高
    :return:
    """
    im = Image.open(in_file)
    # (x, y) = im.size
    out = im.resize((width, height), Image.ANTIALIAS)
    out.save(output_file)
    # print('file {} original size {}*{}, adjust size: {}*{}'.format(in_file, x, y, width, height))


if __name__ == '__main__':
    path = '/Users/wugang/datasets/image/captcha/4/train_10000/'
    images, count = get_images(path)
    for i in range(count):
        input_file = path + images[i]
        out_file = '/Users/wugang/datasets/image/captcha/4/temp1/' + images[i]
        adjust_size(input_file, out_file, 80, 26)
        if (i != 0) & (i % 100 == 0):
            print("now process: %s" % i)
        elif i + 1 == count:
            print("now process: %s" % (i + 1))
