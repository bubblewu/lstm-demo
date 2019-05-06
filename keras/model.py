#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: wu gang
# @Date  : 2019/5/5
# @Desc  : 基于keras的LSTM模型
# @Contact: 752820344@qq.com

from keras.models import Sequential


class Model:
    """构建LSTM模型和预测"""

    def __init__(self):
        self.model = Sequential()  # Sequential 顺序模型
        self.configs = ''
