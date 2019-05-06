#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : time_util.py
# @Author: wu gang
# @Date  : 2019/5/5
# @Desc  : 时间操作
# @Contact: 752820344@qq.com

import datetime as dt


class Timer:
    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))
