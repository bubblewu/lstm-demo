#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : yield_test.py
# @Author: wu gang
# @Date  : 2019/5/7
# @Desc  : 生成器yield函数生成斐波那契数列
# @Contact: 752820344@qq.com
import sys


def fibonacci(n):
    """
    使用yield实现斐波那契数列
    斐波那契数列数列从第3项开始,每一项都等于前两项之和。如：0 1 1 2 3 5 8 13 21 34 55
    :param n:
    :return:
    """
    a, b, counter = 0, 1, 0
    while True:
        if counter > n:
            return
        yield a
        a, b = b, a + b
        counter += 1


if __name__ == '__main__':
    """
    在python中使用了yield的函数被称为生成器函数，生成器函数是一个返回迭代器的函数，只能用于迭代操作。
    即生成器就是一个迭代器。 
    在调用生成器运行的过程中，每次遇到yield时函数会暂停并保存当前所有的运行信息，返回yield的值。
    并在下一次执行next()方法时从当前位置继续运行。
    """
    f = fibonacci(10)
    while True:
        try:
            print(next(f), end=" ")
        except StopIteration:
            sys.exit()
