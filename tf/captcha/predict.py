#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : predict.py
# @Author: wu gang
# @Date  : 2019/5/6
# @Desc  : 根据训练出的模型对验证码进行分类识别
# @Contact: 752820344@qq.com

import numpy as np
import tensorflow as tf

from tf.captcha import image_util
from tf.captcha.config import *


def get_test_set():
    # 获取测试集路径下的所有文件
    target_file_list = os.listdir(test_data_path)
    print("预测的验证码文件:", len(target_file_list))

    # 判断条件
    # 计算待检测验证码个数能被batch size 整除的次数
    flag = len(target_file_list) // batch_size
    # 共有多少个batch
    batch_len = flag if flag > 0 else 1
    # 计算验证码被batch size整除后的取余
    flag2 = len(target_file_list) % batch_size
    # 若不能整除，则batch数量加1
    batch_len = batch_len if flag2 == 0 else batch_len + 1

    print("共生成batch数:", batch_len)
    print("验证码根据batch取余:", flag2)

    batch = np.zeros([batch_len * batch_size, time_steps, n_input])
    for i, file in enumerate(target_file_list):
        batch[i] = image_util.open_image(test_data_path + '/' + file)
    batch = batch.reshape([batch_len, batch_size, time_steps, n_input])
    return batch, target_file_list  # batch_file_name


def write_to_file(predict_list, file_list):
    with open(output_path, 'a') as f:
        for i, res in enumerate(predict_list):
            if i == 0:
                f.write("id\tfile\tresult\n")
            f.write(str(i) + "\t" + file_list[i] + "\t" + res + "\n")
    print("预测结果保存在：", output_path)


def predict(lstm_model=model, checkpoint=model_checkpoint_path):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(lstm_model)
        # 读取已训练模型
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint))
        # 获取原始计算图，并读取其中的tensor
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        y = graph.get_tensor_by_name("y:0")
        pre_arg = graph.get_tensor_by_name("predict:0")
        # 获取测试集
        test_x, file_list = get_test_set()
        predict_result = []
        for i in range(len(test_x)):
            batch_test_x = test_x[i]
            # 创建空的y输入
            batch_test_y = np.zeros([batch_size, captcha_num, n_classes])
            test_predict = sess.run([pre_arg], feed_dict={x: batch_test_x, y: batch_test_y})
            # print(test_predict)
            # predict_result.extend(test_predict)

            # 将预测结果转换为字符
            for line in test_predict[0]:
                character = ""
                for each in line:
                    character += image_util.index2char(each)
                predict_result.append(character)

        # 预测结果
        predict_result = predict_result[:len(file_list)]
        # 保存到文件
        write_to_file(predict_result, file_list)


if __name__ == '__main__':
    predict()
