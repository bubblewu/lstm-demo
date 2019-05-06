#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train_model.py
# @Author: wu gang
# @Date  : 2019/5/6
# @Desc  : LSTM模型构建和训练，用于验证码识别（分类问题）
# @Contact: 752820344@qq.com

import tensorflow as tf

from tf.captcha import image_util
from tf.captcha.config import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 这是默认的显示等级，显示所有信息


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error


def build_model(x, y):
    """
    构建模型训练所需的配置
    :param x:
    :param y:
    :return:
    """
    # 权重。初始化一个num_units*n_classes（128*36）的矩阵，元素均值为0。
    weights = tf.Variable(tf.random_normal([num_units, n_classes]), name="weights")
    # 偏置。 初始值全为0且长度为n_classes（36）的变量
    biases = tf.Variable(tf.random_normal([n_classes]), name='biases')

    # 构建网络
    # 创建两层的lstm
    lstm_layers = [tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True) for _ in range(layer_num)]
    # 将lstm连接在一起
    # 通过MultiRNNCell类实现深层循环神经网络中每一个时刻的前向传播过程。
    # 其中layer_num表示有多少层，也就是从xt到ht需要经过多少个lstm结构。
    # lstm = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
    # mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm * layer_num)
    mlstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers, state_is_tuple=True)
    # cell的初始状态
    # 将lstm中的状态初始化全0数组，和其他神经网络类似，在优化循环神经网络时，每次也会使用一个batch的训练样本。
    init_state = mlstm_cell.zero_state(batch_size, tf.float32)
    # 每个cell的输出
    outputs = list()
    state = init_state
    with tf.variable_scope('RNN'):
        # 计算每一个时刻的前向传播结果
        # 理论上RNN可以处理任意长度的序列，但在训练时为了避免梯度消散但问题，会规定一个最大的序列长度time_steps。
        for step in range(time_steps):
            # 在第一个时刻声明lstm结构中使用的变量，在之后的时刻都需要复用之前定义好的变量
            if step > 0:
                tf.get_variable_scope().reuse_variables()
            # 这里的state保存了每一层 LSTM 的状态。
            # 每一步处理时间序列中的一个时刻。
            # 将当前输入current_input和前一时刻状态state传入定义的lstm结构，
            # 可以得到当前lstm结构的输出lstm_output和更新后的状态state
            current_input = x[:, step, :]
            (lstm_output, state) = mlstm_cell(x[:, step, :], state)
            outputs.append(lstm_output)
    # h_state = outputs[-1] #取最后一个cell输出

    # 计算输出层的第一个元素
    # 获取最后time-step的输出，使用全连接, 得到第一个验证码输出结果
    # 两个矩阵中对应元素各自相乘
    mat = tf.matmul(outputs[-4], weights)
    # 激活函数，返回一个Tensor，与logits(即 mat + biases)具有相同的类型和shape。
    prediction_1 = tf.nn.softmax(mat + biases)
    # 计算输出层的第二个元素, 输出第二个验证码预测结果
    prediction_2 = tf.nn.softmax(tf.matmul(outputs[-3], weights) + biases)
    # 计算输出层的第三个元素, 输出第三个验证码预测结果
    prediction_3 = tf.nn.softmax(tf.matmul(outputs[-2], weights) + biases)
    # 计算输出层的第四个元素, 输出第四个验证码预测结果,size:[batch,num_class]
    prediction_4 = tf.nn.softmax(tf.matmul(outputs[-1], weights) + biases)
    # 输出连接. 4 * [batch, num_class] => [batch, 4 * num_class]
    # 将张量沿一个维度串联.将张量值的列表与维度轴串联在一起.
    prediction_all = tf.concat([prediction_1, prediction_2, prediction_3, prediction_4], 1)
    # [4, batch, num_class] => [batch, 4, num_class]
    prediction_all = tf.reshape(prediction_all, [batch_size, captcha_num, n_classes], name='prediction_merge')

    # loss_function 计算当前时刻输出的损失
    loss = -tf.reduce_mean(y * tf.log(prediction_all), name='loss')
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_all,labels=y))
    # optimization 优化器
    optimization = tf.train.AdamOptimizer(learning_rate=learning_rate, name='opt').minimize(loss)
    # model evaluation 返回沿轴axis最大值的索引
    pre_arg = tf.argmax(prediction_all, 2, name='predict')
    y_arg = tf.argmax(y, 2)
    correct_prediction = tf.equal(pre_arg, y_arg)
    # 准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

    return optimization, loss, accuracy, pre_arg, y_arg


def train():
    """
    模型训练
    # 训练神经网络的三个步骤：
    # 1、定义神经网络的结构和前向传播的输出结果；
    # 2、定义损失函数以及选择反向传播优化的算法；
    # 3、生成会话(tf.Session)并且在训练数据上反复运行反向传播优化算法；
    """
    # tf.placeholder函数用于定义过程，不必指定初始值，可在运行时，通过 Session.run 的函数的 feed_dict 参数指定。返回Tensor张量类型
    # 赋值一般用sess.run(feed_dict={x: xs, y_: ys})，其中x, y_是用placeholder创建出来的
    x = tf.placeholder(
        # 数据类型。常用的是tf.float32,tf.float64等数值类型
        "float",
        # shape：数据形状。默认是None，就是一维值，也可以多维，比如：[None，3]，表示列是3，行不一定
        [None, time_steps, n_input],
        # 名称
        name="x")  # input image placeholder
    y = tf.placeholder("float", [None, captcha_num, n_classes], name="y")  # input label placeholder

    optimization, loss, accuracy, pre_arg, y_arg = build_model(x, y)
    # # 创建训练模型保存类
    saver = tf.train.Saver()
    # tf.global_variables_initializer()函数初始化所有变量,也会自动处理变量直接的依赖关系
    init = tf.global_variables_initializer()

    # 创建一个会话来运行TensorFlow程序。
    with tf.Session() as sess:
        sess.run(init)
        iter = 1
        # 训练迭代次数
        while iter < iteration:
            batch_x, batch_y = image_util.get_batch()
            # 只运行优化迭代计算图
            sess.run(optimization, feed_dict={x: batch_x, y: batch_y})
            if iter % 100 == 0:
                los, acc, parg, yarg = sess.run([loss, accuracy, pre_arg, y_arg], feed_dict={x: batch_x, y: batch_y})
                print("For iter ", iter)
                print("Accuracy ", acc)
                print("Loss ", los)
                if iter % 1000 == 0:
                    print("predict arg:", parg[0:10])
                    print("yarg :", yarg[0:10])
                print("__________________")
                # if acc > 0.95:
                #     print("training complete, accuracy:", acc)
                #     break
            # 保存模型
            if iter % 1000 == 0:
                saver.save(sess, model_path, global_step=iter)
            iter += 1
        # 计算验证集准确率
        valid_x, valid_y = image_util.get_batch(data_path=validation_path, is_training=False)
        print("Validation Accuracy:", sess.run(accuracy, feed_dict={x: valid_x, y: valid_y}))


if __name__ == '__main__':
    train()
