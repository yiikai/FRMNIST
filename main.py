import FRmodel as fr
import LoadData as ld
import numpy as np
import random
import tensorflow as tf
from sklearn import cross_validation

x_train, y_train = ld.mnist_data().load_train()
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x_train, y_train, test_size=0.2, random_state=0)

FR = fr.FRModel()
FR.create_model()

#FR.train(x_train,y_train, 100, 1,40,100)
FR.valid(x_test,y_test)
# # create triplet pair data
#
# def create_pairs(x, digit_indices, num_classes):
#     """
#     创建正例和负例的Pairs
#     :param x: 数据
#     :param digit_indices: 不同类别的索引列表
#     :param num_classes: 类别
#     :return: Triplet Loss 的 Feed 数据
#     """
#
#     pairs = []
#     n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1  # 最小类别数
#     for d in range(num_classes):
#         for i in range(n):
#             z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
#             inc = random.randrange(1, num_classes)
#             dn = (d + inc) % num_classes
#             z3 = digit_indices[dn][i]
#             pairs += [[x[z1], x[z2], x[z3]]]
#     return np.array(pairs)
#
#
# clz_size = len(np.unique(y_train))
# print("clz_size is :", clz_size)
# digit_indices = [np.where(y_train == i)[0] for i in range(clz_size)]
# trip_x = create_pairs(x_train, digit_indices, clz_size)
#
# clz_size_te = len(np.unique(y_test))
# digit_indices_te = [np.where(y_test == i)[0] for i in range(clz_size_te)]
# trip_te_x = create_pairs(x_test, digit_indices_te, clz_size_te)
#
# print(trip_x.shape)
# FR.train(trip_x, 34, 2)

#
#
# anc_input = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
# pos_input = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
# neg_input = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
#
#
# def base_mode(x, reuse):
#     out = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, padding='same')
#     out = tf.nn.relu(out)
#     out = tf.layers.max_pooling2d(out, 2, 1)
#     out = tf.layers.conv2d(inputs=out, filters=32, kernel_size=3, padding='same')
#     out = tf.nn.relu(out)
#     out = tf.layers.max_pooling2d(out, 2, 1)
#     out = tf.layers.dropout(out, rate=0.25)
#     out = tf.layers.flatten(out)
#     out = tf.layers.dense(inputs=out, units=128)
#     out = tf.nn.l2_normalize(out, 1)
#     return out
#
#
# from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, K, merge, Concatenate
#
#
# def triplet_loss(y_pred, alpha):
#     A = y_pred[:, 0:128]
#     P = y_pred[:, 128:256]
#     N = y_pred[:, 256:]
#     pos_dist = K.sum(K.square(A - P), axis=-1, keepdims=True)
#     neg_dist = K.sum(K.square(A - N), axis=-1, keepdims=True)
#     basic_loss = pos_dist - neg_dist + alpha
#
#     loss = K.maximum(basic_loss, 0.0)
#
#     print("[INFO] model - triplet_loss shape: %s" % str(loss.shape))
#     return loss
#     # pos_dist = tf.reduce_sum(tf.square(tf.subtract(A, P)), 1)
#     # neg_dist = tf.reduce_sum(tf.square(tf.subtract(A, N)), 1)
#     # basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
#     # loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), None)
#     # return loss
#
#
# with tf.variable_scope('embedding', reuse=None) as embscope:
#     anc_out = base_mode(anc_input,None)
#     print('anc_out shape:',anc_out.shape)
# with tf.variable_scope(embscope, reuse=True):
#     pos_out = base_mode(pos_input,True)
#     print('pos_out shape:', pos_out.shape)
# with tf.variable_scope(embscope,reuse=True):
#     neg_out = base_mode(neg_input,True)
#     print('neg_out shape:',neg_out.shape)
#
# logits = tf.concat([anc_out,pos_out,neg_out],1)
# print("logits shape:",logits.shape)
# with tf.variable_scope('TRIPLET_LOSS'):
#     loss = triplet_loss(logits, 0.2)
#
# optimizer = tf.train.AdamOptimizer(1e-3)
# global_step = tf.train.get_global_step()
# train_op = optimizer.minimize(loss, global_step=global_step)
#
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# sess = tf.Session()
# summary_writer = tf.summary.FileWriter("log", sess.graph)
# sess.run(tf.global_variables_initializer())
# # import matplotlib
# # print('trip x is size: ',trip_x.shape)
# # matplotlib.use('TkAgg')
# # import matplotlib.pyplot as plt
# # import pylab
# # for j in range(100):
# #     out_a = sess.run(anc_out, feed_dict={anc_input: trip_x[j * 55:j * 55 + 55, 0]})
# #     out_p = sess.run(pos_out,feed_dict={pos_input: trip_x[j * 55:j * 55 + 55, 1]})
# #     out_N = sess.run(neg_out,feed_dict={neg_input: trip_x[j * 55:j * 55 + 55, 2]})
# #     loss = sess.run(triplet_loss(out_a,out_p,out_N,0.2))
# #     print(loss)
# for i in range(2):
#     for j in range(100):
#         _, cost = sess.run([train_op, loss],
#                            feed_dict={anc_input: trip_x[j * 55:j * 55 + 55, 0],
#                                       pos_input: trip_x[j * 55:j * 55 + 55, 1],
#                                       neg_input: trip_x[j * 55:j * 55 + 55, 2]})
#         if j % 10 ==0:
#             print("cost is ", sess.run(tf.reduce_mean(cost)))
# print("Train over====")
# pred_y = sess.run(logits,feed_dict={anc_input: trip_te_x[:,0],
#                                       pos_input: trip_te_x[:,1],
#                                       neg_input: trip_te_x[:, 2]})
#
# def show_acc_facets(y_pred, n, clz_size):
#     """
#     展示模型的准确率
#     :param y_pred: 测试结果数据组
#     :param n: 数据长度
#     :param clz_size: 类别数
#     :return: 打印数据
#     """
#     print("[INFO] trainer - n_clz: %s" % n)
#     for i in range(clz_size):
#         print("[INFO] trainer - clz %s" % i)
#         final = y_pred[n * i:n * (i + 1), :]
#         anchor, positive, negative = final[:, 0:128], final[:, 128:256], final[:, 256:]
#
#         pos_dist = np.sum(np.square(anchor - positive), axis=-1, keepdims=True)
#         neg_dist = np.sum(np.square(anchor - negative), axis=-1, keepdims=True)
#         basic_loss = pos_dist - neg_dist
#         r_count = basic_loss[np.where(basic_loss < 0)].shape[0]
#         print("[INFO] trainer - distance - min: %s, max: %s, avg: %s" % (np.min(basic_loss), np.max(basic_loss), np.average(basic_loss)))
#         print("[INFO] acc: %s" % (float(r_count) / float(n)))
#         print("")
#
#
# show_acc_facets(pred_y, pred_y.shape[0] // clz_size_te, clz_size_te)
