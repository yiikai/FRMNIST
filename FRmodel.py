import tensorflow as tf
import numpy as np
import random


class FRModel:
    def __init__(self):
        self.input_a = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
        self.input_p = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
        self.input_n = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
        self.logits = None
        self.loss = None
        self.train_op = None

    def __base_model(self, x):
        """
        embedding基础模型，可以自行修改
        :param x: 输入原始图像数据 4dims
        :return: 模型output
        """
        out = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, padding='same')
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 1)
        out = tf.layers.conv2d(inputs=out, filters=32, kernel_size=3, padding='same')
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 1)
        out = tf.layers.dropout(out, rate=0.25)
        out = tf.layers.flatten(out)
        out = tf.layers.dense(inputs=out, units=128)
        out = tf.nn.l2_normalize(out, 1)
        return out

    def create_model(self):
        '''
        创建模型参数损失函数
        :return: 返回训练参数
        '''
        with tf.variable_scope("embeddings", initializer=tf.truncated_normal_initializer()) as embeding:
            out_a = self.__base_model(self.input_a)
        with tf.variable_scope(embeding, initializer=tf.truncated_normal_initializer(), reuse=True):
            out_p = self.__base_model(self.input_p)
        with tf.variable_scope(embeding, initializer=tf.truncated_normal_initializer(), reuse=True):
            out_n = self.__base_model(self.input_n)

        self.logits = tf.concat([out_a, out_p, out_n], axis=1)
        self.loss = self.__triplet_loss(self.logits, alpha=0.2)
        optimizer = tf.train.AdamOptimizer(1e-3)
        global_step = tf.train.get_global_step()
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)
        return self.logits, self.loss, self.train_op

    def __triplet_loss(self, logits, alpha):
        A = logits[:, 0:128]
        P = logits[:, 128:256]
        N = logits[:, 256:]
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(A, P)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(A, N)), 1)
        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), None)
        return loss

    def batch_hard(self, batch_size, x, y, anchorsize=1, postivesize=40):
        '''
        batch_hard进行数据划分（使用的是Facenet的方法，对所有的ap对进行计算）
        :param x: 训练sample
        :param y: 训练label
        :param anchorsize:锚
        :param postivesize: 同标签数据
        :return:
        '''
        classify = list(np.unique(y))
        print(type(classify))
        # 随机选出一个anchor
        slice = random.sample(classify, anchorsize)
        print("[info]:anchor label is: ", slice)
        class_len = len(classify)

        # 找到符合slice标签的所有数据的下标
        postive_indices = [np.where(y == slice)][0][0]

        # 从这些下标中随机选出postivesize个
        wanted_postive = random.sample(list(postive_indices), postivesize)
        wanted_anchor = wanted_postive[0]
        wanted_postive = wanted_postive[1:]

        # 从不是anchor的数据中随机挑选出negtive，组成mini-batch
        negetive_indices = [np.where(y != slice)][0][0]
        wanted_negetive = random.sample(list(negetive_indices), batch_size - postivesize)
        return self.create_pairs(x, wanted_anchor, wanted_postive, wanted_negetive)

    def create_pairs(self, x, anchor, postive, negtive):
        pairs = []
        for i in postive:
            for j in negtive:
                pairs += [[x[anchor], x[i], x[j]]]
        return np.array(pairs)

    def create_vaild_test_pairs(self, x, digit_indices, num_classes):
        """
        创建正例和负例的Pairs
        :param x: 数据
        :param digit_indices: 不同类别的索引列表
        :param num_classes: 类别
        :return: Triplet Loss 的 Feed 数据
        """

        pairs = []
        n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1  # 最小类别数
        print("mini size is :", n)
        for d in range(num_classes):
            for i in range(n):
                z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
                inc = random.randrange(1, num_classes)
                dn = (d + inc) % num_classes
                z3 = digit_indices[dn][i]
                pairs += [[x[z1], x[z2], x[z3]]]
        return np.array(pairs), n

    def train(self, x, y, batch_size, anchor=1, postive=40, epochs=100, model_dir='./checkpoint/model', save=False):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.train.write_graph(sess.graph_def, model_dir, name="FRMNIST.pbtxt")
            saver.save(sess, model_dir, global_step=100, write_meta_graph=False)
            for j in range(epochs):
                pairs = self.batch_hard(batch_size=batch_size, x=x, y=y, anchorsize=anchor, postivesize=postive)
                op, cost = sess.run([self.train_op, self.loss],
                                    feed_dict={self.input_a: pairs[:, 0],
                                               self.input_a: pairs[:, 1],
                                               self.input_n: pairs[:, 2]})
                print("cost is ", sess.run(tf.reduce_mean(cost)))

    def valid(self, x, y):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
            clz_size_te = len(np.unique(y))
            digit_indices_te = [np.where(y == i)[0] for i in range(clz_size_te)]
            trip_te_x, minnums = self.create_vaild_test_pairs(x, digit_indices_te, clz_size_te)
            pred_y = sess.run(self.logits, feed_dict={self.input_a: trip_te_x[:, 0],
                                                      self.input_p: trip_te_x[:, 1],
                                                      self.input_n: trip_te_x[:, 2]})
            self.__show_acc_facets(pred_y, minnums, clz_size_te)

    def __show_acc_facets(self, y_pred, n, clz_size):
        """
        展示模型的准确率
        :param y_pred: 测试结果数据组
        :param n: 数据长度
        :param clz_size: 类别数
        :return: 打印数据
        """
        print("[INFO] trainer - n_clz: %s" % n)
        for i in range(clz_size):
            print("[INFO] trainer - clz %s" % i)
            final = y_pred[n * i:n * (i + 1), :]
            anchor, positive, negative = final[:, 0:128], final[:, 128:256], final[:, 256:]
            print('anchor size is ', len(anchor))
            pos_dist = np.sum(np.square(anchor - positive), axis=-1, keepdims=True)
            neg_dist = np.sum(np.square(anchor - negative), axis=-1, keepdims=True)
            basic_loss = pos_dist - neg_dist
            r_count = basic_loss[np.where(basic_loss < 0)].shape[0]
            print("[INFO] trainer - distance - min: %s, max: %s, avg: %s" % (
                np.min(basic_loss), np.max(basic_loss), np.average(basic_loss)))
            print("[INFO] acc: %s" % (float(r_count) / float(n)))
            print("")
