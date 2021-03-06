from __future__ import print_function
import datetime
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid

from PaddleFileWriter.paddle_file_writer import PaddleFileWriter
from PaddleFileWriter import paddleboard_utils as pbu


# Create PaddleFileWriter with log
timestamp_dir = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
train_fw = PaddleFileWriter('./logs/%s/train' % timestamp_dir)

train_lists = []

images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
conv_pool_1 = fluid.nets.simple_img_conv_pool(
    input=images,
    filter_size=5,
    num_filters=20,
    pool_size=2,
    pool_stride=2,
    act="relu")
conv_pool_2 = fluid.nets.simple_img_conv_pool(
    input=conv_pool_1,
    filter_size=5,
    num_filters=50,
    pool_size=2,
    pool_stride=2,
    act="relu")

predict = fluid.layers.fc(input=conv_pool_2, size=10, act="softmax")
cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(x=cost)
optimizer = fluid.optimizer.Adam(learning_rate=0.01)
optimizer.minimize(avg_cost)

accuracy = fluid.evaluator.Accuracy(input=predict, label=label)

BATCH_SIZE = 50
PASS_NUM = 3
train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=500),
    batch_size=BATCH_SIZE)

place = fluid.CPUPlace()
exe = fluid.Executor(place)

exe.run(fluid.default_startup_program())

# Print computation graph
train_fw.write_graph(pbu.convert_program_to_tf_graph_def(fluid.default_main_program()))

batch_id = 0
for pass_id in range(PASS_NUM):
    accuracy.reset(exe)

    for data in train_reader():
        img_data = np.array(map(lambda x: x[0].reshape([1, 28, 28]),
                                data)).astype("float32")
        y_data = np.array(map(lambda x: x[1], data)).astype("int64")
        y_data = y_data.reshape([BATCH_SIZE, 1])

        loss, acc = exe.run(fluid.default_main_program(),
                            feed={"pixel": img_data,
                                  "label": y_data},
                            fetch_list=[avg_cost] + accuracy.metrics)
        pass_acc = accuracy.eval(exe)
        print("pass_id=" + str(pass_id) + " acc=" + str(acc) + " pass_acc=" +
              str(pass_acc))
        # print loss, acc
        if loss < 10.0 and pass_acc > 0.9:
            # if avg cost less than 10.0 and accuracy is larger than 0.9, we think our code is good.
            exit(0)

        # Log training batch cost and error
        train_fw.write("cost", float(loss[0]), batch_id)
        train_fw.write("error", float(1.0-pass_acc[0]), batch_id)
        train_lists.append((loss, float(1.0-pass_acc[0])))

        best = sorted(train_lists, key=lambda list: float(list[0]))[0]
        acc = 100 - float(best[1]) * 100
        print('The training classification accuracy is %.2f%%' % acc)
        train_fw.write("accuracy", acc, batch_id)

        batch_id += 1

    pass_acc = accuracy.eval(exe)
    print("pass_id=" + str(pass_id) + " pass_acc=" + str(pass_acc))


exit(1)
