import os
import datetime
import numpy as np
import paddle.v2 as paddle
from PaddleFileWriter.paddleFileWriter import PaddleFileWriter

def softmax_regression(img):
    predict = paddle.layer.fc(
        input=img, size=10, act=paddle.activation.Softmax())
    return predict


def multilayer_perceptron(img):
    # The first fully-connected layer
    hidden1 = paddle.layer.fc(input=img, size=128, act=paddle.activation.Relu())
    # The second fully-connected layer and the according activation function
    hidden2 = paddle.layer.fc(
        input=hidden1, size=64, act=paddle.activation.Relu())
    # The thrid fully-connected layer, note that the hidden size should be 10,
    # which is the number of unique digits
    predict = paddle.layer.fc(
        input=hidden2, size=10, act=paddle.activation.Softmax())
    return predict


def convolutional_neural_network(img):
    # first conv layer
    conv_pool_1 = paddle.networks.simple_img_conv_pool(
        input=img,
        filter_size=5,
        num_filters=20,
        num_channel=1,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # second conv layer
    conv_pool_2 = paddle.networks.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        num_channel=20,
        pool_size=2,
        pool_stride=2,
        act=paddle.activation.Relu())
    # fully-connected layer
    predict = paddle.layer.fc(
        input=conv_pool_2, size=10, act=paddle.activation.Softmax())
    return predict


def main():
    paddle.init(use_gpu=False, trainer_count=1)

    # define network topology
    images = paddle.layer.data(
        name='pixel', type=paddle.data_type.dense_vector(784))
    label = paddle.layer.data(
        name='label', type=paddle.data_type.integer_value(10))

    # Here we can build the prediction network in different ways. Please
    # choose one by uncomment corresponding line.
    # predict = softmax_regression(images)
    # predict = multilayer_perceptron(images)
    predict = convolutional_neural_network(images)

    cost = paddle.layer.classification_cost(input=predict, label=label)

    parameters = paddle.parameters.create(cost)

    optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1 / 128.0,
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 * 128))

    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)

    train_lists = []
    test_lists = []

    timestamp_dir = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    train_fw = PaddleFileWriter('./logs/%s/train' % timestamp_dir)
    test_fw = PaddleFileWriter('./logs/%s/test' % timestamp_dir)

    def event_handler_train(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 10 == 0:
                print "Train Data: Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
                train_fw.write("cost", event.cost, event.batch_id)
                train_fw.write("error", event.metrics['classification_error_evaluator'], event.batch_id)
                train_lists.append((event.pass_id, event.cost,
                              event.metrics['classification_error_evaluator']))

                best = sorted(train_lists, key=lambda list: float(list[1]))[0]
                accuracy = 100 - float(best[2]) * 100
                print 'The training classification accuracy is %.2f%%' % accuracy
                train_fw.write("accuracy", accuracy, event.batch_id)

    def event_handler_test(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 10 == 0:
                print "Test Data: Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
                test_fw.write("cost", event.cost, event.batch_id)
                test_fw.write("error", event.metrics['classification_error_evaluator'], event.batch_id)
                test_lists.append((event.pass_id, event.cost,
                                    event.metrics['classification_error_evaluator']))

                best = sorted(test_lists, key=lambda list: float(list[1]))[0]
                accuracy = 100 - float(best[2]) * 100
                print 'The training classification accuracy is %.2f%%' % accuracy
                test_fw.write("accuracy", accuracy, event.batch_id)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=8192),
            batch_size=128),
        event_handler=event_handler_train,
        num_passes=1)

    trainer.train(
        reader=paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.test(), buf_size=8192),
            batch_size=128),
        event_handler=event_handler_test,
        num_passes=1)

    # def load_image(file):
    #     im = Image.open(file).convert('L')
    #     im = im.resize((28, 28), Image.ANTIALIAS)
    #     im = np.array(im).astype(np.float32).flatten()
    #     im = im / 255.0
    #     return im
    #
    # test_data = []
    # cur_dir = os.path.dirname(os.path.realpath(__file__))
    # test_data.append((load_image(cur_dir + '/image/infer_3.png'), ))
    #
    # probs = paddle.infer(
    #     output_layer=predict, parameters=parameters, input=test_data)
    # lab = np.argsort(-probs)  # probs and lab are the results of one batch data
    # print "Label of image/infer_3.png is: %d" % lab[0][0]


if __name__ == '__main__':
    main()
