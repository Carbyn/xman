import paddle.v2 as paddle
import math


def fc_net(dict_dim,
           class_num=2,
           hidden_layer_sizes=[2048, 1024, 512, 128, 64, 16, 8],
           is_infer=False):

    data = paddle.layer.data("video",
                             paddle.data_type.dense_vector_sequence(dict_dim))
    if not is_infer:
        lbl = paddle.layer.data("label",
                                paddle.data_type.integer_value(class_num))

    seq_pool = paddle.layer.pooling(
        input=data, pooling_type=paddle.pooling.Max())
    for idx, hidden_size in enumerate(hidden_layer_sizes):
        hidden_init_std = 1.0 / math.sqrt(hidden_size)
        hidden = paddle.layer.fc(
            input=hidden if idx else seq_pool,
            size=hidden_size,
            act=paddle.activation.Tanh(),
            param_attr=paddle.attr.Param(initial_std=hidden_init_std))

    prob = paddle.layer.fc(
        input=hidden,
        size=class_num,
        act=paddle.activation.Softmax(),
        param_attr=paddle.attr.Param(initial_std=1.0 / math.sqrt(class_num)))
    if is_infer:
        return prob
    else:
        return paddle.layer.classification_cost(
            input=prob, label=lbl), prob, lbl


def conv_net(dict_dim,
                    class_dim=2,
                    hid_dim=128,
                    is_infer=False):

    # input layers
    data = paddle.layer.data("video",
                             paddle.data_type.dense_vector_sequence(dict_dim))
    if not is_infer:
        lbl = paddle.layer.data("label",
                                paddle.data_type.integer_value(class_dim))

    # convolution layers with max pooling
    conv_3 = paddle.networks.sequence_conv_pool(name='conv_1',
        input=data, context_len=3, hidden_size=hid_dim)
    conv_4 = paddle.networks.sequence_conv_pool(name='conv_2',
        input=data, context_len=4, hidden_size=hid_dim)

    # fc and output layer
    prob = paddle.layer.fc(
        input=[conv_3, conv_4], size=class_dim, act=paddle.activation.Softmax())

    if is_infer:
        return prob
    else:
        cost = paddle.layer.classification_cost(input=prob, label=lbl)

        return cost, prob, lbl


def lstm_net(input_dim,
                     class_dim=2,
                     hid_dim=512,
                     stacked_num=3,
                     is_infer=False):
    assert stacked_num % 2 == 1

    fc_para_attr = paddle.attr.Param(learning_rate=1e-3)
    lstm_para_attr = paddle.attr.Param(initial_std=0., learning_rate=1.)
    para_attr = [fc_para_attr, lstm_para_attr]
    bias_attr = paddle.attr.Param(initial_std=0., l2_rate=0.)
    relu = paddle.activation.Relu()
    linear = paddle.activation.Linear()

    data = paddle.layer.data("video",
                             paddle.data_type.dense_vector_sequence(input_dim))

    fc1 = paddle.layer.fc(input=data,
                          size=hid_dim,
                          act=linear,
                          bias_attr=bias_attr)
    lstm1 = paddle.layer.lstmemory(
        input=fc1, act=relu, bias_attr=bias_attr)

    inputs = [fc1, lstm1]
    for i in range(2, stacked_num + 1):
        fc = paddle.layer.fc(input=inputs,
                             size=hid_dim,
                             act=linear,
                             param_attr=para_attr,
                             bias_attr=bias_attr)
        lstm = paddle.layer.lstmemory(
            input=fc,
            reverse=(i % 2) == 0,
            act=relu,
            bias_attr=bias_attr)
        inputs = [fc, lstm]

    fc_last = paddle.layer.pooling(input=inputs[0], pooling_type=paddle.pooling.Max())
    lstm_last = paddle.layer.pooling(input=inputs[1], pooling_type=paddle.pooling.Max())
    output = paddle.layer.fc(input=[fc_last, lstm_last],
                             size=class_dim,
                             act=paddle.activation.Softmax(),
                             bias_attr=bias_attr,
                             param_attr=para_attr)

    if not is_infer:
        lbl = paddle.layer.data("label", paddle.data_type.integer_value(class_dim))
        cost = paddle.layer.classification_cost(input=output, label=lbl)
        return cost, output, lbl
    else:
        return output
