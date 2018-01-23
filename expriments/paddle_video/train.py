from network import lstm_net
from reader import reader_creator 
import paddle.v2 as paddle
import gzip
import sys
import math


def train():
    dict_dim = 2048
    num_passes = 100
    window_len = 30 
    model_path = "model/video_lstm" + str(window_len) + ".tar.gz"
    paddle.init(use_gpu=False, trainer_count=1)                                                                       

    # network config                                                                                                  
    cost, prob, label = lstm_net(dict_dim, class_dim=2)

    if os.path.exists(model_path):
        with gzip.open(model_path, 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)
    else:
        parameters = paddle.parameters.create(cost)
    adam_optimizer = paddle.optimizer.Adam(                                                                           
        learning_rate=1e-3,
        regularization=paddle.optimizer.L2Regularization(rate=1e-3),                                                  
        model_average=paddle.optimizer.ModelAverage(average_window=0.5))                                              

    # create trainer
    trainer = paddle.trainer.SGD(
        cost=cost,                                                                                                    
        extra_layers=paddle.evaluator.auc(input=prob, label=label),                                                   
        parameters=parameters,                                                                                        
        update_equation=adam_optimizer)                                                                               

    # begin training network                                                                                          
    feeding = {"video": 0, "label": 1}                                                                                 

    def _event_handler(event):
        """
        Define end batch and end pass event handler                                                                   
        """
        if isinstance(event, paddle.event.EndIteration):                                                              
            if event.batch_id % 10 == 0:                                                                             
                print "Pass %d, Batch %d, Cost %f, %s\n" % (                                                    
                    event.pass_id, event.batch_id, event.cost, event.metrics)

        if isinstance(event, paddle.event.EndPass):                                                                   
            print "Save model"                                                
            with gzip.open(model_path, "w") as f:                                                           
                parameters.to_tar(f)   
            if event.pass_id % 1 == 0:                                                                             
                print 'Test start'
                result = trainer.test(reader=paddle.batch(reader_creator('validation', window_len, class_num=2), 256), feeding=feeding)                                            
                print "Test at Pass %d, %s \n" % (event.pass_id, result.metrics)                                              

    trainer.train(
        reader=paddle.batch(paddle.reader.shuffle(reader_creator('training', window_len, class_num=2), 1024), 256),
        event_handler=_event_handler,
        feeding=feeding,
        num_passes=num_passes)
train()
