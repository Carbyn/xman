import paddle.v2 as paddle
import gzip
import math
def infer(data_type):
    proposal_data = {'results': {}, 'version': "VERSION 1.0"}
    json_data= load_json("data/meta.json")
    database=json_data['database']
    dict_dim = 2048
    class_num =2
    window_lens=[30]
    window_stride = 30
    model_path = "model/video_cnn_lstm.tar.gz"
    prob_layer = lstm_net(dict_dim, class_num, is_infer=True)
    
    # initialize PaddlePaddle
    paddle.init(use_gpu=False, trainer_count=1)

    # load the trained models
    if os.path.exists(model_path) :
        with gzip.open(model_path, 'r') as f:
            parameters = paddle.parameters.Parameters.from_tar(f)
    index = 0
    for video in database.keys():
        dataSet=database[video]["subset"]
        if dataSet != data_type:
            continue
        try:
            with open("data/"+dataSet+"/"+str(video)+".pkl",'rb') as f:
                video_fea=cPickle.load(f)
        except:
            continue
        print index,video
        index+=1
        video_len = np.shape(video_fea)[0]
        this_vid_proposals = []
        for pos in range(0, video_len, window_stride):
            inputs=[]
            for window_len in window_lens:
                if pos+window_len < video_len:
                    inputs.append([video_fea[pos:pos+window_len]])
            probs = paddle.infer(
                output_layer=prob_layer, parameters=parameters,input=inputs, field="value")
            if len(probs) <= 0: continue
            pos_probs = probs[:,1]
            max_probs = np.max(pos_probs)
            window_index = np.argmax(pos_probs)
            if max_probs < 0.5 : continue
            proposal = {
                    'score': max_probs,
                    'segment': [pos, pos+window_lens[window_index]],
                   }
            this_vid_proposals += [proposal]
        proposal_data['results'][video] = this_vid_proposals
    with open("results/"+data_type+".json", 'w') as fobj:
        json.dump(proposal_data, fobj)
infer('training')
