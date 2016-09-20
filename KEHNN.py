import cPickle
import numpy as np
import theano
from RNN import GRU, BiGRU, GRU_Cell
from numpy.random import shuffle
from sklearn.metrics import f1_score, accuracy_score
from gensim.models.word2vec import Word2Vec
from process_ubuntu import WordVecs
from CNN import QALeNetConvPoolLayer,LeNetConvPoolLayer
from Classifier import BilinearLR, MLP
from Optimization import Adadelta, Adam
import theano.tensor as T

def get_idx_from_sent(sent, word_idx_map, max_l=50, filter_h=3):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    x_mask = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
        x_mask.append(0)
    words = sent.split()
    for i, word in enumerate(words):
        if i >= max_l: break
        if word in word_idx_map:
            x.append(word_idx_map[word])
            x_mask.append(1)
    while len(x) < max_l+2*pad:
        x.append(0)
        x_mask.append(0)
    for e in x_mask:
        x.append(e)
    return x

def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
            rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

def predict_nn(datasets,
        U,
        TW,# pre-trained word embeddings
        filter_hs=[3],           # filter width
        feature_map_per_layer=[100,2],
        shuffle_batch=True,
        n_epochs=25,
        batch_size=20,input_embedding=100,hidden_dimension = 100,max_echo = 5,pooling_size = [(3,3)]):          # for optimization
    """
    return: a list of dicts of lists, each list contains (ansId, groundTruth, prediction) for a question
    """
    num_channel = 3
    U = U.astype(dtype=theano.config.floatX)
    rng = np.random.RandomState(3435)
    img_h = (len(datasets[0][0]) - 3) / 4
    img_w = (int)(U.shape[1])
    lsize, rsize = img_h, img_h

    filter_w = 1
    feature_maps = feature_map_per_layer[0]
    filter_shapes = []
    pool_sizes = pooling_size
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, num_channel, filter_h, filter_h))

    print pool_sizes

    parameters = [("image shape",img_h,img_w), ("filter shape",filter_shapes),
                  ("feature_maps ",feature_maps ), ("batch_size",batch_size),
                   ("shuffle_batch",shuffle_batch),("num_channel", num_channel)
                  ,("input_embedding",input_embedding),("hidden_dimension",hidden_dimension)]
    print parameters

    index = T.lscalar()
    lx = T.matrix('lx')
    rx = T.matrix('rx')
    y = T.ivector('y')
    t = T.ivector()
    lxmask = T.matrix()
    rxmask = T.matrix()
    t2 = T.ivector()
    TWords = theano.shared(value = TW, name = "TWords")
    Words = theano.shared(value = U, name = "Words")


    train_set, dev_set, test_set = datasets[0], datasets[1], datasets[2]
    train_set_lx = theano.shared(np.asarray(train_set[:,:lsize],dtype=theano.config.floatX),borrow=True)
    train_set_lx_mask = theano.shared(np.asarray(train_set[:,lsize:2*lsize],dtype=theano.config.floatX),borrow=True)
    train_set_rx = theano.shared(np.asarray(train_set[:,2*lsize:3*lsize],dtype=theano.config.floatX),borrow=True)
    train_set_rx_mask = theano.shared(np.asarray(train_set[:,3*lsize:4 *lsize],dtype=theano.config.floatX),borrow=True)
    train_set_y =theano.shared(np.asarray(train_set[:,-3],dtype="int32"),borrow=True)
    train_set_t =theano.shared(np.asarray(train_set[:,-2],dtype="int32"),borrow=True)
    train_set_t2 =theano.shared(np.asarray(train_set[:,-1],dtype="int32"),borrow=True)

    val_set_lx = theano.shared(np.asarray(dev_set[:,:lsize],dtype=theano.config.floatX),borrow=True)
    val_set_lx_mask = theano.shared(np.asarray(dev_set[:,lsize:2*lsize],dtype=theano.config.floatX),borrow=True)
    val_set_rx = theano.shared(np.asarray(dev_set[:,2*lsize:3*lsize],dtype=theano.config.floatX),borrow=True)
    val_set_rx_mask = theano.shared(np.asarray(dev_set[:,3*lsize:4 *lsize],dtype=theano.config.floatX),borrow=True)
    val_set_y =theano.shared(np.asarray(dev_set[:,-3],dtype="int32"),borrow=True)
    val_set_t =theano.shared(np.asarray(dev_set[:,-2],dtype="int32"),borrow=True)
    val_set_t2 =theano.shared(np.asarray(dev_set[:,-1],dtype="int32"),borrow=True)

    test_set_lx = theano.shared(np.asarray(test_set[:,:lsize],dtype=theano.config.floatX),borrow=True)
    test_set_lx_mask = theano.shared(np.asarray(test_set[:,lsize:2*lsize],dtype=theano.config.floatX),borrow=True)
    test_set_rx = theano.shared(np.asarray(test_set[:,2*lsize:3*lsize],dtype=theano.config.floatX),borrow=True)
    test_set_rx_mask = theano.shared(np.asarray(test_set[:,3*lsize:4 *lsize],dtype=theano.config.floatX),borrow=True)
    test_set_y =theano.shared(np.asarray(test_set[:,-3],dtype="int32"),borrow=True)
    test_set_t =theano.shared(np.asarray(test_set[:,-2],dtype="int32"),borrow=True)
    test_set_t2 =theano.shared(np.asarray(test_set[:,-1],dtype="int32"),borrow=True)

    cell = GRU_Cell(n_in=input_embedding,n_hidden=hidden_dimension,n_out=100)
    session_topic = T.mean(TWords[t],1)
    res_topic = T.mean(TWords[t2],1)
    W = theano.shared(value=ortho_weight(2*hidden_dimension),borrow=True)
    W4 = theano.shared(value=ortho_weight(2*hidden_dimension),borrow=True)

    llayer0_input = Words[T.cast(lx.flatten(),dtype="int32")].reshape((lx.shape[0],lx.shape[1],Words.shape[1]))     # input: word embeddings of the mini batch
    rlayer0_input = Words[T.cast(rx.flatten(),dtype="int32")].reshape((rx.shape[0],rx.shape[1],Words.shape[1]))     # input: word embeddings of the mini batch

    session_topic = T.repeat(session_topic,img_h,1).reshape((batch_size,100,img_h)).dimshuffle(0,2,1)
    res_topic = T.repeat(res_topic,img_h,1).reshape((batch_size,100,img_h)).dimshuffle(0,2,1)


    cell1 = cell(session_topic,llayer0_input)
    cell2 = cell(res_topic,rlayer0_input)

    sentence2vec = BiGRU(n_in=input_embedding,n_hidden=hidden_dimension,n_out=100,batch_size=batch_size)
    sentence2vec2 = BiGRU(n_in=input_embedding,n_hidden=hidden_dimension,n_out=100,batch_size=batch_size)
    sentence2vec(llayer0_input,lxmask,return_list = True)

    q_emb = sentence2vec(llayer0_input,lxmask,return_list = True) * 0.5
    r_emb = sentence2vec(rlayer0_input,rxmask,return_list = True) * 0.5

    q_emb2 = sentence2vec2(cell1,lxmask,return_list = True) * 0.5
    r_emb2 = sentence2vec2(cell2,rxmask,return_list = True) * 0.5


    input1 = ReLU(T.batched_dot(T.dot(q_emb,W),r_emb.dimshuffle(0,2,1)))
    input2 = ReLU(T.batched_dot(T.dot(q_emb2,W4),r_emb2.dimshuffle(0,2,1)))
    input3 = ReLU(T.batched_dot(llayer0_input * lxmask.dimshuffle(0,1,'x')
                           ,(rlayer0_input * rxmask.dimshuffle(0,1,'x')).dimshuffle(0,2,1)))

    input = T.concatenate([input1.dimshuffle(0,'x',1,2),input3.dimshuffle(0,'x',1,2),
                           input2.dimshuffle(0,'x',1,2)],1)


    conv_layer = LeNetConvPoolLayer(rng,input,filter_shape=filter_shapes[0],
                                    image_shape=(batch_size,num_channel,img_h,img_h)
                       ,poolsize=pool_sizes[0],non_linear='relu')

    mlp_in = T.flatten(conv_layer.output,2)
    classifier = MLP(rng,mlp_in,8*17*17,50,3)
    cost = classifier.negative_log_likelihood(y)
    predict_prob = classifier.logRegressionLayer.predict_prob
    predict_y = classifier.logRegressionLayer.predict_y

    error = classifier.errors(y)
    params = classifier.params
    params += conv_layer.params
    params += sentence2vec.params
    params += cell.params
    params += sentence2vec2.params
    params += [W4,W]
    #params +=[Words]
    params += [TWords]

    load_params(params,'model.bin')
    import numpy

    # numpy.savetxt('input2.txt',a[0][0])
    # numpy.savetxt('input1.txt',a[1][1])
    print '########### predicting'

    val_model2 = theano.function([index],[predict_y ,y,cost,error,predict_prob],givens={
        lx: test_set_lx[index*batch_size:(index+1)*batch_size],
        lxmask: test_set_lx_mask[index*batch_size:(index+1)*batch_size],
        rx: test_set_rx[index*batch_size:(index+1)*batch_size],
        rxmask:test_set_rx_mask[index*batch_size:(index+1)*batch_size],
        y: test_set_y[index*batch_size:(index+1)*batch_size],
        t: test_set_t[index*batch_size:(index+1)*batch_size],
        t2: test_set_t2[index*batch_size:(index+1)*batch_size]
    },on_unused_input='ignore')
    best_dev = 0.

    cost = 0
    errors = 0
    pys = []
    labelys = []
    j = 0
    f = open('res.txt','w')
    n_train_batches = datasets[0].shape[0]/batch_size
    for minibatch_index in xrange(datasets[2].shape[0]/batch_size):
        py,labely, tcost, terr,prob = val_model2(minibatch_index)
        cost += tcost
        errors += terr
        for i in range(batch_size):
            pys.append(py[i])
            labelys.append(labely[i])
            f.write("{0}\t{1}\n".format(py[i],labely[i]))
        j = j+1
    print 'test macro f1',f1_score(labelys,pys,labels=[0,1,2],average='macro')
    print 'test acc', accuracy_score(labelys,pys)



def load_params(params,filename):
    f = open(filename)
    num_params = cPickle.load(f)
    for p,w in zip(params,num_params):
        p.set_value(w,borrow=True)
    print "load successfully"
def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')
def kmaxpooling(input,input_shape,k):
    sorted_values = T.argsort(input,axis=3)
    topmax_indexes = sorted_values[:,:,:,-k:]
    # sort indexes so that we keep the correct order within the sentence
    topmax_indexes_sorted = T.sort(topmax_indexes)

    #given that topmax only gives the index of the third dimension, we need to generate the other 3 dimensions
    dim0 = T.arange(0,input_shape[0]).repeat(input_shape[1]*input_shape[2]*k)
    dim1 = T.arange(0,input_shape[1]).repeat(k*input_shape[2]).reshape((1,-1)).repeat(input_shape[0],axis=0).flatten()
    dim2 = T.arange(0,input_shape[2]).repeat(k).reshape((1,-1)).repeat(input_shape[0]*input_shape[1],axis=0).flatten()
    dim3 = topmax_indexes_sorted.flatten()
    return input[dim0,dim1,dim2,dim3].reshape((input_shape[0], input_shape[1], input_shape[2], k))

def train_nn(datasets,
        U,
        TW,# pre-trained word embeddings
        filter_hs=[3],           # filter width
        feature_map_per_layer=[100,2],
        shuffle_batch=True,
        n_epochs=25,
        batch_size=20,input_embedding=100,hidden_dimension = 100,max_echo = 5,pooling_size = [(3,3)]):
    """
    return: a list of dicts of lists, each list contains (ansId, groundTruth, prediction) for a question
    """
    num_channel = 3
    U = U.astype(dtype=theano.config.floatX)
    rng = np.random.RandomState(3435)
    img_h = (len(datasets[0][0]) - 3) / 4
    img_w = (int)(U.shape[1])
    lsize, rsize = img_h, img_h

    filter_w = 1
    feature_maps = feature_map_per_layer[0]
    filter_shapes = []
    pool_sizes = pooling_size

    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, num_channel, filter_h, filter_h))

    print pool_sizes

    parameters = [("image shape",img_h,img_w), ("filter shape",filter_shapes),
                  ("feature_maps ",feature_maps), ("batch_size",batch_size),
                   ("shuffle_batch",shuffle_batch),("num_channel", num_channel)
                  ,("input_embedding",input_embedding),("hidden_dimension",hidden_dimension)]
    print parameters

    index = T.lscalar()
    lx = T.matrix('lx')
    rx = T.matrix('rx')
    y = T.ivector('y')
    t = T.ivector()
    lxmask = T.matrix()
    rxmask = T.matrix()
    t2 = T.ivector()
    TWords = theano.shared(value = TW, name = "TWords")
    Words = theano.shared(value = U, name = "Words")


    train_set, dev_set, test_set = datasets[0], datasets[1], datasets[2]
    train_set_lx = theano.shared(np.asarray(train_set[:,:lsize],dtype=theano.config.floatX),borrow=True)
    train_set_lx_mask = theano.shared(np.asarray(train_set[:,lsize:2*lsize],dtype=theano.config.floatX),borrow=True)
    train_set_rx = theano.shared(np.asarray(train_set[:,2*lsize:3*lsize],dtype=theano.config.floatX),borrow=True)
    train_set_rx_mask = theano.shared(np.asarray(train_set[:,3*lsize:4 *lsize],dtype=theano.config.floatX),borrow=True)
    train_set_y =theano.shared(np.asarray(train_set[:,-3],dtype="int32"),borrow=True)
    train_set_t =theano.shared(np.asarray(train_set[:,-2],dtype="int32"),borrow=True)
    train_set_t2 =theano.shared(np.asarray(train_set[:,-1],dtype="int32"),borrow=True)

    val_set_lx = theano.shared(np.asarray(dev_set[:,:lsize],dtype=theano.config.floatX),borrow=True)
    val_set_lx_mask = theano.shared(np.asarray(dev_set[:,lsize:2*lsize],dtype=theano.config.floatX),borrow=True)
    val_set_rx = theano.shared(np.asarray(dev_set[:,2*lsize:3*lsize],dtype=theano.config.floatX),borrow=True)
    val_set_rx_mask = theano.shared(np.asarray(dev_set[:,3*lsize:4 *lsize],dtype=theano.config.floatX),borrow=True)
    val_set_y =theano.shared(np.asarray(dev_set[:,-3],dtype="int32"),borrow=True)
    val_set_t =theano.shared(np.asarray(dev_set[:,-2],dtype="int32"),borrow=True)
    val_set_t2 =theano.shared(np.asarray(dev_set[:,-1],dtype="int32"),borrow=True)

    test_set_lx = theano.shared(np.asarray(test_set[:,:lsize],dtype=theano.config.floatX),borrow=True)
    test_set_lx_mask = theano.shared(np.asarray(test_set[:,lsize:2*lsize],dtype=theano.config.floatX),borrow=True)
    test_set_rx = theano.shared(np.asarray(test_set[:,2*lsize:3*lsize],dtype=theano.config.floatX),borrow=True)
    test_set_rx_mask = theano.shared(np.asarray(test_set[:,3*lsize:4 *lsize],dtype=theano.config.floatX),borrow=True)
    test_set_y =theano.shared(np.asarray(test_set[:,-3],dtype="int32"),borrow=True)
    test_set_t =theano.shared(np.asarray(test_set[:,-2],dtype="int32"),borrow=True)
    test_set_t2 =theano.shared(np.asarray(test_set[:,-1],dtype="int32"),borrow=True)

    cell = GRU_Cell(n_in=input_embedding,n_hidden=hidden_dimension,n_out=100)
    session_topic = T.mean(TWords[t],1)
    res_topic = T.mean(TWords[t2],1)
    W = theano.shared(value=ortho_weight(2*hidden_dimension),borrow=True)
    W4 = theano.shared(value=ortho_weight(2*hidden_dimension),borrow=True)

    llayer0_input = Words[T.cast(lx.flatten(),dtype="int32")].reshape((lx.shape[0],lx.shape[1],Words.shape[1]))     # input: word embeddings of the mini batch
    rlayer0_input = Words[T.cast(rx.flatten(),dtype="int32")].reshape((rx.shape[0],rx.shape[1],Words.shape[1]))     # input: word embeddings of the mini batch

    session_topic = T.repeat(session_topic,img_h,1).reshape((batch_size,100,img_h)).dimshuffle(0,2,1)
    res_topic = T.repeat(res_topic,img_h,1).reshape((batch_size,100,img_h)).dimshuffle(0,2,1)

    cell1 = cell(session_topic,llayer0_input)
    cell2 = cell(res_topic,rlayer0_input)

    sentence2vec = BiGRU(n_in=input_embedding,n_hidden=hidden_dimension,n_out=100,batch_size=batch_size)
    sentence2vec2 = BiGRU(n_in=input_embedding,n_hidden=hidden_dimension,n_out=100,batch_size=batch_size)
    sentence2vec(llayer0_input,lxmask,return_list = True)


    q_emb = _dropout_from_layer(rng,sentence2vec(llayer0_input,lxmask,return_list = True),0.5)
    r_emb = _dropout_from_layer(rng,sentence2vec(rlayer0_input,rxmask,return_list = True),0.5)

    q_emb2 = _dropout_from_layer(rng, sentence2vec2(cell1,lxmask,return_list = True),0.5)
    r_emb2 = _dropout_from_layer(rng, sentence2vec2(cell2,rxmask,return_list = True),0.5)

    input1 = ReLU(T.batched_dot(T.dot(q_emb,W),r_emb.dimshuffle(0,2,1)))
    input2 = ReLU(T.batched_dot(T.dot(q_emb2,W4),r_emb2.dimshuffle(0,2,1)))
    input3 = ReLU(T.batched_dot(llayer0_input * lxmask.dimshuffle(0,1,'x')
                           ,(rlayer0_input * rxmask.dimshuffle(0,1,'x')).dimshuffle(0,2,1)))

    input = T.concatenate([input1.dimshuffle(0,'x',1,2),input3.dimshuffle(0,'x',1,2),
                           input2.dimshuffle(0,'x',1,2)],1)


    conv_layer = LeNetConvPoolLayer(rng,input,filter_shape=filter_shapes[0],
                                    image_shape=(batch_size,num_channel,img_h,img_h)
                       ,poolsize=pool_sizes[0],non_linear='relu')

    mlp_in = T.flatten(conv_layer.output,2)
    classifier = MLP(rng,mlp_in,8*17*17,50,3)
    cost = classifier.negative_log_likelihood(y)
    predict_y = classifier.logRegressionLayer.predict_y


    opt = Adam()
    error = classifier.errors(y)
    params = classifier.params
    params += conv_layer.params
    params += sentence2vec.params
    params += cell.params
    params += sentence2vec2.params
    params += [W4,W]
    #params +=[Words]
    params += [TWords]

    grad_updates = opt.Adam(cost=cost,params=params,lr = 0.001)
    train_model = theano.function([index], cost,updates=grad_updates, givens={
        lx: train_set_lx[index*batch_size:(index+1)*batch_size],
        lxmask:train_set_lx_mask[index*batch_size:(index+1)*batch_size],
        rx: train_set_rx[index*batch_size:(index+1)*batch_size],
        rxmask:train_set_rx_mask[index*batch_size:(index+1)*batch_size],
        y: train_set_y[index*batch_size:(index+1)*batch_size],
        t: train_set_t[index*batch_size:(index+1)*batch_size],
        t2: train_set_t2[index*batch_size:(index+1)*batch_size]
    },on_unused_input='ignore')
    val_model = theano.function([index],[predict_y,y,cost,error],givens={
        lx: val_set_lx[index*batch_size:(index+1)*batch_size],
        lxmask: val_set_lx_mask[index*batch_size:(index+1)*batch_size],
        rx: val_set_rx[index*batch_size:(index+1)*batch_size],
        rxmask: val_set_rx_mask[index*batch_size:(index+1)*batch_size],
        y: val_set_y[index*batch_size:(index+1)*batch_size],
        t: val_set_t[index*batch_size:(index+1)*batch_size],
        t2: val_set_t2[index*batch_size:(index+1)*batch_size]
    },on_unused_input='ignore')
    val_model2 = theano.function([index],[predict_y ,y,cost,error],givens={
        lx: test_set_lx[index*batch_size:(index+1)*batch_size],
        lxmask: test_set_lx_mask[index*batch_size:(index+1)*batch_size],
        rx: test_set_rx[index*batch_size:(index+1)*batch_size],
        rxmask:test_set_rx_mask[index*batch_size:(index+1)*batch_size],
        y: test_set_y[index*batch_size:(index+1)*batch_size],
        t: test_set_t[index*batch_size:(index+1)*batch_size],
        t2: test_set_t2[index*batch_size:(index+1)*batch_size]
    },on_unused_input='ignore')
    best_dev = 0.

    n_train_batches = datasets[0].shape[0]/batch_size
    for i in xrange(max_echo):
        cost = 0
        total = 0.
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            batch_cost = train_model(minibatch_index)
            total = total + 1
            cost = cost + batch_cost
            if total % 50 == 0:
                print 'batch number:', total, 'avg cost:', cost/total
        cost = cost / n_train_batches
        print "trainning echo %d loss %f" % (i,cost)

        cost=0
        errors = 0
        j = 0
        pys = []
        labelys = []
        for minibatch_index in xrange(datasets[1].shape[0]/batch_size):
            py,labely, tcost, terr = val_model(minibatch_index)
            cost += tcost
            errors += terr
            j = j+1
            for i in range(batch_size):
                pys.append(py[i])
                labelys.append(labely[i])
            #print pys
        print "val marco f1:",f1_score(labelys,pys,labels=[0,1,2],average='macro')
        print "val accuracy:" , accuracy_score(labelys,pys)
        if accuracy_score(labelys,pys) > best_dev:
            best_dev = accuracy_score(labelys,pys)
            save_params(params,'model.bin')
        pys = []
        labelys = []
        cost = 0
        errors = 0
        for minibatch_index in xrange(datasets[2].shape[0]/batch_size):
            py,labely, tcost, terr = val_model2(minibatch_index)
            cost += tcost
            errors += terr
            #print py
            for i in range(batch_size):
                pys.append(py[i])
                labelys.append(labely[i])
            #f.write(py)
            j = j+1
        print 'test macro f1',f1_score(labelys,pys,labels=[0,1,2],average='macro')
        print 'test acc', accuracy_score(labelys,pys)

        cost = cost / j
        errors = errors / j
        if cost < best_dev:
            best_dev = cost


def save_params(params,filename):
    print '###saving model!!!!####'
    num_params = [p.get_value() for p in params]
    f = open(filename,'wb')
    cPickle.dump(num_params,f)

def make_data(revs, word_idx_map, max_l=50, filter_h=3
                  ,train_instance = 16541,val_instance = 1645):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, val, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["m"], word_idx_map, max_l, filter_h)
        sent += get_idx_from_sent(rev["r"], word_idx_map, max_l, filter_h)
        sent.append(int(rev["y"]))
        sent.append(int(rev["t"]))
        sent.append(int(rev["t2"]))

        if len(train) < train_instance:
            train.append(sent)
        elif (len(train) + len(val)) < train_instance + val_instance:
            val.append(sent)
        else:
            test.append(sent)

    shuffle(train)
    train = np.array(train,dtype="int")
    val = np.array(val,dtype="int")
    test = np.array(test,dtype="int")
    print 'trainning data', len(train),'val data', len(val)
    return [train, val, test]


if __name__=="__main__":
    dataset = r"D:\users\wuyu\pythoncode\SemEval2015\preserve_stopc.bin"
    x = cPickle.load(open(dataset,"rb"))
    revs, wordvecs, max_l,tw = x[0], x[1], x[2], x[3]
    datasets = make_data(revs,wordvecs.word_idx_map,max_l=50,
                             train_instance = 16541,val_instance = 1645)

    #train_nn(datasets,wordvecs.W,tw,filter_hs=[3],feature_map_per_layer=[8],batch_size=50
    #         ,input_embedding=100,hidden_dimension = 100,pooling_size = (3,3))

    predict_nn(datasets,wordvecs.W,tw,filter_hs=[3],feature_map_per_layer=[8],batch_size=50
             ,input_embedding=100,hidden_dimension = 100,pooling_size = [(3,3)])