import cPickle
from collections import defaultdict
import logging
import gensim
import numpy as np
from numpy.random import shuffle
from gensim.models.word2vec import Word2Vec
import theano
logger = logging.getLogger('relevance_logger')

def build_data(trainfiles, max_len = 20,isshuffle=False):
    revs = []
    vocab = defaultdict(float)

    for trainfile in trainfiles:
        with open(trainfile) as f:
            for line in f:
                parts = line.strip().split("\t")

                if len(parts) < 5:
                    qt = parts[0]
                    rt = parts[1]
                    lable = parts[2]
                    message = parts[3]
                    response = 'none'
                else:
                    qt = parts[0]
                    rt = parts[1]
                    lable = parts[2]
                    message = parts[3]
                    response = parts[4]

                data = {"y" : lable, "m":message,"r": response,"t":qt,"t2":rt}
                revs.append(data)

                words = set(message.split())
                words.update(set(response.split()))
                for word in words:
                    vocab[word] += 1
        logger.info("processed dataset with %d question-answer pairs " %(len(revs)))
        logger.info("vocab size: %d" %(len(vocab)))
        if isshuffle == True:
            shuffle(revs)
    return revs, vocab, max_len

class WordVecs(object):
    def __init__(self, fname, vocab, binary, gensim):
        if gensim:
            word_vecs = self.load_gensim(fname,vocab)
        self.k = len(word_vecs.values()[0])
        self.W, self.word_idx_map = self.get_W(word_vecs, k=self.k)

    def get_W(self, word_vecs, k=300):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size+1, k))
        W[0] = np.zeros(k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def load_gensim(self, fname, vocab):
         model = Word2Vec.load(fname)
         weights = [[0.] * model.vector_size]
         word_vecs = {}
         total_inside_new_embed = 0
         for pair in vocab:
             word = gensim.utils.to_unicode(pair)
             if word in model:
                total_inside_new_embed += 1
                word_vecs[pair] = np.array([w for w in model[word]])
                #weights.append([w for w in model[word]])
             else:
                word_vecs[pair] = np.array([0.] * model.vector_size)
                #weights.append([0.] * model.vector_size)
         print 'transfer', total_inside_new_embed, 'words from the embedding file, total', len(vocab), 'candidate'
         return word_vecs

def createtopicvec(word2vec_path):
    max_topicword = 20
    model = Word2Vec.load(word2vec_path)
    topicmatrix = np.zeros(shape=(100,max_topicword,100),dtype=theano.config.floatX)
    file = open(r"\\msra-sandvm-001\v-wuyu\Data\SemEvalCQA"
                r"\semeval2015-task3-english-data\pre-process\stemming_preservestop_cate\catedic.txt")
    i = 0
    miss = 0
    for line in file:
        tmp = line.strip().split(' ')
        for j in range(min(len(tmp),max_topicword)):
            if gensim.utils.to_unicode(tmp[j]) in model.vocab:
                topicmatrix[i,j,:] = model[gensim.utils.to_unicode(tmp[j])]
            else:
                miss = miss+1
        i= i+1
    print "miss word2vec", miss
    return topicmatrix

if __name__=="__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    revs, vocab, max_len = build_data([r"\\msra-sandvm-001\v-wuyu\Data\SemEvalCQA\semeval2015-task3-english-data\pre-process\stemming_preservestop_cate\train.txt.preprocess.txt"
                                       ,r"\\msra-sandvm-001\v-wuyu\Data\SemEvalCQA\semeval2015-task3-english-data\pre-process\stemming_preservestop_cate\dev.txt.preprocess.txt"
                                       ,r"\\msra-sandvm-001\v-wuyu\Data\SemEvalCQA\semeval2015-task3-english-data\pre-process\stemming_preservestop_cate\test.txt.preprocess.txt"])
    word2vec = WordVecs(r"\\msra-sandvm-001\v-wuyu\Models\W2V\SemEval2015\word2vec.preservestop.stemming.model", vocab, True, True)
    cPickle.dump([revs, word2vec, max_len,createtopicvec(r"\\msra-sandvm-001\v-wuyu\Models"
                                                         r"\W2V\SemEval2015\word2vec.preservestop.stemming.model")]
                 , open("training.bin",'wb'))

    logger.info("dataset created!")