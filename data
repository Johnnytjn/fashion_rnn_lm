import os
from pyltp import SentenceSplitter
import re
from tqdm import tqdm
from pyltp import Segmentor
import json
import string
from collections import Counter

#data_path = '/home/tongjianing/lm/data/taobao_headline_0412'

UNK_ID = 0
SOS_ID = 1
EOS_ID = 2



LTP_DATA_DIR = '/home/tongjianing/lm/data/ltp_data_v3.4.0'
cws_model_path = os.path.join(LTP_DATA_DIR,'cws.model')

class FashionData(object):
    def __init__(self, save_path,  data_path, batch_size = 5000, max_content_len=1000):

        self.batch_size = batch_size
        self.data_path = data_path
        self.save_path = save_path
        self.max_content_len = max_content_len
        self.LTP_DATA_DIR = '/home/tongjianing/lm/data/ltp_data_v3.4.0'
        self.vocab_path ='/home/tongjianing/lm/data/fashion.words.1123'
        self.max_sentence_len = 0
#         self.segmentor = Segmentor()  # 初始化实例
#         self.cws_model_path = os.path.join(self.LTP_DATA_DIR,'cws.model')
#         self.segmentor.load(self.cws_model_path)  # 加载模型
#         self.segmentor.load_with_lexicon(self.cws_model_path, self.vocab_path) # 加载模型，第二个参数是您的外部词典文件路径
#         self.data_dir_prefix = '/home/tongjianing/lm/data/'




    def create_vocab(self, fenci_filename, vocab_file_i2w, vocab_file_w2i, delimiter=",", min_count=0, max_size=2000):
        """
        words should be split by delimiter, default is space
        """
        counter = Counter()
        print('********************************************')
        with open(fenci_filename) as f:
            line = f.readline().strip()
            i = 0
            print(i)
            while line:
                line = line.strip()
                tokens = line.split(delimiter)
                counter.update([t.strip() for t in tokens if t.strip()])
                line = f.readline()
                
                i += 1
                print('********************************************')
                print(i)
                if i % 100 == 0:
                    print("Processed {0} lines".format(i))
            words = counter.most_common()
            
            words = [word for word in words if word[1] >= min_count]
            if max_size:
                words = words[:max_size]
            word2id = {'UNK': UNK_ID}
            for word, _ in words:
                if word in word2id:
                    continue
                word2id[word] = len(word2id)
            print("total words", len(word2id))
            #print("total tokens", sum([item[1] for item in words]))
            id2word = {i: w for w, i in word2id.items()}
            f1 =  open(vocab_file_i2w, 'w')
            json.dump(id2word, f1, ensure_ascii=False)
            f1.close()
            f2 =  open(vocab_file_w2i, 'w')
            json.dump(word2id, f2,ensure_ascii=False)
            f2.close()
           
        return word2id, id2word   

    def vocab_size(self, word2id):
        return len(word2id)
    def ids_to_sentences(self, ids):
        tokens = [self.i2w.get(t,UNK_ID) for t in ids]
        return ''.join(tokens)

    def worker_file(self, segmentor, data_file_name,fenci_result_fname):
        '''
        fname: 数据文件名 datafile in data_file
        segmentor: 分词器
        fenci_result_fname = 分词结果文件
        '''
        
        print("start to process file {0}".format(data_file_name))
        cnt = 0
        
        en_punctuation = string.punctuation
        cn_punctuation = json.load(open("/data/xueyou/textsum/chinese_punctuation.json"))
        keeps = '^\u4e00-\u9fa50-9a-zA-Z ' + en_punctuation  + cn_punctuation
        keeps = u'[{0}]+'.format(keeps)
        with open(os.path.join(self.save_path ,fenci_result_fname),'w') as f:
            
            for line in open(os.path.join(self.data_path, data_file_name),'r'):

                word_article = []
                tokens = line.strip().split("\x01")
                
                if(len(tokens) != 18):  #每一行包含18个元素
                    continue
                article = tokens[3] #第4项才是文章内容
                article = re.sub(keeps,'',article)
                article = article.split('<para>')
#                 __data = []
#                 __data.extend(data.split('。') for data in datas)
                
                for sentence in article:
                    
                    #print(sentence)
                    if sentence =='\n':
                        sentence= sentence.strip("\n")
                    result = segmentor.segment(sentence)
                    word_article.extend(result)
                cnt += 1
                if cnt % 1000 == 0:
                    print("{0} processed {1} articles".format(data_file_name,cnt))
                f.write(",".join(word_article))
                f.write("\n")

        print("finish processing file {0} ".format(data_file_name))
        segmentor.release() 
        
    def preprocess(self, w2i, data_file_line,  max_content_len = 500,delimiter=","):
        '''process setences to vector
        arguments:
        w2i: dict: wordtoId
        data_file_line : str : 原始淘宝数据文件一行——代表一篇文章
        max_content_len : int : max sentence length  
        data_path:原始淘宝数据路径
        return:
        每一句话转换id后的样子
        e.g.:[SOS_ID] + [w2i.get(t,UNK_ID) for t in line.strip().split()] + [EOS_ID]
        '''
        #print("preprocess data")

        transfered_contents = []
        trans_tokens = [SOS_ID] + [w2i.get(t,UNK_ID) for t in data_file_line.strip().split(delimiter)] + [EOS_ID]
        trans_tokens = trans_tokens[:max_content_len]
        
        transfered_contents.append(trans_tokens)
        #self.num_batches = int(len(transfered_contents) / (self.batch_size))
        return transfered_contents
    def get_next(self):
            batch = []

            for i in range(self.num_batches):
                if len(batch) != self.batch_size:
                    x = raw_data[i*self.batch_size:(i+1)*self.batch_size]
                    x = np.array(x).astype(np.int32)
                    y = raw_data[i*self.batch_size+1:(i+1)*self.batch_size+1]
                    if len(y) < self.batch_size :
                        y = y + self._raw_data[:self.batch_size  - len(y)]
                    y = np.array(y).astype(np.int32)
                    batch.append((x,y))
                else:
                    for j in range(len(batch)):
                        yield batch.pop()
            if len(batch)!=0:
                for j in range(len(batch)):
                    yield batch.pop()
    def save_data(self, datas, outcome_save_path):
        f = open(outcome_save_path,'a')
        for data in datas:
            f.write(str(data))
            f.write("\n")
            
            
        f.close()



if __name__ == "__main__":
    save_path = '/data/share/tongjianing/'
    
    
    data_path = '/home/tongjianing/lm/data/taobao_headline_test'
    fashiondata = FashionData(save_path, data_path = '/home/tongjianing/lm/data/taobao_headline_test')
    LTP_DATA_DIR = '/home/tongjianing/lm/data/ltp_data_v3.4.0'
    vocab_path ='/home/tongjianing/lm/data/fashion.words.1123'
    
    segmentor = Segmentor()  # 初始化实例
    cws_model_path = os.path.join(LTP_DATA_DIR,'cws.model')
    segmentor.load(cws_model_path)  # 加载模型
    segmentor.load_with_lexicon(cws_model_path, vocab_path) # 加载模型，第二个参数是您的外部词典文件路径
    
    data_files = []
    for file in os.listdir(data_path): #select useful files
        
        if file.startswith("."):
            continue
        else:
            #print(file)
            data_files.append(file)
   
    fenci_result_fname = 'afterprocess.txt'
    for data_file in data_files:
        fashiondata.worker_file(segmentor,data_file,fenci_result_fname)
    
    print('Done with fenci procecss!')    
    vocab_file_i2w_name = 'vocab_file_i2w.txt'
    vocab_file_w2i_name = 'vocab_file_w2i.txt'
    
    fashiondata.create_vocab(fenci_filename = os.path.join(save_path, fenci_result_fname) , vocab_file_i2w = os.path.join(save_path,vocab_file_i2w_name ), vocab_file_w2i = os.path.join(save_path,vocab_file_w2i_name))
    #vocab_size = fashiondata.vocab_size(word2id)
    #content_file = '/home/tongjianing/lm/data/taobao_headline_0412/'
    word2id_filepath =  os.path.join(save_path,vocab_file_w2i_name) #=vocab_file_w2i 
    data_dir_prefix = '/data/share/tongjianing/'
    max_content_len = 5000
    
    
    f = open(os.path.join(save_path,fenci_result_fname))   #对分词结果进行w2i
    data_file_line = f.readline()
    max_sentence_len = 0
    
    with open(word2id_filepath, 'r') as f1:
        w2i = json.load(f1)
        #print(len(w2i))
        while data_file_line:
            if len(data_file_line) > max_sentence_len:
            max_sentence_len = len(data_file_line)
            result = fashiondata.preprocess(w2i, data_file_line,  max_content_len, delimiter=",")
            data_file_line = f.readline()
            #print(data)
            fashiondata.save_data(result, (data_dir_prefix + "result.txt"))

    print(max_sentence_len)   
    print('*********Done with saving file!*******************')
