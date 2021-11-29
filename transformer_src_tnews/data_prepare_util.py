
import json
import numpy as np
# 数据预处理的类，生成训练集和测试集


class Dataset(object):
    def __init__(self, config):
        self.labels = config.labels
        self.num_class = config.numClasses
        self.dict_path = '../data/tnews/vocab.txt'
        self.tokenizer = config.tokenizer
        self.max_len = config.sequenceLength
        self.trainX = []
        self.trainY = []
        self.evalX = []
        self.evalY = []

    def _genTrainEvalData(self, x):
        x_pad = []
        y_pad = []
        for i, (news, lab) in enumerate(x):
            train_x = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(news))
            if len(train_x) > self.max_len:
                x_pad.append(train_x[:self.max_len])
            else:
                x_pad.append(train_x + [0] * (self.max_len - len(train_x)))

            y_pad.append(lab)
        return x_pad, y_pad

    def load_data(self, filename):
        D = []
        with open(filename, encoding="utf8") as f:
            for i, l in enumerate(f):
                l = json.loads(l)
                text, label = l['sentence'], l['label']
                D.append((text, self.labels.index(label)))
        return D

    def dataGen(self):
        train_data = self.load_data(
            '../data/tnews/train.json'
        )
        valid_data = self.load_data(
            '../data/tnews/dev.json'
        )

        trainX, train_Y = self._genTrainEvalData(train_data)
        evalX, evalY = self._genTrainEvalData(valid_data)
        self.trainX, self.trainY, self.evalX, self.evalY = np.asarray(trainX, dtype="int64"), \
                                                           np.asarray(train_Y,dtype="float32"), \
                                                           np.asarray(evalX, dtype="int64"), \
                                                           np.asarray(evalY,dtype="float32")


if __name__ == "__main__":
    da = Dataset()
    da.dataGen()
    import pdb;
    pdb.set_trace()
