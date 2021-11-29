import numpy as np
import re
import jieba


# 配置参数
class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 30
    checkpointEvery = 30
    learningRate = 0.001

class ModelConfig(object):
    embeddingSize = 100
    filters = 128  # 内层一维卷积核的数量，外层卷积核的数量应该等于embeddingSize，因为要确保每个layer后的输出维度和输入维度是一致的。
    numHeads = 8  # Attention 的头数
    numBlocks = 1  # 设置transformer block的数量
    epsilon = 1e-8  # LayerNorm 层中的最小除数
    keepProp = 0.9  # multi head attention 中的dropout
    dropoutKeepProb = 0.5  # 全连接层的dropout
    l2RegLambda = 0.0


class Config(object):
    sequenceLength = 12  # 取了所有序列长度的均值
    batchSize = 128
    dataSource = "../data/preProcess/dataForTrain.csv"
    stopWordSource = "../data/chinese"
    numClasses = 35  # 二分类设置为1，多分类设置为类别的数目
    rate = 0.8  # 训练集的比例
    savedModelPath = "../model/transformer/savedModel"
    ckptPath = "../model/transformer/model/my-model"
    training = TrainingConfig()
    model = ModelConfig()


# 输出batch数据集

def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """
    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY


# 生成位置嵌入
def fixedPositionEmbedding(batchSize, sequenceLen):
    embeddedPosition = []
    for batch in range(batchSize):
        x = []
        for step in range(sequenceLen):
            a = np.zeros(sequenceLen)
            a[step] = 1
            x.append(a)
        embeddedPosition.append(x)

    return np.array(embeddedPosition, dtype="float32")


def strB2Q(ustring):
    """半角转全角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 32:  # 半角空格直接转化
            inside_code = 12288
        elif inside_code >= 32 and inside_code <= 126:  # 半角字符（除空格）根据关系转化
            inside_code += 65248
        rstring += chr(inside_code)
    return rstring

def clean_text(text):
    text = strB2Q(text)
    text = re.sub('(（已通知.*?）)|(（已经通知.*?）)','',text)
    text = re.sub('（内线.*?）','',text)
    text = re.sub('（\d+）','',text)
    text = re.sub('（.*?消防(共|协)同(接|收)(听|通).*?）','',text)
    text = re.sub('（消防(同|共|接)听）','',text)
    text = re.sub('[【】（）。，１２５４７８６３９０ＡＢＣＤＥＦＧＨＩＪＫＬＭＮ\u3000]','',text)
    return text

def process_context(line_):
    line = clean_text(line_)
    line = " ".join(list(jieba.cut(line)))
    return line
