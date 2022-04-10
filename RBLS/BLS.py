import torch
import time
from numpy import random
from sklearn import preprocessing
import numpy as np
from scipy import linalg as LA
from net.RNNet import TwitterNet, ImdbNet, ReutersNet, CorpusNet


def shrinkage(a, b):
    z = np.maximum(a - b, 0) - np.maximum(-a - b, 0)
    return z


def sparse_bls(A, b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m, n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1, (ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk


def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def sigmoid(data):
    return 1.0 / (1 + np.exp(-data))


def linear(data):
    return data


def tanh(data):
    return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))


def relu(data):
    return np.maximum(data, 0)


def pinv(A, reg):
    return np.mat(reg * np.eye(A.shape[1]) + A.T.dot(A)).I.dot(A.T)


def show_accuracy(predictLabel, Label):
    count = 0
    label_1 = Label.argmax(axis=1)
    predlabel = predictLabel.argmax(axis=1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count / len(Label), 5))


def one_hot(y, out_dim):
    y = torch.tensor(y)
    y = torch.unsqueeze(y, 1)
    y = torch.zeros(y.shape[0], out_dim).scatter_(1, y, 1)
    return y.detach().numpy()


class BLS():
    def __init__(self, windowSize, windowNum, enhanceNum, S, R):
        self.ymax = 1
        self.ymin = 0
        self.windowSize = windowSize
        self.windowNum = windowNum
        self.enhanceNum = enhanceNum
        self.Beta1OfEachWindow = []
        self.distOfMaxAndMin = []
        self.minOfEachWindow = []
        self.S = S
        self.R = R
        self.parameterOfShrink = self.S
        if self.windowNum * self.windowSize >= self.enhanceNum:
            random.seed(67797325)
            self.weightOfEnhanceLayer = LA.orth(
                2 * random.randn(self.windowNum * self.windowSize + 1, self.enhanceNum)) - 1
        else:
            random.seed(67797325)
            self.weightOfEnhanceLayer = LA.orth(
                2 * random.randn(self.windowNum * self.windowSize + 1, self.enhanceNum).T - 1).T

    def getData(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def loadTwitterRNN(self, vocab_size, out_dim, embedding_dim, hidden_dim, n_layers, device, model):
        lstm = TwitterNet(vocab_size=vocab_size, out_dim=out_dim, embedding_dim=embedding_dim,
                          hidden_dim=hidden_dim, n_layers=n_layers,
                          device=device)
        lstm.to(device)
        pars = torch.load(model, map_location='cpu')
        lstm.load_state_dict(pars)
        self.lstm = lstm

    def loadImdbkerasRNN(self, vocab_size, out_dim, embedding_dim, hidden_dim, n_layers, device, model):
        lstm = ImdbNet(max_words=vocab_size, emb_size=embedding_dim, hid_size=hidden_dim)
        pars = torch.load(model, map_location='cpu')
        lstm.load_state_dict(pars)
        lstm.to(device)
        self.lstm = lstm

    def loadreuterskerasRNN(self, vocab_size, out_dim, embedding_dim, hidden_dim, n_layers, device, model):
        lstm = ReutersNet(max_words=vocab_size, emb_size=embedding_dim, hid_size=hidden_dim)
        pars = torch.load(model, map_location='cpu')
        lstm.load_state_dict(pars)
        lstm.to(device)
        self.lstm = lstm

    def loadcorpusRNN(self, vocab_size, out_dim, embedding_dim, hidden_dim, n_layers, device, model, use_attention,
                      attention_size, bidirectional):
        lstm = CorpusNet(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                         use_attention=use_attention, attention_size=attention_size, bidirectional=bidirectional,
                         device=device)
        pars = torch.load(model, map_location='cpu')
        lstm.load_state_dict(pars)
        lstm.to(device)
        self.lstm = lstm

    def featureMap(self, FeatureOfInputDataWithBias):
        OutputOfFeatureMappingLayer = np.zeros([self.train_x.shape[0], self.windowNum * self.windowSize])
        for i in range(self.windowNum):
            random.seed(i)
            weightOfEachWindow = 2 * random.randn(FeatureOfInputDataWithBias.shape[1], self.windowSize) - 1
            FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
            scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
            FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
            betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
            self.Beta1OfEachWindow.append(betaOfEachWindow)
            outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
            #        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
            self.distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
            self.minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
            outputOfEachWindow = (outputOfEachWindow - self.minOfEachWindow[i]) / self.distOfMaxAndMin[i]
            OutputOfFeatureMappingLayer[:, self.windowSize * i:self.windowSize * (i + 1)] = outputOfEachWindow
            del outputOfEachWindow
            del FeatureOfEachWindow
            del weightOfEachWindow
        return OutputOfFeatureMappingLayer

    def enhanceMap(self, OutputOfFeatureMappingLayer):
        InputOfEnhanceLayerWithBias = np.hstack(
            [OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0], 1))])
        tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, self.weightOfEnhanceLayer)
        #    print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))
        self.parameterOfShrink = self.S / np.max(tempOfOutputOfEnhanceLayer)
        OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * self.parameterOfShrink)
        return OutputOfEnhanceLayer

    def train(self):
        train_x = self.lstm.embed_extract(self.train_x)
        train_x = train_x.reshape(train_x.shape[0], -1)
        # train_x = preprocessing.scale(self.train_x, axis=1)
        FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0], 1))])
        time_start = time.time()  # 计时开始
        OutputOfFeatureMappingLayer = self.featureMap(FeatureOfInputDataWithBias)  # 得到特征层节点
        OutputOfEnhanceLayer = self.enhanceMap(OutputOfFeatureMappingLayer)  # 得到增强曾节点
        InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])  # 生成最终输入
        self.OutputWeight = np.dot(pinv(InputOfOutputLayer, self.R), self.train_y)
        time_end = time.time()
        trainTime = time_end - time_start
        OutputOfTrain = np.dot(InputOfOutputLayer, self.OutputWeight)
        trainAcc = show_accuracy(OutputOfTrain, self.train_y)
        print('Training accurate is', trainAcc * 100, '%')
        print('Training time is ', trainTime, 's')

    def test(self):
        test_x = self.lstm.embed_extract(self.test_x)
        test_x = test_x.reshape(test_x.shape[0], -1)
        # test_x = preprocessing.scale(self.test_x, axis=1)
        FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0], 1))])
        OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0], self.windowSize * self.windowNum])
        time_start = time.time()
        for i in range(self.windowNum):
            outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest, self.Beta1OfEachWindow[i])
            OutputOfFeatureMappingLayerTest[:, self.windowSize * i:self.windowSize * (i + 1)] = (
                                                                                                        self.ymax - self.ymin) * (
                                                                                                        outputOfEachWindowTest -
                                                                                                        self.minOfEachWindow[
                                                                                                            i]) / \
                                                                                                self.distOfMaxAndMin[
                                                                                                    i] - self.ymin

        InputOfEnhanceLayerWithBiasTest = np.hstack(
            [OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0], 1))])
        tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest, self.weightOfEnhanceLayer)
        OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * self.parameterOfShrink)
        InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
        OutputOfTest = np.dot(InputOfOutputLayerTest, self.OutputWeight)
        time_end = time.time()
        testTime = time_end - time_start
        testAcc = show_accuracy(OutputOfTest, self.test_y)
        print('Testing accurate is', testAcc * 100, '%')
        print('Testing time is ', testTime, 's')


class RBLS():
    def __init__(self, windowSize, windowNum, enhanceNum, S, R, seqLen, embeddingDim):
        self.seqLen = seqLen
        self.embeddingDim = embeddingDim
        self.ymax = 1
        self.ymin = 0
        self.windowSize = windowSize
        self.windowNum = windowNum
        self.enhanceNum = enhanceNum
        self.Beta1OfEachWindow = []
        self.distOfMaxAndMin = []
        self.minOfEachWindow = []
        self.S = S
        self.R = R
        self.parameterOfShrink = self.S
        if self.seqLen * self.embeddingDim >= self.enhanceNum:
            random.seed(67797325)
            self.weightOfEnhanceLayer = LA.orth(
                2 * random.randn(self.seqLen * self.embeddingDim + 1, self.enhanceNum)) - 1
        else:
            random.seed(67797325)
            self.weightOfEnhanceLayer = LA.orth(
                2 * random.randn(self.seqLen * self.embeddingDim + 1, self.enhanceNum).T - 1).T

    def getData(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def loadTwitterRNN(self, vocab_size, out_dim, embedding_dim, hidden_dim, n_layers, device, model):
        lstm = TwitterNet(vocab_size=vocab_size, out_dim=out_dim, embedding_dim=embedding_dim,
                          hidden_dim=hidden_dim, n_layers=n_layers,
                          device=device)
        pars = torch.load(model, map_location='cpu')
        lstm.load_state_dict(pars)
        lstm.to(device)
        self.lstm = lstm

    def loadImdbkerasRNN(self, vocab_size, out_dim, embedding_dim, hidden_dim, n_layers, device, model):
        lstm = ImdbNet(max_words=vocab_size, emb_size=embedding_dim, hid_size=hidden_dim)
        pars = torch.load(model, map_location='cpu')
        lstm.load_state_dict(pars)
        lstm.to(device)
        self.lstm = lstm

    def loadreuterskerasRNN(self, vocab_size, out_dim, embedding_dim, hidden_dim, n_layers, device, model):
        lstm = ReutersNet(max_words=vocab_size, emb_size=embedding_dim, hid_size=hidden_dim)
        pars = torch.load(model, map_location='cpu')
        lstm.load_state_dict(pars)
        lstm.to(device)
        self.lstm = lstm

    def loadcorpusRNN(self, vocab_size, out_dim, embedding_dim, hidden_dim, n_layers, device, model, use_attention,
                      attention_size, bidirectional):
        lstm = CorpusNet(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                         use_attention=use_attention, attention_size=attention_size, bidirectional=bidirectional,
                         device=device)
        pars = torch.load(model, map_location='cpu')
        lstm.load_state_dict(pars)
        lstm.to(device)
        self.lstm = lstm

    def featureMap(self, emdedding_x):
        feature_x = self.lstm.feature_extract(emdedding_x)
        OutputOfFeatureMappingLayer = np.zeros([self.train_x.shape[0], feature_x.shape[0] * self.windowSize])
        for i in range(feature_x.shape[0]):
            random.seed(i)
            FeatureOfInputDataWithBias = np.hstack([feature_x[i], 0.1 * np.ones((feature_x[i].shape[0], 1))])
            # FeatureOfInputDataWithBias = feature_x[i]
            weightOfEachWindow = 2 * random.randn(FeatureOfInputDataWithBias.shape[1], self.windowSize) - 1
            FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias, weightOfEachWindow)
            scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
            FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
            betaOfEachWindow = sparse_bls(FeatureOfEachWindowAfterPreprocess, FeatureOfInputDataWithBias).T
            self.Beta1OfEachWindow.append(betaOfEachWindow)
            outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, betaOfEachWindow)
            #        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
            self.distOfMaxAndMin.append(np.max(outputOfEachWindow, axis=0) - np.min(outputOfEachWindow, axis=0))
            self.minOfEachWindow.append(np.min(outputOfEachWindow, axis=0))
            outputOfEachWindow = (outputOfEachWindow - self.minOfEachWindow[i]) / self.distOfMaxAndMin[i]
            OutputOfFeatureMappingLayer[:, self.windowSize * i:self.windowSize * (i + 1)] = outputOfEachWindow
            del outputOfEachWindow
            del FeatureOfEachWindow
            del weightOfEachWindow
        return OutputOfFeatureMappingLayer

    def enhanceMap(self, embedding_x):
        reshape_x = self.lstm.reshape_extract(embedding_x)
        InputOfEnhanceLayerWithBias = np.hstack(
            [reshape_x, 0.1 * np.ones((reshape_x.shape[0], 1))])
        tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias, self.weightOfEnhanceLayer)
        #    print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))
        self.parameterOfShrink = self.S / np.max(tempOfOutputOfEnhanceLayer)
        OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * self.parameterOfShrink)
        return OutputOfEnhanceLayer

    def featureMapTest(self, emdedding_x):
        feature_x = self.lstm.feature_extract(emdedding_x)
        OutputOfFeatureMappingLayer = np.zeros([self.test_x.shape[0], feature_x.shape[0] * self.windowSize])
        for i in range(feature_x.shape[0]):
            random.seed(i)
            FeatureOfInputDataWithBias = np.hstack([feature_x[i], 0.1 * np.ones((feature_x[i].shape[0], 1))])
            outputOfEachWindow = np.dot(FeatureOfInputDataWithBias, self.Beta1OfEachWindow[i])
            outputOfEachWindow = (outputOfEachWindow - self.minOfEachWindow[i]) / self.distOfMaxAndMin[i]
            OutputOfFeatureMappingLayer[:, self.windowSize * i:self.windowSize * (i + 1)] = outputOfEachWindow
            del outputOfEachWindow
        return OutputOfFeatureMappingLayer

    def train(self):
        time_start = time.time()  # 计时开始
        embedding_x = self.lstm.embed_extract(self.train_x)
        OutputOfFeatureMappingLayer = self.featureMap(embedding_x)  # 得到特征层节点
        OutputOfEnhanceLayer = self.enhanceMap(embedding_x)  # 得到增强曾节点
        InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer])  # 生成最终输入
        self.OutputWeight = np.dot(pinv(InputOfOutputLayer, self.R), self.train_y)
        # self.OutputWeight = random.randn(InputOfOutputLayer.shape[1], self.train_y.shape[1])
        time_end = time.time()
        trainTime = time_end - time_start
        OutputOfTrain = np.dot(InputOfOutputLayer, self.OutputWeight)
        trainAcc = show_accuracy(OutputOfTrain, self.train_y)
        print('Training accurate is', trainAcc * 100, '%')
        print('Training time is ', trainTime, 's')

    def test(self):
        time_start = time.time()
        embedding_x = self.lstm.embed_extract(self.test_x)
        OutputOfFeatureMappingLayerTest = self.featureMapTest(embedding_x)
        OutputOfEnhanceLayerTest = self.enhanceMap(embedding_x)
        InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest, OutputOfEnhanceLayerTest])
        OutputOfTest = np.dot(InputOfOutputLayerTest, self.OutputWeight)
        time_end = time.time()
        testTime = time_end - time_start
        testAcc = show_accuracy(OutputOfTest, self.test_y)
        print('Testing accurate is', testAcc * 100, '%')
        print('Testing time is ', testTime, 's')
