class TwitterUtils():
    def __init__(self):
        self.vocab_size = 10002
        self.batch_size = 100
        self.windowSize = 10
        self.enhanceNum = 10
        self.R = 2 ** -10
        self.S = 0.8
        self.embedding_dim = 200
        self.hidden_dim = 400
        self.device = 'cpu'
        self.epochs = 10
        self.seq_len = 30
        self.windowNum = 30
        self.n_layers = 2

class ImdbUtils():
    def __init__(self):
        self.vocab_size = 10002
        self.batch_size = 100
        self.windowSize = 10
        self.enhanceNum = 10
        self.R = 2 ** -10
        self.S = 0.8
        self.embedding_dim = 128
        self.hidden_dim = 128
        self.device = 'cpu'
        self.epochs = 10
        self.seq_len = 200
        self.windowNum = 200
        self.n_layers = 2

class ReutersUtils():
    def __init__(self):
        self.vocab_size = 10002
        self.batch_size = 100
        self.windowSize = 10
        self.enhanceNum = 10
        self.R = 2 ** -10
        self.S = 0.8
        self.embedding_dim = 128
        self.hidden_dim = 128
        self.device = 'cpu'
        self.epochs = 10
        self.seq_len = 200
        self.windowNum = 200
        self.n_layers = 2
        self.out_dim = 46

class CorpusUtils():
    def __init__(self):
        self.vocab_size = 10002
        self.batch_size = 100
        self.windowSize = 100
        self.enhanceNum = 500
        self.R = 2 ** -10
        self.S = 0.8
        self.embedding_dim = 128
        self.hidden_dim = 128
        self.device = 'cpu'
        self.epochs = 10
        self.seq_len = 16
        self.windowNum = 16
        self.n_layers = 2
        self.bidirectional = False
        self.use_attention = False
        self.attention_size = 16

# 当采用lstm_train1hidden400这个pth时，效果最好，且winSize越小越好，enhance也是如此
# 当采用imdbkeras_em128train0hidden128seq_len200out_dim2.pth，效果最好，比hidden400还要好
