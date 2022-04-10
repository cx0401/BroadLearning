import argparse


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
        self.epochs = 0
        self.seq_len = 16
        self.windowNum = 16
        self.n_layers = 2
        self.bidirectional = False
        self.use_attention = False
        self.attention_size = 16

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Broad Learning System',
        usage='main.py [<args>] [-h | --help]'
    )
    # Twitter
    # parser.add_argument('--vocab_size', default=10002, type=int)
    # parser.add_argument('--batch_size', default=100, type=int)
    # parser.add_argument('--windowSize', default=10)
    # parser.add_argument('--enhanceNum', default=10)
    # parser.add_argument('--R', default=2 ** -10)
    # parser.add_argument('--S', default=0.8)
    # parser.add_argument('--embedding_dim', default=200)
    # parser.add_argument('--hidden_dim', default=400)
    # parser.add_argument('--device', default='cpu')
    # parser.add_argument('--epochs', default=10, type=int)
    # parser.add_argument('--seq_len', default=30)
    # parser.add_argument('--windowNum', default=30)
    # parser.add_argument('--n_layers', default=2)

    # IMDB
    # parser.add_argument('--vocab_size', default=10002, type=int)
    # parser.add_argument('--batch_size', default=100, type=int)
    # parser.add_argument('--windowSize', default=10)
    # parser.add_argument('--enhanceNum', default=10)
    # parser.add_argument('--R', default=2 ** -10)
    # parser.add_argument('--S', default=0.8)
    # parser.add_argument('--embedding_dim', default=128)
    # parser.add_argument('--hidden_dim', default=128)
    # parser.add_argument('--device', default='cpu')
    # parser.add_argument('--epochs', default=10, type=int)
    # parser.add_argument('--out_dim', default=2)
    # parser.add_argument('--seq_len', default=200)
    # parser.add_argument('--windowNum', default=200)
    # parser.add_argument('--n_layers', default=2)

    # Reuters
    # parser.add_argument('--vocab_size', default=10002, type=int)
    # parser.add_argument('--batch_size', default=100, type=int)
    # parser.add_argument('--windowSize', default=10)
    # parser.add_argument('--enhanceNum', default=10)
    # parser.add_argument('--R', default=2 ** -10)
    # parser.add_argument('--S', default=0.8)
    # parser.add_argument('--embedding_dim', default=128)
    # parser.add_argument('--hidden_dim', default=128)
    # parser.add_argument('--device', default='cpu')
    # parser.add_argument('--epochs', default=20, type=int)
    # parser.add_argument('--out_dim', default=46)
    # parser.add_argument('--seq_len', default=200)
    # parser.add_argument('--windowNum', default=200)
    # parser.add_argument('--n_layers', default=2)

    # corpus
    parser.add_argument('--vocab_size', default=10002, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--windowSize', default=100)
    parser.add_argument('--enhanceNum', default=500)
    parser.add_argument('--R', default=2 ** -10)
    parser.add_argument('--S', default=0.8)
    parser.add_argument('--embedding_dim', default=128)
    parser.add_argument('--hidden_dim', default=128)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', default=0, type=int)
    parser.add_argument('--seq_len', default=16)
    parser.add_argument('--windowNum', default=16)
    parser.add_argument('--n_layers', default=2)
    parser.add_argument('--use_attention', default=True)
    parser.add_argument('--attention_size', default=16)
    parser.add_argument('--bidirectional', default=False)

    return parser.parse_args(args)

# 当采用lstm_train1hidden400这个pth时，效果最好，且winSize越小越好，enhance也是如此
# 当采用imdbkeras_em128train0hidden128seq_len200out_dim2.pth，效果最好，比hidden400还要好
