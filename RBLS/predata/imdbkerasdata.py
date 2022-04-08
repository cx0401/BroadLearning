import torch
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb

class imdbkerasdata():
    def __init__(self, batchsize=256, seq_len=200, out_dim=2):
        MAX_WORDS = 10002  # imdb’s vocab_size 即词汇表大小
        MAX_LEN = 200  # max length
        batchsize = 256
        EMB_SIZE = 128  # embedding size
        HID_SIZE = 128  # lstm hidden size
        DROPOUT = 0.2
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)
        x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
        x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding="post", truncating="post")
        print(x_train.shape, x_test.shape)

        self.train_x = x_train
        self.train_y = y_train
        self.test_x = x_test
        self.test_y = y_test
        self.out_dim = out_dim
        self.vocab_size = MAX_WORDS
        self.seq_len = seq_len

    def data_convert(self):
        y = torch.tensor(self.train_y)
        y = torch.unsqueeze(y, 1)
        y = torch.zeros(y.shape[0], self.out_dim).scatter_(1, y, 1)
        self.train_y = y.detach().numpy()

        y = torch.tensor(self.test_y)
        y = torch.unsqueeze(y, 1)
        y = torch.zeros(y.shape[0], self.out_dim).scatter_(1, y, 1)
        self.test_y = y.detach().numpy()


