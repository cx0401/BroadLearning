from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
from keras.datasets import reuters
import scipy.io as scio
import numpy as np
import torch
import pandas as pd


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


class matData():

    def loadmat(self, dataFile):
        data = scio.loadmat(dataFile)
        self.train_x = np.double(data['train_x'] / 255)
        self.train_y = np.double(data['train_y'])
        self.test_x = np.double(data['test_x'] / 255)
        self.test_y = np.double(data['test_y'])
        return

    def save(self, output, data):
        return output.write(data)


class Twitterdata():
    def __init__(self, out_dim):
        data = pd.read_csv('data/Tweets.csv')
        data.head()
        reviews = np.array(data['text'])[:14000]
        labels = np.array(data['airline_sentiment'])[:14000]
        data['text'].loc[14639]
        data['airline_sentiment'].loc[14639]
        from collections import Counter
        Counter(labels)
        punctuation = '!"#$%&\'()*+,-./:;<=>?[\\]^_`{|}~'
        all_reviews = 'separator'.join(reviews)
        all_reviews = all_reviews.lower()
        all_text = ''.join([c for c in all_reviews if c not in punctuation])
        reviews_split = all_text.split('separator')
        all_text = ' '.join(reviews_split)
        words = all_text.split()
        new_reviews = []
        for review in reviews_split:
            review = review.split()
            new_text = []
            for word in review:
                if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
                    new_text.append(word)
            new_reviews.append(new_text)
        counts = Counter(words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
        reviews_ints = []
        for review in new_reviews:
            reviews_ints.append([vocab_to_int[word] for word in review])
        encoded_labels = []
        for label in labels:
            if label == 'neutral':
                encoded_labels.append(1)
            elif label == 'negative':
                encoded_labels.append(0)
            else:
                encoded_labels.append(1)

        encoded_labels = np.asarray(encoded_labels)

        def pad_features(reviews_ints, seq_length):
            features = np.zeros((len(reviews_ints), seq_length), dtype=int)
            for i, row in enumerate(reviews_ints):
                features[i, -len(row):] = np.array(row)[:seq_length]

            return features

        seq_length = 30

        features = pad_features(reviews_ints, seq_length=seq_length)

        assert len(features) == len(reviews_ints), "The features should have as many rows as reviews."
        assert len(features[0]) == seq_length, "Each feature row should contain seq_length values."

        split_frac = 0.8

        split_idx = int(len(features) * split_frac)
        train_x, remaining_x = features[:split_idx], features[split_idx:]
        train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

        test_idx = int(len(remaining_x) * 0.5)
        val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
        val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

        self.seq_len = seq_length
        self.vocab_to_int = vocab_to_int
        self.vocab_size = len(vocab_to_int) + 1
        self.punctuation = punctuation
        self.out_dim = out_dim
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.val_x = val_x
        self.val_y = val_y

    def pad_features(self, reviews_ints, seq_length):
        features = np.zeros((len(reviews_ints), seq_length), dtype=int)
        for i, row in enumerate(reviews_ints):
            features[i, -len(row):] = np.array(row)[:seq_length]

        return features

    def one_hot(self, y, out_dim):
        y = torch.tensor(y)
        y = torch.unsqueeze(y, 1)
        y = torch.zeros(y.shape[0], out_dim).scatter_(1, y, 1)
        return y.detach().numpy()

    def data_convert(self):
        self.train_y = self.one_hot(self.train_y, self.out_dim)
        self.test_y = self.one_hot(self.test_y, self.out_dim)
        self.val_y = self.one_hot(self.val_y, self.out_dim)


class reuterskerasdata():
    def __init__(self, batchsize=256, seq_len=200, out_dim=2):
        MAX_WORDS = 10002  # imdb’s vocab_size 即词汇表大小
        MAX_LEN = seq_len  # max length
        (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=MAX_WORDS)
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


class attentiondata():
    def __init__(self, batchsize=256, seq_len=200, out_dim=2):
        batch_size = 32
        seed = 1111
        save = './bilstm_attn_model'
        data = './data/corpus.pt'

        torch.manual_seed(seed)

        data = torch.load(data)

        train_x = data['train']['src']
        test_x = data['valid']['src']
        for i in train_x:
            while len(i) < data["max_len"]:
                i.append(0)
        for i in test_x:
            while len(i) < data["max_len"]:
                i.append(0)
        self.train_x = np.array(train_x)
        self.test_x = np.array(test_x)
        self.train_y = np.array(data['train']['label'])
        self.test_y = np.array(data['valid']['label'])
        self.out_dim = data['dict']['label_size']
        self.vocab_size = data['dict']['vocab_size']
        self.seq_len = data["max_len"]

    def data_convert(self):
        y = torch.tensor(self.train_y)
        y = torch.unsqueeze(y, 1)
        y = torch.zeros(y.shape[0], self.out_dim).scatter_(1, y, 1)
        self.train_y = y.detach().numpy()

        y = torch.tensor(self.test_y)
        y = torch.unsqueeze(y, 1)
        y = torch.zeros(y.shape[0], self.out_dim).scatter_(1, y, 1)
        self.test_y = y.detach().numpy()
