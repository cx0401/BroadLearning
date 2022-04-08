import numpy as np
import pandas as pd
import torch


class Twitterdata():
    def __init__(self, out_dim):
        data = pd.read_csv('../data/Tweets.csv')
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

    def one_hot(self, y, out_dim):
        y = torch.tensor(y)
        y = torch.unsqueeze(y, 1)
        y = torch.zeros(y.shape[0], out_dim).scatter_(1, y, 1)
        return y.detach().numpy()

    def data_convert(self):
        self.train_y = self.one_hot(self.train_y, self.out_dim)
        self.test_y = self.one_hot(self.test_y, self.out_dim)
        self.val_y = self.one_hot(self.val_y, self.out_dim)