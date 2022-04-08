import torch
import numpy as np


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
        self.train_y = np.numpy(data['train']['label'])
        self.test_y = np.numpy(data['valid']['label'])
        for i in train_x:
            while len(i) < data["max_len"]:
                i.append(0)
        for i in test_x:
            while len(i) < data["max_len"]:
                i.append(0)
        self.train_x = np.numpy(train_x)
        self.test_x = np.numpy(test_x)
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
