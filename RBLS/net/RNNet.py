import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torchtext.vocab as vocab


# # cache_dir是保存golve词典的缓存路径
# cache_dir = '.vector_cache/glove'
# # dim是embedding的维度
# glove = vocab.GloVe(name='6B', dim=300, cache=cache_dir)

class TwitterNet(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, out_dim, embedding_dim, hidden_dim, n_layers, drop_prob=0.5, device='cpu'):
        """
        Initialize the model by setting up the layers.
        """
        super(TwitterNet, self).__init__()

        self.out_dim = out_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embedding and LSTM layers
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True)

        # dropout layer
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.fc2 = nn.Linear(256, out_dim)
        self.sig = nn.Sigmoid()
        self.device = device

    def forward(self, x, hidden):
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.permute(1, 0, 2)
        lstm_out = lstm_out[-1]
        sig_out = self.sig(self.fc1(lstm_out))
        out = self.fc2(sig_out)
        # return last sigmoid output and hidden state
        return out, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden

    def feature_extract(self, x):
        x = torch.tensor(x).to(self.device)
        bz = x.shape[0]
        h0 = torch.zeros((self.n_layers, bz, self.hidden_dim)).to(self.device)  # bz这里应该是batch大小
        c0 = torch.zeros((self.n_layers, bz, self.hidden_dim)).to(self.device)
        # 然后将词嵌入交给lstm模型处理
        r_o, _ = self.lstm(x, (h0, c0))
        r_o = r_o.permute(1, 0, 2)
        return r_o.cpu().detach().numpy()

    def embed_extract(self, x):
        x = torch.tensor(x).to(self.device)
        x = self.embedding(x)
        return x.cpu().detach().numpy()

    def reshape_extract(self, x):
        x = torch.tensor(x).to(self.device)
        x = x.reshape([x.shape[0], -1])
        return x.cpu().detach().numpy()


class ImdbNet(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout=0.1, out_dim=2, n_layers=2):
        super(ImdbNet, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.Embedding = nn.Embedding(self.max_words, self.emb_size)
        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=n_layers,
                            batch_first=True, bidirectional=False)  # 2层双向LSTM
        self.dp = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, out_dim)

    def forward(self, x):
        """
        input : [bs, maxlen]
        output: [bs, 2]
        """
        x = self.Embedding(x)  # [bs, ml, emb_size]
        x = self.dp(x)
        x, _ = self.LSTM(x)  # [bs, ml, 2*hid_size]
        x = self.dp(x)
        x = F.relu(self.fc1(x))  # [bs, ml, hid_size]
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()  # [bs, 1, hid_size] => [bs, hid_size]
        out = self.fc2(x)  # [bs, 2]
        return out  # [bs, 2]

    def feature_extract(self, x):
        x = torch.tensor(x)
        x = self.dp(x)
        x, _ = self.LSTM(x)  # [bs, ml, 2*hid_size]
        x = x.permute(1, 0, 2)
        return x.cpu().detach().numpy()

    def embed_extract(self, x):
        x = torch.tensor(x)
        x = self.Embedding(x)
        return x.cpu().detach().numpy()

    def reshape_extract(self, x):
        x = torch.tensor(x)
        x = x.reshape([x.shape[0], -1])
        return x.cpu().detach().numpy()


class ReutersNet(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout=0.1, out_dim=46):
        super(ReutersNet, self).__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.Embedding = nn.Embedding(self.max_words, self.emb_size)
        self.LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=2,
                            batch_first=True, bidirectional=False)  # 2层双向LSTM
        self.dp = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, out_dim)

    def forward(self, x):
        """
        input : [bs, maxlen]
        output: [bs, 2]
        """
        x = self.Embedding(x)  # [bs, ml, emb_size]
        x = self.dp(x)
        x, _ = self.LSTM(x)  # [bs, ml, 2*hid_size]
        x = self.dp(x)
        x = F.relu(self.fc1(x))  # [bs, ml, hid_size]
        x = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()  # [bs, 1, hid_size] => [bs, hid_size]
        out = self.fc2(x)  # [bs, 2]
        return out  # [bs, 2]

    def feature_extract(self, x):
        x = torch.tensor(x)
        x = self.dp(x)
        x, _ = self.LSTM(x)  # [bs, ml, 2*hid_size]
        x = x.permute(1, 0, 2)
        return x.cpu().detach().numpy()

    def embed_extract(self, x):
        x = torch.tensor(x)
        x = self.Embedding(x)
        return x.cpu().detach().numpy()

    def reshape_extract(self, x):
        x = torch.tensor(x)
        x = x.reshape([x.shape[0], -1])
        return x.cpu().detach().numpy()


class CorpusNet(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, use_attention, attention_size, bidirectional=True,
                 dropout=0.5, seq_len=16, out_dim=6, device = 'cuda'):
        super(CorpusNet, self).__init__()
        self.use_attention = use_attention
        self.out_dim = out_dim
        self.hidden_size = hidden_dim
        self.vocab_size = vocab_size
        self.embed_dim = embedding_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.sequence_length = seq_len
        self.Embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.Embedding.weight.data.uniform_(-1., 1.)
        # self.Embedding = nn.Embedding(glove.vectors.size(0), glove.vectors.size(1))
        # self.Embedding.weight.data.copy_(glove.vectors)

        self.layer_size = 1
        self.LSTM = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

        self.attention_size = attention_size
        self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size)).to(device)
        self.u_omega = Variable(torch.zeros(self.attention_size)).to(device)
        self.sig = nn.Sigmoid()
        self.label = nn.Linear(hidden_dim * self.layer_size, out_dim)

    # self.attn_fc_layer = nn.Linear()

    def attention_net(self, lstm_output):
        # print(lstm_output.size()) = (squence_length, batch_size, hidden_size*layer_size)
        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size * self.layer_size])
        # print(output_reshape.size()) = (squence_length * batch_size, hidden_size*layer_size)
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        # print(attn_tanh.size()) = (squence_length * batch_size, attention_size)
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        # print(attn_hidden_layer.size()) = (squence_length * batch_size, 1)
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        # print(exps.size()) = (batch_size, squence_length)
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        # print(alphas.size()) = (batch_size, squence_length)
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        # print(alphas_reshape.size()) = (batch_size, squence_length, 1)
        state = lstm_output.permute(1, 0, 2)
        # print(state.size()) = (batch_size, squence_length, hidden_size*layer_size)
        attn_output = torch.sum(state * alphas_reshape, 1)
        # print(attn_output.size()) = (batch_size, hidden_size*layer_size)

        return attn_output

    def forward(self, input_sentences, batch_size=None):
        input = self.Embedding(input_sentences)
        bz = input.shape[0]
        input = input.permute(1, 0, 2)
        h_0 = Variable(torch.zeros(self.layer_size, bz, self.hidden_size)).cuda()
        c_0 = Variable(torch.zeros(self.layer_size, bz, self.hidden_size)).cuda()

        lstm_output, (final_hidden_state, final_cell_state) = self.LSTM(input, (h_0, c_0))
        if self.use_attention:
            attn_output = self.attention_net(lstm_output)
        else:
            attn_output = self.sig(lstm_output[-1])
        logits = self.label(attn_output)
        return logits

    def feature_extract(self, x):
        x = torch.tensor(x)
        x = x.permute(1, 0, 2)
        out, _ = self.LSTM(x)  # [bs, ml, 2*hid_size]
        if self.use_attention:
            attention_out = self.attention_net(out).unsqueeze(0)
            out = torch.cat((out, attention_out), 0)
        return out.cpu().detach().numpy()

    def embed_extract(self, x):
        x = torch.tensor(x)
        x = self.Embedding(x)
        return x.cpu().detach().numpy()

    def reshape_extract(self, x):
        x = torch.tensor(x)
        x = x.reshape([x.shape[0], -1])
        return x.cpu().detach().numpy()
