from BroadLearningSystem import BLS
from BLS import BLS, RBLS
import utils.utils as utils
from predata.datapre import ImdbData, TwitterData, matData, ReutersData, CorpusData


def loadData(data_file):
    myData = matData()
    myData.loadmat(data_file)
    return myData


def TwitterText():
    args = utils.TwitterUtils()
    print(args)
    # myData = loadData(args.data_file)
    myData = TwitterData()
    myData.data_convert()
    model = "model/Twitter/Train" + str(args.epochs) + "_Embed" + str(args.embedding_dim) + "_Hidden" + str(
            args.hidden_dim) + "_Seq" + str(args.seq_len) + ".pth"
    print("------BLS-------Twitter")
    bls = BLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R)
    bls.loadTwitterRNN(vocab_size=myData.vocab_size, out_dim=myData.out_dim, embedding_dim=args.embedding_dim,
                       hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    bls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    bls.train()
    bls.test()

    print("------RBLS------Twitter")
    rbls = RBLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R,
                seqLen=myData.seq_len, embeddingDim=args.embedding_dim)
    rbls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    rbls.loadTwitterRNN(vocab_size=myData.vocab_size, out_dim=myData.out_dim, embedding_dim=args.embedding_dim,
                        hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    rbls.train()
    rbls.test()


def ImdbText():
    args = utils.ImdbUtils()
    print(args)
    myData = ImdbData(seq_len=args.seq_len)
    myData.data_convert()
    model = "model/Imdb/Train" + str(args.epochs) + "_Embed" + str(args.embedding_dim) + "_Hidden" + str(
        args.hidden_dim) + "_Seq" + str(args.seq_len) + ".pth" # 定义模型保存路径
    print("------BLS-------IMDB")
    bls = BLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R)
    bls.loadImdbkerasRNN(vocab_size=myData.vocab_size, out_dim=myData.out_dim, embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    bls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    bls.train()
    bls.test()

    print("------RBLS------IMDB")
    rbls = RBLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R,
                seqLen=myData.seq_len, embeddingDim=args.embedding_dim)
    rbls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    rbls.loadImdbkerasRNN(vocab_size=myData.vocab_size, out_dim=myData.out_dim, embedding_dim=args.embedding_dim,
                          hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    rbls.train()
    rbls.test()


def ReutersText():
    args = utils.ReutersUtils()
    print(args)
    myData = ReutersData(seq_len=args.seq_len, out_dim=args.out_dim)
    myData.data_convert()
    model = "model/Reuters/Train" + str(args.epochs) + "_Embed" + str(args.embedding_dim) + "_Hidden" + str(
        args.hidden_dim) + "_Seq" + str(args.seq_len) + ".pth" # 定义模型保存路径
    print("------BLS-------Reuters")
    bls = BLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R)
    bls.loadreuterskerasRNN(vocab_size=myData.vocab_size, out_dim=args.out_dim, embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    bls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    bls.train()
    bls.test()

    print("------RBLS------Reuters")
    rbls = RBLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R,
                seqLen=myData.seq_len, embeddingDim=args.embedding_dim)
    rbls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    rbls.loadreuterskerasRNN(vocab_size=myData.vocab_size, out_dim=args.out_dim, embedding_dim=args.embedding_dim,
                             hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    rbls.train()
    rbls.test()


def CorpusText():
    args = utils.CorpusUtils()
    print(args)
    myData = CorpusData(seq_len=args.seq_len)
    myData.data_convert()
    if args.use_attention:
        model = "model/Corpus/Train" + str(args.epochs) + "_Embed" + str(args.embedding_dim) + "_Hidden" + str(
            args.hidden_dim) + "_Seq" + str(args.seq_len) + "_Attention" + str(args.attention_size)
    else:
        model = "model/Corpus/Train" + str(args.epochs) + "_Embed" + str(args.embedding_dim) + "_Hidden" + str(
            args.hidden_dim) + "_Seq" + str(args.seq_len)
    if args.bidirectional:
        model += "_bi"
    model += ".pth"
    print("------BLS-------corpus")
    bls = BLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R)
    bls.loadcorpusRNN(vocab_size=myData.vocab_size, out_dim=myData.out_dim, embedding_dim=args.embedding_dim,
                       hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model,
                       use_attention=args.use_attention, attention_size=args.attention_size, bidirectional=args.bidirectional)
    bls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    bls.train()
    bls.test()
    print("------RBLS------corpus")
    rbls = RBLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R,
                seqLen=myData.seq_len, embeddingDim=args.embedding_dim)

    rbls.loadcorpusRNN(vocab_size=myData.vocab_size, out_dim=myData.out_dim, embedding_dim=args.embedding_dim,
                       hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model,
                       use_attention=args.use_attention, attention_size=args.attention_size, bidirectional=args.bidirectional)
    rbls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    rbls.train()
    rbls.test()

def run(data, model):
    args = utils.parse_args()
    print(args)
    # myData = loadData(args.data_file)
    myData = data(args.out_dim)
    myData.data_convert()
    model = "model/lstm_train" + str(args.epochs) + "hidden" + str(args.hidden_dim) + ".pth"
    print("------BLS-------Twitter")
    bls = BLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R)
    bls.loadTwitterRNN(vocab_size=myData.vocab_size, out_dim=args.out_dim, embedding_dim=args.embedding_dim,
                       hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    bls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    bls.train()
    bls.test()

    print("------RBLS------Twitter")
    rbls = RBLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R,
                seqLen=myData.seq_len, embeddingDim=args.embedding_dim)
    rbls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    rbls.loadTwitterRNN(vocab_size=myData.vocab_size, out_dim=args.out_dim, embedding_dim=args.embedding_dim,
                        hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    rbls.train()
    rbls.test()


if __name__ == "__main__":
    # TwitterText()
    # ImdbText()
    # ReutersText()
    CorpusText()