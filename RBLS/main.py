from BroadLearningSystem import BLS
from BLS import BLS, RBLS
import utils.utils as utils
from predata.datapre import imdbkerasdata, Twitterdata, matData, reuterskerasdata, attentiondata


def loadData(data_file):
    myData = matData()
    myData.loadmat(data_file)
    return myData


def twitter():
    args = utils.parse_args()
    print(args)
    # myData = loadData(args.data_file)
    myData = Twitterdata(args.out_dim)
    myData.data_convert()
    model = "model/lstm_train" + str(args.epochs) + "hidden" + str(args.hidden_dim) + ".pth"
    print("------BLS-------Twitter")
    bls = BLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R)
    bls.loadTwitterRNN(vocab_size=myData.vocab_size, output_size=args.out_dim, embedding_dim=args.embedding_dim,
                       hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    bls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    bls.train()
    bls.test()

    print("------RBLS------Twitter")
    rbls = RBLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R,
                seqLen=myData.seq_len, embeddingDim=args.embedding_dim)
    rbls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    rbls.loadTwitterRNN(vocab_size=myData.vocab_size, output_size=args.out_dim, embedding_dim=args.embedding_dim,
                        hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    rbls.train()
    rbls.test()


def ImdbKeras():
    args = utils.parse_args()
    print(args)
    myData = imdbkerasdata(seq_len=args.seq_len, out_dim=args.out_dim)
    myData.data_convert()
    model = "model/imdbkeras_em" + str(args.embedding_dim) + "train" + str(args.epochs) + "hidden" + str(
        args.hidden_dim) + "seq_len" + str(args.seq_len) + "out_dim" + str(args.out_dim) + ".pth"
    print("------BLS-------IMDB")
    bls = BLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R)
    bls.loadImdbkerasRNN(vocab_size=myData.vocab_size, output_size=args.out_dim, embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    bls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    bls.train()
    bls.test()

    print("------RBLS------IMDB")
    rbls = RBLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R,
                seqLen=myData.seq_len, embeddingDim=args.embedding_dim)
    rbls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    rbls.loadImdbkerasRNN(vocab_size=myData.vocab_size, output_size=args.out_dim, embedding_dim=args.embedding_dim,
                          hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    rbls.train()
    rbls.test()


def ReutersKeras():
    args = utils.parse_args()
    print(args)
    myData = reuterskerasdata(seq_len=args.seq_len, out_dim=args.out_dim)
    myData.data_convert()
    model = "model/reuterskeras_em" + str(args.embedding_dim) + "train" + str(args.epochs) + "hidden" + str(
        args.hidden_dim) + "seq_len" + str(args.seq_len) + "out_dim" + str(args.out_dim) + ".pth"
    print("------BLS-------Reuters")
    bls = BLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R)
    bls.loadreuterskerasRNN(vocab_size=myData.vocab_size, output_size=args.out_dim, embedding_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    bls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    bls.train()
    bls.test()

    print("------RBLS------Reuters")
    rbls = RBLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R,
                seqLen=myData.seq_len, embeddingDim=args.embedding_dim)
    rbls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    rbls.loadreuterskerasRNN(vocab_size=myData.vocab_size, output_size=args.out_dim, embedding_dim=args.embedding_dim,
                             hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model)
    rbls.train()
    rbls.test()


def AttentionText():
    args = utils.parse_args()
    print(args)
    myData = attentiondata(seq_len=args.seq_len, out_dim=args.out_dim)
    myData.data_convert()
    if args.use_attention:
        model = "model/corpus_em" + str(args.embedding_dim) + "train" + str(args.epochs) + "hidden" + str(
            args.hidden_dim) + "seq_len" + str(args.seq_len) + "attention_size" + str(args.attention_size)
    else:
        model = "model/corpus_em" + str(args.embedding_dim) + "train" + str(args.epochs) + "hidden" + str(
            args.hidden_dim) + "seq_len" + str(args.seq_len)
    if args.bidirectional:
        model += "bi"
    model += ".pth"
    print("------BLS-------corpus")
    bls = BLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R)
    bls.loadcorpusRNN(vocab_size=myData.vocab_size, output_size=args.out_dim, embedding_dim=args.embedding_dim,
                       hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model,
                       use_attention=args.use_attention, attention_size=args.attention_size, bidirectional=args.bidirectional)
    bls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    bls.train()
    bls.test()
    print("------RBLS------corpus")
    rbls = RBLS(windowSize=args.windowSize, windowNum=args.windowNum, enhanceNum=args.enhanceNum, S=args.S, R=args.R,
                seqLen=myData.seq_len, embeddingDim=args.embedding_dim)

    rbls.loadcorpusRNN(vocab_size=myData.vocab_size, output_size=args.out_dim, embedding_dim=args.embedding_dim,
                       hidden_dim=args.hidden_dim, n_layers=args.n_layers, device=args.device, model=model,
                       use_attention=args.use_attention, attention_size=args.attention_size, bidirectional=args.bidirectional)
    rbls.getData(myData.train_x, myData.train_y, myData.test_x, myData.test_y)
    rbls.train()
    rbls.test()


if __name__ == "__main__":
    AttentionText()
