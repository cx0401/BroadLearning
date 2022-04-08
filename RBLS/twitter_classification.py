import time
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from predata.datapre import Twitterdata
import utils.utils as utils
from net.RNNet import SentimentRNN
import torch.nn as nn

# dataloaders
args = utils.parse_args()
batch_size = args.batch_size
output_size = args.out_dim
embedding_dim = args.embedding_dim
hidden_dim = args.hidden_dim
n_layers = args.n_layers
seq_length = args.seq_len
epochs = args.epochs
lstm_data = Twitterdata(output_size)

train_x, train_y = lstm_data.train_x, lstm_data.train_y
val_x, val_y = lstm_data.val_x, lstm_data.val_y
test_x, test_y = lstm_data.test_x, lstm_data.test_y
vocab_to_int = lstm_data.vocab_to_int
punctuation = lstm_data.punctuation

# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# make sure the SHUFFLE the training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# In[15]:


# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size())  # batch_size, seq_length
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size())  # batch_size
print('Sample label: \n', sample_y)

train_on_gpu = torch.cuda.is_available()

if (train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# Instantiate the model w/ hyperparams
vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding + our word tokens



net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

print(net)

# ### Training
# 
# Below is the typical training code. We'll use a cross entropy loss, which is designed to work with a single Sigmoid output. [BCELoss](https://pytorch.org/docs/stable/nn.html#bceloss), or **Binary Cross Entropy Loss**, applies cross entropy loss to a single value between 0 and 1. We also have some data and training hyparameters:
# 
# * `lr`: Learning rate for our optimizer.
# * `epochs`: Number of times to iterate through the training dataset.
# * `clip`: The maximum gradient value to clip at (to prevent exploding gradients).

# In[19]:


# loss and optimization functions
lr = 0.001

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

# In[20]:


# training params


counter = 0
print_every = 100
clip = 5  # gradient clipping

# move model to GPU, if available
if (train_on_gpu):
    net.cuda()
time_start = time.time()
net.train()
# train for some number of epochs
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        if (train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output, labels)
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        num_correct = 0
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if (train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output, labels)

                val_losses.append(val_loss.item())
                pred = torch.argmax(output, axis=1)
                num_correct += (pred == labels).sum().item()

            net.train()
            print("Epoch: {}/{}...".format(e + 1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)),
                  "ACC: {:.6f}".format(num_correct / len(valid_loader.dataset))
                  )

# ### Testing
# 
# We'll see how our trained model performs on all of our defined test_data, above. We'll calculate the average loss and accuracy over the test data.

# In[21]:


# Get test data loss and accuracy
print("train over,time:", time.time() - time_start)
time_start = time.time()
test_losses = []  # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if (train_on_gpu):
        inputs, labels = inputs.cuda(), labels.cuda()

    # get predicted outputs
    output, h = net(inputs, h)

    # calculate loss
    test_loss = criterion(output, labels)
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.argmax(output, axis=1)
    num_correct += (pred == labels).sum().item()

# -- stats! -- ##
# avg test loss
print("Test over, time:", time.time() - time_start)
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

# ---
# ## Creating a predict function
# We'll write a predict function that takes in a trained net, a plain text_review, and a sequence length, and prints out a custom statement for a non-negative or negative review.

# In[22]:


# negative test review
test_review = "@AmericanAir you have my money, you change my flight, and don't answer your phones! Any other suggestions so I can make my commitment??"


# In[23]:


def tokenize_review(test_review):
    test_review = test_review.lower()  # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # get rid of web address, twitter id, and digit
    new_text = []
    for word in test_words:
        if (word[0] != '@') & ('http' not in word) & (~word.isdigit()):
            new_text.append(word)

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in new_text])

    return test_ints


# test code and generate tokenized review
test_ints = tokenize_review(test_review)
print(test_ints)

# In[24]:


# test sequence padding

features = lstm_data.pad_features(test_ints, seq_length)

print(features)

# In[25]:


# test conversion to tensor and pass into your model
feature_tensor = torch.from_numpy(features)
print(feature_tensor.size())


# In[26]:


def predict(net, test_review, sequence_length=30):
    net.eval()

    # tokenize review
    test_ints = tokenize_review(test_review)

    # pad tokenized sequence
    seq_length = sequence_length
    features = lstm_data.pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    if (train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.argmax(output, axis=1)

    # print custom response
    if (pred == 1):
        print("Non-negative review detected.")
    else:
        print("Negative review detected.")


# In[27]:


seq_length = 30  # good to use the length that was trained on

# In[28]:


# call function on negative review
test_review_neg = "@AmericanAir you have my money, you change my flight, and don't answer your phones! Any other suggestions so I can make my commitment??"
predict(net, test_review_neg, seq_length)

# In[29]:


# call function on positive review
test_review_pos = "@AmericanAir thank you we got on a different flight to Chicago."
predict(net, test_review_pos, seq_length)

# In[30]:


# call function on neutral review
test_review_neu = "@AmericanAir i need someone to help me out"
predict(net, test_review_neu, seq_length)

torch.save(obj=net.state_dict(), f="model/lstm_train" + str(epochs) + "hidden" + str(hidden_dim) + ".pth")
