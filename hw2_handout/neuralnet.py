import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchtext
import re
import random

from scipy.sparse import csr_matrix, vstack
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec, KeyedVectors
from collections import Counter

# Module for data processing and model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext.vocab import Vectors, Vocab
from torchtext.vocab import GloVe
from torchtext.data import TabularDataset
from torchtext.data import Iterator, BucketIterator

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
#device = torch.device("cuda:0" if use_cuda else "cpu")
device = torch.device("cuda:1" if use_cuda else "cpu")
print(device)

#Define functions
def preprocess(data):
    word = [] #list of cleaned token 
    text = [] #list of cleaned text
    
    for line in data:
        content_text = re.sub(r'\([^)]*\)', '', line) 
        sent_text = sent_tokenize(content_text)

        normalized_text = []
        for string in sent_text:
            tokens = re.sub(r'[^A-Za-z0-9\s]+', '', string.lower())
            tokens = re.sub(r'\d+', '', tokens)
            normalized_text.append(tokens)

        result_content = ' '.join(normalized_text)
        result_sentence = [word_tokenize(sentence) for sentence in normalized_text]
        result = [word for sentence in result_sentence for word in sentence]

        word.append(result)
        text.append(result_content)
    
    return word, text

# Setup model
#Embedding = 'W2V'
Embedding = 'G6B' 
Model = 'LSTM'

# Setup hyper-parameters
NUM_WORDS = 1000
NUM_DIM = 100
BATCH_SIZE = 64
NUM_CLASS = 2
EPOCHS = 50

# Define model
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=True,
                           batch_first=True,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, sent len, emb dim]
        output, (hidden, cell) = self.rnn(embedded)
        # output = [batch size, sent len, hid dim * num directions]
        # hidden = [batch size, num layers * num directions, hid dim]
        # cell = [batch size, num layers * num directions, hid dim]
        # concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)


def train(model, optimizer, train_iter):
    model.train()
    corrects, total_loss = 0, 0
    for batch in train_iter:
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(train_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


def evaluate(model, val_iter):
    model.eval()
    corrects, total_loss = 0, 0
    with torch.no_grad():
        for batch in val_iter:
            x, y = batch.text.to(device), batch.label.to(device)
            y.data.sub_(1)
            logit = model(x)
            loss = F.cross_entropy(logit, y, reduction='sum')
            total_loss += loss.item()
            corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

def predict(model, eval_data):
    eval_iter_pred = Iterator(eval_data, batch_size=len(eval_data),
                              sort_key=lambda x: len(x.text),
                              sort_within_batch = False,
                              shuffle=False, repeat=False,
                              device = device)    
    batch = next(iter(eval_iter_pred))
    x = batch.text.to(device)
    logit = model(x)
    pred_idx =logit.max(1)[1].data.tolist()
    pred = np.array([LABEL.vocab.itos[idx+1] for idx in pred_idx])
    return pred


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('sys.argv[1]: dev_text.txt')
        print('sys.argv[2]: dev_label.txt')
        print('sys.argv[3]: heldout_text.txt')
        print('sys.argv[4]: heldout_pred_nb.txt')
        quit()

    #To setup nltk
    import nltk
    nltk.download('punkt')
        
    dev_text_file = sys.argv[1]
    dev_label_file = sys.argv[2]
    heldout_text_file = sys.argv[3]
    heldout_pred_file = sys.argv[4]

    #Load data
    with open(dev_label_file, 'rt', encoding='UTF8') as f:
        dev_label = np.asarray(f.readlines())
        f.close()
    with open(dev_text_file, 'rt', encoding='UTF8') as f:
        dev_text = np.asarray(f.readlines())
        f.close()
    with open(heldout_text_file, 'rt', encoding='UTF8') as f:
        heldout_text = np.asarray(f.readlines())
        f.close()

    dev_label = [label.replace('\n', '') for label in dev_label]
    dev_text = [text.replace('\n', '') for text in dev_text]
    heldout_text = [text.replace('\n', '') for text in heldout_text]
    
    dev_word_list, dev_text_list = preprocess(dev_text)
    heldout_word_list, heldout_text_list = preprocess(heldout_text)

    #Data into pd.DataFrame format
    Train_word = pd.DataFrame(columns=['label', 'word'])
    Train_text = pd.DataFrame(columns=['label', 'text'])
    for label, text, word in zip(dev_label, dev_text_list, dev_word_list):
        Train_word = Train_word.append([{'label': label, 'word': word}], ignore_index=True, sort=False)
        Train_text = Train_text.append([{'label': label, 'text': text}], ignore_index=True, sort=False)

    Test_word = pd.DataFrame(columns=['label', 'word'])
    Test_text = pd.DataFrame(columns=['label', 'text'])
    for text, word in zip(heldout_text_list, heldout_word_list):
        Test_word = Test_word.append([{'label': None, 'word': word}], ignore_index=True, sort=False)
        Test_text = Test_text.append([{'label': None, 'text': text}], ignore_index=True, sort=False)

    #Random sampling for train data
    Train_text_pos = Train_text[Train_text['label']=='pos']
    Train_text_neg = Train_text[Train_text['label']=='neg']
    Train_text_sampling = Train_text.copy()
    for i in range(10):
        np.random.shuffle(Train_text_pos.values)
        pos_sample = Train_text_pos.sample(n=100)
        Train_text_sampling = Train_text_sampling.append(pos_sample)

        np.random.shuffle(Train_text_neg.values)
        neg_sample = Train_text_neg.sample(n=100)
        Train_text_sampling = Train_text_sampling.append(neg_sample)
    np.random.shuffle(Train_text_sampling.values)
    
    #Random shuffle for train data
    np.random.shuffle(Train_text.values)

    # Save as a csv format 
    #Train_text.to_csv('train.csv', index=False)
    Train_text_sampling.to_csv('train.csv', index=False)
    Test_text.to_csv('test.csv', index=False)

    # Define train_set / val_set / test_set
    #train_set_df, validation_set_df = train_test_split(Train_text, test_size=0.2, random_state=SEED)
    train_set_df, validation_set_df = train_test_split(Train_text_sampling, test_size=0.2, random_state=SEED)
    test_set_df = Test_text.copy()

    # Save as a csv format
    train_set_df.to_csv('train_set_df.csv', index=False)
    validation_set_df.to_csv('validation_set_df.csv', index=False)
    test_set_df.to_csv('test_set_df.csv', index=False)
    
    #Use W2V or pretrained embedding G6B
    if Embedding == 'W2V':
        # Create word2vector model
        W2V_model = Word2Vec(sentences=Train_word['word'], size=100, window=5, min_count=5, sg=0)
        # Load W2V
        W2V_model.wv.save_word2vec_format('w2v_model')
        loaded_model = KeyedVectors.load_word2vec_format('w2v_model')
        W2V_model = loaded_model
        print('----- Save Word2Vector embedding model -----')    
    
    # Define Torchtext structure
    TEXT = torchtext.data.Field(sequential=True, use_vocab=True,
                                tokenize=str.split, lower=True,
                                batch_first=True, fix_length=NUM_WORDS)

    LABEL = torchtext.data.Field(sequential=False, use_vocab=True,
                                batch_first=False, is_target=True)
    
    # Load data
    train_data, valid_data, test_data = TabularDataset.splits(
        path='.', train='train_set_df.csv', validation='validation_set_df.csv', test='test_set_df.csv',
        format='csv', fields=[('label', LABEL), ('text', TEXT)], skip_header=True)
    
    # Build vocab with embedding vector
    if Embedding == 'W2V':
        w2v_vectors = Vectors('w2v_model')
        TEXT.build_vocab(train_data, vectors=w2v_vectors, min_freq=5)
    elif Embedding == 'G6B':
        TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=100) , min_freq=5, max_size=10000)
    LABEL.build_vocab(train_data)

    VOCAB_SIZE = len(TEXT.vocab)
    print('Vocabulary Size: {}'.format(VOCAB_SIZE))

    # Define data bucket and iterator
    train_iter, valid_iter = BucketIterator.splits(
                                            (train_data, valid_data),
                                            batch_size = BATCH_SIZE,
                                            sort_key=lambda x: len(x.text),
                                            sort_within_batch = False,
                                            shuffle=True, repeat=False,
                                            device = device)

    print('The number of mini-batch in train_data : {}'.format(len(train_iter)))
    print('The number of mini-batch in validation_data : {}'.format(len(valid_iter)))   

    # Define model
    INPUT_DIM = VOCAB_SIZE
    EMBEDDING_DIM = NUM_DIM
    HIDDEN_DIM = 100
    OUTPUT_DIM = 2
    N_LAYERS = 2
    DROPOUT = 0.5

    model = RNN(INPUT_DIM,
                EMBEDDING_DIM,
                HIDDEN_DIM,
                OUTPUT_DIM,
                N_LAYERS,
                DROPOUT)
    model.to(device)

    # Setup embedding parameters
    model.embedding.weight.data.copy_(TEXT.vocab.vectors)
    model.embedding.weight.data[0] = (torch.rand(EMBEDDING_DIM)-0.5)*0.001 #<unk>
    model.embedding.weight.data[1] = torch.zeros(EMBEDDING_DIM) #<pad>

    # Train and Evaluate Model
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

    best_val_loss = None
    train_out = []
    valid_out = []
    test__out = []
    for e in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train(model, optimizer, train_iter)
        valid_loss, valid_accuracy = evaluate(model, valid_iter)

        train_out.append([train_loss, train_accuracy])
        valid_out.append([valid_loss, valid_accuracy])

        if e%10==0:
            #print("[Epoch: %d] train loss : %3.3f | train accuracy : %3.3f" % (e, train_loss, train_accuracy))
            print("[Epoch: %d] valid loss : %3.3f | valid accuracy : %3.3f" % (e, valid_loss, valid_accuracy))

        if not best_val_loss or valid_loss < best_val_loss:
            if not os.path.isdir("snapshot"):
                os.makedirs("snapshot")
            torch.save(model.state_dict(), './snapshot/LSTM_classification.pt')
            best_val_loss = valid_loss

    ## Save Figure
    plt.figure()
    plt.plot(np.array(train_out)[:,0])
    plt.plot(np.array(valid_out)[:,0])
    plt.legend(['train', 'valid'])
    plt.title('loss'+'_'+str(Embedding)+'_'+str(Model)+'_'+str(EPOCHS))
    plt.savefig('loss'+'_'+str(Embedding)+'_'+str(Model)+'_'+str(EPOCHS)+'.png')
    #plt.show()

    plt.figure()
    plt.ylim((45,105))
    plt.plot(np.array(train_out)[:,1])
    plt.plot(np.array(valid_out)[:,1])
    plt.legend(['train', 'valid'])
    plt.title('accuracy'+'_'+str(Embedding)+'_'+str(Model)+'_'+str(EPOCHS))
    plt.savefig('accuracy'+'_'+str(Embedding)+'_'+str(Model)+'_'+str(EPOCHS)+'.png')
    #plt.show()    
    
    #Predict results
    model.load_state_dict(torch.load('./snapshot/LSTM_classification.pt'))    
    
    pred_train = predict(model, train_data)
    y_train = np.array([data.label for data in train_data.examples])
    print(' LSTM Train score: {:.5f}'.format(np.sum(pred_train==y_train)/len(y_train)))

    pred_valid = predict(model, valid_data)
    y_valid = np.array([data.label for data in valid_data.examples])
    print(' LSTM Train score: {:.5f}'.format(np.sum(pred_valid==y_valid)/len(y_valid)))
    
    pred_heldout = predict(model, test_data)

    #Save Predict
    with open(heldout_pred_file, 'w') as f:
        for l in pred_heldout:
            f.write(str(l)+'\n')
        f.close()
    print('Prediction saved: {}'.format(heldout_pred_file))
    