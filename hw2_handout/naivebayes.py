import os
import sys
import numpy as np
import pandas as pd

#import torch
import re
import random

from scipy.sparse import csr_matrix, vstack
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter

SEED = 0
np.random.seed(SEED)
#torch.manual_seed(SEED)
random.seed(SEED)

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
    
    #Random shuffle for train data
    np.random.shuffle(Train_text.values)
        
    #Token list
    train_all_token_list = [token for text in Train_word['word'] for token in text]

    #Build vocab from pos and neg
    #2000 each, unique size is 2500
    pos_token_list = [token for words in Train_word[Train_word['label']=='pos']['word'] for token in words]
    pos_vocab = Counter(pos_token_list).most_common(2000)
    pos_vocab = [c[0] for c in pos_vocab]
    
    neg_token_list = [token for words in Train_word[Train_word['label']=='neg']['word'] for token in words]
    neg_vocab = Counter(neg_token_list).most_common(2000)
    neg_vocab = [c[0] for c in neg_vocab]

    vocab = np.unique(pos_vocab+neg_vocab)
    vocab_dict = dict(zip(vocab, range(len(vocab))))
    
    VOCAB_SIZE = len(vocab_dict)
    print('Vocabulary Size: {}'.format(VOCAB_SIZE))
    
    #Set TF-IDF
    vectorizer = TfidfVectorizer(vocabulary=vocab_dict, ngram_range=(1,1))
    vectorizer.fit(train_all_token_list)

    Train_feature = vectorizer.transform(Train_text['text'])
    Test_feature = vectorizer.transform(Test_text['text'])
    
    train_data = []
    for i in range(Train_feature.shape[0]):
        train_data.append([Train_text['label'].iloc[i], Train_feature.getrow(i)])
    train_data = np.asarray(train_data)
    np.random.shuffle(train_data)    
    
    #Define data to train and test
    train_set, validation_set = train_test_split(train_data, test_size=0.2, random_state=SEED)
    test_set = Test_feature.copy()

    #Random Sampling for train data
    train_set_pos = train_set[train_set[:,0]=='pos']
    train_set_neg = train_set[train_set[:,0]=='pos']
    train_set_sampling = [train_set.copy()]
    for i in range(10):
        pos_sample = train_set_pos[np.random.choice(range(len(train_set_pos)), 100),:]
        train_set_sampling.append(pos_sample)

        neg_sample = train_set_pos[np.random.choice(range(len(train_set_neg)), 100),:]
        train_set_sampling.append(neg_sample)
    train_set_sampling = np.vstack(train_set_sampling)
    np.random.shuffle(train_set_sampling)
    
    y_train = train_set[:,0]
    X_train = vstack(train_set[:,1]).toarray()
    y_validation = validation_set[:,0]
    X_validation = vstack(validation_set[:,1]).toarray()
    X_test = test_set.toarray()
    
    skf = StratifiedKFold(n_splits=10)
    params = {'alpha':[0.1, 0.5, 1.0, 2.0]}
    nb = MultinomialNB()
    gs = GridSearchCV(nb, cv=skf, param_grid=params, return_train_score=True)
    
    nb.fit(X_train, y_train)
    print('  Naive Bayes Train score: {:.5f}'.format(nb.score(X_train, y_train)))
    print('  Naive Bayes Valid score: {:.5f}'.format(nb.score(X_validation, y_validation)))
    
    gs.fit(X_train, y_train)
    gs.score(X_validation, y_validation)
    print('  Naive Bayes K-fold Train score: {:.5f}'.format(gs.score(X_train, y_train)))
    print('  Naive Bayes K-fold Valid score: {:.5f}'.format(gs.score(X_validation, y_validation)))

    #Save Predict
    pred_heldout = gs.predict(X_test)
    with open(heldout_pred_file, 'w') as f:
        for l in pred_heldout:
            f.write(str(l)+'\n')
        f.close()   
    print('Prediction saved: {}'.format(heldout_pred_file))