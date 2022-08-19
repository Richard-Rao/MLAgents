

import scipy
from scipy import sparse
import numpy as np
from collections import Counter
import string

# utility functions we provides

def load_data(file_name):
    '''
    @input:
     file_name: a string. should be either "training.txt" or "texting.txt"
    @return:
     a list of sentences
    '''
    with open(file_name, "r") as file:
        sentences = file.readlines()
    return sentences


def tokenize(sentence):
    # Convert a sentence into a list of words
    wordlist = sentence.translate(str.maketrans('', '', string.punctuation)).lower().strip().split(
        ' ')

    return [word.strip() for word in wordlist]


# Main "Feature Extractor" class:
# It takes the provided tokenizer and vocab as an input.

class feature_extractor:
    def __init__(self, vocab, tokenizer):
        self.tokenize = tokenizer
        self.vocab = vocab  # This is a list of words in vocabulary
        self.vocab_dict = {item: i for i, item in
                           enumerate(vocab)}  # This constructs a word 2 index dictionary
        self.d = len(vocab)

    def bag_of_word_feature(self, sentence):
        '''
        Bag of word feature extactor
        :param sentence: A text string representing one "movie review"
        :return: The feature vector in the form of a "sparse.csc_array" with shape = (d,1)
        '''

        words = tokenize(sentence)
        indexRow = np.zeros(len(set(words)))
        data = np.ones(len(set(words)))
        i = 0
        for word in set(words):
            try:
                index = self.vocab_dict[word]
                indexRow[i] = index
                data[i] = words.count(word)
                i+=1
            except KeyError:
                pass
        
        indexColumn = np.zeros(len(set(words)))
        x = sparse.csc_matrix((data,(indexRow, indexColumn)), shape=(self.d, 1))

        return x


    def __call__(self, sentence):
        
        return self.bag_of_word_feature(sentence)


class classifier_agent():
    def __init__(self, feat_map, params):
        #Constructor
        self.feat_map = feat_map
        self.params = np.array(params)

    def batch_feat_map(self, sentences):
        #Processes Data
        if isinstance(sentences, list):
            X = scipy.sparse.hstack([self.feat_map(sentence) for sentence in sentences])
        else:
            X = self.feat_map(sentences)
        return X

    def score_function(self, X):
        #Computes Score

        (d,m) = X.shape
        s = np.zeros(shape=m) 
       
        return self.params.T @ X

        return s



    def predict(self, X, RAW_TEXT=False, RETURN_SCORE=False):
        #Makes a binary prediction/Score
        if RAW_TEXT:
            X = self.batch_feat_map(X)
        
        preds = np.zeros(shape=X.shape[1])
        
        scores = self.score_function(X)
        
        if RETURN_SCORE:
            return scores
        
        for i in range(len(scores)):
            if scores[i] > 0:
                preds[i] = 1


        return preds


    def error(self, X, y, RAW_TEXT=False):
        #Calculates average error
        if RAW_TEXT:
            X = self.batch_feat_map(X)
            y = np.array(y)

        prediction = self.predict(X, False, False)
        numErrors = 0
        for i in range(len(prediction)):
            if prediction[i] != y[i]:
                numErrors += 1


        err =  numErrors/len(prediction)


        return err


    def loss_function(self, X, y):
        #calculates the logistic loss with current params

        score = self.score_function(X)
        loss = 0

        p_hat = np.exp(score) / (1 + np.exp(score))
        
        for i in range(len(y)):
            loss += -( np.log(p_hat) * y[i] + np.log(1-p_hat) * (1 - y[i]))
        


        return loss/len(y)

    def gradient(self, X, y):
        
        #gradient at current params

        score = self.score_function(X) #m by 1


        grad = np.zeros_like(self.params)
        
        grad = (X @ (np.exp(score) / (1+np.exp(score))-y)) / len(y)

        
        return grad


    def train_gd(self, train_sentences, train_labels, niter, lr=0.01):
        #Trains the model niter times using normal Gradient Descent

        Xtrain = self.batch_feat_map(train_sentences)
        ytrain = np.array(train_labels)
        train_losses = [self.loss_function(Xtrain, ytrain)]
        train_errors = [self.error(Xtrain, ytrain)]

        for i in range(niter):
            grad = self.gradient(Xtrain, ytrain)
            self.params -= lr * grad


        return train_losses, train_errors


    def train_sgd(self, train_sentences, train_labels, nepoch, lr=0.001):
        #trains the model niter times using Stochastic Gradient Descent(minibatches)
        

        Xtrain = self.batch_feat_map(train_sentences)
        ytrain = np.array(train_labels)
        train_losses = [self.loss_function(Xtrain, ytrain)]
        train_errors = [self.error(Xtrain, ytrain)]
        
        for i in range(nepoch): 
            for j in range(len(ytrain)):
                idx = np.random.choice(len(ytrain), 1)
                grad = self.gradient(Xtrain[:,idx], np.array(ytrain[idx]))
                self.params -= lr * grad


        return train_losses, train_errors


    def eval_model(self, test_sentences, test_labels):
        #Evaluates the current model on the test data
        
        X = scipy.sparse.hstack([self.feat_map(sentence) for sentence in test_sentences])
        y = np.array(test_labels)
        return self.error(X, y)

    def save_params_to_file(self, filename):
        # The filename should be *.npy
        with open(filename, 'wb') as f:
            np.save(f, self.params)

    def load_params_from_file(self, filename):
        with open(filename, 'rb') as f:
            self.params = np.load(f)

