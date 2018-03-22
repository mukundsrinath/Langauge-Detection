import numpy as np
import pandas as pd
import math
from sklearn.externals import joblib
from sklearn.metrics import classification_report

class test(object):
    def __init__(self, vocabulary):
        data = pd.read_csv('europarl.test', sep='\t', header=None, encoding='utf=8')
        self.test_y = data.iloc[:,0].values
        self.target_names = list(set(self.test_y))
        lang_list = joblib.load('lang_list.pkl')
        self.y = np.zeros(len(self.test_y))
        for index, i in enumerate(self.test_y):
            self.y[index] = lang_list.index(i)
        test_x = data.iloc[:,1].values
        self.vocabulary = {}
        for index, i in enumerate(vocabulary):
            if i in self.vocabulary:
                pass
            else:
                self.vocabulary[i] = index
        self.test_data = np.zeros((len(data), len(vocabulary)))
        test_x = self.extract_ngrams(test_x)
        self.extract_features(test_x)
        self.test_model()

    def extract_ngrams(self, test_x):
        feature_data = []
        for str in test_x:
            str = str.split()
            three_grams = []
            for i in str:
                for j in range(0, len(i), 3):
                    three_grams.append(i[j:j + 3])
            feature_data.append(three_grams)
        return feature_data

    def extract_features(self, test_x):
        for index, row in enumerate(test_x):
            for gram in row:
                if gram in self.vocabulary:
                    self.test_data[index, self.vocabulary[gram]] += 1

        idf = np.sum(self.test_data != 0.0, axis = 0)
        for i, row in enumerate(self.test_data):
            number_of_terms = np.sum(row)
            if number_of_terms == 0:
                continue
            for j, column in enumerate(row):
                self.test_data[i,j] = (column/number_of_terms)*math.log(self.test_data.shape[0]/(1+idf[j]))

    def test_model(self):
        print('testing model')
        clf = joblib.load('regression.pkl')
        print(clf.score(self.test_data, self.y))
        y_predicted = clf.predict(self.test_data)
        print(classification_report(self.y, y_predicted, target_names=self.target_names))

vocab = joblib.load('vocab.pkl')
test(vocab)







