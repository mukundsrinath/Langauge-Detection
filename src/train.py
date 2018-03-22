import numpy as np
import os
import math
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import gc

class LanguageDetection(object):
    def __init__(self):
        self.vocabulary = set()
        self.lang_dict = {}
        self.lang_list = []
        self.root_dir = 'europarl\\txt\\'
        self.extract_files()
        self.create_features()
        self.train_model()
        joblib.dump(self.vocabulary, 'vocab.pkl')

    def build_vocabulary(self, language, str):
        str = str.split()
        three_grams = []
        for i in str:
            for j in range(0, len(i), 3):
                three_grams.append(i[j:j+3])
        if language in self.lang_dict:
            self.build_dictionary(three_grams, language)
        else:
            self.lang_dict[language] = []
            self.build_dictionary(three_grams, language)

    def build_dictionary(self, three_gram, language):
        document_ngram= {}
        for gram in three_gram:
            if gram in document_ngram:
                document_ngram[gram] += 1
            else:
                document_ngram[gram] = 1
                self.vocabulary.add(gram)
        self.lang_dict[language].append(document_ngram)

    def create_features(self):

        self.number_docs = 0
        for i in self.lang_dict:
            self.number_docs += len(self.lang_dict[i])
        idf = np.zeros(len(self.vocabulary))
        self.data = np.zeros((self.number_docs, len(self.vocabulary)+1))
        self.vocabulary = list(self.vocabulary)
        self.lang_list = list(self.lang_dict.keys())
        joblib.dump(self.lang_list, 'lang_list.pkl')
        for i, gram in enumerate(self.vocabulary):
            for lang in self.lang_list:
                for doc in self.lang_dict[lang]:
                    if gram in doc:
                        idf[i] += 1
        for i, gram in enumerate(self.vocabulary):
            index = 0
            for lang in self.lang_list:
                for doc in self.lang_dict[lang]:
                    if gram in doc:
                        self.data[index, i] = (doc[gram]/sum(doc.values()))*math.log(self.number_docs/(1+idf[i]))
                    index += 1

    def train_model(self):
        y = np.zeros(self.number_docs)
        index = 0
        for lang in self.lang_list:
            for i in self.lang_dict[lang]:
                y[index] = self.lang_list.index(lang)
                index += 1
        self.lang_dict = {}
        self.data[:,-1] = y

        gc.collect()

        np.random.shuffle(self.data)
        training = self.data[0:int(0.8 * len(self.data)), :]
        testing = self.data[int(0.8 * len(self.data)):, :]
        data_x_train = training[:, 0:-1]
        data_y_train = training[:, -1]
        data_x_test = testing[:, 0:-1]
        data_y_test = testing[:, -1]
        print(data_y_train)
        lregression = LogisticRegression()
        lregression = lregression.fit(data_x_train, data_y_train)
        print(lregression.score(data_x_test, data_y_test))
        print(lregression.score(data_x_train, data_y_train))
        joblib.dump(lregression, 'regression.pkl')

    def extract_files(self):
        dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        for lang_num, language in enumerate(dirs):
            files = [file for file in os.listdir(self.root_dir+language)]
            index = 0
            for text_file in files:
                with open(self.root_dir+language+'\\'+text_file, 'r', encoding='utf-8') as f:
                    s = f.read()
                    self.build_vocabulary(language, s)
                index+=1
                if index > 500:
                    break
            print(lang_num)

LanguageDetection()