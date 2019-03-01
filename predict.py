"""
CSCI 630(Foundation of Intelligent Systems)
Lab 2
@author: Shubham Patil (sbp5931)
"""

import sys
import train
import re
import pickle

class Predict:
    __slots__ = ["model", "feature_matrix", "features", "attributes"]

    def __init__(self, model, test):
        with open(model, 'rb') as file:
            self.model = pickle.load(file)
        train = []
        self.feature_matrix = []
        with open(test, encoding="utf8") as file:
            for line in file:
                line = re.sub('[^A-Za-z]+', ' ', line)
                line = line.split()
                train.append(line)
        self.read_features(train)
        self.features = {"contains_to": 1, "contains_and": 2, "contains_the": 3, "contains_de": 4,
                         "contains_een": 5, "contains_het": 6, "avg_len_ls_five": 7, "words_start_with_z_v_d": 8,
                         "contains_single_char": 9, "contains_many_ij": 10}
        self.attributes = list(self.features.keys())
        for line in self.feature_matrix:
            self.predict(line, self.model)

    def predict(self, feature, node):
        if node == "en":
            print("en")
        if node == "nl":
            print("nl")
        if type(node) == train.DecisionTree:
            ind = self.features[node.val]
            x = feature[ind]
            if not x:
                self.predict(feature, node.left)
            else:
                self.predict(feature, node.right)

    def read_features(self, train):
        for line in train:
            line = line[1:]
            features = ['True']
            self.contains_to(line, features)
            self.contains_and(line, features)
            self.contains_the(line, features)
            self.contains_de(line, features)
            self.contains_een(line, features)
            self.contains_het(line, features)
            self.avg_len_ls_five(line, features)
            self.words_start_with_z_v_d(line, features)
            self.contains_single_char(line, features)
            self.contains_many_ij(line, features)
            self.feature_matrix.append(features)

    def contains_many_ij(self, line, features):
        en = 0
        for word in line:
            if 'ij' in word:
                en += 1
        if en > 1:
            features.append(True)
        else:
            features.append(False)

    def contains_single_char(self, line, features):
        single = 0
        for word in line:
            if len(word) == 1:
                single += 1
        if single >= 1:
            features.append(True)
        else:
            features.append(False)

    def words_start_with_z_v_d(self, line, features):
        zvd = 0
        for word in line:
            z = word.startswith('z')
            v = word.startswith('v')
            d = word.startswith('d')
            if z or v or d:
                zvd += 1
        if zvd > 2:
            features.append(True)
        else:
            features.append(False)

    def contains_to(self, line, features):
        flag = False
        for word in line:
            if word == "to":
                flag = True
                features.append(flag)
                break
        if not flag:
            features.append(False)

    def contains_and(self, line, features):
        flag = False
        for word in line:
            if word == "and":
                flag = True
                features.append(flag)
                break
        if not flag:
            features.append(False)

    def contains_the(self, line, features):
        flag = False
        for word in line:
            if word == "the":
                flag = True
                features.append(flag)
                break
        if not flag:
            features.append(False)

    def contains_de(self, line, features):
        flag = False
        for word in line:
            if word == "de":
                flag = True
                features.append(flag)
                break
        if not flag:
            features.append(False)

    def contains_een(self, line, features):
        flag = False
        for word in line:
            if word == "een":
                flag = True
                features.append(flag)
                break
        if not flag:
            features.append(False)

    def contains_het(self, line, features):
        flag = False
        for word in line:
            if word == "het":
                flag = True
                features.append(flag)
                break
        if not flag:
            features.append(False)

    def avg_len_ls_five(self, line, features):
        sum = 0
        for word in line:
            sum += len(word)
        avg = sum/len(line)
        if avg < 5:
            features.append(True)
        else:
            features.append(False)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Format: 'python predict.py <hypothesis> <test>'")
        print("eg. for decision_tree 'python predict.py dt.pkl test.dat'")
        print("eg. for adaboost 'python predict.py ada.pkl test.dat'")
        sys.exit(1)
    test = sys.argv[2]
    model = sys.argv[1]
    obj = Predict(model, test)

