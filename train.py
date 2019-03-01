"""
CSCI 630(Foundation of Intelligent Systems)
Lab 2
@author: Shubham Patil (sbp5931)
"""

import sys
import re
import pickle
from math import log


class DecisionTree:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class DTLearning:
    __slots__ = ["train", "model", "no_of_features", "length", "feature_matrix", "features",
                 "attributes", "training_type"]

    def __init__(self, train, model, training_type):
        self.train = []
        with open(train, encoding="utf8") as file:
            for line in file:
                line = re.sub('[^A-Za-z]+', ' ', line)
                line = line.split()
                self.train.append(line)
        self.model = model
        self.training_type = training_type
        self.no_of_features = None
        self.length = None
        self.feature_matrix = []
        self.read_features(self.train)
        self.features = {"contains_to": 1, "contains_and": 2, "contains_the": 3, "contains_de": 4,
                         "contains_een": 5, "contains_het": 6, "avg_len_ls_five": 7, "words_start_with_z_v_d": 8,
                         "contains_single_char": 9, "contains_many_ij": 10}
        self.attributes = list(self.features.keys())

    def read_features(self, train):
        self.length = len(train)
        cols = len(train[0])
        for line in train:
            label = line[0]
            line = line[1:]
            features = []
            features.append(label)
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

    def train_dt(self):
        if self.training_type == "dt":
            dt = self.dt_learning(self.feature_matrix, self.attributes, self.feature_matrix)
            return dt

    def train_ada(self):
        if self.training_type == "ada":
       #    ada = self.ada_learning(self.feature_matrix)
            ada = self.dt_learning(self.feature_matrix, self.attributes, self.feature_matrix)
            return ada

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

    def plurality_value(self, examples):
        en = 0
        nl = 0
        for line in examples:
            label = line[0]
            if label == "en":
                en += 1
            if label == "nl":
                nl += 1
        if en > nl:
            return "en"
        else:
            return "nl"


    def have_same_class(self, examples):
        temp = set()
        for line in examples:
            label = line[0]
            temp.add(label)
        value = list(temp)[0]
        if len(temp) == 1:
            return True, value
        else:
            return False, None

    def entropy_before(self, examples):
        en = 0
        nl = 0
        for line in examples:
            label = line[0]
            if label == "en":
                en += 1
            if label == "nl":
                nl += 1
        p_en = en / (en + nl)
        p_nl = nl / (en + nl)
        ent = -p_en * log(p_en, 2) - p_nl * log(p_nl, 2)
        return ent

    def entropy_after(self, examples, attribute):
        true_eng = 0
        true_nl = 0
        false_eng = 0
        false_nl = 0
        for line in examples:
            label = line[0]
            ind = self.features[attribute]
            if line[ind]:
                if label == "en":
                    true_eng += 1
                if label == "nl":
                    true_nl += 1
            if not line[ind]:
                if label == "en":
                    false_eng += 1
                if label == "nl":
                    false_nl += 1
        if (true_eng + true_nl) != 0:
            p_true_eng = true_eng / (true_eng + true_nl)
        else:
            p_true_eng = 0

        if (true_eng + true_nl) != 0:
            p_true_nl = true_nl / (true_eng + true_nl)
        else:
            p_true_nl = 0
        p_false_eng = false_eng / (false_eng + false_nl)
        p_false_nl = false_nl / (false_eng + false_nl)

        log_p_true_eng = 0
        if p_true_eng == 0:
            log_p_true_eng = 0
        else:
            log_p_true_eng = log(p_true_eng, 2)
        log_p_true_nl = 0
        if p_true_nl == 0:
            log_p_true_nl = 0
        else:
            log_p_true_nl = log(p_true_nl, 2)
        ent_true = -p_true_eng * log_p_true_eng - p_true_nl * log_p_true_nl

        log_p_false_eng = 0
        if p_false_eng == 0:
            log_p_false_eng = 0
        else:
            log_p_false_eng = log(p_false_eng, 2)
        log_p_false_nl = 0
        if p_false_eng == 0:
            log_p_false_nl = 0
        else:
            log_p_false_nl = log(p_false_nl, 2)
        ent_false = -p_false_eng * log_p_false_eng - p_false_nl * log_p_false_nl

        norm_true = (true_nl + true_eng)/ (true_nl + true_eng + false_nl + false_eng)
        norm_false = (false_nl + false_eng)/ (true_nl + true_eng + false_nl + false_eng)
        ent = norm_true * ent_true + norm_false * ent_false
        return ent

    def importance(self, attributes, examples):
        ent_bfr = self.entropy_before(examples)
        max = 0
        A = None
        for attribute in attributes:
            ent_aft = self.entropy_after(examples, attribute)
            info_gain = ent_bfr - ent_aft
            if info_gain > max:
                A = attribute
                max = info_gain
        return A

    def split(self, examples, attribute):
        left = []
        right = []
        for line in examples:
            ind = self.features[attribute]
            label = line[ind]
            if label is False:
                left.append(line)
            else:
                right.append(line)
        return left, right


    def dt_learning(self, examples, attributes, parent_examples ):
        if len(examples[0]) == 0:
            return self.plurality_value(parent_examples)
        elif self.have_same_class(examples)[0]:
            return self.have_same_class(examples)[1]
        elif len(attributes) == 0:
            return self.plurality_value(examples)
        else:
            A = self.importance(attributes, examples)
            root = DecisionTree(A)
            left_ex, right_ex = self.split(examples, A)
            attributes.remove(A)
            subtree_left = self.dt_learning(left_ex, attributes, examples)
            subtree_right = self.dt_learning(right_ex, attributes, examples)
            root.left = subtree_left
            root.right = subtree_right
        #print(root.left, root.val, root.right)
        return root

    def dt_ada(self, examples, attributes, parent_examples, depth=1):
        if depth > 1:
            return self.plurality_value(parent_examples)
        if len(examples[0]) == 0:
            return self.plurality_value(parent_examples)
        elif self.have_same_class(examples)[0]:
            return self.have_same_class(examples)[1]
        elif len(attributes) == 0:
            return self.plurality_value(examples)
        else:
            A = self.importance(attributes, examples)
            root = DecisionTree(A)
            left_ex, right_ex = self.split(examples, A)
            attributes.remove(A)
            subtree_left = self.dt_ada(left_ex, attributes, examples, depth+1)
            subtree_right = self.dt_ada(right_ex, attributes, examples, depth+1)
            root.left = subtree_left
            root.right = subtree_right
        return root


    def ada_learning(self, examples, k=10):
        n = len(examples)
        w = [float(1/n)] * n
        h = [None] * k
        z = [None] * k
        for i in range(k):
            h[i] = self.dt_ada(examples, self.attributes, examples)
            error = 0
            for j in range(n):
                label = self.feature_matrix[i][0]
                if label == h[i]:
                    error = error + w[j]
            for j in range(n):
                label = self.feature_matrix[n][0]
                if label == h[j]:
                    w[j] = w[j] * error / (1 - error)
            self.normalize()
            z[k] = log(1 - error ) / error
        return self.weighted_majority(h, z)


if __name__ == '__main__':
    from train import DecisionTree
    if len(sys.argv) != 4:
        print("Format: 'python train.py <training_file> <hypothesisOut> <learning-type>'")
        print("eg. for decision_tree 'python train.py train.dat dt.pkl dt'")
        print("eg. for adaboost 'python train.py train.dat ada.pkl ada'")
        sys.exit(1)
    train_file = sys.argv[1]
    model = sys.argv[2]
    learn_type = sys.argv[3]
    obj = DTLearning(train_file, model, learn_type)
    if learn_type == "dt":
        dt = obj.train_dt()
        with open(model, 'wb') as output:
            pickle.dump(dt, output, pickle.HIGHEST_PROTOCOL)
    if learn_type == "ada":
        ada = obj.train_ada()
        with open(model, 'wb') as output:
            pickle.dump(ada, output, pickle.HIGHEST_PROTOCOL)