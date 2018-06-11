import sys
import sklearn, sklearn.datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import os, os.path
import shutil
import codecs
import copy
import matplotlib.pyplot as plt
'''
https://github.com/scikit-learn/scikit-learn/blob/a5ab948/sklearn/datasets/twenty_newsgroups.py
'''


def preprocess_data(path):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    preprocessed_path = dir_path + '/preprocessed' + path
    if not os.path.exists(preprocessed_path):
        shutil.copytree(dir_path + path, preprocessed_path)
    else:
        return
    for dirpath, dirnames, files in os.walk(preprocessed_path):
        for name in files:
            f = codecs.open(os.path.join(dirpath, name), 'r+', 'utf8', 'ignore')
            _, _, text = f.read().partition('\n\n')
            f.seek(0)
            f.write(text)
            f.truncate()
            f.close()


def naive_bayessian_classifier(config, train, test, curve_flag):
    if config == 'UB':
        ngram = (1, 1)
    elif config == 'BB':
        ngram = (2, 2)
    classifier = Pipeline([('vect', CountVectorizer(ngram_range=ngram)),
                           ('tfidf', TfidfTransformer()), ('clf',
                                                           MultinomialNB())])
    t = copy.deepcopy(train)
    classification = classifier.fit(t.data, t.target).predict(test.data)
    global f1_score
    if curve_flag:
        f_ones[0].append(
            metrics.f1_score(test.target, classification, average='macro'))
    else:
        global output
        output += 'NB, ' + config + ', ' + str(
            round(
                metrics.precision_score(
                    test.target, classification, average='macro'),
                2)) + ', ' + str(
                    round(
                        metrics.recall_score(
                            test.target, classification, average='macro'),
                        2)) + ', ' + str(
                            round(
                                metrics.f1_score(
                                    test.target,
                                    classification,
                                    average='macro'), 2)) + '\n'


def logistic_regression_classifer(config, train, test, curve_flag):
    if config == 'UB':
        ngram = (1, 1)
    elif config == 'BB':
        ngram = (2, 2)
    classifier = Pipeline([('vect', CountVectorizer(ngram_range=ngram)),
                           ('tfidf',
                            TfidfTransformer()), ('clf', LogisticRegression())])
    t = copy.deepcopy(train)
    classification = classifier.fit(t.data, t.target).predict(test.data)
    global f1_score
    if curve_flag:
        f_ones[1].append(
            metrics.f1_score(test.target, classification, average='macro'))
    else:
        global output
        output += 'LR, ' + config + ', ' + str(
            round(
                metrics.precision_score(
                    test.target, classification, average='macro'),
                2)) + ', ' + str(
                    round(
                        metrics.recall_score(
                            test.target, classification, average='macro'),
                        2)) + ', ' + str(
                            round(
                                metrics.f1_score(
                                    test.target,
                                    classification,
                                    average='macro'), 2)) + '\n'


def support_vectore_machine_classifier(config, train, test, curve_flag):
    if config == 'UB':
        ngram = (1, 1)
    elif config == 'BB':
        ngram = (2, 2)
    classifier = Pipeline([('vect', CountVectorizer(ngram_range=ngram)),
                           ('tfidf', TfidfTransformer()), ('clf', LinearSVC())])
    t = copy.deepcopy(train)
    classification = classifier.fit(t.data, t.target).predict(test.data)
    global f1_score
    if curve_flag:
        f_ones[2].append(
            metrics.f1_score(test.target, classification, average='macro'))
    else:
        global output
        output += 'SVM, ' + config + ', ' + str(
            round(
                metrics.precision_score(
                    test.target, classification, average='macro'),
                2)) + ', ' + str(
                    round(
                        metrics.recall_score(
                            test.target, classification, average='macro'),
                        2)) + ', ' + str(
                            round(
                                metrics.f1_score(
                                    test.target,
                                    classification,
                                    average='macro'), 2)) + '\n'


def random_forest_classifier(config, train, test, curve_flag):
    if config == 'UB':
        ngram = (1, 1)
    elif config == 'BB':
        ngram = (2, 2)
    classifier = Pipeline([('vect', CountVectorizer(ngram_range=ngram)),
                           ('tfidf', TfidfTransformer()),
                           ('clf', RandomForestClassifier())])
    t = copy.deepcopy(train)
    classification = classifier.fit(t.data, t.target).predict(test.data)
    global f1_score
    if curve_flag:
        f_ones[3].append(
            metrics.f1_score(test.target, classification, average='macro'))
    else:
        global output
        output += 'RF, ' + config + ', ' + str(
            round(
                metrics.precision_score(
                    test.target, classification, average='macro'),
                2)) + ', ' + str(
                    round(
                        metrics.recall_score(
                            test.target, classification, average='macro'),
                        2)) + ', ' + str(
                            round(
                                metrics.f1_score(
                                    test.target,
                                    classification,
                                    average='macro'), 2)) + '\n'


'''
http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
'''


def compare_learning_curve(train, test):
    split_size = []
    for i in range(10):
        split = copy.deepcopy(train)
        if float(i + 1) / 10 == 1:
            test_size_ = .99
        else:
            test_size_ = float(i + 1) / 10
        splitTestData, split.data, splitTestTarget, split.target = train_test_split(
            train.data, train.target, test_size=test_size_)
        split_size.append(len(split.data))
        naive_bayessian_classifier('UB', split, test, True)
        logistic_regression_classifer('UB', split, test, True)
        support_vectore_machine_classifier('UB', split, test, True)
        random_forest_classifier('UB', split, test, True)
    global f1_score
    for classifier in range(4):
        plt.plot(split_size, f_ones[classifier])
        plt.xlabel('Training Size')
        plt.ylabel('F1-Score')
        plt.axis([0, split_size[9], 0, 1.0])
    plt.savefig('classifier.png')
    print('Learning Curve created')


def classify(train, test):
    naive_bayessian_classifier('UB', train, test, False)
    naive_bayessian_classifier('BB', train, test, False)

    logistic_regression_classifer('UB', train, test, False)
    logistic_regression_classifer('BB', train, test, False)

    support_vectore_machine_classifier('UB', train, test, False)
    support_vectore_machine_classifier('BB', train, test, False)

    random_forest_classifier('UB', train, test, False)
    random_forest_classifier('BB', train, test, False)


def write_to_output_file(file_path):
    with open(file_path + '.txt', 'w') as f:
        f.write(output)
    print('Output Written to the Output File: ', file_path + '.txt')


if __name__ == "__main__":
    f_ones = [[] for x in range(4)]
    output = ''
    training_file_path = sys.argv[1]
    preprocess_data(training_file_path)
    training = sklearn.datasets.load_files(
        'preprocessed' + training_file_path,
        categories=[
            'rec.sport.hockey', 'sci.med', 'soc.religion.christian',
            'talk.religion.misc'
        ],
        encoding="utf-8",
        decode_error="replace",
        shuffle=True,
        random_state=42)

    test_file_path = sys.argv[2]
    preprocess_data(test_file_path)
    test = sklearn.datasets.load_files(
        'preprocessed' + test_file_path,
        categories=[
            'rec.sport.hockey', 'sci.med', 'soc.religion.christian',
            'talk.religion.misc'
        ],
        encoding="utf-8",
        decode_error="replace",
        shuffle=True,
        random_state=42)

    classify(training, test)
    write_to_output_file(sys.argv[3])

    if sys.argv[4] == '1':
        compare_learning_curve(training, test)
