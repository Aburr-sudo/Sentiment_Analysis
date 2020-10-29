#!/usr/bin/env python
# coding: utf-8

# Libraries that may need installation
#pip install tweepy
#pip install pandas
#pip install nltk
## pip install -U scipy scikit-learn
#pip install matplotlib
#pip install numpy


## import libraries
import json
import csv
import tweepy
import re
import pandas as pd

import string  
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.classify import apply_features
import random
import matplotlib.pyplot as plt


#### SVM
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import roc_curve, auc



def authorize(consumer_key, consumer_secret, access_token, access_token_secret):
    
    #Pass authentication details to login and access the twitter API
    auth_details = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth_details.set_access_token(access_token, access_token_secret)
    
    return auth_details


def Scrapetwitter(auth_details, keyword):
    #initialize Tweepy API
    api = tweepy.API(auth_details)
    
    ## Collect Tweets
    # important to put encoding to utf-8 here, encountered lots of issues without it
    # w for write mode
    with open('Dataset.csv', 'w', encoding='utf8') as file:
        #create csv file
        data_file = csv.writer(file)
        #write header row to spreadsheet
        data_file.writerow(['Timestamp', 'Text', 'Sentiment'])
        query = keyword

        # for every tweet that contains the specified keyword ('Tenet'), write it into row with timestamp and default sentiment value
        # default Neutral value for sentiments
        # filter retweets, this level of missing context would confuse the classification process. 
        # This function currently scrapes 400 tweets at a time
        # only searches for english tweets
        #replaces new lines with a space, allows for easier parsing of strings
        for tweet in tweepy.Cursor(api.search, q=query+' -filter:retweets',                                    lang="en", tweet_mode='extended').items(400):
            data_file.writerow([tweet.created_at,tweet.full_text.replace('\n',' '), ' Neutral'])



def get_keys():
    consumer_key = input('API (consumer) Key ')
    consumer_secret = input('API (consumer) Secret key')
    access_token = input('Access Token ')
    access_token_secret = input('Access Token Secret ')
    return consumer_key, consumer_secret, access_token, access_token_secret


## driver function for collecting data
def get_data():
    consumer_key, consumer_secret, access_token, access_token_secret = get_keys()
    keyword = input('keyword: ')
    print('Authorising data collection...')
    auth_details = authorize(consumer_key, consumer_secret, access_token, access_token_secret)
    print('Collecting data...')
    Scrapetwitter(auth_details, keyword)
    print('Data scraping complete')



#Preprocess data
def process_text(text):
    # get set of stop words
    stop_words = set(stopwords.words('english')) 
    # remove punctuation
    cleaned = [char for char in text if char not in string.punctuation]
    # rejoin characters
    cleaned = ''.join(cleaned)
    # remove numbers, \d = [0-9]
    ## use regular expressions to remove numbers, urls, usertags(@), and hashtags, respectively
    cleaned = re.sub(r'\d+', '', cleaned)
    cleaned = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', cleaned)
    cleaned = re.sub('@[^\s]+', '', cleaned)
    cleaned = re.sub(r'#([^\s]+)', r'\1', cleaned)
    # remove stop words
    filtered_words = [word for word in cleaned.split() if word not in stop_words]
    
    return filtered_words


# this function gets all the words in the corpus
def get_all_text(data):
    all_words = []
    for word in data:
            all_words += word
    return all_words


## methodology from NLTK documentation
# https://www.nltk.org/book/ch06.html

def features_in_document(document, all_words): 
    #make a set for faster computation
    # removes all duplicates
    document_words = set(document) 
    features = {}
    for word in all_words:
        features['Contains: ({})'.format(word)] = (word in document_words)
    return features

def get_feature_set(all_words, positive, negative): 
    positive_feats = []
    negative_feats = []
    for doc in positive:
        positive_feats.append([features_in_document(doc, all_words), 'Pos'])
    for doc in negative:
        negative_feats.append([features_in_document(doc, all_words), 'Neg'])
    feature_set = positive_feats + negative_feats
    return feature_set


def display_statistics(data):
    print('Total number of entries in data set:')
    print(len(data))
    print('Number of Neutral Tweets:')
    neutral = len(data[data.Sentiment == ' Neutral'])
    print(str(neutral))
    print('Number of Positive Tweets:')
    Positive = len(data[data.Sentiment == ' Positive'])
    print(str(Positive))
    print('Number of Negative Tweets:')
    Negative = len(data[data.Sentiment == ' Negative'])
    print(str(Negative))


def read_data():
    ## Read in Data
    data = pd.read_csv('t3_dataset.csv')
    data = data.drop(columns= 'Timestamp')
    
    display_statistics(data)
    # drop neutral values
    data = data[data.Sentiment != ' Sentiment']
    data = data[data.Sentiment != ' Neutral']
    return data



#grabs polarised tweets from preprocessed column
def extract_pos_neg_tweets(data):
    pos_tweets = data.loc[data['Sentiment'] == ' Positive']['Preprocessed_Text']
    neg_tweets = data.loc[data['Sentiment'] == ' Negative']['Preprocessed_Text']
    return pos_tweets, neg_tweets


def get_train_test_sets(features):
    #Randomise the data
    random.shuffle(features)
    # randomly split the data
    # 75/25 train test split
    cutoff = int(len(features) * 0.75)
    training_set = features[:cutoff]
    evaluation_set = features[cutoff:]
    return training_set, evaluation_set



def show_train_test_split(train, test): 
    print('Train on %d instances\nTest on %d instances' % (len(train), len(test)))



def naiveBayesClassification(train_data, test_data,):
    print('--------------------------------------------------------')
    print('                 Naive Bayes Classifier')
    print('--------------------------------------------------------')
    classifier = nltk.NaiveBayesClassifier.train(train_data)

    print('Accuracy:', nltk.classify.util.accuracy(classifier, test_data))
    print(classifier.show_most_informative_features(20))

    accuracy, tp, tn, fp, fn, pred_list = get_confusion_matrix_entry(classifier, test_data)

    prob_list = get_prob_list(classifier, test_data)

    conv_test = np.array(test_data)

    fpr, tpr, thresholds = roc_curve(conv_test[:, 1], prob_list, pos_label='Pos')

    draw_roc(fpr, tpr, 'Naive Bayes Classifier Result')
    print('--------------------------------------------------------')

'''
All the predict result of test data is compared with the
actual result in the dataset, and count the accuracy, tp, 
tn, fp, fn to generate confusion matrix
'''
def get_confusion_matrix_entry(model, test_data):
    pred_list = []
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    correct_entry = 0
    for i in range(len(test_data)):
        rst = model.classify(test_data[i][0])
        pred_list.append(rst)
        target = test_data[i][1]
        if rst == target:
            correct_entry += 1
            if rst == 'Pos':
                tp += 1
            elif rst == 'Neg':
                tn += 1
        else:
            if rst == 'Pos':
                fp += 1
            elif rst == 'Neg':
                fn += 1
    accuracy = correct_entry / len(test_data)
    print("++++++++++++++++++ Result +++++++++++++++++++++++++")
    print("Accuracy: ", accuracy)
    print("Confusion Matrix:")
    print("Predict/Reference\tPositive\tNegative")
    print("Positive\t\t", tp, "\t\t", fp)
    print("Negative\t\t", fn, "\t\t", tn)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")
    return accuracy, tp, tn, fp, fn, pred_list

'''
generate a probability list of the data set
'''
def get_prob_list(model, test_data):
    prob_list = []
    for i in range(len(test_data)):
        prob_rst = model.prob_classify(test_data[i][0])
        prob_list.append(prob_rst.prob('Pos'))
    return prob_list

'''
The function to draw the roc plot
by using matplotlib function
'''
def draw_roc(fpr, tpr, title):
    roc_auc = auc(fpr, tpr)
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area under curve (AUC) = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def decision_tree_classification(training_set, evaluation_set):
    print('--------------------------------------------------------')
    print('             Decision Tree classification')
    print('--------------------------------------------------------')
    DTclassifier = nltk.classify.DecisionTreeClassifier.train(training_set, entropy_cutoff=0.05,support_cutoff=0)
    
    print('Decision Tree Accuracy:', nltk.classify.util.accuracy(DTclassifier, evaluation_set))
    print(DTclassifier.pseudocode(depth=4))

    accuracy, tp, tn, fp, fn, pred_list = get_confusion_matrix_entry(DTclassifier, evaluation_set)

    print('--------------------------------------------------------')


def display_SVM_results(y_test, predictions):
    confusion_mat = metrics.confusion_matrix(y_test, predictions)
    accuracy =metrics.accuracy_score(y_test, predictions)
    print("++++++++++++++++++ Result +++++++++++++++++++++++++")
    print("Accuracy: ", accuracy)
    print("Confusion Matrix:")
    print("Predict/Reference\tPositive\tNegative")
    print("Positive\t\t", confusion_mat[0][0], "\t\t", confusion_mat[0][1])
    print("Negative\t\t", confusion_mat[1][0], "\t\t", confusion_mat[1][1])
    print("++++++++++++++++++++++++++++++++++++++++++++++++++")


def SVM_classification(data):
    print('--------------------------------------------------------')
    print('          Support Vector Machine Classification')
    print('--------------------------------------------------------')
    X = data['Text']
    #Count vectorizor folds the case of the text, so can't pass it preprocessed text
    y = data['Sentiment']
    
    # Split into 75/25 training and test sets
    # random state randomises the selection of features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # This function converts text into a vector of token/text counts
    # Has pre-processing built into it
    count_vect = CountVectorizer()
    # fit the training data to 
    X_train_count_vector = count_vect.fit_transform(X_train)

    # Term frequency, Inverse document Frequency transormfer
    tfidf_transformer = TfidfTransformer()
    # Scale down influence of terms that occur frequently througout the entireity of the corpus
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_count_vector)

    tfidf_vectorizor = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizor.fit_transform(X_train) # remember to use the original X_train set

    # Linear Support Vector Machine
    svm_classifier = LinearSVC()
    svm_classifier.fit(X_train_tfidf, y_train)

    # create pipeline
    sentiment_classifier = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LinearSVC(C=10)),
    ])

    # Feed the training data through the pipeline
    svm_rst = sentiment_classifier.fit(X_train, y_train)

    # call the draw_roc function to plot the roc diagram with auc
    y_score = svm_rst.decision_function(X_test)
    
    # an issue  was encountered running the code on different machines/versions of python.
    # If Y_score has more than 1 dimension, get the 2nd column
    print(type(y_score[0]))
    if type(y_score[0]) is np.float64:
        y_score = y_score
        # Draw the ROC curve, position AUC value in bottom right hand corner
        fpr, tpr, threshold = roc_curve(y_test.array, y_score, pos_label=' Positive')
        draw_roc(fpr, tpr, "SVM Classifier Result")
    elif type(y_score[0]) is np.array:
        y_score = y_score[:,1]
        fpr, tpr, threshold = roc_curve(y_test.array, y_score, pos_label=' Positive')
        draw_roc(fpr, tpr, "SVM Classifier Result")
    elif type(y_score[0]) is np.ndarray:
        y_score = y_score[:,1]
        fpr, tpr, threshold = roc_curve(y_test.array, y_score, pos_label=' Positive')
        draw_roc(fpr, tpr, "SVM Classifier Result")
    else:
        print("Error calculating the ROC curve..")
    
    # Form a prediction set
    predictions = sentiment_classifier.predict(X_test)

    # Report the result and confusion matrix
    display_SVM_results(y_test, predictions)
    print('--------------------------------------------------------')


def compareResults(X_test, y_test):
    ## compare actual values to predicted values
    testing_predictions = []
    for i in range(len(X_test)):
        if predictions[i] == 1:
            testing_predictions.append('Negative')
        else:
            testing_predictions.append('Positive')

    check_df = pd.DataFrame({'actual_label': list(y_test), 'prediction': testing_predictions, 'tweet_text':list(X_test)})
    check_df.replace(to_replace=0, value='Positive', inplace=True)
    check_df.replace(to_replace=1, value='Negative', inplace=True)
    check_df.head


def most_common_words(all_words):
    all_words = nltk.FreqDist(all_words)
    print(all_words.most_common(10))


#################  Main  ####################
if __name__ == '__main__':
    #get_data() #to scrape Twitter for data
    data = read_data()
    #get_data() #to scrape Twitter for data
    # Read data in file from web scraped data set
    # data is read into a pandas data frame for easy manipulation
    
    #performed in main to avoid passing by value
    # Pre-Process text and place in a new column
    data['Preprocessed_Text']= data['Text'].map(lambda tweet:process_text(tweet)) 
    
    #extracts positive and negative tweets from pre-processed column
    pos_tweets, neg_tweets = extract_pos_neg_tweets(data)
    
    # collect all words in data set
    corpus_dictionary = get_all_text(data['Preprocessed_Text'])
    # show most frequent words in data set
    print('most frequent words: ')
    most_common_words(corpus_dictionary)
    
    # extract features from the dictionary
    features = get_feature_set(corpus_dictionary, pos_tweets, neg_tweets)
    
    # Randomise and split the features into train and test set
    training_set, evaluation_set = get_train_test_sets(features)
    
    # Show amount of training and test features
    show_train_test_split(training_set, evaluation_set)
    
    ### Classification ###
    
    ## Naive Bayes 
    naiveBayesClassification(training_set, evaluation_set)

    ### Decision tree
    decision_tree_classification(training_set, evaluation_set)
    
    # Support vector machine
    # has a different format due to sklearn library
    # sklearn library methods such as CountVectorizor have built in pre-processing measures so
    # -- passing in the pre-processed text throws an error, original data is passed instead.
    SVM_classification(data)
    




