# < ---- Import Necessary Packages ---- >

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
import nltk
import numpy as np
import joblib
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

nltk.download('stopwords')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # using pretrained Bert Tokenizer

def tokenize_text(text):
    return tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='np'
    )


def classifier(xtrain, ytrain):
    # define
    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(xtrain, ytrain) # train
    joblib.dump(dt_classifier, 'model.pkl') # save the trained Model


def preprocessing():
    # load
    data = pd.read_csv('articles.csv', encoding='cp1252')

    # doing necessary preprocessing steps lowercasing, remove numbers , punctuations with most frequent words (stop words is, are etc.,)
    data['Heading'] = data['Heading'].str.lower()
    data['Heading'] = data['Heading'].str.replace('[^\w\s]', '')  # Punctuations
    data['Heading'] = data['Heading'].str.replace('\d', '')  # Numbers
    sw = stopwords.words('english')
    data['Heading'] = data['Heading'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    data['tokenized'] = data['Heading'].apply(lambda x: tokenize_text(x)['input_ids'][0])
    lab_enc = preprocessing.LabelEncoder()
    data['label'] = lab_enc.fit_transform(data['Article_Type'])

    feat = list()
    lab = list()
    for i in range(data.shape[0]):
        x = data['tokenized'][i]
        y = data['label'][i]
        x = x.reshape(1, x.shape[0])
        feat.append(x)
        lab.append(y)

    Features = np.vstack(feat)
    return Features, np.array(lab)


x, y = preprocessing()
 ####################  cross validation  #######################
k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = []
for train, test in k_fold.split(x, y):
    xtrain = x[train, :]
    xtest = x[test, :]
    ytrain = y[train]
    ytest = y[test]

    # model
    classifier(xtrain, ytrain)
    m = joblib.load('model.pkl')  # saved model
    pred = m.predict(xtest)
    acc = accuracy_score(ytest, pred)
    acc_scores.append(acc)

accuracy = (sum(acc_scores) / len(acc_scores))


print('Accuracy Scores ------ > ', accuracy)

