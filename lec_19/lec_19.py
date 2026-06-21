# фильтрация спама
# бинарная классификация
# векторизация
# столбцы = слова (в тексте)
# строки = образы текста
# ячейка = кол-во слов в данном тексте
# отчистка - удаляем стоп слова, знгаки препинания, строчные символы

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
#
#data=pd.read_csv("spam.csv", encoding='latin-1')
#data = data[['v1', 'v2']]
#data.columns = ['Category', 'Message']
#vectorizer=CountVectorizer()
#X=vectorizer.fit_transform(data["Message"])
#w=vectorizer.get_feature_names_out()
#data["Spam"] = data["Category"].apply(lambda x: 1 if x == "spam" else 0)
#
#X_tr, X_tst, y_tr, y_tst = train_test_split(data["Message"], data["Spam"], test_size=0.25)
#
#md = Pipeline([("vectorizer", CountVectorizer()), ("nb", MultinomialNB())])
#md.fit(X_tr, y_tr)
#
#texts = [
#    "Hi! How are you?",              # 0
#    "Win the lottery",               # 0
#    "Free subscription",             # 1
#    "Black Friday big discount shop offer", # 0
#    "Nice to meet you"               # 0
#]
#
#print(md.predict(texts))



#data=pd.read_csv("phishing.csv")
#
#X = data.drop(columns=["class"])
#print(X.columns)
#
#y = pd.DataFrame(data=["class"])
#print(y.columns)
#
#
#X_tr, X_tst, y_tr, y_tst = train_test_split(
#    X, y, test_size=0.25
#)
#
#from sklearn.tree import DecisionTreeClassifier
#dt = DecisionTreeClassifier()
#model = dt.fit(X_tr, y_tr)
#predict = model.predict(X_tst)
#
#from sklearn.metrics import accuracy_score
#print(accuracy_score(predict, y_tst))

# Классификации: бинарные (двоичные), мультиклассовые, многометочные
# - точность (precision) - стоимость ложных срабатываний высока
# - полнота (recall) - стоимость ложноотрицательных срабатываний высока
# - специфичность (specificity) = полнота (наоборот). насколько точно определяются отрицательные образцы
# - чувствительность (sensitivity) = полнота.
# - F1-мера

# Метрики: - процент ошибок, процент правильных ответов (accuracy)
# Типы ошибочных ответов: ложноположительные (ложная тревога), ложноотрицательные (ложный пропуск)
# Типы правильных ответов: истиноположительные, истиноотрицательные


data=pd.read_csv("creditcard.csv")
legit=data[data["Class"]==0]
fraud=data[data["Class"]==1]

X = data.drop(["Time", "Class"])
y = data["Class"]

from sklearn.model_selection import train_test_split

X_tr, X_tst, y_tr, y_tst = train_test_split(
    X, y, test_size=0.25
)

from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()
model1.fit(X_tr, y_tr)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    model1,
    X_tst,
    y_tst,
    display_labels=["Легитимная", "Мошенническая"],
)

from sklearn.metrics import precision_score, recall_score

# Точность
y_pred = model1.predict(X_tst)
print(precision_score(y_tst, y_pred))

# Полнота
print(recall_score(y_tst, y_pred))

# Специфичность
print(recall_score(y_tst, y_pred, pos_label=0))

from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier()
model2.fit(X_tr, y_tr)
ConfusionMatrixDisplay.from_estimator(
    model2,
    X_tst,
    y_tst,
    display_labels=["Легитимная", "Мошенническая"],
)

from sklearn.ensemble import GradientBoostingClassifier
model3 = GradientBoostingClassifier()
model3.fit(X_tr, y_tr)
ConfusionMatrixDisplay.from_estimator(
    model3,
    X_tst,
    y_tst,
    display_labels=["Легитимная", "Мошенническая"],
)
