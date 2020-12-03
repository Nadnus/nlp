from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import copy
from nltk.corpus import stopwords
import nltk
import string
import collections
from typing import Counter
import numpy as np

#Accuracy
def accuracy(M):
    total = 0
    correct = 0
    for i in range(len(M)):
        for j in range(len(M)):
            total += M[i][j]
            if i == j:
                correct+=M[i][j]
    return correct/total

# Precision
def precision(M, index): #index es la posicion de la variable de interes
    correct = M[index][index]
    total = 0
    for i in range(len(M)):
        for j in range(len(M)):
            if i == index:
                total+= M[i][j]
    if total == 0:
        return 0

    return correct/total

# Recall
def recall(M, index): #index es la posicion de la variable de interes
    correct = M[index][index]
    total = 0
    for i in range(len(M)):
        for j in range(len(M)):
            if j == index:
                total+= M[i][j]
    if total == 0:
        return 0

    return correct/total

# F1
def F1_score(M,index):
    P = precision(M,index)
    R = recall(M,index)
    if P+R == 0:
        return 0
    return (2*P*R)/(P+R)

def macro_avg(func, M): #func es la metrica que queremos usar
    accum = 0
    for i in range(len(M)):
        accum+=func(M,i)
    return accum/len(M)


f = open("dataset.txt", "r")
o = open("processed.txt", "w")
printable = set(string.printable)
to_delete = ['<', '/', '|', '(', '+', '%', ':', '{', '}', '.', ',']

bracketsOpen = False
for line in f:
    line = line.replace(" - ", "-")
    for word in line.split():
        word = ''.join(filter(lambda x: x in printable, word))
        word = word.replace('=', ' ')
        word = word.replace('>', ' ')
        word = word.replace(')', ' ')

        for letter in to_delete:
            word = word.replace(letter, '')
        o.write(word + " ")
f.close()
o.close()

# Step 2

stop = stopwords.words('english')
bag = collections.defaultdict(int)
f = open("processed.txt", "r")
for line in f:
    for word in line.split():
        if word in stop:
            continue
        bag[word] += 1
bag = dict(sorted(bag.items(), key=lambda item: item[1], reverse=True))


# Step 3: Data split
feature = "modestly"
pageSize = 500

frequent_words = list(bag.keys())
feature_index = frequent_words.index(feature)
incidence_vector = [0] * len(frequent_words)
pages = []
pages.append('')
f = open("processed.txt", "r")
for line in f:
    for word in line.split():
        pages[-1] += " " + word
        if len(pages[-1].split()) >= pageSize:
            pages.append(word)
# Y ahora tenemos nuestra data paginada, sera cosa de cojer nuestro feature Y proceder a tratar de predecirlo
# Antes que nada, tenemos que crear nuestros vectores para cada pagina

pageVectors = []

for page in pages:
    vector = copy.deepcopy(incidence_vector)
    for word in page.split():
        for i in range(len(frequent_words)):
            if word == frequent_words[i]:
                vector[i] += 1
    pageVectors.append(vector)
print(len(pageVectors))
# vamos a separar la variable clase
clases = []
for page in pageVectors:
    clase = page[feature_index]
    clases.append(clase)
    page.pop(feature_index)

# Aqui se hace el split de la data
split = 80

# Casteamos las variables a arreglos de numpy:
index_percent = int(len(pageVectors)*split/100)
X = np.array(pageVectors[:index_percent])
Y = np.array(clases[:index_percent])
X_test = np.array(pageVectors[index_percent:])
Y_test = np.array(clases[index_percent:])


# algoritmo 1
# Naive Bayes
from sklearn.metrics import confusion_matrix

clf = GaussianNB()
clf.fit(X, Y)
confusion = confusion_matrix(Y_test,clf.predict(X_test))
acc = round(accuracy(confusion)*100)
prec = round(macro_avg(precision,confusion)*100)
rec = macro_avg(recall,confusion)*100
F1 = macro_avg(F1_score,confusion)*100


# Algoritmo 2:
#Decision Tree
from sklearn.metrics import confusion_matrix

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
confusion = confusion_matrix(Y_test,clf.predict(X_test))
acc = round(accuracy(confusion)*100)
prec = round(macro_avg(precision,confusion)*100)
rec = macro_avg(recall,confusion)*100
F1 = macro_avg(F1_score,confusion)*100


#Algoritmo 3:
#Redes neurales:
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y,
                                                    random_state=1)

clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
clf.predict_proba(X_test)
confusion = confusion_matrix(Y_test,clf.predict(X_test))
acc = round(accuracy(confusion)*100)
prec = round(macro_avg(precision,confusion)*100)
rec = macro_avg(recall,confusion)*100
F1 = macro_avg(F1_score,confusion)*100


#Algoritmo 4:
#Regresion
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
reg = LinearRegression().fit(X, Y)
reg.score(X, Y)
confusion = confusion_matrix(Y_test,clf.predict(X_test))
acc = round(accuracy(confusion)*100)
prec = round(macro_avg(precision,confusion)*100)
rec = macro_avg(recall,confusion)*100
F1 = macro_avg(F1_score,confusion)*100





# print(f)
