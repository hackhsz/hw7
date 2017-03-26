import scipy.io as sio
from PIL import Image
import numpy as np
from scipy.misc import toimage
from scipy.misc import imshow
import matplotlib.pyplot as plt
import pdb
import math
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
clf = MultinomialNB()

model = linear_model.LinearRegression()
mat_content = sio.loadmat('bodyfat_data.mat')

#pdb.set_trace()

print(mat_content.keys())

fit_x = mat_content['X']

fit_y = mat_content['y']

model.fit(fit_x,fit_y)
print(model.coef_)

print(model.intercept_)
listone= model.coef_
cof1 = 0.85547131

cof2 = -0.33975859

pdb.set_trace()
print(fit_y[0]-cof1*fit_x[0][0]-cof2*fit_x[0][1])


import scipy.io as sio
from PIL import Image
import numpy as np
from scipy.misc import toimage
from scipy.misc import imshow
import matplotlib.pyplot as plt
import pdb
import math
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
mat_content = sio.loadmat('spambase.mat')



# Using the class method..
dict11={}
dict12={}
dict1={}
dict0={}
numberof0 = newy.count(0)/len(newy)
numberof1 = newy.count(1)/len(newy)

#Pr(x|y=1) = Pr(x1=2|y=1)

#Pr(y=1|x=2) = Pr(x1=2|y=1)*Pr(x2=2|y=1)......*Pr(y=1)

#Pr(y=1|x=1) = Pr(x1=1 |y=1)........Pr(y=1)

#Compare Pr(x | y=0) *Pr(y=0)  versus Pr(x | y=1) *Pr(y=1)

listone = [x for (x,y) in zip(x,newy) if y==1]
listzero = [x for (x,y) in zip(x,newy) if y==0]



for i in range(57):
    dict1[i] ={}
    dict0[i] = {}
    for j in range(1,3):
        dict1[i][j] = 0.0
        dict0[i][j] = 0.0

for item in listone:
    for i in range(len(item)):
        number = item[i]
        if number == 2:
            dict1[i][number]+=1
        else:
            dict1[i][number]+=1

for item in listzero:
    for i in range(len(item)):
        number = item[i]
        if number == 2:
            dict0[i][number]+=1
        else:
            dict0[i][number]+=1
for i in range(len(dict1.keys())):
    dict1[i][1] = dict1[i][1] / len(listone)
    dict0[i][1] = dict0[i][1] / len(listzero)
    dict1[i][2] = dict1[i][2] / len(listone)
    dict0[i][2] = dict0[i][2] / len(listzero)

#pdb.set_trace()

#print(dict1[1][1],dict1[1][2])

# Prediction Stage:


prd=[]

#pdb.set_trace()

for item in predict:
    numberlist = item
    probability1 = 1.0
    probability0 = 1.0
    for i in range(len(numberlist)):
        thisnumber = numberlist[i]
        probability1 = dict1[i][thisnumber]*probability1
        probability0 = dict0[i][thisnumber]*probability0
    #pdb.set_trace()
    if probability1 > probability0:
        prd.append(1)
    else:
        prd.append(0)
# print(probability1,probability0)
# prd.append(probability1>probability0)

#pdb.set_trace()

print(prd)

print(len(prd))

print(predictionresult)
print(len(predictionresult))

final = []


for i in range(len(prd)):
    if prd[i] == predictionresult[i]:
        final.append(1)
    else:
        final.append(0)
result = [(x==y==1) for x,y in zip(prd,predictionresult)]

print(final.count(1)/len(result))


# Using Skilearn method..

x = mat_content['X'][:2000]
#pdb.set_trace()
y = [ item for sublist in  mat_content['y'] for item in sublist ]
newy = y[:2000]
#print(len(newy))
#print(y)
predict = mat_content['X'][2001:]
predictionresult = mat_content['y'][2001:]
clf.fit(x,newy)
predicted = clf.predict(predict)
#print(predicted)
#pdb.set_trace()

#result = [(x==y==1) for x,y in zip(mat_content['y'][2001:],predicted)]
#print(listone.count(1)/len(listone))
#print(result)

