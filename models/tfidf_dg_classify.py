# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
# D:\localE\python
# -*-coding:utf-8-*-
# Author ycx
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest,chi2
from  sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV  # 调试参数并打分，获取最佳参数
import pickle
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import time
t1=time.time()

column = "word_seg"
train = pd.read_csv('train_set_filter.csv')
train['add']=train['word_seg']+' '+train['article']

# train = shuffle(train)
test = pd.read_csv('test_set.csv')
test['test_add']=test['word_seg']+' '+test['article']

vec = TfidfVectorizer(ngram_range=(1,3), min_df=3, max_df=0.5,
                      use_idf=1, smooth_idf=1, sublinear_tf=1)

# tr_te=np.concatenate((train['add'],test['test_add'][0:3000]),axis=0)
trn_term_doc = vec.fit_transform(train['add'])  #102277-1097+51999*key_word_length

test_term_doc = vec.transform(test['test_add'])

print('1:',trn_term_doc.shape)

y = (train["class"] - 1).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
   trn_term_doc, y, test_size=0.01, random_state=0)
del trn_term_doc
#
# grid.fit(X_train, y_train)
# print(grid.best_params_,grid.best_score_)

# model= svm.LinearSVC(C=grid.best_params_['C'], max_iter=grid.best_params_['max_iter'])
model= svm.LinearSVC()

# model= SGDClassifier()
model.fit(X_train, y_train)
y_true, y_pred = y_test, model.predict(X_test)
y_train_pred=model.predict(X_train)
# train_pred=model.fit(X_test)
# test_pred=model.fit(y_test)

# model.fit(X_train, y_train)
print(classification_report(y_test, y_pred,digits=4))
print(classification_report(y_train,y_train_pred,digits=4))

t2=time.time()
print('耗时：',t2-t1)

preds = model.predict(test_term_doc)
preds+=1
test_id=test['id']
df_preds=pd.DataFrame({'id':test_id,'class':preds})
df_preds.to_csv('1-4add_article_baseline3-0.2.csv',index=None)

logits=model._predict_proba_lr(test_term_doc)
with open('1-4add_article_baseline3-0.2_logits.pkl','wb') as f:
    pickle.dump(logits,f)
print('wancheng!')