■ word embedding
일상의 언어를 컴퓨터가 이해할 수 있는 벡터 (숫자)로 표현하는 방법

■ BOW
문서를 숫자 벡터로 변환하는 가장 기본적인 방법(Bag of Words), 인코딩 방법

■ 문서 단어 행렬(Document Term Matrix)
문서에서 등장하는 각 단어들의 빈도를 행렬로 표현

[문제193] mom.txt 데이터를 가지고 윗 화면과 같은 문서단어행렬을 수동으로 생성해주세요.
import pandas as pd
from pandas import Series,DataFrame
from collections import Counter
from konlpy.tag import Kkma

mom = pd.read_csv('c:/data/mom.txt',header=None,names=['sentences'])

mom['sentences'] = mom['sentences'].str.lower()

mom['sentences'].str.count('mommy').sum()
mom['sentences'].str.count('i').sum() # 이건 단어 i가 아니라 알파벳'i'를 다 세어버려
mom['sentences'].str.count('mommy').sum()

data = ' '.join(mom['sentences'])
words = set(data.split())
words = [i for i in words if len(i) >= 2]
words.sort()
words

freq_table = {}
for i in words:
    freq_table.setdefault(i,0)

mom['sentences'][0]
j = mom['sentences'][0].split()
j

for j in mom['sentences'][0].split():
    if j in freq_table.keys():
        freq_table[j] += 1

freq_table
freq_df = pd.DataFrame()
freq_df

temp = DataFrame(Series(freq_table)).T
freq_df = pd.concat([freq_df, temp],ignore_index=True)
freq_df

# 같음
temp = pd.DataFrame.from_dict([freq_table])
freq_df = pd.concat([freq_df, temp],ignore_index=True)
freq_df

###
data = ' '.join(mom['sentences'])
words = set(data.split())
words = [i for i in words if len(i) >= 2]
words.sort()
words

freq_df = pd.DataFrame()
freq_df

for i in mom['sentences']:
    freq_table ={}
    for w in words:
        freq_table.setdefault(w,0)
        
    for j in i.split():
        if j in freq_table.keys():
            freq_table[j] += 1
    temp = pd.DataFrame.from_dict([freq_table])
    freq_df = pd.concat([freq_df,temp],ignore_index=True)

freq_df.sum(axis=0)
freq_df.sum(axis=1)

freq_df['document'] = mom['sentences']
del freq_df['document']

freq_df.insert(0,'document', mom['sentences']) # 특정 인덱스 열 위치에 새로운 열을 추가하는 방법
freq_df.iloc[0]

freq_df[words].sum()
freq_df[words].sum().sort_values(ascending=False)

freq_df.columns.difference(['document']) # 특정한 열을 제외
freq_df[freq_df.columns.difference(['document'])].sum().sum()

■ CountVectorizer
문서집합에서 단어 토큰을 생성하고 각 단어의 수를 세어 BOW 인코딩 벡터를 만든다.


import operator
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
vect.fit(mom['sentences'])
vect.vocabulary_ # 숫자는 인덱스
sorted(vect.vocabulary_.items(),key=operator.itemgetter(1))
vect.get_feature_names() # 단어만 확인

vect.fit_transform(mom['sentences'])
feature_vector= vect.transform(mom['sentences'])
feature_vector
feature_vector.shape

feature_vector.toarray()

df = pd.DataFrame(feature_vector.toarray(),columns=vect.get_feature_names())
df.sum(axis=0).sort_values(ascending=False)

df # d-t-m 결과


■ 불용어 처리
stopwords = ['am','are','be','and','is','the','then']

vect = CountVectorizer(stop_words=stopwords)
feature_vector= vect.fit_transform(mom['sentences'])
feature_vector
vect.vocabulary_
vect.get_feature_names() 
df = pd.DataFrame(feature_vector.toarray(),columns=vect.get_feature_names())
df.sum(axis=0).sort_values(ascending=False)

# 원래 있는 불용어 사전 이용하기
mom = pd.read_csv('c:/data/mom.txt',header=None,names=['sentences'])

vect = CountVectorizer(stop_words="english") # 있던 불용어사전 이걸로 이용하면 한철자짜리 없애고 소문자로도 자동으로 바
feature_vector= vect.fit_transform(mom['sentences'])
feature_vector
vect.vocabulary_
vect.get_feature_names() 
df = pd.DataFrame(feature_vector.toarray(),columns=vect.get_feature_names())
df.sum(axis=0).sort_values(ascending=False)

■ 토큰
vect = CountVectorizer(analyzer='word') # 기본값, 공백 기준으로
feature_vector= vect.fit_transform(mom['sentences'])
feature_vector
vect.vocabulary_
vect.get_feature_names() 
df = pd.DataFrame(feature_vector.toarray(),columns=vect.get_feature_names())
df.sum(axis=0).sort_values(ascending=False)

vect = CountVectorizer(analyzer='char') # 알파벳 철자를 기준으로 분리 
feature_vector= vect.fit_transform(mom['sentences'])
feature_vector
vect.vocabulary_
vect.get_feature_names() 
df = pd.DataFrame(feature_vector.toarray(),columns=vect.get_feature_names())
df.sum(axis=0).sort_values(ascending=False)

vect = CountVectorizer(token_pattern='t\w+') # 정규표현식을 활용해서 내가 원하는 글자 패턴의 사전을 만들 수 있다
feature_vector= vect.fit_transform(mom['sentences'])
feature_vector
vect.vocabulary_
vect.get_feature_names() 
df = pd.DataFrame(feature_vector.toarray(),columns=vect.get_feature_names())
df.sum(axis=0).sort_values(ascending=False)

■ n그램
1개의 단어로 하나의 토큰을 생성 1-gram,monogram
2개의 단어로 하나의 토큰을 생성 2-gram,bigram
3개의 단어로 하나의 토큰을 생성 3-gram,trigram

vect = CountVectorizer(ngram_range=(1,1)) # 1gram 기본값
vect = CountVectorizer(ngram_range=(2,2)) # 2-gram,bigram
vect = CountVectorizer(ngram_range=(1,2)) # 1gram, 2gram 같이 생성
vect = CountVectorizer(ngram_range=(1,2),token_pattern ='t\w+')
feature_vector= vect.fit_transform(mom['sentences'])
feature_vector
vect.vocabulary_
vect.get_feature_names() 
df = pd.DataFrame(feature_vector.toarray(),columns=vect.get_feature_names())
df.sum(axis=0).sort_values(ascending=False)


■ 빈도수
토큰의 빈도(등장한 문서 수)가 max_df로 지정한 값을 초과하거나
min_df로 지정한 값보다 작은 경우에는 무시한다.

vect = CountVectorizer(analyzer='word',min_df=2,max_df=4) 
feature_vector= vect.fit_transform(mom['sentences'])
feature_vector
vect.vocabulary_
vect.get_feature_names() 
df = pd.DataFrame(feature_vector.toarray(),columns=vect.get_feature_names())
df.sum(axis=0).sort_values(ascending=False)

■ TF-IDF(Term Frequency - Inverse Document Frequency)

# Term Frequency(기본)

- 한개 문서 안에서 특정 단어의 등장 빈도수
- 단어 빈도


vect = CountVectorizer(analyzer='word')
feature_vector= vect.fit_transform(mom['sentences'])
feature_vector
vect.vocabulary_
vect.get_feature_names() 
df = pd.DataFrame(feature_vector.toarray(),columns=vect.get_feature_names())
df.sum(axis=0).sort_values(ascending=False)

# DF(Document Frequency)
- 특정 단어가 나타나는 문서의 수
- 문서빈도
- DF가 클수록 여러 문서에 흔하게 사용된 일반적인 단어라고 할 수 있다.(중요하지 않은 단어)

mommy
TF = ? 문서1 1, 문서2 0, 문서3 0, 문서4 1, 문서5 1
DF = ? 3

you 
TF = ? 문서1 2, 문서2 1, 문서3 0, 문서4 1, 문서5 1
DF = ? 4

# Inverse Document Frequency
- DF에 역수로 변환해준 값(작을 수록 중요하니까)
- DF의 역수이므로 DF가 클수록 IDF값은 작아지고 DF작을수록 IDF 커진다.
- IDF-SMOOTHING에 따라 달라진다.
- 여러 문서에서 자주 등장하는 단어들에 대해서 페널티를 주는 방법



TF-IDF = TF * IDF

IDF = log(N+1/DF+1) # 0이 되면 안되니까 +1

N : 전체문서 수
DF : 특정 단어가 나타나는 문서의 수

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer()
tf_idf_vect = tfidf_vect.fit(mom['sentences'])
tf_idf_vect.vocabulary_
tf_idf_vect.get_feature_names()
tf_idf_vect.idf_

tfidf_vect.transform(mom['sentences']).toarray()

pd.DataFrame(tf_idf_vect.idf_,index=tf_idf_vect.get_feature_names(),columns=['idf'])

poem = pd.read_csv('c:/data/poem.txt', header=None,names=['sentences'])
poem

vect = CountVectorizer()
vect.fit(poem['sentences'])
vect.vocabulary_
vect.get_feature_names()

feature_vector = vect.transform(poem['sentences'])
feature_vector

df = pd.DataFrame(feature_vector.toarray(),columns=vect.get_feature_names())
df.sum(axis=0)

from konlpy.tag import Okt
okt = Okt()

def okt_pos(arg):
    token_corpus =[]
    for i in okt.pos(arg):
        if i[1] in ['Noun','Adjective']:
            token_corpus.append(i[0])
    return token_corpus

[okt_pos(i) for i in poem['sentences']]


cv = CountVectorizer(tokenizer=okt_pos)
cv_trans = cv.fit_transform(poem['sentences'])
cv.vocabulary_
cv.get_feature_names()
cv
df=pd.DataFrame(cv_trans.toarray(),columns=cv.get_feature_names())
df

■ vectorizer로 이미 만든 걸 TF-IDF로 바꾸는법

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_trans = TfidfTransformer()
tf_idf_vect = tfidf_trans.fit_transform(cv_trans)

df_tfidf = pd.DataFrame(tf_idf_vect.toarray(),columns=cv.get_feature_names())
df_tfidf



[문제194] 중앙일보 인공지능 뉴스 기사 검색 데이터를 이용해서 tf-idf를 생성해주세요.
(다른 기사에는 없는 단어 찾기)

contents = pd.read_csv('c:/data/contents.csv')
con10 = contents.news[:10]

from konlpy.tag import Okt
okt = Okt()

def okt_pos(arg):
    token_corpus =[]
    for i in okt.pos(arg):
        if i[1] in ['Noun','Adjective']:
            token_corpus.append(i[0])
    token_corpus = [x for x in token_corpus if len(x) >1]
    return token_corpus


cv = CountVectorizer(tokenizer=okt_pos)
cv_trans = cv.fit_transform(con10)
cv.vocabulary_
cv.get_feature_names()
cv
df=pd.DataFrame(cv_trans.toarray(),columns=cv.get_feature_names())
df.sum(axis=0)

df.iloc[0].max()
df.iloc[0].idxmax()
df.loc[0,'디지털']
df.iloc[0].sort_values(ascending=False)[:10]

df.iloc[1].max()
df.iloc[1].idxmax()
df.loc[1,'산업']
df.iloc[1].sort_values(ascending=False)[:10]


w = WordCloud(font_path='c:/windows/fonts/HMKMMAG.TTF',
              background_color='white',
              width=900,height=500).generate_from_frequencies(dict(df.sum(axis=0))) 
plt.imshow(w)
plt.axis('off')

# vectorize한거 tfidf로 변환
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_trans = TfidfTransformer()
tf_idf_vect = tfidf_trans.fit_transform(cv_trans)

df_tfidf = pd.DataFrame(tf_idf_vect.toarray(),columns=cv.get_feature_names())
df_tfidf.sum(axis=0).sort_values(ascending=False)

df_tfidf.iloc[0].max()
df_tfidf.iloc[0].idxmax()
df_tfidf.loc[0,'디지털']
df_tfidf.iloc[0].sort_values(ascending=False)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

survey = pd.read_csv('c:/data/naive_survey.csv',header=None,names=['sentences','class'])
survey

cv = CountVectorizer(analyzer='word').fit(survey['sentences'])
cv.vocabulary_
cv.get_feature_names()
cv_trans = cv.transform(survey['sentences'])
cv_trans.toarray()

cv_trans[0:2].toarray()

# 인코딩된 벡터를 실제 단어로 변환
cv.inverse_transform(cv_trans[0:2]) # 0을 제외한 빈도수가 구해진 단어들 확인하는 방법 inverse_transform

df = pd.DataFrame(cv_trans.toarray(),columns = cv.get_feature_names())
df

from konlpy.tag import Kkma
kkma = Kkma()

def kkma_pos(arg):
    return kkma.morphs(arg)

kkma_pos(survey['sentences'][0])
stopwords = ['은','는','이','가','에','고'] # 다른사람의 불용어 사전을 활용해도 된다

cv = CountVectorizer(tokenizer=kkma_pos,stop_words=stopwords).fit(survey['sentences'])
cv.vocabulary_
cv.get_feature_names()
cv_trans = cv.transform(survey['sentences'])
cv_trans.toarray()

cv_trans[0:2].toarray()

cv.inverse_transform(cv_trans[0:2])

df = pd.DataFrame(cv_trans.toarray(),columns = cv.get_feature_names())
df



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(survey['sentences'])
tfidf_vect.vocabulary_
tfidf_vect.get_feature_names()
tfidf_vect.idf_ # 이건 그냥 idf값
tfidf_vect.transform(survey['sentences']).toarray() # transform 하게 되면 tfidf 변환값이 나온다

pd.DataFrame(tfidf_vect.idf_,index=tfidf_vect.get_feature_names(),columns=['idf']) # idf 값 확인
df_tfidf = pd.DataFrame(tfidf_vect.transform(survey['sentences']).toarray(),columns = tfidf_vect.get_feature_names())
df_tfidf # tfidf값 확인
#####################################################
바로 tfidf 구하기
from konlpy.tag import Kkma
kkma = Kkma()

def kkma_pos(arg):
    return kkma.morphs(arg)

tfidf_vect = TfidfVectorizer(tokenizer=kkma_pos,stop_words=stopwords)
tfidf_vect.fit(survey['sentences'])
tfidf_vect.vocabulary_
tfidf_vect.get_feature_names()
tfidf_vect.idf_ # 이건 그냥 idf값
tfidf_vect.transform(survey['sentences']).toarray() # transform 하게 되면 tfidf 변환값이 나온다

pd.DataFrame(tfidf_vect.idf_,index=tfidf_vect.get_feature_names(),columns=['idf']) # idf 값 확인
df_tfidf = pd.DataFrame(tfidf_vect.transform(survey['sentences']).toarray(),columns = tfidf_vect.get_feature_names())
df_tfidf # tfidf값 확인

################################################################################

CountVectorizer -> tfidf 변환

from konlpy.tag import Kkma
kkma = Kkma()

def kkma_pos(arg):
    return kkma.morphs(arg)

kkma_pos(survey['sentences'][0])
stopwords = ['은','는','이','가','에','고'] # 다른사람의 불용어 사전을 활용해도 된다

cv = CountVectorizer(tokenizer=kkma_pos,stop_words=stopwords).fit(survey['sentences'])
cv.vocabulary_
cv.get_feature_names()
cv_trans = cv.transform(survey['sentences'])
cv_trans.toarray()

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_trans = TfidfTransformer()
pd.DataFrame(tfidf_trans.fit_transform(cv_trans).toarray(),columns = cv.get_feature_names())


################### 나이브 베이즈 ######################

[문제195] 메일 안에 복권이라는 단어가 있을 경우에 스팸일 확률은?
p(스팸) = 0.22 # 사전확률
P(복권|스팸) = 0.136 # 가능성(likelyhood)
P(복권|햄) = 0.025 

P(스팸|복권) = P(스팸 ∩ 복권)/P(복권) = P(복권|스팸)*p(스팸)/ P(복권)

P(복권) = P(복권|스팸)*p(스팸) + P(복권|햄)*P(햄) = (0.136*0.22) + (0.025*0.78)

P(스팸 ∩ 복권)/P(복권) = P(복권|스팸)*p(스팸)/ P(복권) = (0.136*0.22) / ((0.136*0.22) + (0.025*0.78))



from konlpy.tag import Kkma
kkma = Kkma()

def kkma_morphs(arg):
    return kkma.morphs(arg)

kkma_pos(survey['sentences'][0])
stopwords = ['은','는','이','가','에','고'] # 다른사람의 불용어 사전을 활용해도 된다

cv = CountVectorizer(tokenizer=okt_pos,stop_words=stopwords).fit(survey['sentences'])
cv.vocabulary_
cv.get_feature_names()
x_train = cv.transform(survey['sentences']) # 학습데이터
x_train.toarray()
y_train = survey['class'] # 정답 레이블
y_train

import numpy as np
from pandas import Series, DataFrame
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(x_train,y_train)


x_test = cv.transform(Series(' '.join(kkma_pos("함께 살고 있는 강아지가 너무 좋아")))) #위에 survey['senectences']는 이미 시리즈니까 오류 안뜨지. 시리즈로 바꿔줘야해
x_test.toarray()
cv.inverse_transform(x_test)
nb.predict(x_test)

x_test = cv.transform(Series(' '.join(kkma_pos("오늘 하루는 기분이 우울하다."))))
x_test.toarray()
cv.inverse_transform(x_test)
nb.predict(x_test)

x_test = cv.transform(Series(' '.join(kkma_pos("너는 무지 짜증나.")))) # 데이터가 적으니까 부정인데도 긍정으로 나와
x_test.toarray()
cv.inverse_transform(x_test)
nb.predict(x_test)

# 분류 클래스 정보
nb.classes_

# 분류 클래스 개수
nb.class_count_
survey['class'].value_counts()

# 사전확률
np.exp(nb.class_log_prior_)
8/15 긍정 사전확률
7/15 부정 사전확률

# 분류 클래스별 컬럼의 값의 빈도수
nb.feature_count_

df = pd.DataFrame(nb.feature_count_,columns=cv.get_feature_names(),index=nb.classes_)
df['짜증스러운']


# 저장하기. 학습이 원래 오래걸리잖아. 그러니까 한번 만든걸 저장을 잘 해놓고 다음에 이용해야해
import pickle
file= open('c:/data/classifier_1.pkl','wb')
pickle.dump(nb,file)
file.close()

file= open('c:/data/cv.pkl','wb')
pickle.dump(cv,file)
file.close()

file= open('c:/data/classifier_1.pkl','rb')
classifier_new = pickle.load(file)
file.close()

file= open('c:/data/cv.pkl','rb')
cv_new = pickle.load(file)
file.close()

classifier_new.classes_
classifier_new.class_count_
np.exp(classifier_new.class_log_prior_)

cv_new.vocabulary_
cv_new.get_feature_names()

x_test = cv_new.transform(Series(' '.join(kkma_pos("함께 살고 있는 강아지가 너무 좋아")))) 
x_test.toarray()
cv_new.inverse_transform(x_test)
nb.predict(x_test)

# 학습 테스트 분류
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from collections import Counter


survey = pd.read_csv('c:/data/naive_survey.csv',header=None,names=['sentences','class'])

x_train,x_test,y_train,y_test = train_test_split(survey['sentences'],survey['class'],test_size=0.2)
Counter(y_train)
Counter(x_train)

cv = CountVectorizer(ngram_range=(2,2))
x_train = cv.fit_transform(x_train)
cv.get_feature_names()
x_train.toarray()
x_test= cv.transform(x_test)
x_test.toarray()

nb = MultinomialNB()
nb.fit(x_train,y_train)

y_predict = nb.predict(x_test)
sum(y_predict == y_test)
accuracy_score(y_test,y_predict)

from konlpy.tag import Okt
okt = Okt()

def okt_pos(arg):
    token_corpus = []
    for i in okt.pos(arg):
        if i[1] in ['Noun','Adjective']:
            token_corpus.append(i[0])
    token_corpus = [word for word in token_corpus if len(word) >=2]
    return token_corpus

cv = CountVectorizer(tokenizer=okt_pos)
x_train = cv.fit_transform(x_train)
cv.get_feature_names()
x_train.toarray()
x_test= cv.transform(x_test)
x_test.toarray()

nb = MultinomialNB()
nb.fit(x_train,y_train)

y_predict = nb.predict(x_test)
sum(y_predict == y_test)
accuracy_score(y_test,y_predict)


■ 혼동행렬(confusion matrix)
- 모델 성능을 평가할 때 사용되는 지표
- 예측값이 실제값을 얼마나 정확하게 예측했는지를 보여주는 행렬

pd.crosstab(y_test,y_predict)

               predict(예측)
     col_0     긍정   부정
실    class
제    긍정       1      0
      부정       0      2

        
 
           예측(긍정)        예측(부정)
-----------------------------------------
실제(긍정)    TP                FN
실제(부정)    FP                 TN

TP : 참긍정
TN : 참부정
FP : 거짓긍정
FN : 거짓부정


from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test,y_predict)
print(classification_report(y_test,y_predict))

              precision    recall  f1-score   support

          긍정       0.67      1.00      0.80         2
          부정       0.00      0.00      0.00         1

    accuracy                           0.67         3
   macro avg       0.33      0.50      0.40         3
weighted avg       0.44      0.67      0.53         3


정확도(accuracy)
모델이 입력된 데이터에 대해 얼마나 정확하게 예측하는지를 나타내는 지표

정확도 = 예측결과와 실제값이 동일한 건수 / 전체 데이터 수
        =(TP+TN)/(TP+TN+FP+FN)

정밀도(precision)
정밀도는 긍정 클래스에 속한다고 예측한 값이 실제도 긍정 클래스에 속하는 비율
정밀도는 부정 클래스에 속한다고 예측한 값이 실제도 부정 클래스에 속하는 비율

긍정 정밀도 = TP / TP+FP
부정 정밀도 = TN / TN+FN

재현율(recall)
실제값 중에서 모델이 검출한 실제값의 비율을 나타내는 비율
긍정재현율 = TP / TP+FN
부정재현율 = TN / FP+TN

f1-score
정밀도도 중요하고 재현율도 중요한데 둘 중 무엇을 사용해야 할지 고민될 때
두 값을 조화평균해서 하나의 수치로 나타낸 지표

긍정f1-score = (긍정재현율*긍정정밀도*2) / (긍정재현율+긍정정밀도)
부정f1-score = (부정재현율*부정정밀도*2) / (부정재현율+부정정밀도)


############### 나이브베이즈 세미프로젝트 #################
from selenium import webdriver
import urllib
from urllib.request import urlopen
import time
from pandas import DataFrame, Series
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
from selenium.webdriver import ActionChains
import time
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from konlpy.tag import Kkma
kkma = Kkma()

opt = Options()
opt.add_experimental_option('prefs',{'profile.default_content_setting_values.notifications':1})

# 캐치카페 기업 들어가기
url = 'https://www.catch.co.kr/Comp/CompMajor?flag=Search'
driver = webdriver.Chrome('c:/data/chromedriver.exe',options=opt)
driver.get(url)

# IT/통신 
btn = driver.find_element(By.XPATH, '//*[@id="Contents"]/div[2]/div[1]/dl[1]/dd/ul/li[6]/label')
action = ActionChains(driver)
action.move_to_element(btn).perform()
driver.find_element(By.XPATH, '//*[@id="Contents"]/div[2]/div[1]/dl[1]/dd/ul/li[6]/label').click()

# 검색
btn = driver.find_element(By.XPATH, '//*[@id="imgSearch"]')
action = ActionChains(driver)
action.move_to_element(btn).perform()
driver.find_element(By.XPATH, '//*[@id="imgSearch"]').click()


it_df = DataFrame()
# 페이징
for i in range(2,7):
    btn = driver.find_element(By.XPATH, '//*[@id="Contents"]/p[3]/a['+str(i)+']')
    action = ActionChains(driver)
    action.move_to_element(btn).perform()
    driver.find_element(By.XPATH, '//*[@id="Contents"]/p[3]/a['+str(i)+']').click()
    time.sleep(2)
    # 기업 선택
    for i in range(1,11):
        try:
            btn = driver.find_element(By.XPATH, '//*[@id="updates"]/tbody/tr['+str(i)+']/td[1]/dl/dt[2]/a')
            action = ActionChains(driver)
            action.move_to_element(btn).perform()
            driver.find_element(By.XPATH, '//*[@id="updates"]/tbody/tr['+str(i)+']/td[1]/dl/dt[2]/a').click()
            time.sleep(2)
        # 기업 소개란 수집
            html = driver.page_source
            soup = BeautifulSoup(html,'html.parser')
            
            for i in soup.select('div.corp_bizexp2 > div.left > p'):
                intro_txt = i.text.strip()
                for i in kkma.sentences(intro_txt):
                    intro = i
                    field = 'IT/통신' 
                    it_df = it_df.append({'intro':intro,'field':field},ignore_index=True)
             
           
            driver.back()
            time.sleep(2)
        except:
            print('정보없음')
it통신 - 245

# 체크 리셋
driver.get(url)


# 제조생산, 건설토목, 은행금융, 교육출판, 공공기관 체크
driver.find_element(By.XPATH, '//*[@id="Contents"]/div[2]/div[1]/dl[1]/dd/ul/li[2]/label').click()
driver.find_element(By.XPATH, '//*[@id="Contents"]/div[2]/div[1]/dl[1]/dd/ul/li[4]/label').click()
driver.find_element(By.XPATH, '//*[@id="Contents"]/div[2]/div[1]/dl[1]/dd/ul/li[7]/label').click()
driver.find_element(By.XPATH, '//*[@id="Contents"]/div[2]/div[1]/dl[1]/dd/ul/li[9]/label').click()
driver.find_element(By.XPATH, '//*[@id="Contents"]/div[2]/div[1]/dl[1]/dd/ul/li[10]/label').click()

# 검색
btn = driver.find_element(By.XPATH, '//*[@id="imgSearch"]')
action = ActionChains(driver)
action.move_to_element(btn).perform()
driver.find_element(By.XPATH, '//*[@id="imgSearch"]').click()


# 페이징
for i in range(2,7):
    btn = driver.find_element(By.XPATH, '//*[@id="Contents"]/p[3]/a['+str(i)+']')
    action = ActionChains(driver)
    action.move_to_element(btn).perform()
    driver.find_element(By.XPATH, '//*[@id="Contents"]/p[3]/a['+str(i)+']').click()
    time.sleep(2)
    # 기업 선택
    for i in range(1,11):
        try:
            btn = driver.find_element(By.XPATH, '//*[@id="updates"]/tbody/tr['+str(i)+']/td[1]/dl/dt[2]/a')
            action = ActionChains(driver)
            action.move_to_element(btn).perform()
            driver.find_element(By.XPATH, '//*[@id="updates"]/tbody/tr['+str(i)+']/td[1]/dl/dt[2]/a').click()
            time.sleep(2)
        # 기업 소개란 수집
            html = driver.page_source
            soup = BeautifulSoup(html,'html.parser')
            
            for i in soup.select('div.corp_bizexp2 > div.left > p'):
                intro_txt = i.text.strip()
                for i in kkma.sentences(intro_txt):
                    intro = i
                    field = '타업종' 
                    it_df = it_df.append({'intro':intro,'field':field},ignore_index=True)
             
           
            driver.back()
            time.sleep(2)
        except:
            print('정보없음')


it/통신 - 245
타업종 - 334
전체 - 579

it_df.to_csv('c:/data/it_df.csv')


# 기업
//*[@id="updates"]/tbody/tr[1]/td[1]/dl/dt[2]/a
//*[@id="updates"]/tbody/tr[2]/td[1]/dl/dt[2]/a
//*[@id="updates"]/tbody/tr[3]/td[1]/dl/dt[2]/a
....
//*[@id="updates"]/tbody/tr[1]/td[2]/dl/dt[2]/a
//*[@id="updates"]/tbody/tr[2]/td[2]/dl/dt[2]/a
//*[@id="updates"]/tbody/tr[3]/td[2]/dl/dt[2]/a

# 소개
//*[@id="Contents"]/div[2]/div[2]/div[2]/div[1]/p

# 페이징
//*[@id="Contents"]/p[3]/a[2]
//*[@id="Contents"]/p[3]/a[3]
//*[@id="Contents"]/p[3]/a[4]
//*[@id="Contents"]/p[3]/a[4]
//*[@id="Contents"]/p[3]/a[5]
//*[@id="Contents"]/p[3]/a[6]

# 학습, 테스트 분류
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from collections import Counter

143 249 254 323 # 데이터수집오류 인덱스
it_df.drop([143,249,254,323],inplace=True)


x_train,x_test,y_train,y_test = train_test_split(it_df['intro'],it_df['field'],test_size=0.2)


def kkma_pos(arg):
    token_corpus = []
    for i in kkma.pos(arg):
        if i[1] in ['Noun','Adjective']:
            token_corpus.append(i[0])
    token_corpus = [word for word in token_corpus if len(word) >=2]
    return token_corpus


import operator
from sklearn.feature_extraction.text import CountVectorizer




cv = CountVectorizer(tokenizer=kkma_pos) 
x_train = cv.fit_transform(x_train) # 질문 : ValueError: empty vocabulary; perhaps the documents only contain stop words
cv.get_feature_names()
x_train.toarray()
x_test= cv.transform(x_test)
x_test.toarray()

nb = MultinomialNB()
nb.fit(x_train,y_train)

y_predict = nb.predict(x_test)
sum(y_predict == y_test)
accuracy_score(y_test,y_predict)

from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test,y_predict)
print(classification_report(y_test,y_predict))

pd.crosstab(y_test,y_predict)


■ numpy
- 다차원 배열을 지원하는 라이브러리
- 단일 데이터 타입의 값을 갖는다

import pandas as pd
import numpy as np

x = np.array([1,2,3])
x
type(x)
x.dtype
x.shape
x.ndim

x1 = np.array([[1,2,3],[4,5,6]])
type(x1)
x1.dtype
x1.shape
x1.ndim

x = np.array([1,2,'3'])
x
type(x)
x.dtype
x.shape
x.ndim

lst = [[1,2,3],[4,5,6],[7,8,9]]
type(lst)
x2 = np.array(lst)
x2.shape
x2.ndim

pd.DataFrame(x2)
pd.Series(x2[0])

x2[0]
x2[1]
x2[2]
x2[:,0]
x2[:,1]
x2[:,2]
x2[0:2,0]
x2[0:2,0:2]

x2[0,0]
x2[2,2]
x2[1,2]
x2[[0,2,1],[0,2,2]]


b = np.array([[False,True,False],
             [True,False,True],
             [False,True,False]])

b.dtype
b.shape
x2.shape
x2[b]

x2[x2%2==0]

# 배열을 모두 0으로 채우는 함수
np.zeros((3,3))

# 배열을 모두 1으로 채우는 함수
np.ones((3,3))

# 배열의 사용자가 지정한 값으로 채우는 함수
np.full((5,5),3)

# 대각선으로 1로 채우고 나머지는 0으로 채우는 함수
np.eye(3)

list(range(20))

x = np.array(range(20))
x.shape
x.ndim
x = x.reshape((4,5)) # 미리보기
x.shape
x.reshape((5,4))
x = x.reshape((20,))
x.shape
x.ndim

x = np.array([1,2,3])
y = np.array([4,5,6])

x[0]
y[0]

x+y # 인덱스 끼리 더함
x[0] + y[0]

np.add(x,y)

x-y
np.subtract(x,y)

x*y
np.multiply(x,y)

x/y
np.divide(x,y)

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])


x[0][0] + y[0][0]

x+y # 인덱스 끼리 더함
np.add(x,y)

x-y
np.subtract(x,y)

x*y
np.multiply(x,y)

x/y
np.divide(x,y)

# 행렬의 곱
np.dot(x,y)

x.shape
y.shape

np.sum(x) # 전체합
np.sum(x,axis=0) # 열기준 행들을 더함
np.sum(x,axis=1) # 행기준 열들을 더함

np.mean(x) # 전체 평균
np.mean(x,axis=0) # 열기준 행들 평균
np.mean(x,axis=1) # 행기준 열들 평균

np.var(x) # 전체 분산
np.var(x,axis=0) # 열기준 행들 분산
np.var(x,axis=1) # 행기준 열들 분산

np.std(x) # 전체 표준편차
np.std(x,axis=0) # 열기준 행들 표준편차
np.std(x,axis=1) # 행기준 열들 표준편차

np.max(x) # 전체 최대값
np.max(x,axis=0) # 열기준 행들 최대값
np.max(x,axis=1) # 행기준 열들 최대값

np.min(x) # 전체 최소값
np.min(x,axis=0) # 열기준 행들 최소값
np.min(x,axis=1) # 행기준 열들 최소값

np.median(x) # 전체 중앙값
np.median(x,axis=0) # 열기준 행들 중앙값
np.median(x,axis=1) # 행기준 열들 중앙값


x= x.reshape((4,))
np.max(x)

np.argmax(x) # 최대값이 있는 인덱스
np.argmin(x) # 최소값이 있는 인덱스

x = np.array([[1,2],[3,4],[0,15]])
x
np.argmax(x)
np.argmax(x.reshape((6,)))
np.argmax(x,axis=0)
np.argmax(x,axis=1)

np.cumsum(x)
np.cumsum(x.reshape((6,)))
np.cumsum(x,axis=0)
np.cumsum(x,axis=1)

np.cumprod(x) # 누적곱
np.cumprod(x.reshape((6,))) 
np.cumprod(x,axis=0)
np.cumprod(x,axis=1)

np.prod(x)
np.prod(x.reshape((6,)))
np.prod(x,axis=0)
np.prod(x,axis=1)

list(range(5))
np.arange(5)

y = np.arange(10)
y.reshape((5,2)) # 행우선으로 값이 채워진다
y.reshape((5,2),order='C') # 행우선으로 값이 채워진다. 기본값
y.reshape((5,2),order='F') # 열우선으로 값이 채워진다. 

y = np.arange(10).reshape((5,2),order='F')
y
y.reshape((10,),order='F')
y.flatten('C')
y.flatten('F')

y.ravel()
y.ravel('C')
y.ravel('F')


# array 합치기
x = np.array([[1,2,3],[4,5,6]])
y = np.array([[7,8,9],[10,11,12]])

np.concatenate([x,y])
np.concatenate([x,y],axis=0)
np.vstack([x,y])

np.concatenate([x,y],axis=1)
np.hstack([x,y])

x = np.array([[1,2],[3,4]])
x + 4 # broadcast, 논리적으로 안될 것 같지만 다 더해주는게 된다
x + np.full((2,2),4)

z = np.array([10,20])
z.shape

x + z # z는 1차원인데 2차원으로 바꾸고 더해준다

x = np.arange(3)
x.repeat(2)
x.repeat([2,3,4])

x = np.array([[1,2],[3,4]])
x.repeat(2)
x.repeat(2,axis=1)
x.repeat(2,axis=0)
np.tile(x,2)

np.unique([1,2,3,1,2,3,3,4,2])
x = np.array(['a','a','b','b'])
np.unique(x)
np.unique(x,return_counts=True)
word, cnt = np.unique(x,return_counts=True)
word
cnt

y = np.array([[1,0,0],[1,0,0],[1,0,0]])
np.unique(y)
np.unique(y,axis=0)
np.unique(y,axis=1)

x = np.arange(0,20,2)
y = np.arange(0,30,3)
x > y
x < y

np.maximum(x,y) # 큰 값 찾기
np.minimum(x,y) # 작은 값 찾기

np.union1d(x,y) # 합집합
np.intersect1d(x,y) # 교집합
np.setdiff1d(x,y) # 차집합

x = np.array([5,6,3,2,8,4,1])
x

np.sort(x) # 미리보기
x

np.sort(x)[::-1] # 미리보기

y = np.array([[3,1,7],[2,8,4],[5,4,6]])
y
np.sort(y,axis=1) # 기본값
np.sort(y,axis=0)

np.sort(y,axis=1)[::-1]
np.sort(y,axis=0)[::-1]

x = np.array([5,6,3,2,8,4,1])
np.sort(x)

x.argsort() # 오름차순 인덱스만 뽑기

x[x.argsort()]
x[x.argsort()[::-1]]

■ kNN(k-Nearest Neighbors), k 번째 가장 가까운(최근접) 이웃
- 사회적인 관점
    비슷한 사람끼리 모이는 성질
    비슷한 취향의 사람끼리 모여서 동호회를 만든다.
    비슷한 부류의 계층 사람끼리 친분을 맺기도 한다.
    

- 공간적인 관점
    가구점
    맛집
    중고차 매장

- 거리계산 알고리즘
- 유클리드 거리(Euclidean distance)

두점 사이의 거리 
1. 수직선

-----|----|------
    a     b
    |큰값 - 작은값|

2. 평면 (유클리드 거리)
- 피타고라스 정리

p(x1,y1)
q(x2,y2)

import math
math.sqrt((x1-x2)**2 + (y1-y2)**2)
np.sqrt((x1-x2)**2 + (y1-y2)**2)

[문제196] 거리계산을 한 후 가장 가까운 3개만 추출해주세요.
pointlst = [(1,1),(1,0),(2,0),(0,1),(2,2),(1,5),(2,3)]
point = [2,1]

dist = []
for i in pointlst:
    dist.append(np.sqrt((i[0]-point[0])**2+(i[1]-point[1])**2))

sorted(dist)

enumerate

for idx, value in enumerate(dist):
    print(idx,value)

[i for i in sorted(enumerate(dist),key=lambda x : x[1])][:3]
k = [i[0] for i in sorted(enumerate(dist),key=lambda x : x[1])][:3]

for i in k:
    print(pointlst[i])

np.argsort(dist)
for i in np.argsort(dist)[:3]:
    print(pointlst[i])

pointlst[np.argsort(dist)] #리스트의 인덱스를 array값으로 할 수는 없다

point_array = np.array(pointlst)
point_array[np.argsort(x)][:3]



########################################
array로 바꿔서 하는 법


pointlst = [(1,1),(1,0),(2,0),(0,1),(2,2),(1,5),(2,3)]
point = [2,1]

point1 =np.array(pointlst)
point2 = np.array(point)
dist = np.sqrt(np.sum(pow(point1 - point2,2),axis=1)) # for 문 없이 array broadcast 계산으로 간단히 해결
dist = np.sqrt(np.sum(np.square(point1 - point2),axis=1))

np.argsort(dist)[:3]

for i in np.argsort(dist)[:3]:
    print(pointlst[i])


■ take

np.take(point1, np.argsort(dist)[:3], axis=0) # for문 없이 바로 뽑기





재료       단맛         아삭한맛        종류          거리
----------------------------------------------------------
포도        8             5            과일          np.sqrt((6-8)**2 +(4-5)**2) = 2.2       
콩          3             7            채소         np.sqrt((6-3)**2 +(4-7)**2) = 4.2
견과        3             6            단백질        np.sqrt((6-3)**2 +(4-6)**2) = 3.6
오렌지      7             3            과일          np.sqrt((6-7)**2 +(4-3)**2) = 1.4


토마토 단맛 6, 아삭한맛 4
k=1
오렌지와 토마토 거리는 1.4 로 가까운 이웃하여 과일로 분류

k=3 # 이래서 k값을 일반적으로는 홀수로. 다수결로 결정이 나야하니까
오렌지,포도,견과 세가지 사이에서 다수결로 정한다.
종류 과일, 과일, 단백질 중에 다수결로 인해 과일로 분류한다.


[문제197] food.csv읽어 들인후 토마토 단맛 6, 아삭한맛 4를 이용해서 거리계산한 값을 dist 컬럼에 입력해주세요.
dist 컬럼의 값을 오름차순으로 rank컬럼을 추가해주세요.

food = pd.read_csv('c:/data/food.csv')
food

feature : 변수, 예측, 분류를 하기 위해서 사용되는 입력변수
독립변수, 설명변수, 입력변수 : 종속변수에 영향을 주는 변수
종속변수, 결과변수 : 영향을 받는 변수

food['dist'] = np.sqrt((food['sweetness'] - 6)**2 + (food['crunchiness']-4)**2)
food['rank'] = food['dist'].rank(ascending=True,method='dense').astype(int)
y = food.loc[food['rank'] <= 3,'class'].value_counts() == food.loc[food['rank'] <= 3,'class'].value_counts().max()
x = food.loc[food['rank'] <= 3,'class'].value_counts()[y]
x.index[0]


# array로 바꿔서 해결하기
x_train = np.array(food.iloc[:,1:3])
x_train.shape
y = np.array([[6,4]]) # 괄호 두개로 1행2열 맞춰줘 위에꺼랑
y.shape
food['dist'] = np.sqrt(np.sum(np.square(x_train - y),axis=1))
food['rank'] = food['dist'].rank(ascending=True,method='dense').astype(int)
food[food['rank']<=3]['class']

from collections import Counter
Counter(food[food['rank']<=3]['class']).most_common(1)[0][0]




# knn으로 바로 해결하기
from sklearn.neighbors import KNeighborsClassifier
x_train = np.array(food.iloc[:,1:3])
x_train.shape
label = food['class']
label

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,label)
clf.predict([[6,4]])
clf.predict([[6,4]])[0]

[문제198] bmi 데이터를 이용해서 키 : 178, 몸무게 : 71 일 때 분류해주세요.
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

bmi = pd.read_csv('c:/data/bmi.csv')
bmi
x_train = np.array(bmi.iloc[:,0:2])
x_train.shape
label = bmi.label
label

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,label)
clf.predict([[186,100]])
clf.predict([[178,71]])[0]

x_train,x_test,y_train,y_test = train_test_split(bmi.iloc[:,0:2],bmi.label,test_size=0.2)
x_train.shape
y_train.shape
x_test.shape
y_test.shape

# 균일하게 학습되었는지 확인하기
Counter(y_train)
Counter(y_test)

clf = KNeighborsClassifier(n_neighbors=21)
clf.fit(x_train,y_train)
clf.classes_

# 정확도 구하는 3가지 방법
y_predict = clf.predict(x_test)
sum(y_predict == y_test)/len(y_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)

clf.score(x_test,y_test)

pd.crosstab(y_test,y_predict)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))

clf.predict(np.array([[186,100]]))[0]

import pickle
file = open('c:/data/clf_knn.pkl','wb')
pickle.dump(clf,file)
file.close()

file = open('c:/data/clf_knn.pkl','rb')
clf_knn = pickle.load(file)
file.close()

clf_knn.classes_

[문제199] k값의 변화에 따른 정확도를 그래프로 시각화 해주세요.
x_train,x_test,y_train,y_test = train_test_split(bmi.iloc[:,0:2],bmi.label,test_size=0.2)

accuracy = []
for k in range(1,50,2):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(x_train,y_train)
    accuracy.append(clf.score(x_test,y_test))

import matplotlib.pyplot as plt

plt.plot(range(1,50,2),accuracy)

# 설명(입력, 독립) 변수 단위가 다를 때 
■ Feature Scailing
- 서로 다른 변수의 값 범위를 일정한 수준으로 맞추는 작업
- 표준화(standardization)
    평균 0, 표준편차 1 인 표준정규분포를 가진 값으로 변환
    
표준화 =  (관측값 - 평균)/표준편차
    
- 정규화(normalization)
    최소값과 최대값을 사용해서 0 ~ 1 사이의 데이터로 변환
    
정규화 = x - x.min()/ x.max() - x.min()


x = np.arange(9,dtype=np.float) - 3
x
x.shape
x.reshape(9,1)
np.reshape(x,(9,1))
x = x.reshape(-1,1) # 행의 개수 모를 때 열의 개수 기준 맞추고싶을때 -1
x.shape
pd.DataFrame(x).describe()
x = np.vstack([x,[100]])
pd.DataFrame(x).describe()

# 표준화
y = (x - np.mean(x)) / np.std(x)
pd.DataFrame(y).describe()

# 정규화
(x - x.min()) / (x.max() - x.min())

from sklearn.preprocessing import StandardScaler

s = StandardScaler()
s.fit_transform(x)

s = StandardScaler().fit(x)
r = s.transform(x)

np.mean(r)
np.std(r)

s.mean_
s.scale_
s.var_

from sklearn.preprocessing import scale
# 이거는 관측값의 원래 평균, 표준편차는 알 수 없어 그래서 위에 StandardScaler로 많이 함
scale(x)

np.mean(scale(x))
np.std(scale(x))


from sklearn.preprocessing import MinMaxScaler

m = MinMaxScaler()
m.fit_transform(x)

m = MinMaxScaler().fit(x)
m.transform(x)
m.data_min_
m.data_max_

from sklearn.preprocessing import minmax_scale
minmax_scale(x) # 이것도 바로 할 수 있지만, 관측값의 원래 최대값 최소값 따로 구해야 해

[문제200] bmi 데이터를 이용해서 키 : 178, 몸무게 : 71 일 때 분류해주세요.
단, 표준화로 변환한 후 수행하세요.
standscaler = StandardScaler()

data = standscaler.fit_transform(bmi.iloc[:,0:2]) # 모델 pickle할 때 standscaler 얘도 저장해야돼

standscaler.mean_
standscaler.scale_
standscaler.var_


x_train,x_test, y_train, y_test = train_test_split(data,bmi.label,test_size=0.2)


clf = KNeighborsClassifier(n_neighbors=21)
clf.fit(x_train,y_train)
clf.classes_

# 정확도 구하는 3가지 방법
y_predict = clf.predict(x_test)
sum(y_predict == y_test)/len(y_test)

clf.score(x_test,y_test)
clf.predict(standscaler.transform([[178,71]]))[0] # test 데이터도 transform 해줘야한다(스케일링)

# clf 저장
[문제201] bmi 데이터를 이용해서 키 : 178, 몸무게 : 71 일 때 분류해주세요.
단, 정규화로 변환한 후 수행하세요.


minmaxscaler = MinMaxScaler()


data = minmaxscaler.fit_transform(bmi.iloc[:,0:2]) # 모델 pickle할 때 minmaxscaler 얘도 저장해야돼

minmaxscaler.data_max_
minmaxscaler.data_min_

x_train,x_test, y_train, y_test = train_test_split(data,bmi.label,test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=21)
clf.fit(x_train,y_train)
clf.classes_

# 정확도 구하는 3가지 방법
y_predict = clf.predict(x_test)
sum(y_predict == y_test)/len(y_test)

clf.score(x_test,y_test)
clf.predict(standscaler.transform([[178,71]]))[0] # test 데이터도 transform 해줘야한다(스케일링)

# clf 저장

[문제202] 붓꽃데이터입니다. KNN으로 분류해주세요.
iris = pd.read_csv('c:/data/iris.csv')
iris.info()
iris

 0   SepalLength  꽃받침 길이
 1   SepalWidth   꽃받침 폭(너비)
 2   PetalLength  꽃잎 길이
 3   PetalWidth   꽃잎 폭(너비)
 4   Name         붓꽃 이름

iris.Name.unique()

plt.scatter(iris.SepalLength, iris.SepalWidth,c=iris.Name) # 컬러에 문자값들어가니까 오류.
########################
iris_name_labels = iris.Name.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}) # map !!!!!!!!!!!!!!!!! 문자형을 수치형으로 변환 바로 가능
########################
plt.scatter(iris.SepalLength, iris.SepalWidth,c=iris_name_labels)
plt.scatter(iris.PetalLength, iris.PetalWidth,c=iris_name_labels, marker='v')
plt.legend(['data2'], loc='lower right')


x_train,x_test,y_train,y_test = train_test_split(iris.iloc[:,0:-1],iris.Name,test_size=0.2)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train)
clf.classes_
clf.predict(x_test)
clf.score(x_test,y_test)


pd.crosstab(y_test,clf.predict(x_test))



x_train,x_test,y_train,y_test = train_test_split(iris.iloc[:,2:4],iris.Name,test_size=0.2)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train,y_train)
clf.classes_
clf.predict(x_test)
clf.score(x_test,y_test)
pd.crosstab(y_test,clf.predict(x_test))

test = np.array([[1.0,0.1],
                 [5.1,1.8]])


clf.predict(test)

# scatter plot 그리기

setosa = iris[iris.Name=='Iris-setosa']
versicolor = iris[iris.Name=='Iris-versicolor']
virginica = iris[iris.Name=='Iris-virginica']

fig, ax = plt.subplots()
fig.set_size_inches(10,10)
ax.scatter(setosa['PetalLength'],setosa['PetalWidth'],label = 'Iris-setosa',marker='v',facecolor='blue')
ax.scatter(versicolor['PetalLength'],versicolor['PetalWidth'],label = 'Iris-versicolor',marker='s',facecolor='red')
ax.scatter(virginica['PetalLength'],virginica['PetalWidth'],label = 'Iris-virginica',marker='d',facecolor='yellow')
ax.legend()


# 그래프 쉽게 그리기
# seaborn으로 하자
import seaborn as sns

sns.scatterplot(data=iris,x='PetalLength',y='PetalWidth',hue='Name')


sns.pairplot(iris)



https://archive.ics.uci.edu/ml/index.php
[문제203] 유방암 데이터입니다. KNN 알고리즘을 이용해서 분류해주세요.

wisc = pd.read_csv('c:/data/wisc_bc_data.csv')
wisc.info()
wisc.diagnosis

wisc.diagnosis.unique()

B -> Benign (양성)
M -> Malignant (악성)

wisc.isnull().sum()

표준화



■ overfitting
- 학습데이터(training set)에 대해 과하게 학습된 상황을 의미
- 학습데이터 이외의 데이터(test set)에 대해서는 모델 정확도가 낮다.
- 데이터 모델의 특성에 비해 모델이 너무 복잡한 경우
- 학습데이터가 부족할 경우

■ underfitting
- 학습데이터도 정확도가 낮다.
- 데이터 양이 너무 적을 경우 발생
- 반복횟수가 너무 적을 경우 발생
- 데이터 모델의 특성에 비해 모델이 너무 간단한 경우


■ k-fold 교차검증
기존데이터를 k개로 나눠서 k번 정확도를 검증하는 방법

- 5-fold면 학습데이터를 5조각으로 나눠 1조각을 검증데이터로 사용하고 나머지 4조각을 학습데이터로 사용한다.
- 첫번째 조각부터 5번째 조각까지 한번씩 검증하게 되어 결과적으로 5개의 검증결과에 해당하는 정확도의 평균을 검증결과의 점수로 표현
- 레이블(정답)의 분포도가 좋을 때 사용한다.

stratified k fold
불균형한 분포도를 가진 레이블(정답) 데이터셋을 위한 방식

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from collections import Counter



iris = pd.read_csv('c:/data/iris.csv')

iris.iloc[:,0:4]
iris.Name

x_train, x_test, y_train, y_test = train_test_split(iris.iloc[:,0:4],iris.Name,test_size=0.1,random_state=123)
x_train.shape
y_train.shape
Counter(y_train)
Counter(y_test)

knn_clf = KNeighborsClassifier()
parameters = {'n_neighbors':[3,5,7,9,11]}
grid_knn = GridSearchCV(estimator=knn_clf, param_grid = parameters,cv=5,refit=True,return_train_score=True)
grid_knn.fit(x_train,y_train)
pd.DataFrame(grid_knn.cv_results_)[['params','mean_test_score','rank_test_score']]

grid_knn.best_params_ # 최적 파라미터
grid_knn.best_score_ # 최고 정확도

y_pred = grid_knn.predict(x_test)
accuracy_score(y_test,y_pred)

knn_estimator = grid_knn.best_estimator_ # GridSearchCV 의 refit으로 이미 학습된 estimator를 반환
y_pred = knn_estimator.predict(x_test)
accuracy_score(y_test,y_pred)

■ clustering(군집화)
- unsupervised learning
- 유사한 성질을 가지는 데이터끼리 cluster(군집) 
- 동일한 집단에 소속된 관측치들은 서로 유사할 수록 좋다.
- 상이한 집단에 소속된 관측치들은 서로 다를수록 좋다.
- 군집내 데이터들의 거리는 가깝고 군집 간 거리는 먼 경우 군집이 잘되어있다.
- 유사한 뉴스그룹끼리 묶어서 놓은 것
- 특허문서분석
- 관련된 문서들을 모을 때

■ k-means
주어진 데이터에서 k개의 cluster 묶는 알고리즘

k-means 동작방식
1. 랜덤으로 중심지정(k-means++에서는 극단에있는 값들 골라줌)
2. 각 중심으로부터 가까운 데이터들에 레벨을 붙인다.
3. 각 레벨의 평균으로 중심이 이동된다.
4. 클러스터의 중심이 더 이상 변하지 않으면 중단.


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


iris = pd.read_csv('c:/data/iris.csv')

model = KMeans(n_clusters=3)
model.fit(iris.iloc[:,0:-1])
model.labels_
model.cluster_centers_

iris['cluster'] = model.labels_
iris

colormap = np.array(['red','blue','black'])
plt.scatter(iris.PetalLength,iris.PetalWidth,c=colormap[model.labels_],s=40)

model.inertia_ # 이 값이 작을수록 응집도가 크다
inertia 값은 군집화된 후에 각 중심점에서 군집의 데이터간의 거리를 합산한 것이므로 군집의 응집도를 나타내는 값이다.



inertia = []
for k in range(1,10):
    model = KMeans(n_clusters=k)
    model.fit(iris.iloc[:,0:4])
    inertia.append(model.inertia_)
    
    
plt.plot(range(1,10),inertia,'-o') # 꺾이는 지점 - 엘보우점 (보통 k값 이 근처에서 정함)


teens = pd.read_csv('c:/data/snsdata.csv')

# 결측값 확인
teens.info()
teens.isnull().sum()

teens['gender'].value_counts()
teens['gender'].value_counts(dropna=True) #기본값
teens['gender'].value_counts(dropna=False)
teens['gender'].isnull().sum()
teens['gender'].describe()

teens['gender'].value_counts(dropna=False).plot(kind='bar')

teens_df = pd.DataFrame(teens['gender'].value_counts(dropna=False)).reset_index()
teens_df.columns=['gender','freq']
teens_df.loc[teens_df.gender.isnull(),'gender'] = 'NA' # NA값 출력하도록 만들기


import seaborn as sns
sns.barplot(x='gender',y='freq',data=teens_df)

teens['age'].isnull().sum()
teens['age'].describe()
teens.describe()

teens.boxplot(column='age')
sns.boxplot(x=teens['age'])
sns.boxplot(y=teens['age'])

len([i for i in teens['age'] if (i < 13) | (i >= 20) ])

teens['age'] = [np.nan  if (i < 13) | (i >= 20) else i for i in teens['age']]
teens['age'].describe()
teens['age'].isnull().sum()

# 졸업연도 기준으로 평균나이
teens[['gradyear','age']]

gradage = teens['age'].groupby(teens['gradyear']).mean()
gradage.index

gradage.loc[2006]
teens['agemean'] = [gradage.loc[i] for i  in teens['gradyear']]
teens

teens['age'] = np.where(teens['age'].isnull(),teens['agemean'],teens['age']) # np.where(조건,True,False)
teens['age'].isnull().sum()
teens['age'].describe()

teens['female'] = [1 if i == 'F' else 0 for i in teens['gender']]
teens['female'].value_counts()

teens['no_gender'] = [ 1 if pd.isnull(i) else 0 for i in teens['gender'] ]
teens['no_gender'].value_counts()

teens.info()
data = teens.iloc[:,4:40]

from sklearn.cluster import KMeans
model = KMeans(n_clusters=5)
model.fit(data)
model.labels_
model.cluster_centers_.shape
teens['cluster'] = model.labels_
teens.loc[1:5,['cluster','gender','age','friends']]

teens['age'].groupby(teens['cluster']).mean()
teens['female'].groupby(teens['cluster']).mean()
teens['friends'].groupby(teens['cluster']).mean()

for name,group in teens.groupby('cluster'):
    print(name)
    print(group)

teens.info()
teens[teens['cluster']==0][teens.columns]

# 필요없는 컬럼 제외하고 보기
col = teens.columns.difference(['gradyear','gender','age','agemean','female','no_gender','cluster','friends'])

# 군집별 특징 확인하기
teens_0 = teens[teens['cluster']==0][col]
teens_0.sum().sort_values(ascending=False)[:10]

teens_1 = teens[teens['cluster']==1][col]
teens_1.sum().sort_values(ascending=False)[:10]

teens_2 = teens[teens['cluster']==2][col]
teens_2.sum().sort_values(ascending=False)[:10]

teens_3 = teens[teens['cluster']==3][col]
teens_3.sum().sort_values(ascending=False)[:10]

teens_4 = teens[teens['cluster']==4][col]
teens_4.sum().sort_values(ascending=False)[:10]

data['cluster'] = model.labels_
data.columns[:-1]

# 중심점 중 가장 핵심이 되는 것 확인하기
model.cluster_centers_.argsort()[:,::-1][0]

# 위에서 본 거랑 똑같은 10개가 나옴
data.columns[:-1][model.cluster_centers_.argsort()[:,::-1][0][:10]]
data.columns[:-1][model.cluster_centers_.argsort()[:,::-1][1][:10]]
data.columns[:-1][model.cluster_centers_.argsort()[:,::-1][2][:10]]
data.columns[:-1][model.cluster_centers_.argsort()[:,::-1][3][:10]]
data.columns[:-1][model.cluster_centers_.argsort()[:,::-1][4][:10]]


ks = range(1,10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(data)
    inertias.append(model.inertia_)
    
# plot ks vs inertias

plt.plot(ks, inertias, '-o')
plt.xlabel('number of cluster, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.ticklabel_format(style='plain') # 지수형식을 보통으로 바꾸기
plt.show()


■ 연관분석(association analysis)
대량의 데이터에 숨겨진 항목간의 연관규칙을 찾아내는 기법으로서 장바구니 분석(market basket analysis)라고 한다.
실제 연관성분석은 모든 기업에서 다양한 마케팅활동에 활용하고 있고, 나아가 사회 네트워크 분석에도 활용할 수 있다.

장점
- 대규모 거래 데이터에 대해 작업할 수 있다.
- 이해하기 쉬운 규칙을 생성한다.
- 데이터마이닝과 데이터베이스에서 예상치 못한 지식을 발굴하는데 유용하다.

단점
- 작은데이터셋에는 그다지 유용하지 않다.
- 통찰력과 상식을 분리하기 위한 노력 필요

지지도(support)
- 전체 거래 중 연관성 규칙을 구성하는 항목들이 포함된 거래의 비율
- 전체 거래 중 연관성 규칙이 얼마나 빈번히 발생하는지 확인하는 지표
- 항목에 대한 거래 개수 / 전체 거래 수


신뢰도(confidence)
- 조건이 발생했을 때 동시에 일어날 확률. 신뢰도가 1에 가까울수록 의미있는 연관성을 가지고 있다.
{조건} -> {결과}
- 조건을 포함하는 거래 중 연관성 규칙이 얼마나 빈번히 발생하는 지 확인하는 지표
- 조건과 결과 항목을 포함하는 거래 수 / 조건 항목을 포함한 거래 수


confidence(x -> y) = support(x,y) / support(x)

거래번호           구매물품
----------        -----------
1                 우유, 버터, 시리얼
2                 우유, 시리얼
3                 우유, 빵
4                 버터, 맥주, 오징어

support(우유 -> 시리얼) : 우유와 시리얼을 동시에 구매할 확률, {우유,시리얼} 포함한 거래수 / 전체거래수 , 2/4
confidence(우유 -> 시리얼) : 우유를 구매할 때 시리얼 같이 구매할 조건부확률, {우유,시리얼} / {우유}

support(우유 -> 시리얼) = 50%
confidence(우유 -> 시리얼) = 66.7%

support(시리얼 -> 우유) = 50%
confidence(시리얼 -> 우유) = 100%

향상도(lift)

- 지지도와 신뢰도를 동시에 고려하는 지표

lift(시리얼,우유) = confidence(시리얼 -> 우유) / support(우유) = 1/ (3/4) = 1.33

or

lift(시리얼,우유) = support(시리얼 -> 우유) / support(시리얼) * support(우유)


향상도 값이 1인 경우 조건과 결과는 우연에 의한 관계이며, 1보다 클수록 우연이 아닌 의미있는 연관성을 가진 규칙으로 해석

거래번호          항목
1               A,C,D
2               B,C,E
3               A,B,C,E
4               B,E

항목      지지도
A         2
B         3
C         3
D         1
E         3

지지도 2이상인 항목만 추출
항목      지지도
A         2
B         3
C         3
E         3

항목       지지도
A B        1
A C        2
A E        1
B C        2
B E        3
C E        2 

지지도가 2이상인 항목만 추출

항목      지지도
A C       2
B C       2
B E       3
C E       2

각각 항목에서 첫번째 항목을 기준으로 동일한 것을 찾아보세요.
B C E

  거래번호          항목
  1               A,C,D              
  2               B,C,E
  3               A,B,C,E
  4               B,E


발견된 규칙 항목     지지도
B,C,E              2

■ apriori algorithm
- 집합의 크기가 1인 경우부터 차례로 늘려가면서 처리하는 알고리즘
- k개인 빈도가 높은 항목을 구했다면 그 다음에는 K+1개인 항목의 집합을 계산한다.


거래번호           구매물품
----------        -----------
1                 우유, 버터, 시리얼
2                 우유, 시리얼
3                 우유, 빵
4                 버터, 맥주, 오징어


dataset = [['우유', '버터', '시리얼'],
           ['우유','시리얼'],
           ['우유','빵'],
           ['버터','맥주','오징어']]
dataset

(base) C:\Users\USER>pip install mlxtend

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
# transaction : 논리적으로 DML을 수행하는 작업 단위

te = TransactionEncoder()
te_data = te.fit_transform(dataset)
te_data
te.columns_

df = pd.DataFrame(te_data,columns = te.columns_)
df
apriori(df,use_colnames=True) # support 0.5이상만
f = apriori(df,min_support = 0.1, use_colnames=True)
f
association_rules(f,metric='confidence')

# pandas 말고 애초에 [[]] 이렇게 읽어들이는 방법
import csv

groc = []

with open('c:/data/groceries.csv','r') as file:
    csv_data = csv.reader(file)
    for row in csv_data:
        groc.append(row)

groc


te = TransactionEncoder()
te_data = te.fit_transform(groc)
te_data
te.columns_

df = pd.DataFrame(te_data,columns = te.columns_)
df
apriori(df,use_colnames=True) 
f = apriori(df,min_support = 0.005, use_colnames=True)
f
rules = association_rules(f,metric='confidence',min_threshold=0.01)
rules[rules['antecedents'] == {'whole milk','butter'}]

■ Decision Tree(결정트리 의사결정나무)
- 머신러닝 알고리즘 중 직관적으로 이해하기 쉬운 알고리즘
- 데이터에 있는 규칙을 학습을 통해 자동으로 찾아내는 트리기반의 분류규칙을 만든다.
- if else 기반
- 지도학습
- 스무고개 게임, 룰기반의 프로그램
- CART(classification and Regression Tree) : 지니지수(Gini index)
- C4.5, C5 : 엔트로피(Entropy index)
- 분류 알고리즘

import pandas as pd
from pandas import Series,DataFrame
from collections import Counter
import math

성별 = ['남','남','남','남','남','남','여','여','여','여']
결혼 = ['기혼','미혼','기혼','기혼','미혼','미혼','미혼','기혼','미혼','기혼']
구매 = ['예','예','예','아니오','예','예','아니오','아니오','아니오','아니오']
customer = DataFrame({'성별':성별,'결혼':결혼,'구매':구매})
customer

result = pd.crosstab(customer['구매'],customer['성별'])
result

result2 = pd.crosstab(customer['구매'],customer['결혼'])
result2

if 성별 =='남':
    if 결혼 = '미혼':
        구매 = '예'
    else:
        구매 = '예','아니오'
else:
    구매 = '아니오'


CART : 지니지수
- 불확실성을 의미
- 지니지수는 얼마나 불확실한가(얼마나 많이 섞여있는가)
- 지니지수 0 이라는 것은 불확실성이 0이라는 것으로 같은 특성을 가진 데이터끼리 잘 모여 있다는 뜻
- 엔트로피와 거의 동일하지만 훨씬 더 빨리 계산 가능

1 - ∑P²

성별 지니지수
result

G(상위) = 1-(5/10)**2 -(5/10)**2= 0.5
G(남자) = 1-(5/6)**2 -(1/6)**2= 0.278
G(여자) = 1-(0/4)**2 -(4/4)**2= 0 # 불확실성이 낮다. 
G(성별) = (6/10)*0.278 + (4/10)*0 = 0.1668 # 0.5 -> 0.1668 불확실성 감소되었다.

result2






C4.5, C5 : 엔트로피(Entropy index)
- Entropy는 주어진 데이터 집합의 혼잡도를 의미한다.
- 서로 다른 값이 섞여있으면 엔트로피가 높고 같은 값이 섞여있으면 엔트로피는 낮다.
 -∑P*log2(P)

성별 엔트로피
result

G(상위) = -(5/10)*math.log2(5/10) -(5/10)*math.log2(5/10) = 1
G(남자) = -(5/6)*math.log2(5/6) -(1/6)*math.log2(1/6) = 0.65
G(여자) = -(0/4)*math.log2(0/4) -(4/4)*math.log2(4/4) = 0 
G(성별) = (6/10)*0.65 + (4/10)*0 = 0.39
IG(성별) = E(상위) - E(성별) = 1-0.39 = 0.61 # 0.61만큼 감소 
IG(Information Gain) : 클수록 가치가 있다.



결혼 엔트로피
result2

G(상위) = -(5/10)*math.log2(5/10) -(5/10)*math.log2(5/10) = 1
G(기혼) = -(2/5)*math.log2(2/5) -(3/5)*math.log2(3/5) = 0.97
G(여자) = -(3/5)*math.log2(3/5) -(2/5)*math.log2(2/5) = 0.97 
G(결혼) = (5/10)*0.97 + (5/10)*0.97 = 0.97
IG(성별) = E(상위) - E(성별) = 1-0.97 = 0.03 # 0.03만큼 감소 별 의미 없음 


# 의사결정나무 알고리즘 이용
성별 = ['남','남','남','남','남','남','여','여','여','여']
결혼 = ['기혼','미혼','기혼','기혼','미혼','미혼','미혼','기혼','미혼','기혼']
구매 = ['예','예','예','아니오','예','예','아니오','아니오','아니오','아니오']
customer = DataFrame({'성별':성별,'결혼':결혼,'구매':구매})
customer

-카테고리형 데이터(categorical data)를 수치형데이터(numerical data)로 변환
customer['성별'].map({'남':0,'여':1})

customer['성별'].astype('category').cat.codes

from sklearn.preprocessing import LabelEncoder
gender_le = LabelEncoder()
gender = gender_le.fit_transform(customer['성별'])
gender
gender_le.classes_
gender_le.inverse_transform(gender) # 역변환

marry_le = LabelEncoder()
marry = marry_le.fit_transform(customer['결혼'])
marry_le.classes_
marry_le.inverse_transform(marry)

# one hot encoding
pd.get_dummies(customer['성별'])

from sklearn.preprocessing import OneHotEncoder
onehot_en = OneHotEncoder()
gender_onehot = onehot_en.fit_transform(customer[['성별']])
gender_onehot
gender_onehot.toarray()

DataFrame(gender_onehot.toarray().astype('int'))

df = DataFrame([gender,marry]).T
df.columns = ['성별','결혼']
df['구매'] = customer['구매']
df

from sklearn.tree import DecisionTreeClassifier
# 지니가 기본값

# model = DecisionTreeClassifier(criterion='gini',max_depth=2) # 기본값
model = DecisionTreeClassifier(criterion='entropy',max_depth=1)
model.fit(df[['성별','결혼']],df['구매']) # 학습, 정답
y_predict = model.predict(df[['성별','결혼']])
(y_predict == df['구매']).mean()
model.score(df[['성별','결혼']],df['구매'])

model.classes_
model.feature_importances_

model.predict([[0,1]])[0] # 남, 미혼
model.predict([[0,0]])[0] # 남, 기혼
model.predict([[1,1]])[0] # 여, 미혼
model.predict([[1,0]])[0] # 여, 기혼

# 의사결정나무 시각화

1. 다운로드 받아서 설치, 환경path설정 체크
https://graphviz.gitlab.io/_pages/Download/Download_windows.html

(base) C:\Users\USER>pip install graphviz

2. pip instal graphviz

import graphviz
from sklearn.tree import export_graphviz

dot_data = export_graphviz(model,out_file=None,
                           feature_names=['성별','결혼'],
                           class_names=model.classes_,
                           filled=True,rounded=True, special_characters=True)
graphviz.Source(dot_data)


iris = pd.read_csv('c:/data/iris.csv')
iris.info()
iris.iloc[:,0:-1]
iris.Name

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(iris.iloc[:,0:-1],iris.Name,test_size=0.2)

from collections import Counter
Counter(y_train)
Counter(y_test)

iris_model = DecisionTreeClassifier(max_depth=3)
iris_model.fit(x_train,y_train)
iris_model.classes_
iris_model.feature_importances_ # 결정적인 독립변수가 뭐였는지 확인

y_pred = iris_model.predict(x_test)
(y_pred == y_test).mean()

iris_model.score(x_test,y_test)

iris.columns[:-1]

dot_data = export_graphviz(iris_model,out_file=None,
                           feature_names=iris.columns[:-1],
                           class_names=iris_model.classes_,
                           filled=True,rounded=True, special_characters=True)
graphviz.Source(dot_data)


iris_model.feature_importances_

for i in range(0,4):
    print('{} : {}'.format(iris.columns[:-1][i],iris_model.feature_importances_[i]))

# 둘의 인덱스끼리 연결시키는 법
grid CV

or

for i,j in zip(iris.columns[:-1],iris_model.feature_importances_):
    print('{} : {:.3F}'.format(i,j))
    
iris.columns[:-1]
iris_model.feature_importances_

import matplotlib.pyplot as plt
import seaborn as sns

plt.bar(iris.columns[:-1],iris_model.feature_importances_)
plt.barh(iris.columns[:-1],iris_model.feature_importances_)

sns.barplot(x=iris_model.feature_importances_, y =iris.columns[:-1])
sns.barplot(y=iris_model.feature_importances_, x =iris.columns[:-1])

# max_depth 값 결정하기

max_depth : 트리의 최대 깊이, 기본값 None
min_samples_split: 기본값 2, 노드로 분할하기 위한 최소한의 샘플데이터 수 # sample수가 2만 넘으면 계속 분할하겠다.
min_samples_leaf : leaf가 되기 위한 최소한의 샘플데이터 수  # leaf의 샘플수가 30이상이면 더이상 분할하지 말아봐
max_leaf_nodes : leaf의 최대개수
max_features : 분할하기 위해 최대로 사용할 수 있는 피쳐 개수, 기본값 None(모든 feature를 다 사용),
                'sqrt': sprt(피쳐의 수),
                'auto': sqrt 동일
                'log': log2(피쳐의 수)

iris_model = DecisionTreeClassifier(max_features='sqrt')
iris_model.fit(x_train,y_train)
iris_model.classes_
iris_model.feature_importances_ # 결정적인 독립변수가 뭐였는지 확인

y_pred = iris_model.predict(x_test)
(y_pred == y_test).mean()

iris_model.score(x_test,y_test)

iris.columns[:-1]

dot_data = export_graphviz(iris_model,out_file=None,
                           feature_names=iris.columns[:-1],
                           class_names=iris_model.classes_,
                           filled=True,rounded=True, special_characters=True)
graphviz.Source(dot_data)

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

iris
x_train,x_test,y_train,y_test = train_test_split(iris.iloc[:,0:-1],iris.Name,test_size=0.2)

# 선언
dtree = DecisionTreeClassifier()
parameters = {'max_depth':[3,4,5],'min_samples_split':[3,4]}
grid_tree = GridSearchCV(dtree, parameters,cv=5,refit=True,return_train_score=True)

# 과적합 되었는지 확인하는 방법은 학습하지 않은 새로운 데이터를 넣는 것인데, 
#지금 grid에 0.2로 학습 안시킨 test 데이터 넣어서 최적을 찾은거라서 정확도가 과적합 안 되었다는 것을 어느정도 보장할 수 있음

# fit으로 넣기
grid_tree.fit(x_train,y_train)

# 확인
score_df = pd.DataFrame(grid_tree.cv_results_)
score_df[['params','mean_test_score','rank_test_score']]

print('최적 파라미터',grid_tree.best_params_)
print('최적 정확도',grid_tree.best_score_)

y_pred = grid_tree.predict(x_test)
accuracy_score(y_pred,y_test)

tree_model = grid_tree.best_estimator_
accuracy_score(tree_model.predict(x_test),y_test)
# 여기서 tree_model을 피클 저장

tree_model.feature_importances_

import pandas as pd

titanic = pd.read_csv('c:/data/titanic.csv')

# 종속변수, 목표변수, 결과변수, 정답레이블
titanic.survived.unique()
0 = 사망, 1 = 생존

# 독립변수, 설명변수, 입력변수, 티켓의 등급
titanic.pclass.unique()

# 성별
titanic.gender.unique() # 문자형임. 수치형으로 바꿔서 넣어줘야 함

# 나이
titanic.age.describe()

# 함께 탑승한 형제 자매, 배우자 수
titanic.sibsp.describe()

# 함께 탑승한 부모님, 자식 수
titanic.parch.describe()

# 티켓 번호
titanic.ticket

# 운임
titanic.fare 

# 객실번호
titanic.cabin

# 탑승항구
titanic.embarked.unique()
S = Southampton, Q = Queenstown, C = Cherbourg


# 수치형으로 바꾸기
titanic.gender.map({'female':0,'male':1})

from sklearn.preprocessing import LabelEncoder
gender_le = LabelEncoder()
titanic.gender = gender_le.fit_transform(titanic['gender'])
Counter(titanic.gender)

# 열별로 null 값 확인
titanic.isnull().sum()

# 특정 열의 null값 확인
titanic.age.isnull().sum()

# age 중앙값
titanic.age.describe()
titanic.age.median()

# age null값을 중앙값으로 수정
titanic.age.fillna(titanic.age.median(),inplace=True)


Counter(titanic.embarked)

# one-hot-encoding

     C  Q  S
0    0  0  1
1    1  0  0
2    0  0  1

pd.get_dummies(titanic.embarked,prefix='embarked')

     embarked_C  embarked_Q  embarked_S
0             0           0           1
1             1           0           0
2             0           0           1
3             0           0           1
4             0           0           1

embarked_dummies = pd.get_dummies(titanic.embarked,prefix='embarked')

# 원래 DF에 원핫인코딩한거 추가하기
titanic = pd.concat([titanic, embarked_dummies],axis=1)
titanic.info()

feature_col = ['pclass','gender','age','embarked_S','embarked_Q']
x = titanic[feature_col]
y = titanic.survived
x
y

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x,y)
model.classes_
model.feature_importances_
pd.DataFrame({'features':feature_col, 
              'importances':model.feature_importances_})



[문제204] 유방암 데이터를 의사결정나무를 이용해서 분류해주세요. # knn, k-means 다 해보자

wisc = pd.read_csv('c:/data/wisc_bc_data.csv')
wisc.info()
wisc.diagnosis = wisc.diagnosis.map({'B':'Benign','M':'Malignant'})

wisc

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(wisc.iloc[:,2:],wisc.diagnosis,test_size=0.2)

from collections import Counter
Counter(y_train)
Counter(y_test)

wisc_model = DecisionTreeClassifier()
wisc_model.fit(x_train,y_train)
wisc_model.classes_

y_pred = wisc_model.predict(x_test)
(y_pred == y_test).mean()

# 만약 train은 정확도가 높은데 test는 정확도가 낮으면 과적합이야
wisc_model.score(x_train,y_train)
wisc_model.score(x_test,y_test)

wisc_model.feature_importances_ # 결정적인 독립변수가 뭐였는지 확인

import graphviz
dot_data = export_graphviz(wisc_model,out_file=None,
                           feature_names=wisc.columns[2:],
                           class_names=wisc_model.classes_,
                           filled=True,rounded=True, special_characters=True)
graphviz.Source(dot_data)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# 선언
dtree = DecisionTreeClassifier()
parameters = {'max_depth':[5,6,7,8,9],'min_samples_split':[3,4,5,6,7,8,9,10]}
grid_tree = GridSearchCV(dtree, parameters,cv=5,refit=True,return_train_score=True)

# 과적합 되었는지 확인하는 방법은 학습하지 않은 새로운 데이터를 넣는 것인데, 
#지금 grid에 0.2로 학습 안시킨 test 데이터 넣어서 최적을 찾은거라서 정확도가 과적합 안 되었다는 것을 어느정도 보장할 수 있음

# fit으로 넣기
grid_tree.fit(x_train,y_train)

# 확인
score_df = pd.DataFrame(grid_tree.cv_results_)
score_df[['params','mean_test_score','rank_test_score']]

print('최적 파라미터',grid_tree.best_params_)
print('최적 정확도',grid_tree.best_score_)

y_pred = grid_tree.predict(x_test)
accuracy_score(y_pred,y_test)

tree_model = grid_tree.best_estimator_
accuracy_score(tree_model.predict(x_test),y_test)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# 여기서 tree_model을 피클 저장
dot_data = export_graphviz(tree_model,out_file=None,
                           feature_names=wisc.columns[2:],
                           class_names=wisc_model.classes_,
                           filled=True,rounded=True, special_characters=True)
graphviz.Source(dot_data)

tree_model.feature_importances_

■ RandomForest
- decision tree 와 bagging을 결합한 알고리즘
- 매 실행시 마다 랜덤하게 관측치와 변수(feature)를 선택하므로 실행결과가 조금씩 달라지게 된다.
- 학습데이터에서 중복을(복원추출) 허용하여 랜덤샘플링을 통해서 의사결정나무모델을 구축
    학습데이터 수 만큼 샘플링한다(bagging - bootstrap aggregating)
- 의사결정나무 모델 구축시 변수도 무작위 선택한다. 단, 하나의 모델 안에서 변수는 중복불가능
- 의사결정나무 모델 결과의 투표를 통해서 클래스를 선택한다.
- 앙상블 기법 중 하나다.

######### 랜덤 포레스트 구조 ####################################################
iris = pd.read_csv('c:/data/iris.csv')
iris.info()
iris.sample(n=150,replace=True) # 복원추출로 선택. 중복O, 선택되지 않은 애들은 검증데이터로

# feature 뽑기
import random

col_idx = []
for i in range(2):
    idx = random.randint(0,3)
    while idx in col_idx:
        idx = random.randint(0,3)
    col_idx.append(idx)

col_idx

data = iris.sample(n=150,replace=True)
x_train = data[data.columns[col_idx]]
y_train = data.Name

iris_model = DecisionTreeClassifier()
iris_model.fit(x_train,y_train)
iris_model.score(x_train,y_train)


iris_result = pd.DataFrame({'feature': x_train.columns, 'importance': iris_model.feature_importances_})
iris_result.sort_values(by='importance',ascending=False)

pred_data = [5.1,3.5,1.4,0.2]
pred_data[col_idx[0]]
pred_data[col_idx[1]]

q = []
q.append(iris_model.predict([[pred_data[col_idx[0]],pred_data[col_idx[1]]]])[0])
q

Counter(q).most_common(1)[0][0]
######################################################################################

wisc = pd.read_csv('c:/data/wisc_bc_data.csv')
wisc.info()
wisc.diagnosis = wisc.diagnosis.map({'B':'Benign','M':'Malignant'})

x_train,x_test,y_train,y_test = train_test_split(wisc.iloc[:,2:],wisc.diagnosis,test_size=0.2)
Counter(y_train)
Counter(y_test)

from sklearn.ensemble import RandomForestClassifier

# oob = out-of-bag 부트스트랩 샘플링 시 선택되지 않은 샘플
# n_estimators=100 100개의 모델 만든다
# 샘플 개수는 중복을 허용하면서 데이터 개수 전체로 뽑음
model_rf = RandomForestClassifier(n_estimators=1000,oob_score=True) # oob_score=True - 이용하지 않은 데이터를 검증 데이터처럼 쓰겠다.
model_rf.fit(x_train,y_train)
model_rf.score(x_train,y_train)
model_rf.score(x_test,y_test)
model_rf.oob_score_

confusion_matrix(y_test,model_rf.predict(x_test))

# rf도 GridSearchCV 해보기
rftree = RandomForestClassifier()
parameters = {'max_depth':[5,6,7,8,9],
              'min_samples_split':[3,4,5,6,7,8,9,10],
              'max_features':[5,6,7,8,9,10],
              'n_estimators':[100,200,300,400,500]}

grid_tree = GridSearchCV(rftree, parameters,cv=5,refit=True,return_train_score=True,n_jobs=-1)
# fit으로 넣기
grid_tree.fit(x_train,y_train)

# 확인
score_df = pd.DataFrame(grid_tree.cv_results_)
score_df[['params','mean_test_score','rank_test_score']]

print('최적 파라미터',grid_tree.best_params_)
print('최적 정확도',grid_tree.best_score_)

y_pred = grid_tree.predict(x_test)
accuracy_score(y_pred,y_test)

tree_model = grid_tree.best_estimator_
accuracy_score(tree_model.predict(x_test),y_test)

■ 분산(variance)
- 내가 가진 자료(데이터)가 평균값을 중심으로 퍼져있는 평균적인 거리(정도)



■ 공분산(covariance)
- 두 변수가 얼마나 함께 변하는지를 측정하는 지표
- cov(x,y) > 0 : x와 y의 변화가 같은 방향으로 변화
- cov(x,y) < 0 : x와 y의 변화가 다른 방향으로 변화
 - cov(x,y) = 0 : x와 y의 값이 서로 상관없이 움직인다

공분산(x,y) = 합((x-x평균)*(y-y평균)) / 관측값의수 - 1


 
 
■ 상관분석(correlation)
- 공분산을 표준화하는 방법
- 변수들 간의 연관성 파악
    - 피어슨 상관계수, 스피어만 상관계수, 겐달 순위 상관계수
    - 상관계수는 -1 <= r <= 1



상관분석(x,y) = 공분산(x,y) / x표준편차 * y표준편차 = 상관계수


x = [184,170,180]
y = [85,70,82]

import numpy as np

np.mean(x)
np.mean(y)
np.var(x) # 모집단 분산
np.var(y)
np.std(x) # 모집단 표준편차

(((184-178) * (85-79)) + ((170-178)*(70-79)) + ((180-178)*(82-79))) / 2

np.cov(x,y)[0][1] # 공분산

(((184-178) * (85-79)) + ((170-178)*(70-79)) + ((180-178)*(82-79))) / 2    /   (np.std(x)*(np.std(y)))

np.corrcoef(x,y)[0][1]  # 상관계수

import pandas as pd

df = pd.concat([pd.DataFrame(x),pd.DataFrame(y)],axis=1)
df.columns = ['x','y']
df['x'].var() # 표본 분산

df['x'].cov(df['y'])
df['x'].corr(df['y'])


■ 회귀분석(regression)
- 인과관계를 분석하는 방법
    어떤 변수가 어떤 변수에게 어떤 영향을 주는지를 판단(원인, 결과)
- 주어진 데이터의 독립변수(원인)로 종속변수(결과)를 예측

독립변수, 설명변수, 입력변수, 원인

종속변수, 목표변수, 결과변수

인과관계의 조건
1. x가 변화될 때 y도 변한다.
    교육연수 -> 생활만족도
2. 시간적으로 선행되어야 한다.
    독립변수가 종속변수보다 시간적으로 앞선다.
3. 외생변수를 통제해야 한다.
    다른 요인을 통제하고 인과관계 분석


회귀분석 시
1. 산점도를 그려본다
    x 가 커지면 y도 커진다. 선형
    x 가 작아지면 y는 커진다. 선형
    x 가 커지면서 y가 커지다가 작아진다. 비선형
    
2. 모델의 선을 그려본다. 추세선을 예측하는 것이 회귀분석의 목적
    -최소제곱법을 이용해서 선을 그린다. (오차의 제곱의 합을 최소로 만드는 방법)
    -이 직선은 평균을 지난다.(평균으로 회귀)
    y = ax + b: a(회귀계수, 기울기)
    
기울기 = y 증가분 / x 증가분
기울기 = 공분산(x,y) / x 분산


b(절편) = 평균(y) - a * 평균(x)

height = [176,172,182,160,163,165,168,163,182,182]
weight = [72,70,70,43,48,54,51,52,73,88]


height = 185 일 때 몸무게 예측

# 표본 분산을 구해야하기 때문에 모집단 거를 구하는 np.var는 이용하면 안돼

df = pd.DataFrame({'height':height,'weight':weight})
df

a = df['height'].cov(df['weight']) / df['height'].var()

b = df['weight'].mean() -a*df['height'].mean()

b
w = a*185 + b


from scipy import stats
slope, intercept, r_value, p_value, stderr = stats.linregress(height,weight)

print('기울기',slope)
print('절편',intercept)

import matplotlib.pyplot as plt

plt.scatter(df['height'],df['weight'])
plt.plot(df['height'],df['height']*slope+intercept,c='red')

[문제205] IQ 130일 때 시험성적 예측해주세요.

score = pd.read_csv('c:/data/score.txt',encoding='euc-kr')
score

# 선형 상관관계 맞는지 먼저 확인
plt.scatter(score['IQ'],score['성적'])
# 독립변수가 한 개일 경우 단순회귀분석
slope,intercept,r_value,p_value,stderr = stats.linregress(score['IQ'],score['성적'])
slope*130 + intercept

# p_value값이 0.05보다 작으면 slope값은 의미가 있는 값이다.

plt.scatter(score['IQ'],score['성적'])
plt.plot(score['IQ'],score['IQ']*slope+intercept,c='red')



■ 회귀분석(단순, 다중) 모델

from sklearn.linear_model import LinearRegression
# 모델에 넣으려면 2차원으로 바꿔야한다.
lr = LinearRegression()
# .values 하면 array로 바꿔주고 array에서 reshape 해야한다.
lr.fit(score['IQ'].values.reshape(-1,1),score['성적'])

score['IQ'].values.reshape(-1,1)

print('기울기', lr.coef_)
print('절변',lr.intercept_)

# 예측할 때도 값 넣을때 2차원으로! 
lr.predict([[130]])

# 독립변수가 2개 이상일 때 다중회귀분석이라고 한다.

score.iloc[:,2:]
# 변수 2개이상이면 바로 2차원 되니까 reshape필요 없음
mul_lr = LinearRegression()
mul_lr.fit(score.iloc[:,2:],score['성적'])

print('기울기', mul_lr.coef_)
print('절변',mul_lr.intercept_)

y = a1x1+a2x2+a3x3 + b

mul_lr.predict([[120,3,5,2]])[0]

[문제206] 보험청구액을 예측해주세요.

ins = pd.read_csv('c:/data/insurance.csv')
ins

from sklearn.preprocessing import OneHotEncoder
onehot_en = OneHotEncoder()
gender_onehot = onehot_en.fit_transform(ins[['gender']])
gender_onehot
gender_onehot.toarray()

ins.info()
ins.head()

dummy_gender = pd.get_dummies(ins['gender'],prefix='gender')
dummy_gender

dummy_smoker = pd.get_dummies(ins['smoker'],prefix='smoker')
dummy_smoker


# index끼리 연결할 때 .join
data1 = dummy_gender.join(dummy_smoker)
data2 = data1.join(ins[['age','bmi','children','charges']])
data2

# 1,0,1,0 이걸 다 넣으면 안돼 gender_female or gender_male 하나, smoker_yes,smoker_no 중 하나만 넣어야돼. (dummy trap에 빠지지않기 위해)
# 만약 상,중,하 세개라면 그 중 2개만 넣어야해

col=['age','bmi','children','gender_male','smoker_yes']
data2[col]
data2['charges']

lr = LinearRegression()
lr.fit(data2[col],data2['charges'])

print('기울기', lr.coef_)
print('절편', lr.intercept_)

gender_male의 보험청구금액이 gender_female보다 -128.63985357

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(data2[col],data2['charges'],test_size=0.2)

lr = LinearRegression()
lr.fit(x_train,y_train)
lr.score(x_train,y_train)
lr.score(x_test,y_test)

# 각 독립변수가 유의미한지(p_value < 0.05) 확인하는 법
from statsmodels.formula.api import ols

ols_lr = ols('charges ~ age + bmi + children + gender_male + smoker_yes', data=data2).fit() 
ols_lr.summary()

# gender_male -> p = 0.700 (의미 없을 수도 있다. 무조건 빼는 건 아님)
# R-squared: 설명력(1에 가까우면 설명력이 좋다.)
# std err : 표준오차 작을수록 좋다. 이게 크면 선에서 관측치들이 멀리 떨어져 있다.


■ 회귀평가
회귀의 평가를 위한 지표는 실제값과 회귀예측값의 차이값을 기반으로 하는 지표

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error,r2_score

y_test = np.array([1,2,1]) # 실제값
y_pred = np.array([1,2,2]) # 예측값

MAE(Mean Absolute Error)
실제값과 예측값의 차이를 절대값으로 변환해 평균한 값
(abs(1-1)+abs(2-2)+abs(1-2)) / 3
mean_absolute_error(y_test,y_pred)

MSE(Mean Squared Error)
실제값과 예측값의 차이를 제곱해 평균한 값
((1-1)**2 + (2-2)**2 + (1-2)**2) /3
mean_squared_error(y_test,y_pred)

MSLE(Mean Squared Log Error)
mean_squared_log_error(y_test,y_pred)

R2(R squared)
R2 분산기반으로 예측성능을 평가한다. 1에 가까울수록 예측정확도가 높다.

r2_score(y_test,y_pred)






MAE
MSE
RMSE
MSLE
RMSLE
R²





[문제207] boston 데이터를 이용해서 보스턴 주택 가격 예측을 해보세요.
CRIM: 도시의 범죄율
INDUS: 비소매상업지역 면적 비율
NOX: 일산화질소 농도
RM: 주택당 방 수
LSTAT: 인구 중 하위 계층 비율
B: 인구 중 흑인 비율
PTRATIO: 학생/교사 비율
ZN: 25,000 평방피트를 초과 거주지역 비율
CHAS: 찰스강의 경계에 위치한 경우는 1, 아니면 0
AGE: 1940년 이전에 건축된 주택의 비율
RAD: 방사형 고속도로까지의 거리
DIS: 직업센터의 거리
TAX: 재산세율
MEDV: 주택 가격 중앙값 (단위 1,000 달러)

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 변수 상관관계 미리 파악하기
boston = pd.read_csv('c:/data/boston.csv')
boston.head()

boston.isnull().sum()

sns.pairplot(boston,height=2.5)

cols = ['CRIM','RM','LSTAT','CHAS','PTRATIO','MEDV']
sns.pairplot(boston[cols],height=2.5)
sns.pairplot(boston, vars=['RM','MEDV'])

# 변수 간의 상관계수
corr_matrix = boston.corr().round(2)
sns.heatmap(data = corr_matrix, annot=True)

# 종속변수 시각화
sns.distplot(boston['MEDV'],bins=30)

X = boston[boston.columns.difference(['MEDV'])]
Y = boston['MEDV']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
x_train.shape
y_train.shape

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

print('기울기',lr.coef_)
print('절편',lr.intercept_)

y_pred = lr.predict(x_test)

np.sqrt(mean_squared_error(y_test, y_pred))
r2_score(y_test,y_pred)


from statsmodels.fomula.api import ols

col = boston[boston.columns.difference(['MEDV'])].columns
'+'.join(col)
ols_lr = ols('MEDV ~ '+' + '.join(col),data=boston).fit()
ols_lr.summary()

# 전진선택법, 후진선택법

■ 모형의 선형성
- 예측값(predicted)과 잔차(residual)
- 모든 예측값에서 잔차가 비슷하게 있어야 한다.
- 빨간 실선은 잔차의 추세선
- 빨간 실선이 점선에서 크게 벗어나면 예측값에 따라 잔차가 크게 달라진다는 것을 의미



x = boston[boston.columns.difference(['MEDV'])]
y = boston['MEDV']

predicted = ols_lr.predict(x)
residual = y - predicted

import seaborn as sns

sns.regplot(predicted,residual,line_kws={'color':'red'})
plt.plot([predicted.min(),predicted.max()],[0,0],'--',color='grey')

■ 잔차의 정규성
- 잔차가 정규분포를 따른다는 가정
- Q-Q plot으로 확인할 수 있다.(Quantile-Quantile Plot)
- 잔차가 정규분포를 띄면 Q-Q plot 에서 점들이 점선을 따라 배치되어 있어야 한다.

import scipy.stats as ss
z = ss.zscore(residual)
ss.probplot(z,plot=plt) # Q-Q plot

sns.distplot(residual)
sns.histplot(residual)

import numpy as np

■ 잔차의 등분산성
- 회귀모형을 통해 예측된 값이 크든, 작든 모든 값들에 대하여 잔차의 분산이 동일한 것으로 가정
- 빨간색 실선이 수평선을 그리는 것이 가장 이상적
- 패턴을 보이지 않는 것이 좋음

sns.regplot(predicted,np.sqrt(np.abs(z)),line_kws={'color':'red'})


■ 잔차의 독립성
- 자료 수집 과정에서 무작위 표본을 이용해서 모델을 학습했다면 잔차의 독립성은 만족하는 것으로 본다.
- Durbin-Watson 검정을 이용해서 확인
- 보통 1.5 ~ 2.5 사이이면 독립으로 판단하고 회귀모형이 적합하다고 판단하면 된다.
- 0(양의 자기상관) 또는 4(음의 자기상관)에 가깝다는 것은 잔차들이 자기 상관을 갖는다는 뜻
    이는 t값,F값,R2 값을 실제보다 증가시켜 유의미하지 않은 결과를 유의미한 것으로 왜곡

■ 극단값 
- cook's distance는 극단값을 나타내는 지표
- 데이터가 특히 예측에서 많이 벗어남을 알 수 있다.
- 모델 학습시 학습데이터에서 극단값 제거




from statsmodels.stats.outliers_influence import OLSInfluence
cd, _ = OLSInfluence(ols_lr).cooks_distance
cd.sort_values(ascending=False).head()

ols_lr.predict(boston.iloc[368,])
# 예측값 368  =  23.800
# MEDV          50.00000


R2(R-squared)
종속변수의 분산을 독립변수의 분산으로 설명하는 지표
R2 = 0 : 모델의 설명력이 0
R2 = 1 : 모델의 설명력이 100%

R2 = 1 - SSR/SST


■ 다중공선성(multicollinearity)
-독립변수들 끼리 상관관계를 가지고 있다.

예시)
혈압(종속변수), 키, 몸무게, BMI,,,,(독립변수 지들끼리 영향)
...
...

■ VIF를 통한 다중공선성 확인
- VIF = Variance inflation Factor
- 분산팽창요인
-VIF가 10보다 크면 다중공선성이 있다고 판단
- ONE HOT encoding 의 변수는 VIF가 3이상이면 다중공선성이 있다고 판단

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['VIF Factor'] = [variance_inflation_factor(boston.values,i) for i in range(boston.shape[1])]
vif['features'] = boston.columns
vif

data = boston[boston.columns.difference(['NOX','PTRATIO','RM','TAX'])]

data_vif = pd.DataFrame()
data_vif['VIF Factor'] = [variance_inflation_factor(data.values,i) for i in range(data.shape[1])]
data_vif['features'] = data.columns
data_vif

########## 그런데ㅡ 다중공선성 발생한다고 해서 무조건 빼면 예측에 영향을 많이 주는 경우 발생할 수 있음

corr_matrix = boston.corr().round(2)
sns.heatmap(data = corr_matrix, annot=True)
VIF 말고 heatmap으로 판단 가능



■ Logistic Regression
- 선형회귀방식을 분류에 적용한 알고리즘
- 분류를 하는데 있어서 가장 흔한 경우는 이분법을 기준으로 분류하는 경우, 이진분류(0,1)
- 기업부도예측, 주가, 환율, 금리 등의 up/down 예측

오즈비(odds ratio)
- 오즈비는 확률과 관련된 의미로 P가 주어졌을 때 사건이 발생할 확률(성공확률)이 발생하지 않을 확률(실패확률)에 비해 
몇 배 더 높은가의 의미
- 예)
    종속변수의 범주가 1(성공),0(실패)인 이분형을 가정할 때 P가 0.8이라면
    오즈비는 (0.8/(2-0.8)) = 4가 되고 이 값은 성공이 될 확률이 실패가 될 확률보다 4배 높다는 의미
    
    odds = P/ 1-P
    
    
    logit = log(odds)

■ sigmoid (시그모이드) 함수
- 0 ~ 1 사이의 실수값으로 리턴하는 함수가 필요하다.
- e 자연상수 : 2.7182

np.exp(-10)
np.exp(10)

sigmoid(x) = 1 / 1 + np.exp(-x)

def sigmoid(x):
    return 1 / (1+np.exp(-x))


sigmoid(1000)
sigmoid(12) # 어떤 값이든 0 혹은 1에 매우 근접

x = np.arange(-5,5,0.1)
y = sigmoid(x)
plt.plot(x,y)

sigmoid(y(예측)) = ax + b
예측값이 0 ~ 1 값만 출력할 수 있도록 시그모이드 함수를 이용한다

sigmoid(y(예측)) >= 0.5  -> 1
sigmoid(y(예측)) < 0.5  ->  0

from sklearn.linear_model import LogisticRegression

iris = pd.read_csv('c:/data/iris.csv')
x = iris.iloc[:,:-1]
y = iris.Name

log_lr = LogisticRegression()
log_lr.fit(x,y)

log_lr.classes_

log_lr.predict([[5.1,3.5,1.4,0.2]])
log_lr.predict([[6.9,3.2,5.4,2.2]])

# 나이브베이즈 한 거 로지스틱에도 넣어보자

import seaborn as sns

sns.stripplot(x='PetalLength',y='Name',data=iris)
sns.stripplot(x='SepalLength',y='Name',data=iris)

sns.swarmplot(x='PetalLength',y='Name',data=iris)


import pandas as pd

titanic = pd.read_csv('c:/data/titanic.csv')
gender_dummy = pd.get_dummies(titanic.gender,prefix='gender')
data = pd.concat([titanic,gender_dummy],axis=1)

import statsmodels.api as sm

logit = sm.Logit.from_formula('survived ~ age + fare + sibsp + gender_male ',data = data).fit()
logit.summary()

import numpy as np

(np.exp(0.0159) -1) * 100
요금이 1$ 증가할 때 생존률은 1.6% 증가한다

(np.exp(-0.0190) -1) * 100
나이가 1살 증가할 때 생존 가능성은 1.9 감소한다

# gender_male
역수 취해준다( 기준이 여성이니까 !)
((1/np.exp(-2.4413)) -1) * 100
남성에 비해 여성의 생존률은 1048% 높다

■ 규제선형회귀(Regularized Linear Regression)
회귀분석에서 과적합하게 되면 회귀계수가 크게 설정되며 테스트데이터에 대해서 예측성능이 좋지않게 된다.
이를 해결하기 위해 하나의 방법 - 회귀계수가 기하급수적으로 커지는 것을 제어한다.

최적모델을 위한 cost = 잔차(residual) 최소화 + 회귀계수 크기 

cost(비용) 목표 = 최소화(RSS(Residual Sum of Square)+ alpha*(회귀계수)²)
- alpha는 학습데이터 적합 정도와 회귀계수값의 크기를 제어하는 파라미터
- alpha가 0이면, 비용함수 최소화만
- alpha가 무한이면, RSS에 비해 alpha*(회귀계수)² 값이 커지므로 회귀계수값을 작게 만들어야 비용함수 목표에 도달할 수 있다.
- alpha 값 감소하면 RSS 최소화하는 방향으로 가고
- alpha 값을 증가하면 회귀계수를 감소하는 방향으로 간다.
- alpha 값으로 패널티를 부여해 회귀계수 값의 크기를 감소시켜 과적합을 개선하는 방식이 규제

- L2 규제 alpha*(회귀계수)² 이 방식을 적용한 회귀를 Ridge라고 함
- L1 규제 alpha*|회귀계수| 이 방식을 적용한 회귀를 Lasso라고 함
    L1 규제는 영향력이 크지 않은 회귀계수값은 0으로 변환한다. 그렇다보니 불필요한 feature들은 제거하고 적절한 피쳐만 회귀에 포함
    
- ElasticNet 회귀는 L1,L2규제를 함계 결합한 모델

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

x_train = boston.iloc[:,:-1]
y_train = boston.MEDV

lr = LinearRegression()
lr.fit(x_train,y_train)
lr.coef_
lr.intercept_
lr.score(x_train,y_train)

y_predict = lr.predict(x_train)
rmse = np.sqrt(mean_squared_error(y_train,y_predict))
rmse
r2_score(y_train,y_test)

from sklearn.linear_model import Ridge

ridge = Ridge(alpha=10)
ridge.fit(x_train,y_train)

ridge.score(x_train,y_train)

# 사용된 피쳐 수 
np.sum(ridge.coef_ !=0)

np.sum(lasso.coef_ !=0)

elasticnet도 해보기

from sklearn.model_selection import GridSearchCV

ridge = Ridge()
alphas = [0,0.1,1,10,100]
parameters = {'alpha':alphas}

ridge_cv = GridSearchCV(estimator=ridge,param_grid=parameters,scoring='r2',cv=5)
ridge_cv.fit(x_train,y_train)
ridge_cv.best_params_
ridge_cv.best_score_

# 잔차는 작을수록 좋으니까 neg를 꼭 붙여줘야 해
ridge_cv = GridSearchCV(estimator=ridge,param_grid=parameters,scoring='neg_mean_squared_error',cv=5)
ridge_cv.fit(x_train,y_train)
ridge_cv.best_params_
ridge_cv.best_score_

■ 퍼셉트론(perceptron)
- 1957년에 프랑크 로젠브라트가 고안했다
- 사람의 뇌의 동작을 전기신호 on/off로 흉내낼 수 있다는 이론
- 특정한 자극이 있다면 그 자극이 어느 thresholde(역치, 임계값) 이상이어야지만 세포가 반응한다
- 다수의 신호를 입력받아 하나의 신호를 출력한다
- 0 : 신호가 없다
- 1 : 신호가 있다

회귀식
y = ax + b

퍼셉트론 동작
y = wx
x : 입력값(입력신호)
w : weight(가중치)
Θ : theta(임계값)
y : 출력

y : 0 w1*x1 + w2*x2 <= Θ
y : 1 w1*x1 + w2*x2 > Θ

논리회로
컴퓨터는 두가지 디지털 0, 1을 입력해서 하나의 값을 출력하는 회로가 모여 만들어지는데 이 회로를 gate라고 한다.

AND gate

x1      x2       y
0        0       0
0        1       0
1        0       0  
1        1       1

AND(0,0) => 0
AND(0,1) => 0
AND(1,0) => 0
AND(1,1) => 1

y : 0 w1*x1 + w2*x2 <= Θ
y : 1 w1*x1 + w2*x2 > Θ

AND(0,0) => 0
    w1*0 + w2*0 <= Θ

w1 = ?
w2 = ?
Θ = ?

def AND(arg1,arg2):
    w1 = 0.5
    w2 = 0.5
    theta = 0.5
    tmp = w1*arg1 + w2*arg2
    if tmp <= theta:
        return 0
    else:
        return 1

AND(0,0)
AND(0,1)
AND(1,0)
AND(1,1)

import numpy as np

input = np.array([[0,0],[0,1],[1,0],[1,1]])
input

def AND(arg1,arg2):
    w1 = 0.5
    w2 = 0.5
    theta = 0.5
    w = np.array([w1,w2])
    x = np.array([arg1,arg2])
    tmp = np.sum(w*x)
    if tmp <= theta:
        return 0
    else:
        return 1

print("AND perceptron")
for i in input:
    print(str(i)+' => '+str(AND(i[0],i[1])))

OR gate

x1      x2       y
0        0       0
0        1       1
1        0       1  
1        1       1

OR(0,0) => 0
OR(0,1) => 1
OR(1,0) => 1
OR(1,1) => 1

input = np.array([[0,0],[0,1],[1,0],[1,1]])
input

def OR(arg1,arg2):
    w1 = 0.5
    w2 = 0.5
    theta = 0
    w = np.array([w1,w2])
    x = np.array([arg1,arg2])
    tmp = np.sum(w*x)
    if tmp <= theta:
        return 0
    else:
        return 1

print("OR perceptron")
for i in input:
    print(str(i)+' => '+str(OR(i[0],i[1])))


NAND(NOT AND) gate

x1      x2       y
0        0       1
0        1       1
1        0       1  
1        1       0

NAND(0,0) => 1
NAND(0,1) => 1
NAND(1,0) => 1
NAND(1,1) => 0

input = np.array([[0,0],[0,1],[1,0],[1,1]])
input

def NAND(arg1,arg2):
    w1 = -0.5
    w2 = -0.5
    theta = -0.7
    w = np.array([w1,w2])
    x = np.array([arg1,arg2])
    tmp = np.sum(w*x)
    if tmp <= theta:
        return 0
    else:
        return 1

print("NAND perceptron")
for i in input:
    print(str(i)+' => '+str(NAND(i[0],i[1])))
    
    
XOR gate

x1      x2       y
0        0       0
0        1       1
1        0       1  
1        1       0

XOR(0,0) => 0
XOR(0,1) => 1
XOR(1,0) => 1
XOR(1,1) => 0

input = np.array([[0,0],[0,1],[1,0],[1,1]])
input

def XOR(arg1,arg2):
    w1 = 0.5
    w2 = 0.5
    theta = 0
    w = np.array([w1,w2])
    x = np.array([arg1,arg2])
    tmp = np.sum(w*x)
    if tmp <= theta:
        return 0
    else:
        return 1

print("XOR perceptron")
for i in input:
    print(str(i)+' => '+str(XOR(i[0],i[1])))
    
    
■ 단층퍼셉트론의 한계
- 직선하나로 XOR gate의 출력을 구분할 수 없다.
- 퍼셉트론(단층퍼셉트론)은 직선하나로 나눈 영역만 표현할 수 있다는 한계 # XOR은 직선으로 나눌 수 없으니까
- 1969 민스키가 기존 퍼셉트론의 문제점을 지적했는데, XOR 분류 못하는 문제점 지적

XOR gate

x1      x2    OR층  NAND층     AND(OR,NAND) => y
0        0     0      1               0
0        1     1      1               1
1        0     1      1               1  
1        1     1      0               0


■ 다층퍼셉트론(Multi Layer perceptron)
다층퍼셉트론(OR,NAND)로 AND 연산작업을 하면 XOR를 만들 수 있다(1986)
1. x1, x2를 통해 OR층 만든다.
2. x1, x2를 통해 NAND 층 만든다.
3. 중간에 만든 OR층, NAND층을 AND 계산작업을 하면 XOR를 만들 수 있다.

XOR gate

x1      x2    OR층  NAND층     AND(OR,NAND) => y
0        0     0      1               0
0        1     1      1               1
1        0     1      1               1  
1        1     1      0               0


def XOR(arg1,arg2):
    s1 = OR(arg1,arg2)
    s2 = NAND(arg1,arg2)
    s3 = AND(s1,s2)
    return s3

input = np.array([[0,0],[0,1],[1,0],[1,1]])
input

print("XOR Multi Layer perceptron")
for i in input:
    print(str(i)+' => '+str(XOR(i[0],i[1])))
    

■ 딥러닝(Deep Neural Network)
- 생각할 수 있도록 고안된 인공지능
- 사람의 뇌세포를 프로그램으로 표현한 기술
- 인간의 신경망(neural network)이론을 이용한 인공신경망(ANN, Artificial Neural Network)의 일종으로
계층구조로 구성하면서 입력층(input layer)과 출력층(output layer) 사이에 하나 이상의 은닉층(hidden layer)을 가지고 있는
심층신경망이다.(Deep Neural Network)

- 지도학습

         weight
입력값(x) -------- sum() ---------> 출력값(y)
         bias


y(출력) = weight * x(입력값) + bias 
y(출력) = weight(회귀계수) * x(입력) + bias(절편)

■ Activation Function(활성화함수)
- synapse는 전달된 전기신호가 최소한의 자극값을 초과하면 활성화되어 다음 뉴런으로 전기신호를 전달
- 활성화 함수는 synapse를 모방하여 값이 작을 때는 출력값을 작은값으로 막고
일정한값을 초과하면 급격히 커지는 함수를 이용한다.

■ 은닉층에서 사용하는 활성화 함수
1. 계단함수(step Function)
- 입력값이 0을 초과하면 1을 출력하고 그외에는 0을 출력하는 함수

def step_function(arg):
    if arg > 0:
        return 1
    else:
        return 0

step_function(100)

def step_function(arg):
    return [1 if i > 0 else 0 for i in arg]
            
step_function(np.array([1,100,-200]))

x = np.array([1,100,-200])
y = x > 0
y
#############
bool -> int 변환 : True -> 1, False -> 0
y.astype(np.int64)
np.sum(y)

def step_function(arg):
    y = np.array(arg) > 0
    return y.astype(np.int32)

step_function(np.array([100]))
step_function(np.array([1,100,-200]))

import matplotlib.pylab as plt

x = np.arange(-5,5,0.1)
y = step_function(x)
plt.plot(x,y)

2. sigmoid
- 0과 1 사이의 실수값으로 전달하는 함수
- e 자연상수 : 2.7182.....
- 요즘 트렌드는 시그모이드 함수를 은닉층에서 사용 X
이유는 우리가 구하고자 하는 weight값과 bias을 구하기 위해
미분을 이용하다보면 sigmoid값이 없어지는 현상이 발생한다.
vanishing gradient(gradient 값 손실)

- 출력층에서는 이진분류를 수행할 때 사용한다.

x = 10
np.exp(-x)

sigmoid = 1/(1+np.exp(-x))

def sigmoid(arg):
    return 1/ (1+np.exp(-arg))


sigmoid(100)
sigmoid(-100)

x = np.arange(-5,5,0.1)
y = sigmoid(x)
plt.plot(x,y)


# Relu를 이용한다!!!!!!!!!!!!!!!!!
3. Relu(Rectified Linear Unit)
- 입력값이 0을 넘으면 그 입력값으로 출력하고 0이하면 0을 출력한다


def relu(arg):
    if arg > 0:
        return arg
    else:
        return 0
    
relu(100)
relu(-1000)

def relu(arg):
    return np.maximum(arg,0)

relu(np.array([1,-10,2]))

x = np.arange(-5,5,0.1)
y = relu(x)
plt.plot(x,y)


x1 = np.array([[1,2],[3,4]])
x2 = np.array([[5,6],[7,8]])
x3 = np.array([[1,2,3],[4,5,6]])
x4 = np.array([[5,6],[7,8],[9,10]])
x5 = np.array([1,2])

x1.shape
x2.shape


np.dot(x1,x2)

x3.shape
x4.shape

[문제] 그림을 보고 행렬의 곱으로 표현해보세요.

x = np.array([1,2])
weight = np.array([[1,3,5],[2,4,6]])
np.dot(x,weight)
relu(np.dot(x,weight))


x = np.array([1,2])
weight = np.array([[1,3,5],[2,4,6]])
bias = np.array([0.2,0.5,0.7])
np.dot(x,weight) + bias
relu(np.dot(x,weight)+bias)

[문제] 3층 신경망을 구현해 보세요.

x = np.array([0.1,0.5])
weight1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
bias1 = np.array([0.1,0.2,0.3])
x2 = relu(np.dot(x,weight1) + bias1)


weight2=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
bias2 = np.array([0.1,0.2])
x3 = relu(np.dot(x2,weight2) + bias2)

weight3 = np.array([[0.1,0.3],[0.2,0.4]])
bias3 = np.array([0.1,0.2])
x4 = relu(np.dot(x3,weight3)+bias3)
x4


■ 출력층에서 사용하는 활성화 함수
- 분류 : sigmoid, softmax function
- 회귀 : 회귀는 쓰지 않음. 항등함수(identity function)

1. sigmoid
- binary classification,이진분 류

2. softmax function
- multi classification
- 0 ~ 1 사이의 숫자로 출력되는 함수
- 확률값처럼 사용
- 값이 가장 크게 나온 것으로(확률이 높은 것) 분류

# 출 력 층에서 이걸 써서 확인해야한다.

import numpy as np
x = np.array([0.7,0.1,0.2])
s = np.exp(x)/np.sum(np.exp(x))
np.sum(s)
np.argmax(s)

def softmax_function(arg):
    return np.exp(arg) / np.sum(np.exp(arg))

# softmax_function 오버플로우를 막기 위해 최대값을 빼준 다음에 계산하자
x = np.array([100,1000,10000]) # 에러
softmax_function(x)

모두 -10000 을 해주거나, 로그를 취하거나, 등등 조정을 해줘야해

def softmax_function(arg):
    return np.exp(arg-np.max(arg)) / np.sum(np.exp(arg-np.max(arg)))

x = np.array([100,1000,10000])
softmax_function(x)


[문제] 7를 입력하면 출력값을 예측해주세요.

입력(x)   출력(y)
1           2
2           4
3           6
4           8
5           10
6           12

linear regression

y = 기울기 * x + 절편

최소제곱법(ordinary least square), 경사하강법(미분)

기울기 = y 증가량 / x 증가량 = 합((x-x평균)*(y-y평균)) / 합((x - x평균)**2)

절편 = y평균 -(x평균 * 기울기)

x = np.array([1,2,3,4,5,6]) # 입력값
y = np.array([2,4,6,8,10,12]) # 정답, 출력값

기울기
ss = sum((x-np.mean(x)) * (y-np.mean(y))) / sum((x - np.mean(x))**2)

intercept
i = np.mean(y) - (np.mean(x)*ss)

ss*7+i

import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.plot(x,x*ss+i,c='red')


from scipy import stats

stats.linregress(x,y)


feed forward(순방향) 방식
========================================================>
y(목표) = weight * 입력 + bias

입력 : 1                     출력(목표) : 4

4 = weight * 1 + 1

weight 설정       예측값     오차(목표 - 예측값) 오차가 0에 가까운 값의 weight값을 사용
1                   2             2
2                   3             1
2.5                 3.5           0.5
3                   4             0

4 = 2*1 + bias

bias 설정       예측값     오차(목표 - 예측값) 오차가 0에 가까운 값의 bias값을 사용
1                3            1
1.5              3.5          0.5
2                4            0

경사하강법(Gradient descendent method)
- 미분계수 : 기울기, 평균변화율, 순간변화율
- 한점에서 접선의 기울기

MSE(Mean Squared Error)
실제값과 예측값의 차이를 제곱해 평균한 값

y = 2 # 실제값
y_hat = 0.5 * 1 + 0

error = ((y_hat - y)**2).mean()
error # 2.25

x = np.array([1])
y = np.array([2])

weight = 0.5
bias = 0

y_hat = weight * x + bias
error = ((y_hat-y)**2).mean()
error # 2.25                  ---------> 0 나올 때 까지 weight, bias 조정

# 이 부분 반복
learning_rate = 0.25
weight = weight - learning_rate * ((y_hat-y)*x).mean() # 0.5 ---> 0.875 ---> 1.0625 ---> 1.1562 -> 1.20
bias = bias - learning_rate * (y_hat -y).mean() # 0 ---> 0.375---> 0.5625 ---> 0.65 ---> 0.7

y_hat = weight * x + bias # 0.5 -> 1.25 -> 1.625 -> 1.8125 -> 1.9
error = ((y_hat-y)**2).mean() 
error # 2.25 -> 0.5625-> 0.14 -> 0.035 -> 0.008

# 최종결과
weight = 1.2
bias = 0.7

y_hat = weight * x + bias
y_hat

learning_rate = 0.01
n_epoch = 500 # 반복 몇번했는지
errors = []

x = np.array([1,2,3,4,5,6]) # 입력값
y = np.array([2,4,6,8,10,12]) # 정답, 출력값

w = np.random.uniform(low=-1.0,high=1.0) # 난수값
b = np.random.uniform(low=-1.0,high=1.0)

######## 반복 #########
for epoch in range(n_epoch):
    y_hat = w * x + b
    loss = ((y_hat - y)**2).mean()
    errors.append(loss)
    
    if loss < 0.0005:
        break
    
    w = w - learning_rate * ((y_hat-y)*x).mean()
    b = b - learning_rate * (y_hat-y).mean(0)
    
    if epoch % 100 == 0:
        print("{}, w={}, b={}, loss={}".format(epoch,w,b,loss))
        
#######################

plt.plot(errors)
plt.xlabel('epochs')
plt.ylabel('loss')


TensorFlow
- 구글이 오픈소스로 공개한 머신러닝 라이브러리
- 다차원 행렬계산(tensor), 대규모 숫자 계산
- 빅데이터 처리를 위한 병렬컴퓨터지원을 한다.
- C++로 만들어진 라이브러리
- CPU, GPU
- C++, PYTHON, JAVA

(base) C:\Users\USER> pip install tensorflow-cpu

import tensorflow as tf
tf.__version__

t = tf.constant('tensorflow')
print(t)
t.numpy()

x = tf.constant(1234)
y = tf.constant(5678)
add_op = x+y
add_op.numpy()

a = tf.constant(1,name='a')
b = tf.constant(2,name='b')
c = tf.constant(3,name='c')
z = tf.Variable(0,name='z')
z

z = a+b*c
z
y = a+b*c # 이렇게 그냥 하면 됨
y

x1 = tf.constant([[1,2,3],[4,5,6]])
x1.shape
x2 = tf.constant([[1,2],[3,4],[5,6]])
tf.matmul(x1,x2)

np.dot(x1,x2)

tf.math.add(10,20)
tf.add(100,200)

tf.subtract(100,90)
tf.math.subtract(100,90)

tf.math.multiply(2,3)
tf.multiply(2,3)

tf.truediv(3,6)
tf.divide(3,6)

tf.math.mod(7,2)
tf.mod(7,2) # 오류. 이래서 math 쓰는 습관 들여야돼

[문제]tensorflow 상수를 이용해서 아래와 같이 결과를 출력하는 프로그램을 만드세요. 
Add : 6
Multiply : 8

import numpy as np
import tensorflow as tf

a = tf.Variable(2)
b = tf.Variable(4)

add = tf.math.add(a,b)
mul = tf.math.multiply(a,b)

print('Add : {}'.format(add))
print('Multiply : {}'.format(mul))

텐서 자료구조
- 텐서는 텐서플로의 기본 자료 구조
- 텐서는 다차원배열

1차원텐서
a = np.array([1,2,3])
a.shape
a.ndim
a.dtype

# numpy array -> tensor 전환
tf.convert_to_tensor(a,dtype='int32')
tf.constant(a,tf.float64)
t = tf.convert_to_tensor(a,dtype=tf.int32)
t.ndim
t.shape
t.dtype
t1 = tf.constant(a)
t1.dtype

2차원텐서
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.array([[1,1,1],[2,2,2],[3,3,3]])

a.shape
a.ndim

t1 = tf.constant(a)
t2 = tf.constant(b)

tf.math.add(t1,t2)
tf.matmul(t1,t2) # 행렬의 곱
np.dot(a,b) # 행렬의 곱

3차원텐서
x = np.array([[[1,2],[3,4],[5,6],[7,8]]])
x.shape # 면,행,열

x1 = tf.constant(x)
x1
x1.shape
x1.get_shape()

[문제]
x변수는 1행 3열 모양의 1,2,3
w변수는 3행 1열 모양의 2,2,2
y변수는 x와 w를 행렬의 곱을 이용한 결과를 수행하는 프로그램을 작성하세요.

x = np.array([[1,2,3]])
w = np.array([[1],[2],[3]])
t1 = tf.constant(x)
t2 = tf.constant(w)
tf.matmul(t1,t2)

tf.constant([[1,2,3]]) # 바로가능

# 상수를 생성하는 방법
1. 0의 값으로 텐서를 생성
tf.zeros_like([1,2,3],dtype=tf.int32,name='zeros')
tf.zeros([4,5]) # 2차원행렬(행,열)
tf.zeros([1,4,5])

2. 1의 값으로 텐서를 생성
tf.ones([3,3])
tf.ones([2,3,3])

3. 특정한 값으로 텐서를 생성

tf.fill([3,3],7)
tf.constant(7,shape=[3,3])
# 비교
tf.constant([1,2,3,4],shape=[2,2])

4. 정규분포난수
x = tf.random.normal([3,3],mean=0,stddev=1) # mean=0, stddev=1 기본값
np.mean(x)
np.std(x)

5. 균등분포난수
tf.random.uniform([2,3],minval=1,maxval=3)
tf.random.uniform([5,5],minval=1,maxval=10)

6. 시퀀스
tf.linspace(10,12,5)

tf.range(start=1,limit=10,delta=1)
tf.range(start=1,limit=11,delta=1.5)

# 주어진 값을 shuffle

x = tf.constant([1,2,3,4,5,6],shape=[3,2])
x
tf.random.shuffle(x)

# 행,열의 모습을 수정
tf.reshape(x,shape=[2,3])

y = weight * x + bias

x_data = np.array([1,2,3,4,5,6])
y_data = np.array([2,4,6,8,10,12])

x = tf.constant(x_data,tf.float32) # 입력값
y = tf.constant(y_data,tf.float32) # 종속변수, 정답

w = tf.Variable(tf.random.normal([1]),name='weight')
b = tf.Variable(tf.random.normal([1]),name='bias')

learning_rate = 0.01
for i in range(100):
    with tf.GradientTape() as tape:
        hypothesis = w*x+b
        cost = tf.reduce_mean(tf.square(hypothesis-y)) #MSE
    w_grad,b_grad = tape.gradient(cost,[w,b])
    w.assign_sub(learning_rate*w_grad)
    b.assign_sub(learning_rate*b_grad)
    #cost = tf.reduce_mean(tf.square(hypothesis-y)) #MSE
    #w = w - learning_rate*(∂cot(loss) / ∂w) # weight 값의 수정 -> assing_sub
    #b = b - learning_rate*(∂cot(loss) / ∂b) # bias 값의 수정
    
    if i % 10 == 0:
        print("step:{}, weight:{}, bias:{}, cost:{}".format(i,float(w),float(b),cost))
# cost 값이 거의 0이면 끝난건데, 굳이 너무 많이 돌릴필요없다 조정.
    
    
#cost = tf.reduce_mean(tf.square(hypothesis-y)) #MSE
#w = w - learning_rate*(∂cot(loss) / ∂w) # weight 값의 수정
#b = b - learning_rate*(∂cot(loss) / ∂b) # bias 값의 수정

[문제] linear regression 학습을 통해서 입력값에 대한 예측값을 출력하세요. 
hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b

x1 x2 x3  y
-------------
73 80 75  152
93 88 93  185
89 91 90  180
96 98 100 196
73 66 70  142

x1_data = [73,93,89,96,73]
x2_data = [80,88,91,98,66]
x3_data = [75,93,90,100,70]
y_data = [152,185,180,196,142]

x1 = tf.constant(x1_data,tf.float32)
x2 = tf.constant(x2_data,tf.float32)
x3 = tf.constant(x3_data,tf.float32)
y = tf.constant(y_data,tf.float32)

w1 = tf.Variable(tf.random.normal([1]),name='weight1')
w2 = tf.Variable(tf.random.normal([1]),name='weight2')
w3 = tf.Variable(tf.random.normal([1]),name='weight3')
b = tf.Variable(tf.random.normal([1]),name='bias')



learning_rate = 0.00001
for i in range(30000):
    with tf.GradientTape() as tape:
        hypothesis = w1*x1 +w2*x2 + w3*x3 +b
        cost = tf.reduce_mean(tf.square(hypothesis-y)) #MSE
    w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost,[w1,w2,w3,b])
    w1.assign_sub(learning_rate*w1_grad)
    w2.assign_sub(learning_rate*w2_grad)
    w3.assign_sub(learning_rate*w3_grad)
    b.assign_sub(learning_rate*b_grad)
    #cost = tf.reduce_mean(tf.square(hypothesis-y)) #MSE
    #w = w - learning_rate*(∂cot(loss) / ∂w) # weight 값의 수정 -> assing_sub
    #b = b - learning_rate*(∂cot(loss) / ∂b) # bias 값의 수정
    
    if i % 1000 == 0:
        print("step:{}, weight1:{}, weight2:{}, weight3:{}, bias:{}, cost:{}".format(i,float(w1),float(w2),float(w3),float(b),cost))
    
learning_rate이 너무 커서 NAN 나옴
    -> 0.001 으로 바꿔보자. 해보면서 조정하는거
    -> 0.00001
    -> cost 0에 가깝게 나오도록 반복 조정



# 데이터들을 하나로 묶어서 처리하기

hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b

x1 x2 x3  y
-------------
73 80 75  152
93 88 93  185
89 91 90  180
96 98 100 196
73 66 70  142

x1_data = np.array([73,93,89,96,73]).reshape(-1,1)
x2_data = np.array([80,88,91,98,66]).reshape(-1,1)
x3_data = np.array([75,93,90,100,70]).reshape(-1,1)
y_data = np.array([152,185,180,196,142]).reshape(-1,1)

data = np.hstack((x1_data, x2_data, x3_data))


x = tf.constant(data,tf.float32)
y = tf.constant(y_data,tf.float32)

w = tf.Variable(tf.random.normal([3,1]),name='weight') # 3행 1열로
b = tf.Variable(tf.random.normal([1,]),name='bias')



learning_rate = 0.00001
for i in range(30000):
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(x,w) +b # 행렬의 곱으로
        cost = tf.reduce_mean(tf.square(hypothesis-y)) #MSE
    w_grad, b_grad = tape.gradient(cost,[w,b])
    w.assign_sub(learning_rate*w_grad)
    b.assign_sub(learning_rate*b_grad)
    
    if i % 1000 == 0:
        print("step:{}, weight:{}, bias:{}, cost:{}".format(i,w.numpy,float(b),cost))
    
new = np.array([73,93,72]).reshape(1,3) # w와 행렬곱 하기 위해
new.shape

new = tf.constant(new,tf.float32)
new
predict = tf.matmul(new,w) + b
predict

# 회귀분석
독립변수, 입력변수, 설명변수 : 수치형 자료
종속변수, 결과변수, 목표변수 : 수치형자료(연속형)

예측값(연속형) = weight * 입력값 + bias

LOGIT(Logistic Regression), binary classfication
- 이분법을 기준으로 분류
- 로지스틱 회귀분석은 입력값이 연속적인 값을 가지더라도 로지스틱함수의 결과값은 0~1사이의 값을 갖도록
설계되어 있기 때문에 이분법적인 분류 문제를 해결할 수 있다.

독립변수, 입력변수, 설명변수 : 수치형 자료
종속변수, 결과변수, 목표변수 : 번주형 자료 (0,1)

예측값(0,1) = sigmoid(weight * 입력값 + bias)

# sigmoid
- 0과 1사이의 실수값으로 전달하는 함수
- e 자연상수 : 2.7182.....

def sigmoid(x):
    return 1/(1+np.exp(-x))

sigmoid(100)
sigmoid(-1000)
sigmoid(10000)
sigmoid(-250)

목표 1 : 학습된 결과는 0 이 나왔다. 틀린 답
이를 해결하기 위해 weight 값, bias 조정

binary classification에서는 어떤 cost 함수를 사용해야 하나?
MSE 로 하면 안됨

binary crossentropy cost function을 사용
z = 1
z = 0
target = 1 : -target * np.log(z)
target = 0 : -(1-target) * np.log(1-z)

target = 1
z = 1
-target * np.log(z)

target = 0
z = 0
-(1-target) * np.log(1-z)

target = 0
z = 1
-(1-target) * np.log(1-z)


h = sigmoid(weight*x + bias)
target = 1
h = 0.1

if target == 1:
    -np.log(z)
else:
    -np.log(1-z)

(-target*np.log(z))-(-(1-target) * np.log(1-z))


target = 1
h = 0.1
(-target*np.log(h))-(-(1-target) * np.log(1-h))

target = 1
h = 0.99 # 1
(-target*np.log(h))-(-(1-target) * np.log(1-h))

target = 0
h = 0.001
(-target*np.log(h))-(-(1-target) * np.log(1-h))


target = 0
h = 0.9
(-target*np.log(h))-(-(1-target) * np.log(1-h))


# logistic reg 의 cost
tf.reduce_mean(-tf.reduce_sum(target * tf.math.log(h) + (1-target)*tf.math.log(1-h)))

# 분류
x_data = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y_data = [[0],[0],[0],[0],[0],[1],[1],[1],[1],[1]]

x = tf.constant(x_data,tf.float32)
y = tf.constant(y_data,tf.float32)

w = tf.Variable(tf.random.normal([1,1]),name='weight')
b = tf.Variable(tf.random.normal([1]),name='bias')

learning_rate = 0.01
for i in range(20000):
    with tf.GradientTape() as tape:
        hypothesis = tf.sigmoid(tf.matmul(x,w) +b) # 행렬의 곱으로
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis) + (1-y)*tf.math.log(1-hypothesis))) # binary crossentropy cost function

    w_grad, b_grad = tape.gradient(cost,[w,b])
    w.assign_sub(learning_rate*w_grad)
    b.assign_sub(learning_rate*b_grad)
    
    if i % 1000 == 0:
        print("step:{}, weight:{}, bias:{}, cost:{}".format(i,float(w),float(b),cost))
    
predict = sigmoid(float(w) * 10 + float(b)) # def sigmoid 안만들고 tf.sigmoid 사용 가능
predict

[문제] XOR Logistic regression을 이용해서 분류해주세요. 신경망 이용

x1 x2 y
0  0  0
0  1  1
1  0  1
1  1  0 


import tensorflow as tf
import numpy as np

x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

x = tf.constant(x_data,tf.float32)
y = tf.constant(y_data,tf.float32)
x
y

w = tf.Variable(tf.random.normal([2,1]),name='weight')
b = tf.Variable(tf.random.normal([1]),name='bias')

learning_rate = 0.1

for i in range(20000):
    with tf.GradientTape() as tape:
        hypothesis = tf.sigmoid(tf.matmul(x,w) + b)
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis) +(1-y)*tf.math.log(1-hypothesis)))
        w_grad, b_grad = tape.gradient(cost,[w,b])
        w.assign_sub(learning_rate*w_grad)
        b.assign_sub(learning_rate*b_grad)
    if i%1000 ==0:
        print('>>#{}\n weight:{}\n bias:{}\n cost:{}'.format(i,w.numpy(),b.numpy(),cost.numpy()))


# 2.77 쯤에서 줄어들지 않음. -> XOR 이기 때문에 단층으로는 해결할 수가 없는 것
# 다층퍼셉트론으로 해결하기

x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [[0],[1],[1],[0]]

x = tf.constant(x_data,tf.float32)
y = tf.constant(y_data,tf.float32)
x
y

# 은닉층
w1 = tf.Variable(tf.random.normal([2,4]),name='weight') # 은닉층 뉴런개수 4개 -> 이건 내가 임의로 설정
b1 = tf.Variable(tf.random.normal([4]),name='bias')

# 출력층
w2 = tf.Variable(tf.random.normal([4,1]),name='weight2')
b2 = tf.Variable(tf.random.normal([1]),name='bias2')

learning_rate = 0.01

for i in range(20000):
    with tf.GradientTape() as tape:
        layer1 = tf.nn.relu(tf.matmul(x,w1) + b1)
        hypothesis = tf.sigmoid(tf.matmul(layer1,w2) + b2)
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis) +(1-y)*tf.math.log(1-hypothesis)))
        w1_grad, w2_grad, b1_grad, b2_grad = tape.gradient(cost,[w1,w2,b1,b2])
        w1.assign_sub(learning_rate*w1_grad)
        w2.assign_sub(learning_rate*w2_grad)
        b1.assign_sub(learning_rate*b1_grad)
        b2.assign_sub(learning_rate*b2_grad)
    if i%1000 ==0:
        print('>>#{}\n weight2:{}\n bias2:{}\n cost:{}'.format(i,w2.numpy(),b2.numpy(),cost.numpy()))

hypothesis > 0.5

predict = tf.cast(hypothesis > 0.5,dtype=tf.float32)
predict == y
accuracy = np.mean(predict == y)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(8,input_dim=2, activation='relu')) # layer1,  8 -> 뉴런 수, input2 x 8 = 16, bias =8 > 화살표 24개(param 수)
model.add(Dense(4,activation='relu')) # layer2
model.add(Dense(1,activation='sigmoid')) # 출력층
model.summary()
model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['binary_accuracy'])
model.fit(x,y,epochs=1000)

y_predict = model.predict(x) > 0.5
y_predict.astype(np.int32)

model.evaluate(x,y)


■ multinomial classification

softmax function
- 0 ~ 1 사이의 값으로 출력된다
- 확률값으로 출력
- 모든 출력의 합은 반드시 1이 되어야 한다.

x = 100
np.exp(x) / np.sum(np.exp(x))

target = [0,0,1]
y_hat = [0.1,0.1,0.8]

-(0*np.log(0.6) + 0*np.log(0.1) + 1*np.log(0.3))  # -> cost 값
-(0*np.log(0.1) + 0*np.log(0.1) + 1*np.log(0.8)) 


- 다중분류 cost function
cross entropy

-np.sum(y*np.log(x))

def cross_entropy_function(x,y):
    delta = 1e-7
    return -np.sum(y*np.log(x)+delta)

cross_entropy_function(np.array(y_hat),np.array(target))

0,0,1 -> 0,0,1 이면 NaN 결과나옴. log0이 되어서





■ one hot encoding
한 개의 값만 1이고 나머지 값은 0으로 표현하는 기법

import pandas as pd
from pandas import Series, DataFrame

iris = pd.read_csv('c:/data/iris.csv')

x_data = iris.iloc[:,0:-1]
y_data = iris.Name
y_data.unique() # one-hot encoding 해줘야 해

type(y_data)
labels = {
'Iris-setosa':[1,0,0], 
'Iris-versicolor':[0,1,0], 
'Iris-virginica':[0,0,1]}

labels[y_data[0]]

y_one_hot = list(map(lambda v : labels[v],y_data))

x = tf.constant(x_data,tf.float32)
y = tf.constant(y_one_hot,tf.float32)
w = tf.Variable(tf.random.normal([4,3]),name='weight')
b = tf.Variable(tf.random.normal([3]),name='bias')

learning_rate = 0.01

for i in range(20000):
    with tf.GradientTape() as tape:
        hypothesis = tf.nn.softmax(tf.matmul(x,w) + b) # multinomial 이니까 softmax
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis),axis=1))
        w_grad, b_grad = tape.gradient(cost,[w,b])
        w.assign_sub(learning_rate*w_grad)
        b.assign_sub(learning_rate*b_grad)
    if i%1000 ==0:
        print('>>#{}\n weight:{}\n bias:{}\n cost:{}'.format(i,w.numpy(),b.numpy(),cost.numpy()))

hypothesis
tf.argmax(hypothesis,axis=1)
tf.argmax(y,axis=1)

np.mean(tf.argmax(hypothesis,axis=1) == tf.argmax(y,axis=1))

predict = tf.argmax(hypothesis,axis=1)
real = tf.argmax(y,axis=1)

tf.equal(predict,real)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,real),dtype=tf.float32))
accuracy.numpy()

# 새로운 값을 입력해서 꽃 분류 예측
# 찾은 애들로 넣기
w
b
new = tf.constant(np.array([5.9,3.,5.1,1.8]),shape=(1,4),dtype=tf.float32)
new
new_predict = tf.nn.softmax(tf.matmul(new,w) + b)
np.argmax(new_predict)



 # one hot encoding 방법 2

import pandas as pd

iris = pd.read_csv('c:/data/iris.csv')

x_data = iris.iloc[:,0:-1]
y_data = iris.Name
y_data.unique() # one-hot encoding 해줘야 해

# 문자형 -> 수치형 변환

iris.Name = iris.Name.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

# 수치형 데이터를 one hot encoding 변환
0 -> [1,0,0]
1 -> [0,1,0]
2 -> [0,0,1]

y_one_hot = tf.one_hot(iris.Name,3)
y_one_hot

x = tf.constant(x_data,tf.float32)
y = tf.constant(y_one_hot,tf.float32)
w = tf.Variable(tf.random.normal([4,3]),name='weight')
b = tf.Variable(tf.random.normal([3]),name='bias')

learning_rate = 0.01

for i in range(20000):
    with tf.GradientTape() as tape:
        hypothesis = tf.nn.softmax(tf.matmul(x,w) + b) # multinomial 이니까 softmax
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis),axis=1))
        w_grad, b_grad = tape.gradient(cost,[w,b])
        w.assign_sub(learning_rate*w_grad)
        b.assign_sub(learning_rate*b_grad)
    if i%1000 ==0:
        print('>>#{}\n weight:{}\n bias:{}\n cost:{}'.format(i,w.numpy(),b.numpy(),cost.numpy()))

hypothesis
tf.argmax(hypothesis,axis=1)
tf.argmax(y,axis=1)

np.mean(tf.argmax(hypothesis,axis=1) == tf.argmax(y,axis=1))

predict = tf.argmax(hypothesis,axis=1)
real = tf.argmax(y,axis=1)

tf.equal(predict,real)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predict,real),dtype=tf.float32))
accuracy.numpy()

# 새로운 값을 입력해서 꽃 분류 예측
# 찾은 애들로 넣기
w
b
new = tf.constant(np.array([5.9,3.,5.1,1.8]),shape=(1,4),dtype=tf.float32)
new
new_predict = tf.nn.softmax(tf.matmul(new,w) + b)
np.argmax(new_predict)


 # one hot encoding 방법 3

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

iris = pd.read_csv('c:/data/iris.csv')

x_data = iris.iloc[:,0:-1]
y_data = iris.Name
y_data.unique() # one-hot encoding 해줘야 해

문자형을 one hot encoding으로 변환하는 단계

1. 문자형 -> 수치형 변환 : LabelEncoder
2. 수치형 -> onehotencoding : OneHotEncoder

1.

le = LabelEncoder()
y_integer = le.fit_transform(iris.Name)
y_integer
le.inverse_transform([0])
le.inverse_transform([1])
le.inverse_transform([2])
le.inverse_transform(y_integer)

2. 

# 수치형 데이터를 one hot encoding 변환
0 -> [1,0,0]
1 -> [0,1,0]
2 -> [0,0,1]

one = OneHotEncoder(sparse=False) # sparse=True가 디폴트라서 sparse matrix로 밖에 안보여 False 해줘 -> array변로 환
y_one_hot = one.fit_transform(y_integer.reshape(-1,1))
np.argmax(y_one_hot[0])

le.inverse_transform([np.argmax(y_one_hot[0])])

x = tf.constant(x_data,tf.float32)
y = tf.constant(y_one_hot,tf.float32)
w = tf.Variable(tf.random.normal([4,3]),name='weight')
b = tf.Variable(tf.random.normal([3]),name='bias')

learning_rate = 0.01

for i in range(20000):
    with tf.GradientTape() as tape:
        hypothesis = tf.nn.softmax(tf.matmul(x,w) + b) # multinomial 이니까 softmax
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis),axis=1))
        w_grad, b_grad = tape.gradient(cost,[w,b])
        w.assign_sub(learning_rate*w_grad)
        b.assign_sub(learning_rate*b_grad)
    if i%1000 ==0:
        print('>>#{}\n weight:{}\n bias:{}\n cost:{}'.format(i,w.numpy(),b.numpy(),cost.numpy()))

hypothesis
tf.argmax(hypothesis,axis=1)
tf.argmax(y,axis=1)

np.mean(tf.argmax(hypothesis,axis=1) == tf.argmax(y,axis=1))

predict = tf.argmax(hypothesis,axis=1)
real = tf.argmax(y,axis=1)


 # one hot encoding 방법 4 - keras 이용

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from tensorflow.keras.utils import to_categorical

iris = pd.read_csv('c:/data/iris.csv')

x_data = iris.iloc[:,0:-1]
y_data = iris.Name
y_data.unique() # one-hot encoding 해줘야 해

# 문자형 -> 수치형 변환

iris.Name = iris.Name.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

# 수치형 데이터를 one hot encoding 변환
0 -> [1,0,0]
1 -> [0,1,0]
2 -> [0,0,1]

y_one_hot = to_categorical(iris.Name,3)
y_one_hot

x = tf.constant(x_data,tf.float32)
y = tf.constant(y_one_hot, tf.float32)

model = Sequential()
model.add(Dense(4,input_dim=4,activation='relu')) # layer1, input(feature) -> 4 개임. 맨 왼쪽 뉴런의 수는 상관 ㄴ
model.add(Dense(4,activation='relu')) # layer2
model.add(Dense(3,activation='softmax')) # 출력층
model.summary()
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x,y,epochs=1000)


new = tf.constant(np.array([5.9,3.,5.1,1.8]),shape=(1,4),dtype=tf.float32)
new_predict = model.predict(new)
np.argmax(new_predict)


[문제] 유방암 데이터를 신경망을 이용해서 분류
# 스케일링
from collections import Counter
from sklearn.preprocessing import scale

wisc = pd.read_csv('c:/data/wisc_bc_data.csv')
wisc
wisc.describe()
wisc = wisc.iloc[:,1:]

wisc.diagnosis.unique()
wisc.diagnosis = wisc.diagnosis.map({"B":0,"M":1})
wisc.diagnosis.unique()
Counter(wisc.diagnosis)

y_one_hot = tf.one_hot(wisc.diagnosis,2)
y_one_hot

x = tf.constant(scale(wisc.iloc[:,1:]),tf.float32)
y = tf.constant(y_one_hot,tf.float32)
w = tf.Variable(tf.random.normal([30,2]),name='weight')
b = tf.Variable(tf.random.normal([2]),name='bias')

learning_rate = 0.01

for i in range(20000):
    with tf.GradientTape() as tape:
        hypothesis = tf.nn.softmax(tf.matmul(x,w) + b) 
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis),axis=1))
        w_grad, b_grad = tape.gradient(cost,[w,b])
        w.assign_sub(learning_rate*w_grad)
        b.assign_sub(learning_rate*b_grad)
    if i%1000 ==0:
        print('>>#{}\n weight:{}\n bias:{}\n cost:{}'.format(i,w.numpy(),b.numpy(),cost.numpy()))


# 스케일링 2

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

wisc = pd.read_csv('c:/data/wisc_bc_data.csv')
wisc
wisc.describe()
wisc = wisc.iloc[:,1:]

# 데이터 표준화
wisc_scale = StandardScaler()
wisc_scale.fit(wisc.iloc[:,1:])
wisc_scale.mean_
wisc_scale.scale_

x_scale = wisc_scale.transform(wisc.iloc[:,1:])
x_scale.shape

# 정답 one hot encoding 변환
# 바이너리여도 걍 one hot encoding 해


wisc.diagnosis.unique()
wisc.diagnosis = wisc.diagnosis.map({"B":0,"M":1})
wisc.diagnosis.unique()
Counter(wisc.diagnosis)

y_one_hot = to_categorical(wisc.diagnosis,num_classes=2)
y_one_hot

x_train, x_test, y_train, y_test = train_test_split(x_scale,y_one_hot,test_size=0.2)
x_train.shape
y_train.shape

x = tf.constant(x_train,tf.float32)
y = tf.constant(y_train,tf.float32)

model = Sequential()
model.add(Dense(64,input_dim=30, activation='relu')) # layer1,
model.add(Dense(32,activation='relu')) # layer2
model.add(Dense(2,activation='softmax')) # 출력층
model.summary()
model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(x,y,epochs=10,batch_size=32) # 455/32 , 32개씩 나눠서 미분을 구하자

한 번 epochs 시 에
데이터를 32개씩 15번 올려서 weight값 과 bias값을 계산한다.

score = model.evaluate(x_test,y_test)
print('loss : ',score[0])
print('accuracy : ',score[1])


■ Flatten
- 이미지가 중앙에 있을 때는 픽셀을 세워놓고 판단하는 flatten만 가지고도 된다.

from tensorflow.keras import datasets

(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

x_train.shape # (60000, 28, 28)
y_train.shape

x_test.shape # (10000, 28, 28)
y_test.shape

# grayscale
# 0(검정색) ~ 255(흰색)
x_train[0].shape
y_train[0]

import matplotlib.pyplot as plt

plt.imshow(x_train[0],'gray')
plt.title(y_train[0])
plt.show()

plt.imshow(x_train[1000],'gray')
plt.title(y_train[1000])
plt.show()

# 정답레이블을 one hot encoding으로 변환

from tensorflow.keras.utils import to_categorical
to_categorical(0,10)
to_categorical(5,10)

train_ohe = to_categorical(y_train)
test_ohe = to_categorical(y_test)
train_ohe.shape
test_ohe.shape

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
import tensorflow as tf

model = Sequential([
    Flatten(input_shape=(28,28)), # 784개 픽셀 인풋 세우기
    Dense(100,activation='relu'), # 1번 은닉층 
    Dense(30,activation='relu'),
    Dense(10,activation='softmax')
    ])


model.summary()

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,train_ohe,epochs=20,batch_size=32) # 455/32 , 32개씩 나눠서 미분을 구하자

history.history['loss']
history.history['accuracy']

model.evaluate(x_test,test_ohe,batch_size=128)


■ CNN

Convolutional Neural Network(합성곱)

image(입력) - > Feature Extracotr(특징추출),CNN layer - > Fully connected layer(분류) # 위 Flatten은 이부분만 한것 -> 정답


Convolutional Neural Network(합성곱)
- computer vision(사람의 눈) 에서 많이 사용되는 딥러닝 모델
- 컴퓨터 비전은 시각적 세계를 해석하고 이해하도록 컴퓨터를 학습시키는 인공지능 분야
- 이미지 인식과 음성인식, 텍스트 분석 등 다양한 곳에서 사용되는 딥러닝 모델

convolution -> relu -> pooling -> fully connected

convolution layer
- 이미지의 특징을 찾는 층
- feature map을 만들고 그 feature map을 선명하게 하는 층이다.


합성곱 연산
- 이미지 3차원(세로,가로,색상(채널)) 데이터의 형상을 유지하면서 연산하는 작업,  grayscale 1, RGB 3
- 입력 데이터에 필터를 적용한 것이 합성곱 연산
- 이미지에 대한 특성을 감지하기 위해 커널(kernel)이 필터(filter)이다.

이미지        필터
1 2 3 0      2 0 1
0 1 2 3  *   0 1 2
3 0 1 2      1 0 2
2 3 0 1
입력(4*4)    필터(3*3)      출력(2*2)

stride
합성곱 연산을 하기 위해서 필터를 적용하는 위치의 간격

padding
- 합성곱 연산을 수행하기 전에 입력 데이터 주변을 특정한 값으로 채워 늘리는 작업
- 패딩을 하지 않을 경우 데이터의 공간 크기는 합성곱 계층을 지닐 때 마다 작아지게 되므로
가장 자리 정보들이 사라지게 되는 문제 발생하기 때문에 이를 방지하고자 패딩 이용

입력(4,4), 필터(3,3), 패딩 0 , 스트라이드 1 -> 출력(2,2)
입력(4,4), 필터(3,3), 패딩 1 , 스트라이드 1 -> 출력(4,4)

입력(h,w), 필터크기(fh,fw), 패딩(p), 스트라이드(s), -> 출력(oh, ow)



       h + 2*p - fh
oh = ----------------- + 1
            s



       w + 2*p - fw
ow = ----------------- + 1
            s


pooling
- 출력값에서 일부분만 취하는 가정
- 주변의 픽셀을 묶어서 하나의 대표 픽셀로 바꾼다.
- 이미지 차원을 축소

풀링의 종류
- 최대 풀링 : 컨볼루션 데이터에서 가장 큰 값을 대표값으로 선정
- 평균 풀링 : 컨볼루션 데이터에서 모든 값의 평균을 대표값으로 선정

# sequential model 생성 방법

model = Sequential([
    Flatten(input_shape=(28,28)), # 784개 픽셀 인풋 세우기
    Dense(100,activation='relu'), # 1번 은닉층 
    Dense(30,activation='relu'),
    Dense(10,activation='softmax')
    ])
model.summary()



model = Sequential()
model.add(Flatten(input_shape=(28,28))
model.add(Dense(100,activation='relu'))  
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

# functional API 이용 방법 -  이 방법을 이용하자

from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model

input_size = 28
input_tensor = Input(shape=(input_size,input_size))
x = Flatten()(input_tensor)
x = Dense(100,activation='relu')(x)
x = Dense(30,activation='relu')(x)
output = Dense(10,activation='softmax')(x)
model=Model(inputs=input_tensor,outputs=output)
model.summary()

image = x_train[0]
image.shape
label = y_train[0]
label

plt.imshow(image,'gray')
plt.title(label)
plt.show()

tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1),padding='same',activation='relu') # 필터개수, 필터사이즈
tf.keras.layers.Conv2D(filters=3, kernel_size=(3,3), strides=(1,1),padding='valid')
tf.keras.layers.Conv2D(3, 3, 1,'same')

x_train.shape # (60000, 28, 28) -> (60000, 28, 28, 1) CNN에서 학습할 때 4개 이런 모습으로 되어있어야 해

image.shape # 28 * 28 -> (batchsize,height,width,channel) (1, 28, 28, 1)

# 차원수 늘리기
image = image[tf.newaxis,...,tf.newaxis]
image.shape

image.dtype
image = tf.cast(image,dtype=tf.float32)
image.dtype

import numpy as np
np.min(image)
np.max(image)

# 합성곱
layer = tf.keras.layers.Conv2D(3,3,1,'same')
output = layer(image)
output
np.min(output)
np.max(output)

plt.subplot(1,2,1)
plt.imshow(image[0,:,:,0],'gray') # 원본이미지
plt.subplot(1,2,2)
plt.imshow(output[0,:,:,0],'gray') # 합성곱 연산 수행 뒤 이미지, 필터 3개니까 3개의 이미지가 만들어진 것
plt.show()

filter 에 있는 값은 기존의 weight 값과 같다.

w_0 = layer.get_weights()[0] # weight
w_0.shape # (height,width,channel,filter 번호)

w_1 = layer.get_weights()[1] # bias


plt.imshow(w_0[:,:,0,0],'gray')



plt.subplot(1,3,1)
plt.imshow(image[0,:,:,0],'gray') # 원본이미지
plt.subplot(1,3,2)
plt.imshow(w_0[:,:,0,0],'gray') # 첫번째 필터
plt.subplot(1,3,3)
plt.imshow(output[0,:,:,0],'gray') # 합성곱 연산 수행 뒤 이미지, 필터 3개니까 3개의 이미지가 만들어진 것
plt.show()

plt.subplot(1,3,1)
plt.imshow(image[0,:,:,0],'gray') # 원본이미지
plt.subplot(1,3,2)
plt.imshow(w_0[:,:,0,1],'gray') # 두번째 필터
plt.subplot(1,3,3)
plt.imshow(output[0,:,:,0],'gray') # 합성곱 연산 수행 뒤 이미지, 필터 3개니까 3개의 이미지가 만들어진 것
plt.show()

plt.subplot(1,3,1)
plt.imshow(image[0,:,:,0],'gray') # 원본이미지
plt.subplot(1,3,2)
plt.imshow(w_0[:,:,0,2],'gray') # 세번째 필터
plt.subplot(1,3,3)
plt.imshow(output[0,:,:,0],'gray') # 합성곱 연산 수행 뒤 이미지, 필터 3개니까 3개의 이미지가 만들어진 것
plt.show()


np.min(image)
np.max(image)
# 합성곱
layer = tf.keras.layers.Conv2D(3,3,1,'same')
output = layer(image)
output
np.min(output)
np.max(output)

# ReLU
layer = tf.keras.layers.ReLU()
output = layer(output)

np.min(output) # relu니까 0보다 작은값 없어지지
np.max(output)

plt.subplot(1,2,1)
plt.imshow(image[0,:,:,0],'gray') # 원본이미지
plt.subplot(1,2,2)
plt.imshow(output[0,:,:,0],'gray') # relu 수행 후 이미지
plt.show()

# pooling
layer = tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2),padding='same')
output = layer(output)
output #shape=(1, 14, 14, 3)

plt.subplot(1,2,1)
plt.imshow(image[0,:,:,0],'gray') # 원본이미지
plt.subplot(1,2,2)
plt.imshow(output[0,:,:,0],'gray') # maxpooling 수행 후 이미지
plt.show()

# Flatten
layer = tf.keras.layers.Flatten()
output = layer(output)
output #shape=(1, 588)

plt.subplot(1,2,1)
plt.imshow(image[0,:,:,0],'gray') # 원본이미지
plt.subplot(1,2,2)
plt.imshow(output[:,:100],'gray') # Flatten 수행 후 이미지
plt.show()

# Dense
layer = tf.keras.layers.Dense(32,activation='relu')
output = layer(output)
output.shape



caltech_dir = "C:/data/101_ObjectCategories/"
categories = ['chair','camera','butterfly','elephant','flamingo']

# one-hot-encoding 정답을 원핫인코딩해줘야돼. 여기서 정답은 카테고리 이름이지

'chair' => [1,0,0,0,0]
'camera' => [0,1,0,0,0]

"C:/data/101_ObjectCategories/chair/*.jpg"
"C:/data/101_ObjectCategories/camera/*.jpg"

import glob
from PIL import Image


glob.glob("C:/data/101_ObjectCategories/chair/*.jpg")
glob.glob("C:/data/101_ObjectCategories/camera/*.jpg")
x = []
y = []
image_w = 64
image_h = 64

for idx,cate in enumerate(categories):
    label = [0 for i in range(5)]
    label[idx] = 1
    image_dir = caltech_dir+cate
    files = glob.glob(image_dir+'/*.jpg')
    for i in files:
        print(i)
        img = Image.open(i)
        img = img.convert("RGB")
        img = img.resize((image_w,image_h))
        data = np.asarray(img)
        x.append(data)
        y.append(label) 
        
        
        
x[0].shape #(64, 64, 3) grayscale 아니고 RGB 니까 3

len(x)
len(y)

x = np.array(x)
x.shape
y = np.array(y)
y.shape

x[0]
y[0]
x[333]
y[333]

x[0].shape # (64, 64, 3)
plt.imshow(x[0][:,:,0])
plt.title(y[0])

plt.imshow(x[333][:,:,0])
plt.title(y[333])

x_train,x_test,y_train,y_test = train_test_split(x,y)

x_train.shape
y_train.shape

x_test.shape
y_test.shape

# 데이터셋 저장, 불러오기 #######################################
xy = (x_train,x_test,y_train,y_test)
xy

np.save("C:/data/101_ObjectCategories/my_image.npy",xy)

train_x, test_x, train_y, test_y = np.load("C:/data/101_ObjectCategories/my_image.npy",allow_pickle=True)
train_x.shape
train_y.shape
■ 이미지 학습시에 데이터에 대한 표준화작업(보통 최대값으로 나눈다)을 수행하는 게 학습효과가 좋다.

x_train[0]
x_test[0]
y_train,y_test

x_train = x_train.astype('float')/255
x_test = x_test.astype('float')/255
x_train[0] # 250, 64, 64 ,3
x_test[0]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same', input_shape=(64,64,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu')) # 정답률 높이려면 이 Conv층을 더넣어봐
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(5,activation='softmax'))
model.summary()

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=20,batch_size=32) 

model.evaluate(x_test,y_test,batch_size=64)

# overfitting -> Dense 과정에서 weight 선을 몇개 끊는 dropout 쓰자
# overfitting -> 이미지 변형 작업 (다양한 각도 등)

# 새로운 이미지를 모델에 적용시켜 예측받기
test = []
new = Image.open("C:/data/101_ObjectCategories/chair/image_0061.jpg")
new = new.convert("RGB")
new = new.resize((64,64))
new = np.asarray(new)
test.append(new)
new_data = np.array(test)
np.argmax(model.predict(new_data.astype('float')/255))
categories[np.argmax(model.predict(new_data.astype('float')/255))]




############################### 개 고양이 연습 ##############################

from selenium import webdriver
import urllib
from urllib.request import urlopen
import time
from pandas import DataFrame, Series
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.common.keys import Keys
import urllib.request as req


url = 'https://search.daum.net/search?nil_suggest=btn&w=img'
driver = webdriver.Chrome('c:/data/chromedriver.exe')
driver.get(url)
driver.implicitly_wait(2)
element = driver.find_element(By.CLASS_NAME,'tf_keyword')
element = driver.find_element(By.XPATH,'')
element = driver.find_element(By.CSS_SELECTOR,'div.inner_searcher > input')

element.clear()
element.send_keys('고양이')
driver.implicitly_wait(1)
element.submit()
element.send_keys(Keys.ENTER)

cnt=0
while True:
    
    for i in range(4):
        driver.find_element(By.TAG_NAME,'body').send_keys(Keys.END)
        time.sleep(1)
        
    driver.find_element(By.CLASS_NAME,'open').click()
    time.sleep(1)
    cnt += 1
    if cnt ==5:
        break

html = driver.page_source
driver.quit()

soup = BeautifulSoup(html,'html.parser')
soup
img_url = []
for i in soup.select('img.thumb_img'):
    img_url.append(i.attrs['src'])
    
for i in range(len(img_url)):    
    req.urlretrieve(img_url[i],'c:/cat/daumcat'+str(i)+'.jpg')
    
    
    
dc_dir = "C:/image/"
catego = ['dog','cat']

x = []
y = []
image_w = 64
image_h = 64

for idx,cate in enumerate(catego):
    label = [0 for i in range(2)]
    label[idx] = 1
    image_dir = dc_dir+cate
    files = glob.glob(image_dir+'/*.jpg')
    for i in files:
        print(i)
        img = Image.open(i)
        img = img.convert("RGB")
        img = img.resize((image_w,image_h))
        data = np.asarray(img)
        x.append(data)
        y.append(label) 

len(x)
len(y)
x[0].shape

x = np.array(x)
x.shape
y = np.array(y)
y.shape

x[0]
y[0]
x[333]
y[333]

x[0].shape # (64, 64, 3)
plt.imshow(x[0][:,:,0])
plt.title(y[0])

plt.imshow(x[333][:,:,0])
plt.title(y[333])

x_train,x_test,y_train,y_test = train_test_split(x,y)

x_train.shape
y_train.shape

x_test.shape
y_test.shape

# 데이터셋 저장, 불러오기 #######################################
xy = (x_train,x_test,y_train,y_test)
xy

np.save("C:/image/dogcat.npy",xy)

train_x, test_x, train_y, test_y = np.load("C:/data/101_ObjectCategories/my_image.npy",allow_pickle=True)
train_x.shape

■ 이미지 학습시에 데이터에 대한 표준화작업(보통 최대값으로 나눈다)을 수행하는 게 학습효과가 좋다.

x_train[0]
x_test[0]
y_train,y_test

x_train = x_train.astype('float')/255
x_test = x_test.astype('float')/255
x_train.shape # 250, 64, 64 ,3
y_train.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same', input_shape=(64,64,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu')) # 정답률 높이려면 이 Conv층을 더넣어봐
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2,activation='softmax'))
model.summary()

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,epochs=20,batch_size=32) 

model.evaluate(x_test,y_test,batch_size=64)

# overfitting -> Dense 과정에서 weight 선을 몇개 끊는 dropout 쓰자
# overfitting -> 이미지 변형 작업 (다양한 각도 등)

# 새로운 이미지를 모델에 적용시켜 예측받기
test = []
new = Image.open("C:/data/101_ObjectCategories/dalmatian/image_0014.jpg")
new = new.convert("RGB")
new = new.resize((64,64))
new = np.asarray(new)
test.append(new)
new_data = np.array(test)
np.argmax(model.predict(new_data.astype('float')/255))
catego[np.argmax(model.predict(new_data.astype('float')/255))]

# conv2d 3번 -> 65퍼
# conv2d 5번 -> 66퍼
# model.add(Dropout(0.5)) 2번 -> 65퍼
# 정제 후 conv2d 5번, dropout 5번 -> 68퍼

# 결론 - 이미지 품질( 정제)가 제일 중요

overfitting
- 학습데이터에 대해 과하게 학습된 상황
- 학습데이터에 대해서는 정확도가 좋은데 학습데이터 이외의 데이터에 대해서는 정확도가 좋지않음
- 학습데이터가 부족하거나 데이터의 특성에 ㅂ해 모델이 너무 복잡한 경우

underfitting
-학습데이터도 학습을 하지 못한 상태
- 학습 반복횟수가 적을때
- 데이터의 특성에 비해 모델이 너무 간단하게 설계
- 데이터의 양이 적을 경우

데이터 증강(Data Augmentation)
- CNN 모델의 성능을 높이고 overfitting, underfitting을 줄일 수 있는 방법
- data augmentation 은 학습시에 원본 이미지에 다양한 변형을 가해서 학습 이미지 데이터를 늘리는 효과를 가하는 기법
- data augmentation은 원본 학습 이미지 개수를 늘리는것이 아니고 학습시마다 개별 원본 이미지를 변현해서 학습을 수행한다.

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img


ImageDataGenerator(
    rotation_range=10,# 이미지 회전(0~180)
    width_shift_range=0.2, # 이미지 수평(좌우) 랜덤 평행이동
    height_shift_range=0.2, #    이미지 수직(상하) 랜덤 평행이동
    shear_range=0.1, # y 축 방향으로 각도를 증가시켜 변형
    zoom_range=0.1, # 확대(1보다 큰경우), 축소(1보다 작은경우)
    horizontal_flip=True, # True로 설정 -> 50% 확률로 이미지 수평 뒤집기
    # vertical_flip=True, # True로 설정 -> 50% 확률로 이미지 수직 뒤집기
    fill_mode = 'nearest' # 이미지를 회전, 이동, 축소, 할 때 공간을 채우는 방식
                # 'nearest' : 빈공간에 가장 근접한 픽셀로 채우기
                # 'reflelct' : 빈공간 만큼 영역을 근처 공간으로 채우되 거울로 반사되는 이미지 처럼 채운다.
                # 'wrap' : 빈 공간을 잘려나간 이미지로 채움
                # 'content' : 특정 픽셀값으로 채운다 cval = 0(검은색)
    brightness_range=(0.1,0.9) # 밝기 조절 0~ 1 사이의값, 0에 가까울수록 어둡다
    channel_shift_range = 100 # R,G,B 값을 -100 ~100 사이 임의의 값을 더하여 변환
    )

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2)

img = load_img("C:/data/101_ObjectCategories/dalmatian/image_0014.jpg")
img
x = img_to_array(img)
x.shape

# (294, 300, 3) -> (1, 294, 300, 3)

x = x[np.newaxis,:,:,:]
x.shape

i = 0
for batch in datagen.flow(x,save_to_dir='C:/data/101_ObjectCategories/dalmatian/',save_prefix='20220511',save_format='jpg'):
    i +=1
    if i> 3:
        break

import os

os.listdir('c:/cats_dogs')
os.listdir('c:/cats_dogs/train')
os.listdir('c:/cats_dogs/train/cats')



1. binary

train_path = 'c:/cats_dogs/train'
validation_path = 'c:/cats_dogs/validation'

training_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255.)

training_generator = training_datagen.flow_from_directory(train_path,
                                      batch_size=32,
                                      target_size=(64,64),
                                      class_mode='binary')

training_generator.classes
training_generator.class_indices # flow_from_directory 이용하면 자동으로 레이블이 달린다.

validation_generator = valid_datagen.flow_from_directory(validation_path,
                                      batch_size=32,
                                      target_size=(64,64),
                                      class_mode='binary')


validation_generator.classes
validation_generator.class_indices

# 모델 적용

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same', input_shape=(64,64,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu')) # 정답률 높이려면 이 Conv층을 더넣어봐
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(training_generator, validation_data = validation_generator, epochs=20) 

model.evaluate(x_test,y_test,batch_size=64)

img = load_img("C:/data/101_ObjectCategories/dalmatian/image_0014.jpg")
x = img_to_array(img)
x.shape
# 새로운 데이터의 사이즈 조절
x = tf.image.resize(x,[64,64])


(64,64,3) - > (1,64,64,3)
x = np.array([x])
x.shape

predict = model.predict(x.astype('float')/255) > 0.5

list(training_generator.class_indices.keys())[int(predict)]


img = load_img("C:/data/101_ObjectCategories/cougar_face/image_0014.jpg")
x = img_to_array(img)
x.shape
# 새로운 데이터의 사이즈 조절
x = tf.image.resize(x,[64,64])


(64,64,3) - > (1,64,64,3)
x = np.array([x])
x.shape

predict = model.predict(x.astype('float')/255) > 0.5

list(training_generator.class_indices.keys())[int(predict)]






2. categorical

train_path = 'c:/cats_dogs/train'
validation_path = 'c:/cats_dogs/validation'

training_datagen = ImageDataGenerator(
    rescale=1./255.,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255.)

training_generator = training_datagen.flow_from_directory(train_path,
                                      batch_size=32,
                                      target_size=(64,64),
                                      class_mode='categorical')

training_generator.classes
training_generator.class_indices # flow_from_directory 이용하면 자동으로 레이블이 달린다.

validation_generator = valid_datagen.flow_from_directory(validation_path,
                                      batch_size=32,
                                      target_size=(64,64),
                                      class_mode='categorical')


validation_generator.classes
validation_generator.class_indices

# 모델 적용

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),strides=(1,1),padding='same', input_shape=(64,64,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu')) # 정답률 높이려면 이 Conv층을 더넣어봐
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.summary()

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(training_generator, validation_data = validation_generator, epochs=20) 

model.evaluate(x_test,y_test,batch_size=64)

img = load_img("C:/data/101_ObjectCategories/dalmatian/image_0014.jpg")
x = img_to_array(img)
x.shape
# 새로운 데이터의 사이즈 조절
x = tf.image.resize(x,[64,64])


(64,64,3) - > (1,64,64,3)
x = np.array([x])
x.shape

predict = model.predict(x.astype('float')/255)

list(training_generator.class_indices.keys())[np.argmax(predict)]


img = load_img("C:/data/101_ObjectCategories/cougar_face/image_0014.jpg")
x = img_to_array(img)
x.shape
# 새로운 데이터의 사이즈 조절
x = tf.image.resize(x,[64,64])


(64,64,3) - > (1,64,64,3)
x = np.array([x])
x.shape

predict = model.predict(x.astype('float')/255)

list(training_generator.class_indices.keys())[np.argmax(predict)]




from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


from tensorflow.keras import datasets

(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

다중분류 정답은 one hot encoding 변환

0 -> to_categorical(0,10)
5 -> to_categorical(5,10)

train_ohe = to_categorical(y_train)
test_ohe = to_categorical(y_test)



input_size = 28

image = x_train[100]
label = y_train[100]

model = Sequential([
    Flatten(input_shape=[input_size,input_size]),
    Dense(100,activation='relu'),
    Dense(30,activation='relu'),
    Dense(10,activation='softmax')    
    ])

model.summary()

model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x=x_train,y=train_ohe, batch_size=32,epochs=20)

x_train.shape
60000/32

1 epochs 시에 전체 데이터 60000 데이터를 32개씩 총 1875번 메모리에 올려서 미분을 구한다

print(history.history['loss'])
print(history.history['accuracy'])


# 테스트 데이터로 모델 성능 검증
score = model.evaluate(x_test,test_ohe,batch_size=64)
print('loss :',score[0])
print('accuracy :',score[1])

# 모델 저장
model.save('c:/data/my_mnist_model.h5')

# 모델 불러오기

from tensorflow.keras.models import Sequential, load_model

new_model = load_model('c:/data/my_mnist_model.h5')
new_model.summary()


(base) C:\Users\USER>pip install opencv-python

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('c:/data/4.png',cv2.IMREAD_GRAYSCALE)
img

plt.imshow(x_test[0],'gray')
plt.imshow(255-img,'gray')

img_4 = cv2.resize(255-img, (28,28))
plt.imshow(img_4,'gray')

img_4.shape
28, 28 -> 1, 28, 28, 1
img_4.reshape(1,28,28,1)


predict = new_model.predict(img_4.reshape(1,28,28,1))
predict

import numpy as np

np.argmax(predict)

# 검증 데이터셋을 이용해서 학습 수행 -> overfitting 줄임
from sklearn.model_selection import train_test_split

train_x,valid_x, train_y,valid_y = train_test_split(x_train,y_train, test_size=0.15)
train_x.shape
train_y.shape

train_ohe = to_categorical(train_y)
valid_ohe = to_categorical(valid_y)


input_size = 28

image = x_train[100]
label = y_train[100]

model = Sequential([
    Flatten(input_shape=[input_size,input_size]),
    Dense(100,activation='relu'),
    Dense(30,activation='relu'),
    Dense(10,activation='softmax')    
    ])

model.summary()

model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x=train_x,y=train_ohe,validation_data= (valid_x,valid_ohe), batch_size=32,epochs=20)


callback

1.
modelcheckpoint
특정조건에 맞춰서 모델의 weight 값을 파일로 저장

from tensorflow.keras.callbacks import ModelCheckpoint

mcp_cb = ModelCheckpoint(filepath = 'c:/mnist/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                monitor='val_loss',mode='min',# loss니까 min, accu면 max
                save_best_only=True,save_weight_only=True,period=3,verbose=1) 


input_size=28
mnist_model = Sequential([
    Flatten(input_shape=[input_size,input_size]),
    Dense(100,activation='relu'),
    Dense(30,activation='relu'),
    Dense(10,activation='softmax')    
    ])

mnist_model.summary()

mnist_model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])

# weight값을 새로운 모델에 적용하기

mnist_model.load_weights('c:/mnist/weights.03-0.21.hdf5')
score = mnist_model.evaluate(valid_x,valid_ohe,batch_size=32)



2. ReduceLROnPlateau
epochs 회수동안 성능이 개선되지 않을 경우 learning rate을 동적으로 감소시키는 기능

from tensorflow.keras.callbacks import ReduceLROnPlateau

rlp_cb = ReduceLROnPlateau(monitor='val_loss',mode='min',patience=1,factor=0.3,verbose=1)

LR 0.001 * factor 0.3


input_size=28
model = Sequential([
    Flatten(input_shape=[input_size,input_size]),
    Dense(100,activation='relu'),
    Dense(30,activation='relu'),
    Dense(10,activation='softmax')    
    ])

model.summary()

model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x=train_x,y=train_ohe,validation_data= (valid_x,valid_ohe), 
                    batch_size=32,epochs=20,callbacks=[rlp_cb])


3. EarlyStopping
epochs 동안 성능이 개선되지 않은 경우 학습을 조기에 중단하는 방법

from tensorflow.keras.callbacks import EarlyStopping

es_cb = EarlyStopping(monitor='val_loss',mode='min',patience=1,verbose=1)


input_size=28
model = Sequential([
    Flatten(input_shape=[input_size,input_size]),
    Dense(100,activation='relu'),
    Dense(30,activation='relu'),
    Dense(10,activation='softmax')    
    ])

model.summary()

model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x=train_x,y=train_ohe,validation_data= (valid_x,valid_ohe), 
                    batch_size=32,epochs=20,callbacks=[es_cb])



4. 콜백들 한번에 수행하기


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

mcp_cb = ModelCheckpoint(filepath = 'c:/mnist/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                monitor='val_loss',mode='min',# loss니까 min, accu면 max
                save_best_only=True,save_weight_only=True,period=3,verbose=1) 

rlp_cb = ReduceLROnPlateau(monitor='val_loss',mode='min',patience=1,factor=0.3,verbose=1)
es_cb = EarlyStopping(monitor='val_loss',mode='min',patience=1,verbose=1)


input_size=28
model = Sequential([
    Flatten(input_shape=[input_size,input_size]),
    Dense(100,activation='relu'),
    Dense(30,activation='relu'),
    Dense(10,activation='softmax')    
    ])

model.summary()

model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(x=train_x,y=train_ohe,validation_data= (valid_x,valid_ohe), 
                    batch_size=32,epochs=20,callbacks=[mcp_cb,es_cb,rlp_cb])





train_path = 'c:/cats_dogs/train'
validation_path = 'c:/cats_dogs/validation'

#preprocessing_function

training_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
    rescale=1./255.,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255.)

training_generator = training_datagen.flow_from_directory(train_path,
                                      batch_size=32,
                                      shuffle=True
                                      target_size=(64,64),
                                      class_mode='binary')


training_generator.classes
training_generator.class_indices # flow_from_directory 이용하면 자동으로 레이블이 달린다.

validation_generator = valid_datagen.flow_from_directory(validation_path,
                                      batch_size=32,
                                      shuffle=False,
                                      target_size=(64,64),
                                      class_mode='binary')


validation_generator.classes
validation_generator.class_indices

# VGG16
vgg_model = VGG16(input_shape=(64,64,3),include_top=False,weights='imagenet')
vgg_model.summary()
vgg_model.trainable=False


model = Sequential([
    vgg_model,
    Flatten(),
    Dense(100,activation='relu'),
    Dense(30,activation='relu'),
    Dense(2,activation='softmax')    
    ])

model.summary()

model.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(traing_generator,validation_data= validation_generator, 
                    batch_size=32,epochs=20,callbacks=[rlp_cb])

challenge?















