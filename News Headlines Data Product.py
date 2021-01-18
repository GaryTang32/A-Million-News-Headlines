import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE 
import pandas as pd 
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
import chardet
from nltk.stem import WordNetLemmatizer
from joblib import dump, load
from numpy import savetxt
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Read CSV 
Raw_headlines = pd.read_csv('abcnews-date-text.csv')
print(Raw_headlines.info())
Process_Headline = Raw_headlines.copy(deep = True)

#drop duplicated head_lines.
Process_Headline.drop_duplicates(subset ="headline_text", keep = False, inplace = True) 
print(Process_Headline.info())


# Determine tokenizer 
tokenizer = RegexpTokenizer(r'[a-zA-Z\']+')

# Determine Stemming method
stemmer = SnowballStemmer('english')
lemmas = WordNetLemmatizer()

#get the stop word list 
# explicly add punciation to the stop word list from the SKLearn
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
StopWord = text.ENGLISH_STOP_WORDS.union(punc)

# Process the headline to lower letter 
Process_Headline['headline_text'] = Process_Headline['headline_text'].str.lower()

#Lemmatizer approach
Process_Headline['headline_text'] = Process_Headline['headline_text'].apply(lambda x: ' '.join([lemmas.lemmatize(word) for word in tokenizer.tokenize(x)]))

#Stemmar approach
#Process_Headline['headline_text'] = Process_Headline['headline_text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in tokenizer.tokenize(x)]))

headline_content = Process_Headline['headline_text'].values
vectorizer = TfidfVectorizer(stop_words = StopWord, max_features = 40000)
TD_IDF_DATA = vectorizer.fit_transform(headline_content)
TD_IDF_DATA = TD_IDF_DATA.sorted_indices()


#Failed AE
'''
### Encoder
encoder = Sequential()
encoder.add(Dense(9000,input_dim=len(word_features2),activation="relu"))
encoder.add(Dense(8000,activation="relu"))
encoder.add(Dense(7000,activation="relu"))
 
 
### Decoder
decoder = Sequential()
decoder.add(Dense(8000,input_shape=[7000],activation='relu'))
decoder.add(Dense(9000,activation='relu'))
decoder.add(Dense(len(word_features2),activation ="sigmoid"))

### Autoencoder
autoencoder = Sequential([encoder,decoder])
autoencoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
autoencoder.fit(Stem_header,Stem_header,epochs=200,batch_size=128,verbose = 1)

AE_Headlines = encoder.predict(Stem_header)
print(AE_Headlines[:10])
'''

#Dimensionally reduce all the featres into 100 topics. 

'''
LDA_Model = load('LDA_Model.joblib') 
LDA_Data = pd.read_csv('LDA_Data.csv')
LDA_Data = LDA_Data.sample(n = 500000) 
LDA_Data.sort_index()
'''

print("Construct LDA model")
LDA_Model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=42,max_iter=3,verbose= 1) 
print("Transforming orginal data to LDA data")
LDA_Data=LDA_Model.fit_transform(TD_IDF_DATA)
print("Saving LDA Model")
dump(LDA_Model, 'LDA_Model.joblib')
savetxt('LDA_Data.csv', LDA_Data, delimiter=',')
print("Saved LDA Model")


'''
LDA_Model = load('LDA_Model.joblib') 
LDA_Data = pd.read_csv('LDA_Data.csv', header = None)
'''



print('tsne2')
TSNE_Model = TSNE(n_components=2, n_jobs = 4, verbose = 2, random_state = 42).fit_transform(LDA_Data)
dump(TSNE_Model, 'TSNE_Model.joblib')
savetxt('TSNE_Model.csv', TSNE_Model, delimiter=',')
print(TSNE_Model[:10])

'''
print('tsne3')
TSNE_Model_3 = TSNE(n_components=3, n_jobs = 4, verbose = 2, random_state = 42).fit_transform(LDA_Data)
dump(TSNE_Model_3, 'TSNE_Model_3.joblib')
savetxt('TSNE_Model_3.csv', TSNE_Model_3, delimiter=',')
print(TSNE_Model_3[:10])
'''

vocab = vectorizer.get_feature_names() 
for i, comp in enumerate(LDA_Model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0]+', ', end = '')
    print("\n")

'''
LDA_Data_5K = LDA_Data.sample(n = 5000)
LDA_Data_10K = LDA_Data.sample(n = 10000)
LDA_Data_20K = LDA_Data.sample(n = 20000)
LDA_Data_50K = LDA_Data.sample(n = 50000)
LDA_Data_100K = LDA_Data.sample(n = 100000)
LDA_Data_150K = LDA_Data.sample(n = 150000)
LDA_Data_250K = LDA_Data.sample(n = 250000)
LDA_Data_500K = LDA_Data.sample(n = 500000)
'''

print("Construct KMeans Model")
KMeans_Clustering = KMeans(n_clusters=10, random_state=42, verbose = 2, n_jobs = 3)
print("Clustering KMeans Model")
KMeans_label = KMeans_Clustering.fit_predict(TSNE_Model)
savetxt('KMeans_label.csv', KMeans_label,delimiter=',')
dump(KMeans_Clustering, 'KMeans_Clustering.joblib')

'''
KMeans_label_5k = KMeans_Clustering.fit_predict(LDA_Data_5K)
KMeans_label_10k = KMeans_Clustering.fit_predict(LDA_Data_10K)
KMeans_label_20k = KMeans_Clustering.fit_predict(LDA_Data_20K)
KMeans_label_50k = KMeans_Clustering.fit_predict(LDA_Data_50K)
KMeans_label_100k = KMeans_Clustering.fit_predict(LDA_Data_100K)
KMeans_label_150k = KMeans_Clustering.fit_predict(LDA_Data_150K)
KMeans_label_250k = KMeans_Clustering.fit_predict(LDA_Data_250K)
KMeans_label_500k = KMeans_Clustering.fit_predict(LDA_Data_500K)

print(KMeans_label[:1000])
print(KMeans_Clustering.labels_.tolist().count(0))
print(KMeans_Clustering.labels_.tolist().count(1))
print(KMeans_Clustering.labels_.tolist().count(2))
print(KMeans_Clustering.labels_.tolist().count(3))
print(KMeans_Clustering.labels_.tolist().count(4))
print(KMeans_Clustering.labels_.tolist().count(5))
print(KMeans_Clustering.labels_.tolist().count(6))
print(KMeans_Clustering.labels_.tolist().count(7))
print(KMeans_Clustering.labels_.tolist().count(8))
print(KMeans_Clustering.labels_.tolist().count(9))
print(KMeans_Clustering.cluster_centers_)
exit()

print("Construct DBSCAN Model")
DBSCAN_Clustering = DBSCAN()
print("Clustering DBSCAN Model")
DBSCAN_Label = DBSCAN_Clustering.fit_predict(LDA_Data)

print("Saving DBSCAN_Clustering Model")
dump(DBSCAN_Clustering, 'DBSCAN_Clustering.joblib')
savetxt('DBSCAN_Label.csv', DBSCAN_Label, delimiter=',')
print("Saved DBSCAN_Clustering Model")

print(DBSCAN_Label[:1000])

print("2015")
print(KMeans_label[:10])
KMeans_label_2015 = KMeans_label[925481:1002770+1]
KMeans_label_2015 = KMeans_label_2015.iloc[:,0]
print(KMeans_label_2015.tolist().count(0))
print(KMeans_label_2015.tolist().count(1))
print(KMeans_label_2015.tolist().count(2))
print(KMeans_label_2015.tolist().count(3))
print(KMeans_label_2015.tolist().count(4))
print(KMeans_label_2015.tolist().count(5))
print(KMeans_label_2015.tolist().count(6))
print(KMeans_label_2015.tolist().count(7))
print(KMeans_label_2015.tolist().count(8))
print(KMeans_label_2015.tolist().count(9))
print("2016")
KMeans_label_2016 = KMeans_label[1002771:1064344+1]
KMeans_label_2016 = KMeans_label_2016.iloc[:,0]
print(KMeans_label_2016.tolist().count(0))
print(KMeans_label_2016.tolist().count(1))
print(KMeans_label_2016.tolist().count(2))
print(KMeans_label_2016.tolist().count(3))
print(KMeans_label_2016.tolist().count(4))
print(KMeans_label_2016.tolist().count(5))
print(KMeans_label_2016.tolist().count(6))
print(KMeans_label_2016.tolist().count(7))
print(KMeans_label_2016.tolist().count(8))
print(KMeans_label_2016.tolist().count(9))
print("2017")
KMeans_label_2017 = KMeans_label[1064345:1111855+1]
KMeans_label_2017 = KMeans_label_2017.iloc[:,0]
print(KMeans_label_2017.tolist().count(0))
print(KMeans_label_2017.tolist().count(1))
print(KMeans_label_2017.tolist().count(2))
print(KMeans_label_2017.tolist().count(3))
print(KMeans_label_2017.tolist().count(4))
print(KMeans_label_2017.tolist().count(5))
print(KMeans_label_2017.tolist().count(6))
print(KMeans_label_2017.tolist().count(7))
print(KMeans_label_2017.tolist().count(8))
print(KMeans_label_2017.tolist().count(9))
print("2018")
KMeans_label_2018 = KMeans_label[1111856:1151957+1]
KMeans_label_2018 = KMeans_label_2018.iloc[:,0]
print(KMeans_label_2018.tolist().count(0))
print(KMeans_label_2018.tolist().count(1))
print(KMeans_label_2018.tolist().count(2))
print(KMeans_label_2018.tolist().count(3))
print(KMeans_label_2018.tolist().count(4))
print(KMeans_label_2018.tolist().count(5))
print(KMeans_label_2018.tolist().count(6))
print(KMeans_label_2018.tolist().count(7))
print(KMeans_label_2018.tolist().count(8))
print(KMeans_label_2018.tolist().count(9))
exit()
'''

train_X, test_X, train_y, test_y = train_test_split(TSNE_Model,  KMeans_label, test_size=0.15, shuffle=True, random_state = 0)


print("Start KNN")
KNN_Model = KNeighborsClassifier(n_neighbors=7, n_jobs = 3)
print("Fit KNN")
KNN_Model.fit(train_X, train_y)
dump(KNN_Model, 'KNN_Model.joblib')
print("Predict KNN")
KNN_pred = KNN_Model.predict(test_X)
savetxt('KNN_pred.csv', KNN_pred, delimiter=',')
print(classification_report(test_y, KNN_pred))
print(confusion_matrix(test_y, KNN_pred))

RF_Model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_features='auto', oob_score=True, random_state=42, verbose=2, warm_start=False, n_jobs = 4)
print("Fit RF_Model")
RF_Model.fit(train_X, train_y)         
dump(RF_Model, 'RF_Model.joblib')      
print("Predict RF_Model")
RF_pred = RF_Model.predict(test_X)
savetxt('RF_pred.csv', RF_pred, delimiter=',')
print(classification_report(test_y, RF_pred))
print(confusion_matrix(test_y, RF_pred))

#Cross validation of XGB
'''
kf = KFold(n_splits=20)
XGB_Model = xg.XGBClassifier( verbosity = 1,  random_state = 42, n_jobs = 4)
print("Fit XGB_Model")
results = cross_val_score(XGB_Model, TSNE_Model, KMeans_label, cv=kf)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
exit()
'''

XGB_Model = xg.XGBClassifier( verbosity = 1,  random_state = 42, n_jobs = 4)
print("Fit XGB_Model")
XGB_Model.fit(train_X, train_y)
dump(XGB_Model, 'XGB_Model.joblib')
print("Predict XGB_Model")
XGB_pred = XGB_Model.predict(test_X)
savetxt('XGB_pred.csv', XGB_pred, delimiter=',')
print(classification_report(test_y, XGB_pred))
print(confusion_matrix(test_y, XGB_pred))

XGBRF_Model = xg.XGBRFClassifier( verbosity =2, random_state = 42, n_jobs = 3)
print("Fit XGBRF_Model")
XGBRF_Model.fit(train_X, train_y)
dump(XGBRF_Model, 'XGBRF_Model.joblib')
print("Predict XGBRF_Model")
print("Predict XGBRF_Model")
XGBRF_pred = XGBRF_Model.predict(test_X)
savetxt('XGBRF_pred.csv', XGBRF_pred, delimiter=',')
print(classification_report(test_y, XGBRF_pred))
print(confusion_matrix(test_y, XGBRF_pred))

print("Create ADA_Model")
ADA_Model = AdaBoostClassifier(random_state = 42, n_estimators = 100)
print("Fit ADA_Model")
ADA_Model.fit(train_X, train_y)
dump(ADA_Model, 'ADA_Model.joblib')
ADA_pred = ADA_Model.predict(test_X)
savetxt('ADA_pred.csv', ADA_pred, delimiter=',')
print(classification_report(test_y, ADA_pred))
print(confusion_matrix(test_y, ADA_pred))

print("Create BAG_Model")
BAG_Model = BaggingClassifier(base_estimator=SVC(), n_jobs = 4, random_state = 42, verbose = 2)
print("Fit BAG_Model")
BAG_Model.fit(train_X, train_y)
dump(BAG_Model, 'BAG_Model.joblib')
BAG_pred = BAG_Model.predict(test_X)
savetxt('BAG_pred.csv', BAG_pred, delimiter=',')
print(classification_report(test_y, BAG_pred))
print(confusion_matrix(test_y, BAG_pred))

print("Create GBC_Model")
GBC_Model = GradientBoostingClassifier(random_state = 42, verbose = 2) 
print("Fit GBC_Model")
GBC_Model.fit(train_X, train_y)
dump(GBC_Model, 'GBC_Model.joblib')
GBC_pred = GBC_Model.predict(test_X)
savetxt('GBC_pred.csv', GBC_pred, delimiter=',')
print(classification_report(test_y, GBC_pred))
print(confusion_matrix(test_y, GBC_pred))

print("Create HGBC_Model")
HGBC_Model = HistGradientBoostingClassifier(random_state = 42, verbose = 2) 
print("Fit HGBC_Model")
HGBC_Model.fit(train_X, train_y)
dump(HGBC_Model, 'HGBC_Model.joblib')
HGBC_pred = HGBC_Model.predict(test_X)
savetxt('HGBC_pred.csv', HGBC_pred, delimiter=',')
print(classification_report(test_y, HGBC_pred))
print(confusion_matrix(test_y, HGBC_pred))

print("Create SVC_Model")
SVC_Model = SVC(random_state = 42, verbose = 2) 
print("Fit SVC_Model")
SVC_Model.fit(train_X, train_y)
dump(SVC_Model, 'SVC_Model.joblib')
SVC_pred = SVC_Model.predict(test_X)
savetxt('SVC_pred.csv', SVC_pred, delimiter=',')
print(classification_report(test_y, SVC_pred))
print(confusion_matrix(test_y, SVC_pred))

