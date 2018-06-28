# 0.2197
#Initially forked from Bojan's kernel here: https://www.kaggle.com/tunguz/bow-meta-text-and-dense-features-lb-0-2242/code
#improvement using kernel from Nick Brook's kernel here: https://www.kaggle.com/nicapotato/bow-meta-text-and-dense-features-lgbm
#Used oof method from Faron's kernel here: https://www.kaggle.com/mmueller/stacking-starter?scriptVersionId=390867
#Used some text cleaning method from Muhammad Alfiansyah's kernel here: https://www.kaggle.com/muhammadalfiansyah/push-the-lgbm-v19
#Forked From - https://www.kaggle.com/him4318/avito-lightgbm-with-ridge-feature-v-2-0

import time
notebookstart= time.time()

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
print("Data:\n",os.listdir("../input"))

# Models Packages
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn import feature_selection
from sklearn.model_selection import train_test_split, KFold
from sklearn import cross_validation 

from sklearn import preprocessing

# Gradient Boosting
import lightgbm as lgb
from sklearn.linear_model import Ridge

# Tf-Idf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

# Viz
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

NFOLDS = 10
SEED = 414
VALID = False
DEBUG=0

class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool = True):
        if(seed_bool == True):
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
        
def get_oof(clf, x_train, y, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"
    
    
def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))


def to_number(x):
    try:
        if not x.isdigit():
            return 0
        x = int(x)
        if x > 100:
            return 100
        else:
            return x
    except:
        return 0

def sum_numbers(desc):
    if not isinstance(desc, str):
        return 0
    try:
        return sum([to_number(s) for s in desc.split()])
    except:
        return 0


print("\nData Load Stage")

if DEBUG:
    training = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
    training = training.sample(200000)
    traindex = training.index
    testing = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"], nrows=10000)
    testdex = testing.index
else:
    training = pd.read_csv('../input/train.csv', index_col = "item_id", parse_dates = ["activation_date"])
    traindex = training.index
    testing = pd.read_csv('../input/test.csv', index_col = "item_id", parse_dates = ["activation_date"])
    testdex = testing.index

ntrain = training.shape[0]
ntest = testing.shape[0]

kf = cross_validation.KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

y = training.deal_probability.copy()
training.drop("deal_probability",axis=1, inplace=True)
print('Train shape: {} Rows, {} Columns'.format(*training.shape))
print('Test shape: {} Rows, {} Columns'.format(*testing.shape))

print("Combine Train and Test")
df = pd.concat([training,testing],axis=0)
del training, testing
gc.collect()

categorical = ["region","city","parent_category_name",
"category_name","user_type","image_top_1","param_1","param_2","param_3"]

df[categorical] = df[categorical].fillna('NAN')
#***************************************************************************

df.reset_index(inplace=True)

# blurness
img_tr = pd.read_csv('train_blurrness.csv')
img_te = pd.read_csv('test_blurrness.csv')
img_fe = pd.concat([img_tr,img_te],axis=0)
df = df.merge(img_fe, on='item_id',how='left')
del img_tr,img_te,img_fe; gc.collect()
# keypoint
img_fe = pd.read_csv('img_features_keypoint_good.csv')
df = df.merge(img_fe, on='image',how='left')
del img_fe; gc.collect()

# NiMa
img_tr = pd.read_csv('train_mobilenet_score.csv')
img_te = pd.read_csv('test_mobilenet_score.csv')
img_fe = pd.concat([img_tr,img_te],axis=0)
df = df.merge(img_fe, on='image', how='left')
del img_tr, img_te, img_fe; gc.collect()

img_tr = pd.read_csv('train_inception_resnet_score.csv')
img_te = pd.read_csv('test_inception_resnet_score.csv')
img_fe = pd.concat([img_tr,img_te],axis=0)
df = df.merge(img_fe, on='image', how='left')
del img_tr, img_te, img_fe; gc.collect()

# mobile-net
# 
img_fe = pd.read_csv('mobilenet_features.csv')
img_fe.dropna(inplace=True)
img_fe.set_index('image',inplace=True)
img_fe = img_fe.astype(np.float32).reset_index()
df = df.merge(img_fe, on='image',how='left')
del img_fe; gc.collect()

# res-net
# 
img_fe = pd.read_csv('resnet_features.csv')
img_fe.dropna(inplace=True)
img_fe.set_index('image',inplace=True)
img_fe = img_fe.astype(np.float32).reset_index()
df = df.merge(img_fe, on='image',how='left')
del img_fe; gc.collect()

# inceptionv3-net
# 
img_fe = pd.read_csv('inceptionv3_features.csv')
img_fe.dropna(inplace=True)
img_fe.set_index('image',inplace=True)
img_fe = img_fe.astype(np.float32).reset_index()
df = df.merge(img_fe, on='image',how='left')
del img_fe; gc.collect()

# user_id
gp = pd.read_csv('aggregated_features_ben.csv')
print(gp.info())
print(df.info())
df = df.merge(gp, on='user_id', how='left')
del gp; gc.collect()

# city
gp = pd.read_csv('city_features.csv')
df = df.merge(gp, on='city', how='left')
del gp; gc.collect()

# activate date
gp = pd.read_csv('activation_date_features.csv', parse_dates=['activation_date'])
df = df.merge(gp, on='activation_date', how='left')
del gp; gc.collect()

# user type
gp = pd.read_csv('user_type_features.csv')
df = df.merge(gp, on='user_type', how='left')
del gp; gc.collect()

# item seq number
gp = pd.read_csv('item_seq_number_features.csv')
df = df.merge(gp, on='item_seq_number', how='left')
del gp; gc.collect()

# city population
city_df = pd.read_csv('city_population.csv',usecols=[1,2])
df = df.merge(city_df, on='city', how='left')
# df['population'] = df['population'].fillna(df['population'].mean())
del city_df; gc.collect()

# image feature
print('image_feature')
image_tr = pd.read_csv('../image-features-agg/train_image_features-header.csv')
image_te = pd.read_csv('../image-features-agg/test_image_features-header.csv')
image_fe = pd.concat([image_tr,image_te], ignore_index=True)
image_fe['image'] = image_fe['image'].apply(lambda x: x[:-4])
df = df.merge(image_fe, on='image', how='left')
del image_fe,image_te,image_tr; gc.collect()


#***************************************************
# City names are duplicated across region, HT: Branden Murray 
#https://www.kaggle.com/c/avito-demand-prediction/discussion/55630#321751
df['city'] = df['city'] + "_" + df['region']
cat_cols = ['category_name', 'image_top_1', 'parent_category_name', 'param_1', 'param_2',
            'param_3','city','user_type']
num_cols = ['price']

for c in cat_cols:
    for c2 in num_cols:
        print(c)
        enc = df.groupby(c)[c2].agg(['median']).astype(np.float32).reset_index()
        enc.columns = ['_'.join([str(c), str(c2), str(c3)]) if c3 != c else c for c3 in enc.columns]
        df = pd.merge(df, enc, how='left', on=c)


bi_cols = [['category_name','param_1'],['category_name','city'],['category_name','user_type'],
           ['parent_category_name','param_1'],['parent_category_name','city'],['parent_category_name','user_type'],
           ['param_1','param_2'],['param_1','param_3'],['param_2','param_3'],['param_1','city'],['param_1','user_type'],
           ['param_2','user_type'],['param_3','user_type'],
           ['city','param_1','param_2'],['param_1','param_3','city'],['param_2','param_3','city'],
           ['user_type','param_1','param_2'],['param_1','param_3','user_type'],['param_2','param_3','user_type'],
           ['param_1','city','user_type','category_name'],
           ['param_2','city','user_type','category_name'],['param_3','city','user_type','category_name'],
           ['image_top_1','city']]


for c in bi_cols:
    enc = df.groupby(c)['price'].agg(['median']).astype(np.float32).reset_index()
    enc.columns = c+['_'.join(c)+"_price_median"]
    #enc = enc.reset_index()
    df = pd.merge(df, enc, how='left', on=c)

#***************************************************

df.set_index('item_id',inplace=True)
#***************************************************************************

print('\nAll Data shape: {} Rows, {} Columns'.format(*df.shape))


print("Feature Engineering")
df["price"] = np.log1p(df["price"])
df["price"].fillna(df.price.mean(),inplace=True)
#df["image_top_1"].fillna(-999,inplace=True)

print("\nCreate Time Variables")
df["Weekday"] = df['activation_date'].dt.weekday


df.drop(["activation_date","image",'user_id'],axis=1,inplace=True)

print("\nEncode Variables")

print("Encoding :",categorical)

# Encoder:
lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col].fillna('Unknown')
    df[col] = lbl.fit_transform(df[col].astype(str))
    
print("\nText Features")

# Feature Engineering 

# Meta Text Features
textfeats = ["description", "title"]
russian_stop = set(stopwords.words('russian'))

for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Words
    df[cols + '_num_letters'] = df[cols].apply(lambda comment: len(comment)) # Count number of Letters
    df[cols + '_letters_per_word'] = df[cols + '_num_letters'] / df[cols + '_num_words']
    df[cols + '_mean'] = df[cols].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10
    df[cols + '_punc_count'] = df[cols].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df[cols + '_punc_count_ratio'] = df[cols+'_punc_count']/df[cols+'_num_words']
    df[cols + '_digit_count'] = df[cols].str.count('[0-9]')
    df[cols + '_digit_count_ratio'] = df[cols + '_digit_count']/df[cols+'_num_words']
    df[cols + '_stopword_ratio'] = df[cols].apply(
        lambda x: len([w for w in x.split() if w in russian_stop])) / df[cols+'_num_words']

df['description_num_sum'] = df['description'].apply(sum_numbers) 


# Extra Feature Engineering
df['title_desc_len_ratio'] = df['title_num_letters']/df['description_num_letters']


df[['param_1','param_2','param_3','title',
    'parent_category_name','category_name',
    'region','city','user_type']] = \
        df[['param_1','param_2','param_3','title',
            'parent_category_name','category_name',
            'region','city','user_type']].fillna('NAN')

df['title'] = df.apply(lambda x: \
    x['title']+' '+str(x['param_1'])+' '+str(x['param_2'])+' '+str(x['param_3'])+' '\
        +str(x['parent_category_name'])+' '+str(x['category_name'])+' '\
        +str(x['region'])+' '+str(x['city'])+' '+str(x['user_type']),axis=1)

df["description"]   = df["description"].apply(lambda x: cleanName(x))
df['description'] = df.apply(lambda x: \
    x['title']+' '+str(x['description']),axis=1)

print("\n[TF-IDF] Term Frequency Inverse Document Frequency Stage")

tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    #"min_df":5,
    #"max_df":.9,
    "smooth_idf":False
}


def get_col(col_name): return lambda x: x[col_name]
##I added to the max_features of the description. It did not change my score much but it may be worth investigating
vectorizer = FeatureUnion([
        ('description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=40000,
            **tfidf_para,
            preprocessor=get_col('description'))),
        ('title',CountVectorizer(
            #min_df = 10,
            #max_df = .9,
            stop_words = russian_stop,
            max_features=20000,
            preprocessor=get_col('title')))
    ])
    
start_vect=time.time()

#Fit my vectorizer on the entire dataset instead of the training rows
#Score improved by .0001
vectorizer.fit(df.to_dict('records'))

ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()
print("Vectorization Runtime: %0.2f Minutes"%((time.time() - start_vect)/60))

# Drop Text Cols
textfeats = ["description", "title"]
df.drop(textfeats, axis=1,inplace=True)

from sklearn.metrics import mean_squared_error
from math import sqrt

ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}

#Ridge oof method from Faron's kernel
#I was using this to analyze my vectorization, but figured it would be interesting to add the results back into the dataset
#It doesn't really add much to the score, but it does help lightgbm converge faster
ridge = SklearnWrapper(clf=Ridge, seed = SEED, params = ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, ready_df[:ntrain], y, ready_df[ntrain:])

rms = sqrt(mean_squared_error(y, ridge_oof_train))
print('Ridge OOF RMSE: {}'.format(rms))

print("Modeling Stage")

ridge_preds = np.concatenate([ridge_oof_train, ridge_oof_test])

df['ridge_preds'] = ridge_preds




# Combine Dense Features with Sparse Text Bag of Words Features
X = hstack([csr_matrix(df.loc[traindex,:].values),ready_df[0:traindex.shape[0]]]) # Sparse Matrix
testing = hstack([csr_matrix(df.loc[testdex,:].values),ready_df[traindex.shape[0]:]])
tfvocab = df.columns.tolist() + tfvocab
for shape in [X,testing]:
    print("{} Rows and {} Cols".format(*shape.shape))
print("Feature Names Length: ",len(tfvocab))
del df
gc.collect();

print("\nModeling Stage")

del ridge_preds,vectorizer,ready_df
gc.collect();
    
print("Light Gradient Boosting Regressor")
lgbm_params =  {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    # 'max_depth': 15,
    'num_leaves': 400,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.75,
    'bagging_freq': 2,
    'learning_rate': 0.016,
    'verbose': 0,
    'random_seed': SEED,
    'num_thread': 6
}  


if VALID == True:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.10, random_state=23)
        
    # LGBM Dataset Formatting 
    lgtrain = lgb.Dataset(X_train, y_train,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    lgvalid = lgb.Dataset(X_valid, y_valid,
                    feature_name=tfvocab,
                    categorical_feature = categorical)
    del X, X_train; gc.collect()
    
    # Go Go Go
    lgb_clf = lgb.train(
        lgbm_params,
        lgtrain,
        num_boost_round=20000,
        valid_sets=[lgtrain, lgvalid],
        valid_names=['train','valid'],
        early_stopping_rounds=50,
        verbose_eval=100
    )
    print("Model Evaluation Stage")
    print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
    del X_valid ; gc.collect()
    # Feature Importance Plot
    f, ax = plt.subplots(figsize=[7,10])
    lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
    plt.title("Light GBM Feature Importance")
    plt.savefig('feature_import.png')

    print("Model Evaluation Stage")
    lgpred = lgb_clf.predict(testing) 

    #Mixing lightgbm with ridge. I haven't really tested if this improves the score or not
    #blend = 0.95*lgpred + 0.05*ridge_oof_test[:,0]
    lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
    lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
    lgsub.to_csv("lgsub.csv",index=True,header=True)
    #print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
    print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))

else:

    lgpred = np.empty((NFOLDS,(testing.shape[0])))
    kf = KFold(n_splits=NFOLDS,shuffle=True,random_state=SEED)
    idx=0
    for train_index, test_index in kf.split(X):
        X_train, X_valid = X.tocsr()[train_index,:], X.tocsr()[test_index,:]
        y_train, y_valid = y[train_index], y[test_index]
        
        # LGBM Dataset Formatting 
        lgtrain = lgb.Dataset(X_train, y_train,
                        feature_name=tfvocab,
                        categorical_feature = categorical)
        lgvalid = lgb.Dataset(X_valid, y_valid,
                        feature_name=tfvocab,
                        categorical_feature = categorical)
        
        
        # Go Go Go
        lgb_clf = lgb.train(
            lgbm_params,
            lgtrain,
            num_boost_round=20000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=['train','valid'],
            early_stopping_rounds=300,
            verbose_eval=100
        )
        print("Model Evaluation Stage")
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_valid, lgb_clf.predict(X_valid))))
        lgpred[idx,:] = lgb_clf.predict(testing) 
        idx += 1
        del X_train, X_valid ; gc.collect()

        # Feature Importance Plot
        # f, ax = plt.subplots(figsize=[7,10])
        # lgb.plot_importance(lgb_clf, max_num_features=50, ax=ax)
        # plt.title("Light GBM Feature Importance")
        # plt.savefig('feature_import.png')

    lgpred = lgpred.mean(axis=0)
    print("Model Evaluation Stage")
    del X; gc.collect()

    #Mixing lightgbm with ridge. I haven't really tested if this improves the score or not
    #blend = 0.95*lgpred + 0.05*ridge_oof_test[:,0]
    lgsub = pd.DataFrame(lgpred,columns=["deal_probability"],index=testdex)
    lgsub['deal_probability'].clip(0.0, 1.0, inplace=True) # Between 0 and 1
    lgsub.to_csv("lgsub_{}fold-0627.csv".format(NFOLDS),index=True,header=True)
    #print("Model Runtime: %0.2f Minutes"%((time.time() - modelstart)/60))
    print("Notebook Runtime: %0.2f Minutes"%((time.time() - notebookstart)/60))



