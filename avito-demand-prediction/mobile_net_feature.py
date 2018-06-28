from keras.applications.mobilenet import MobileNet

from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
from sklearn.decomposition import PCA
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
import numpy as np
import keras
from tqdm import tqdm
import gc

import os
import pandas as pd

use_cols = ['image','image_top_1']
df_train = pd.read_csv('../input/train.csv',
                 usecols=use_cols)#,nrows=10000)
df_train.dropna(subset=['image'],inplace=True)
df_train['image'] = df_train.image.map(lambda x : '../input/train_jpg/'+x+'.jpg')
df_train = df_train.loc[df_train['image'].apply(
    lambda x: isinstance(x,str) and os.path.isfile(x)),:]

use_cols = ['image','image_top_1']
df_test = pd.read_csv('../input/test.csv',
                 usecols=use_cols)#,nrows=10000)
df_test.dropna(subset=['image'],inplace=True)
df_test['image'] = df_test.image.map(lambda x : '../input/test_jpg/'+x+'.jpg')
df_test = df_test.loc[df_test['image'].apply(
    lambda x: isinstance(x,str) and os.path.isfile(x)),:]

df = pd.concat([df_train,df_test],ignore_index=True)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

batch_size=8092

# resnet
model = MobileNet(weights='imagenet', include_top=False, pooling='avg')

total_features = []
for img_paths in tqdm(chunks(df.image,batch_size),
                      total=len(list(chunks(df.image,batch_size)))):
    imgs = [image.load_img(img_path, target_size=(224, 224)) for img_path in img_paths]
    # convert image to numpy array
    xs = [image.img_to_array(img) for img in imgs]
    # the image is now in an array of shape (3, 224, 224) 
    # need to expand it to (1, 3, 224, 224) as it's expecting a list
    xs = np.concatenate([np.expand_dims(x, axis=0) for x in xs])
    xs = preprocess_input(xs)
    # extract the features
    features = model.predict(xs)
    total_features.append(features)
    # convert from Numpy to a list of values
    #features_arr = np.char.mod('%f', features)

pca = PCA(n_components=40)
resnet_fea = pca.fit_transform(np.concatenate(total_features))
del pca,model,total_features;gc.collect()


resnet_cols = ['mobilenet_'+str(i) for i in range(40)]
resnet_df = pd.DataFrame(resnet_fea, columns=resnet_cols, index=df.index)
resnet_df['image'] = df['image'].map(lambda x: x[(x.rfind('/')+1):-4])
resnet_df.to_csv('mobilenet_features.csv',index=None)