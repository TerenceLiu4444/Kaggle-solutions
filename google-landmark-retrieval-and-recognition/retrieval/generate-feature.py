import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import StringIO
import tensorflow as tf
import tensorflow_hub as hub
from six.moves.urllib.request import urlopen
import os
import gc
import pickle



def image_input_fn(JPG_LIST):
  filename_queue = tf.train.string_input_producer(
      JPG_LIST, shuffle=False)
  reader = tf.WholeFileReader()
  _, value = reader.read(filename_queue)
  image_tf = tf.image.decode_jpeg(value, channels=3)
  # print(image_tf.shape)

  # image_tf_resize = tf.image.resize_images(image_tf, [224, 224])
  # print(image_tf_resize.shape)
  return tf.image.convert_image_dtype(image_tf, tf.float32)


file_id = 1
BATCH_SIZE = 10000


# filelist_df = pd.read_csv('../input/each_class_pick_12_to_100.csv')
# filelist = ['../input/train/'+x+'.jpg' for x in filelist_df['id'].values]
# filelist = [x for x in filelist if os.path.isfile(x)]

# JPG_TRAIN_LIST = filelist

# #sanity check first
# JPG_LIST = JPG_TRAIN_LIST


with open('../input/test_landmarks_884.csv') as f:
  file_list1 = f.readlines()

with open('../input/test_landmarks_1919.csv') as f:
  file_list2 = f.readlines()

with open('../input/test_landmarks_batch3_1218.csv') as f:
  file_list3 = f.readlines()

with open('../input/test_landmarks_batch4_406.csv') as f:
  file_list4 = f.readlines()

with open('../input/test_landmarks_batch5_1111.csv') as f:
  file_list5 = f.readlines()


filelist = file_list1+file_list2+file_list3+file_list4+file_list5
filelist = ['../input/test/'+x[:-1] for x in filelist]
JPG_LIST = [x for x in filelist if os.path.isfile(x)]


# filelist = ['../input/train/'+x+'.jpg' for x in filelist_df['id'].values]
# filelist = [x for x in filelist if os.path.isfile(x)]

# JPG_TRAIN_LIST = filelist

# #sanity check first
# JPG_LIST = JPG_TRAIN_LIST


tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.FATAL)

m = hub.Module('https://tfhub.dev/google/delf/1')

# The module operates on a single image at a time, so define a placeholder to
# feed an arbitrary image in.
image_placeholder = tf.placeholder(
    tf.float32, shape=(None, None, 3), name='input_image')

module_inputs = {
    'image': image_placeholder,
    'score_threshold': 100.0,
    'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
    'max_feature_num': 1000,
}


module_outputs = m(module_inputs, as_dict=True)

image_tf = image_input_fn(JPG_LIST)



file_id = 1
with tf.train.MonitoredSession() as sess:
  results_dict = {}  # Stores the locations and their descriptors for each image
  for image_path in tqdm(JPG_LIST):
    image = sess.run(image_tf)
    #print(image_tf.shape)
    #print('Extracting locations and descriptors from %s' % image_path)
    results_dict[image_path] = sess.run(
        [module_outputs['locations'], module_outputs['descriptors']],
        feed_dict={image_placeholder: image})
    if len(results_dict) == BATCH_SIZE:
        output = open('features/results_dict_test_{}.pkl'.format(file_id), 'wb')
        pickle.dump(results_dict, output)
        output.close()
        results_dict.clear()
        gc.collect()
        file_id += 1

output = open('features/results_dict_test_{}.pkl'.format(file_id), 'wb')
pickle.dump(results_dict, output)
output.close()
