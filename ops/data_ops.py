'''

   Operations commonly used in tensorflow

'''

import tensorflow as tf
import numpy as np
import math
import glob
import scipy.misc as misc
import ntpath
from tqdm import tqdm
import os
import requests
import gzip
import cPickle as pickle
import gzip
#import mnist_reader

'''
   Helper function that returns string names for the attributes

   0     1      2           3           4           5           6       7          8             9     10        11        12       13           14
   bald, bangs, black_hair, blond_hair, brown_hair, eyeglasses, goatee, gray_hair, heavy_makeup, male, mustache, no_beard, smiling, wearing_hat, wearing_necklace
   4,5,8,9,11,15,16,17,18,20,22,24,31,35,37

'''
def get_attr_name(attr):
   s = ''
   if attr[0] == -1: s+='not bald, '
   else: s+='bald, '
   
   if attr[1] == -1: s+='no bangs, '
   else: s+='bangs, '
   
   if attr[2] == -1: s+='no black hair, '
   else: s+='black hair, '
   
   if attr[3] == -1: s+='no blonde hair, '
   else: s+='black hair, '
   
   if attr[4] == -1: s+='no brown hair, '
   else: s+='brown hair, '
   
   if attr[5] == -1: s+='no eyeglasses, '
   else: s+='eyeglasses, '
   
   if attr[6] == -1: s+='no goatee, '
   else: s+='goatee, '
   
   if attr[7] == -1: s+='no gray hair, '
   else: s+='gray hair, '
   
   if attr[8] == -1: s+='no heavy makeup, '
   else: s+='heavy makeup, '
   
   if attr[9] == -1: s+='not male, '
   else: s+='male, '
   
   if attr[10] == -1: s+='no mustache, '
   else: s+='mustache, '
   
   if attr[11] == -1: s+='beard, '
   else: s+='no beard, '
   
   if attr[12] == -1: s+='not smiling, '
   else: s+='smiling, '
   
   if attr[13] == -1: s+='not wearing hat, '
   else: s+='wearing hat, '
   
   if attr[14] == -1: s+='not wearing necklace, '
   else: s+='wearing necklace'
   
   return s


'''

   Loading up the galaxy dataset. Only considering the following:
   
   1  T               - EFIGI morphological type
   7  Arm strength    - Strength of spiral arms
   10 Arm curvature   - Average curvature of the spiral arms
   13 Arm Rotation    - Direction of the winding of the spiral arms
   16 Bar length      - Length of the central bar
   19 Inner Ring      - Strength of the inner ring, lens or inner pseudo-ring
   22 Outer Ring      - Strength of outer ring
   25 Pseudo Ring     - Type and strength of outer pseudo-ring
   28 Perturbation    - Deviation from rotational symmetry
   31 Visible Dust    - Strength of dust features
   34 Dust Dispersion - Patchiness of dust features
   40 Hot Spots       - Strength of regions of strong star formation, active nuclei, or stellar nuclie
   49 Multiplicity    - Abundance of neighbouring galaxies

'''
def load_galaxy(data_dir):

   idx = np.array([0, 1, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 40, 49])

   train_images     = glob.glob(data_dir+'images/train/*.png')
   test_images      = glob.glob(data_dir+'images/test/*.png')

   # get train ids from train folder
   train_ids = [ntpath.basename(x.split('.')[0]) for x in train_images]
   test_ids  = [ntpath.basename(x.split('.')[0]) for x in test_images]

   iptr = data_dir+'images/train/'
   ipte = data_dir+'images/test/'

   train_images = []
   test_images  = []
   train_attributes = []
   test_attributes  = []

   paths = []
   with open(data_dir+'EFIGI_attributes.txt', 'r') as f:
      for line in f:
         line     = line.rstrip().split()
         image_id = line[0]
         line     = np.asarray(line[1:])
         line     = line[idx].astype('float32')
         if image_id in train_ids:
            img = misc.imread(iptr+image_id+'.png').astype('float32')
            img = misc.imresize(img, (64, 64))
            img = normalize(img)
            train_images.append(img)
            train_attributes.append(line)
         elif image_id in test_ids:
            paths.append(ipte+image_id+'.png')
            img = misc.imread(ipte+image_id+'.png').astype('float32')
            img = misc.imresize(img, (64, 64))
            img = normalize(img)
            test_images.append(img)
            test_attributes.append(line)

   train_images = np.asarray(train_images)
   test_images = np.asarray(test_images)
   train_attributes = np.asarray(train_attributes)
   test_attributes = np.asarray(test_attributes)

   return train_images, train_attributes, test_images, test_attributes, paths

def load_mnist(data_dir, mode='train'):

   url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
   # check if it's already downloaded
   if not os.path.isfile(data_dir+'/mnist.pkl.gz'):
      print 'Downloading mnist...'
      with open('mnist.pkl.gz', 'wb') as f:
         r = requests.get(url)
         if r.status_code == 200:
            f.write(r.content)
         else:
            print 'Could not connect to ', url

   print 'opening mnist'
   f = gzip.open('mnist.pkl.gz', 'rb')
   train_set, val_set, test_set = pickle.load(f)
   
   if mode == 'train':
      mnist_train_images = []
      mnist_train_labels = []
      for t,l in zip(*train_set):
         label = np.zeros((10))
         label[l] = 1
         mnist_train_images.append(np.reshape(t, (28, 28, 1)))
         mnist_train_labels.append(label)
      return np.asarray(mnist_train_images), np.asarray(mnist_train_labels)

   if mode == 'val':
      mnist_val_images = []
      mnist_val_labels = []
      for t,l in zip(*val_set):
         label = np.zeros((10))
         label[l] = 1
         mnist_val_images.append(np.reshape(t, (28, 28, 1)))
         mnist_val_labels.append(label)
      return np.asarray(mnist_val_images), np.asarray(mnist_val_labels)

   if mode == 'test':
      mnist_test_images  = []
      mnist_test_labels = []
      for t,l in zip(*test_set):
         label = np.zeros((10))
         label[l] = 1
         mnist_test_images.append(np.reshape(t, (28, 28, 1)))
         mnist_test_labels.append(label)
      return np.asarray(mnist_test_images), np.asarray(mnist_test_labels)

   return 'mode error'


def load_fashion(data_dir):

   X_train, y_train = mnist_reader.load_mnist(data_dir, kind='train')
   X_test, y_test = mnist_reader.load_mnist(data_dir, kind='t10k')

   train_images = []
   test_images  = []
   train_labels = []
   test_labels = []

   for t,l in zip(X_train, y_train):
      label = np.zeros((10))
      label[l] = 1
      train_labels.append(label)
      train_images.append(np.reshape(t, (28, 28, 1)))
   for t,l in zip(X_test, y_test):
      label = np.zeros((10))
      label[l] = 1
      test_labels.append(label)
      test_images.append(np.reshape(t, (28, 28, 1)))

   return np.asarray(train_images), np.asarray(train_labels), np.asarray(test_images), np.asarray(test_labels)


'''
   mode can be train/test/val
'''
def load_celeba(data_dir, mode='train', filters=[4, 5, 8, 9, 15, 18, 20, 26, 31]):
   # load up annotations
   '''
      0  5_o_Clock_Shadow
      1  Arched_Eyebrows
      2  Attractive
      3  Bags_Under_Eyes
      4  Bald
      5  Bangs
      6  Big_Lips
      7  Big_Nose
      8  Black_Hair
      9  Blond_Hair
      10 Blurry
      11 Brown_Hair
      12 Bushy_Eyebrows
      13 Chubby
      14 Double_Chin
      15 Eyeglasses
      16 Goatee
      17 Gray_Hair
      18 Heavy_Makeup
      19 High_Cheekbones
      20 Male
      21 Mouth_Slightly_Open
      22 Mustache
      23 Narrow_Eyes
      24 No_Beard
      25 Oval_Face
      26 Pale_Skin
      27 Pointy_Nose
      28 Receding_Hairline
      29 Rosy_Cheeks
      30 Sideburns
      31 Smiling
      32 Straight_Hair
      33 Wavy_Hair
      34 Wearing_Earrings
      35 Wearing_Hat
      36 Wearing_Lipstick
      37 Wearing_Necklace
      38 Wearing_Necktie
      39 Young

      only considering: bald, bangs, black_hair, blond_hair, eyeglasses, heavy_makeup, male, pale_skin, smiling
      4, 5, 8, 9, 15, 18, 20, 26, 31
   '''
   dum = 0
   train_image_attr = {}
   i = 0
   print 'Loading attributes...'
   with open(data_dir+'list_attr_celeba.txt', 'r') as f:
      for line in tqdm(f):
         line = line.rstrip().split()
         if dum < 2:
            dum += 1
            continue
         image_id = line[0]
         attr = line[1:]
         attr = np.asarray(list(attr[x] for x in filters), dtype=np.float32)
         attr = np.asarray([0 if x == -1 else 1 for x in attr])
         train_image_attr[data_dir+'img_align_celeba/'+image_id] = attr

         i += 1

   train_images = train_image_attr.keys()
   train_attrs  = train_image_attr.values()
   '''
   if mode == 'train':
      train_images = train_image_attr.keys()
      train_attrs  = train_image_attr.values()
      return np.asarray(train_images), np.asarray(train_attrs)

   if mode == 'test':
      test_images = test_image_attr.keys()
      test_attrs  = test_image_attr.values()
      return np.asarray(test_images), np.asarray(test_attrs)
   '''
   return np.asarray(train_images), np.asarray(train_attrs)

def normalize(image):
   return (image/127.5)-1.0

'''
   Converts a single image from [0,255] range to [-1,1]
'''
def preprocess(image):
   with tf.name_scope('preprocess'):
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      return (image/127.5)-1.0

'''
   Converts a single image from [-1,1] range to [0,255]
'''
def deprocess(image):
   with tf.name_scope('deprocess'):
      return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)
   
'''
   Converts a batch of images from [-1,1] range to [0,255]
'''
def batch_deprocess(images):
   with tf.name_scope('batch_deprocess'):
      return tf.map_fn(deprocess, images, dtype=tf.uint8)

'''
   Converts a batch of images from [0,255] to [-1,1]
'''
def batch_preprocess(images):
   with tf.name_scope('batch_preprocess'):
      return tf.map_fn(preprocess, images, dtype=tf.float32)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return misc.imread(path, flatten=True).astype(np.float)
    else:
        return misc.imread(path).astype(np.float)

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return misc.imsave(path, image)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image
    return img

def inverse_transform(image):
    return (image + 1.) / 2.
