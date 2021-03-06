'''
   conditional gan
'''
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import scipy.misc as misc
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np
import argparse
import random
import ntpath
import time
import sys
import os

sys.path.insert(0, '../ops/')

from tf_ops import *
import data_ops
from nets import *

if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--loss',          required=False,help='Type of GAN loss to use', type=str,default='wgan')
   parser.add_argument('--dataset',       required=False,help='The DATASET to use',      type=str,default='celeba')
   parser.add_argument('--data_dir',      required=False,help='Directory where data is', type=str,default='./data/')
   parser.add_argument('--image_dir',     required=False,help='Directory where image is',type=str,default='./samples/')
   parser.add_argument('--log_dir',       required=False,help='Directory where logs is', type=str,default='./logs/')
   parser.add_argument('--checkpoint_dir',required=False,help='Directory where checkpoint is', type=str,default='./checkpoint/')
   parser.add_argument('--epochs',        required=False,help='Maximum training steps',  type=int,default=25)
   parser.add_argument('--batch_size',    required=False,help='Batch size',              type=int,default=64)
   parser.add_argument('--dist',          required=False,help='Distribution to use',     type=str,default='normal')
   parser.add_argument('--match',         required=False,help='Match discriminator',     type=int,default=0)
   a = parser.parse_args()

   LOSS           = a.loss
   DIST           = a.dist
   MATCH          = bool(a.match)
   EPOCHS         = a.epochs
   DATASET        = a.dataset
   DATA_DIR       = a.data_dir
   IMAGES_DIR     = a.image_dir
   LOG_DIR        = a.log_dir
   BATCH_SIZE     = a.batch_size
   CHECKPOINT_DIR = a.checkpoint_dir

   FILTERS        = [4, 5, 8, 9, 15, 18, 20, 26, 31]
   Y_DIM          = len(FILTERS)


   try: os.makedirs(IMAGES_DIR)
   except: pass
   try: os.makedirs(LOG_DIR)
   except: pass
   try: os.makedirs(CHECKPOINT_DIR)
   except: pass

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   real_images = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 64, 64, 3), name='real_images')
   z           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 100), name='z')
   y           = tf.placeholder(tf.float32, shape=(BATCH_SIZE, Y_DIM), name='y')
   fy          = tf.placeholder(tf.float32, shape=(BATCH_SIZE, Y_DIM), name='fy')

   # generated images
   gen_images = netG(z, y, BATCH_SIZE)

   # get the output from D on the real and fake data
   errD_real = netD(real_images, y, BATCH_SIZE, LOSS)
   # matching aware discriminator - send real images in with fake labels and mark as fake
   '''
   if MATCH == 'True':
      errD_fake1 = 0.5*netD(gen_images, y, BATCH_SIZE, LOSS, reuse=True)
      errD_fake2 = 0.5*netD(real_images, fy, BATCH_SIZE, LOSS, reuse=True)
      errD_fake = errD_fake1 + errD_fake2
      print 'using match'
   '''
   errD_fake = netD(gen_images, y, BATCH_SIZE, LOSS, reuse=True)

   # Important! no initial activations done on the last layer for D, so if one method needs an activation, do it
   e = 1e-12
   if LOSS == 'gan':
      errD_real = tf.nn.sigmoid(errD_real)
      errD_fake = tf.nn.sigmoid(errD_fake)
      errG = tf.reduce_mean(-tf.log(errD_fake + e))
      errD = tf.reduce_mean(-(tf.log(errD_real+e)+tf.log(1-errD_fake+e)))

   if LOSS == 'lsgan':
      errD_real = tf.nn.sigmoid(errD_real)
      errD_fake = tf.nn.sigmoid(errD_fake)
      errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))
      errG = tf.reduce_mean(0.5*(tf.square(errD_fake - 1)))

   if LOSS == 'wgan':
      # cost functions
      errD = tf.reduce_mean(errD_real) - tf.reduce_mean(errD_fake)
      errG = tf.reduce_mean(errD_fake)

      # gradient penalty
      epsilon = tf.random_uniform([], 0.0, 1.0)
      x_hat = real_images*epsilon + (1-epsilon)*gen_images
      d_hat = netD(x_hat, y, BATCH_SIZE, LOSS, reuse=True)
      gradients = tf.gradients(d_hat, x_hat)[0]
      slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
      gradient_penalty = 10*tf.reduce_mean((slopes-1.0)**2)
      errD += gradient_penalty

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   if LOSS == 'wgan':
      n_critic = 5
      beta1    = 0.0
      beta2    = 0.9
      lr       = 1e-4

   if LOSS == 'lsgan':
      n_critic = 1
      beta1    = 0.5
      beta2    = 0.999
      lr       = 0.001

   if LOSS == 'gan':
      n_critic = 1
      beta1    = 0.5
      beta2    = 0.999
      lr       = 0.0002

   # optimize G
   G_train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2).minimize(errG, var_list=g_vars, global_step=global_step)
   # optimize D
   D_train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=beta1,beta2=beta2).minimize(errD, var_list=d_vars)

   saver = tf.train.Saver(max_to_keep=1)
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   run_config = tf.ConfigProto()
   run_config.gpu_options.allow_growth=True
   sess  = tf.Session(config=run_config)
   sess.run(init)

   # write losses to tf summary to view in tensorboard
   try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   except:pass
   try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   except:pass

   summary_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())

   tf.add_to_collection('G_train_op', G_train_op)
   tf.add_to_collection('D_train_op', D_train_op)
   
   # restore previous model if there is one
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   ########################################### training portion

   step = sess.run(global_step)

   print 'Loading data...'
   #images, annots, test_images, test_annots = data_ops.load_celeba(DATA_DIR)
   images, annots = data_ops.load_celeba(data_dir=DATA_DIR, filters=FILTERS)

   train_len = len(annots)
   #test_len  = len(test_annots)

   print 'train num:',train_len
   #print 'test num:',test_len
   
   epoch_num = step/(train_len/BATCH_SIZE)

   sample_y     = data_ops.sample_label(BATCH_SIZE, Y_DIM, 0.0, 1.0)
   sample_z     = np.random.normal(0.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
   for i in range(min(BATCH_SIZE, Y_DIM)):
      sample_z[i] = sample_z[0]
   
   while epoch_num < EPOCHS:
      epoch_num = step/(train_len/BATCH_SIZE)
      start = time.time()

      # train the discriminator
      for critic_itr in range(n_critic):
         idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
         batch_z      = np.random.normal(0.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
         batch_y      = annots[idx]
         batch_img    = images[idx]
         batch_images = [data_ops.get_image(img,108,108,64,64,True,False) for img in batch_img]
         batch_images = np.array(batch_images).astype(np.float32)

         '''
         batch_images = np.empty((BATCH_SIZE, 64, 64, 3), dtype=np.float32)
         i = 0
         for img in batch_img:
            img = data_ops.normalize(misc.imread(img))
            batch_images[i, ...] = img
            i+=1
         '''

         if MATCH == True:
            batch_fy = 1-batch_y
            sess.run(D_train_op, feed_dict={z:batch_z, y:batch_y, fy:batch_fy, real_images:batch_images})
         else:
            sess.run(D_train_op, feed_dict={z:batch_z, y:batch_y, real_images:batch_images})
      
      # now train the generator once! use normal distribution, not uniform!!
      idx          = np.random.choice(np.arange(train_len), BATCH_SIZE, replace=False)
      batch_z      = np.random.normal(0.0, 1.0, size=[BATCH_SIZE, 100]).astype(np.float32)
      batch_y      = annots[idx]
      batch_img    = images[idx]
      batch_images = [data_ops.get_image(img,108,108,64,64,True,False) for img in batch_img]
      batch_images = np.array(batch_images).astype(np.float32)
      '''
      batch_images = np.empty((BATCH_SIZE, 64, 64, 3), dtype=np.float32)
      
      
      i = 0
      for img in batch_img:
         img = data_ops.normalize(misc.imread(img))
         batch_images[i, ...] = img
         i+=1
      '''

      if MATCH == True:
         batch_fy = 1-batch_y
         sess.run(G_train_op, feed_dict={z:batch_z, y:batch_y, fy:batch_fy, real_images:batch_images})
         D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op],
                                 feed_dict={z:batch_z, y:batch_y, fy: batch_fy, real_images:batch_images})
      else:
         # now get all losses and summary *without* performing a training step - for tensorboard and printing
         sess.run(G_train_op, feed_dict={z:batch_z, y:batch_y, real_images:batch_images})
         D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op],
                                 feed_dict={z:batch_z, y:batch_y, real_images:batch_images})

      summary_writer.add_summary(summary, step)

      print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-start
      step += 1
    
      if step%100 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')

         if MATCH == True:
            sample_fy = 1-sample_y
            # comes out as (1, batch, 64, 64, 3), so squeezing it
            gen_imgs = np.squeeze(np.asarray(sess.run([gen_images],
                                    feed_dict={z:sample_z, y:sample_y, fy:sample_fy})))
         else:
            gen_imgs = np.squeeze(np.asarray(sess.run([gen_images],
                                    feed_dict={z:sample_z, y:sample_y})))

         data_ops.save_images(gen_imgs[0:64] , [8, 8], '{}train_{:02d}_{:04d}.png'.format(IMAGES_DIR, epoch_num, step))
         num = 0
         for img,atr in zip(gen_imgs, batch_y):
            img = (img+1.)
            img *= 127.5
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = np.reshape(img, (64, 64, -1))
            misc.imsave(IMAGES_DIR+'step_'+str(step)+'_num_'+str(num)+'.png', img)
            with open(IMAGES_DIR+'attrs.txt', 'a') as f:
               f.write('step_'+str(step)+'_num_'+str(num)+','+str(atr)+'\n')
            num += 1
            if num == 5: break
   saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
   saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')


