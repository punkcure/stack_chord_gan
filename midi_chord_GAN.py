from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time,datetime,os,sys

import numpy as np 
import tensorflow as tf

from model import *
from midi_utils import *
from Chord_utils import *

flags = tf.flags

#flags.DEFINE_string('data_dir','E:\\MLCode\\TrainningSource\\Nottingham\\Nottingham\\train\\','directory of training,validating and testing data')
flags.DEFINE_string('data_dir','/home/punkcure/TrainningResource/Nottingham/train/','')
flags.DEFINE_string('checkpoint_dir','checkpoint','')
flags.DEFINE_string('result_dir','result','')
flags.DEFINE_string('sample_dir','sample','')
flags.DEFINE_boolean('is_training',True,'')
flags.DEFINE_integer('epoches_to_save',10,'')
flags.DEFINE_integer('epoches',500,'')
flags.DEFINE_float('learning_rate',0.001,'')
flags.DEFINE_float('d_lr_factor',0.5,'')
flags.DEFINE_float('max_grad_norm',5.0,'')
flags.DEFINE_float('keep_prob',0.5,'')
flags.DEFINE_float('lr_decay',1.0,'')
flags.DEFINE_integer('epochs_before_decay',60,'')
flags.DEFINE_integer('num_layers_g',2,'')
flags.DEFINE_integer('num_layers_d',2,'')
flags.DEFINE_integer('chord_length',30,'')
flags.DEFINE_integer('hidden_size_g',350,'')
flags.DEFINE_integer('hidden_size_d',350,'')
flags.DEFINE_integer('batch_size',20,'')
flags.DEFINE_integer('pretraining_epochs',6,'')
flags.DEFINE_boolean('Adam',False,'')
flags.DEFINE_float('reg_scale',1.0,'')

FLAGS = flags.FLAGS

midi_tools = midi_utils()

def main(_):
    try:os.makedirs(FLAGS.checkpoint_dir)
    except:pass

    try:os.makedirs(FLAGS.result_dir)
    except:pass

    try:os.makedirs(FLAGS.sample_dir)
    except:pass

    #a = midi_tools.read_midi_file_and_process('/home/punkcure/TrainningResource/Nottingham/train/jigs_simple_chords_225.mid')
    midi_tools.read_midi_files(FLAGS.data_dir)

    train_start_time = time.time()
    
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True
    with tf.Session(config = run_config) as sess:
        with tf.variable_scope('model',regularizer=tf.contrib.layers.l2_regularizer(scale=FLAGS.reg_scale),reuse=None) as scope:
            midi_gan_model = midiGAN(is_training=FLAGS.is_training,sess=sess,config=FLAGS,chord_feature_num=midi_tools.FEATURE_SHAPE)
            midi_gan_model.load_and_init_variables()

            if FLAGS.is_training:
                train(midi_gan_model,sess,train_start_time)
            else:
                sample(midi_gan_model,sess,False)
    

def train(model,sess,start_time):
    g_loss,d_loss = 1.0,1.0
    save,do_exit = False,False

    for epoch in range(FLAGS.epoches):
    #set learning rate by epoches
        lr_decay = FLAGS.lr_decay**max(epoch-FLAGS.epochs_before_decay,0.0)
        model.assign_lr(FLAGS.learning_rate*lr_decay)

        g_loss,d_loss = run_epoch(epoch,sess,model,midi_tools,g_loss,d_loss)
        print('***Epoch:{} G Loss:{} D Loss{}***'.format(epoch,g_loss,d_loss))

        #todo valide current model
        #g_loss_valid,d_loss_valid = run_epoch(sess,model,midi_tools,g_loss_valid,d_loss_valid)
        #print('Epoch:{} G Loss valid:{} D Loss valid{}'.format(epoch,g_loss,d_loss))

        if g_loss == 0.0 and d_loss == 0.0:
            print('******************loss value are both 0**********************')
            save = True
            do_exit = True
        if epoch % FLAGS.epoches_to_save == 0:
            save = True           
        if save:
            model.save(epoch)
            sample(model,sess,False)

def run_epoch(epoch_id,sess,model,midi_tools,cur_g_loss,cur_d_loss):
    epoch_start_time = time.time()
    g_loss,d_loss = 10.0,10.0
    g_losses,d_losses = cur_g_loss,cur_d_loss
    iters = 0

    batch_num_per_epoch = int(midi_tools.get_chord_info_len()/FLAGS.batch_size)
    for batch in range(batch_num_per_epoch):
        reset_batch = False
        if batch == 0 : reset_batch = True
        batch_chord = midi_tools.get_batch_chord(FLAGS.batch_size,FLAGS.chord_length,reset_batch)

        fetches_g = [model.g_loss,model.opt_g]
        fetches_d = [model.d_loss,model.opt_d]
        feed_dict = {model.input_chords:batch_chord}

        g_update_time = 2
        d_update_time = 1

        if d_loss == 0 and g_loss == 0:
            print('****************Warning:Both G and D train losses are zero in run_epoch.********************')
            break
        elif d_loss >= g_loss*3:
            g_update_time = 0
        elif g_loss >= d_loss*3:
            d_update_time = 0

        for i in range(g_update_time):
            g_loss,_ = sess.run(fetches_g,feed_dict)
        for i in range(d_update_time):
            d_loss,_ = sess.run(fetches_d,feed_dict) 

        print(' Batch/Epoch:{}/{} g_loss {} d_loss {}'.format(batch,epoch_id,g_loss,d_loss))

        g_losses += g_loss
        d_losses += d_loss
    
    g_mean_losses = g_losses / batch_num_per_epoch
    d_mean_losses = d_losses / batch_num_per_epoch

    return g_mean_losses,d_mean_losses

def sample(model,sess,try_load_ckpt=True):
    if try_load_ckpt:
        model.load_and_init_variables()
    chord_list = sess.run(model.G,feed_dict={})

    #change the order of batch and step index
    chord_list_batch_first = []
    for i in xrange(chord_list[0].shape[0]):
        chord_list_batch_first.append([x[i,:] for x in chord_list])

    current_time = time.strftime('./sample/%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))
    #chord_tesor:[batch_size,length,FEATURE_SHAPE]
    midi_tools.save_batch_files(current_time,chord_list_batch_first)

if __name__ == '__main__':
    tf.app.run()

