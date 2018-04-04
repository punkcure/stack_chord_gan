import tensorflow as tf
import numpy as np
from Chord_utils import *
from midi_utils import *
from utils import *

class midiGAN(object):
    def __init__(self,is_training,sess,config,chord_feature_num):
        #Init related params here
        self.sess = sess
        self.config = config
        self.is_training = is_training
        self.chord_feature_num = chord_feature_num

        self.input_chords = tf.placeholder(dtype = tf.float32,shape = [config.batch_size,config.chord_length,chord_feature_num],name = 'input_chord')
        #To get a list of 'config.chord_length' items ,each has shape[config.batch_size,chord_feature_num]
        #squeeze removes dimensions of size 1 from the shape of a tensor.
        self.input_chords_list = tf.unstack(self.input_chords,axis=1)
        #self.input_chords_list = [tf.squeeze(input_,[1]) 
        #    for input_ in tf.split(self.input_chords,self.config.chord_length,1)]

           
        #calc l2 reg loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.1
        self.reg_loss = reg_constant*sum(reg_losses)

        #build G and D
        self.build_model()

        if not is_training:
            return
     

        #Gradient Descent d
        d_optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate*self.config.d_lr_factor)
        d_grads = tf.gradients(self.d_loss,self.d_params)
        d_grads,_ = tf.clip_by_global_norm(d_grads,config.max_grad_norm)
        self.opt_d = d_optimizer.apply_gradients(zip(d_grads,self.d_params))
        
        #Gradient Descent g
        g_optimizer = tf.train.AdamOptimizer(config.learning_rate)
        g_grads = tf.gradients(self.g_loss,self.g_params)
        g_grads,_ = tf.clip_by_global_norm(g_grads,config.max_grad_norm)
        self.opt_g = g_optimizer.apply_gradients(zip(g_grads,self.g_params))

        self.lr = tf.Variable(self.config.learning_rate,trainable=False,dtype=tf.float32)
        self.new_lr = tf.placeholder(shape=[],name='new_learning_rate',dtype=tf.float32)
        self.lr_update = tf.assign(self.lr,self.new_lr)

    def build_model(self):
        #constuct G
        self.G = self.Generator()
        self.g_params = [v for v in tf.trainable_variables() if v.name.startswith('model/G/')]
        
        #construct D
        self.D_real = self.Discriminator(self.input_chords_list) 
        self.D_fake = self.Discriminator(self.G,reuse=True)
        self.d_params = [v for v in tf.trainable_variables() if v.name.startswith('model/D/')]

        #construct D loss with clip value
        #todo:why not use sigmoid instead of clippin? 
        self.d_loss_real = tf.reduce_mean(tf.clip_by_value(self.D_real,1e-1000000,1.0))
        self.d_loss_fake = tf.reduce_mean(1-tf.clip_by_value(self.D_fake,0.0,1.0-1e-1000000))
        self.d_loss = -tf.log(self.d_loss_real) - tf.log(self.d_loss_fake)
        self.d_loss += self.reg_loss

        #construct G loss
        self.g_loss = tf.reduce_mean(-tf.log(tf.clip_by_value(self.D_fake,1e-100000,1.0)))
        self.g_loss += self.reg_loss 

        self.saver = tf.train.Saver()

    def Generator(self):
        with tf.variable_scope('G',regularizer=tf.contrib.layers.l2_regularizer(scale=self.config.reg_scale)) as scope:
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size_g)
            if (self.is_training) and (self.config.keep_prob < 1):
                lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=self.config.keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*self.config.num_layers_g,state_is_tuple=True)

            initial_state = cell.zero_state(self.config.batch_size,tf.float32)
            #todo: init one_hot code and other code separately and make final concat 
            random_rnninputs = tf.random_uniform(shape=[self.config.batch_size,self.config.chord_length,self.chord_feature_num],minval=0.0,maxval=1.0)
            #make a list of training tensor with shape [batch_size,chord_feature_num],one per step in recurrence
            random_rnninputs_list = [tf.squeeze(input_,[1]) for input_ in tf.split(random_rnninputs,self.config.chord_length,1)]

            #randomly init outputs
            rnnoutputs = tf.random_uniform(shape=[self.config.batch_size,self.chord_feature_num],minval=0.0,maxval=1.0)
            outputs,results = [],[]

            #train step by step
            for i,input_ in enumerate(random_rnninputs_list):
                if i > 0 : scope.reuse_variables()
                concat_inputs = [input_]
                concat_inputs.append(rnnoutputs)
                input_ = tf.concat(values = concat_inputs,axis = 1)
                input_ = tf.nn.relu(full_connect(input_,self.config.hidden_size_g,scope_name='input_layer',reuse=(i!=0)))
                
                output,state = cell(input_,initial_state)
                outputs.append(output)

                #softmax the last dimensions of code for chord type
                rnnoutputs = full_connect(output,self.chord_feature_num,scope_name='output_layer',reuse=(i!=0))
                rnnoutputs_chord_one_hot = rnnoutputs[:,CHORD_DATA_TYPE:]
                rnnoutputs_feature = rnnoutputs[:,:CHORD_DATA_TYPE]

                rnnoutputs_chord_one_hot = tf.nn.softmax(rnnoutputs_chord_one_hot)
                rnnoutputs_new = tf.concat([rnnoutputs_feature,rnnoutputs_chord_one_hot],axis=1)
                
                results.append(rnnoutputs_new)

            self.final_state_g = state
            
            return results



    def Discriminator(self,inputs,reuse=False):
        #input:a list with size 'length' of tensors,each tensor has shape[batch_size,chord_feature_num]
        #input after stack(axis=1):[batch_size,length,neuron_num*2]
        #after passing multi_bidirectional_rnn,get[batch_size,length,neuron_num*2]
        #output:[length] [batch_size,length,1]
        with tf.variable_scope('D') as scope:
            if reuse:
                scope.reuse_variables()
            if (self.is_training) and (self.config.keep_prob < 1.0):
                inputs = [tf.nn.dropout(input,self.config.keep_prob) for input in inputs]
        
            inputs = tf.stack(inputs,axis=1)
            outputs = Multi_layer_bidirectional_rnn(self.config.hidden_size_d,self.config.num_layers_d,inputs,reuse=reuse)

            #todo:why unstack
            decisions_list = tf.unstack(outputs)
            decisions = [tf.sigmoid(full_connect(output,1,'decision',reuse=(i!=0))) for i,output in enumerate(decisions_list)]
            decisions = tf.stack(decisions)
            decisions = tf.transpose(decisions,[1,0,2])
            #decisions_final = tf.reduce_mean(decisions,reduction_indices=[1,2])

        return decisions
        #return decisions_final,decisions

    def assign_lr(self,learning_rate):
        self.sess.run(self.lr_update,feed_dict={self.new_lr:learning_rate})

    def load_and_init_variables(self):
        if self.saver is None:
            self.saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
        self.global_step = 0
        if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
            print("reading check point from:{}".format(ckpt.model_checkpoint_path))
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.global_step = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
        else:
            print("Init variables without checkpoint!")
            self.sess.run(tf.initialize_all_variables())

    def save(self,counter):
        if self.saver is None:
            self.saver = tf.train.Saver()
        
        ckpt_path = os.path.join(self.config.checkpoint_dir,'model.ckpt')
        self.saver.save(self.sess,ckpt_path,global_step=(self.global_step+counter))



