import tensorflow as tf 

def full_connect(input_tensor,output_size,scope_name = 'full_connect',stdev=0.02,bias_start=0.0,reuse = False):
    with tf.variable_scope(scope_name) as scope:
        if reuse:
            scope.reuse_variables()
        w = tf.get_variable(name = 'fc_w',shape=[input_tensor.get_shape()[-1],output_size],dtype=tf.float32,
                            initializer=tf.random_normal_initializer(stddev=stdev))
        b = tf.get_variable(name = 'fc_b',shape=[output_size],dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_start))
        
        return tf.matmul(input_tensor,w) + b
        
def Multi_layer_bidirectional_rnn(num_units,num_layers,inputs,reuse=False):
    if len(inputs.get_shape().as_list()) != 3:
        raise ValueError("the inputs must be 3-dimensional Tensor!")
    
    _inputs = inputs

    with tf.variable_scope('bidirectional_rnn',default_name='bidirectional_rnn') as scope:
        if reuse:
            scope.reuse_variables()
        for i in range(num_layers):
            rnn_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units)            
            rnn_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units)

            initial_state_fw = rnn_cell_fw.zero_state(inputs.get_shape()[0],dtype=tf.float32)
            initial_state_bw = rnn_cell_fw.zero_state(inputs.get_shape()[0],dtype=tf.float32)
            
            rnn_scope_name = 'layer_'+str(i)
            (output,state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw,rnn_cell_bw,_inputs,initial_state_bw=initial_state_bw, \
                initial_state_fw=initial_state_fw,scope=rnn_scope_name)
            
            _inputs = tf.concat(output,2)

    return _inputs