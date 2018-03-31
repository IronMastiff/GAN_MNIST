import tensorflow as tf
import yaml

def model_inputs( real_dim, z_dim ):
    inputs_real = tf.placeholder( tf.float32, ( None, real_dim ), name = 'input_real' )
    inputs_z = tf.placeholder( tf.float32, ( None, z_dim ), name = 'input_z' )

    return inputs_real, inputs_z

def generator( z, out_dim, n_units = 128, reuse = False, alpha = 0.01 ):
    with tf.variable_scope( 'generator', reuse = reuse ):
        # Hidden layer
        h1 = tf.layers.dense( z, n_units, activation = None )# Leaky ReLUense( z, n_units, activation = None )    # 全链接层的高级封装接口

        h1 = tf.maximum( alpha * h1, h1 )

        # Logits and tanh output
        logits = tf.layers.dense( h1, out_dim, activation = None )
        out = tf.tanh( logits )

        return out

def discriminator( x, n_units = 128, reuse = False, alpha = 0.01 ):
    with tf.variable_scope( 'discriminator', reuse = reuse ):
        # Hidden layer
        h1 = tf.layers.dense( x, n_units, activation = None )    # 全链接层的高级封装接口
        # Leacy ReLU
        h1 = tf.maximum( alpha * h1, h1 )

        logits = tf.layers.dense( h1, 1, activation = None )
        out = tf.sigmoid( logits )

        return out, logits

def print_training_loss( sess ):
    # tf.summary.scalar( 'd_loss', d_loss )
    # tf.summary.scalar( 'g_loss', g_loss )

    # merge all summary together
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter( 'logs/', sess.graph )

    return merged, writer

class Flag( object ):
    def __init__( self,**entries  ):
        self.__dict__.update( entries )

def read_config_file( config_file ):
    with open( config_file ) as f:
        FLAGES = Flag( **yaml.load( f ) )
    return FLAGES

def select_data( mnist, label ):
    data_idx = []
    for i in range( mnist.train.images.shape[0] ):
        if mnist.train.labels[i] == label:
            data_idx.append( i )
    datas = mnist.train.images[data_idx]
    return datas

def batch_data( datas, batch_size ):
    batches = []
    for i in range( datas.shape[0] // batch_size ):
        batch = datas[i * batch_size : ( i + 1 ) * batch_size, :]
        batches.append( batch )

    return batches