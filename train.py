import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import os

import utils

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets( 'MNIST_data' )


'''--------Load the config file--------'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '-c', '--config', default = 'config.yml', help = 'The path to the config file' )

    return parser.parse_args()


args = parse_args()
FLAGS = utils.read_config_file( args.config )

if not( os.path.exists( FLAGS.generate_file ) ):
    os.makedirs( FLAGS.generate_file )


'''--------Preprocessing data--------'''
if( FLAGS.select_label != 'All' ):
    datas = utils.select_data( mnist, FLAGS.select_label )
else:
    datas = mnist.train.images    # shape ( 55000, 784 )

batches = utils.batch_data( datas, FLAGS.batch_size )


'''-----------Hyperparameters------------'''
# Size of input image to discriminator
input_size = 784
# Size of latent vector to genorator
z_size = 100
# Size of hidden layers in genorator and discriminator
g_hidden_size = FLAGS.g_hidden_size
d_hidden_size = FLAGS.d_hidden_size
# Leak factor for leaky ReLU
alpha = FLAGS.alpha
# Smoothing
smooth = 0.1


'''------------Build network-------------'''
tf.reset_default_graph()

# Creat out input placeholders
input_real, input_z = utils.model_inputs( input_size, z_size )

# Build the model
g_model = utils.generator( input_z, input_size, n_units = g_hidden_size, alpha = alpha )
# g_model is the generator output

d_model_real, d_logits_real = utils.discriminator( input_real, n_units = d_hidden_size, alpha = alpha )
d_model_fake, d_logits_fake = utils.discriminator( g_model, reuse = True, n_units = d_hidden_size, alpha = alpha )


'''---Discriminator and Generator Losses---'''
# Calculate losses
d_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits = d_logits_real,
                                                                       labels = tf.ones_like( d_logits_real ) * ( 1 - smooth ), name = 'd_loss_real' ) )

d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits = d_logits_fake,
                                                                       labels = tf.zeros_like( d_logits_fake ), name = 'd_loss_fake' ) )
d_loss = d_loss_real + d_loss_fake
# add d_loss to summary scalar
tf.summary.scalar('d_loss', d_loss)

g_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits = d_logits_fake,
                                                                  labels = tf.ones_like( d_logits_fake ), name = 'g_loss' ) )
# add g_loss to summary scalar
tf.summary.scalar('g_loss', g_loss)


'''---------------Optimizers----------------'''
# Optimizers
learning_rate = FLAGS.learning_rate

# Get the trainable_variables, split into G and D parts
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith( 'generator' )]
d_vars = [var for var in t_vars if var.name.startswith( 'discriminator' )]

d_train_opt = tf.train.AdamOptimizer( learning_rate ).minimize( d_loss, var_list = d_vars )
g_train_opt = tf.train.AdamOptimizer( learning_rate ).minimize( g_loss, var_list = g_vars )


'''-----------------Traing---------------------'''
batch_size = FLAGS.batch_size
epoches = FLAGS.epoches
samples = []
# losses = []
# Only save generator variables
saver = tf.train.Saver( var_list = g_vars )
with tf.Session() as sess:
    # Tensorboard Print Loss
    merged, writer = utils.print_training_loss(sess)

    sess.run( tf.global_variables_initializer() )
    for e in range( epoches ):
        for batch in batches:
            # batch = mnist.train.next_batch( batch_size )
            # Get images, reshape and rescale to pass to D
            batch_images = batch
            batch_images = batch_images * 2 - 1

            # Sample random noise for G
            batch_z = np.random.uniform( -1, 1, size = ( batch_size, z_size ) )

            # Run optimizers
            _ = sess.run( d_train_opt, feed_dict = {input_real : batch_images, input_z : batch_z} )
            _ = sess.run( g_train_opt, feed_dict = {input_z : batch_z} )

        # At the end of each epoch, get the losses and print them out
        train_loss_d = sess.run( d_loss, {input_z : batch_z, input_real : batch_images} )
        train_loss_g = g_loss.eval( {input_z : batch_z} )

        print( 'Epoch {}/{}...' . format( e + 1, epoches ),
               'Discriminator Loss: {:.4f}...' . format( train_loss_d ),
               'Generator Loss: {:.4f}' . format( train_loss_g ) )
        # Save losses to view after training
        # losses.append( ( train_loss_d, train_loss_g ) )

        # Add data to tensorboard
        rs = sess.run(merged, feed_dict={input_z: batch_z, input_real: batch_images})
        writer.add_summary(rs, e)

        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform( -1, 1, size = ( 16, z_size ) )
        gen_samples = sess.run(
            utils.generator( input_z, input_size, n_units = g_hidden_size, reuse = True, alpha = alpha),
            feed_dict = {input_z : sample_z} )


        gen_image = gen_samples.reshape( ( -1, 28, 28, 1 ) )
        gen_image = tf.cast( np.multiply( gen_image, 255 ), tf.uint8 )
        for r in range( gen_image.shape[0] ):
            with open( FLAGS.generate_file + str(e) + ' ' + str( r ) + '.jpg', 'wb' ) as img:
                img.write( sess.run( tf.image.encode_jpeg( gen_image[r] ) ) )

        samples.append( gen_samples )
        saver.save( sess, './checkpoint/generator.ckpt' )