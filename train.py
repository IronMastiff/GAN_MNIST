import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import utils

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets( 'MNIST_data' )


'''-----------Hyperparameters------------'''
# Size of input image to discriminator
input_seze = 784
# Size of latent vector to genorator
z_size = 100
# Size of hidden layers in genorator and discriminator
g_hidden_size = 128
d_hidden_size = 128
# Leak factor for leaky ReLU
alpha = 0.01
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
                                                                       labels = tf.ones_like( d_logits_real ) * ( 1 - smooth ) ) )

d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits = d_logits_fake,
                                                                       labels = tf.zeros_like( d_logits_fake ) ) )
d_loss = d_loss_real + d_loss_fake
# add d_loss to summary scalar
tf.summary.scalar( 'd_loss', d_loss )

g_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits( logits = d_logits_fake,
                                                                  labels = tf.ones_like( d_logits_fake ) ) )
# add g_loss to summary scalar
tf.summary.scalar( 'g_loss', g_loss )


'''---------------Optimizers----------------'''
# Optimizers
learning_rate = 0.002

# Get the trainable_variables, split into G and D parts
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startwith( 'generator' )]
d_vars = [var for var in t_vars if var.name.startwith( 'discriminator' )]

d_train_opt = tf.train.AdamOptimizer( learning_rate ).minimize( d_loss, var_list = d_vars )
g_train_opt = tf.train.AdamOptimizer( learning_rate ).minimize( g_loss, var_list = g_vars )


'''-----------------Traing---------------------'''
batch_size = 100
epoches = 100
samples = []
# losses = []
# Only save generator variables
saver = tf.train.Saver( var_list = g_vars )
with tf.InteractiveSession() as sess:
    # Tensorboard Print Loss
    merged, writer = utils.print_training_loss( sess )

    sess.run( tf.global_variables_initializer() )
    for e in range( epoches ):
        for i in range( mnist.train.num_examples // batch_size ):
            batch = mnist.train.next_batch( batch_size )

            # Get images, reshape and rescale to pass to D
            batch_images = batch[0].reshape( ( batch_size, 784 ) )
            batch_images = batch_images * 2 - 1

            # Sample random noise for G
            batch_z = np.random.uniform( -1, 1, size = ( batch_size, z_size ) )

            # Run optimizers
            _ = sess.run( d_train_opt, feed_dict = {input_real : batch_images, input_z : batch_z} )
            _ = sess.run( g_train_opt, feed_dict = {input_z : batch_z} )

        # At the end of each epoch, get the losses and print them out
        train_loss_d = sess.run( d_loss, {input_z : batch_z, input_real : batch_images} )
        train_loss_g = g_loss.eval( g_loss, {input_z : batch_z} )


        # Add data to tensorboard
        rs = sess.run( merged, feed_dict = {input_z : batch_z, input_real : batch_images} )
        writer.add_summary( rs, e )

        print( 'Epoch {}/{}...' . format( e + 1, epochs ),
               'Discriminator Loss: {:.4f}...' . format( train_loss_d ),
               'Generator Loss: {:.4f}' . format( train_loss_g ) )
        # Save losses to view after training
        # losses.append( ( train_loss_d, train_loss_g ) )

        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform( -1, 1, size = ( 16, z_size ) )
        gen_samples = sess.run(
            utils.generator( input_z, input_size, n_uints = g_hidden_size, reuse = True, alpha = alpha),
            feed_dict = {input_z : sample_z} )
        samples.append( gen_samples )
        saver.save( sess, './checkpoint/generator.ckpt' )

# Save training generator samples
with open( 'trian_samples.pkl', wb ) as f:
    pkl.dump( samples, f )


'''----------Print Training Loss----------'''
# fig, ax = plt.subplot()
# losses = np.array( losses )
# plt.plot( losses.T[0], label = 'Discriminator' )
# plt.plot( losses.T[1], label = 'Generator' )
# plt.title( 'Training Losses' )
# plt.legend()


'''----------Tensorboard Printing Loss of Step 1------------'''
merged, writer = utils.print_training_loss( d_loss, g_loss, sess )


'''----------Generator samples from training----------'''
def view_samples( epoch, samples ):
    fig, axes = plt.subplot( figsize = ( 7, 7 ), nrows = 4, sharey = True, sharex = True )
    for ax, img in zip( axes.flatten(), samples[epoch] ):
        ax.xaxis.set_visible( False )
        ax.yaxis.set_visible( False )
        im = ax.imshow( img.reshpae( ( 28, 28 ) ), cmap = 'Greys_r' )
    return fig, axes

# Load samples from generatro taken while training
with open( 'train_samples.pkl', 'rb' ) as f:
    samples = pkl.load( f )