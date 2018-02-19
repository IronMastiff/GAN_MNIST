import tensorflow as tf
import utils

'''Hyperparameters'''

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

'''------------Build network------------'''

tf.reset_default_graph()

# Creat out input placeholders
input_real, input_z = utils.model_inputs( input_size, z_size )

# Build the model
g_model = utils.generator( input_z, input_size, n_units = g_hidden_size, alpha = alpha )
# g_model is the generator output

d_model_real, d_logits_real = utils.discriminator( input_real, n_units = d_hidden_size, alpha = alpha )
d_model_fake, d_logits_fack = utils.discriminator( g_model, reuse = True, n_units = d_hidden_size, alpha = alpha )

'''---------------------------------------'''