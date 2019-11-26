# BNN - Uncertainty SCRNN - ATI Project - PhD Computer Science
# TODO(akanni-ade) replace use of input weights with input channels
import os
import sys

import utility

import tensorflow as tf
import tensorflow_probability as tfp
try:
    import tensorflow_addons as tfa
except:
    pass
from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd
from tensorboard.plugins.hparams import api as hp
import pandas as pd

import pickle

import math
import numpy as np

import argparse 
from tqdm import tqdm
import glob
import itertools

import traceback

import time

@tf.function
def replace_inf_nan(_tensor):
    nan_bool_ind_tf = tf.math.is_nan( tf.dtypes.cast(_tensor,dtype=tf.float32 ) )
    inf_bool_ind_tf = tf.math.is_inf( tf.dtypes.cast( _tensor, dtype=tf.float32 ) )

    bool_ind_tf = tf.math.logical_or( nan_bool_ind_tf, inf_bool_ind_tf )
    
    _tensor = tf.where( bool_ind_tf, tf.constant(0.0,dtype=tf.float32), _tensor )
    return tf.dtypes.cast(_tensor, dtype=tf.float32)

#region Model

class SuperResolutionModel( tf.keras.Model ):
    def __init__(self, hparams ):
        super(SuperResolutionModel, self).__init__()
        
        self.SRCNN_1 = SRCNN( hparams )
         
    def call(self, inputs, tape):
        x = self.SRCNN_1(inputs, tape=tape)
        #self.losses += self.SRCNN_1.losses()
        #self.losses += self.SRCNN_1.posterior_entropy()
        #self.losses += self.SRCNN_1.prior_cross_entropy()
        return x
    
class SRCNN( tf.keras.layers.Layer ):
    """ 
        Super Resolution Convolutional Module
        ..info: Each SRCNN will have its own seperate set of horseshoe parameters. 
    """
    def __init__(self, hparams ):
        super( SRCNN, self ).__init__()
        
        self.hparams = hparams                
        
        self.initialize_priors_dist()

        self.initialize_posteriors_vars()

        self.update_priors_dists()        
        self.update_posteriors_dists()

        if self.hparams['var_model_type'] == 'reparam':

            self.conv1 = tfpl.Convolution2DReparameterization( **self.hparams['conv1_params'] ,
                kernel_posterior_fn = horseshoe_kernel_posterior_distribution( self, 1 ), 
                kernel_posterior_tensor_fn= ( lambda dist: horsehoe_kernel_posterior_tensor_fn(dist, self.conv1_nu , self.c ,
                                            self.hparams['conv1_output_node_count'], self.hparams['conv1_params']['kernel_size'], self.hparams['conv1_inp_channels'],
                                            self.tape, 1, self) )  ,
                kernel_prior_fn = None,
                
                kernel_divergence_fn= None) #TODO: Figure out the appropriate posterior and prior for the bias
                                            #TODO: Move the calculation for the kl divergence into this constructor

            self.upSample = UpSampler( self.hparams['input_dims'], self.hparams['output_dims'] )
            
            self.conv2 = tfpl.Convolution2DReparameterization ( **self.hparams['conv2_params'] ,
                            kernel_posterior_fn = horseshoe_kernel_posterior_distribution( self, 2), 
                            
                            kernel_posterior_tensor_fn= ( lambda dist: horsehoe_kernel_posterior_tensor_fn(dist, self.conv2_nu , self.c,
                                                        self.hparams['conv2_output_node_count'], self.hparams['conv2_params']['kernel_size'], self.hparams['conv2_inp_channels'],
                                                        self.tape, 2, self) )  ,

                            kernel_prior_fn = None,
                            kernel_divergence_fn= None)

            self.conv3 = tfpl.Convolution2DReparameterization( **self.hparams['conv3_params'] , 
                    kernel_posterior_fn = HalfCauchy_Guassian_posterior_distribution( self ) , 
                    kernel_posterior_tensor_fn= lambda dist: HalfCauchy_Guassian_posterior_tensor_fn(self, dist, self.hparams['conv3_params']['kernel_size'], self.hparams['conv3_inp_channels'] ) ,

                    kernel_prior_fn = None , 
                    kernel_divergence_fn= None
                    )   
                
    def call( self, _input, tape ,upsample_method=tf.constant("zero_padding") ): #( batch, height, width)
        self.tape = tape
        self.update_priors_dists()        
        self.update_posteriors_dists()
        self.sample_params()
        
        x = self.conv1( _input )    #( batch, height_lr, width_lr, conv1_filters ) #TODO:(akanni-ade) alot of zero values for x at this output
        
        x = self.upSample( x )           #(batch, height_hr, width_hr, conv1_filters )

        x1 = self.conv2( x )         #(batch, height_hr, width_hr, conv2_filters )
        #TODO:(akanni-ade) add layer norm or batch norm here
        x = self.conv3( x1 + x )       #(batch, height_hr, width_hr, 1 )
        
        self.prior_cross_entropy()
        self.posterior_entropy() #TODO check that layer losses are sent to model losses

        return x
    
    def initialize_priors_dist(self):
        # Global C Prior
        self.global_c_priorshape = tf.constant( 2.0, name="global_c_priorshape" ) 
        self.global_c_priorscale = tf.constant( 6.0 , name="global_c_priorscale" )
        self.c_prior_dist = tfd.InverseGamma(self.global_c_priorshape, self.global_c_priorscale )

        # Layerwise nu Prior
        global_nu_scale = tf.constant(1.0, name="global_nu_scale") #b_g in Soumya paper #This is set to 1 in soumya paper 1, and 1e-5 in Soumya Paper 2
        self.conv1_nu_prior_dist  = tfd.HalfCauchy( 0.0, global_nu_scale, name="conv1_nu_prior_dist")
        self.conv2_nu_prior_dist  = tfd.HalfCauchy( 0.0, global_nu_scale, name="conv2_nu_prior_dist")

        # Betas for Nodes Prior
        conv1_beta_prior_loc = tf.zeros([self.hparams['conv1_output_node_count'],self.hparams['conv1_input_weights_per_filter']], dtype=tf.float32  )
        conv1_beta_prior_scale_diag = tf.ones( [self.hparams['conv1_input_weights_per_filter']], dtype=tf.float32 )
        self.conv1_Beta_prior_dist = tfd.MultivariateNormalDiag(conv1_beta_prior_loc, conv1_beta_prior_scale_diag)

        conv2_beta_prior_loc = tf.zeros((self.hparams['conv2_output_node_count'],self.hparams['conv2_input_weights_per_filter']), dtype=tf.float32  )
        conv2_beta_prior_scale_diag = tf.ones([self.hparams['conv2_input_weights_per_filter']],dtype=tf.float32 )
        self.conv2_Beta_prior_dist = tfd.MultivariateNormalDiag(conv2_beta_prior_loc, conv2_beta_prior_scale_diag )

        # Taus for Nodes Prior
        conv1_taus_prior_loc = tf.zeros((self.hparams['conv1_output_node_count']), dtype=tf.float32  )
        conv1_taus_prior_scale = tf.constant( 1.0 , name="conv1_taus_prior_scale")  #b_0 in Soumya Paper #NOTE: eventually this needs to learn to be learnt(variable)
        self.conv1_tau_prior_dist = tfd.HalfCauchy(conv1_taus_prior_loc, conv1_taus_prior_scale)

        conv2_taus_prior_loc = tf.zeros((self.hparams['conv2_output_node_count']), dtype=tf.float32  )
        conv2_taus_prior_scale = tf.constant( 1.0 , name="conv2_taus_prior_scale")  #b_0 in Soumya Paper #NOTE: eventually this needs to learn to be learnt(variable)
        self.conv2_tau_prior_dist = tfd.HalfCauchy(conv2_taus_prior_loc, conv2_taus_prior_scale)

        self.conv3_weights_prior_loc = tf.Variable( initial_value =tf.zeros( (self.hparams['conv3_input_weights_count']), dtype=tf.float32  ), trainable=True, name="conv3_weights_prior_loc"  )
        self.conv3_weights_prior_scale = tf.Variable( initial_value =tf.random.uniform( shape=[self.hparams['conv3_input_weights_count']], minval=15, maxval=20.0 ), trainable=True, name="conv3_weights_prior_scale"  )

    def initialize_posteriors_vars(self):
        #Global C Posterior
        self.global_c_post_shape = tf.Variable( initial_value=1., trainable=True, name="global_c_post_shape" ) #TODO: Create Initialisation Strategy dependent on size of network. :Base it on the ending size of the network
        self.global_c_post_scale = tf.Variable( initial_value=1.,  trainable=True, name="global_c_post_scale" ) #TODO: Create Initialisation Strategy dependent on size of network. :Base it on the ending size of the network 
        
        # Layerwise nu Posterior
        self.conv1_nu_post_mean = tf.Variable( initial_value = tf.random.uniform( [1],0.75,1.25 )[0], trainable=True, name="conv1_nu_post_mean") #TODO: Create reason for why an initialization scheme should be best
        self.conv1_nu_post_scale = tf.Variable( initial_value = tf.random.uniform( [1],0.75,1.25)[0], trainable=True, name="conv1_nu_post_scale" ) #TODO: Create reason for why an initialization scheme should be best

        self.conv2_nu_post_mean = tf.Variable( initial_value = tf.random.uniform( [1],0.75,1.25 )[0], trainable=True, name="conv1_nu_post_mean") #TODO: Create reason for why an initialization scheme should be best
        self.conv2_nu_post_scale = tf.Variable( initial_value = tf.random.uniform( [1],0.75,1.25)[0], trainable=True, name="conv1_nu_post_scale" ) #TODO: Create reason for why an initialization scheme should be best
        
        # Betas_LogTaus for Nodes Posterior
        self.conv1_beta_post_loc = tf.Variable( initial_value=tf.random.uniform( [self.hparams['conv1_output_node_count'], self.hparams['conv1_input_weights_per_filter']] ,-1, 1), trainable=True, name="conv1_beta_post_loc")  #TODO: change this to use glorot or He initialization strategy
        self.conv1_tau_post_loc = tf.Variable( initial_value=np.random.lognormal(mean=0.5, sigma=1, size=[self.hparams['conv1_output_node_count']]), trainable=True, name="conv1_tau_post_loc" )
        self.conv1_U_psi = tf.Variable( initial_value= tf.random.uniform( tf.reshape( self.hparams['conv1_input_weights_count'] + 1*self.hparams['conv1_output_node_count'], [-1]), minval=0.05, maxval=1 ), trainable=True, name="conv1_U_psi") #TODO: Create reason for why an initialization scheme should be best
        self.conv1_U_h =   tf.Variable( initial_value= tf.random.normal( tf.reshape(self.hparams['conv1_input_weights_count'] + 1*self.hparams['conv1_output_node_count'], [-1]) , mean=0.05, stddev=1 ), trainable=True, name="conv1_U_h") #TODO: Create reason for why an initialization scheme should be best
        
        self.conv2_beta_post_loc = tf.Variable( initial_value=tf.random.uniform( [self.hparams['conv2_output_node_count'], self.hparams['conv2_input_weights_per_filter']] ,-1, 1), trainable=True, name="conv2_beta_post_loc")  #TODO: change this to use glorot or He initialization strategy
        self.conv2_tau_post_loc = tf.Variable( initial_value=np.random.lognormal(mean=0.5, sigma=1, size=[self.hparams['conv2_output_node_count']]), trainable=True, name="conv2_tau_post_loc" )
        self.conv2_U_psi = tf.Variable( initial_value= tf.random.uniform( tf.reshape( self.hparams['conv2_input_weights_count'] + 1*self.hparams['conv2_output_node_count'], [-1]), minval=0.05, maxval=1 ), trainable=True, name="conv2_U_psi") #TODO: Create reason for why an initialization scheme should be best
        self.conv2_U_h =   tf.Variable( initial_value= tf.random.normal( tf.reshape(self.hparams['conv2_input_weights_count'] + 1*self.hparams['conv2_output_node_count'], [-1]) , mean=0.05, stddev=1 ), trainable=True, name="conv2_U_h") #TODO: Create reason for why an initialization scheme should be best

        # Output Layer weights Posterior
        self.conv3_kappa_post_loc = tf.Variable( initial_value = tf.random.uniform([self.hparams['conv3_input_weights_count']],0.25,0.75 ), trainable=True, name="conv3_kappa_posterior_loc" )
        self.conv3_kappa_post_scale = tf.Variable( initial_value = tf.random.uniform([self.hparams['conv3_input_weights_count']],0.25,0.75 ), trainable=True, name="conv3_kappa_posterior_scale")    
        self.conv3_post_loc = tf.Variable( initial_value = tf.random.uniform( [self.hparams['conv3_input_weights_count']] ,-1,1 ), trainable=True, name="conv3_post_loc" )

    def update_priors_dists(self):
        self.conv3_weights_prior_dist = tfd.Normal(self.conv3_weights_prior_loc, self.conv3_weights_prior_scale)

    def update_posteriors_dists(self):
        #Global C Posterior
        self.c_post_dist = tfd.LogNormal(self.global_c_post_shape, self.global_c_post_scale)
        
        #Layerwise nu Posteriors
        self.conv1_nu_post_dist = tfd.LogNormal( self.conv1_nu_post_mean, self.conv1_nu_post_scale )
        self.conv2_nu_post_dist = tfd.LogNormal( self.conv2_nu_post_mean, self.conv2_nu_post_scale )

        #Layer weights
        conv1_logtau_post_loc = tf.expand_dims(tf.cast(tf.math.log( self.conv1_tau_post_loc) , dtype=tf.float32 ),-1)
        conv1_li_U_psi = tf.split( self.conv1_U_psi, self.hparams['conv1_params']['filters']  ) #len 10 shape (filterh*filter2w + 1) #TODO: This is new - explain in paper
        conv1_li_U_h = tf.split( self.conv1_U_h, self.hparams['conv1_params']['filters'] ) #len 10 shape (filterh*filter2w + 1) #TODO: This is new - explain in paper
        conv1_tf_U = tf.map_fn( lambda x: tf.linalg.diag(x[0]) + tf.einsum('i,j->ji',x[1], x[1]), tf.stack([conv1_li_U_psi, conv1_li_U_h], axis=1) ) #Matrix Normal Structured Variance for Weights connecting to a convolutional layer
        conv1_betalogtau_post_loc = tf.concat( [self.conv1_beta_post_loc, conv1_logtau_post_loc ], axis=-1) #TODO:(akanni-ade ) check if this is concat the correct way
        conv1_betalogtau_post_scale = tf.linalg.cholesky(conv1_tf_U)
        self.conv1_beta_logtau_post_dist = tfd.MultivariateNormalTriL(conv1_betalogtau_post_loc, conv1_betalogtau_post_scale)

        conv2_logtau_post_loc = tf.expand_dims(tf.cast(tf.math.log( self.conv2_tau_post_loc ), dtype=tf.float32),-1)
        conv2_li_U_psi = tf.split( self.conv2_U_psi, self.hparams['conv2_params']['filters']  ) #len 10 shape (filterh*filter2w + 1) #TODO: This is new - explain in paper
        conv2_li_U_h = tf.split( self.conv2_U_h, self.hparams['conv2_params']['filters'] ) #len 10 shape (filterh*filter2w + 1) #TODO: This is new - explain in paper
        conv2_tf_U = tf.map_fn( lambda x: tf.linalg.diag(x[0]) + tf.einsum('i,j->ji',x[1], x[1]), tf.stack([conv2_li_U_psi, conv2_li_U_h], axis=1) ) #Matrix Normal Structured Variance for Weights connecting to a convolutional layer
        conv2_betalogtau_post_loc = tf.concat( [self.conv2_beta_post_loc, conv2_logtau_post_loc ], axis=-1) #TODO:(akanni-ade ) check if this is concat the correct way
        conv2_betalogtau_post_scale = tf.linalg.cholesky(conv2_tf_U)
        self.conv2_beta_logtau_post_dist = tfd.MultivariateNormalTriL(conv2_betalogtau_post_loc, conv2_betalogtau_post_scale)

        self.conv3_kappa_post_dist = tfd.LogNormal(self.conv3_kappa_post_loc, self.conv3_kappa_post_scale )
        self.conv3_kappa = self.conv3_kappa_post_dist.sample()
        self.conv3_weights_dist = tfd.Normal( loc=self.conv3_post_loc , scale=self.conv3_kappa )

    def sample_params(self):
        self.conv1_nu = self.conv1_nu_post_dist.sample()
        self.conv2_nu = self.conv2_nu_post_dist.sample()
        self.c = self.c_post_dist.sample()

    def prior_cross_entropy(self):
        """
            This calculates the log likelihood of intermediate parameters
            I.e. equation 11 of Soumya Ghouse Structured Varioation Learning - This accounts for all terms except the KL_Div and the loglik(y|params)
            
            This is the prior part of the KL Divergence
        """
        #TODO: Check that the sign of all the losses is correct
        # Global Scale Param c
        ll_c = self.c_prior_dist.log_prob(self.c)


        # Layer-wise variance scaling nus
        ll_conv1_nu = tf.reduce_sum( self.conv1_nu_prior_dist.log_prob( self.conv1_nu ) )
        ll_conv2_nu = tf.reduce_sum( self.conv2_nu_prior_dist.log_prob( self.conv2_nu ) )
        ll_nu = ll_conv1_nu + ll_conv2_nu

        # Node level variaace scaling taus
        ll_conv1_tau = tf.reduce_sum( self.conv1_tau_prior_dist.log_prob( tf.reshape(self.conv1_taus,[-1]) ) ) 
        ll_conv2_tau = tf.reduce_sum( self.conv2_tau_prior_dist.log_prob( tf.reshape(self.conv2_taus,[-1]) ) )
        ll_tau = ll_conv1_tau + ll_conv2_tau

        # Nodel level mean centering Betas
        ll_conv1_beta = tf.reduce_sum( self.conv1_Beta_prior_dist.log_prob( self.conv1_Beta ) )
        ll_conv2_beta = tf.reduce_sum( self.conv2_Beta_prior_dist.log_prob(self.conv2_Beta ))  #TODO: come back to do this after you have refactored the beta code
        ll_conv3_weights = tf.reduce_sum( self.conv3_weights_prior_dist.log_prob( tf.reshape(self.conv3_weights,[-1]) ) ) #TODO: come back to do this after you have refactored the beta code

        ll_beta = ll_conv1_beta + ll_conv2_beta + ll_conv3_weights 
 
        #sum
        prior_cross_entropy = ll_c + ll_nu + ll_tau + ll_beta
        batch_avg_prior_cross_entropy = prior_cross_entropy/self.hparams['batch_size']
        
        self.add_loss(batch_avg_prior_cross_entropy)
    
    def posterior_entropy(self):

        # Node level mean: Beta and scale: Taus
        ll_conv1_beta_logtau = tf.reduce_sum(self.conv1_beta_logtau_post_dist.log_prob( self.conv1_beta_logtau )) #This needs to be made as in the other one
        ll_conv2_beta_logtau = tf.reduce_sum(self.conv2_beta_logtau_post_dist.log_prob( self.conv2_beta_logtau ))
        ll_beta_logtau = ll_conv1_beta_logtau + ll_conv2_beta_logtau

        # Node Level Mean: Output Layer
        ll_conv3_weights = tf.reduce_sum( self.conv3_weights_dist.log_prob( self.conv3_weights ) )
        
        # Nodel Level Variance scale: Output Layer
        #ll_conv3_logkappa = tf.reduce_sum( self.conv3_logkappa_post_dist.log_prob( self.conv3_logkappa )  )

        # global C
        ll_c = self.c_post_dist.log_prob(self.c)

        # Layer-wise variance scaling: nus
        ll_conv1_nu = self.conv1_nu_post_dist.log_prob(self.conv1_nu)
        ll_conv2_nu = self.conv2_nu_post_dist.log_prob(self.conv2_nu)
        ll_nu = ll_conv1_nu + ll_conv2_nu 

        # Sum
        posterior_entropy = ll_beta_logtau + ll_conv3_weights  + ll_c + ll_nu
        batch_avg_posterior_entropy = posterior_entropy/self.hparams['batch_size']
        
        self.add_loss(batch_avg_posterior_entropy)

class UpSampler():
    def __init__(self, input_dims, output_dims, upsample_method="ZeroPadding" , extra_outside_padding_method= "" ):

        """
            :params  input_dims: ( hieght, weight )
            :params  output_dims: ( hieght, weight )
            :params extra_outside_padding_method: ['CONSTANT' or ]
        """

        self.inp_dim_h = input_dims[0]
        self.inp_dim_w = input_dims[1]

        outp_dim_h = output_dims[0]
        outp_dim_w = output_dims[1]
        
        self.upsample_h = outp_dim_h - self.inp_dim_h                                   #amount to expand in height dimension
        self.upsample_w = outp_dim_w - self.inp_dim_w

        self.upsample_h_inner = outp_dim_h - (self.upsample_h % (self.inp_dim_h-1) )    #amount to expand in height dimension, w/ inner padding
        self.upsample_w_inner = outp_dim_w - (self.upsample_w % (self.inp_dim_w-1) )    #amount to expand in width dimension, w/ inner padding

        self.create_transformation_matrices()                                           #Creating the transformation matrices
        
        self.outside_padding  =  tf.constant( self.upsample_h_inner!=self.upsample_h or self.upsample_w_inner!=self.upsample_w ) #bool representing whether we need outside padding or not
        
        if( self.outside_padding ):
            outside_padding_h = (outp_dim_h - self.upsample_h_inner)
            self.outside_padding_h_u = tf.cast( tf.math.ceil(  outside_padding_h/2 ), dtype=tf.int32)
            self.outside_padding_h_d =  tf.cast( outside_padding_h - self.outside_padding_h_u, dtype=tf.int32)

            outside_padding_w = (outp_dim_w - self.upsample_w_inner)
            self.outside_padding_w_l = tf.cast( tf.math.ceil(  outside_padding_w/2 ), dtype=tf.int32)
            self.outside_padding_w_r = tf.cast( outside_padding_w - self.outside_padding_w_l, dtype=tf.int32)
            

    #@tf.function
    def __call__(self, _images):
        """
            :param _images: shape(batch, h, w, c)
        """
        upsampled_images =  tf.einsum( 'ij, hjkl -> hikl', self.T_1,_images)
        upsampled_images =  tf.einsum( 'ijkl, km -> ijml', upsampled_images,self.T_2)

        if( self.outside_padding ) :
            top = tf.expand_dims( upsampled_images[:, 1, ... ], 1)
            top = tf.tile(top, [1,self.outside_padding_h_u,1,1])
            bottom = tf.expand_dims( upsampled_images[:, -1, ... ], 1)
            bottom = tf.tile(bottom, [1,self.outside_padding_h_d,1,1])
            upsampled_images = tf.concat( [top, upsampled_images, bottom], axis=1 )

            left = tf.expand_dims( upsampled_images[:,:,1,...], 2)
            left = tf.tile(left, [1,1,self.outside_padding_w_l,1])
            right = tf.expand_dims( upsampled_images[:,:,-1,...], 2)
            right = tf.tile(right, [1,1,self.outside_padding_w_r,1])
            upsampled_images = tf.concat( [left, upsampled_images, right], axis=2 )

        return upsampled_images 

    def create_transformation_matrices(self ):
        """
            This creates the transformation matrices T_1 and T_2 which when applied to an image A in the form T_1 * A * T_2, lead to 2D upscaling with zeros in the middle
        """
        stride_h = int( (self.upsample_h_inner-self.inp_dim_h)/(self.inp_dim_h-1) )
        stride_w = int( (self.upsample_w_inner-self.inp_dim_w)/(self.inp_dim_w-1) )

        # Creating transformation matrices
        T_1 = np.zeros( (self.upsample_h_inner, self.inp_dim_h) )
        T_2 = np.zeros( (self.inp_dim_w, self.upsample_w_inner ) )

        d1 = np.einsum('ii->i', T_1[ ::stride_h+1,:] ) 
        d1 += 1

        d2 = np.einsum('ii->i', T_2[ :, ::stride_w+1] )
        d2 += 1

        self.T_1 = tf.constant( T_1, dtype=tf.float32)
        self.T_2 = tf.constant( T_2, dtype=tf.float32)

def horseshoe_kernel_posterior_distribution(obj, _conv_layer_no ):
    """
        Implements the Non-Centred Weight Distribution which is used in the kernels
        Samples from multi-variate Guassian distribution for values of Beta and tau
    """
    conv_layer = _conv_layer_no
    def _fn(dtype, shape, name, trainable, add_variable_fn ):
        betalogtau_dist = tf.case( [ (tf.equal(conv_layer,tf.constant(1)),lambda: obj.conv1_beta_logtau_post_dist), (tf.equal(conv_layer,tf.constant(2)), lambda: obj.conv2_beta_logtau_post_dist) ] , exclusive=True )
        return betalogtau_dist
    
    return _fn
   
def horsehoe_kernel_posterior_tensor_fn( betalogtau_dist, layer_nu , model_global_c, output_count, kernel_shape, inp_channels, tape=None, conv_layer=1, obj=None ):
    """
        :param filter_shape list: [h, w]
    """
    #TODO: use tf.case methodology here as opposed to passing in layer_nu, input_count, output_count and kernel hape
    _samples = betalogtau_dist.sample()

    beta = _samples[ : , :-1 ]
    taus_local_scale = tf.math.exp(_samples[ : , -1: ])

    def f1(): obj.conv1_Beta, obj.conv1_taus, obj.conv1_beta_logtau = beta,  taus_local_scale, _samples
    def f2(): obj.conv2_Beta, obj.conv2_taus, obj.conv2_beta_logtau = beta,  taus_local_scale, _samples
    tf.case( [ (tf.equal(conv_layer,tf.constant(1)),f1), (tf.equal(conv_layer,tf.constant(2)), f2) ], exclusive=True )

    #creating the non_centred_version of tau
    model_global_c2 = tf.square( model_global_c )
    taus_local_scale_2 = tf.square( taus_local_scale )
    layer_nu_2 = tf.square( layer_nu )
    taus_local_scale_regularlized = tf.math.sqrt( tf.math.divide( model_global_c2 * taus_local_scale_2, model_global_c2 + taus_local_scale_2*layer_nu_2 ) )

    _weights = tf.multiply( tf.multiply( beta,  taus_local_scale_regularlized), layer_nu ) #TODO:(akanni-ade) check this formula is correct #shape(10, 153)
        #shape ([filter_height, filter_width, in_channels, output_channels]) 
    _weights = tf.transpose( _weights )         #shape

    _weights = tf.reshape( _weights, [kernel_shape[0], kernel_shape[1], inp_channels, output_count] ) #shape
    return _weights

def HalfCauchy_Guassian_posterior_distribution(obj):
    
    def _fn(dtype, shape, name, trainable, add_variable_fn):
        dist = obj.conv3_weights_dist
        return dist
    return _fn

def HalfCauchy_Guassian_posterior_tensor_fn(obj, dist_weights, kernel_shape, inp_channels):    
    _weights = dist_weights.sample()
    _weights = tf.transpose( _weights )
    _weights = tf.reshape( _weights, [kernel_shape[0], kernel_shape[1], inp_channels, 1] )
    obj.conv3_weights = _weights
    return _weights

# endregion


# data pipeline
def load_data( batches_to_skip, hparams, _num_parallel_calls =tf.data.experimental.AUTOTUNE):

    # region prepare elevation
    _path = "Data/Preprocessed/elevation.pkl" #TODO:(akanni-ade) change to value passed in via a h-parameters dictionary
    with open(_path,"rb") as f: #TODO:(akanni-ade) change to value passed in via a h-parameters dictionary
        arr_elev = pickle.load(f)
        
    arr_elev = arr_elev[::4, ::4]  #shape( 156,352 ) #16kmby16km
        #creating layered representation of 16kmby16km arr_elev such that it is same shape as 64kmby64km precip
            ##take cell i,j in 2d array. each cell in the square matrix around cell i,j is stacked underneath i,j. 
            ## The square has dimensions (rel to i,j): 2 to the right, 2 down, 1 left, 1 right
            ## This achieves a dimension reduction of 4
    #region Elevation Preprocess
    AVG_ELEV = np.nanmean(arr_elev) #TODO: (akanni-ade) Find actual max elev
    arr_elev = arr_elev / AVG_ELEV 
        #now making all elevation values above 1, then converting all nan values to 0
    arr_elev = arr_elev + np.min(arr_elev) + 1
    arr_elev = np.nan_to_num( arr_elev, nan=0.0, posinf=0.0, neginf=0.0 )
    # endregion

    def stacked_reshape( arr, first_centre, downscale_x, downscale_y, batch_size = hparams['batch_size'] ):
        """
            This produces a list of tiled arrays. This ensures higher resolution data _arr has the same shape as lower resolution data, ignoring a depth dimension
            i.e.

            The array is stacked to create a new dimension (axis=-1). The stack happens on the following cells:
                first centre (i,j)
                (i+n*downscale_x, j+n*downscale_y ) for integer n

            :param nparray arr: 2D array  #shape( 156,352 ) #16kmby16km
            :param tuple first centre: tuple (i,j) indexing where the upperleft most position to stack on
            

            returns arr_stacked
        """
        # region standardisation
        
        MAX_ELEV = 2500 #TODO: (akanni-ade) Find true value, incorporate into Hparams
        arr = arr / MAX_ELEV 
        # end region

        new_depth = downscale_x * downscale_y
        dim_x, dim_y = arr.shape
        li_arr = []

        idxs_x = list( range(downscale_x) )
        idxs_y = list( range(downscale_y) )
        starting_idxs = list( itertools.product( idxs_x, idxs_y ) )

        for x,y in starting_idxs:
            #This helps define the different boundaries for the tiled elevation
            end_x = dim_x - ( downscale_x-first_centre[0] - x) 
            end_y = dim_y - ( downscale_y-first_centre[1] - y)
            arr_cropped = arr[ x:end_x, y:end_y ]

            li_arr.append(arr_cropped)

        li_tnsr = [ tf.expand_dims(_arr[::downscale_x, ::downscale_y],0) for _arr in li_arr ]
        li_tnsr_elev =  [ tf.tile(_tnsr,[batch_size,1,1]) for _tnsr in li_tnsr ]
        #tnsr_elev_tiled = tf.stack(li_tnsr_elev, axis=-1)
        
        #arr_stacked = np.stack( li_arr, axis=-1 )
        #arr_stacked = arr_stacked[::downscale_x, ::downscale_y] 

        return li_tnsr_elev

    li_elev_tiled = stacked_reshape( arr_elev, (1,1), 4, 4  ) # list[ (1, 39, 88), (1, 39, 88), ... ] #16kmby16km 

    # endregion


    # region features, targets
    _dir_precip = "./Data/PRISM/daily_precip"
    file_paths_bil = list( glob.glob(_dir_precip+"/*/*.bil" ) )
    file_paths_bil.sort(reverse=False)

    ds_fns_precip = tf.data.Dataset.from_tensor_slices(file_paths_bil)

    ds_precip_imgs = ds_fns_precip.map( lambda fn: tf.py_function(utility.read_prism_precip,[fn], [tf.float32] ) )#, num_parallel_calls=_num_parallel_calls ) #shape(bs, 621, 1405) #4km by 4km

    ds_precip_imgs = ds_precip_imgs.batch(hparams['batch_size'] ,drop_remainder=True)

    def features_labels_mker( arr_images, li_elev_tiled=li_elev_tiled ):
        """Produces the precipitation features and the target for training
        shape(bs, rows, cols)
        """
        #standardisation and preprocess of nans
        MAX_RAIN = 200 #TODO:(akanni-ade) Find actual max rain
        arr_images = arr_images / MAX_RAIN
        arr_images = arr_images + 1 #TODO:(akanni-ade) remember to undo these preprocessing steps when looking to predict the future
        arr_images = replace_inf_nan(arr_images)

        #features
        precip_feat = reduce_res( arr_images, 16, 16 ) #shape(bs, 621/16, 1405/16) (bs, 39, 88)  64km by 64km


        feat = tf.stack( [precip_feat,*li_elev_tiled], axis=-1 ) #shape(bs, 39, 88, 17)  64km by 64km

        #targets        
        precip_tar = reduce_res( arr_images, 4, 4)   #shape( bs, 621/4, 1405/4 ) (bs,156,352) 16km by 16km

        #TODO(akanni-ade): consider applying cropping to remove large parts that are just water 
            # #cropping
            # precip_tar = precip_tar[:, : , : ]
            # feat = feat[:, :, :, :]

        return feat, precip_tar

    def reduce_res(arr_imgs, x_axis, y_axis):
        arr_imgs_red = arr_imgs[:,::x_axis, ::y_axis]
        return arr_imgs_red
    
    ds_precip_feat_tar = ds_precip_imgs.map( features_labels_mker)#, num_parallel_calls=_num_parallel_calls ) #shape( (bs, 39, 88, 17 ) (bs,156,352) )
    ds_precip_feat_tar = ds_precip_feat_tar
        # endregion

    ds_precip_feat_tar = ds_precip_feat_tar.prefetch(buffer_size=_num_parallel_calls)

    return ds_precip_feat_tar

# region train --- Debugging Eagerly
def train_loop(train_params):
    print("GPU Available: ", tf.test.is_gpu_available() )
    
    model = SuperResolutionModel( train_params)

    # region ----- Defining Losses and Metrics   
    optimizer = tf.optimizers.Adam(lr=1e-2) 
    # radam = tfa.optimizers.RectifiedAdam( learning_rate=1e-2, total_steps=5000, warmup_proportion=0.1 ,min_lr = 1e-4, beta_1=0.9, beta_2=0.98, epsilon=1e-1, sma_threshold=5.0 , decay=1e-5 )
    # optimizer = tfa.optimizers.Lookahead(radam, sync_period = 9, slow_step_size=0.5 )

    train_metric_mse_mean_groupbatch = tf.keras.metrics.Mean(name='train_loss_mse_obj')
    train_metric_mse_mean_epoch = tf.keras.metrics.Mean(name="train_loss_mse_obj_epoch")

    train_loss_elbo_mean_groupbatch = tf.keras.metrics.Mean(name='train_loss_elbo_obj ')
    train_loss_elbo_mean_epoch = tf.keras.metrics.Mean(name="train_loss_elbo_obj_epoch")

    val_metric_mse_mean = tf.keras.metrics.Mean(name='val_metric_mse_obj')

    try:
        df_training_info = pd.read_csv( "checkpoints/checkpoint_scores_model_{}.csv".format(train_params['model_version']), header=0, index_col =False   )
        print("Recovered checkpoint scores model csv")

    except Exception as e:
        df_training_info = pd.DataFrame(columns=['Epoch','Train_loss_MSE','Val_loss_MSE','Checkpoint_Path', 'Last_Trained_Batch'] ) #key: epoch number #Value: the corresponding loss #TODO: Implement early stopping
        print("Did not recover checkpoint scores model csv")
  
    # endregion

    # region ----- Setting up Checkpoint (For Epochs)
    checkpoint_path_epoch = "checkpoints/epoch/{}".format(train_params['model_version'])
    if not os.path.exists(checkpoint_path_epoch):
        os.makedirs(checkpoint_path_epoch)
        
        #Create the checkpoint path and the checpoint manager. This will be used to save checkpoints every n epochs
    ckpt_epoch = tf.train.Checkpoint(att_con=model, optimizer=optimizer)
    ckpt_manager_epoch = tf.train.CheckpointManager(ckpt_epoch, checkpoint_path_epoch, max_to_keep=train_params['checkpoints_to_keep'], keep_checkpoint_every_n_hours=None)
    # endregion
    
    # region --- Setting up Checkpoints (For Batches) and restoring from last batch if it exists
    checkpoint_path_batch = "checkpoints/batch/{}".format(train_params['model_version'])
    if not os.path.exists(checkpoint_path_batch):
        os.makedirs(checkpoint_path_batch)
        #Create the checkpoint path and the checpoint manager. This will be used to save checkpoints every n epochs
    ckpt_batch = tf.train.Checkpoint(att_con=model, optimizer=optimizer)
    ckpt_manager_batch = tf.train.CheckpointManager(ckpt_batch, checkpoint_path_batch, max_to_keep=train_params['checkpoints_to_keep'], keep_checkpoint_every_n_hours=None)


    if ckpt_manager_batch.latest_checkpoint: #restoring last checkpoint if it exists
        ckpt_batch.restore(ckpt_manager_batch.latest_checkpoint)
        print ('Latest checkpoint restored from {}'.format(ckpt_manager_batch.latest_checkpoint  ) )

    else:
        print (' Initializing from scratch')

    # endregion     

    # region --- Setting up training parameters - to be moved to hparams file
    train_set_size_batches= 64*200
    val_set_size = 64*25
    
    train_batch_reporting_freq = train_set_size_batches*train_params['dataset_trainval_batch_reporting_freq']
    val_batch_reporting_freq = val_set_size*train_params['dataset_trainval_batch_reporting_freq']
    #endregion

    # region Logic for setting up resume location
    starting_epoch =  int(max( df_training_info['Epoch'], default=0 )) 
    df_batch_record = df_training_info.loc[ df_training_info['Epoch'] == starting_epoch,'Last_Trained_Batch' ]

    if( len(df_batch_record)==0 ):
        batches_to_skip = 0
    elif( df_batch_record.iloc[0]==-1 ):
        batches_to_skip = 0
    else:
        batches_to_skip = int(df_batch_record.iloc[0])   
    
    batches_to_skip_on_error = 2
    # endregion

    # region --- Tensorboard
    os.makedirs("log_tensboard/{}".format(train_params['model_version']), exist_ok=True )
    writer = tf.summary.create_file_writer( "log_tensboard/{}/tblog".format(train_params['model_version']) )
    # endregion

    # region --- Train and Validation
    for epoch in range(starting_epoch, int(train_params['epochs']+1) ):
        #region metrics, loss, dataset, and standardization
        train_metric_mse_mean_groupbatch.reset_states()
        train_loss_elbo_mean_groupbatch.reset_states()
        train_metric_mse_mean_epoch.reset_states()
        train_loss_elbo_mean_epoch.reset_states()

        val_metric_mse_mean.reset_states()

        if( epoch == starting_epoch):            
            ds = load_data( batches_to_skip, hparams  )

        df_training_info = df_training_info.append( { 'Epoch':epoch, 'Last_Trained_Batch':0 }, ignore_index=True )
        
        start_epoch = time.time()
        start_epoch_val = None
        
        inp_time = None
        start_batch_time = time.time()
        #endregion
        epoch_finished = False  
        while(epoch_finished==False):
            try:
                print("\nStarting EPOCH {} Batch {}/{}".format(epoch, batches_to_skip+1, train_set_size_batches))
                with writer.as_default():
                    for batch, (feature, target) in enumerate( ds, 1+batches_to_skip ):
                        #shapes
                            #feature ( batch_size, 64 , 64 , 17 )
                            #target (batch_size, 64, 64, 2)
                        
                        #Train Loop
                        if( batch<= train_set_size_batches):

                            with tf.GradientTape(persistent=True) as tape:
                                preds = model( feature, tape=tape ) #TODO Debug, remove tape variable from model later

                                                                    #TODO(akanni-ade): remove (mask) eror for predictions that are water i.e. null
                                
                                #likelihood 1 - Independent Normal
                                noise_std = tfd.HalfNormal(scale=2)   #TODO:(akanni-ade) This should decrease exponentially during training #RESEARCH: NOVEL Addition #TODO:(akanni-ade) create tensorflow function to add this
                                                                        #NOTE: In the original Model Selection paper they use Guassian Likelihoods for loss with a precision (noise_std) that is Gamma(6,6)
                                preds = tf.reshape( preds, [-1] )
                                target = tf.reshape( target, [-1] )
                                preds_distribution_norm = tfd.Normal( loc=preds, scale=noise_std.sample() )  #Ensure you are drawing independent samples along the batch dimensions #Consider adding Multivariate Normal since within each batch elem there is a degree of corr                                
                                
                                #likelihood 2 - 0 Inflated Log-Normal
                                    #preds_distributions_lgnorm = tfd.LogNormal( loc=preds, scale =noise_std.sample() )
                                    #TODO(akanni-ade): To implement this you need to add another value to the value to predict; This will be binary 1, 0 representing probability it rained or not
                                                    #Then we have two likelihoods conditional upon rain. First the zero inflated one, 2nd a log normal one
                                                    # so likelihood is p1 + (1-p1)* 
                                
                                neg_log_likelihood = -tf.reduce_mean( preds_distribution_norm.log_prob( target ) )    
                                kl_loss = tf.math.reduce_sum( model.losses ) / hparams['batch_size']
                                elbo_loss = neg_log_likelihood - kl_loss
                                metric_mse = tf.keras.losses.MSE( target , preds )
                            
                            gradients = tape.gradient( elbo_loss, model.trainable_variables )
                            optimizer.apply_gradients( zip( gradients, model.trainable_variables ) )
                            
                            #region tensorboard weights
                            tf.summary.scalar('train_loss_elbo', elbo_loss , step = batch + epoch*train_set_size_batches )
                            tf.summary.scalar('train_metric_mse', metric_mse , step = batch + epoch*train_set_size_batches )

                            for grad, _tensor in zip( gradients, model.trainable_variables):
                                if grad is not None:
                                    tf.summary.histogram( "Grads: {}".format( _tensor.name ) , grad, step = batch + epoch*train_set_size_batches  )
                                    tf.summary.histogram( "Weights: {}".format(_tensor.name), _tensor , step = batch + epoch*train_set_size_batches )
                            #endregion

                            #region loss updates
                            train_loss_elbo_mean_groupbatch( elbo_loss )
                            train_loss_elbo_mean_epoch( elbo_loss )
                            train_metric_mse_mean_groupbatch( metric_mse )
                            train_metric_mse_mean_epoch( metric_mse )
                            # endregion
                        
                            #region training batch reporting
                            ckpt_manager_batch.save()
                            if( (batch%train_batch_reporting_freq)==0):
                                batches_report_time =  time.time() - start_batch_time 

                                print("\n\tBatch:{}/{}\t\tTrain MSE Loss: {:.4f} \t Time:{:.4f}".format(batch, train_set_size_batches, train_metric_mse_mean_groupbatch.result(), batches_report_time ) )
                                
                                est_completion_time_seconds = (batches_report_time/train_params['dataset_trainval_batch_reporting_freq']) * (train_set_size_batches - batch)/train_set_size_batches
                                est_completion_time_mins = est_completion_time_seconds/60
                                est_completion_time_hours = est_completion_time_mins/60
                                est_completion_time_days = est_completion_time_hours/24

                                print("\tEst.Epoch Time: mins:{:.1f}\t hours:{:.3f}\t days:{:.4f}".format(est_completion_time_mins,est_completion_time_hours,est_completion_time_days ) )

                            # Updating record of the last batch to be operated on in training epoch
                            df_training_info.loc[ ( df_training_info['Epoch']==epoch) , ['Last_Trained_Batch'] ] = batch
                            df_training_info.to_csv( path_or_buf="checkpoints/checkpoint_scores_model_{}.csv".format(train_params['model_version']), header=True, index=False )
                            # endregion
                            
                            continue  

                        # region  - prep params for val loop
                        if( batch == train_set_size_batches+1 ):
                            print('EPOCH {}:\tELBO: {:.3f} \tMSE: {:.3f}\tTime: {:.2f}'.format(epoch, train_loss_elbo_mean_epoch.result() ,train_metric_mse_mean_epoch.result(), (time.time()-start_epoch ) ) )
                            raise Exception("Skipping forward to Validation")

                        if( (batch == train_set_size_batches + batches_to_skip_on_error + 2 ) or ( batch>train_set_size_batches+ batches_to_skip_on_error + 2 and batch==batches_to_skip+1 ) ):                        
                            start_epoch_val = time.time()
                            start_batch_time = time.time()
                            
                        # endregion
                        
                        # region -- Validation Loop
                        if( batch >= train_set_size_batches + batches_to_skip_on_error + 2 ):
                            preds = model( feature )

                            val_metric_mse_mean( tf.keras.metrics.MSE(  feature, target )  )
                                
                            if ( batch%val_batch_reporting_freq==0 ):
                                batches_report_time =  - time.time() - start_batch_time

                                print("\tCompleted Validation Batch:{}/{} \t Time:{:.4f}".format( batch, val_set_size ,batches_report_time))
                                start_batch_time = time.time()
                                
                                est_completion_time_seconds = (batches_report_time/train_params['dataset_trainval_batch_reporting_freq']) *( 1 -  (batch/val_set_size ) )
                                est_completion_time_mins = est_completion_time_seconds/60
                                est_completion_time_hours = est_completion_time_mins/60
                                est_completion_time_days = est_completion_time_hours/24
                                print("\tEst. Epoch Validation Completion Time: mins:{:.1f}\t hours:{:.1}\t days:{:.2f}".format(est_completion_time_mins,est_completion_time_hours,est_completion_time_days ) )

                            continue
                        print("Epoch {} Validation Loss: MSE:{:.4f}\tTime:{:.4f}".format(epoch, val_metric_mse_mean.result(), time.time()-start_epoch_val  ) )
                        
                        # endregion
                    epoch_finished = True
                    batches_to_skip = 0  
            except (Exception, tf.errors.InvalidArgumentError, ValueError) as e1: #Incase some bad data has been read, then we are skipping over it and continuing the training/validation process
                try:
                    batches_to_skip = batch + batches_to_skip_on_error
                except Exception as e2:
                    batches_to_skip = batches_to_skip + batches_to_skip_on_error
                    #This was in case the model isnt able to produce one new iteration before failing - either from fresh start or from prev fail 
                
                if( batch == train_set_size_batches + 1 ):
                    #start of validation
                    ds = load_data( train_set_size_batches, hparams  ) 
                    continue

                ds = load_data( batches_to_skip, hparams )

                print("\Error Raised During inference loop batch {}. Check log_custom.csv and log_custom_errormsg.txt for details".format(batches_to_skip - batches_to_skip_on_error) )
                
                with open('log_custom_errormsg.txt', "a+") as f:
                    f.write("\n\n\nError occured with last batch = {}".format( batches_to_skip - batches_to_skip_on_error))
                    f.write("Number of data per training batch = {}\n".format(  train_params['batch_size']   )  )
                    f.write( traceback.format_exc() )

                print("Skipping forward {} batches in epoch".format(batches_to_skip_on_error))

        #region EPOCH Checkpoints 
        df_training_info = df_training_info[ df_training_info['Epoch'] != epoch ] #rmv current batch records for compatability with code below
        if( ( val_metric_mse_mean.result().numpy() <= max( df_training_info.loc[ : ,'Val_loss_MSE' ], default= val_metric_mse_mean.result().numpy()+1 ) ) ):
            print('Saving Checkpoint for epoch {}'.format(epoch)) 
            ckpt_save_path = ckpt_manager_epoch.save()

            
            # Possibly removing old non top5 records from end of epoch
            if( len(df_training_info.index) >= train_params['checkpoints_to_keep'] ):
                df_training_info = df_training_info.sort_values(by=['Val_loss_MSE'], ascending=False)
                df_training_info = df_training_info.iloc[:-1]
                df_training_info.reset_index(drop=True)

            
            df_training_info = df_training_info.append( other={ 'Epoch':epoch,'Train_loss_MSE':train_metric_mse_mean_epoch.result().numpy(), 'Val_loss_MSE':val_metric_mse_mean.result().numpy(),
                                                                'Checkpoint_Path': ckpt_save_path, 'Last_Trained_Batch':-1 }, ignore_index=True ) #A Train batch of -1 represents final batch of training step was completed

            print("\nTop {} Performance Scores".format(train_params['checkpoints_to_keep']))
            print(df_training_info[['Epoch','Val_loss_MSE']] )
            df_training_info = df_training_info.sort_values(by=['Val_loss_MSE'], ascending=True)
            df_training_info.to_csv( path_or_buf=train_params['scr_dir']+"/checkpoints/checkpoint_scores_model_{}.csv".format(train_params['model_version']), header=True, index=False ) #saving df of scores                      
            

        #Early iteration Stop Check
            #If largest epoch in dictionary is at least train_params['early_stopping_period']
        if( epoch >  max( df_training_info.loc[:, 'Epoch'], default=0 ) + train_params['early_stopping_period'] ):
            print("Model Early Stopping at EPOCH {}".format(epoch))
            break
        #endregion
            
    # endregion

    print("Model Training Finished")

# endregion

if __name__ == "__main__":
# region Defining Hyperparameters
    #TODO create new Hparams class to hold these values

    _num_parallel_calls =tf.data.experimental.AUTOTUNE 
    checkpoints_to_keep = 10
    model_version =1
    EPOCHS=2

    input_dims = [39, 88]
    output_dims = [156,352]

    #TODO: (change filter sizes back to the ones used in the paper)

    CONV1_params = {    'filters':10 ,
                        'kernel_size': [3,3] , #TODO:use size from paper later
                        'activation':'relu',
                        'padding':'same',
                        'data_format':'channels_last',
                        'name':"Conv1" }

    conv2_kernel_size = np.ceil( np.ceil( np.array(output_dims)/np.array(input_dims) )*1.5 )  #This makes sure that each filter in conv2, sees at least two of the real non zero values. The zero values occur due to the upscaling
    CONV2_params = {    'filters':10,
                        'kernel_size':  conv2_kernel_size.astype(np.int32).tolist() , #TODO:use size from paper later
                        #each kernel covers 2 non original values from the upsampled tensor
                        'activation':'relu',
                        'padding':'same',
                        'data_format':'channels_last',
                        "name":"Conv2" }

    CONV3_params = {
                        'filters':1,
                        'kernel_size':[2,2], #TODO:use size from paper later
                        'activation':'relu',
                        'padding':'same',
                        'data_format':'channels_last',
                        "name":"Conv3"  }

    var_model_type = "reparam"

    conv1_inp_channels = 17
    conv1_input_weights_per_filter = np.prod(CONV1_params['kernel_size']) * conv1_inp_channels
    conv1_input_weights_count =  CONV1_params['filters'] * conv1_input_weights_per_filter
    conv1_output_node_count = CONV1_params['filters']

    conv2_inp_channels = CONV1_params['filters']
    conv2_input_weights_per_filter = np.prod(CONV2_params['kernel_size']) * conv2_inp_channels
    conv2_input_weights_count = CONV2_params['filters'] * conv2_input_weights_per_filter
    conv2_output_node_count = CONV2_params['filters']

    conv3_inp_channels = CONV2_params['filters']
    conv3_input_weights_count = CONV3_params['filters'] * np.prod(CONV3_params['kernel_size'] ) * conv3_inp_channels
    conv3_output_node_count = CONV3_params['filters']

    hparams = {
        'input_dims':input_dims,
        'output_dims':output_dims,

        'conv1_params': CONV1_params,
        'conv2_params': CONV2_params,
        'conv3_params': CONV3_params,

        'conv1_input_weights_count':conv1_input_weights_count,
        'conv1_output_node_count':conv1_output_node_count,
        'conv1_inp_channels':conv1_inp_channels,
        'conv1_input_weights_per_filter': conv1_input_weights_per_filter,

        'conv2_input_weights_count':conv2_input_weights_count,
        'conv2_output_node_count':conv2_output_node_count,
        'conv2_inp_channels': conv2_inp_channels,
        'conv2_input_weights_per_filter':conv2_input_weights_per_filter,

        'conv3_input_weights_count': conv3_input_weights_count,
        'conv3_output_node_count':conv3_output_node_count,
        'conv3_inp_channels':conv3_inp_channels ,

        'var_model_type': var_model_type,
        'batch_size':10,
        'checkpoints_to_keep':checkpoints_to_keep,
        'model_version':model_version,
        'epochs':EPOCHS,
        'dataset_trainval_batch_reporting_freq':0.1
    }

    train_loop(hparams)

    # endregion



# region train -- Actual Implementation

    # from keras.layers import Input

    # x_in = Input( shape( 64,64,2 ) )
    # x = SRCNN(hparams)(x_in)
    # model2 = Model(x_in, x)

    # def neg_log_likelihood_func(y_obs, y_pred, sigma=noise):
    #     dist = tfd.Normal(loc=y_pred, scale=sigma)
    #     return -tf.reduce_mean(dist.log_prob(y_obs))

    # kl_loss = tf.math.reduce_sum( model.losses ) / batch_size
    # neg_log_likelihood = neg_log_likelihood_func
    # elbo_loss = neg_log_likelihood + kl_loss

    # _optimizer = tf.train.AdamOptimizer()

    # #train_op = tf.keras.optimizer.minimize( loss, var_lis=model2.weights )

    # model2.compile( loss = neg_log_likelihood_func  , optimizer=tf.keras.optimizers.Adam(lr=0.03) , metrics=['mse'])
    #ds = load_data()
    # model2.fit( ds.make_one_shot_iterator() , test_data, batch_size = hparams['batch_size'], epochs=100, verbose=3 )
    #https://github.com/tensorflow/probability/blob/r0.8/tensorflow_probability/python/layers/conv_variational.py#L660-L805


# endregion



# region predictions -- Actual Implementation

    # y_pred_list = []

    # for i in tqdm.tqdm(range(500)):

# endregion