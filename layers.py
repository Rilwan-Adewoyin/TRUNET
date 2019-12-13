import tensorflow as tf
import tensorflow_probability as tfp
try:
    import tensorflow_addons as tfa
except:
    tfa=None
from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd

import os
import sys

import utility

import pandas as pd

import pickle

import math
import numpy as np



import time

    
class SRCNN( tf.keras.layers.Layer ):
    """ 
        Super Resolution Convolutional Module
        ..info: Each SRCNN will have its own seperate set of horseshoe parameters. 
    """
    def __init__(self, train_params, model_params ):
        super( SRCNN, self ).__init__()
        
        self.model_params = model_params
        self.train_params = train_params                            
        
        if self.model_params['var_model_type'] == 'reparam':
            self.initialize_priors_dist()
            self.initialize_posteriors_vars()
            self.update_posteriors_dists() 
            self.update_priors_dists()  

            self.conv1 = tfpl.Convolution2DReparameterization( **self.model_params['conv1_params'] ,
                kernel_posterior_fn = horseshoe_kernel_posterior_distribution( self, 1 ), 
                kernel_posterior_tensor_fn= ( lambda dist: horsehoe_kernel_posterior_tensor_fn(dist, self.conv1_nu , self.c ,
                                            self.model_params['conv1_output_node_count'], self.model_params['conv1_params']['kernel_size'], self.model_params['conv1_inp_channels'],
                                            self.tape, 1, self) )  ,
                kernel_prior_fn = None,
                
                kernel_divergence_fn= None) #TODO: Figure out the appropriate posterior and prior for the bias
                                            #TODO: Move the calculation for the kl divergence into this constructor

            self.upSample = UpSampler( self.model_params['input_dims'], self.model_params['output_dims'] )
            
            self.conv2 = tfpl.Convolution2DReparameterization ( **self.model_params['conv2_params'] ,
                kernel_posterior_fn = horseshoe_kernel_posterior_distribution( self, 2), 
                
                kernel_posterior_tensor_fn= ( lambda dist: horsehoe_kernel_posterior_tensor_fn(dist, self.conv2_nu , self.c,
                                            self.model_params['conv2_output_node_count'], self.model_params['conv2_params']['kernel_size'], self.model_params['conv2_inp_channels'],
                                            self.tape, 2, self) )  ,
                kernel_prior_fn = None,
                kernel_divergence_fn= None)

            self.conv3 = tfpl.Convolution2DReparameterization( **self.model_params['conv3_params'] , 
                    kernel_posterior_fn = HalfCauchy_Guassian_posterior_distribution( self ) , 
                    kernel_posterior_tensor_fn= lambda dist: HalfCauchy_Guassian_posterior_tensor_fn(self, dist, self.model_params['conv3_params']['kernel_size'], self.model_params['conv3_inp_channels'] ) ,

                    kernel_prior_fn = None , 
                    kernel_divergence_fn= None
                    )   

        if self.model_params['var_model_type'] == 'reparam_standard':

            self.conv1 = tfpl.Convolution2DFlipout( **self.model_params['conv1_params'] )

            self.upSample = UpSampler( self.model_params['input_dims'], self.model_params['output_dims'] )
            
            self.conv2 = tfpl.Convolution2DFlipout ( **self.model_params['conv2_params'] )

            self.conv3 = tfpl.Convolution2DFlipout( **self.model_params['conv3_params'] )                  
    
    def call( self, _input, tape ,upsample_method=tf.constant("zero_padding") ): #( batch, height, width)
        self.tape = tape
        
        if self.model_params['var_model_type'] == 'reparam':
            self.update_posteriors_dists()
            self.update_priors_dists()        
            self.sample_variational_params()
        
        x = self.conv1( _input )    #( batch, height_lr, width_lr, conv1_filters ) #TODO:(akanni-ade) alot of zero values for x at this output
        
        x = self.upSample( x )           #(batch, height_hr, width_hr, conv1_filters )

        x = self.conv2( x )         #(batch, height_hr, width_hr, conv2_filters )
        #TODO:(akanni-ade) add layer norm or batch norm here
        x = self.conv3( x )       #(batch, height_hr, width_hr, 1 )
        
        if self.model_params['var_model_type'] == 'reparam':
            self.prior_cross_entropy()
            self.posterior_entropy() #TODO check that layer losses are sent to model losses

        return x
    
    def initialize_priors_dist(self):
        #TODO: create intialization method for all these hyper priors

        # Global C Prior
        self.global_c2_priorshape = tf.constant( 2.0, name="global_c_priorshape" ) # ? in bayesian compression paper,  2 in Soumya Paper 2
        self.global_c2_priorscale = tf.constant( 6.0 , name="global_c_priorscale" ) # ? in bayesina compression,  6 in Soumya Paper 2
        self.c2_prior_dist = tfd.InverseGamma(self.global_c2_priorshape, self.global_c2_priorscale )
        

        # Layerwise nu Prior #NOTE: here we put one nu per filter which has 153 weights
        conv1_nu_shape = tf.constant( 0.0, dtype=tf.float32  ) #This value of 0 used in the microsoft paper
        conv1_nu_scale = tf.constant(tf.ones_like(conv1_nu_shape, dtype=tf.float32 ))/2 #b_g in Soumya paper #This is set to 1 in soumya paper 1, and 1 in microoft paper
        conv2_nu_shape = tf.constant( 0.0, dtype=tf.float32 ) #This value of 0 used in the microsoft paper
        conv2_nu_scale = tf.constant(tf.ones_like(conv2_nu_shape, dtype=tf.float32 ))/2 #b_g in Soumya paper #This is set to 1 in soumya paper 1, and 1 in microoft paper
        self.conv1_nu_prior_dist  = tfd.HalfCauchy( conv1_nu_shape, conv1_nu_scale) # Microsoft BNN use a value of 1
        self.conv2_nu_prior_dist  = tfd.HalfCauchy( conv2_nu_shape, conv2_nu_scale) # Microsoft BNN use a value of 1

        # Betas for Nodes Prior
        conv1_beta_prior_loc = tf.constant(tf.zeros([self.model_params['conv1_output_node_count'],self.model_params['conv1_input_weights_per_filter']], dtype=tf.float32  ))
        conv1_beta_prior_scale_diag = tf.constant(tf.ones( [self.model_params['conv1_input_weights_per_filter']], dtype=tf.float32 ))
        self.conv1_Beta_prior_dist = tfd.MultivariateNormalDiag(conv1_beta_prior_loc, conv1_beta_prior_scale_diag)

        conv2_beta_prior_loc = tf.constant(tf.zeros((self.model_params['conv2_output_node_count'],self.model_params['conv2_input_weights_per_filter']), dtype=tf.float32  ))
        conv2_beta_prior_scale_diag = tf.constant( tf.ones([self.model_params['conv2_input_weights_per_filter']],dtype=tf.float32 ) )
        self.conv2_Beta_prior_dist = tfd.MultivariateNormalDiag(conv2_beta_prior_loc, conv2_beta_prior_scale_diag )

        # Taus for Nodes Prior
        conv1_taus_prior_loc = tf.constant( tf.zeros([self.model_params['conv1_output_node_count'] , 1] , dtype=tf.float32) )
        conv1_taus_prior_scale = tf.constant( tf.ones_like( conv1_taus_prior_loc, dtype=tf.float32) ) #b_0 in Soumya Paper #1.0 used in microsoft paper
        self.conv1_tau_prior_dist = tfd.HalfCauchy(conv1_taus_prior_loc, conv1_taus_prior_scale)

        conv2_taus_prior_loc = tf.constant( tf.zeros( [self.model_params['conv2_output_node_count'] , 1], dtype=tf.float32 ) ) 
        conv2_taus_prior_scale = tf.constant( tf.ones_like( conv2_taus_prior_loc, dtype=tf.float32) ) #b_0 in Soumya Paper #1.0 used in microsoft paper
        self.conv2_tau_prior_dist = tfd.HalfCauchy(conv2_taus_prior_loc, conv2_taus_prior_scale)

        self.conv3_weights_prior_loc = tf.constant( tf.zeros( (self.model_params['conv3_input_weights_count']), dtype=tf.float32  ) )
        
        self.conv3_weights_prior_hyperprior_scaledist = tfd.HalfCauchy(0, 5)  #prior for kappa
        #self.conv3_weights_prior_dist = tfd.Normal(self.conv3_weights_prior_loc, self.conv3_kappa )

    def initialize_posteriors_vars(self):
        #NOTE: I think we treat c as a value that changes during training, but that doesnt have a distribution. In fact it just gets updated at each step
        #TODO: create an initialization method for all these params
        #Global C Posterior
        self.global_c_post_shape = tf.Variable( initial_value=1.0, trainable=True, name="global_c_post_shape",dtype=tf.float32) 
        self.global_c_post_scale = tf.Variable( initial_value=0.2,  trainable=True, name="global_c_post_scale",dtype=tf.float32) 
        
        # Layerwise nu Posterior
        
        self.conv1_nu_post_mean = tf.Variable( initial_value = tfd.HalfCauchy(loc=self.conv1_nu_prior_dist.mode(), scale=0.1).sample(), trainable=True, dtype=tf.float32, name="conv1_nu_post_mean") 
        self.conv1_nu_post_scale = tf.Variable( initial_value = tf.random.uniform( self.conv1_nu_post_mean.shape, 0.05 ,0.2), trainable=True, dtype=tf.float32, name="conv1_nu_post_scale")  #TODO: In the future this should be based on the variance used in the HS prior half Cauchy be relating the Half Cuachy scale to the Lognormal variance

        self.conv2_nu_post_mean = tf.Variable( initial_value = tfd.HalfCauchy(loc=self.conv2_nu_prior_dist.mode(), scale=0.1).sample(), trainable=True, name="conv2_nu_post_mean",dtype=tf.float32) 
        self.conv2_nu_post_scale = tf.Variable( initial_value = tf.random.uniform( self.conv2_nu_post_mean.shape, .05, 0.2), trainable=True, name="conv2_nu_post_scale",dtype=tf.float32) #TODO: change this intialization back
        
        # Betas_LogTaus for Nodes Posterior
        conv1_scale = tf.constant(tf.cast(1.0 *tf.sqrt( 6. / (self.model_params['conv1_output_node_count'] + self.model_params['conv1_input_weights_per_filter'] )), dtype=tf.float32)) #glorot uniform used in microsoft horeshoe
        self.conv1_beta_post_loc = tf.Variable( initial_value=tf.random.uniform( [self.model_params['conv1_output_node_count'], self.model_params['conv1_input_weights_per_filter']] ,-conv1_scale, conv1_scale), trainable=True, name="conv1_beta_post_loc",dtype=tf.float32)  #This uses Xavier init
        self.conv1_tau_post_loc = tf.Variable( initial_value= tfd.HalfCauchy(loc=self.conv1_tau_prior_dist.mode(),scale=0.1).sample()  , trainable=True, name="conv1_tau_post_loc",dtype=tf.float32 )
        self.conv1_U_psi = tf.Variable( initial_value= tf.random.uniform( [self.model_params['conv1_output_node_count'], self.model_params['conv1_input_weights_per_filter'] + 1], minval=0.95, maxval=1.05 ), trainable=True, name="conv1_U_psi",dtype=tf.float32) # My init strategy
        self.conv1_U_h =   tf.Variable( initial_value= tf.random.uniform( [self.model_params['conv1_output_node_count'], self.model_params['conv1_input_weights_per_filter'] + 1] , minval=-0.001, maxval=0.001 ), trainable=True, name="conv1_U_h",dtype=tf.float32) # My init strategy
        
        conv2_scale = tf.constant(tf.cast(1.0 * tf.sqrt( 6. / (self.model_params['conv2_output_node_count'] + self.model_params['conv2_input_weights_per_filter'])  ),dtype=tf.float32)) #glorot uniform used in microsoft horeshoe
        self.conv2_beta_post_loc = tf.Variable( initial_value=tf.random.uniform( [self.model_params['conv2_output_node_count'], self.model_params['conv2_input_weights_per_filter']] ,-conv2_scale, conv2_scale), trainable=True, name="conv2_beta_post_loc",dtype=tf.float32)  #This uses Xavier init
        self.conv2_tau_post_loc = tf.Variable( initial_value= tfd.HalfCauchy(loc=self.conv2_tau_prior_dist.mode(),scale=0.1).sample()  , trainable=True, name="conv2_tau_post_loc",dtype=tf.float32 )
        self.conv2_U_psi = tf.Variable( initial_value= tf.random.uniform( [self.model_params['conv2_output_node_count'],self.model_params['conv2_input_weights_per_filter'] + 1], minval=0.95, maxval=1.05 ), trainable=True, name="conv2_U_psi",dtype=tf.float32) # My init strategy
        self.conv2_U_h =   tf.Variable( initial_value= tf.random.uniform( [self.model_params['conv2_output_node_count'],self.model_params['conv2_input_weights_per_filter'] + 1],  minval=-0.001, maxval=0.001 ), trainable=True, name="conv2_U_h",dtype=tf.float32) # My init strategy

        # Output Layer weights Posterior
        self.conv3_kappa_post_loc = tf.Variable( initial_value = 1.0, trainable=True, name="conv3_kappa_posterior_loc",dtype=tf.float32) #My init strat
        self.conv3_kappa_post_scale = tf.Variable( initial_value = 0.1 , trainable=True, name="conv3_kappa_posterior_scale",dtype=tf.float32)    #This needs to be low since, log_normal distribution has extrememly large tails, so using a value higher, save above 1, can lead to a large scale term and then exploding gradientss
        
        conv3_scale = tf.constant( tf.cast(1.0 * tf.sqrt( 6. / (self.model_params['conv3_output_node_count'] + self.model_params['conv3_input_weights_count'] )  ),dtype=tf.float32))
        self.conv3_weights_post_loc = tf.Variable( initial_value = tf.random.uniform( [self.model_params['conv3_input_weights_count']] , -conv3_scale, conv3_scale,dtype=tf.float32 ), trainable=True, name="conv3_weights_post_loc",dtype=tf.float32 )

    def update_priors_dists(self):
        self.conv3_weights_prior_dist = tfd.Normal(self.conv3_weights_prior_loc, self.conv3_kappa )

    def update_posteriors_dists(self):
        #Global C Posterior
        #self.global_c_post_dist = tfd.LogNormal(self.global_c_post_shape, self.global_c_post_scale)
        self.global_c_post_dist = tfd.HalfCauchy(self.global_c_post_shape, self.global_c_post_scale)
        
        #Layerwise nu Posteriors
        self.conv1_nu_post_dist = tfd.LogNormal( self.conv1_nu_post_mean, self.conv1_nu_post_scale )
        self.conv2_nu_post_dist = tfd.LogNormal( self.conv2_nu_post_mean, self.conv2_nu_post_scale )

        #Layer weights
        # conv1_logtau_post_loc = tf.expand_dims(tf.cast(tf.math.log( self.conv1_tau_post_loc) , dtype=tf.float32 ),-1)
        conv1_logtau_post_loc = tf.math.log( self.conv1_tau_post_loc) 
        conv1_li_U_psi = tf.split( self.conv1_U_psi, self.model_params['conv1_params']['filters']  ) #len 10 shape (filterh*filter2w + 1) #TODO: This is new - explain in paper
        conv1_li_U_h = tf.split( self.conv1_U_h, self.model_params['conv1_params']['filters'] ) #len 10 shape (filterh*filter2w + 1) #TODO: This is new - explain in paper
        conv1_tf_U = tf.map_fn( lambda x: tf.linalg.diag(x[0]) + tf.einsum('i,j->ji',x[1], x[1]), tf.concat([conv1_li_U_psi, conv1_li_U_h], axis=1) ) #Matrix Normal Structured Variance for Weights connecting to a convolutional layer
        conv1_betalogtau_post_loc = tf.concat( [self.conv1_beta_post_loc, conv1_logtau_post_loc ], axis=-1) #TODO:(akanni-ade ) check if this is concat the correct way
        conv1_betalogtau_post_scale = tf.linalg.cholesky(conv1_tf_U)
        self.conv1_beta_logtau_post_dist = tfd.MultivariateNormalTriL(conv1_betalogtau_post_loc, conv1_betalogtau_post_scale)

        conv2_logtau_post_loc = tf.math.log( self.conv2_tau_post_loc )
        conv2_li_U_psi = tf.split( self.conv2_U_psi, self.model_params['conv2_params']['filters']  ) #len 10 shape (filterh*filter2w + 1) #TODO: This is new - explain in paper
        conv2_li_U_h = tf.split( self.conv2_U_h, self.model_params['conv2_params']['filters'] ) #len 10 shape (filterh*filter2w + 1) #TODO: This is new - explain in paper
        conv2_tf_U = tf.map_fn( lambda x: tf.linalg.diag(x[0]) + tf.einsum('i,j->ji',x[1], x[1]), tf.concat([conv2_li_U_psi, conv2_li_U_h], axis=1) ) #Matrix Normal Structured Variance for Weights connecting to a convolutional layer
        conv2_betalogtau_post_loc = tf.concat( [self.conv2_beta_post_loc, conv2_logtau_post_loc ], axis=-1) #TODO:(akanni-ade ) check if this is concat the correct way
        conv2_betalogtau_post_scale = tf.linalg.cholesky(conv2_tf_U)
        self.conv2_beta_logtau_post_dist = tfd.MultivariateNormalTriL(conv2_betalogtau_post_loc, conv2_betalogtau_post_scale, allow_nan_stats=False )

        self.conv3_kappa_post_dist = tfd.LogNormal(self.conv3_kappa_post_loc, self.conv3_kappa_post_scale )
        self.conv3_kappa = self.conv3_kappa_post_dist.sample() #This should technically be in sample_variational_params
        self.conv3_weights_dist = tfd.Normal( loc=self.conv3_weights_post_loc , scale=self.conv3_kappa )

    def sample_variational_params(self):
        self.conv1_nu = self.conv1_nu_post_dist.sample() #TODO: currently this outputs 1, when in actuality it should be 
        self.conv2_nu = self.conv2_nu_post_dist.sample()

        self.c = self.global_c_post_dist.sample()

    def prior_cross_entropy(self):
        
        """
            This calculates the log likelihood of intermediate parameters
            I.e. equation 11 of Soumya Ghouse Structured Varioation Learning - This accounts for all terms except the KL_Div and the loglik(y|params)
            
            This is the prior part of the KL Divergence
            The negative of this should be passed in the return
        """
        
        # Global Scale Param c
        ll_c = self.c2_prior_dist.log_prob(tf.square(self.c)) #TODO: Check the right prior distr for c is used

        # Layer-wise variance scaling nus
        ll_conv1_nu = tf.reduce_sum( self.conv1_nu_prior_dist.log_prob( self.conv1_nu ) )
        ll_conv2_nu = tf.reduce_sum( self.conv2_nu_prior_dist.log_prob( self.conv2_nu ) )
        ll_nus = ll_conv1_nu + ll_conv2_nu

        # Node level variaace scaling taus
        ll_conv1_tau = tf.reduce_sum( self.conv1_tau_prior_dist.log_prob( self.conv1_taus ) ) 
        ll_conv2_tau = tf.reduce_sum( self.conv2_tau_prior_dist.log_prob( self.conv2_taus ) )
        ll_taus = ll_conv1_tau + ll_conv2_tau

        # Nodel level mean centering Betas
        ll_conv1_beta = tf.reduce_sum( self.conv1_Beta_prior_dist.log_prob( self.conv1_Beta ) )
        ll_conv2_beta = tf.reduce_sum( self.conv2_Beta_prior_dist.log_prob(self.conv2_Beta ))  #TODO: come back to do this after you have refactored the beta code
        ll_betas = ll_conv1_beta + ll_conv2_beta

        ll_conv3_kappa = tf.reduce_sum( self.conv3_weights_prior_hyperprior_scaledist.log_prob(self.conv3_kappa) ) 
        ll_conv3_weights = tf.reduce_sum( self.conv3_weights_prior_dist.log_prob( tf.reshape(self.conv3_weights,[-1]) ) ) #TODO: come back to do this after you have refactored the beta code
        ll_output_layer = ll_conv3_kappa + ll_conv3_weights 
 
        #sum
        prior_cross_entropy = ll_c + ll_nus + ll_taus + ll_betas + ll_output_layer
        
        self.add_loss(-prior_cross_entropy)
    
    def posterior_entropy(self):
        
        # Node level mean: Beta and scale: Taus
        ll_conv1_beta_logtau = tf.reduce_sum(self.conv1_beta_logtau_post_dist.log_prob( self.conv1_beta_logtau )) 
        ll_conv2_beta_logtau = tf.reduce_sum(self.conv2_beta_logtau_post_dist.log_prob( self.conv2_beta_logtau ))
        ll_beta_logtau = ll_conv1_beta_logtau + ll_conv2_beta_logtau

        # Layer-wise variance scaling: nus
        ll_conv1_nu = tf.reduce_sum( self.conv1_nu_post_dist.log_prob(self.conv1_nu) )
        ll_conv2_nu = tf.reduce_sum( self.conv2_nu_post_dist.log_prob(self.conv2_nu) )
        ll_nu = ll_conv1_nu + ll_conv2_nu 

        # Output Layer
        ll_conv3_kappa = tf.reduce_sum( self.conv3_kappa_post_dist.log_prob( self.conv3_kappa )  )
        ll_conv3_weights = tf.reduce_sum( self.conv3_weights_dist.log_prob( self.conv3_weights ) )
        ll_output_layer = ll_conv3_kappa + ll_conv3_weights
        
        # global C
        ll_c = self.global_c_post_dist.log_prob(self.c)

        # Sum
        posterior_shannon_entropy = ll_beta_logtau + ll_conv3_weights  + ll_c + ll_nu + ll_output_layer
        self.add_loss(posterior_shannon_entropy)

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
    _weights = tf.transpose( _weights )         

    _weights = tf.reshape( _weights, [kernel_shape[0], kernel_shape[1], inp_channels, output_count] ) 
    return _weights #shape (filter_height, filter_width, k , inp_channels)

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
