import tensorflow as tf
import tensor2tensor as t2t #NOTE: using tensor2tensors implementation, may cause tensorflow2 incompatibility bugs
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

from tensorflow.keras.layers import Bidirectional, ConvLSTM2D
from tensor2tensor.layers.common_attention import multihead_attention

import layers_ConvLSTM2D

import time

# region DeepSD layers    
class SRCNN( tf.keras.layers.Layer ):
    """ 
        Super Resolution Convolutional Module
        ..info: Each SRCNN will have its own seperate set of horseshoe parameters. 
    """
    def __init__(self, train_params, model_params ):
        super( SRCNN, self ).__init__()
        
        self.model_params = model_params
        self.train_params = train_params                            
        
        if self.model_params['var_model_type'] == 'horseshoe_structured':
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
                    kernel_divergence_fn= None )
        
        if self.model_params['var_model_type'] == 'horseshoe_factorized':
            self.initialize_priors_dist()
            self.initialize_posteriors_vars()
            self.update_posteriors_dists() 
            self.update_priors_dists()
            self.sample_variational_params()  

            self.conv1 = tfpl.Convolution2DReparameterization( **self.model_params['conv1_params'] ,
                kernel_posterior_fn = horseshoe_kernel_posterior_distribution( self, 1 ), 
                kernel_posterior_tensor_fn= ( lambda dist: horsehoe_kernel_posterior_tensor_fn(dist, None , self.c ,
                                            self.model_params['conv1_output_node_count'], self.model_params['conv1_params']['kernel_size'], self.model_params['conv1_inp_channels'],
                                            self.tape, 1, self, self.conv1_beta, self.conv1_taus, self.conv1_nu) )  ,
                kernel_prior_fn = None,
                
                kernel_divergence_fn= None) #TODO: Figure out the appropriate posterior and prior for the bias
                                            #TODO: Move the calculation for the kl divergence into this constructor

            self.upSample = UpSampler( self.model_params['input_dims'], self.model_params['output_dims'] )
            
            self.conv2 = tfpl.Convolution2DReparameterization ( **self.model_params['conv2_params'] ,
                kernel_posterior_fn = horseshoe_kernel_posterior_distribution( self, 2), 
                
                kernel_posterior_tensor_fn= ( lambda dist: horsehoe_kernel_posterior_tensor_fn(dist, None , self.c ,
                                            self.model_params['conv2_output_node_count'], self.model_params['conv2_params']['kernel_size'], self.model_params['conv2_inp_channels'],
                                            self.tape, 1, self, self.conv2_beta, self.conv2_taus, self.conv2_nu) )  ,
                kernel_prior_fn = None,
                kernel_divergence_fn= None)

            self.conv3 = tfpl.Convolution2DReparameterization( **self.model_params['conv3_params'] , 
                    kernel_posterior_fn = HalfCauchy_Guassian_posterior_distribution( self ) , 
                    kernel_posterior_tensor_fn= lambda dist: HalfCauchy_Guassian_posterior_tensor_fn(self, dist, self.model_params['conv3_params']['kernel_size'], self.model_params['conv3_inp_channels'] ) ,

                    kernel_prior_fn = None , 
                    kernel_divergence_fn= None )   

        if self.model_params['var_model_type'] == 'guassian_factorized':

            self.conv1 = tfpl.Convolution2DFlipout( **self.model_params['conv1_params'] )

            self.upSample = UpSampler( self.model_params['input_dims'], self.model_params['output_dims'] )
            
            self.conv2 = tfpl.Convolution2DFlipout ( **self.model_params['conv2_params'] )

            self.conv3 = tfpl.Convolution2DFlipout( **self.model_params['conv3_params'] )                  
    
    def call( self, _input, tape=None ,upsample_method=tf.constant("zero_padding"), pred=False ): #( batch, height, width)
        self.tape = tape
        
        if pred==False and self.model_params['var_model_type'] in ['guassian_factorized']:
            self.conv1._built_kernel_divergence = False
            self.conv1._built_bias_divergence = False
            self.conv2._built_kernel_divergence = False
            self.conv2._built_bias_divergence = False
            self.conv3._built_kernel_divergence = False
            self.conv3._built_bias_divergence = False

        if self.model_params['var_model_type'] in ['horseshoe_factorized','horseshoe_structured']:
            if pred==False:
                self.update_posteriors_dists()
                self.update_priors_dists()        
            if pred==True:
                self.sample_variational_params()
        
        x = self.conv1( _input )    #( batch, height_lr, width_lr, conv1_filters ) #TODO:(akanni-ade) alot of zero values for x at this output
        
        x = self.upSample( x )           #(batch, height_hr, width_hr, conv1_filters )

        x = self.conv2( x )         #(batch, height_hr, width_hr, conv2_filters )
        #TODO:(akanni-ade) add layer norm or batch norm here
        x = self.conv3( x )       #(batch, height_hr, width_hr, 1 )
        
        if self.model_params['var_model_type'] in ['horseshoe_factorized','horseshoe_structured']:
            if pred==False:
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
        
        if(self.model_params['var_model_type'] == "horseshoe_structured"):
            #NOTE: I think we treat c as a value that changes during trainable, but that doesnt have a distribution. In fact it just gets updated at each step
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
        
        elif(self.model_params['var_model_type'] == "horseshoe_factorized"):
            #Global C Posterior
            self.global_c_post_shape = tf.Variable( initial_value=1.0, trainable=True, dtype=tf.float32) 
            self.global_c_post_scale = tf.Variable( initial_value=0.2,  trainable=True, dtype=tf.float32)

            # Layerwise nu Posterior
            self.conv1_nu_post_mean = tf.Variable( initial_value = tfd.HalfCauchy(loc=self.conv1_nu_prior_dist.mode(), scale=0.1).sample(), trainable=True, dtype=tf.float32 ) 
            self.conv1_nu_post_scale = tf.Variable( initial_value = tf.random.uniform( self.conv1_nu_post_mean.shape, 0.05 ,0.2), trainable=True, dtype=tf.float32)  #TODO: In the future this should be based on the variance used in the HS prior half Cauchy be relating the Half Cuachy scale to the Lognormal variance

            self.conv2_nu_post_mean = tf.Variable( initial_value = tfd.HalfCauchy(loc=self.conv2_nu_prior_dist.mode(), scale=0.1).sample(), trainable=True, dtype=tf.float32) 
            self.conv2_nu_post_scale = tf.Variable( initial_value = tf.random.uniform( self.conv2_nu_post_mean.shape, .05, 0.2), trainable=True, dtype=tf.float32)

            # Betas_LogTaus for Nodes Posterior
            conv1_scale = tf.constant(tf.cast(1.0 *tf.sqrt( 6. / (self.model_params['conv1_output_node_count'] + self.model_params['conv1_input_weights_per_filter'])), dtype=tf.float32)) #glorot uniform used in microsoft horeshoe
            self.conv1_beta_post_loc = tf.Variable( initial_value=tf.random.uniform( [self.model_params['conv1_output_node_count'], self.model_params['conv1_input_weights_per_filter']] ,-conv1_scale, conv1_scale), trainable=True, dtype=tf.float32)  #This uses Xavier init
            self.conv1_beta_post_scale = tf.Variable( initial_value=tf.random.uniform( self.conv1_beta_post_loc.shape, 0.75, 1.1 ), trainable=True, dtype=tf.float32 )
            self.conv1_tau_post_loc = tf.Variable( initial_value= tfd.HalfCauchy(loc=self.conv1_tau_prior_dist.mode(),scale=0.1).sample(), trainable=True, name="conv1_tau_post_loc",dtype=tf.float32 )
            self.conv1_tau_post_scale = tf.Variable( initial_value = tf.random.uniform( self.conv1_tau_post_loc.shape, .05, 0.2), trainable=True, dtype=tf.float32)
            
            conv2_scale = tf.constant(tf.cast(1.0 * tf.sqrt( 6. / (self.model_params['conv2_output_node_count'] + self.model_params['conv2_input_weights_per_filter'])), dtype=tf.float32)) #glorot uniform used in microsoft horeshoe
            self.conv2_beta_post_loc = tf.Variable( initial_value=tf.random.uniform( [self.model_params['conv2_output_node_count'], self.model_params['conv2_input_weights_per_filter']] ,-conv2_scale, conv2_scale), trainable=True, dtype=tf.float32)  #This uses Xavier init
            self.conv2_beta_post_scale = tf.Variable( initial_value=tf.random.uniform( self.conv2_beta_post_loc.shape, 0.75, 1.1 ), trainable=True, dtype=tf.float32 )
            self.conv2_tau_post_loc = tf.Variable( initial_value= tfd.HalfCauchy(loc=self.conv2_tau_prior_dist.mode(),scale=0.1).sample()  , trainable=True, name="conv2_tau_post_loc",dtype=tf.float32 )
            self.conv2_tau_post_scale = tf.Variable( initial_value = tf.random.uniform( self.conv2_tau_post_loc.shape, .05, 0.2), trainable=True, dtype=tf.float32)

            # Output Layer weights Posterior
            self.conv3_kappa_post_loc = tf.Variable( initial_value = 1.0, trainable=True, dtype=tf.float32) #My init strat
            self.conv3_kappa_post_scale = tf.Variable( initial_value = 0.1 , trainable=True, dtype=tf.float32)    #This needs to be low since, log_normal distribution has extrememly large tails, so using a value higher, save above 1, can lead to a large scale term and then exploding gradientss

            conv3_scale = tf.constant( tf.cast(1.0 * tf.sqrt( 6. / (self.model_params['conv3_output_node_count'] + self.model_params['conv3_input_weights_count'] )  ),dtype=tf.float32))
            self.conv3_weights_post_loc = tf.Variable( initial_value = tf.random.uniform( [self.model_params['conv3_input_weights_count']] , -conv3_scale, conv3_scale,dtype=tf.float32 ), trainable=True, dtype=tf.float32 )

    def update_priors_dists(self):
        self.conv3_weights_prior_dist = tfd.Normal(self.conv3_weights_prior_loc, self.conv3_kappa )

    def update_posteriors_dists(self):
        if(self.model_params['var_model_type'] == "horseshoe_structured"):
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
        
        elif(self.model_params['var_model_type'] == "horseshoe_factorized"):
            #Global C Posterior
            #self.global_c_post_dist = tfd.LogNormal(self.global_c_post_shape, self.global_c_post_scale)
            self.global_c_post_dist = tfd.HalfCauchy(self.global_c_post_shape, self.global_c_post_scale)
            
            #Layerwise nu Posteriors
            self.conv1_nu_post_dist = tfd.LogNormal( self.conv1_nu_post_mean, self.conv1_nu_post_scale )
            self.conv2_nu_post_dist = tfd.LogNormal( self.conv2_nu_post_mean, self.conv2_nu_post_scale )

            # Layer local scales
            # conv1_logtau_post_loc = tf.expand_dims(tf.cast(tf.math.log( self.conv1_tau_post_loc) , dtype=tf.float32 ),-1)
            self.conv1_tau_post_dist = tfd.LogNormal( loc = self.conv1_tau_post_loc, scale=self.conv1_tau_post_scale )
            self.conv2_tau_post_dist = tfd.LogNormal( loc = self.conv2_tau_post_loc, scale=self.conv2_tau_post_scale )

            # Layer local means
            self.conv1_beta_post_dist = tfd.Normal( loc=self.conv1_beta_post_loc, scale=self.conv1_beta_post_scale  )
            self.conv2_beta_post_dist = tfd.Normal( loc=self.conv2_beta_post_loc, scale=self.conv2_beta_post_scale  )

            self.conv3_kappa_post_dist = tfd.LogNormal(self.conv3_kappa_post_loc, self.conv3_kappa_post_scale )
            self.conv3_kappa = self.conv3_kappa_post_dist.sample() #This should technically be in sample_variational_params
            self.conv3_weights_dist = tfd.Normal( loc=self.conv3_weights_post_loc , scale=self.conv3_kappa )

    def sample_variational_params(self):
        if(self.model_params['var_model_type'] == "horseshoe_structured"):
            self.conv1_nu = self.conv1_nu_post_dist.sample() #TODO: currently this outputs 1, when in actuality it should be 
            self.conv2_nu = self.conv2_nu_post_dist.sample()

            self.c = self.global_c_post_dist.sample()

        elif(self.model_params['var_model_type'] == "horseshoe_factorized"):
            self.conv1_beta =   self.conv1_beta_post_dist.sample()
            self.conv2_beta =   self.conv2_beta_post_dist.sample()

            self.conv1_taus =   self.conv1_tau_post_dist.sample()
            self.conv2_taus =   self.conv2_tau_post_dist.sample()
            
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
        if(self.model_params['var_model_type'] == "horseshoe_structured"):
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
        
        elif(self.model_params['var_model_type'] == "horseshoe_factorized"):
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
            ll_conv1_beta = tf.reduce_sum( self.conv1_Beta_prior_dist.log_prob( self.conv1_beta ) )
            ll_conv2_beta = tf.reduce_sum( self.conv2_Beta_prior_dist.log_prob( self.conv2_beta ))  #TODO: come back to do this after you have refactored the beta code
            ll_betas = ll_conv1_beta + ll_conv2_beta

            ll_conv3_kappa = tf.reduce_sum( self.conv3_weights_prior_hyperprior_scaledist.log_prob(self.conv3_kappa) ) 
            ll_conv3_weights = tf.reduce_sum( self.conv3_weights_prior_dist.log_prob( tf.reshape(self.conv3_weights,[-1]) ) ) #TODO: come back to do this after you have refactored the beta code
            ll_output_layer = ll_conv3_kappa + ll_conv3_weights
            
            prior_cross_entropy = ll_c + ll_nus + ll_taus + ll_betas + ll_output_layer

        
        self.add_loss(-prior_cross_entropy)
    
    def posterior_entropy(self):
        
        if(self.model_params['var_model_type'] == "horseshoe_structured"):
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
            posterior_shannon_entropy = ll_beta_logtau + ll_c + ll_nu + ll_output_layer
            self.add_loss(posterior_shannon_entropy)
        
        elif(self.model_params['var_model_type'] == "horseshoe_factorized"):
            # Node level mean: Beta
            ll_conv1_beta = tf.reduce_sum(self.conv1_beta_post_dist.log_prob( self.conv1_beta ))
            ll_conv2_beta = tf.reduce_sum(self.conv2_beta_post_dist.log_prob( self.conv2_beta ))
            ll_conv_beta = ll_conv1_beta+ ll_conv2_beta 

            # Noe level scale: Taus
            ll_conv1_taus = tf.reduce_sum(self.conv1_beta_post_dist.log_prob( self.conv1_taus ))
            ll_conv2_taus = tf.reduce_sum(self.conv2_beta_post_dist.log_prob( self.conv2_taus ))
            ll_conv_taus = ll_conv1_taus + ll_conv2_taus 

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
            posterior_shannon_entropy = ll_conv_beta + ll_conv_taus  + ll_c + ll_nu + ll_output_layer
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

    
    var_model_type = obj.model_params['var_model_type']

    def _fn(dtype, shape, name, trainable, add_variable_fn ):
        
        if( var_model_type == "horseshoe_structured" ): 
            dist = tf.case( [ (tf.equal(conv_layer,tf.constant(1)),lambda: obj.conv1_beta_logtau_post_dist), (tf.equal(conv_layer,tf.constant(2)), lambda: obj.conv2_beta_logtau_post_dist) ] , exclusive=True )
        
        elif( var_model_type == "horseshoe_factorized" ):
            dist = tf.case( [ (tf.equal(conv_layer,tf.constant(1)),lambda: obj.conv1_beta_post_dist), (tf.equal(conv_layer,tf.constant(2)), lambda: obj.conv2_beta_post_dist) ] , exclusive=True )

        return dist
    
    return _fn
   
def horsehoe_kernel_posterior_tensor_fn( dist, layer_nu , model_global_c, output_count, kernel_shape, inp_channels, tape=None, conv_layer=1, obj=None,
                                        factorised_beta=None, factorised_tau=None, factorised_nu=None ):
    """
        :param filter_shape list: [h, w]
    """
        
    if( obj.model_params['var_model_type'] == "horseshoe_structured"): 
        _samples = dist.sample()
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
    
    elif(obj.model_params['var_model_type'] == "horseshoe_factorized"):
        #creating the non_centred_version of tau
        model_global_c2 = tf.square( model_global_c )
        taus_local_scale_2 = tf.square( factorised_tau )
        layer_nu_2 = tf.square( factorised_nu )
        taus_local_scale_regularlized = tf.math.sqrt( tf.math.divide( model_global_c2 * taus_local_scale_2, model_global_c2 + taus_local_scale_2*layer_nu_2 ) )

        _weights = tf.multiply( tf.multiply( factorised_beta,  taus_local_scale_regularlized), factorised_nu ) #TODO:(akanni-ade) check this formula is correct #shape(10, 153)
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

# endregion

# region THST layers
class THST_Encoder(tf.keras.layers.Layer ):
    def __init__(self, train_params, encoder_params):
        super( THST_Encoder, self ).__init__()
        self.encoder_params = encoder_params
        self.train_params = train_params

        #TODO: REFACTOR so that each of these conv layers is made automatically using a forloop and and iterating through a dictionary of params
        self.CLSTM_1 = THST_CLSTM_Input_Layer( train_params, encoder_params['CLSTMs_params'][0] )
        self.CLSTM_2 = THST_CLSTM_Attention_Layer( train_params, encoder_params['CLSTMs_params'][1], encoder_params['ATTN_params'][0], encoder_params['seq_len_factor_reduction'][0], self.encoder_params['num_of_splits'][0] )
        self.CLSTM_3 = THST_CLSTM_Attention_Layer( train_params, encoder_params['CLSTMs_params'][2], encoder_params['ATTN_params'][1], encoder_params['seq_len_factor_reduction'][1], self.encoder_params['num_of_splits'][1] )
        self.CLSTM_4 = THST_CLSTM_Attention_Layer( train_params, encoder_params['CLSTMs_params'][3], encoder_params['ATTN_params'][2], encoder_params['seq_len_factor_reduction'][2], self.encoder_params['num_of_splits'][2] )
        self.CLSTM_5 = THST_CLSTM_Attention_Layer( train_params, encoder_params['CLSTMs_params'][4], encoder_params['ATTN_params'][3], encoder_params['seq_len_factor_reduction'][3], self.encoder_params['num_of_splits'][3] )

        #Building Neccessary hidden states before calls
        # self.CLSTM_2.convLSTM.build( input_shape = ( train_params['batch_size'] , self.encoder_params['num_of_splits'][0], 100 , 140 , encoder_params['CLSTMs_params'][0]['filters']*2 ) )
        # self.CLSTM_3.convLSTM( tf.zeros(shape= [train_params['batch_size'] , self.encoder_params['num_of_splits'][1], 100 , 140 , encoder_params['CLSTMs_params'][0]['filters']*2] ) )

        # #self.CLSTM_3.convLSTM.build( input_shape = [ train_params['batch_size'] , self.encoder_params['num_of_splits'][1], 100 , 140 , encoder_params['CLSTMs_params'][1]['filters']*2 ] )
        # self.CLSTM_4.convLSTM.build( input_shape = [ train_params['batch_size'] , self.encoder_params['num_of_splits'][2], 100 , 140 , encoder_params['CLSTMs_params'][2]['filters']*2 ] )
        # self.CLSTM_5.convLSTM.build( input_shape = [ train_params['batch_size'] , self.encoder_params['num_of_splits'][3], 100 , 140 , encoder_params['CLSTMs_params'][3]['filters']*2 ] )

    def call(self, _input):
        """
            _input #shape( )
        """
        hidden_states_1 =  self.CLSTM_1( _input ) #(bs, seq_len_1, h, w, c1)

        hidden_states_2 = self.CLSTM_2( hidden_states_1) #(bs, seq_len_2, h, w, c2)

        hidden_states_3 = self.CLSTM_3( hidden_states_2 ) #(bs, seq_len_3, h, w, c3)

        hidden_states_4 = self.CLSTM_4( hidden_states_3 ) #(bs, seq_len_4, h, w, c3)

        hidden_states_5 = self.CLSTM_5( hidden_states_4 ) #(bs, seq_len_5, h, w, c3)

        return hidden_states_2, hidden_states_3, hidden_states_4, hidden_states_5    

class THST_Decoder(tf.keras.layers.Layer):
    def __init__(self, train_params ,decoder_params):
        """
        :param list decoder_params: a list of dictionaries of the contained LSTM's params
        """
        super( THST_Decoder, self ).__init__()
        self.decoder_params = decoder_params
        self.train_params = train_params

        #TODO: REFACTOR so that each of these conv layers is made automatically using a forloop and and iterating through a dictionary of params
        self.CLSTM_L4 = THST_CLSTM_Decoder_Layer( train_params, self.decoder_params['CLSTMs_params'][2], decoder_params['seq_len_factor_reduction'][2] )
        self.CLSTM_L3 = THST_CLSTM_Decoder_Layer( train_params, self.decoder_params['CLSTMs_params'][1],  decoder_params['seq_len_factor_reduction'][1] )
        self.CLSTM_L2 = THST_CLSTM_Decoder_Layer( train_params, self.decoder_params['CLSTMs_params'][0],  decoder_params['seq_len_factor_reduction'][0] )
        #self.CLSTM_L1 = THST_CLSTM_Decoder_Layer( train_params, self.decoder_params['CLSTMs_params'][0],  decoder_params['seq_len_factor_reduction'][0] )
        
    def call(self, hidden_states_2_enc, hidden_states_3_enc, hidden_states_4_enc, hidden_states_5_enc  ):

        #NOTE: make sure to include some sort of iterative reduction of span in each descent in decoder level

        hidden_states_l4 = self.CLSTM_L4( hidden_states_4_enc , hidden_states_5_enc)

        hidden_states_l3 = self.CLSTM_L3( hidden_states_3_enc, hidden_states_l4 ) #(bs, output_len2, height, width)

        hidden_states_l2 = self.CLSTM_L2( hidden_states_2_enc, hidden_states_l3 )     #(bs, output_len1, height, width)     

        #hidden_states_l1 = self.CLSTM_L1( hidden_states_1_enc, hidden_states_l2 )     #(bs, output_len1, height, width)     

        return hidden_states_l2

class THST_CLSTM_Input_Layer(tf.keras.layers.Layer):
    """
        This corresponds to the lower input layer
        rmrbr to add spatial LSTM
    """
    def __init__(self, train_params, layer_params ):
        super( THST_CLSTM_Input_Layer, self ).__init__()
        
        self.trainable = train_params['trainable']
        self.layer_params = layer_params #list of dictionaries containing params for all layers

        self.convLSTM = Bidirectional( ConvLSTM2D( **self.layer_params ), merge_mode=None ) 

    def call( self, _input ):
        #NOTE: consider addding multiple LSTM Layers to extract more latent features

        hidden_states_f, hidden_states_b = self.convLSTM( _input, training=self.trainable ) #(bs, seq_len_1, h, w, c)
        hidden_states = tf.concat([hidden_states_f, hidden_states_b],axis=-1)
               

        return hidden_states #(bs, seq_len_1, h, w, c*2)

class THST_CLSTM_Attention_Layer(tf.keras.layers.Layer):
    """
        This corresponds to all layers which use attention to weight the inputs from the layer below
    """
    def __init__(self, train_params, clstm_params, attn_params, seq_len_factor_reduction, num_of_splits ):
        super( THST_CLSTM_Attention_Layer, self ).__init__()

        self.trainable = train_params['trainable']
        self.num_of_splits = num_of_splits
        self.seq_len_factor_reduction = seq_len_factor_reduction
        #self.clstm_params = clstm_params #list of dictionaries containing params for all layers

        self.convLSTM = Bidirectional( ConvLSTM2D( **clstm_params ), merge_mode=None ) #stateful possibly set to True, return_state=True, return_sequences=True
        
        self.li_Attention2D_f = [ MultiHead2DAttention( trainable=self.trainable, layer_params = attn_params ) for attn_group in range( num_of_splits ) ]
        self.li_Attention2D_b = [ MultiHead2DAttention( trainable= self.trainable, layer_params = attn_params  ) for attn_group in range( num_of_splits ) ]

        self.shapes_set = 0
        self.h, self.w = (100,140)
        self.bs = train_params['batch_size']
        self.c = clstm_params['filters']*2
        # pass in output dimensions of previous layer

        self.convLSTM_attn = ConvLSTM2D_attn( **clstm_params, merge_mode=None )

    def call(self, input_hidden_states):
        """
        1) Break the hidden_states into chunks compatible with the number of CLSTM cells in this layer
        2a) DO Self attention on them to extract alternative pattern
        2b) Use the initial cell state (or hidden state) of the of the LSTMcellas 
        """
        # chunked_inp_hid_states = tf.split( input_hidden_states, num_or_size_splits=self.num_of_splits ,axis=1 ) #axis=1, split along non batch dimension
        #     #Note: input_hidden_states: if this is from a previous LSTM layer, Then the dimension will be (bs, seq_len, h, w, channels_forward + channels_backward) as we should concatenate feature maps from both directions
        
        # #Note: Commented out is the depracated attention idea, where I would use the cells current hidden state as the key
        #     # hidden_states_f, cell_states_f = self.convLSTM.forward_layer.states #NOTE: check that hidden state and cell state are output in correct order
        #     # hidden_states_b, cell_states_b = self.convLSTM.backward_layer.states #shape( bs, seq_len, h, w, c) #hidden states will be the queries for attn
            
        #     # li_queries_f = tf.split( hidden_states_f, hidden_states_f.shape[1] , axis=1 ) #[ ( bs, 1, h, w, c), ... ] 
        #     # li_queries_b = tf.split( hidden_states_b, hidden_states_b.shape[1] , axis=1 )

        # li_queries_f = [ tf.reduce_mean(chunk, axis=1, keepdims=True )  for chunk in chunked_inp_hid_states ]
        # li_queries_b = li_queries_f

        # if(self.shapes_set==0):
        #     #bs, seq_len, h, w, c = tf.unstack(hidden_states_f.shape)
        #     self.seq_len = tf.cast( tf.shape(input_hidden_states)[1]/self.seq_len_factor_reduction, tf.int32 )
        #     self.shapes_set=1

        # attn_avg_inp_hid_states_f = tf.zeros( shape=[self.bs, self.seq_len, self.h, self.w, self.c], dtype=tf.float32 )
        # attn_avg_inp_hid_states_b = tf.zeros( shape=[self.bs, self.seq_len, self.h, self.w, self.c], dtype=tf.float32 )

        # for idx in tf.range(self.num_of_splits):
        #     attn_avg_inp_hid_state_f = self.li_Attention2D_f[idx]( li_queries_f[idx], chunked_inp_hid_states[idx] ) #(bs, 1, h, w, f)
        #     attn_avg_inp_hid_state_b = self.li_Attention2D_b[idx]( li_queries_b[idx], chunked_inp_hid_states[idx] ) #(bs, 1, h, w, f)

        #     attn_avg_inp_hid_states_f = tf.concat( [ attn_avg_inp_hid_states_f[ :, :idx , :, :, :] , attn_avg_inp_hid_state_f[ :, : ,:, :, :] , 
        #                                         tf.zeros([self.bs, (self.seq_len-1)-idx , self.h, self.w, self.c]) ] , axis=1 )     #shape( bs, seq_len, h, w, c1)
            
        #     attn_avg_inp_hid_states_b = tf.concat( [ attn_avg_inp_hid_states_b[ :, :idx , :, :, :] , attn_avg_inp_hid_state_b[ :, : ,:, :, :] , 
        #                                         tf.zeros([self.bs, (self.seq_len-1)-idx , self.h, self.w, self.c]) ] , axis=1 )     #shape( bs, seq_len, h, w, c1)

        # attn_avg_inp_hid_states = tf.concat( [attn_avg_inp_hid_states_f, attn_avg_inp_hid_states_b] , axis=-1 ) #shape( bs, seq_len, h, w, 2*c1)

        # forward_h, backward_h = self.convLSTM( attn_avg_inp_hid_states, training=self.trainable ) #shape( bs, seq_len, h, w, 2*c2) #c2 specified in ConvLSTM creation

        # output_hidden_states = tf.concat( [forward_h, backward_h], axis=-1) #shape( bs, seq_len, h, w, 2*c2)

        
        output_hidden_states = self.convLSTM_attn(.)

        return output_hidden_states #shape(bs, seq_len, h, w, 2*c2)

class THST_CLSTM_Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self, train_params ,layer_params, input_2_factor_increase ):
        super( THST_CLSTM_Decoder_Layer, self ).__init__()
        
        self.layer_params = layer_params
        self.input_2_factor_increase = input_2_factor_increase
        self.trainable= train_params['trainable']

        self.convLSTM =  tf.keras.layers.Bidirectional( layers_ConvLSTM2D.ConvLSTM2D_custom(**layer_params)  , merge_mode=None)

    def call(self, input1, input2 ):
        """

            input1 : Contains the hidden representations from the corresponding layer in the encoder #(bs, seq_len1, h,w,c1)
            input2: Contains the hidden repr from the previous decoder layer #(bs, seq_len2, h,w,c2)

        """
        input2 = tf.keras.backend.repeat_elements( input2, self.input_2_factor_increase, axis=1)

        tf.debugging.assert_equal(tf.shape(input1),tf.shape(input2), message="Decoder Input1, and Input2 not the same size")

        inputs = tf.concat( [input1, input2], axis=-1 ) #NOTE: At this point both input1 and input2, must be the same shape
        hidden_states_f, hidden_states_b = self.convLSTM( inputs, training=self.trainable )
        hidden_states = tf.concat( [ hidden_states_f, hidden_states_b], axis=-1 )
        return hidden_states

class THST_OutputLayer(tf.keras.layers.Layer):
    def __init__(self, train_params, layer_params):
        """
            :param list layer_params: a list of dicts of params for the layers
        """
        super( THST_OutputLayer, self ).__init__()

        self.trainable = train_params['trainable']
        self.conv_hidden = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[0] ) )
        #TODO: may have to add an upscaling mechanism here if model fields are on a lower dimension than the rain data for ati data
        self.conv_output = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[1] ) )
    
    def call(self, _inputs ):
        """
        :param tnsr inputs: (bs, seq_len, h,w,c)
        """
        x = self.conv_hidden( _inputs )
        x = self.conv_output( x ) #shape (bs, height, width)
        return x

class MultiHead2DAttention(tf.keras.layers.Layer):
    def __init__(self, ,layer_params, kv_downscale_kernelshape, kv_downscale_stride, vector_kv_downscale_factor, trainable):

        """
            #TODO: prior to the attention possibly add something like squeeze and excitation to reweight the feature maps. But only in the first layer since taking in the original feature maps, as it shouldnt be needed after

            Either use 2D attention or try flattening nromal tensors to vectors so normal attention can be used
            Flattening used in https://arxiv.org/pdf/1904.09925.pdf, so will use there flattening method
        """
        super( MultiHead2DAttention, self ).__init__()

        self.trainable = trainable
        
        self.kv_downscale_kernelshape = layer_params['kv_downscale_kernelshape']
        self.kv_downscale_stride = layer_params['kv_downscale_stride']
        self.vector_downscale_factor = layer_params['vector_downscale_factor']
        
        for k in ['kv_downscale_kernelshape', 'kv_downscale_stride', 'vector_downscale_factor']: del layer_params[k]
        self.layer_params = layer_params

        self.ln1 = tf.keras.layers.LayerNormalization(axis=-1 , epsilon=1e-4 , trainable=trainable )

    def call(self, queries, keys, values):
        """
            Note: In the attention layer, keys and values are the same
            #TODO:Later consider alternative implementation where you cmbine the attention and lstm into one new layer. so the previous hidden state is used as the query for the current attention aggregation as opposed to just averaging
        """
        #https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

        #queries.shape #( batch_size, seq_len ,height,width,filters_in) # This the shape to convert objects back into at the end
        
        # region size reduction
        
        queries = tf.nn.avg_pool3d( queries, strides=self.kv_downscale_stride,
                                ksize=self.kv_downscale_kernelshape, padding="VALID")

        keys = tf.nn.avg_pool3d( keys, strides=self.kv_downscale_stride,
                                ksize=self.kv_downscale_kernelshape, padding="VALID")
        # endregion 

        if( self.layer_params['total_key_depth'] == 0) :
            key_depth = tf.math.reduce_prod( queries.shape[-3:] )
            value_depth = tf.math.reduce_prod( values.shape[-3:] )
            self.layer_params['total_key_depth'] = key_depth // self.vector_kv_downscale_factor
            self.layer_params['total_value_depth'] = value_depth // self.vector_kv_downscale_factor
            self.layer_params['output_depth'] = value_depth


        queries_flat = tf.reshape(queries, queries.shape.as_list()[:2]  + [-1] ) #( batch_size, seq_len, height*width*filters_in)
        keys_flat = tf.reshape( keys, keys.shape.as_list()[:2] +[-1] )
        values_flat = tf.reshape(values, values.shape.as_list()[:2] + [-1] )

        x = multihead_attention_custom(
            query_antecedent = queries_flat ,
            memory_antecedent = keys_flat,
            values_antecedent = values_flat,
            trainable=self.trainable,
            **self.layer_params   ) #( batch_size, seq_len, height*width*filters_in
        x = self.ln1(x) #( batch_size, seq_len, height*width*filters_in
        
        x = tf.reshape( x ,  values.shape ) #( batch_size, seq_len, height, width, filters_in)

        return x
