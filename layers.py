import os
import sys
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import tensorflow as tf
try:
	import tensorflow_addons as tfa
except:
	tfa=None
from tensorflow_probability import layers as tfpl
from tensorflow_probability import distributions as tfd
import tensor2tensor as t2t #NOTE: using tensor2tensors implementation, may cause tensorflow2 incompatibility bugs
import tensorflow_probability as tfp


import utility

import pickle
import math

import time

from tensorflow.keras.layers import Bidirectional
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import math_ops, nn, clip_ops

import layers_ConvLSTM2D


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
		
		if self.model_params['model_type_settings']['var_model_type'] == 'horseshoestructured':
			self.initialize_priors_dist()
			self.initialize_posteriors_vars()
			self.update_posteriors_dists() 
			self.update_priors_dists()
			self.sample_variational_params()  

			self.conv1 = tfpl.Convolution2DReparameterization( **self.model_params['conv1_params'] ,
				kernel_posterior_fn = horseshoe_kernel_posterior_distribution( self, 1 ), 
				kernel_posterior_tensor_fn= ( lambda dist: horsehoe_kernel_posterior_tensor_fn(dist, self.conv1_nu , self.c ,
											self.model_params['conv1_output_node_count'], self.model_params['conv1_params']['kernel_size'], self.model_params['conv1_inp_channels'],
											 1, self) )  ,
				kernel_prior_fn = None,
				kernel_divergence_fn= None) 

			self.upSample = UpSampler( self.model_params['input_dims'], self.model_params['output_dims'] )
			
			self.conv2 = tfpl.Convolution2DReparameterization ( **self.model_params['conv2_params'] ,
				kernel_posterior_fn = horseshoe_kernel_posterior_distribution( self, 2), 
				
				kernel_posterior_tensor_fn= ( lambda dist: horsehoe_kernel_posterior_tensor_fn(dist, self.conv2_nu , self.c,
											self.model_params['conv2_output_node_count'], self.model_params['conv2_params']['kernel_size'], self.model_params['conv2_inp_channels'],
											2, self) )  ,
				kernel_prior_fn = None,
				kernel_divergence_fn= None)

			self.conv3 = tfpl.Convolution2DReparameterization( **self.model_params['conv3_params'] , 
					kernel_posterior_fn = HalfCauchy_Guassian_posterior_distribution( self ) , 
					kernel_posterior_tensor_fn= lambda dist: HalfCauchy_Guassian_posterior_tensor_fn(self, dist, self.model_params['conv3_params']['kernel_size'], self.model_params['conv3_inp_channels'] ) ,

					kernel_prior_fn = None , 
					kernel_divergence_fn= None )
		
		if self.model_params['model_type_settings']['var_model_type'] == 'horseshoefactorized':
			self.initialize_priors_dist()
			self.initialize_posteriors_vars()
			self.update_posteriors_dists() 
			self.update_priors_dists()
			self.sample_variational_params()  

			self.conv1 = tfpl.Convolution2DReparameterization( **self.model_params['conv1_params'] ,
				kernel_posterior_fn = horseshoe_kernel_posterior_distribution( self, 1 ), 
				kernel_posterior_tensor_fn= ( lambda dist: horsehoe_kernel_posterior_tensor_fn(dist, None , self.c ,
											self.model_params['conv1_output_node_count'], self.model_params['conv1_params']['kernel_size'], self.model_params['conv1_inp_channels'],
											 1, self, self.conv1_beta, self.conv1_taus, self.conv1_nu) )  ,
				kernel_prior_fn = None,
				
				kernel_divergence_fn= None) #TODO: Figure out the appropriate posterior and prior for the bias
											#TODO: Move the calculation for the kl divergence into this constructor

			self.upSample = UpSampler( self.model_params['input_dims'], self.model_params['output_dims'] )
			
			self.conv2 = tfpl.Convolution2DReparameterization ( **self.model_params['conv2_params'] ,
				kernel_posterior_fn = horseshoe_kernel_posterior_distribution( self, 2), 
				
				kernel_posterior_tensor_fn= ( lambda dist: horsehoe_kernel_posterior_tensor_fn(dist, None , self.c ,
											self.model_params['conv2_output_node_count'], self.model_params['conv2_params']['kernel_size'], self.model_params['conv2_inp_channels'],
											 1, self, self.conv2_beta, self.conv2_taus, self.conv2_nu) )  ,
				kernel_prior_fn = None,
				kernel_divergence_fn= None)

			self.conv3 = tfpl.Convolution2DReparameterization( **self.model_params['conv3_params'] , 
					kernel_posterior_fn = HalfCauchy_Guassian_posterior_distribution( self ) , 
					kernel_posterior_tensor_fn= lambda dist: HalfCauchy_Guassian_posterior_tensor_fn(self, dist, self.model_params['conv3_params']['kernel_size'], self.model_params['conv3_inp_channels'] ) ,

					kernel_prior_fn = None , 
					kernel_divergence_fn= None )   

		if self.model_params['model_type_settings']['var_model_type'] == 'flipout':

			self.conv1 = tfpl.Convolution2DFlipout( **self.model_params['conv1_params'] )
			self.conv1._dtype = 'float16'

			self.upSample = UpSampler( self.model_params['input_dims'], self.model_params['output_dims'] )
			
			self.conv2 = tfpl.Convolution2DFlipout ( **self.model_params['conv2_params'] )
			self.conv2._dtype = 'float16'

			self.conv3 = tfpl.Convolution2DFlipout( **self.model_params['conv3_params'] ) 
			self.conv3._dtype = 'float16'

			#self.conv1.dtype =                  

		if self.model_params['model_type_settings']['var_model_type'] == 'concrete_dropout':
			self.conv1 = SpatialConcreteDropout( tf.keras.layers.Conv2D( **self.model_params['conv1_params'] ) )
			
			self.upSample = UpSampler( self.model_params['input_dims'], self.model_params['output_dims'] )
			
			self.conv2 = SpatialConcreteDropout( tf.keras.layers.Conv2D( **self.model_params['conv2_params'] ) )

			self.conv3 = tf.keras.layers.Conv2D( **self.model_params['conv3_params'] )

	#@tf.function
	def call( self, _input ,upsample_method=tf.constant("zero_padding"), pred=False ): #( batch, height, width)
		
		
		if pred==False and self.model_params['model_type_settings']['var_model_type'] in ['flipout']:
			self.conv1._built_kernel_divergence = False
			self.conv1._built_bias_divergence = False
			self.conv2._built_kernel_divergence = False
			self.conv2._built_bias_divergence = False
			self.conv3._built_kernel_divergence = False
			self.conv3._built_bias_divergence = False

		elif self.model_params['model_type_settings']['var_model_type'] in ['horseshoefactorized','horseshoestructured']:
			if pred==False:
				self.update_posteriors_dists()
				self.update_priors_dists()
				self.sample_variational_params()        
			if pred==True:
				self.sample_variational_params()
		x = self.conv1( _input )    #( batch, height_lr, width_lr, conv1_filters ) #TODO:(akanni-ade) alot of zero values for x at this output
		
		x = self.upSample( x )           #(batch, height_hr, width_hr, conv1_filters )

		x = self.conv2( x )         #(batch, height_hr, width_hr, conv2_filters )

		x = self.conv3( x )       #(batch, height_hr, width_hr, 1 )
		
		if self.model_params['model_type_settings']['var_model_type'] in ['dropout']:
			if pred == False:
				self.conv1.add_loss()
				self.conv2.add_loss()

		if self.model_params['model_type_settings']['var_model_type'] in ['horseshoefactorized','horseshoestructured']:
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
		conv1_nu_shape = tf.constant( 0.0, dtype=tf.float16  ) #This value of 0 used in the microsoft paper
		conv1_nu_scale = tf.constant(tf.ones_like(conv1_nu_shape, dtype=tf.float16 )/2.0) #b_g in Soumya paper #This is set to 1 in soumya paper 1, and 1 in microoft paper
		
		conv2_nu_shape = tf.constant( 0.0, dtype=tf.float16 ) #This value of 0 used in the microsoft paper
		conv2_nu_scale = tf.constant(tf.ones_like(conv2_nu_shape, dtype=tf.float16 ))/2 #b_g in Soumya paper #This is set to 1 in soumya paper 1, and 1 in microoft paper
		self.conv1_nu_prior_dist  = tfd.HalfCauchy( conv1_nu_shape, conv1_nu_scale) # Microsoft BNN use a value of 1
		self.conv2_nu_prior_dist  = tfd.HalfCauchy( conv2_nu_shape, conv2_nu_scale) # Microsoft BNN use a value of 1

		# Betas for Nodes Prior
		conv1_beta_prior_loc = tf.constant(tf.zeros([self.model_params['conv1_output_node_count'],self.model_params['conv1_input_weights_per_filter']], dtype=tf.float16  ))
		conv1_beta_prior_scale_diag = tf.constant(tf.ones( [self.model_params['conv1_input_weights_per_filter']], dtype=tf.float16 ))
		self.conv1_Beta_prior_dist = tfd.MultivariateNormalDiag(conv1_beta_prior_loc, conv1_beta_prior_scale_diag)

		conv2_beta_prior_loc = tf.constant(tf.zeros((self.model_params['conv2_output_node_count'],self.model_params['conv2_input_weights_per_filter']), dtype=tf.float16  ))
		conv2_beta_prior_scale_diag = tf.constant( tf.ones([self.model_params['conv2_input_weights_per_filter']],dtype=tf.float16 ) )
		self.conv2_Beta_prior_dist = tfd.MultivariateNormalDiag(conv2_beta_prior_loc, conv2_beta_prior_scale_diag )

		# Taus for Nodes Prior
		conv1_taus_prior_loc = tf.constant( tf.zeros([self.model_params['conv1_output_node_count'] , 1] , dtype=tf.float16) )
		conv1_taus_prior_scale = tf.constant( tf.ones_like( conv1_taus_prior_loc, dtype=tf.float16) ) #b_0 in Soumya Paper #1.0 used in microsoft paper
		self.conv1_tau_prior_dist = tfd.HalfCauchy(conv1_taus_prior_loc, conv1_taus_prior_scale)

		conv2_taus_prior_loc = tf.constant( tf.zeros( [self.model_params['conv2_output_node_count'] , 1], dtype=tf.float16 ) ) 
		conv2_taus_prior_scale = tf.constant( tf.ones_like( conv2_taus_prior_loc, dtype=tf.float16) ) #b_0 in Soumya Paper #1.0 used in microsoft paper
		self.conv2_tau_prior_dist = tfd.HalfCauchy(conv2_taus_prior_loc, conv2_taus_prior_scale)

		self.conv3_weights_prior_loc = tf.Variable( tf.zeros( (self.model_params['conv3_input_weights_count']), dtype=tf.float16  ) )
		
		self.conv3_weights_prior_hyperprior_scaledist = tfd.HalfCauchy(0, 5)  #prior for kappa
		#self.conv3_weights_prior_dist = tfd.Normal(self.conv3_weights_prior_loc, self.conv3_kappa )

	def initialize_posteriors_vars(self):
		
		if(self.model_params['model_type_settings']['var_model_type'] == "horseshoestructured"):
			#NOTE: I think we treat c as a value that changes during trainable, but that doesnt have a distribution. In fact it just gets updated at each step
			#TODO: create an initialization method for all these params
			#Global C Posterior
			self.global_c_post_shape = tf.Variable( initial_value=1.0, trainable=True, name="global_c_post_shape",dtype=tf.float16) 
			self.global_c_post_scale = tf.Variable( initial_value=0.2,  trainable=True, name="global_c_post_scale",dtype=tf.float16) 
			
			# Layerwise nu Posterior
			self.conv1_nu_post_mean = tf.Variable( initial_value = tfd.HalfCauchy(loc=self.conv1_nu_prior_dist.mode(), scale=0.1).sample(), trainable=True, dtype=tf.float16, name="conv1_nu_post_mean") 
			self.conv1_nu_post_scale = tf.Variable( initial_value = tf.random.uniform( self.conv1_nu_post_mean.shape, 0.05 ,0.2), trainable=True, dtype=tf.float16, name="conv1_nu_post_scale")  #TODO: In the future this should be based on the variance used in the HS prior half Cauchy be relating the Half Cuachy scale to the Lognormal variance

			self.conv2_nu_post_mean = tf.Variable( initial_value = tfd.HalfCauchy(loc=self.conv2_nu_prior_dist.mode(), scale=0.1).sample(), trainable=True, name="conv2_nu_post_mean",dtype=tf.float16) 
			self.conv2_nu_post_scale = tf.Variable( initial_value = tf.random.uniform( self.conv2_nu_post_mean.shape, .05, 0.2), trainable=True, name="conv2_nu_post_scale",dtype=tf.float16) #TODO: change this intialization back
			
			# Betas_LogTaus for Nodes Posterior
			conv1_scale = tf.constant(tf.cast(1.0 *tf.sqrt( 6. / (self.model_params['conv1_output_node_count'] + self.model_params['conv1_input_weights_per_filter'] )), dtype=tf.float16)) #glorot uniform used in microsoft horeshoe
			self.conv1_beta_post_loc = tf.Variable( initial_value=tf.random.uniform( [self.model_params['conv1_output_node_count'], self.model_params['conv1_input_weights_per_filter']] ,-conv1_scale, conv1_scale), trainable=True, name="conv1_beta_post_loc",dtype=tf.float16)  #This uses Xavier init
			self.conv1_tau_post_loc = tf.Variable( initial_value= tfd.HalfCauchy(loc=self.conv1_tau_prior_dist.mode(),scale=0.1).sample()  , trainable=True, name="conv1_tau_post_loc",dtype=tf.float16 )
			self.conv1_U_psi = tf.Variable( initial_value= tf.random.uniform( [self.model_params['conv1_output_node_count'], self.model_params['conv1_input_weights_per_filter'] + 1], minval=0.95, maxval=1.05 ), trainable=True, name="conv1_U_psi",dtype=tf.float16) # My init strategy
			self.conv1_U_h =   tf.Variable( initial_value= tf.random.uniform( [self.model_params['conv1_output_node_count'], self.model_params['conv1_input_weights_per_filter'] + 1] , minval=-0.001, maxval=0.001 ), trainable=True, name="conv1_U_h",dtype=tf.float16) # My init strategy
			
			conv2_scale = tf.constant(tf.cast(1.0 * tf.sqrt( 6. / (self.model_params['conv2_output_node_count'] + self.model_params['conv2_input_weights_per_filter'])  ),dtype=tf.float16)) #glorot uniform used in microsoft horeshoe
			self.conv2_beta_post_loc = tf.Variable( initial_value=tf.random.uniform( [self.model_params['conv2_output_node_count'], self.model_params['conv2_input_weights_per_filter']] ,-conv2_scale, conv2_scale), trainable=True, name="conv2_beta_post_loc",dtype=tf.float16)  #This uses Xavier init
			self.conv2_tau_post_loc = tf.Variable( initial_value= tfd.HalfCauchy(loc=self.conv2_tau_prior_dist.mode(),scale=0.1).sample()  , trainable=True, name="conv2_tau_post_loc",dtype=tf.float16 )
			self.conv2_U_psi = tf.Variable( initial_value= tf.random.uniform( [self.model_params['conv2_output_node_count'],self.model_params['conv2_input_weights_per_filter'] + 1], minval=0.95, maxval=1.05 ), trainable=True, name="conv2_U_psi",dtype=tf.float16) # My init strategy
			self.conv2_U_h =   tf.Variable( initial_value= tf.random.uniform( [self.model_params['conv2_output_node_count'],self.model_params['conv2_input_weights_per_filter'] + 1],  minval=-0.001, maxval=0.001 ), trainable=True, name="conv2_U_h",dtype=tf.float16) # My init strategy

			# Output Layer weights Posterior
			self.conv3_kappa_post_loc = tf.Variable( initial_value = 1.0, trainable=True, name="conv3_kappa_posterior_loc",dtype=tf.float16) #My init strat
			self.conv3_kappa_post_scale = tf.Variable( initial_value = 0.1 , trainable=True, name="conv3_kappa_posterior_scale",dtype=tf.float16)    #This needs to be low since, log_normal distribution has extrememly large tails, so using a value higher, save above 1, can lead to a large scale term and then exploding gradientss
			
			conv3_scale = tf.constant( tf.cast(1.0 * tf.sqrt( 6. / (self.model_params['conv3_output_node_count'] + self.model_params['conv3_input_weights_count'] )  ),dtype=tf.float16))
			self.conv3_weights_post_loc = tf.Variable( initial_value = tf.random.uniform( [self.model_params['conv3_input_weights_count']] , -conv3_scale, conv3_scale,dtype=tf.float16 ), trainable=True, name="conv3_weights_post_loc",dtype=tf.float16 )
		
		elif(self.model_params['model_type_settings']['var_model_type'] == "horseshoefactorized"):
			#Global C Posterior
			self.global_c_post_shape = tf.Variable( initial_value=1.0, trainable=True, dtype=tf.float16) 
			self.global_c_post_scale = tf.Variable( initial_value=0.2,  trainable=True, dtype=tf.float16)

			# Layerwise nu Posterior
			self.conv1_nu_post_mean = tf.Variable( initial_value = tfd.HalfCauchy(loc=self.conv1_nu_prior_dist.mode(), scale=0.1).sample(), trainable=True, dtype=tf.float16 ) 
			self.conv1_nu_post_scale = tf.Variable( initial_value = tf.random.uniform( self.conv1_nu_post_mean.shape, 0.05 ,0.2), trainable=True, dtype=tf.float16)  #TODO: In the future this should be based on the variance used in the HS prior half Cauchy be relating the Half Cuachy scale to the Lognormal variance

			self.conv2_nu_post_mean = tf.Variable( initial_value = tfd.HalfCauchy(loc=self.conv2_nu_prior_dist.mode(), scale=0.1).sample(), trainable=True, dtype=tf.float16) 
			self.conv2_nu_post_scale = tf.Variable( initial_value = tf.random.uniform( self.conv2_nu_post_mean.shape, .05, 0.2), trainable=True, dtype=tf.float16)

			# Betas_LogTaus for Nodes Posterior
			conv1_scale = tf.constant(tf.cast(1.0 *tf.sqrt( 6. / (self.model_params['conv1_output_node_count'] + self.model_params['conv1_input_weights_per_filter'])), dtype=tf.float16)) #glorot uniform used in microsoft horeshoe
			self.conv1_beta_post_loc = tf.Variable( initial_value=tf.random.uniform( [self.model_params['conv1_output_node_count'], self.model_params['conv1_input_weights_per_filter']] ,-conv1_scale, conv1_scale), trainable=True, dtype=tf.float16)  #This uses Xavier init
			self.conv1_beta_post_scale = tf.Variable( initial_value=tf.random.uniform( self.conv1_beta_post_loc.shape, 0.75, 1.1 ), trainable=True, dtype=tf.float16 )
			self.conv1_tau_post_loc = tf.Variable( initial_value= tfd.HalfCauchy(loc=self.conv1_tau_prior_dist.mode(),scale=0.1).sample(), trainable=True, name="conv1_tau_post_loc",dtype=tf.float16 )
			self.conv1_tau_post_scale = tf.Variable( initial_value = tf.random.uniform( self.conv1_tau_post_loc.shape, .05, 0.2), trainable=True,name="conv1_tau_post_scale" , dtype=tf.float16)
			
			conv2_scale = tf.constant(tf.cast(1.0 * tf.sqrt( 6. / (self.model_params['conv2_output_node_count'] + self.model_params['conv2_input_weights_per_filter'])), dtype=tf.float16)) #glorot uniform used in microsoft horeshoe
			self.conv2_beta_post_loc = tf.Variable( initial_value=tf.random.uniform( [self.model_params['conv2_output_node_count'], self.model_params['conv2_input_weights_per_filter']] ,-conv2_scale, conv2_scale), trainable=True, dtype=tf.float16)  #This uses Xavier init
			self.conv2_beta_post_scale = tf.Variable( initial_value=tf.random.uniform( self.conv2_beta_post_loc.shape, 0.75, 1.1 ), trainable=True, dtype=tf.float16 )
			self.conv2_tau_post_loc = tf.Variable( initial_value= tfd.HalfCauchy(loc=self.conv2_tau_prior_dist.mode(),scale=0.1).sample()  , trainable=True, name="conv2_tau_post_loc",dtype=tf.float16 )
			self.conv2_tau_post_scale = tf.Variable( initial_value = tf.random.uniform( self.conv2_tau_post_loc.shape, .05, 0.2), trainable=True, name="conv1_tau_post_scale",dtype=tf.float16)

			# Output Layer weights Posterior
			self.conv3_kappa_post_loc = tf.Variable( initial_value = 1.0, trainable=True, dtype=tf.float16) #My init strat
			self.conv3_kappa_post_scale = tf.Variable( initial_value = 0.1 , trainable=True, dtype=tf.float16)    #This needs to be low since, log_normal distribution has extrememly large tails, so using a value higher, save above 1, can lead to a large scale term and then exploding gradientss

			conv3_scale = tf.constant( tf.cast(1.0 * tf.sqrt( 6. / (self.model_params['conv3_output_node_count'] + self.model_params['conv3_input_weights_count'] )  ),dtype=tf.float16))
			self.conv3_weights_post_loc = tf.Variable( initial_value = tf.random.uniform( [self.model_params['conv3_input_weights_count']] , -conv3_scale, conv3_scale,dtype=tf.float16 ), trainable=True, dtype=tf.float16 )

	def update_priors_dists(self):
		self.conv3_weights_prior_dist = tfd.Normal(self.conv3_weights_prior_loc, self.conv3_kappa )

	def update_posteriors_dists(self):
		if(self.model_params['model_type_settings']['var_model_type'] == "horseshoestructured"):
			#Global C Posterior
			#self.global_c_post_dist = tfd.LogNormal(self.global_c_post_shape, self.global_c_post_scale)
			self.global_c_post_dist = tfd.HalfCauchy(self.global_c_post_shape, self.global_c_post_scale)
			
			#Layerwise nu Posteriors
			self.conv1_nu_post_dist = tfd.LogNormal( self.conv1_nu_post_mean, self.conv1_nu_post_scale )
			self.conv2_nu_post_dist = tfd.LogNormal( self.conv2_nu_post_mean, self.conv2_nu_post_scale )

			#Layer weights
			# conv1_logtau_post_loc = tf.expand_dims(tf.cast(tf.math.log( self.conv1_tau_post_loc) , dtype=tf.float16 ),-1)
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
		
		elif(self.model_params['model_type_settings']['var_model_type'] == "horseshoefactorized"):
			#Global C Posterior
			#self.global_c_post_dist = tfd.LogNormal(self.global_c_post_shape, self.global_c_post_scale)
			self.global_c_post_dist = tfd.HalfCauchy(self.global_c_post_shape, self.global_c_post_scale)
			
			#Layerwise nu Posteriors
			self.conv1_nu_post_dist = tfd.LogNormal( self.conv1_nu_post_mean, self.conv1_nu_post_scale )
			self.conv2_nu_post_dist = tfd.LogNormal( self.conv2_nu_post_mean, self.conv2_nu_post_scale )

			# Layer local scales
			# conv1_logtau_post_loc = tf.expand_dims(tf.cast(tf.math.log( self.conv1_tau_post_loc) , dtype=tf.float16 ),-1)
			self.conv1_tau_post_dist = tfd.LogNormal( loc = self.conv1_tau_post_loc, scale=self.conv1_tau_post_scale )
			self.conv2_tau_post_dist = tfd.LogNormal( loc = self.conv2_tau_post_loc, scale=self.conv2_tau_post_scale )

			# Layer local means
			self.conv1_beta_post_dist = tfd.Normal( loc=self.conv1_beta_post_loc, scale=self.conv1_beta_post_scale  )
			self.conv2_beta_post_dist = tfd.Normal( loc=self.conv2_beta_post_loc, scale=self.conv2_beta_post_scale  )

			self.conv3_kappa_post_dist = tfd.LogNormal(self.conv3_kappa_post_loc, self.conv3_kappa_post_scale )
			self.conv3_kappa = self.conv3_kappa_post_dist.sample() #This should technically be in sample_variational_params
			self.conv3_weights_dist = tfd.Normal( loc=self.conv3_weights_post_loc , scale=self.conv3_kappa )

	def sample_variational_params(self):
		if(self.model_params['model_type_settings']['var_model_type'] == "horseshoestructured"):
			self.conv1_nu = self.conv1_nu_post_dist.sample() #TODO: currently this outputs 1, when in actuality it should be 
			self.conv2_nu = self.conv2_nu_post_dist.sample()

			self.c = self.global_c_post_dist.sample()

		elif(self.model_params['model_type_settings']['var_model_type'] == "horseshoefactorized"):
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
		if(self.model_params['model_type_settings']['var_model_type'] == "horseshoestructured"):
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
		
		elif(self.model_params['model_type_settings']['var_model_type'] == "horseshoefactorized"):
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
		
		if(self.model_params['model_type_settings']['var_model_type'] == "horseshoestructured"):
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
		
		elif(self.model_params['model_type_settings']['var_model_type'] == "horseshoefactorized"):
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

		self.T_1 = tf.constant( T_1, dtype=tf.float16)
		self.T_2 = tf.constant( T_2, dtype=tf.float16)

def horseshoe_kernel_posterior_distribution(obj, _conv_layer_no ):
	"""
		Implements the Non-Centred Weight Distribution which is used in the kernels
		Samples from multi-variate Guassian distribution for values of Beta and tau
	"""
	conv_layer = _conv_layer_no

	
	var_model_type = obj.model_params['model_type_settings']['var_model_type']

	def _fn(dtype, shape, name, trainable, add_variable_fn ):
		
		if( var_model_type == "horseshoestructured" ): 
			dist = tf.case( [ (tf.equal(conv_layer,tf.constant(1)),lambda: obj.conv1_beta_logtau_post_dist), (tf.equal(conv_layer,tf.constant(2)), lambda: obj.conv2_beta_logtau_post_dist) ] , exclusive=True )
		
		elif( var_model_type == "horseshoefactorized" ):
			dist = tf.case( [ (tf.equal(conv_layer,tf.constant(1)),lambda: obj.conv1_beta_post_dist), (tf.equal(conv_layer,tf.constant(2)), lambda: obj.conv2_beta_post_dist) ] , exclusive=True )

		return dist
	
	return _fn
   
def horsehoe_kernel_posterior_tensor_fn( dist, layer_nu , model_global_c, output_count, kernel_shape, inp_channels, conv_layer=1, obj=None,
										factorised_beta=None, factorised_tau=None, factorised_nu=None ):
	"""
		:param filter_shape list: [h, w]
	"""
		
	if( obj.model_params['model_type_settings']['var_model_type'] == "horseshoestructured"): 
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
	
	elif(obj.model_params['model_type_settings']['var_model_type'] == "horseshoefactorized"):
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
	def __init__(self, train_params, encoder_params, h_w):
		super( THST_Encoder, self ).__init__()
		self.encoder_params = encoder_params
		self.train_params = train_params
		self.layer_count = encoder_params['enc_layer_count']
		

		self.CLSTM_Input_Layer = THST_CLSTM_Input_Layer( train_params, encoder_params['CLSTMs_params'][0] )

		self.CLSTM_Attn_layers = []
		for idx in range( encoder_params['attn_layers_count'] ):
			_layer = THST_CLSTM_Attention_Layer( train_params, encoder_params['CLSTMs_params'][idx+1],
						encoder_params['ATTN_params'][idx], encoder_params['ATTN_DOWNSCALING_params_enc'] ,
						encoder_params['seq_len_factor_reduction'][idx], self.encoder_params['attn_layers_num_of_splits'][idx],
						h_w )
			self.CLSTM_Attn_layers.append(_layer)

	#@tf.function -> cant be used since output hs_list is a TensorArray which Tensorflow does not correctly handle yet
	def call(self, _input, training=True):
		"""
			_input #shape( )
		"""
		#old
			# hidden_states_1 =  self.CLSTM_1( _input, training ) #(bs, seq_len_1, h, w, c1)

			# hidden_states_2 = self.CLSTM_2( hidden_states_1, training=training) #(bs, seq_len_2, h, w, c2)

			# hidden_states_3 =  self.CLSTM_3( hidden_states_2,training=training ) #(bs, seq_len_3, h, w, c3)

			# hidden_states_4 =  self.CLSTM_4( hidden_states_3,training=training ) #(bs, seq_len_4, h, w, c3)

			# hidden_states_5 =  self.CLSTM_5( hidden_states_4,training=training ) #(bs, seq_len_5, h, w, c3)

			# return hidden_states_2, hidden_states_3, hidden_states_4, hidden_states_5  

		hs_list = tf.TensorArray(dtype=self._compute_dtype, size=self.encoder_params['attn_layers_count'], infer_shape=False, dynamic_size=False, clear_after_read=False )
		
		hidden_state =  self.CLSTM_Input_Layer( _input, training ) #(bs, seq_len_1, h, w, c1)
		
		#Note: Doing the foor loop this way so more operations can be given to gpu 1
		#with tf.device('/GPU:0'):
		for idx in range(self.encoder_params['attn_layers_count'] -1):
			hidden_state = self.CLSTM_Attn_layers[idx]( hidden_state, training=training)
			hs_list = hs_list.write( idx, hidden_state )
		
		#with tf.device('/GPU:1'):
		hidden_state = self.CLSTM_Attn_layers[idx+1]( hidden_state, training=training)
		hs_list = hs_list.write( idx+1, hidden_state )
		
		return hs_list

class THST_Decoder(tf.keras.layers.Layer):
	def __init__(self, train_params ,decoder_params, h_w):
		"""
		:param list decoder_params: a list of dictionaries of the contained LSTM's params
		"""
		super( THST_Decoder, self ).__init__()
		self.decoder_params = decoder_params
		self.train_params = train_params
		self.layer_count = decoder_params['decoder_layer_count']
		#self.encoder_hidden_state_count = self.layer_count + 1
		
		
		self.CLSTM_2cell_layers = []
		for idx in range( self.layer_count ):
			_layer = THST_CLSTM_Decoder_Layer( train_params, self.decoder_params['CLSTMs_params'][idx], decoder_params['seq_len_factor_expansion'][idx],
														decoder_params['seq_len'][idx], h_w )
			self.CLSTM_2cell_layers.append(_layer)

	# @tf.function
	# def call(self, hidden_states_2_enc, hidden_states_3_enc, hidden_states_4_enc, hidden_states_5_enc, training=True  ):

	#@tf.function -> cant be used since inpuut hs_list is a TensorArray which Tensorflow does not correctly handle yet
	def call(self, hs_list, training=True):

		# hidden_states_l4 = self.CLSTM_L4( hidden_states_4_enc , hidden_states_5_enc, training)
		#     #2, 4, 100, 140, 2
		# hidden_states_l3 = self.CLSTM_L3( hidden_states_3_enc, hidden_states_l4, training ) #(bs, output_len2, 100, 140, layer_below_filters*2)
		#     #2, 8, 100, 140, 2
		# hidden_states_l2 = self.CLSTM_L2( hidden_states_2_enc, hidden_states_l3, training )     #(bs, output_len1, height, width)     
		#     #2, 16, 100, 140, 4

		dec_hs_outp = hs_list.read(self.layer_count)

		for idx in range(1, self.layer_count+1):
			dec_hs_outp =  self.CLSTM_2cell_layers[-idx]( hs_list.read( self.layer_count -idx ), dec_hs_outp, training )
		hs_list = hs_list.close()
		return dec_hs_outp
			
class THST_CLSTM_Input_Layer(tf.keras.layers.Layer):
	"""
		This corresponds to the lower input layer
		rmrbr to add spatial LSTM
	"""
	def __init__(self, train_params, layer_params ):
		super( THST_CLSTM_Input_Layer, self ).__init__()
		
		self.trainable = train_params['trainable']
		self.layer_params = layer_params #list of dictionaries containing params for all layers

		self.convLSTM = Bidirectional( layers_ConvLSTM2D.ConvLSTM2D( **self.layer_params ), merge_mode=None ) 
	
	@tf.function
	def call( self, _input, training ):
		#NOTE: consider addding multiple LSTM Layers to extract more latent features

		hidden_states_f, hidden_states_b = self.convLSTM( _input, training=training ) #(bs, seq_len_1, h, w, c)
		hidden_states = tf.concat([hidden_states_f, hidden_states_b],axis=-1)
			   
		return hidden_states #(bs, seq_len_1, h, w, c*2)

class THST_CLSTM_Attention_Layer(tf.keras.layers.Layer):
	"""
		This corresponds to all layers which use attention to weight the inputs from the layer below
	"""
	def __init__(self, train_params, clstm_params, attn_params, attn_downscaling_params ,seq_len_factor_reduction, num_of_splits, h_w ):
		super( THST_CLSTM_Attention_Layer, self ).__init__()

		self.trainable = train_params['trainable']
		self.num_of_splits = num_of_splits
		self.seq_len_factor_reduction = seq_len_factor_reduction
		
		self.convLSTM_attn = Bidirectional( layers_ConvLSTM2D.ConvLSTM2D_attn( **clstm_params,
												attn_params=attn_params , attn_downscaling_params=attn_downscaling_params ,
												attn_factor_reduc=seq_len_factor_reduction ,trainable=self.trainable ),
												merge_mode=None ) #stateful possibly set to True, return_state=True, return_sequences=True

		self.shape = ( train_params['batch_size'], self.num_of_splits, h_w[0], h_w[1], clstm_params['filters'] )

	@tf.function
	def call(self, input_hidden_states, training=True):
		
		hidden_states_f, hidden_states_b = self.convLSTM_attn(input_hidden_states, training=training)
		hidden_states_f.set_shape( self.shape )
		hidden_states_b.set_shape( self.shape )
		hidden_states = tf.concat( [hidden_states_f,hidden_states_b], axis=-1 )

		return hidden_states #shape(bs, seq_len, h, w, 2*c2)

class THST_CLSTM_Decoder_Layer(tf.keras.layers.Layer):
	def __init__(self, train_params ,layer_params, input_2_factor_increase, seq_len, h_w ):
		super( THST_CLSTM_Decoder_Layer, self ).__init__()
		
		self.layer_params = layer_params
		self.input_2_factor_increase = input_2_factor_increase
		#self.trainable= train_params['trainable']
		self.seq_len = seq_len
		
		self.shape2 = ( train_params['batch_size'], self.seq_len//self.input_2_factor_increase, h_w[0], h_w[1], layer_params['filters']*2 ) #TODO: the final dimension only works rn since all layers have same filter count. It should be equal to 2* filters of previous layer
		self.shape1 = ( train_params['batch_size'], self.seq_len, h_w[0], h_w[1], layer_params['filters']*2 ) #same comment as above
		self.shape3 = ( train_params['batch_size'], self.seq_len, h_w[0], h_w[1], layer_params['filters'] )
		self.convLSTM =  tf.keras.layers.Bidirectional( layers_ConvLSTM2D.ConvLSTM2D_custom(**layer_params )  , merge_mode=None)
	
	@tf.function
	def call(self, input1, input2, training=True ):
		"""

			input1 : Contains the hidden representations from the corresponding layer in the encoder #(bs, seq_len1, h,w,c1)
			input2: Contains the hidden repr from the previous decoder layer #(bs, seq_len2, h,w,c2)

		"""
		input1.set_shape(self.shape1)
		input2.set_shape(self.shape2)
		input2 = tf.keras.backend.repeat_elements( input2, self.input_2_factor_increase, axis=1) #(bs, seq_len1, h,w,c2)

		tf.debugging.assert_equal(tf.shape(input1),tf.shape(input2), message="Decoder Input1, and Input2 not the same size")
		
		inputs = tf.concat( [input1, input2], axis=-1 ) #NOTE: At this point both input1 and input2, must be the same shape

		hidden_states_f, hidden_states_b = self.convLSTM( inputs, training=training )
		
		hidden_states_f.set_shape( self.shape3 )
		hidden_states_b.set_shape( self.shape3 )
		hidden_states = tf.concat( [hidden_states_f,hidden_states_b], axis=-1 ) 
		return hidden_states

class THST_OutputLayer(tf.keras.layers.Layer):
	def __init__(self, train_params, layer_params, model_type_settings):
		"""
			:param list layer_params: a list of dicts of params for the layers
		"""
		super( THST_OutputLayer, self ).__init__()

		self.trainable = train_params['trainable']
		
		if(model_type_settings['deformable_conv'] ==False):
			self.conv_hidden = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[0] ) )
			self.conv_output = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[1] ) )
		
		elif( model_type_settings['deformable_conv'] ==True ):
			self.conv_hidden = tf.keras.layers.TimeDistributed( layers_ConvLSTM2D.DeformableConvLayer( **layer_params[0] ) )
			self.conv_output = tf.keras.layers.TimeDistributed( layers_ConvLSTM2D.DeformableConvLayer( **layer_params[1] ) )
	
	@tf.function
	def call(self, _inputs, training=True ):
		"""
		:param tnsr inputs: (bs, seq_len, h,w,c)
		"""
		x = self.conv_hidden( _inputs,training=training )
		x = self.conv_output( x, training=training ) #shape (bs, height, width)
		return x

class SpatialConcreteDropout(tf.keras.layers.Wrapper):
	"""This wrapper allows to learn the dropout probability for any given Conv2D input layer.
		```python
			model = Sequential()
			model.add(ConcreteDropout(Conv2D(64, (3, 3)),
									input_shape=(299, 299, 3)))
		```
		# Arguments
			layer: a layer instance.
			weight_regularizer:
				A positive number which satisfies
					$weight_regularizer = l**2 / (\tau * N)$
				with prior lengthscale l, model precision $\tau$ (inverse observation noise),
				and N the number of instances in the dataset.
				Note that kernel_regularizer is not needed.
			dropout_regularizer:
				A positive number which satisfies
					$dropout_regularizer = 2 / (\tau * N)$
				with model precision $\tau$ (inverse observation noise) and N the number of
				instances in the dataset.
				Note the relation between dropout_regularizer and weight_regularizer:
					$weight_regularizer / dropout_regularizer = l**2 / 2$
				with prior lengthscale l. Note also that the factor of two should be
				ignored for cross-entropy loss, and used only for the eculedian loss.
	"""
	def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
				 init_min=0.1, init_max=0.1, is_mc_dropout=True, data_format=None, **kwargs):
		assert 'kernel_regularizer' not in kwargs
		super(SpatialConcreteDropout, self).__init__(layer, **kwargs)
		self.weight_regularizer =  1 # in your model, this regularlization is done in the train code  2/(1e-5*365*24 ) #weight_regularizer
		self.dropout_regularizer = 1 # in your model, this regularlization is done in the train code 2/(1e-5*365*24 ) #dropout_regularizer
		self.is_mc_dropout = is_mc_dropout
		self.supports_masking = True
		self.p_logit = None
		self.p = None
		self.init_min = np.log(init_min) - np.log(1. - init_min)
		self.init_max = np.log(init_max) - np.log(1. - init_max)
		self.data_format = 'channels_last' if data_format is None else 'channels_first'

	def build(self, input_shape=None):
		
		self.input_spec = tf.keras.layers.InputSpec(shape=input_shape)

		super(SpatialConcreteDropout, self).build(input_shape)
		if self.layer.built: #Strange behaviour but layer needs to be built twice to work
			self.layer.build(input_shape)
			self.layer.built = True

		# self.kernel = self.add_weight(
		#     name='kernel',
		#     shape=kernel_shape,
		#     initializer=self.kernel_initializer,
		#     regularizer=self.kernel_regularizer,
		#     constraint=self.kernel_constraint,
		#     trainable=True,
		#     dtype=self.dtype)
		# if self.use_bias:
		#     self.bias = self.add_weight(
		#         name='bias',
		#         shape=(self.filters,),
		#         initializer=self.bias_initializer,
		#         regularizer=self.bias_regularizer,
		#         constraint=self.bias_constraint,
		#         trainable=True,
		#         dtype=self.dtype)
		
				
			# initialise p
		self.p_logit = self.layer.add_weight(name='p_logit',
											shape=(1,),
											initializer=tf.keras.initializers.RandomUniform(self.init_min, self.init_max),
											trainable=True,
											dtype=tf.float16)
		# self.p = K.sigmoid(self.p_logit[0])

		# initialise regulariser / prior KL term
		assert len(input_shape) == 4, 'this wrapper only supports Conv2D layers'
		if self.data_format == 'channels_first':
			self.input_dim = input_shape[1] # we drop only channels
		else:
			self.input_dim = input_shape[3]
		
		# weight = self.layer.kernel
		# kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
		# dropout_regularizer = self.p * K.log(self.p)
		# dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
		# dropout_regularizer *= self.dropout_regularizer * input_dim
		# regularizer = K.sum(kernel_regularizer + dropout_regularizer)
		# self.layer.add_loss(regularizer)

	def compute_output_shape(self, input_shape):
		return self.layer.compute_output_shape(input_shape)

	def spatial_concrete_dropout(self, x):
		'''
		Concrete dropout - used at training time (gradients can be propagated)
		:param x: input
		:return:  approx. dropped out input
		'''
		self.p = K.sigmoid(self.p_logit[0])
		#eps = K.cast_to_floa6tx(K.epsilon())
		eps = tf.cast( K.epsilon(), dtype=tf.float16)
		temp = 2. / 3.

		input_shape = K.shape(x)
		if self.data_format == 'channels_first':
			noise_shape = (input_shape[0], input_shape[1], 1, 1)
		else:
			noise_shape = (input_shape[0], 1, 1, input_shape[3])
		unif_noise = K.random_uniform(shape=noise_shape,dtype=tf.float16)
		
		drop_prob = (
			K.log(self.p + eps)
			- K.log(1. - self.p + eps)
			+ K.log(unif_noise + eps)
			- K.log(1. - unif_noise + eps)
		)
		drop_prob = K.sigmoid(drop_prob / temp)
		random_tensor = 1. - drop_prob

		retain_prob = 1. - self.p
		x *= random_tensor
		x /= retain_prob
		return x
	
	def call(self, inputs, training=None, pred=False):
		
		if self.is_mc_dropout:
			return self.layer.call(self.spatial_concrete_dropout(inputs))
		else:
			def relaxed_dropped_inputs():
				return self.layer.call(self.spatial_concrete_dropout(inputs))
			return K.in_train_phase(relaxed_dropped_inputs,
									self.layer.call(inputs),
									training=training)
	
	def add_loss(self):
		weight = self.layer.kernel
		kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
		dropout_regularizer = self.p * K.log(self.p)
		dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
		dropout_regularizer *= self.dropout_regularizer * self.input_dim
		regularizer = K.sum(kernel_regularizer + dropout_regularizer)
		self.layer.add_loss(regularizer)
		return True

# endregion

# region general layers/functions

class OutputReluFloat32(tf.keras.layers.Layer):
	def __init__(self, t_params):
		super(OutputReluFloat32, self).__init__()

		self.custom_relu = CustomRelu_maker(t_params)
		self.outputf32 = tf.keras.layers.Activation('linear', dtype='float32')
	
	@tf.function
	def call(self, inputs):
		outp = self.custom_relu(inputs)
		outp = self.outputf32(outp)
		return outp

def CustomRelu_maker(t_params):
	CustomRelu = ReLU_correct_layer( threshold= utility.standardize_ati( 0, t_params['normalization_shift']['rain'], 
															t_params['normalization_scales']['rain'], reverse=False) )
	return CustomRelu

class ReLU_correct_layer(tf.keras.layers.Layer):
    """Akanni Corrected Rectified Linear Unit activation function.
        With default values, it returns element-wise `max(x, 0)`.
        Otherwise, it follows:
        `f(x) = max_value` for `x >= max_value`,
        `f(x) = x` for `threshold <= x < max_value`,
        `f(x) = negative_slope * (x - threshold)` otherwise.
        Input shape:
            Arbitrary. Use the keyword argument `input_shape`
            (tuple of integers, does not include the samples axis)
            when using this layer as the first layer in a model.
        Output shape:
            Same shape as the input.
        Arguments:
            max_value: Float >= 0. Maximum activation value.
            negative_slope: Float >= 0. Negative slope coefficient.
            threshold: Float. Threshold value for thresholded activation.
    """

    def __init__(self, max_value=None, negative_slope=0, threshold=0, **kwargs):
        super(ReLU_correct_layer, self).__init__(**kwargs)
        if max_value is not None and max_value < 0.:
            raise ValueError('max_value of Relu layer '
                            'cannot be negative value: ' + str(max_value))
        if negative_slope < 0.:
            raise ValueError('negative_slope of Relu layer '
                            'cannot be negative value: ' + str(negative_slope))

        self.support_masking = True
        if max_value is not None:
            max_value = K.cast_to_floatx(max_value)
        self.max_value = max_value
        self.negative_slope = K.cast_to_floatx(negative_slope)
        self.threshold = K.cast_to_floatx(threshold)
    
    @tf.function
    def call(self, inputs):
        # alpha is used for leaky relu slope in activations instead of
        # negative_slope.
        return ReLU_corrected(inputs,
                    alpha=self.negative_slope,
                    max_value=self.max_value,
                    threshold=self.threshold)
        

    def get_config(self):
        config = {
            'max_value': self.max_value,
            'negative_slope': self.negative_slope,
            'threshold': self.threshold
        }
        base_config = super(ReLU_correct_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


def ReLU_corrected(x, alpha=0., max_value=None, threshold=0):
    """Rectified linear unit.
        With default values, it returns element-wise `max(x, 0)`.
        Otherwise, it follows:
        `f(x) = max_value` for `x >= max_value`,
        `f(x) = x` for `threshold <= x < max_value`,
        `f(x) = threshold + alpha * (x - threshold)` otherwise.
        Arguments:
            x: A tensor or variable.
            alpha: A scalar, slope of negative section (default=`0.`).
            max_value: float. Saturation threshold.
            threshold: float. Threshold value for thresholded activation.
        Returns:
            A tensor.
    """

    if alpha != 0.:
        if max_value is None and threshold == 0:
            return nn.leaky_relu(x, alpha=alpha)

        if threshold != 0:
            negative_part = nn.relu(-x + threshold)
        else:
            negative_part = nn.relu(-x)

    clip_max = max_value is not None

    if threshold != 0:
        # computes x for x > threshold else 0
        #x = x * math_ops.cast(math_ops.greater(x, threshold), K.floatx())
        x = x * math_ops.cast(math_ops.greater(x, threshold), K.floatx()) + threshold * math_ops.cast(math_ops.greater_equal(threshold, x), K.floatx())
    elif max_value == 6:
        # if no threshold, then can use nn.relu6 native TF op for performance
        x = nn.relu6(x)
        clip_max = False
    else:
        x = nn.relu(x)

    if clip_max:
        max_value = K._to_tensor(max_value, x.dtype.base_dtype)
        #zero = K._to_tensor(0., x.dtype.base_dtype)
        x = clip_ops.clip_by_value(x, threshold, max_value)

    if alpha != 0.:
        alpha = K._to_tensor(alpha, x.dtype.base_dtype)
        x -= alpha * negative_part
    return x


def LeakyRelu_mkr(t_params):
	CustomRelu = tf.keras.layers.ReLU( threshold=utility.standardize_ati( 0, t_params['normalization_shift']['rain'], 
															t_params['normalization_scales']['rain'], reverse=False), negative_slope=0.1 )

	return CustomRelu
# endregion

# region SimpleLSTM GRU layers


# endregion