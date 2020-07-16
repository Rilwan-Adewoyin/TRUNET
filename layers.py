import copy
import math
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import tensor2tensor as t2t  
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import clip_ops, math_ops, nn
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl

import layers_convgru2D
import utility

#os.environ["OMP_NUM_THREADS"] = "1"

try:
	import tensorflow_addons as tfa
except:
	tfa=None


# region --- TRU-NET Encoder Decoder Net sublayers
class TRUNET_Encoder(tf.keras.layers.Layer):
	"""TRU-NET Encoder-Decoder Encoder
	"""	
	def __init__(self, t_params, encoder_params, h_w, attn_ablation=0):
		"""

		Args:
            t_params (dict): params related to training/testing
			encoder_params (dict): params related to encoder 
			h_w ([type]): height and width of convolution output for this layer
			attn_ablation (int, optional): ablation mode for encoder layers. 
				Defaults to 0 for cross attention
		"""		
		super( TRUNET_Encoder, self ).__init__()
		self.encoder_params = encoder_params
		self.t_params = t_params
		self.layer_count = encoder_params['enc_layer_count']	
		
		self.CGRU_Input_Layer = TRUNET_CGRU_Input_Layer( t_params, encoder_params['CGRUs_params'][0] )

		#Dynamically init ConvGRU w/ ILCA layers
		self.CGRU_Attn_layers = []
		for idx in range( encoder_params['attn_layers_count'] ):
			_layer = TRUNET_CGRU_Attention_Layer( t_params, encoder_params['CGRUs_params'][idx+1],
						encoder_params['ATTN_params'][idx], encoder_params['ATTN_DOWNSCALING_params_enc'] ,
						encoder_params['seq_len_factor_reduction'][idx], self.encoder_params['attn_layers_num_of_splits'][idx],
						h_w, attn_ablation )

			self.CGRU_Attn_layers.append(_layer)
				
	def call(self, _input, training=True):
		"""[summary]

		Args:
			_input ([type]): (batch_size, seq_len, h, w, c)
			training (bool, optional): [description]. Defaults to True.

		Returns:
			[type]: (batch_size, seq_len1, h1, w1, c1)
		"""						
		hidden_state =  self.CGRU_Input_Layer( _input, training ) #(bs, seq_len_1, h, w, c1)
		
		hidden_state = self.CGRU_Attn_layers[0]( hidden_state, training=training)
		hidden_states = hidden_state
		
		for idx in range(1, self.encoder_params['attn_layers_count']):
			hidden_state = self.CGRU_Attn_layers[idx]( hidden_state, training=training)
			hidden_states = tf.concat( [ hidden_states, hidden_state ], axis=1 )
					
		return hidden_states

class TRUNET_Decoder(tf.keras.layers.Layer):
	def __init__(self, t_params ,decoder_params, h_w):
		"""
		:param list decoder_params: a list of dictionaries of the contained LSTM's params
		"""
		super( TRUNET_Decoder, self ).__init__()
		self.decoder_params = decoder_params
		self.t_params = t_params
		self.layer_count = decoder_params['decoder_layer_count']
		#self.h_w = h_w
		#self.encoder_hidden_state_count = self.layer_count + 1
		
		self.CGRU_2cell_layers = []
		for idx in range( self.layer_count ):
			_layer = TRUNET_CGRU_Decoder_Layer( t_params, self.decoder_params['CGRUs_params'][idx], 
												decoder_params['seq_len_factor_expansion'][idx],
												decoder_params['seq_len'][idx], h_w )
			self.CGRU_2cell_layers.append(_layer)

		self.seq_lens = self.decoder_params['attn_layer_no_splits']
		#self.shape1 = [ sum( self.seq_lens ) ] + [ self.t_params['batch_size'] ] + [h_w[0], h_w[1], self.decoder_params['CGRUs_params'][0]['filters']*2 ] 
	
	def call(self, hidden_states, training=True):

		li_hs = tf.split(hidden_states, self.seq_lens, axis=1 )

		dec_hs_outp = li_hs[-1]

		for idx in range(1, self.layer_count+1):
			dec_hs_outp =  self.CGRU_2cell_layers[-idx]( li_hs[ self.layer_count-idx], dec_hs_outp, training )

		return dec_hs_outp

class TRUNET_OutputLayer(tf.keras.layers.Layer):
	def __init__(self, t_params,layer_params, model_type_settings, dropout_rate):
		"""
			:param list layer_params: a list of dicts of params for the layers
		"""
		super( TRUNET_OutputLayer, self ).__init__()

		self.trainable = t_params['trainable']
		
		self.dc = model_type_settings['discrete_continuous']
		
		self.do0 = tf.keras.layers.TimeDistributed( tf.keras.layers.SpatialDropout2D( rate=dropout_rate, data_format = 'channels_last') )
		self.do1 = tf.keras.layers.TimeDistributed( tf.keras.layers.SpatialDropout2D( rate=dropout_rate, data_format = 'channels_last') )

		if not self.dc:			
			self.conv_hidden = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[0] ) )
			self.conv_output = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[1] ) )
			self.float32_custom_relu = OutputReluFloat32(t_params) 
		
		else:
			# layer_params_pr = copy.deepcopy(layer_params[0])
			# layer_params_pr['filters'] = layer_params_pr['filters']//2
			self.conv_hidden_val = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[0] ) )
			self.conv_hidden_prob = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[0] ) )

			# self.conv_hidden_val1 = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[0] ) )
			#self.conv_hidden_prob1 = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[0] ) )

			self.conv_output_val = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[1] ) )
			self.conv_output_prob = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **layer_params[1] ) )

			self.float32_output = tf.keras.layers.Activation('linear', dtype='float32')

			self.output_activation_val = CustomRelu_maker(t_params, dtype='float32')
			self.output_activation_prob = tf.keras.layers.Activation('sigmoid', dtype='float32')
				

	def call(self, _inputs, training=True ):
		"""
			:param tnsr inputs: (bs, seq_len, h,w,c)
		"""

		if self.dc == False:
			x = self.conv_hidden( self.do0(_inputs,training=training) ) 	
			outp = self.conv_output( x, training=training ) #shape (bs, height, width)
			outp = self.float32_custom_relu(outp)   
		
		else:

			# x_val = self.conv_hidden_val( self.do0( tf.gather(_inputs,indexes2,axis=4), training=training))
			# x_prob = self.conv_hidden_prob( self.do0( tf.gather(_inputs,indexes1, axis=4),training=training))

			x_val 	= self.conv_hidden_val( self.do0( _inputs, training=training))
			x_prob 	= self.conv_hidden_prob( self.do0( tf.stop_gradient(_inputs), training=training))	

			# x_val 	= self.conv_hidden_val1( self.do1( x_val, training=training))
			#x_prob 	= self.conv_hidden_prob1( self.do1( x_prob, training=training))			
			
			x_val = self.conv_output_val( x_val, training=training)
			x_prob = self.conv_output_prob( x_prob, training=training)

			outp_val = self.float32_output(x_val)
			outp_prob = self.float32_output(x_prob)

			outp_val = self.output_activation_val(outp_val)
			outp_prob = self.output_activation_prob(outp_prob)
			#outp_prob = tf.keras.activations.relu(outp_prob, max_value=1.0 )

			outp = tf.stack([outp_val, outp_prob], axis=0)

		return outp

class TRUNET_CGRU_Input_Layer(tf.keras.layers.Layer):
	"""Convolutional GRU Input Layer
	"""	
	def __init__(self, t_params, layer_params):
		super( TRUNET_CGRU_Input_Layer, self ).__init__()
			
		self.layer_params = layer_params #list of dictionaries containing params for all layers
		self.convGRU = Bidirectional( layer=layers_convgru2D.ConvGRU2D( **self.layer_params ), 
										backward_layer=layers_convgru2D.ConvGRU2D( **copy.deepcopy(self.layer_params), go_backwards=True ),
										merge_mode=None ) 		
	def call( self, _input, training ):
		hidden_states_f, hidden_states_b = self.convGRU(_input, training=training ) #(bs, seq_len_1, h, w, c)
		hidden_states = tf.concat([hidden_states_f, hidden_states_b],axis=-1) 
		return hidden_states #(bs, seq_len_1, h, w, c*2)

class TRUNET_CGRU_Attention_Layer(tf.keras.layers.Layer):
	"""ConvGRU Layer w/ Inter Layer Cross Attention
		Inputs:
					tensor of shape (bs, seq_len, h, w, c1)
		Returns:
			[type]: tensor of shape (bs, seq_len/n, h2, w2, c2)
	"""	
	def __init__(self, t_params, CGRU_params, attn_params, attn_downscaling_params ,seq_len_factor_reduction, num_of_splits, h_w, attn_ablation=0 ):
		super( TRUNET_CGRU_Attention_Layer, self ).__init__()

		self.trainable 					= t_params['trainable']
		self.num_of_splits 				= num_of_splits
		self.slfr 						= seq_len_factor_reduction
		self.attn_ablation				= attn_ablation
		
		self.convGRU_attn		= Bidirectional( layer=layers_convgru2D.ConvGRU2D_attn( **CGRU_params,
													attn_params=attn_params , attn_downscaling_params=attn_downscaling_params ,
													attn_factor_reduc=self.slfr ,trainable=self.trainable, attn_ablation=self.attn_ablation  ),

												 backward_layer=layers_convgru2D.ConvGRU2D_attn( go_backwards=True, **copy.deepcopy(CGRU_params),
													attn_params=attn_params , attn_downscaling_params=attn_downscaling_params ,
													attn_factor_reduc=self.slfr ,trainable=self.trainable,
													attn_ablation=self.attn_ablation ),
															merge_mode=None  ) 

	def call(self, input_hidden_states, training=True):
		
		hidden_states_f, hidden_states_b = self.convGRU_attn(input_hidden_states, training=training)
		hidden_states = tf.concat( [hidden_states_f, hidden_states_b], axis=-1 )

		return hidden_states #shape(bs, seq_len, h, w, 2*c2)

class TRUNET_CGRU_Decoder_Layer(tf.keras.layers.Layer):
	def __init__(self, t_params ,layer_params, input_2_factor_increase, seq_len, h_w ):
		super( TRUNET_CGRU_Decoder_Layer, self ).__init__()
		
		self.layer_params = layer_params
		# The factor increase in repeated GRU operations between this decoder layer and the next
		self.input_2_factor_increase = input_2_factor_increase
		self.seq_len = seq_len
		
		# Shapes to facilitate tensorflow graph operations
		self.convGRU =  tf.keras.layers.Bidirectional( layer=layers_convgru2D.ConvGRU2D_Dualcell(**layer_params,trainable=self.trainable ),
														backward_layer=layers_convgru2D.ConvGRU2D_Dualcell( **copy.deepcopy(layer_params),go_backwards=True,trainable=self.trainable ),
														merge_mode=None)
	
	def call(self, input1, input2, training=True ):
		"""[summary]

			Args:
				input1 : hidden representations from the corresponding layer 
							in the encoder #(bs, seq_len1, h,w,c1)
				input2 : hidden repr from the previous decoder layer 
							#(bs, seq_len2, h,w,c2)
				training (bool, optional): [description]. Defaults to True.

			Returns:
				[type]: tensor wth shape #(bs, seq_len1, h,w,c3)
		"""		

		input2 = tf.keras.backend.repeat_elements( input2, self.input_2_factor_increase, axis=1) #(bs, seq_len1, h,w,c2)
		
		inputs = tf.concat( [input1, input2], axis=-1 ) 
		hidden_states_f, hidden_states_b = self.convGRU( inputs, training=training )
		hidden_states = tf.concat( [hidden_states_f,hidden_states_b], axis=-1 ) 
		return hidden_states

# endregion

# region --- TRU-NET Encoder-Forecaster
class TRUNET_EF_Encoder(tf.keras.layers.Layer):
	def __init__(self, t_params, encoder_params, h_w):
		super( TRUNET_EF_Encoder, self ).__init__()
		self.encoder_params = encoder_params
		self.t_params = t_params
		self.layer_count = encoder_params['enc_layer_count']
		self.seq_len_factor = encoder_params['seq_len_factor_reduction']
		
		self.CGRU_Input_Layer = TRUNET_CGRU_Input_Layer( t_params, encoder_params['CGRUs_params'][0] )

		self.CGRU_dsample = []
		self.CGRU_Attn_layers = []
		
		for idx in range( encoder_params['attn_layers_count'] ):
			_layer_dsample = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( filters= encoder_params['CGRUs_params'][idx]['filters'] , kernel_size=(2,2), strides=(2,2), activation=None, padding='same'   ) )
			_layer = TRUNET_CGRU_Attention_Layer( t_params, encoder_params['CGRUs_params'][idx+1],
						encoder_params['ATTN_params'][idx], encoder_params['ATTN_DOWNSCALING_params_enc'] ,
						encoder_params['seq_len_factor_reduction'][idx], self.encoder_params['attn_layers_num_of_splits'][idx],
						h_w )
			
			self.CGRU_dsample.append(_layer_dsample)
			self.CGRU_Attn_layers.append(_layer)
				
	def call(self, _input, training=True):
		"""
			_input #shape( )
		"""
		hidden_state =  self.CGRU_Input_Layer( _input, training ) 						#(bs, seq_len_1, h, w, c1)
		hidden_state, last_state = self.CGRU_Attn_layers[0]( hidden_state, training=training)

		last_states = tf.RaggedTensor.from_tensor( last_state ) # top2bot: ,16, 4,1

		for idx in range(1, self.encoder_params['attn_layers_count']):
			hidden_state = self.CGRU_dsample[idx]( hidden_state )
			hidden_outp, last_state = self.CGRU_Attn_layers[idx]( hidden_state, training=training)

			last_states = tf.concat( [ last_states, tf.RaggedTensor._from_tensor(last_state) ], axis=0 ) #(bs, h, w, c1)
					
		return last_states

class TRUNET_EF_Forecaster(tf.keras.layers.Layer):
	def __init__(self, t_params ,decoder_params, h_w):
		"""
		:param list decoder_params: a list of dictionaries of the contained LSTM's params
		"""
		super( TRUNET_EF_Forecaster, self ).__init__()
		self.decoder_params = decoder_params
		self.t_params = t_params
		self.layer_count = decoder_params['decoder_layer_count']
		self.seq_len_factor_expansion = decoder_params['seq_len_factor_expansion']
		self.h_w = h_w
		
		self.CGRU_layers = []
		self.upscale_layers = []
		for idx in range( self.layer_count ):
			# _layer = TRUNET_CGRU_Decoder_Layer( t_params, self.decoder_params['CGRUs_params'][idx], 
			# 									decoder_params['seq_len_factor_expansion'][idx],
			# 									decoder_params['seq_len'][idx], h_w )
			_layer = layers_convgru2D.ConvGRU2D(**self.decoder_params['CGRUs_params'][idx],trainable=self.trainable )
			_layer_upscale = tf.keras.layers.TimeDistributed( SubpixelConv2D(2) )
			self.CGRU_layers.append(_layer)
			self.upscale_layers.append(_layer_upscale)

		self.seq_lens = self.decoder_params['attn_layer_no_splits']
		self.shape1 = [ sum( self.seq_lens ) ] + [ self.t_params['batch_size'] ] + [h_w[0], h_w[1], self.decoder_params['CGRUs_params'][0]['filters']*2 ] 
	
	def call(self, last_states, training=True):

		li_lss_outp = tf.split(last_states, self.seq_lens, axis=0 ) #last_states outp

		dec_hs_outp = tf.RaggedTensor.to_tensor( li_lss_outp[-1] )
		dec_hs_outp = tf.keras.backend.repeat_elements( dec_hs_outp, self.seq_len_factor_expansion[0] , axis=1) #(bs, seq_len1, h,w,c2)

		for idx in range(2, self.layer_count+1):
			
			initial_state = tf.RaggedTensor.to_tensor( li_lss_outp[ self.layer_count-idx] )
			
			dec_hs_outp =  self.CGRU_layers[-idx]( dec_hs_outp, initial_state = initial_state )

			dec_hs_outp = self.upscale_layers[idx](dec_hs_outp)

			dec_hs_outp = tf.keras.backend.repeat_elements( dec_hs_outp,self.seq_len_factor_expansion[idx-1] , axis=1)

		return dec_hs_outp

#endregion


# Upscaling Layers
class SubpixelConv2D(tf.keras.layers.Layer):
    """ Subpixel Conv2D Layer
    upsampling a layer from (h, w, c) to (h*r, w*r, c/(r*r)),
    where r is the scaling factor, default to 4
    # Arguments
    upsampling_factor: the scaling factor
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        the second and the third dimension increased by a factor of
        `upsampling_factor`; the last layer decreased by a factor of
        `upsampling_factor^2`.
    # References
        Real-Time Single Image and Video Super-Resolution Using an Efficient
        Sub-Pixel Convolutional Neural Network Shi et Al. https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upsampling_factor=4, **kwargs):
        super(SubpixelConv2D, self).__init__(**kwargs)
        self.upsampling_factor = upsampling_factor

    def build(self, input_shape):
        last_dim = input_shape[-1]
        factor = self.upsampling_factor * self.upsampling_factor
        if last_dim % (factor) != 0:
            raise ValueError('Channel ' + str(last_dim) + ' should be of '
                             'integer times of upsampling_factor^2: ' +
                             str(factor) + '.')

    def call(self, inputs, **kwargs):
        return tf.nn.depth_to_space( inputs, self.upsampling_factor )

    def get_config(self):
        config = { 'upsampling_factor': self.upsampling_factor, }
        base_config = super(SubpixelConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        factor = self.upsampling_factor * self.upsampling_factor
        input_shape_1 = None
        if input_shape[1] is not None:
            input_shape_1 = input_shape[1] * self.upsampling_factor
        input_shape_2 = None
        if input_shape[2] is not None:
            input_shape_2 = input_shape[2] * self.upsampling_factor
        dims = [ input_shape[0],
                 input_shape_1,
                 input_shape_2,
                 int(input_shape[3]/factor)
               ]
        return tuple( dims )
# endregion

# region activation layers
class OutputReluFloat32(tf.keras.layers.Layer):
	def __init__(self, t_params):
		super(OutputReluFloat32, self).__init__()
		self.custom_relu = CustomRelu_maker(t_params, dtype='float32')
		self.outputf32 = tf.keras.layers.Activation('linear', dtype='float32')
	
	#@tf.function
	def call(self, inputs):
		outp = self.outputf32(inputs)
		outp = self.custom_relu(outp)
		return outp

def CustomRelu_maker(t_params, dtype):
	CustomRelu = ReLU_correct_layer( threshold= utility.standardize_ati( 0.0, t_params['normalization_shift']['rain'], 
									t_params['normalization_scales']['rain'], reverse=False), sdtype=dtype )
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
            threshold: Float. Threshold value for thresholded activation."""
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
            A tensor.    """
    def __init__(self, max_value=None, negative_slope=0.0, threshold=0.0, sdtype='float32' ,**kwargs):
        super(ReLU_correct_layer, self).__init__()
        if max_value is not None and max_value < 0.:
            raise ValueError('max_value of Relu layer '
                            'cannot be negative value: ' + str(max_value))
        if negative_slope < 0.:
            raise ValueError('negative_slope of Relu layer '
                            'cannot be negative value: ' + str(negative_slope))

        self.support_masking = True
        if max_value is not None:
            #max_value = K.cast_to_floatx(max_value)
            max_value = K.cast(max_value, sdtype)
        self.max_value = max_value
        #self.negative_slope = K.cast_to_floatx(negative_slope)
        #self.negative_slope = K.cast(negative_slope,sdtype)
        #self.negative_slope = tf.constant(negative_slope,sdtype )
        self.negative_slope = np.array([negative_slope],dtype=sdtype)		
        #self.threshold = K.cast_to_floatx(threshold)
        #self.threshold = K.cast(threshold, sdtype)
        self.threshold = np.array([threshold],dtype=sdtype)		
        self.sdtype = sdtype
        self._dtype = sdtype
    
    def call(self, inputs):
        # alpha is used for leaky relu slope in activations instead of
        # negative_slope.

        x = inputs
        if self.negative_slope != 0.0:
            #tf.print(self.negative_slope, output_stream=sys.stdout)
        #if tf.math.not_equal( self.negative_slope, 0.0 ):

            if self.max_value is None and self.threshold == 0:
                #return nn.leaky_relu(x, alpha=self.negative_slope)
                return K.relu(x, alpha=self.negative_slope)

            if self.threshold != 0:
                #negative_part = nn.relu(-x + self.threshold)
                negative_part = K.relu( -x + self.threshold)
            else:
                #negative_part = nn.relu(-x)
                negative_part = K.relu(-x)

        #clip_max = max_value != None #Note: This may not evaluate to false in graph mode
        clip_max = False

        if self.threshold != 0:
            # computes x for x > threshold else 0
            #x = x * math_ops.cast(math_ops.greater(x, threshold), K.floatx())
            #x = x * math_ops.cast(math_ops.greater(x, self.threshold), x.dtype.base_dtype) + self.threshold * math_ops.cast(math_ops.greater_equal(self.threshold, x), x.dtype.base_dtype)
		    #x = x * math_ops.greater(x, threshold) + threshold * math_ops.greater_equal(threshold, x)
            x = x * tf.cast(tf.math.greater(x, self.threshold), x.dtype.base_dtype) + self.threshold * tf.cast(tf.greater(self.threshold, x), x.dtype.base_dtype)
        elif self.max_value == 6:
            # if no threshold, then can use nn.relu6 native TF op for performance
            x = nn.relu6(x)
            clip_max = False
        else:
            x = nn.relu(x)

        if clip_max == True:
            self.max_value = K._to_tensor(self.max_value, x.dtype.base_dtype)
            #zero = K._to_tensor(0., x.dtype.base_dtype)
            x = clip_ops.clip_by_value(x, self.threshold, self.max_value)

        if self.negative_slope != 0.:
            self.negative_slope = K._to_tensor(self.negative_slope, x.dtype.base_dtype)
            x -= self.negative_slope * negative_part
		
        return x
        
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

def LeakyRelu_mkr(t_params):
	CustomRelu = tf.keras.layers.ReLU( threshold=utility.standardize_ati( 0, t_params['normalization_shift']['rain'], 
															t_params['normalization_scales']['rain'], reverse=False), negative_slope=0.1 )

	return CustomRelu

# endregion
