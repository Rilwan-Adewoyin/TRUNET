
import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float16')

from tensor2tensor.layers.common_attention import split_heads, combine_heads, maybe_upcast
from tensor2tensor.layers.common_attention import dot_product_attention, dot_product_attention_relative, dot_product_unmasked_self_attention_relative_v2, dot_product_self_attention_relative_v2
from tensor2tensor.layers.common_attention import compute_attention_component, harden_attention_weights
from tensor2tensor.layers.common_layers import dense as t2t_dense, dropout_with_broadcast_dims, cast_like
from tensorflow.python.ops import inplace_ops

##New imports
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import regularizers

from tensorflow.keras.layers import Layer
from tensorflow.python.keras.utils import conv_utils, generic_utils, tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export

from tensorflow.keras.layers import Conv2D, RNN

"""This script contains code to support the multi-head cross attention operation
"""

class MultiHead2DAttention_v2(Layer):
    """Multihead scaled-dot-product attention with input/output transformations.
        Adapted from the Tensor2Tensor attention mechanism
        Args:
            bias: bias Tensor (see attention_bias())
            total_key_depth: an integer
            total_value_depth: an integer
            output_depth: an integer
            num_heads: an integer dividing total_key_depth and total_value_depth
            dropout_rate: a floating point number
            attn_factor_reduc: an integer indicating the ratio of temporal size
                                of input to output.
            value_conv: dictionary containing params for the convolution operation on
                        the value precendent       
            output_conv: dictionary containing params for the convolution operation on
                        the output precedent                                                    
            compat_dict: dictionary to assist backwards compatibility
            transform_value_antecedent: whether or not to include conv op on value
                        antecedent
            transform_output: whether or not to include conv op on output
            max_relative_position: Maximum distance between inputs to generate
                                unique relation embeddings for. Only relevant
                                when using "dot_product_relative" attention.
            heads_share_relative_embedding: boolean to share relative embeddings
            add_relative_to_values: a boolean for whether to add relative component to
                                        values.
            name: an optional string.
            dropout_broadcast_dims:  an optional list of integers less than 4  
                                    specifying in which dimensions to broadcast
                                     the dropout decisions. saves memory.
            hard_attention_k: integer, if > 0 triggers hard attention 
            training: indicating if it is in the training mode.
            **kwargs (dict): Parameters for the attention function.

            #TODO: introduce params below to assist with visualizing attention
            image_shapes: optional tuple of integer scalars.
                        see comments for attention_image_summary()
            save_weights_to: an optional dictionary to capture attention weights
                            for vizualization; the weights tensor will be appended 
                            there under a string key created from the variable scope 
                            (including name).
            make_image_summary: Whether to make an attention image summary.

        Returns:
                The result of the attention transformation. The output shape is
                [batch_size, seq_len, h, w, c]
        Raises:
                ValueError: if the key depth or value depth are not divisible by the
                number of attention heads.
    """
    def __init__(self, attention_scaling_params,trainable,
                    bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        attn_factor_reduc,
                        value_conv,
                        output_conv,
                        compat_dict,
                        transform_value_antecedent=True,
                        transform_output=True,
                        max_relative_position=None, 
                        heads_share_relative_embedding=False,
                        add_relative_to_values=False,
                        name="multihead_rel_attention",
                        dropout_broadcast_dims=None, 
                        hard_attention_k=0,
                        training=True,
                        conv_ops_qk = False,
                        key_conv = None,
                        query_conv = None,
                        **kwargs):

 
        #region --- arguments
        self.trainable = trainable
        self.bias = bias
        self.total_key_depth = total_key_depth
        self.total_value_depth = total_value_depth
        self.output_depth = output_depth
        self.num_heads = num_heads
        self.key_depth_per_head = total_key_depth // num_heads
        self.dropout_rate = dropout_rate
        self.hard_attention_k = hard_attention_k
        self.attn_factor_reduc = attn_factor_reduc
        
        self.compat_dict = compat_dict
        self.transform_value_antecedent = transform_value_antecedent
        self.value_conv = value_conv
        self.output_conv = output_conv
        self.transform_output = transform_output
        self.add_relative_to_values = add_relative_to_values
        self.max_relative_position = max_relative_position                    
        self.heads_share_relative_embedding = heads_share_relative_embedding 
        
        self.conv_ops_qk = conv_ops_qk
        self.dropout_broadcast_dims = dropout_broadcast_dims

        self.kq_downscale_kernelshape = attention_scaling_params['kq_downscale_kernelshape']
        self.kq_downscale_stride = attention_scaling_params['kq_downscale_stride']
        # endregion       
        
        #region Layer Checks & Prep
        super( MultiHead2DAttention_v2, self ).__init__()

        assert_op1 = tf.Assert( tf.equal( tf.math.floormod(total_key_depth, num_heads), 0 ), [total_key_depth, tf.constant(num_heads)] )
        assert_op2 = tf.Assert( tf.equal( tf.math.floormod(total_value_depth, num_heads), 0 ), [total_value_depth, tf.constant(num_heads)] )
        # endregion

        #region attention layers
        with tf.control_dependencies([assert_op1, assert_op2]):
            if self.conv_ops_qk == True:
                #model variant - convolution operations on query and key antecedents
                self.conv_query = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **query_conv ) )
                self.conv_key =   tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D( **key_conv ) )
            else:
                #model variant - dense operations on query and key antecedents
                self.dense_query =  tf.keras.layers.Dense( total_key_depth, use_bias=False, activation="linear", name="q")
                self.dense_key   =  tf.keras.layers.Dense( total_key_depth, use_bias=False, activation="linear", name="k")  
        
        if self.transform_value_antecedent == True:
            #model variant - convolution operation on value antecedent
            self.conv_value = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D(  **self.value_conv ) ) # This has been used for all other THST models
        
        if self.transform_output == True:
            #model vairant - convolution operation of value output
            #self.dense_output = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D(  **output_conv) ) 
            self.conv_output = tf.keras.layers.TimeDistributed( tf.keras.layers.Conv2D(  **output_conv) ) 

        #Maximum relative attention
        if( self.max_relative_position==None ):
           self.max_relative_position =  tf.constant( int(self.attn_factor_reduc/2 - 1) , dtype=tf.int32 )

        embding_size = int( self.attn_factor_reduc ) #int(self.max_relative_position * 2 + 1)
        self.embeddings_table_k = tf.Variable( tf.keras.initializers.glorot_uniform()(shape=[embding_size, total_key_depth//num_heads],   dtype=self._dtype ), name="embedding_table_k" )
        self.embeddings_table_v = tf.Variable( tf.keras.initializers.glorot_uniform()(shape=[embding_size, total_value_depth//num_heads], dtype=self._dtype ), name="embedding_table_v" ) 

        #endregion

    
    def call(self, inputs , k_antecedent, v_antecedent, training=True):
        """Performs the multi-head attention as described in the paper

        Args:
            inputs : a Tensor with shape (batch_size, 1, h, w, channels); query_antecedent
            k_antecedent : a Tensor with shape (batch_size, seq_len, h, w, channels)
            v_antecedent : a Tensor with shape (batch_size, seq_len, h, w, channels)
                
            training (bool, optional): whetehr variables should be updated

        Returns:
            tensor: A Tensor with shape (batch_size, 1, h, w, c)
        """
        # region --- calculating q k v #Link To Paper: Equation 2
            # Note: In the paper, we explain the methodology used when self.conv_ops_qk == False
        output_shape = v_antecedent.shape.as_list() 
        output_shape[1] = 1 # inputs.shape[1]

        if self.conv_ops_qk == False:
            #Link To Paper - Equation 2, 3D avg pool operations
            q_antecedent = tf.cast( tf.nn.avg_pool3d( tf.cast(inputs,tf.float32), strides=self.kq_downscale_stride,
                                        ksize=self.kq_downscale_kernelshape, padding="SAME"), tf.float16) #( batch_size, seq_len, height,width,filters_in) 
            k_antecedent = tf.cast(tf.nn.avg_pool3d( tf.cast(k_antecedent,tf.float32), strides=self.kq_downscale_stride,
                                    ksize=self.kq_downscale_kernelshape, padding="SAME"), tf.float16)
        else:
            q_antecedent = inputs
            k_antecedent = inputs
                                    
        if self.conv_ops_qk == False:
            # reshping from 3D to 2D  for attention along the temporal dimension
            q_antecedent_flat = tf.reshape(q_antecedent, q_antecedent.shape.as_list()[:2] + [-1] ) #( batch_size, seq_len, height*width*filters_in) 
            k_antecedent_flat = tf.reshape(k_antecedent, k_antecedent.shape.as_list()[:2] + [-1] ) #( batch_size, seq_len, height*width*filters_in) 
            # Dense operations on reshaped/flattened query and key antecedents
            q = self.dense_query(q_antecedent_flat)
            k = self.dense_key(k_antecedent_flat)
        else:
            #Using convolution operations on query and key antecedents
            q_antecedent = self.conv_query( q_antecedent, training=True ) #(bs, seq_len, h, w, c)
            k_antecedent = self.conv_key( k_antecedent, training=True )

            q_antecedent_flat = tf.reshape(q_antecedent, q_antecedent.shape.as_list()[:2] + [-1] ) #(bs, seq_len, h*w*c)
            k_antecedent_flat = tf.reshape(k_antecedent, k_antecedent.shape.as_list()[:2] + [-1] ) 
            q = q_antecedent_flat
            k = k_antecedent_flat

        # convolution operation on value antecedent
        if self.transform_value_antecedent == True:
            v_antecedent = self.conv_value( v_antecedent, training=True  )
        # endregion

        # flattening value antecedent for compatibility reasons   
        v_antecedent_flat = tf.reshape(v_antecedent, v_antecedent.shape.as_list()[:2] + [-1] ) 
        v = v_antecedent_flat
        
        #region Scaled --- Relative Multi-Head Dot-Product Attention
        
        # gathering multiple heads # Link to Paper: Equation 
        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        
        q *= tf.cast(self.key_depth_per_head,dtype=q.dtype)**-0.5      

        # relative positional embeddings for keys and values
        q_length = q.shape.as_list()[2]
        k_length = k.shape.as_list()[2]
        relations_keys = _generate_relative_positions_embeddings( q_length, k_length,
                                self.max_relative_position, self.embeddings_table_k, self._compute_dtype )

        relations_values = _generate_relative_positions_embeddings(q_length, k_length,
                                self.max_relative_position, self.embeddings_table_v, self._compute_dtype )
        
        # Compute attention w/ relative positional embeddings
        logits = _relative_attention_inner(q, k, relations_keys, transpose=True) #Link To Paper: Equations (3) - Score operation

        # masking attention logits using bias #In our implementation no bias is used
        if self.bias is not None:
            bias = cast_like(self.bias, logits)
            logits += bias

        # If logits are fp16, upcast before softmax
        logits = maybe_upcast(logits, self._compute_dtype, self.dtype)
        weights = tf.nn.softmax(logits, name="attention_weights") #Link To Paper: Equations (3) - normalizing exp()/sum(exp()) operation
        if self.hard_attention_k > 0:
            weights = harden_attention_weights(weights, self.hard_attention_k)
        weights = cast_like(weights, q)

        # Dropping out attention links for each head.
        weights = dropout_with_broadcast_dims(
            weights, 1.0 - self.dropout_rate, broadcast_dims=self.dropout_broadcast_dims) 

        outp = _relative_attention_inner(weights, v, relations_values, False) #Link To Paper: Equations (3) - calculating \hat(A}

        outp = combine_heads(outp)
        
        if self.transform_output == True:
            # convolution ops on output precedent \hat{A}
            outp.set_shape(outp.shape.as_list()[:-1] + [self.total_value_depth]) 
            outp = tf.reshape( outp, output_shape )
            outp = self.conv_output( outp, training=training)
        else:
            outp = tf.reshape( outp, output_shape ) 
        
        # endregion

        return outp    #( batch_size, seq_len, height, width, filters_in)

    def get_config(self):
        config = {
            'trainable':
                self.trainable,
            'bias':
                self.bias,
            'total_key_depth':
                self.total_key_depth,
            'total_value_depth':
                self.total_value_depth,
            'output_depth':
            self.output_depth,
            'num_heads':
                self.num_heads,
            'key_depth_per_head':
                self.key_depth_per_head,
            'dropout_rate':
                self.dropout_rate,
            'hard_attention_k':
                self.hard_attention_k,
            'attn_factor_reduc':
                self.attn_factor_reduc,
            'compat_dict':
                self.compat_dict,
            'trainsform_value_antecedent':
                self.transform_value_antecedent,
            'value_conv':
                self.value_conv,
            'output_conv':
                self.output_conv,
            'transoform_output':
                self.transform_output,
            'add_relative_to_values':
                self.add_relative_to_values,
            'max_relative_position':
                self.max_relative_position,
            'heads_share_relative_embedding':
                self.heads_share_relative_embedding
                
        }

        return config

def _generate_relative_positions_embeddings( length_q, length_k,
                                        max_relative_position, embeddings_table, dtype):
    """ Generates tensor of size [length_q, length_k, depth],
            encoding the relative positional embedding

        Refer to Self-Attention with Relative Position Representations
            Peter Shaw, Jakob Uszkoreit, Ashish Vaswani
    """
    if length_q == length_k:
        range_vec_q = range_vec_k = tf.range(length_q)
    else:
        range_vec_k = tf.range(length_k)
        range_vec_q = range_vec_k[-length_q:]
    
    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
    distance_mat_clipped = tf.clip_by_value( distance_mat, -max_relative_position,
                                                max_relative_position )
    
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    relative_positions_matrix = distance_mat_clipped + max_relative_position
    
    embeddings = tf.gather(embeddings_table, relative_positions_matrix)

    return tf.cast(embeddings,dtype)

def _relative_attention_inner(x, y, z, transpose):
    """Relative position-aware dot-product attention inner calculation.

        This batches matrix multiply calculations to avoid unnecessary broadcasting.

        Args:
            x: Tensor with shape [batch_size, heads, length or 1, length or depth].
            y: Tensor with shape [batch_size, heads, length or 1, depth].
            z: Tensor with shape [length or 1, length, depth].
            transpose: Whether to transpose inner matrices of y and z. Should be true if
                last dimension of x is depth, not length.

        Returns:
            A Tensor with shape [batch_size, heads, length, length or depth].
    """
    batch_size = tf.shape(x)[0]
    heads = x.get_shape().as_list()[1]
    length = tf.shape(x)[2]

        # xy_matmul is [batch_size, heads, length or 1, length or depth]
    xy_matmul = tf.matmul(x, y, transpose_b=transpose)
        # x_t is [length or 1, batch_size, heads, length or depth]
    x_t = tf.transpose(x, [2, 0, 1, 3])
        # x_t_r is [length or 1, batch_size * heads, length or depth]
    x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
        # x_tz_matmul is [length or 1, batch_size * heads, length or depth]
    x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
        # x_tz_matmul_r is [length or 1, batch_size, heads, length or depth]
    x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
        # x_tz_matmul_r_t is [batch_size, heads, length or 1, length or depth]
    x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
    return xy_matmul + x_tz_matmul_r_t

def attn_shape_adjust(inputs, attn_factor_reduc, reverse=False):

    """ Used to adjust the size of the time dimension, 
        This is ideal when passing multiple 3D tensors to an RNN cell which only accepts one input
            - input data must be reshaped so time dim t -> 1 and channel dim c -> c*t 
        if(reverse=False):
            :param tnsr inputs: (bs, tss, h, w, c)
            return outputs : (bs, tss/seq_len_factor_reduc, h, w, c*seq_len_factor_reduc )
        if(reverse=True):
            :param tnsr inputs: (bs, 1, h, w, c ) 
            return outputs : (bs, seq_len_factor_reduc, h, w, c//seq_len_factor_reduc)
    """

    if reverse==False:
        shape = inputs.shape

        outp = tf.reshape(inputs, shape[:1]+shape[1]//attn_factor_reduc+shape[2:4]+shape[4]*attn_factor_reduc )
    else:
        shape = tf.expand_dims(inputs, axis=1).shape
        outp = tf.reshape(inputs, shape[:1]+shape[1]*attn_factor_reduc+shape[2:4]+shape[4]//attn_factor_reduc )

    return outp