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
#from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D

from tensorflow.keras.layers import Conv2D, RNN

class MultiHead2DAttention_v2(Layer):
    def __init__(self, attention_scaling_params,trainable,
                                bias,
                                total_key_depth,
                                total_value_depth,
                                output_depth,
                                num_heads,
                                dropout_rate,
                                attn_factor_reduc,
                                transform_value_antecedent=True,
                                transform_output=True,
                                max_relative_position=None, #TODO: add code for this much later
                                heads_share_relative_embedding=False,
                                add_relative_to_values=False,
                                name="multihead_rel_attention",
                                dropout_broadcast_dims=None, 
                                chunk_number=None,
                                hard_attention_k=0,
                                training=True,
                                model_location="wholeregion",
                                **kwargs):

        """
            TODO: prior to the attention possibly add something like squeeze and excitation to reweight the feature maps. But only in the first layer since taking in the original feature maps, as it shouldnt be needed after

            Either use 2D attention or try flattening nromal tensors to vectors so normal attention can be used
            Flattening used in https://arxiv.org/pdf/1904.09925.pdf, so will use their flattening method
        """
        """Multihead scaled-dot-product attention with input/output transformations.
            Args:
                query_antecedent: a Tensor with shape [batch, length_q, channels]
                memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
                bias: bias Tensor (see attention_bias())
                total_key_depth: an integer
                total_value_depth: an integer
                output_depth: an integer
                num_heads: an integer dividing total_key_depth and total_value_depth
                dropout_rate: a floating point number
                max_relative_position: Maximum distance between inputs to generate
                                    unique relation embeddings for. Only relevant
                                    when using "dot_product_relative" attention.
                heads_share_relative_embedding: boolean to share relative embeddings
                add_relative_to_values: a boolean for whether to add relative component to
                                            values.
                image_shapes: optional tuple of integer scalars.
                            see comments for attention_image_summary()
                block_length: an integer - relevant for "local_mask_right"
                block_width: an integer - relevant for "local_unmasked"
                q_filter_width: An integer specifying how wide you want the query to be.
                kv_filter_width: An integer specifying how wide you want the keys and values
                                to be.
                q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
                        kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
                        no padding.
                cache: dict containing Tensors which are the results of previous
                        attentions, used for fast decoding. Expects the dict to contrain two
                        keys ('k' and 'v'), for the initial call the values for these keys
                        should be empty Tensors of the appropriate shape.
                        'k' [batch_size, 0, key_channels]
                        'v' [batch_size, 0, value_channels]
                gap_size: Integer option for dilated attention to indicate spacing between
                        memory blocks.
                num_memory_blocks: Integer option to indicate how many memory blocks to look
                                at.
                name: an optional string.
                save_weights_to: an optional dictionary to capture attention weights
                for vizualization; the weights tensor will be appended there under
                a string key created from the variable scope (including name).
                make_image_summary: Whether to make an attention image summary.
                dropout_broadcast_dims:  an optional list of integers less than 4
                specifying in which dimensions to broadcast the dropout decisions.
                saves memory.
                vars_3d: use 3-dimensional variables for input/output transformations
                layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
                KFAC optimizer. Default is None.
                recurrent_memory: An optional transformer_memory.RecurrentMemory, which
                retains state across chunks. Default is None.
                chunk_number: an optional integer Tensor with shape [batch] used to operate
                the recurrent_memory.
                hard_attention_k: integer, if > 0 triggers hard attention (picking top-k).
                gumbel_noise_weight: if > 0, apply Gumbel noise with weight
            `gumbel_noise_weight` before picking top-k. This is a no op if
                hard_attention_k <= 0.
                max_area_width: the max width allowed for an area.
                max_area_height: the max height allowed for an area.
                memory_height: the height of the memory.
                area_key_mode: the mode for computing area keys, which can be "mean",
            "concat", "sum", "sample_concat", and "sample_sum".
                area_value_mode: the mode for computing area values, which can be either
            "mean", or "sum".
                training: indicating if it is in the training mode.
            **kwargs (dict): Parameters for the attention function.
            Caching:
                    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
                    the caching assumes that the bias contains future masking.
                    The caching works by saving all the previous key and value values so that
                    you are able to send just the last query location to this attention
                    function. I.e. if the cache dict is provided it assumes the query is of the
                    shape [batch_size, 1, hidden_dim] rather than the full memory.
            Returns:
                    The result of the attention transformation. The output shape is
                    [batch_size, length_q, hidden_dim]
                    unless the cache dict is provided in which case only the last memory
                    position is calculated and the output shape is [batch_size, 1, hidden_dim]
                    Optionally returns an additional loss parameters (ex: load balance loss for
                    the experts) returned by the attention_type function.
            Raises:
                    ValueError: if the key depth or value depth are not divisible by the
                    number of attention heads.
        """
        #region attach args
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
        
        self.transform_value_antecedent = transform_value_antecedent
        self.transform_output = transform_output
        self.heads_share_relative_embedding = heads_share_relative_embedding
        self.add_relative_to_values = add_relative_to_values
        self.max_relative_position = max_relative_position                    #TODO: add this functionality much later
        self.heads_share_relative_embedding = heads_share_relative_embedding #TODO: add this functionality much later
        
        self.dropout_broadcast_dims = dropout_broadcast_dims

        self.kq_downscale_kernelshape = attention_scaling_params['kq_downscale_kernelshape']
        self.kq_downscale_stride = attention_scaling_params['kq_downscale_stride']
        # endregion       
        
        #region Layer Checks & Prep
        super( MultiHead2DAttention_v2, self ).__init__()

        assert_op1 = tf.Assert( tf.equal( tf.math.floormod(total_key_depth, num_heads), 0 ), [total_key_depth, tf.constant(num_heads)] )
        assert_op2 = tf.Assert( tf.equal( tf.math.floormod(total_value_depth, num_heads), 0 ), [total_value_depth, tf.constant(num_heads)] )

        with tf.control_dependencies([assert_op1, assert_op2]):
            self.ln1 = tf.keras.layers.LayerNormalization(axis=-1 , epsilon=1e-4 , trainable=self.trainable )
        # endregion

        #region scaling
        # if model_location == "wholeregion":
        #     self.scaling_layer = tf.keras.layers.AveragePooling3D( pool_size=tuple(self.kq_downscale_kernelshape),
        #                         strides=tuple(self.kq_downscale_stride), padding='same' )
        # elif model_location == "region-grid":
        #     self.scaling_layer = tf.keras.layers.AveragePooling3D( pool_size=tuple(self.kq_downscale_kernelshape),
        # #                         strides=tuple(self.kq_downscale_stride), padding='same' )
        # else:
        #     raise ValueError
        # endregion

        #region attention layers
        self.dense_query =  tf.keras.layers.Dense( total_key_depth, use_bias=False, activation="linear", name="q")
        self.dense_key =    tf.keras.layers.Dense( total_key_depth, use_bias=False, activation="linear", name="k")  
        
        if transform_value_antecedent:
            self.dense_value = tf.keras.layers.Dense( total_value_depth, use_bias=False, activation="linear", name="v" )
        else:
            self.dense_value = tf.keras.layers.Activation("linear")
        
        if( self.max_relative_position==None ):
           self.max_relative_position =  tf.constant( int(self.attn_factor_reduc/2 - 1) , dtype=tf.int32 )

        vocab_size = int(self.attn_factor_reduc) #int(self.max_relative_position * 2 + 1)
        self.embeddings_table_k = tf.Variable( tf.keras.initializers.glorot_uniform()(shape=[vocab_size, total_key_depth//num_heads ], dtype=self._compute_dtype  ))
        self.embeddings_table_v = tf.Variable( tf.keras.initializers.glorot_uniform()(shape=[vocab_size, total_value_depth//num_heads ], dtype=self._compute_dtype  )) 

        if transform_output:
            self.dense_output = tf.keras.layers.Dense( output_depth, use_bias=False  )
        elif not transform_output:
            self.dense_output = tf.keras.layers.Activation("linear")
        #endregion

    @tf.function
    def call(self, inputs , k_antecedent, v_antecedent):
        """
            :param inputs: q_antecedent This is required due to keras' need for layers to have an input argument

            :inputs: is queries
        """
      
        # region size reduction
        output_shape = v_antecedent.shape.as_list() #NOTE shape.as_list()[:-1] may not work in graph mode
        output_shape[1] = 1 # inputs.shape[1]


        q_antecedent = tf.cast( tf.nn.avg_pool3d( tf.cast(inputs,tf.float32), strides=self.kq_downscale_stride,
                                ksize=self.kq_downscale_kernelshape, padding="SAME"), tf.float16)
        k_antecedent = tf.cast(tf.nn.avg_pool3d( tf.cast(k_antecedent,tf.float32), strides=self.kq_downscale_stride,
                                ksize=self.kq_downscale_kernelshape, padding="SAME"), tf.float16)
        # q_antecedent = tf.cast( self.scaling_layer( tf.cast(inputs,tf.float32)  ), tf.float16)
        # k_antecedent = tf.cast( self.scaling_layer( tf.cast(k_antecedent,tf.float32)  ), tf.float16)

        # endregion 

        # region reshping from 3D to 2D reshaping for attention
        q_antecedent_flat = tf.reshape(q_antecedent, q_antecedent.shape.as_list()[:2]  + [-1] ) #( batch_size, seq_len, height*width*filters_in) #NOTE shape.as_list()[:-1] may not work in graph mode
        k_antecedent_flat = tf.reshape( k_antecedent, k_antecedent.shape.as_list()[:2] +[-1] ) #NOTE shape.as_list()[:-1] may not work in graph mode
        v_antecedent_flat = tf.reshape(v_antecedent, v_antecedent.shape.as_list()[:2] + [-1] ) #NOTE shape.as_list()[:-1] may not work in graph mode
        # endregion

        #region Dot-Product Attention
        #calculating q k v
        q = self.dense_query(q_antecedent_flat)
        k = self.dense_key(  k_antecedent_flat)
        v = self.dense_value(v_antecedent_flat)

        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)

        q *= tf.cast(self.key_depth_per_head,dtype=q.dtype)**-0.5      #scaled dot production attn   

        #Adding relative attn
        # Use separate embeddings suitable for keys and values.
        q_length = q.shape.as_list()[2]
        k_length = k.shape.as_list()[2]
        relations_keys = _generate_relative_positions_embeddings( q_length, k_length,
                                        self.max_relative_position, self.embeddings_table_k, self._compute_dtype )
        relations_values = _generate_relative_positions_embeddings(q_length, k_length,
                                        self.max_relative_position, self.embeddings_table_v, self._compute_dtype )
        
        # Compute self attention considering the relative position embeddings.
        logits = _relative_attention_inner(q, k, relations_keys, transpose=True)

        if self.bias is not None:
            bias = cast_like(self.bias, logits)
            logits += bias

        # If logits are fp16, upcast before softmax
        logits = maybe_upcast(logits, self._compute_dtype, self.dtype)
        weights = tf.nn.softmax(logits, name="attention_weights")
        if self.hard_attention_k > 0: #TODO: fix for graph mode
            weights = harden_attention_weights(weights, self.hard_attention_k)
        weights = cast_like(weights, q)

        # Drop out attention links for each head.
        weights = dropout_with_broadcast_dims(
            weights, 1.0 - self.dropout_rate, broadcast_dims=self.dropout_broadcast_dims)

        x = _relative_attention_inner(weights, v, relations_values, False)

        x = combine_heads(x)
        x.set_shape(x.shape.as_list()[:-1] + [self.total_value_depth]) #NOTE: x.shape.as_list()[:-1] may not work in graph mode
        x = self.dense_output(x)
        # endregion

        #x = self.ln1(x) #( batch_size, seq_len, height*width*filters_in #NOTE: doesnt work on cpu, add tf.test.is_gpu_available() to make this layer conditional
        
        x = tf.reshape( x ,  output_shape ) #( batch_size, seq_len, height, width, filters_in)

        return x

def _generate_relative_positions_embeddings( length_q, length_k,
                                        max_relative_position, embeddings_table,dtype):
    if length_q == length_k:
        range_vec_q = range_vec_k = tf.range(length_q)
    else:
        range_vec_k = tf.range(length_k)
        range_vec_q = range_vec_k[-length_q:]
    distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
    distance_mat_clipped = tf.clip_by_value( distance_mat, -max_relative_position,
                                          max_relative_position)
    # Shift values to be >= 0. Each integer still uniquely identifies a relative
    # position difference.
    final_mat = distance_mat_clipped + max_relative_position

    relative_positions_matrix = final_mat
    
    embeddings = tf.gather(embeddings_table, relative_positions_matrix)
    return embeddings

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
