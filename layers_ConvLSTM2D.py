import tensorflow as tf
import numpy as np

from tensor2tensor.layers.common_attention import split_heads, combine_heads
from tensor2tensor.layers.common_attention import dot_product_attention, dot_product_attention_relative, dot_product_unmasked_self_attention_relative_v2, dot_product_self_attention_relative_v2
from tensor2tensor.layers.common_attention import compute_attention_component
from tensor2tensor.layers.common_layers import dense as t2t_dense
from tensorflow.python.ops import inplace_ops

##New imports
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.keras import regularizers

from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers.recurrent import _standardize_args, DropoutRNNCellMixin, RNN, _is_multiple_state
from tensorflow.python.keras.utils import conv_utils, generic_utils, tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D

from tensorflow.keras.layers import Conv2D


class ConvLSTM2D_custom(ConvRNN2D):
    """
        CUSTOM Convolutional LSTM.

        My key change is that I allow input to be two tensors [ input1 and input2 so our LSTM cell can operate on information from two time lengths+]
        Init Arguments Added:

        Call Arguments Added:

        It is similar to an LSTM layer, but the input transformations
        and recurrent transformations are both convolutional.
        Arguments:
            filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the convolution).
            kernel_size: An integer or tuple/list of n integers, specifying the
                dimensions of the convolution window.
            strides: An integer or tuple/list of n integers,
                specifying the strides of the convolution.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            padding: One of `"valid"` or `"same"` (case-insensitive).
            data_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, time, ..., channels)`
                while `channels_first` corresponds to
                inputs with shape `(batch, time, channels, ...)`.
                It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
                If you never set it, then it will be "channels_last".
            dilation_rate: An integer or tuple/list of n integers, specifying
                the dilation rate to use for dilated convolution.
                Currently, specifying any `dilation_rate` value != 1 is
                incompatible with specifying any `strides` value != 1.
            activation: Activation function to use.
                By default hyperbolic tangent activation function is applied
                (`tanh(x)`).
            recurrent_activation: Activation function to use
                for the recurrent step.
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix,
                used for the linear transformation of the inputs.
            recurrent_initializer: Initializer for the `recurrent_kernel`
                weights matrix,
                used for the linear transformation of the recurrent state.
            bias_initializer: Initializer for the bias vector.
            unit_forget_bias: Boolean.
                If True, add 1 to the bias of the forget gate at initialization.
                Use in combination with `bias_initializer="zeros"`.
                This is recommended in [Jozefowicz et al.]
                (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix.
            recurrent_regularizer: Regularizer function applied to
                the `recurrent_kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to.
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix.
            recurrent_constraint: Constraint function applied to
                the `recurrent_kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
            return_sequences: Boolean. Whether to return the last output
                in the output sequence, or the full sequence.
            go_backwards: Boolean (default False).
                If True, process the input sequence backwards.
            stateful: Boolean (default False). If True, the last state
                for each sample at index i in a batch will be used as initial
                state for the sample of index i in the following batch.
            dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the inputs.
            recurrent_dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the recurrent state.
        Call arguments:
            inputs: A 5D tensor.
            mask: Binary tensor of shape `(samples, timesteps)` indicating whether
                a given timestep should be masked.
            training: Python boolean indicating whether the layer should behave in
                training mode or in inference mode. This argument is passed to the cell
                when calling it. This is only relevant if `dropout` or `recurrent_dropout`
                are set.
            initial_state: List of initial state tensors to be passed to the first
                call of the cell.
        Input shape:
            - If data_format='channels_first'
                    5D tensor with shape:
                    `(samples, time, channels, rows, cols)`
            - If data_format='channels_last'
                    5D tensor with shape:
                    `(samples, time, rows, cols, channels)`
        Output shape:
            - If `return_sequences`
                    - If data_format='channels_first'
                        5D tensor with shape:
                        `(samples, time, filters, output_row, output_col)`
                    - If data_format='channels_last'
                        5D tensor with shape:
                        `(samples, time, output_row, output_col, filters)`
            - Else
                - If data_format ='channels_first'
                        4D tensor with shape:
                        `(samples, filters, output_row, output_col)`
                - If data_format='channels_last'
                        4D tensor with shape:
                        `(samples, output_row, output_col, filters)`
                where `o_row` and `o_col` depend on the shape of the filter and
                the padding
        Raises:
            ValueError: in case of invalid constructor arguments.
        References:
            - [Convolutional LSTM Network: A Machine Learning Approach for
            Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
            The current implementation does not include the feedback loop on the
            cells output.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 gates_version,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):

        self.gates_version = gates_version

        cell = ConvLSTM2DCell_custom(filters=filters,
                                     kernel_size=kernel_size,
                                     gates_version=gates_version,
                                     strides=strides,
                                     padding=padding,
                                     data_format=data_format,
                                     dilation_rate=dilation_rate,
                                     activation=activation,
                                     recurrent_activation=recurrent_activation,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     recurrent_initializer=recurrent_initializer,
                                     bias_initializer=bias_initializer,
                                     unit_forget_bias=unit_forget_bias,
                                     kernel_regularizer=kernel_regularizer,
                                     recurrent_regularizer=recurrent_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     kernel_constraint=kernel_constraint,
                                     recurrent_constraint=recurrent_constraint,
                                     bias_constraint=bias_constraint,
                                     dropout=dropout,
                                     recurrent_dropout=recurrent_dropout,
                                     dtype=kwargs.get('dtype'))

        super(ConvLSTM2D_custom, self).__init__(cell,
                                                return_sequences=return_sequences,
                                                go_backwards=go_backwards,
                                                stateful=stateful,
                                                **kwargs)
        
        self.activity_regularizer = regularizers.get(activity_regularizer)



    def call(self, inputs, mask=None, training=None, initial_state=None):
        #self._maybe_reset_cell_dropout_mask(self.cell)
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        return super(ConvLSTM2D_custom, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state)

    # region build custom
    # def build(self, input_shape):
    #     if isinstance(input_shape, list):
    #         input_shape = input_shape[0]
    #     # The input_shape here could be a nest structure.
    #     if self.gates_version==2:
    #         input_shape = input_shape[:4] +  input_shape[4]//2 #akanni changed Here

    #     # do the tensor_shape to shapes here. The input could be single tensor, or a
    #     # nested structure of tensors.
    #     def get_input_spec(shape):
    #         if isinstance(shape, tensor_shape.TensorShape):
    #             input_spec_shape = shape.as_list()
    #         else:
    #             input_spec_shape = list(shape)
    #         batch_index, time_step_index = (1, 0) if self.time_major else (0, 1)
    #         if not self.stateful:
    #             input_spec_shape[batch_index] = None
    #         input_spec_shape[time_step_index] = None
    #         return InputSpec(shape=tuple(input_spec_shape))

    #     def get_step_input_shape(shape):
    #         if isinstance(shape, tensor_shape.TensorShape):
    #             shape = tuple(shape.as_list())
    #         # remove the timestep from the input_shape
    #         return shape[1:] if self.time_major else (shape[0],) + shape[2:]

    #     # Check whether the input shape contains any nested shapes. It could be
    #     # (tensor_shape(1, 2), tensor_shape(3, 4)) or (1, 2, 3) which is from numpy
    #     # inputs.
    #     try:
    #         input_shape = tensor_shape.as_shape(input_shape)
    #     except (ValueError, TypeError):
    #     # A nested tensor input
    #         pass

    #     if not nest.is_sequence(input_shape):
    #     # This indicates the there is only one input.
    #         if self.input_spec is not None:
    #             self.input_spec[0] = get_input_spec(input_shape)
    #         else:
    #             self.input_spec = [get_input_spec(input_shape)]
    #         step_input_shape = get_step_input_shape(input_shape)
    #     else:
    #         if self.input_spec is not None:
    #             self.input_spec[0] = nest.map_structure(get_input_spec, input_shape)
    #         else:
    #             self.input_spec = generic_utils.to_list(
    #                 nest.map_structure(get_input_spec, input_shape))
    #         step_input_shape = nest.map_structure(get_step_input_shape, input_shape)

    #     # allow cell (if layer) to build before we set or validate state_spec
    #     if isinstance(self.cell, Layer):
    #         if not self.cell.built:
    #             self.cell.build(step_input_shape)

    #     # set or validate state_spec
    #     if _is_multiple_state(self.cell.state_size):
    #         state_size = list(self.cell.state_size)
    #     else:
    #         state_size = [self.cell.state_size]

    #     if self.state_spec is not None:
    #     # initial_state was passed in call, check compatibility
    #         self._validate_state_spec(state_size, self.state_spec)
    #     else:
    #         self.state_spec = [
    #             InputSpec(shape=[None] + tensor_shape.as_shape(dim).as_list())
    #             for dim in state_size
    #         ]
    #     if self.stateful:
    #         self.reset_states()
    #     self.built = True    
    # endregion
    # region pre exists properties

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout
    

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                      self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'gates_version':self.gates_version}
        base_config = super(ConvLSTM2D_custom, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    # endregion 

    def get_initial_state(self, inputs):
        
        #region Adapting for two cell state LSTMs
        initial_state = K.zeros_like(inputs)
        # (samples, rows, cols, filters)
        initial_state = K.sum(initial_state, axis=1)

        shape_h_state = list(self.cell.kernel_shape)
        shape_h_state[-1] = self.cell.filters

        shape_c_state = list(self.cell.kernel_shape)
        shape_c_state[-1] = self.cell.filters*2
        
        initial_hidden_state = self.cell.input_conv(initial_state,
                                            array_ops.zeros(tuple(shape_h_state)),
                                            padding=self.cell.padding)
        
        initial_carry_state = self.cell.input_conv( initial_state,
                                            array_ops.zeros(tuple(shape_c_state)),
                                            padding=self.cell.padding)

        if hasattr(self.cell.state_size, '__len__'):
            return [initial_hidden_state, initial_carry_state ]
        else:
            return [initial_hidden_state]
        #endregion

class ConvLSTM2DCell_custom(DropoutRNNCellMixin, Layer):
    """
        Cell class for the ConvLSTM2D layer.
        Arguments:
            filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
            kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
            strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
            padding: One of `"valid"` or `"same"` (case-insensitive).
            data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
            dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
            activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
            recurrent_activation: Activation function to use
            for the recurrent step.
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            bias_initializer: Initializer for the bias vector.
            unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Use in combination with `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.]
            (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
            recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
            recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
            dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
            recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        Call arguments:
            inputs: A 4D tensor.
            states:  List of state tensors corresponding to the previous timestep.
            training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.
    """

    def __init__(self,
               filters,
                kernel_size,
                gates_version,
                # attn_f,
                # attn_b,
                strides=(1, 1),
                padding='valid',
                data_format=None,
                dilation_rate=(1, 1),
                activation='tanh',
                recurrent_activation='hard_sigmoid',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                unit_forget_bias=True,
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.,
                recurrent_dropout=0.,
                **kwargs):
        super(ConvLSTM2DCell_custom, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters, int(2*self.filters) )

        self.gates_version=gates_version

    def build(self, input_shape):

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                            'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        
        
        # if self.gates_version==1:
        kernel_shape = self.kernel_size + (input_dim, self.filters * 4) #Changed here
        # elif self.gates_version ==2:
        #     kernel_shape = self.kernel_size + (input_dim//2, self.filters * 8)


        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4) #NOT Changed Here

        self.kernel = self.add_weight(shape=kernel_shape,
                                    initializer=self.kernel_initializer,
                                    name='kernel',
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.filters*2,), *args, **kwargs),
                        initializers.Ones()((self.filters*2,), *args, **kwargs),
                        self.bias_initializer((self.filters * 4,), *args, **kwargs),
                    ]) #changed here
            else:
                bias_initializer = self.bias_initializer

            self.bias = self.add_weight(
                shape=(self.filters * 8,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)

        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        #inputs #shape (bs, h, w, c)

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        if self.gates_version==2:
            #TODO remove the first if statemeent below
            if tf.shape(c_tm1)[-1]  == self.filters:
                c_tm1_1 = c_tm1
                c_tm1_2 = c_tm1
            else:
                c_tm1_1 = c_tm1[:, :, :, :self.filters]
                c_tm1_2 = c_tm1[:, :, :, self.filters:]

        #so now inputs will be 


        inputs1, inputs2 = tf.split( inputs, 2, axis=-1)
        # dropout matrices for input units
        dp_mask1 = self.get_dropout_mask_for_cell(inputs1, training, count=4)
        dp_mask2 = self.get_dropout_mask_for_cell(inputs2, training, count=4)
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)


        if 0 < self.dropout < 1.:
            inputs1_i = inputs1 * dp_mask1[0]
            inputs1_f = inputs1 * dp_mask1[1]
            inputs1_c = inputs1 * dp_mask1[2]
            inputs1_o = inputs1 * dp_mask1[3]

            inputs2_i = inputs2 * dp_mask2[0]
            inputs2_f = inputs2 * dp_mask2[1]
            inputs2_c = inputs2 * dp_mask2[2]
            inputs2_o = inputs2 * dp_mask2[3]
        else:
            inputs1_i = inputs1 
            inputs1_f = inputs1 
            inputs1_c = inputs1 
            inputs1_o = inputs1 

            inputs2_i = inputs2 
            inputs2_f = inputs2 
            inputs2_c = inputs2 
            inputs2_o = inputs2 

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        
        if self.gates_version==2:
            _shape = self.kernel.shape.as_list()

            if _shape[3] == self.filters * 4:  
                self.kernel = tf.reshape(self.kernel,  _shape[:2]+[_shape[2]//2]+[-1] )
            elif _shape[3] == self.filters * 8: #fix this line later
                pass
            else:
                pass
                #raise Exception("Dev_akanni: This case not handled")

        (kernel1_i, kernel2_i,
        kernel1_f, kernel2_f,
        kernel1_c, kernel2_c,
        kernel1_o, kernel2_o) = array_ops.split(self.kernel, 8, axis=3)

        (recurrent_kernel_i,
        recurrent_kernel_f,
        recurrent_kernel_c,
        recurrent_kernel_o) = array_ops.split(self.recurrent_kernel, 4, axis=3)

        if self.use_bias:
            (bias1_i, bias2_i,
            bias1_f, bias2_f, 
            bias1_c, bias2_c,
            bias1_o, bias2_o) = array_ops.split(self.bias, 8)
        else:
            (bias1_i, bias2_i,
            bias1_f, bias2_f, 
            bias1_c, bias2_c,
            bias1_o, bias2_o) = None, None, None, None, None, None, None, None

        x1_i = self.input_conv(inputs1_i, kernel1_i, bias1_i, padding=self.padding)
        x1_f = self.input_conv(inputs1_f, kernel1_f, bias1_f, padding=self.padding)
        x1_c = self.input_conv(inputs1_c, kernel1_c, bias1_c, padding=self.padding)
        x1_o = self.input_conv(inputs1_o, kernel1_o, bias1_o, padding=self.padding)

        x2_i = self.input_conv(inputs2_i, kernel2_i, bias2_i, padding=self.padding)
        x2_f = self.input_conv(inputs2_f, kernel2_f, bias2_f, padding=self.padding)
        x2_c = self.input_conv(inputs2_c, kernel2_c, bias2_c, padding=self.padding)
        x2_o = self.input_conv(inputs2_o, kernel2_o, bias2_o, padding=self.padding)

        h_i = self.recurrent_conv(h_tm1_i, recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f, recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c, recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o, recurrent_kernel_o)

        if(self.gates_version==1):
            i = self.recurrent_activation(x1_i + x2_i + h_i)
            f = self.recurrent_activation(x1_f + x2_f + h_f)
            c = f * c_tm1 + i * self.activation(x1_c + x2_c + h_c)
            o = self.recurrent_activation(x1_o + x2_o + h_o)
            h = o * self.activation(c)
        
        elif(self.gates_version==2):
            i_1 = self.recurrent_activation(x1_i + h_i)
            i_2 = self.recurrent_activation(x2_i + h_i)

            f_1 = self.recurrent_activation(x1_f + h_f)
            f_2 = self.recurrent_activation(x2_f + h_f)
            
            c_t1 = f_1 * c_tm1_1 + i_1 * self.activation(x1_c + h_c) 
            c_t2 = f_2 * c_tm1_2 + i_2 * self.activation(x2_c + h_c)
            
            o_1 = self.recurrent_activation(x1_o + h_o)
            o_2 = self.recurrent_activation(x2_o + h_o)

            h = ( o_1*self.activation(c_t1) + o_2 * self.activation(c_t2) )/2

        return h, [h, tf.concat( [c_t1, c_t2], axis=-1) ]

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.conv2d(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out

    def get_config(self):
        config = {'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding,
                'data_format': self.data_format,
                'dilation_rate': self.dilation_rate,
                'activation': activations.serialize(self.activation),
                'recurrent_activation': activations.serialize(
                    self.recurrent_activation),
                'use_bias': self.use_bias,
                'kernel_initializer': initializers.serialize(
                    self.kernel_initializer),
                'recurrent_initializer': initializers.serialize(
                    self.recurrent_initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'unit_forget_bias': self.unit_forget_bias,
                'kernel_regularizer': regularizers.serialize(
                    self.kernel_regularizer),
                'recurrent_regularizer': regularizers.serialize(
                    self.recurrent_regularizer),
                'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                'kernel_constraint': constraints.serialize(
                    self.kernel_constraint),
                'recurrent_constraint': constraints.serialize(
                    self.recurrent_constraint),
                'bias_constraint': constraints.serialize(self.bias_constraint),
                'dropout': self.dropout,
                'recurrent_dropout': self.recurrent_dropout,
                'gates_version':self.gates_version
                # 'num_of_splits':self.num_of_splits,
                # 'attn_params':self.attn_params
                 }
        base_config = super(ConvLSTM2DCell_custom, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ConvLSTM2D_attn(ConvRNN2D):
    """
        CUSTOM Convolutional LSTM.

        My key change is that I will ensure attention on the inputs
        Init Arguments Added:

        Call Arguments Added:

        It is similar to an LSTM layer, but the input transformations
        and recurrent transformations are both convolutional.
        Arguments:
            filters: Integer, the dimensionality of the output space
                (i.e. the number of output filters in the convolution).
            kernel_size: An integer or tuple/list of n integers, specifying the
                dimensions of the convolution window.
            strides: An integer or tuple/list of n integers,
                specifying the strides of the convolution.
                Specifying any stride value != 1 is incompatible with specifying
                any `dilation_rate` value != 1.
            padding: One of `"valid"` or `"same"` (case-insensitive).
            data_format: A string,
                one of `channels_last` (default) or `channels_first`.
                The ordering of the dimensions in the inputs.
                `channels_last` corresponds to inputs with shape
                `(batch, time, ..., channels)`
                while `channels_first` corresponds to
                inputs with shape `(batch, time, channels, ...)`.
                It defaults to the `image_data_format` value found in your
                Keras config file at `~/.keras/keras.json`.
                If you never set it, then it will be "channels_last".
            dilation_rate: An integer or tuple/list of n integers, specifying
                the dilation rate to use for dilated convolution.
                Currently, specifying any `dilation_rate` value != 1 is
                incompatible with specifying any `strides` value != 1.
            activation: Activation function to use.
                By default hyperbolic tangent activation function is applied
                (`tanh(x)`).
            recurrent_activation: Activation function to use
                for the recurrent step.
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix,
                used for the linear transformation of the inputs.
            recurrent_initializer: Initializer for the `recurrent_kernel`
                weights matrix,
                used for the linear transformation of the recurrent state.
            bias_initializer: Initializer for the bias vector.
            unit_forget_bias: Boolean.
                If True, add 1 to the bias of the forget gate at initialization.
                Use in combination with `bias_initializer="zeros"`.
                This is recommended in [Jozefowicz et al.]
                (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            kernel_regularizer: Regularizer function applied to
                the `kernel` weights matrix.
            recurrent_regularizer: Regularizer function applied to
                the `recurrent_kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            activity_regularizer: Regularizer function applied to.
            kernel_constraint: Constraint function applied to
                the `kernel` weights matrix.
            recurrent_constraint: Constraint function applied to
                the `recurrent_kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
            return_sequences: Boolean. Whether to return the last output
                in the output sequence, or the full sequence.
            go_backwards: Boolean (default False).
                If True, process the input sequence backwards.
            stateful: Boolean (default False). If True, the last state
                for each sample at index i in a batch will be used as initial
                state for the sample of index i in the following batch.
            dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the inputs.
            recurrent_dropout: Float between 0 and 1.
                Fraction of the units to drop for
                the linear transformation of the recurrent state.
        Call arguments:
            inputs: A 5D tensor.
            mask: Binary tensor of shape `(samples, timesteps)` indicating whether
                a given timestep should be masked.
            training: Python boolean indicating whether the layer should behave in
                training mode or in inference mode. This argument is passed to the cell
                when calling it. This is only relevant if `dropout` or `recurrent_dropout`
                are set.
            initial_state: List of initial state tensors to be passed to the first
                call of the cell.
        Input shape:
            - If data_format='channels_first'
                    5D tensor with shape:
                    `(samples, time, channels, rows, cols)`
            - If data_format='channels_last'
                    5D tensor with shape:
                    `(samples, time, rows, cols, channels)`
        Output shape:
            - If `return_sequences`
                    - If data_format='channels_first'
                        5D tensor with shape:
                        `(samples, time, filters, output_row, output_col)`
                    - If data_format='channels_last'
                        5D tensor with shape:
                        `(samples, time, output_row, output_col, filters)`
            - Else
                - If data_format ='channels_first'
                        4D tensor with shape:
                        `(samples, filters, output_row, output_col)`
                - If data_format='channels_last'
                        4D tensor with shape:
                        `(samples, output_row, output_col, filters)`
                where `o_row` and `o_col` depend on the shape of the filter and
                the padding
        Raises:
            ValueError: in case of invalid constructor arguments.
        References:
            - [Convolutional LSTM Network: A Machine Learning Approach for
            Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
            The current implementation does not include the feedback loop on the
            cells output.
    """

    def __init__(
                self,
                 filters,
                 kernel_size,
                 attn_params,
                 attn_downscaling_params,
                 attn_factor_reduc,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 trainable=True,
                 **kwargs):

        #Amending Initialisation -> since the init is called after the sub layer MultiHead2DAtt is made
        self._trainable = trainable
        self.attn_params= attn_params
        self.attn_downscaling_params = attn_downscaling_params
        self.attn_factor_reduc = attn_factor_reduc

        self.Attention2D = MultiHead2DAttention( layer_params = attn_params, attention_scaling_params=attn_downscaling_params ,trainable=self.trainable )
        
        
        cell = ConvLSTM2DCell_attn(filters=filters,
                                     kernel_size=kernel_size,
                                     attn = self.Attention2D,
                                     attn_factor_reduc = self.attn_factor_reduc,
                                     strides=strides,
                                     padding=padding,
                                     data_format=data_format,
                                     dilation_rate=dilation_rate,
                                     activation=activation,
                                     recurrent_activation=recurrent_activation,
                                     use_bias=use_bias,
                                     kernel_initializer=kernel_initializer,
                                     recurrent_initializer=recurrent_initializer,
                                     bias_initializer=bias_initializer,
                                     unit_forget_bias=unit_forget_bias,
                                     kernel_regularizer=kernel_regularizer,
                                     recurrent_regularizer=recurrent_regularizer,
                                     bias_regularizer=bias_regularizer,
                                     kernel_constraint=kernel_constraint,
                                     recurrent_constraint=recurrent_constraint,
                                     bias_constraint=bias_constraint,
                                     dropout=dropout,
                                     recurrent_dropout=recurrent_dropout,
                                     dtype=kwargs.get('dtype'))

        super(ConvLSTM2D_attn, self).__init__(cell,
                                                return_sequences=return_sequences,
                                                go_backwards=go_backwards,
                                                stateful=stateful,
                                                **kwargs)
        
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        #self._maybe_reset_cell_dropout_mask(self.cell)
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)
        
        # temporary shape adjustment to ensure each time chunk is passed to a cell (cells do not take in a time dimension, so move time dimension to channel dimension)
        inputs = attn_shape_adjust(inputs, self.attn_factor_reduc ,reverse=False)


        return super(ConvLSTM2D_attn, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state)
    # region pre exists properties

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout
    

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'attn_params':self.attn_params,
                  'attn_downscaling_params':self.attn_downscaling_params,
                  'attn_factor_reduc':self.attn_factor_reduc,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                      self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvLSTM2D_attn, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))
    

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    # endregion 

    def get_initial_state(self, inputs):
        # inputs (samples, expanded_timesteps, rows, cols, filters)
            # The expanded_timesteps relates to the fact the input has the same spatial time dimension as the lower heirachy

            #Note: now inputs will have an extra last dimension, which represents the stacking of all the input vectors
        #region Adapting input_shape for attention
        shape_pre_attention = K.zeros_like(inputs)
        shape_post_attention = shape_pre_attention[:, ::self.attn_factor_reduc, :, :, :]
        inputs = shape_post_attention
        #endregion

        #region Adapting for two cell state LSTMs
        initial_state = K.zeros_like(inputs)
        # (samples, rows, cols, filters)
        initial_state = K.sum(initial_state, axis=1)

        shape_h_state = list(self.cell.kernel_shape)
        shape_h_state[-1] = self.cell.filters

        # shape_c_state = list(self.cell.kernel_shape)
        # shape_c_state[-1] = self.cell.filters
        
        initial_hidden_state = self.cell.input_conv(initial_state,
                                            array_ops.zeros(tuple(shape_h_state)),
                                            padding=self.cell.padding)
        
        # initial_carry_state = self.cell.input_conv( initial_state,
        #                                     array_ops.zeros(tuple(shape_c_state)),
        #                                     padding=self.cell.padding)

        if hasattr(self.cell.state_size, '__len__'):
            return [initial_hidden_state, initial_hidden_state ]
        else:
            return [initial_hidden_state]
        #endregion

class ConvLSTM2DCell_attn(DropoutRNNCellMixin, Layer):
    """
        Cell class for the ConvLSTM2D layer.
        Arguments:
            filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
            kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
            strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
            padding: One of `"valid"` or `"same"` (case-insensitive).
            data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
            dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
            activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
            recurrent_activation: Activation function to use
            for the recurrent step.
            use_bias: Boolean, whether the layer uses a bias vector.
            kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            bias_initializer: Initializer for the bias vector.
            unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Use in combination with `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.]
            (http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
            recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
            bias_regularizer: Regularizer function applied to the bias vector.
            kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
            recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix.
            bias_constraint: Constraint function applied to the bias vector.
            dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
            recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        Call arguments:
            inputs: A 4D tensor.
            states:  List of state tensors corresponding to the previous timestep.
            training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.
    """

    def __init__(
            self,
                filters,
                kernel_size,
                attn,
                attn_factor_reduc,
                strides=(1, 1),
                padding='valid',
                data_format=None,
                dilation_rate=(1, 1),
                activation='tanh',
                recurrent_activation='hard_sigmoid',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                unit_forget_bias=True,
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.,
                recurrent_dropout=0.,
                **kwargs):
        super(ConvLSTM2DCell_attn, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
                                                        'dilation_rate')
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = (self.filters,self.filters)# int(2*self.filters) )

        #self.gates_version=gates_version
        #region NEW - Attn Related params
        self.attn = attn
        self.attn_factor_reduc = attn_factor_reduc 

        #endregion

    def build(self, input_shape):

        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                            'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        
        kernel_shape = self.kernel_size + (input_dim, self.filters * 4)

        self.kernel_shape = kernel_shape
        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4) 

        self.kernel = self.add_weight(shape=kernel_shape,
                                    initializer=self.kernel_initializer,
                                    name='kernel',
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)

        self.recurrent_kernel = self.add_weight(
            shape=recurrent_kernel_shape,
            initializer=self.recurrent_initializer,
            name='recurrent_kernel',
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.filters*1,), *args, **kwargs),
                        initializers.Ones()((self.filters*1,), *args, **kwargs),
                        self.bias_initializer((self.filters * 2,), *args, **kwargs),
                    ]) #changed here
            else:
                bias_initializer = self.bias_initializer

            self.bias = self.add_weight(
                shape=(self.filters * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)

        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        #inputs #shape (bs, h, w, c*self.attn_factor_reduc)


        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        #region new: attn part
        inputs = attn_shape_adjust( inputs, self.attn_factor_reduc, reverse=True ) #shape (bs, self.attn_factor_reduc ,h, w, c )

        q = tf.expand_dims( c_tm1, axis=1)
        k = inputs
        v = inputs
        
        attn_avg_inp_hid_state = self.attn( inputs=q,
                                            inputs_k=k,
                                            inputs_v=v ) #(bs, 1, h, w, f)
        
        # endregion

        inputs = tf.squeeze( attn_avg_inp_hid_state)
        # dropout matrices for input units
        dp_mask1 = self.get_dropout_mask_for_cell(inputs, training, count=4)
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)


        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask1[0]
            inputs_f = inputs * dp_mask1[1]
            inputs_c = inputs * dp_mask1[2]
            inputs_o = inputs * dp_mask1[3]

        else:
            inputs_i = inputs 
            inputs_f = inputs 
            inputs_c = inputs 
            inputs_o = inputs

        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        
        (kernel_i,
        kernel_f, 
        kernel_c, 
        kernel_o) = array_ops.split(self.kernel, 4, axis=3)

        (recurrent_kernel_i,
        recurrent_kernel_f,
        recurrent_kernel_c,
        recurrent_kernel_o) = array_ops.split(self.recurrent_kernel, 4, axis=3)

        if self.use_bias:
            (bias_i, 
            bias_f,  
            bias_c, 
            bias_o) = array_ops.split(self.bias, 4)
        else:
            (bias_i,
            bias_f,
            bias_c,
            bias_o) = None, None, None, None

        x_i = self.input_conv(inputs_i, kernel_i, bias_i, padding=self.padding)
        x_f = self.input_conv(inputs_f, kernel_f, bias_f, padding=self.padding)
        x_c = self.input_conv(inputs_c, kernel_c, bias_c, padding=self.padding)
        x_o = self.input_conv(inputs_o, kernel_o, bias_o, padding=self.padding)

        h_i = self.recurrent_conv(h_tm1_i, recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f, recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c, recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o, recurrent_kernel_o)

        i = self.recurrent_activation(x_i + h_i)
        f = self.recurrent_activation(x_f + h_f)
        c = f * c_tm1 + i * self.activation(x_c + h_c)
        o = self.recurrent_activation(x_o + h_o)
        h = o * self.activation(c)
        
        return h, [h, c ] 

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.conv2d(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                data_format=self.data_format)
        return conv_out

    def recurrent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out

    def get_config(self):
        config = {'filters': self.filters,
                'kernel_size': self.kernel_size,
                'strides': self.strides,
                'padding': self.padding,
                'data_format': self.data_format,
                'dilation_rate': self.dilation_rate,
                'activation': activations.serialize(self.activation),
                'recurrent_activation': activations.serialize(
                    self.recurrent_activation),
                'use_bias': self.use_bias,
                'kernel_initializer': initializers.serialize(
                    self.kernel_initializer),
                'recurrent_initializer': initializers.serialize(
                    self.recurrent_initializer),
                'bias_initializer': initializers.serialize(self.bias_initializer),
                'unit_forget_bias': self.unit_forget_bias,
                'kernel_regularizer': regularizers.serialize(
                    self.kernel_regularizer),
                'recurrent_regularizer': regularizers.serialize(
                    self.recurrent_regularizer),
                'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                'kernel_constraint': constraints.serialize(
                    self.kernel_constraint),
                'recurrent_constraint': constraints.serialize(
                    self.recurrent_constraint),
                'bias_constraint': constraints.serialize(self.bias_constraint),
                'dropout': self.dropout,
                'recurrent_dropout': self.recurrent_dropout,
                
                'attn':self.attn,
                'attn_factor_reduc':self.attn_factor_reduc

                 }
        base_config = super(ConvLSTM2DCell_attn, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MultiHead2DAttention(Layer):
    def __init__(self, layer_params, attention_scaling_params,trainable):

        """
            TODO: prior to the attention possibly add something like squeeze and excitation to reweight the feature maps. But only in the first layer since taking in the original feature maps, as it shouldnt be needed after

            Either use 2D attention or try flattening nromal tensors to vectors so normal attention can be used
            Flattening used in https://arxiv.org/pdf/1904.09925.pdf, so will use there flattening method
        """
        

        self.trainable = trainable
        super( MultiHead2DAttention, self ).__init__()
        
        self.kq_downscale_kernelshape = attention_scaling_params['kq_downscale_kernelshape']
        self.kq_downscale_stride = attention_scaling_params['kq_downscale_stride']
        self.vector_k_downscale_factor = attention_scaling_params['vector_k_downscale_factor']
        self.vector_v_downscale_factor = attention_scaling_params['vector_v_downscale_factor']

        self.layer_params = layer_params

        self.ln1 = tf.keras.layers.LayerNormalization(axis=-1 , epsilon=1e-4 , trainable=trainable )

    def attn_key_depth_init(self, queries, inputs_v ):
        key_depth = tf.math.reduce_prod( queries.shape[-3:] )
        value_depth = tf.math.reduce_prod( inputs_v.shape[-3:] )

        tkd = tf.cast( (( key_depth / self.vector_k_downscale_factor ) // self.layer_params['num_heads'] ) * self.layer_params['num_heads'], dtype=tf.int32 )
        tvd = tf.cast( ( ( value_depth / self.vector_v_downscale_factor ) // self.layer_params['num_heads'] ) * self.layer_params['num_heads'], dtype=tf.int32 )
        od = tvd
        return  tkd, tvd, od

    def call(self, inputs, inputs_k, inputs_v):
        """
            :param inputs: The query 
            Note: In the attention layer, keys and values are tShe same
        """
      
        # region size reduction
        output_shape = inputs_v.shape.as_list()
        output_shape[1] = 1
        queries = tf.nn.avg_pool3d( inputs, strides=self.kq_downscale_stride,
                                ksize=self.kq_downscale_kernelshape, padding="SAME")
        keys = tf.nn.avg_pool3d( inputs_k, strides=self.kq_downscale_stride,
                                ksize=self.kq_downscale_kernelshape, padding="SAME")
        # endregion 

        # #version-1
            # if( self.layer_params['total_key_depth'] == 0) :
            #     key_depth = tf.math.reduce_prod( queries.shape[-3:] )
            #     value_depth = tf.math.reduce_prod( inputs_v.shape[-3:] )
            #     self.layer_params['total_key_depth'] = tf.cast( (( key_depth / self.vector_k_downscale_factor ) // self.layer_params['num_heads'] ) * self.layer_params['num_heads'], dtype=tf.int32 )
            #         #Key depth is used for key and queries in attention func
            #     self.layer_params['total_value_depth'] = tf.cast( ( ( value_depth / self.vector_v_downscale_factor ) // self.layer_params['num_heads'] ) * self.layer_params['num_heads'], dtype=tf.int32 )
            #     self.layer_params['output_depth'] = value_depth
        
        # version 2
            # self.layer_params['total_key_depth'], self.layer_params['total_value_depth'],self.layer_params['output_depth']   = tf.cond( tf.equal(self.layer_params['total_key_depth'],0), 
            #                 lambda: self.attn_key_depth_init(queries, inputs_v), 
            #                 lambda: (self.layer_params['total_key_depth'], self.layer_params['total_value_depth'], self.layer_params['output_depth'] )  )
        
        # version 3
        queries_flat = tf.reshape(queries, queries.shape.as_list()[:2]  + [-1] ) #( batch_size, seq_len, height*width*filters_in)
        keys_flat = tf.reshape( keys, keys.shape.as_list()[:2] +[-1] )
        values_flat = tf.reshape(inputs_v, inputs_v.shape.as_list()[:2] + [-1] )

        x = multihead_attention_custom(
            query_antecedent = queries_flat ,
            memory_antecedent = keys_flat,
            value_antecedent = values_flat,
            trainable=self.trainable,
            **self.layer_params   ) #( batch_size, seq_len, height*width*filters_in

        x = self.ln1(x) #( batch_size, seq_len, height*width*filters_in
        
        x = tf.reshape( x ,  output_shape ) #( batch_size, seq_len, height, width, filters_in)

        return x

def attn_shape_adjust(inputs, attn_factor_reduc ,reverse=False):

    """ 
        This is used to ensure that the number of dimensions in the time dimension is equal to the number of desired time dimensions.
            This is ideal when we need chunks of time to be passed to each RNN Cell for the RNN cell to perform attn to get one input from the time chunk
        if(reverse=False):
            :param tnsr inputs: (bs, tss, h, w, c)

            return outputs : (bs, tss/seq_len_factor_reduc, h, w, c*seq_len_factor_reduc )

        if(reverse=True):
            :param tnsr inputs: (bs, 1, h, w, c ) 

            return outputs : (bs, seq_len_factor_reduc, h, w, c//seq_len_factor_reduc)
    """
    
    #TODO: change _inputs to inputs to save memory
    if reverse==False:
        shape = inputs.shape.as_list()
        shape[1] = shape[1]//attn_factor_reduc
        shape[4] = shape[4]*attn_factor_reduc
        _inputs = tf.reshape(inputs, shape )
    else:
        shape = tf.expand_dims(inputs, axis=1).shape.as_list()
        shape[1] = shape[1]*attn_factor_reduc
        shape[4] = shape[4]//attn_factor_reduc
        _inputs = tf.reshape(inputs, shape )
    

    return _inputs

def multihead_attention_custom(query_antecedent,
                               memory_antecedent,
                                value_antecedent,
                                bias,
                                total_key_depth,
                                total_value_depth,
                                output_depth,
                                num_heads,
                                dropout_rate,
                                transform_value_antecedent=True,
                                transform_output=True,
                                attention_type="dot_product",
                                max_relative_position=None,
                                heads_share_relative_embedding=False,
                                add_relative_to_values=False,
                                image_shapes=None,
                                block_length=128,
                                block_width=128,
                                q_filter_width=1,
                                kv_filter_width=1,
                                q_padding="VALID",
                                kv_padding="VALID",
                                cache=None,
                                gap_size=0,
                                num_memory_blocks=2,
                                name="multihead_attention",
                                save_weights_to=None,      #TODO: Change for visualization
                                make_image_summary=False, #NOTE: Change layer for visualization
                                dropout_broadcast_dims=None, 
                                vars_3d=False,
                                layer_collection=None,
                                recurrent_memory=None,
                                chunk_number=None,
                                hard_attention_k=0,
                                gumbel_noise_weight=0.0,
                                max_area_width=1,
                                max_area_height=1,
                                memory_height=1,
                                area_key_mode="mean",
                                area_value_mode="sum",
                                training=True,
                                **kwargs):
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
                attention_type: a string, either "dot_product", "dot_product_relative",
                            "local_mask_right", "local_unmasked", "masked_dilated_1d",
                            "unmasked_dilated_1d", graph, or any attention function
                                with the signature (query, key, value, **kwargs)
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
    # replaced 
    # if tf.math.floormod(total_key_depth ,num_heads) != 0:
    #     print(total_key_depth)
    #     print("\n")
    #     print(num_heads)
    #     raise ValueError("Key depth (%d) must be divisible by the number of "
    #                     "attention heads (%d)." % (total_key_depth, num_heads))
    # if tf.math.floormod(total_value_depth,num_heads) != 0:
    #     raise ValueError("Value depth (%d) must be divisible by the number of "
    #                     "attention heads (%d)." % (total_value_depth, num_heads))
    #"Key/Value depth  must be divisible by the number of attention heads
    assert_op1 = tf.Assert( tf.equal( tf.math.floormod(total_key_depth, num_heads), 0 ), [total_key_depth, tf.constant(num_heads)] )
    assert_op2 = tf.Assert( tf.equal( tf.math.floormod(total_value_depth, num_heads), 0 ), [total_key_depth, tf.constant(num_heads)] )

    with tf.control_dependencies([assert_op1, assert_op2]):
        vars_3d_num_heads = num_heads if vars_3d else 0

    if layer_collection is not None:
        if cache is not None:
            raise ValueError("KFAC implementation only supports cache is None.")
        if vars_3d:
            raise ValueError("KFAC implementation does not support 3d vars.")

    if recurrent_memory is not None:
        if memory_antecedent is not None:
            raise ValueError("Recurrent memory requires memory_antecedent is None.")
        if cache is not None:
            raise ValueError("Cache is not supported when using recurrent memory.")
        if vars_3d:
            raise ValueError("3d vars are not supported when using recurrent memory.")
        if layer_collection is not None:
            raise ValueError("KFAC is not supported when using recurrent memory.")
        if chunk_number is None:
            raise ValueError("chunk_number is required when using recurrent memory.")

    with tf.compat.v1.variable_scope(name, default_name="multihead_attention",
                                values=[query_antecedent, memory_antecedent]):

        if recurrent_memory is not None:
            (
                    recurrent_memory_transaction,
                    query_antecedent, memory_antecedent, bias,
            ) = recurrent_memory.pre_attention(
                    chunk_number,
                    query_antecedent, memory_antecedent, bias,
                    )

        if cache is None or memory_antecedent is None:
            q, k, v = compute_qkv_custom(query_antecedent, memory_antecedent, value_antecedent,
                                total_key_depth, total_value_depth, q_filter_width,
                                kv_filter_width, q_padding, kv_padding, transform_value_antecedent=transform_value_antecedent,
                                vars_3d_num_heads=vars_3d_num_heads,
                                layer_collection=layer_collection)
        if cache is not None:
            if attention_type not in ["dot_product", "dot_product_relative"]:
            # TODO(petershaw): Support caching when using relative position
            # representations, i.e. "dot_product_relative" attention.
                raise NotImplementedError(
                    "Caching is not guaranteed to work with attention types other than"
                    " dot_product.")
            if bias is None:
                raise ValueError("Bias required for caching. See function docstring "
                                "for details.")

            if memory_antecedent is not None:
            # Encoder-Decoder Attention Cache
                q = compute_attention_component(query_antecedent, total_key_depth,
                                            q_filter_width, q_padding, "q",
                                            vars_3d_num_heads=vars_3d_num_heads)
                k = cache["k_encdec"]
                v = cache["v_encdec"]
            else:
                k = split_heads(k, num_heads)
                v = split_heads(v, num_heads)
                decode_loop_step = kwargs.get("decode_loop_step")
                if decode_loop_step is None:
                    k = cache["k"] = tf.concat([cache["k"], k], axis=2)
                    v = cache["v"] = tf.concat([cache["v"], v], axis=2)
                else:
                    # Inplace update is required for inference on TPU.
                    # Inplace_ops only supports inplace_update on the first dimension.
                    # The performance of current implementation is better than updating
                    # the tensor by adding the result of matmul(one_hot,
                    # update_in_current_step)
                    tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
                    tmp_k = inplace_ops.alias_inplace_update(
                        tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
                    k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
                    tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
                    tmp_v = inplace_ops.alias_inplace_update(
                        tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
                    v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])

        q = split_heads(q, num_heads)
        if cache is None:
            k = split_heads(k, num_heads)
            v = split_heads(v, num_heads)

        key_depth_per_head = total_key_depth // num_heads
        if not vars_3d:
            q *= tf.cast(key_depth_per_head,dtype=tf.float32)**-0.5

        additional_returned_value = None

        if attention_type == "dot_product":
            x = dot_product_attention(q, k, v, bias, dropout_rate, image_shapes,
                            save_weights_to=save_weights_to,
                            make_image_summary=make_image_summary,
                            dropout_broadcast_dims=dropout_broadcast_dims,
                            activation_dtype=kwargs.get(
                                "activation_dtype"),
                            hard_attention_k=hard_attention_k)

        #TODO: remove chunk below
        elif attention_type == "dot_product_relative":
            x = dot_product_attention_relative(
                q,
                k,
                v,
                bias,
                max_relative_position,
                dropout_rate,
                image_shapes,
                make_image_summary=make_image_summary,
                cache=cache is not None,
                allow_memory=recurrent_memory is not None,
                hard_attention_k=hard_attention_k)
        elif attention_type == "dot_product_unmasked_relative_v2":
            x = dot_product_unmasked_self_attention_relative_v2(
                q,
                k,
                v,
                bias,
                max_relative_position,
                dropout_rate,
                image_shapes,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=dropout_broadcast_dims,
                heads_share_relative_embedding=heads_share_relative_embedding,
                add_relative_to_values=add_relative_to_values)
        elif attention_type == "dot_product_relative_v2":
            x = dot_product_self_attention_relative_v2(
                q,
                k,
                v,
                bias,
                max_relative_position,
                dropout_rate,
                image_shapes,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=dropout_broadcast_dims,
                heads_share_relative_embedding=heads_share_relative_embedding,
                add_relative_to_values=add_relative_to_values)

        x = combine_heads(x)

        # Set last dim specifically.
        x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])
        if( transform_output):
            if vars_3d:
                o_var = tf.compat.v1.get_variable(
                "o", [num_heads, total_value_depth // num_heads, output_depth])
                o_var = tf.cast(o_var, x.dtype)
                o_var = tf.reshape(o_var, [total_value_depth, output_depth])
                x = tf.tensordot(x, o_var, axes=1)
            else:
                x = t2t_dense(
                    x, output_depth, use_bias=False, name="output_transform",
                    layer_collection=layer_collection)

            if recurrent_memory is not None:
                x = recurrent_memory.post_attention(recurrent_memory_transaction, x)
            if additional_returned_value is not None:
                return x, additional_returned_value
        return x

def compute_qkv_custom(query_antecedent,
                        memory_antecedent,
                            value_antecedent,
                            total_key_depth,
                            total_value_depth,
                            q_filter_width=1,
                            kv_filter_width=1,
                            q_padding="VALID",
                            kv_padding="VALID",
                            vars_3d_num_heads=0,
                            layer_collection=None,
                                transform_value_antecedent = True):
    """Computes query, key and value.
        Args:
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels]
        total_key_depth: an integer
        total_value_depth: an integer
        q_filter_width: An integer specifying how wide you want the query to be.
        kv_filter_width: An integer specifying how wide you want the keys and values
        to be.
        q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
        kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
        vars_3d_num_heads: an optional (if we want to use 3d variables)
        layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
            KFAC optimizer. Default is None.
        Returns:
        q, k, v : [batch, length, depth] tensors
    """
    if memory_antecedent is None:
        memory_antecedent = query_antecedent
    q = compute_attention_component(
        query_antecedent,
        total_key_depth,
        q_filter_width,
        q_padding,
            "q",
        vars_3d_num_heads=vars_3d_num_heads,
        layer_collection=layer_collection)
    k = compute_attention_component(
        memory_antecedent,
        total_key_depth,
        kv_filter_width,
        kv_padding,
            "k",
        vars_3d_num_heads=vars_3d_num_heads,
        layer_collection=layer_collection)
    if(transform_value_antecedent==True):
        v = compute_attention_component(
            value_antecedent,
            total_value_depth,
            kv_filter_width,
            kv_padding,
                "v",
            vars_3d_num_heads=vars_3d_num_heads,
            layer_collection=layer_collection)
    else:
        v = value_antecedent
    return q, k, v

class DeformableConvLayer(Conv2D):
    """Only support "channel last" data format
    source: https://github.com/DHZS/tf-deformable-conv-layer/blob/master/nets/deformable_conv_layer.py
    # AUTHOR: An Jiaoyang
    # DATE: 2018-10-11
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 num_deformable_group=None,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """`kernel_size`, `strides` and `dilation_rate` must have the same value in both axis.
        :param num_deformable_group: split output channels into groups, offset shared in each group. If
        this parameter is None, then set  num_deformable_group=filters.
        """
        super().__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.kernel = None
        self.bias = None
        self.offset_layer_kernel = None
        self.offset_layer_bias = None
        if num_deformable_group is None:
            num_deformable_group = filters
        if filters % num_deformable_group != 0:
            raise ValueError('"filters" mod "num_deformable_group" must be zero')
        self.num_deformable_group = num_deformable_group

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        # kernel_shape = self.kernel_size + (input_dim, self.filters)
        # we want to use depth-wise conv
        kernel_shape = self.kernel_size + (self.filters * input_dim, 1)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)

        # create offset conv layer
        offset_num = self.kernel_size[0] * self.kernel_size[1] * self.num_deformable_group
        self.offset_layer_kernel = self.add_weight(
            name='offset_layer_kernel',
            shape=self.kernel_size + (input_dim, offset_num * 2),  # 2 means x and y axis
            initializer=tf.zeros_initializer(),
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.offset_layer_bias = self.add_weight(
            name='offset_layer_bias',
            shape=(offset_num * 2,),
            initializer=tf.zeros_initializer(),
            # initializer=tf.random_uniform_initializer(-5, 5),
            regularizer=self.bias_regularizer,
            trainable=True,
            dtype=self.dtype)
        self.built = True

    def call(self, inputs, training=None, **kwargs):
        # get offset, shape [batch_size, out_h, out_w, filter_h, * filter_w * channel_out * 2]
        offset = tf.nn.conv2d(inputs,
                              filters=self.offset_layer_kernel,
                              strides=[1, *self.strides, 1],
                              padding=self.padding.upper(),
                              dilations=[1, *self.dilation_rate, 1])
        offset += self.offset_layer_bias

        # add padding if needed
        inputs = self._pad_input(inputs)

        # some length
        batch_size = int(inputs.get_shape()[0])
        channel_in = int(inputs.get_shape()[-1])
        in_h, in_w = [int(i) for i in inputs.get_shape()[1: 3]]  # input feature map size
        out_h, out_w = [int(i) for i in offset.get_shape()[1: 3]]  # output feature map size
        filter_h, filter_w = self.kernel_size

        # get x, y axis offset
        offset = tf.reshape(offset, [batch_size, out_h, out_w, -1, 2])
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]

        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])
        y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
        y, x = [tf.tile(i, [batch_size, 1, 1, 1, self.num_deformable_group]) for i in [y, x]]
        y, x = [tf.reshape(i, [*i.shape[0: 3], -1]) for i in [y, x]]
        y, x = [tf.cast(i,tf.float32)  for i in [y, x]]

        # add offset
        y, x = y + y_off, x + x_off
        y = tf.clip_by_value(y, 0, in_h - 1)
        x = tf.clip_by_value(x, 0, in_w - 1)

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.cast(tf.floor(i),tf.float32 ) for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1
        # clip
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]

        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [DeformableConvLayer._get_pixel_values_at_point(inputs, i) for i in indices]

        # cast to float
        x0, x1, y0, y1 = [tf.cast(i,tf.float32) for i in [x0, x1, y0, y1]]
        # weights
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)
        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])

        # reshape the "big" feature map
        pixels = tf.reshape(pixels, [batch_size, out_h, out_w, filter_h, filter_w, self.num_deformable_group, channel_in])
        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, self.num_deformable_group, channel_in])

        # copy channels to same group
        feat_in_group = self.filters // self.num_deformable_group
        pixels = tf.tile(pixels, [1, 1, 1, 1, feat_in_group])
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, -1])

        # depth-wise conv
        out = tf.nn.depthwise_conv2d(pixels, self.kernel, [1, filter_h, filter_w, 1], 'VALID')
        # add the output feature maps in the same group
        out = tf.reshape(out, [batch_size, out_h, out_w, self.filters, channel_in])
        out = tf.reduce_sum(out, axis=-1)
        if self.use_bias:
            out += self.bias
        return self.activation(out)

    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.
        :param inputs:
        :return: padded input feature map
        """
        # When padding is 'same', we should pad the feature map.
        # if padding == 'same', output size should be `ceil(input / stride)`
        if self.padding == 'same':
            in_shape = inputs.get_shape().as_list()[1: 3]
            padding_list = []
            for i in range(2):
                filter_size = self.kernel_size[i]
                dilation = self.dilation_rate[i]
                dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
                same_output = (in_shape[i] + self.strides[i] - 1) // self.strides[i]
                valid_output = (in_shape[i] - dilated_filter_size + self.strides[i]) // self.strides[i]
                if same_output == valid_output:
                    padding_list += [0, 0]
                else:
                    p = dilated_filter_size - 1
                    p_0 = p // 2
                    padding_list += [p_0, p - p_0]
            if sum(padding_list) != 0:
                padding = [[0, 0],
                           [padding_list[0], padding_list[1]],  # top, bottom padding
                           [padding_list[2], padding_list[3]],  # left, right padding
                           [0, 0]]
                inputs = tf.pad(inputs, padding)
        return inputs

    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map
        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """
        feat_h, feat_w = [int(i) for i in feature_map_size[0: 2]]

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.compat.v1.extract_image_patches(i,
                                               [1, *self.kernel_size, 1],
                                               [1, *self.strides, 1],
                                               [1, *self.dilation_rate, 1],
                                               'VALID')
                for i in [x, y]]  # shape [1, out_h, out_w, filter_h * filter_w]
        return y, x

    @staticmethod
    def _get_pixel_values_at_point(inputs, indices):
        """get pixel values
        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        batch, h, w, n = y.get_shape().as_list()[0: 4]

        batch_idx = tf.reshape(tf.range(0.0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, tf.cast(pixel_idx, tf.int32 ) )