import tensorflow as tf
import numpy as np

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import _standardize_args
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D
from tensorflow.python.keras.layers.recurrent import _is_multiple_state

from layers import MultiHead2DAttention
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
                #  attn_params,
                #  attn_factor_reduc,
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
        # self.Attention2D_f = MultiHead2DAttention( trainable=self.trainable, layer_params = attn_params )
        # self.Attention2D_b = MultiHead2DAttention( trainable= self.trainable, layer_params = attn_params  )
        self.attn_factor_reduc = attn_factor_reduc

        cell = ConvLSTM2DCell_custom(filters=filters,
                                     kernel_size=kernel_size,
                                     gates_version=gates_version,
                                     attn_f = self.Attention2D_f,
                                     attn_b = self.Attention2D_b,
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
        #region NEW - Attn Related params
        self.attn_f = attn_f 
        self.attn_b = attn_b

        # self.num_of_splits = num_of_splits
        # self.seq_len_factor_reduction = seq_len_factor_reduction 
        # self.shapes_set = 0
        # self.h, self.w = (100,140)
        # self.bs = 0
        # self.c = self.filters*2
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

            if _shape[2] == self.filters * 4:  
                self.kernel = tf.reshape(self.kernel,  _shape[:2]+[_shape[2]//2]+[-1] )
            elif _shape[2] == self.filters * 2: #fix this line later
                pass
            else:
                raise Exception("Dev_akanni: This case not handled")

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

            h = o_1*self.activation(c_t1) + o_2 * self.activation(c_t2)

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
                 #gates_version,
                 
                 attn_params,
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
                 **kwargs):

        #self.gates_version = gates_version
        self.Attention2D = MultiHead2DAttention( trainable=self.trainable, layer_params = attn_params )
        self.attn_factor_reduc = attn_factor_reduc

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
                        self.bias_initializer((self.filters*2,), *args, **kwargs),
                        initializers.Ones()((self.filters*2,), *args, **kwargs),
                        self.bias_initializer((self.filters * 4,), *args, **kwargs),
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
        inputs = attn_shape_adjust( inputs, self.attn_factor_re) #shape (bs, self.attn_factor_reduc ,h, w, c )

        query = tf.expand_dims( c_tm1, axis=1)
        
        attn_avg_inp_hid_state = self.attn( query, inputs ) #(bs, 1, h, w, f)
        
        forward_h, backward_h = self.convLSTM( attn_avg_inp_hid_state, training=self.trainable ) #shape( bs, seq_len, h, w, 2*c) #c2 specified in ConvLSTM creation

        inputs = tf.concat( [forward_h, backward_h], axis=-1) #shape( bs, 1, h, w, 2*c)
        inputs = tf.squeeze(inputs)
        # endregion

        inputs1 = inputs
        # dropout matrices for input units
        dp_mask1 = self.get_dropout_mask_for_cell(inputs1, training, count=4)
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)


        if 0 < self.dropout < 1.:
            inputs1_i = inputs1 * dp_mask1[0]
            inputs1_f = inputs1 * dp_mask1[1]
            inputs1_c = inputs1 * dp_mask1[2]
            inputs1_o = inputs1 * dp_mask1[3]

        else:
            inputs1_i = inputs1 
            inputs1_f = inputs1 
            inputs1_c = inputs1 
            inputs1_o = inputs1

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
        
        (kernel1_i,
        kernel1_f, 
        kernel1_c, 
        kernel1_o) = array_ops.split(self.kernel, 4, axis=3)

        (recurrent_kernel_i,
        recurrent_kernel_f,
        recurrent_kernel_c,
        recurrent_kernel_o) = array_ops.split(self.recurrent_kernel, 4, axis=3)

        if self.use_bias:
            (bias1_i, 
            bias1_f,  
            bias1_c, 
            bias1_o) = array_ops.split(self.bias, 4)
        else:
            (bias1_i,
            bias1_f,
            bias1_c,
            bias1_o) = None, None, None, None

        x1_i = self.input_conv(inputs1_i, kernel1_i, bias1_i, padding=self.padding)
        x1_f = self.input_conv(inputs1_f, kernel1_f, bias1_f, padding=self.padding)
        x1_c = self.input_conv(inputs1_c, kernel1_c, bias1_c, padding=self.padding)
        x1_o = self.input_conv(inputs1_o, kernel1_o, bias1_o, padding=self.padding)

        h_i = self.recurrent_conv(h_tm1_i, recurrent_kernel_i)
        h_f = self.recurrent_conv(h_tm1_f, recurrent_kernel_f)
        h_c = self.recurrent_conv(h_tm1_c, recurrent_kernel_c)
        h_o = self.recurrent_conv(h_tm1_o, recurrent_kernel_o)

        i = self.recurrent_activation(x1_i + h_i)
        f = self.recurrent_activation(x1_f + h_f)
        c = f * c_tm1 + i * self.activation(x1_c + h_c)
        o = self.recurrent_activation(x1_o + h_o)
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
        shape = tf.shape(inputs).as_list()
        _inputs = tf.reshape(inputs, shape[:1] + shape[1]//attn_factor_reduc + shape[2:4] + shape[4]*attn_factor_reduc )
    else:
        shape = tf.shape(tf.expand_dims(inputs, axis=1) ).as_list()
        _inputs = tf.reshape(inputs, shape[:1] + shape[1]*attn_factor_reduc + shape[2:4] + shape[4]//attn_factor_reduc )
    
    return _inputs
>>>>>>> d5dd1fe... Completed new heirachical attention LSTM module
