import tensorflow as tf
import numpy as np

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
from tensorflow.python.keras.layers.recurrent import _standardize_args, DropoutRNNCellMixin, RNN, _is_multiple_state
from tensorflow.python.keras.utils import conv_utils, generic_utils, tf_utils
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
#from tensorflow.python.keras.layers.convolutional_recurrent import ConvRNN2D

from tensorflow.keras.layers import Conv2D, RNN



#The calss below is adated to work with mixed precision
class ConvRNN2D(RNN):
  """Base class for convolutional-recurrent layers.

    Arguments:
      cell: A RNN cell instance. A RNN cell is a class that has:
        - a `call(input_at_t, states_at_t)` method, returning
          `(output_at_t, states_at_t_plus_1)`. The call method of the
          cell can also take the optional argument `constants`, see
          section "Note on passing external constants" below.
        - a `state_size` attribute. This can be a single integer
          (single state) in which case it is
          the number of channels of the recurrent state
          (which should be the same as the number of channels of the cell
          output). This can also be a list/tuple of integers
          (one size per state). In this case, the first entry
          (`state_size[0]`) should be the same as
          the size of the cell output.
      return_sequences: Boolean. Whether to return the last output.
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      input_shape: Use this argument to specify the shape of the
        input when this layer is the first one in a model.

    Call arguments:
      inputs: A 5D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is for use with cells that use dropout.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
      constants: List of constant tensors to be passed to the cell at each
        timestep.

    Input shape:
      5D tensor with shape:
      `(samples, timesteps, channels, rows, cols)`
      if data_format='channels_first' or 5D tensor with shape:
      `(samples, timesteps, rows, cols, channels)`
      if data_format='channels_last'.

    Output shape:
      - If `return_state`: a list of tensors. The first tensor is
        the output. The remaining tensors are the last states,
        each 4D tensor with shape:
        `(samples, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
        if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
      - If `return_sequences`: 5D tensor with shape:
        `(samples, timesteps, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 5D tensor with shape:
        `(samples, timesteps, new_rows, new_cols, filters)`
        if data_format='channels_last'.
      - Else, 4D tensor with shape:
        `(samples, filters, new_rows, new_cols)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)`
        if data_format='channels_last'.

    Masking:
      This layer supports masking for input data with a variable number
      of timesteps.

    Note on using statefulness in RNNs:
      You can set RNN layers to be 'stateful', which means that the states
      computed for the samples in one batch will be reused as initial states
      for the samples in the next batch. This assumes a one-to-one mapping
      between samples in different successive batches.
      To enable statefulness:
        - Specify `stateful=True` in the layer constructor.
        - Specify a fixed batch size for your model, by passing
          - If sequential model:
              `batch_input_shape=(...)` to the first layer in your model.
          - If functional model with 1 or more Input layers:
              `batch_shape=(...)` to all the first layers in your model.
              This is the expected shape of your inputs
              *including the batch size*.
              It should be a tuple of integers,
              e.g. `(32, 10, 100, 100, 32)`.
              Note that the number of rows and columns should be specified
              too.
        - Specify `shuffle=False` when calling fit().
      To reset the states of your model, call `.reset_states()` on either
      a specific layer, or on your entire model.

    Note on specifying the initial state of RNNs:
      You can specify the initial state of RNN layers symbolically by
      calling them with the keyword argument `initial_state`. The value of
      `initial_state` should be a tensor or list of tensors representing
      the initial state of the RNN layer.
      You can specify the initial state of RNN layers numerically by
      calling `reset_states` with the keyword argument `states`. The value of
      `states` should be a numpy array or list of numpy arrays representing
      the initial state of the RNN layer.

    Note on passing external constants to RNNs:
      You can pass "external" constants to the cell using the `constants`
      keyword argument of `RNN.__call__` (as well as `RNN.call`) method. This
      requires that the `cell.call` method accepts the same keyword argument
      `constants`. Such constants can be used to condition the cell
      transformation on additional static inputs (not changing over time),
      a.k.a. an attention mechanism.
  """

  def __init__(self,
               cell,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if unroll:
      raise TypeError('Unrolling isn\'t possible with '
                      'convolutional RNNs.')
    if isinstance(cell, (list, tuple)):
      # The StackedConvRNN2DCells isn't implemented yet.
      raise TypeError('It is not possible at the moment to'
                      'stack convolutional cells.')
    super(ConvRNN2D, self).__init__(cell,
                                    return_sequences,
                                    return_state,
                                    go_backwards,
                                    stateful,
                                    unroll,
                                    **kwargs)
    self.input_spec = [InputSpec(ndim=5)]
    self.states = None
    self._num_constants = None

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    cell = self.cell
    if cell.data_format == 'channels_first':
      rows = input_shape[3]
      cols = input_shape[4]
    elif cell.data_format == 'channels_last':
      rows = input_shape[2]
      cols = input_shape[3]
    rows = conv_utils.conv_output_length(rows,
                                         cell.kernel_size[0],
                                         padding=cell.padding,
                                         stride=cell.strides[0],
                                         dilation=cell.dilation_rate[0])
    cols = conv_utils.conv_output_length(cols,
                                         cell.kernel_size[1],
                                         padding=cell.padding,
                                         stride=cell.strides[1],
                                         dilation=cell.dilation_rate[1])

    if cell.data_format == 'channels_first':
      output_shape = input_shape[:2] + (cell.filters, rows, cols)
    elif cell.data_format == 'channels_last':
      output_shape = input_shape[:2] + (rows, cols, cell.filters)

    if not self.return_sequences:
      output_shape = output_shape[:1] + output_shape[2:]

    if self.return_state:
      output_shape = [output_shape]
      if cell.data_format == 'channels_first':
        output_shape += [(input_shape[0], cell.filters, rows, cols)
                         for _ in range(2)]
      elif cell.data_format == 'channels_last':
        output_shape += [(input_shape[0], rows, cols, cell.filters)
                         for _ in range(2)]
    return output_shape

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    # Note input_shape will be list of shapes of initial states and
    # constants if these are passed in __call__.
    if self._num_constants is not None:
      constants_shape = input_shape[-self._num_constants:]  # pylint: disable=E1130
    else:
      constants_shape = None

    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    batch_size = input_shape[0] if self.stateful else None
    self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:5])

    # allow cell (if layer) to build before we set or validate state_spec
    if isinstance(self.cell, Layer):
      step_input_shape = (input_shape[0],) + input_shape[2:]
      if constants_shape is not None:
        self.cell.build([step_input_shape] + constants_shape)
      else:
        self.cell.build(step_input_shape)

    # set or validate state_spec
    if hasattr(self.cell.state_size, '__len__'):
      state_size = list(self.cell.state_size)
    else:
      state_size = [self.cell.state_size]

    if self.state_spec is not None:
      # initial_state was passed in call, check compatibility
      if self.cell.data_format == 'channels_first':
        ch_dim = 1
      elif self.cell.data_format == 'channels_last':
        ch_dim = 3
      if [spec.shape[ch_dim] for spec in self.state_spec] != state_size:
        raise ValueError(
            'An initial_state was passed that is not compatible with '
            '`cell.state_size`. Received `state_spec`={}; '
            'However `cell.state_size` is '
            '{}'.format([spec.shape for spec in self.state_spec],
                        self.cell.state_size))
    else:
      if self.cell.data_format == 'channels_first':
        self.state_spec = [InputSpec(shape=(None, dim, None, None))
                           for dim in state_size]
      elif self.cell.data_format == 'channels_last':
        self.state_spec = [InputSpec(shape=(None, None, None, dim))
                           for dim in state_size]
    if self.stateful:
      self.reset_states()
    self.built = True

  def get_initial_state(self, inputs):
    # (samples, timesteps, rows, cols, filters)
    initial_state = K.zeros_like(inputs)
    # (samples, rows, cols, filters)
    initial_state = K.sum(initial_state, axis=1)
    shape = list(self.cell.kernel_shape)
    shape[-1] = self.cell.filters
    initial_state = self.cell.input_conv(initial_state,
                                         array_ops.zeros(tuple(shape)),
                                         padding=self.cell.padding)

    if hasattr(self.cell.state_size, '__len__'):
      return [initial_state for _ in self.cell.state_size]
    else:
      return [initial_state]

  def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
    inputs, initial_state, constants = _standardize_args(
        inputs, initial_state, constants, self._num_constants)

    if initial_state is None and constants is None:
      return super(ConvRNN2D, self).__call__(inputs, **kwargs)

    # If any of `initial_state` or `constants` are specified and are Keras
    # tensors, then add them to the inputs and temporarily modify the
    # input_spec to include them.

    additional_inputs = []
    additional_specs = []
    if initial_state is not None:
      kwargs['initial_state'] = initial_state
      additional_inputs += initial_state
      self.state_spec = []
      for state in initial_state:
        shape = K.int_shape(state)
        self.state_spec.append(InputSpec(shape=shape))

      additional_specs += self.state_spec
    if constants is not None:
      kwargs['constants'] = constants
      additional_inputs += constants
      self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                             for constant in constants]
      self._num_constants = len(constants)
      additional_specs += self.constants_spec
    # at this point additional_inputs cannot be empty
    for tensor in additional_inputs:
      if K.is_keras_tensor(tensor) != K.is_keras_tensor(additional_inputs[0]):
        raise ValueError('The initial state or constants of an RNN'
                         ' layer cannot be specified with a mix of'
                         ' Keras tensors and non-Keras tensors')

    if K.is_keras_tensor(additional_inputs[0]):
      # Compute the full input spec, including state and constants
      full_input = [inputs] + additional_inputs
      full_input_spec = self.input_spec + additional_specs
      # Perform the call with temporarily replaced input_spec
      original_input_spec = self.input_spec
      self.input_spec = full_input_spec
      output = super(ConvRNN2D, self).__call__(full_input, **kwargs)
      self.input_spec = original_input_spec
      return output
    else:
      return super(ConvRNN2D, self).__call__(inputs, **kwargs)

  def call(
        self,
            inputs,
            mask=None,
            training=None,
            initial_state=None,
            constants=None):
    # note that the .build() method of subclasses MUST define
    # self.input_spec and self.state_spec with complete input shapes.
    if isinstance(inputs, list):
      inputs = inputs[0]
    if initial_state is not None:
      pass
    elif self.stateful:
      initial_state = self.states
    else:
      initial_state = self.get_initial_state(inputs)

    if isinstance(mask, list):
      mask = mask[0]

    if len(initial_state) != len(self.states):
      raise ValueError('Layer has ' + str(len(self.states)) +
                       ' states but was passed ' +
                       str(len(initial_state)) +
                       ' initial states.')
    timesteps = K.int_shape(inputs)[1]

    kwargs = {}
    if generic_utils.has_arg(self.cell.call, 'training'):
      kwargs['training'] = training

    if constants:
      if not generic_utils.has_arg(self.cell.call, 'constants'):
        raise ValueError('RNN cell does not support constants')

      def step(inputs, states):
        constants = states[-self._num_constants:]
        states = states[:-self._num_constants]
        return self.cell.call(inputs, states, constants=constants,
                              **kwargs)
    else:
      def step(inputs, states):
        return self.cell.call(inputs, states, **kwargs)

    last_output, outputs, states = K.rnn(step,
                                         inputs,
                                         initial_state,
                                         constants=constants,
                                         go_backwards=self.go_backwards,
                                         mask=mask,
                                         input_length=timesteps)
    if self.stateful:
      updates = []
      for i in range(len(states)):
        updates.append(K.update(self.states[i], states[i]))
      self.add_update(updates)

    if self.return_sequences:
      output = outputs
    else:
      output = last_output

    if self.return_state:
      if not isinstance(states, (list, tuple)):
        states = [states]
      else:
        states = list(states)
      return [output] + states
    else:
      return output

  def reset_states(self, states=None):
    if not self.stateful:
      raise AttributeError('Layer must be stateful.')
    input_shape = self.input_spec[0].shape
    state_shape = self.compute_output_shape(input_shape)
    if self.return_state:
      state_shape = state_shape[0]
    if self.return_sequences:
      state_shape = state_shape[:1].concatenate(state_shape[2:])
    if None in state_shape:
      raise ValueError('If a RNN is stateful, it needs to know '
                       'its batch size. Specify the batch size '
                       'of your input tensors: \n'
                       '- If using a Sequential model, '
                       'specify the batch size by passing '
                       'a `batch_input_shape` '
                       'argument to your first layer.\n'
                       '- If using the functional API, specify '
                       'the time dimension by passing a '
                       '`batch_shape` argument to your Input layer.\n'
                       'The same thing goes for the number of rows and '
                       'columns.')

    # helper function
    def get_tuple_shape(nb_channels):
      result = list(state_shape)
      if self.cell.data_format == 'channels_first':
        result[1] = nb_channels
      elif self.cell.data_format == 'channels_last':
        result[3] = nb_channels
      else:
        raise KeyError
      return tuple(result)

    # initialize state if None
    if self.states[0] is None:
      if hasattr(self.cell.state_size, '__len__'):
        self.states = [K.zeros(get_tuple_shape(dim),dtype=tf.float16)
                       for dim in self.cell.state_size]
      else:
        self.states = [K.zeros(get_tuple_shape(self.cell.state_size))]
    elif states is None:
      if hasattr(self.cell.state_size, '__len__'):
        for state, dim in zip(self.states, self.cell.state_size):
          K.set_value(state, np.zeros(get_tuple_shape(dim),dtype=tf.float16))
      else:
        K.set_value(self.states[0],
                    np.zeros(get_tuple_shape(self.cell.state_size)))
    else:
      if not isinstance(states, (list, tuple)):
        states = [states]
      if len(states) != len(self.states):
        raise ValueError('Layer ' + self.name + ' expects ' +
                         str(len(self.states)) + ' states, ' +
                         'but it received ' + str(len(states)) +
                         ' state values. Input received: ' + str(states))
      for index, (value, state) in enumerate(zip(states, self.states)):
        if hasattr(self.cell.state_size, '__len__'):
          dim = self.cell.state_size[index]
        else:
          dim = self.cell.state_size
        if value.shape != get_tuple_shape(dim):
          raise ValueError('State ' + str(index) +
                           ' is incompatible with layer ' +
                           self.name + ': expected shape=' +
                           str(get_tuple_shape(dim)) +
                           ', found shape=' + str(value.shape))
        # TODO(anjalisridhar): consider batch calls to `set_value`.
        K.set_value(state, value)

  def _maybe_reset_cell_dropout_mask(cell):
      return super(ConvRNN2D,self)._maybe_reset_cell_dropout_mask(cell)

##Input layer
class ConvLSTM2D(ConvRNN2D):
    """Convolutional LSTM.

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
        cell = ConvLSTM2DCell(filters=filters,
                            kernel_size=kernel_size,
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
        super(ConvLSTM2D, self).__init__(cell,
                                        return_sequences=return_sequences,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    @tf.function
    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        return super(ConvLSTM2D, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state)
    #region
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
                'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvLSTM2D, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    #endregion

class ConvLSTM2DCell(DropoutRNNCellMixin, Layer):
    """Cell class for the ConvLSTM2D layer.

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
        super(ConvLSTM2DCell, self).__init__(**kwargs)
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
        self.state_size = (self.filters, self.filters)

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
                        self.bias_initializer((self.filters,), *args, **kwargs),
                        initializers.Ones()((self.filters,), *args, **kwargs),
                        self.bias_initializer((self.filters * 2,), *args, **kwargs),
                    ])
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

    @tf.function
    def call(self, inputs, states, training=None):
        h_tm1 = tf.cast(states[0],dtype=inputs.dtype)  # previous memory state
        c_tm1 = tf.cast( states[1], dtype=inputs.dtype)  # previous carry state

        # dropout matrices for input units
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
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

        (kernel_i, kernel_f,
            kernel_c, kernel_o) = array_ops.split(self.kernel, 4, axis=3)
        (recurrent_kernel_i,
            recurrent_kernel_f,
            recurrent_kernel_c,
            recurrent_kernel_o) = array_ops.split(self.recurrent_kernel, 4, axis=3)

        if self.use_bias:
            bias_i, bias_f, bias_c, bias_o = array_ops.split(self.bias, 4)
        else:
            bias_i, bias_f, bias_c, bias_o = None, None, None, None

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
        return h, [h, c]

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
                'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvLSTM2DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

##Decoder Layer
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

    @tf.function
    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)

        if self.stateful and (initial_state is not None):
            initial_state = self.states
        elif (initial_state is not None) :
            initial_state = self.get_initial_state(inputs)
        
        #self.states = [ tf.cast(_state, inputs.dtype ) for _state in self.states ]
        return super(ConvLSTM2D_custom, self).call(inputs,
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
        
        

        kernel_shape = self.kernel_size + (input_dim, self.filters * 4) #Changed here
        self.kernel_shape = kernel_shape
        #if self.corrected_kernel_shape == None:
        self.corrected_kernel_shape = tf.TensorShape(self.kernel_size + (input_dim//2, self.filters * 8) )

        recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4) #NOT Changed Here

        self.kernel = self.add_weight(shape=self.corrected_kernel_shape,
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

    @tf.function
    def call(self, inputs, states, training=None):
        #inputs #shape (bs, h, w, c)

        #self.kernel = tf.reshape(self.kernel, self.corrected_kernel_shape, name="This" )
        h_tm1 = tf.cast( states[0], dtype=inputs.dtype) # previous memory state
        c_tm1 = tf.cast( states[1], dtype=inputs.dtype)  # previous carry state
        
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
            #dp_mask1 = self.get_dropout_mask_for_cell(inputs1, training, count=4)
            # dp_mask2 = self.get_dropout_mask_for_cell(inputs2, training, count=4)
            # dropout matrices for recurrent units
            # rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            #     h_tm1, training, count=4)

        if 0 < self.dropout < 1. and training:
            #inputs1_i = inputs1 * dp_mask1[0]
            # inputs1_f = inputs1 * dp_mask1[1]
            # inputs1_c = inputs1 * dp_mask1[2]
            # inputs1_o = inputs1 * dp_mask1[3]

            # inputs2_i = inputs2 * dp_mask2[0]
            # inputs2_f = inputs2 * dp_mask2[1]
            # inputs2_c = inputs2 * dp_mask2[2]
            # inputs2_o = inputs2 * dp_mask2[3]

            inputs1_i = tf.nn.dropout(inputs1,self.dropout)
            inputs1_f = tf.nn.dropout(inputs1,self.dropout) 
            inputs1_c = tf.nn.dropout(inputs1,self.dropout) 
            inputs1_o = tf.nn.dropout(inputs1,self.dropout) 

            inputs2_i = tf.nn.dropout(inputs2, self.dropout) 
            inputs2_f = tf.nn.dropout(inputs2, self.dropout) 
            inputs2_c = tf.nn.dropout(inputs2, self.dropout) 
            inputs2_o = tf.nn.dropout(inputs2, self.dropout) 
        else:
            inputs1_i = inputs1 
            inputs1_f = inputs1 
            inputs1_c = inputs1 
            inputs1_o = inputs1 

            inputs2_i = inputs2 
            inputs2_f = inputs2 
            inputs2_c = inputs2 
            inputs2_o = inputs2 

        if 0 < self.recurrent_dropout < 1. and training:
            # h_tm1_i = h_tm1 * rec_dp_mask[0]
                # h_tm1_f = h_tm1 * rec_dp_mask[1]
                # h_tm1_c = h_tm1 * rec_dp_mask[2]
                # h_tm1_o = h_tm1 * rec_dp_mask[3]

            h_tm1_i = tf.nn.dropout(h_tm1,self.recurrent_dropout)
            h_tm1_f = tf.nn.dropout(h_tm1,self.recurrent_dropout)
            h_tm1_c = tf.nn.dropout(h_tm1,self.recurrent_dropout)
            h_tm1_o = tf.nn.dropout(h_tm1,self.recurrent_dropout)
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        
        #_shape = self.kernel.shape#.as_list()
 
        # if tf.equal(_shape[3], self.filters * 4):
        #    self.kernel = tf.reshape(self.kernel,  _shape[:2]+_shape[2]//2+_shape[3]*2, name="Here" )

        
        # self.kernel = tf.reshape(self.kernel, self.corrected_kernel_shape, name="This" )

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

        # if(self.gates_version==1):
        #     i = self.recurrent_activation(x1_i + x2_i + h_i)
        #     f = self.recurrent_activation(x1_f + x2_f + h_f)
        #     c = f * c_tm1 + i * self.activation(x1_c + x2_c + h_c)
        #     o = self.recurrent_activation(x1_o + x2_o + h_o)
        #     h = o * self.activation(c)
        
        #elif(self.gates_version==2
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

# Encoder Layer
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

        self.Attention2D = MultiHead2DAttention_v2( **attn_params, attention_scaling_params=attn_downscaling_params , attn_factor_reduc=attn_factor_reduc ,trainable=self.trainable )
        
        
        cell = ConvLSTM2DCell_attn(filters=filters,
                                     kernel_size=kernel_size,
                                     attn_2D = self.Attention2D,
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

    @tf.function
    def call(self, inputs, mask=None, training=None, initial_state=None):
        self._maybe_reset_cell_dropout_mask(self.cell)
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs) 
        
        # temporary shape adjustment to ensure each time chunk is passed to a cell (cells do not take in a time dimension, so move time dimension to channel dimension)
        inputs = attn_shape_adjust(inputs, self.attn_factor_reduc ,reverse=False)

        #self.states = [ tf.cast(_state, inputs.dtype ) for _state in self.states ]
        return super(ConvLSTM2D_attn, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state) #Note: or here
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
                attn_2D,
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
        self.attn_2D = attn_2D
        self.attn_factor_reduc = attn_factor_reduc 

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

    @tf.function
    def call(self, inputs, states, training=None):
        #inputs #shape (bs, h, w, c*self.attn_factor_reduc)

        h_tm1 = tf.cast( states[0], dtype=inputs.dtype)  # previous memory state
        c_tm1 = tf.cast( states[1],dtype=inputs.dtype)  # previous carry state

        #region new: attn part
        inputs = attn_shape_adjust( inputs, self.attn_factor_reduc, reverse=True ) #shape (bs, self.attn_factor_reduc ,h, w, c )
        q = tf.expand_dims( c_tm1, axis=1)
        k = inputs
        v = inputs
        
        attn_avg_inp_hid_state = self.attn_2D( inputs=q,
                                            k_antecedent=k,
                                            v_antecedent=v ) #(bs, 1, h, w, f)
        
        # endregion

        inputs = tf.squeeze( attn_avg_inp_hid_state)
        # dropout matrices for input units
        dp_mask1 = self.get_dropout_mask_for_cell(inputs, training, count=4)
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)


        if 0 < self.dropout < 1. and training:
            inputs_i = inputs * dp_mask1[0]
            inputs_f = inputs * dp_mask1[1]
            inputs_c = inputs * dp_mask1[2]
            inputs_o = inputs * dp_mask1[3]

        else:
            inputs_i = inputs 
            inputs_f = inputs 
            inputs_c = inputs 
            inputs_o = inputs

        if 0 < self.recurrent_dropout < 1. and training:
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
                
                'attn_2D':self.attn_2D,
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

        # self.vector_k_downscale_factor = attention_scaling_params['vector_k_downscale_factor']
        # self.vector_v_downscale_factor = attention_scaling_params['vector_v_downscale_factor']

        self.layer_params = layer_params

        self.ln1 = tf.keras.layers.LayerNormalization(axis=-1 , epsilon=1e-4 , trainable=trainable )

    # def attn_key_depth_init(self, queries, inputs_v ):
    #     key_depth = tf.math.reduce_prod( queries.shape[-3:] )
    #     value_depth = tf.math.reduce_prod( inputs_v.shape[-3:] )

    #     tkd = tf.cast( (( key_depth / self.vector_k_downscale_factor ) // self.layer_params['num_heads'] ) * self.layer_params['num_heads'], dtype=tf.int32 )
    #     tvd = tf.cast( ( ( value_depth / self.vector_v_downscale_factor ) // self.layer_params['num_heads'] ) * self.layer_params['num_heads'], dtype=tf.int32 )
    #     od = tvd
    #     return  tkd, tvd, od

    @tf.function
    def call(self, q_antecedent, k_antecedent, v_antecedent):
        """
            :param inputs: The query 
            Note: In the attention layer, keys and values are tShe same

            :inputs: is queries
        """
      
        # region size reduction
        output_shape = v_antecedent.shape.as_list()
        output_shape[1] = 1 # inputs.shape[1]
        # queries = tf.nn.avg_pool3d( inputs, strides=self.kq_downscale_stride,
        #                         ksize=self.kq_downscale_kernelshape, padding="SAME")
        # keys = tf.nn.avg_pool3d( inputs_k, strides=self.kq_downscale_stride,
        #                         ksize=self.kq_downscale_kernelshape, padding="SAME")

        q_antecedent = tf.cast( tf.nn.avg_pool3d( tf.cast(q_antecedent,tf.float32), strides=self.kq_downscale_stride,
                                ksize=self.kq_downscale_kernelshape, padding="SAME"), tf.float16)
                                
        k_antecedent = tf.cast(tf.nn.avg_pool3d( tf.cast(k_antecedent,tf.float32), strides=self.kq_downscale_stride,
                                ksize=self.kq_downscale_kernelshape, padding="SAME"), tf.float16)

        # endregion 

        # region reshaping for attention
        q_antecedent_flat = tf.reshape(q_antecedent, q_antecedent.shape.as_list()[:2]  + [-1] ) #( batch_size, seq_len, height*width*filters_in)
        k_antecedent_flat = tf.reshape( k_antecedent, k_antecedent.k_antecedent.as_list()[:2] +[-1] )
        k_antecedent_flat = tf.reshape(v_antecedent, v_antecedent.shape.as_list()[:2] + [-1] )
        # endregion

        # region 


        # endregion


        x = multihead_attention_custom(
            query_antecedent = q_antecedent_flat ,
            memory_antecedent = k_antecedent_flat,
            value_antecedent = k_antecedent_flat,
            trainable=self.trainable,
            **self.layer_params   ) #( batch_size, seq_len, height*width*filters_in



        #x = self.ln1(x) #( batch_size, seq_len, height*width*filters_in #NOTE: doesnt work on cpu, add tf.test.is_gpu_available() to make this layer conditional
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
        shape = inputs.shape
        # s1 = shape[1]//attn_factor_reduc
        # s4 = shape[4]*attn_factor_reduc
        _inputs = tf.reshape(inputs, shape[:1]+shape[1]//attn_factor_reduc+shape[2:4]+shape[4]*attn_factor_reduc )
    else:
        shape = tf.expand_dims(inputs, axis=1).shape
        # shape[1] = shape[1]*attn_factor_reduc
        # shape[4] = shape[4]//attn_factor_reduc
        _inputs = tf.reshape(inputs, shape[:1]+shape[1]*attn_factor_reduc+shape[2:4]+shape[4]//attn_factor_reduc )
    

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
    #region Done- copy to call
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
    #endregion

    #drop this with
    with tf.compat.v1.variable_scope(name, default_name="multihead_attention",
                                values=[query_antecedent, memory_antecedent]):
        #region add this to call
        if recurrent_memory is not None:
            (
                    recurrent_memory_transaction,
                    query_antecedent, memory_antecedent, bias,
            ) = recurrent_memory.pre_attention(
                    chunk_number,
                    query_antecedent, memory_antecedent, bias,
                    )
        #endregion

        #region DONE- add to call and create dense layers to be used
        if cache is None or memory_antecedent is None:
            q, k, v = compute_qkv_custom(query_antecedent, memory_antecedent, value_antecedent,
                                total_key_depth, total_value_depth, q_filter_width,
                                kv_filter_width, q_padding, kv_padding, transform_value_antecedent=transform_value_antecedent,
                                vars_3d_num_heads=vars_3d_num_heads,
                                layer_collection=layer_collection)
        #endregion

        #region Done remove this section -> just put "if cache is not None: raise error"
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
        #endregion

        #region Done add to call
        q = split_heads(q, num_heads)
        if cache is None:
            k = split_heads(k, num_heads)
            v = split_heads(v, num_heads)

        key_depth_per_head = total_key_depth // num_heads
        if not vars_3d:
            q *= tf.cast(key_depth_per_head,dtype=q.dtype)**-0.5

        additional_returned_value = None
        #endregion
        
        #region Done add to call
        if attention_type == "dot_product":
            x = dot_product_attention(q, k, v, bias, dropout_rate, image_shapes,
                            save_weights_to=save_weights_to,
                            make_image_summary=make_image_summary,
                            dropout_broadcast_dims=dropout_broadcast_dims,
                            activation_dtype=kwargs.get(
                                "activation_dtype"),
                            hard_attention_k=hard_attention_k)

        x = combine_heads(x)
        # endregion

        # region Done Add to your attention model Copy to call but create a dense layer to handle opers below
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
        # end region
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
        assert_op2 = tf.Assert( tf.equal( tf.math.floormod(total_value_depth, num_heads), 0 ), [total_key_depth, tf.constant(num_heads)] )

        with tf.control_dependencies([assert_op1, assert_op2]):
            self.ln1 = tf.keras.layers.LayerNormalization(axis=-1 , epsilon=1e-4 , trainable=self.trainable )
        # endregion

        #region attention layers
        self.dense_query =  tf.keras.layers.Dense( total_key_depth, use_bias=False, activation="linear", name="q")
        self.dense_key =    tf.keras.layers.Dense( total_key_depth, use_bias=False, activation="linear", name="k")  
        if transform_value_antecedent:
            self.dense_value = tf.keras.layers.Dense( total_value_depth, use_bias=False, activation="linear", name="v" )
        else:
            self.dense_value = tf.keras.layers.Activation("linear")
        if(self.max_relative_position==None):
           self.max_relative_position =  tf.constant( int(self.attn_factor_reduc/2 - 1) , dtype=tf.int32 )

        vocab_size = int(self.attn_factor_reduc) #int(self.max_relative_position * 2 + 1)
        self.embeddings_table_k = tf.Variable( tf.keras.initializers.glorot_uniform()(shape=[vocab_size, total_key_depth ], dtype=self._compute_dtype  ))
        self.embeddings_table_v = tf.Variable( tf.keras.initializers.glorot_uniform()(shape=[vocab_size, total_value_depth ], dtype=self._compute_dtype  )) 

        if transform_output:
            self.dense_output = tf.keras.layers.Dense( output_depth, use_bias=False  )
        elif not transform_output:
            self.dense_output = tf.keras.layers.Activation("linear")
        #endregion

        
    def call(self, inputs , k_antecedent, v_antecedent):
        """
            :param inputs: q_antecedent This is required due to keras' need for layers to have an input argument

            :inputs: is queries
        """
      
        # region size reduction
        output_shape = v_antecedent.shape.as_list() #NOTE shape.as_list()[:-1] may not work in graph mode
        output_shape[1] = 1 # inputs.shape[1]
        # queries = tf.nn.avg_pool3d( inputs, strides=self.kq_downscale_stride,
        #                         ksize=self.kq_downscale_kernelshape, padding="SAME")
        # keys = tf.nn.avg_pool3d( inputs_k, strides=self.kq_downscale_stride,
        #                         ksize=self.kq_downscale_kernelshape, padding="SAME")

        q_antecedent = tf.cast( tf.nn.avg_pool3d( tf.cast(inputs,tf.float32), strides=self.kq_downscale_stride,
                                ksize=self.kq_downscale_kernelshape, padding="SAME"), tf.float16)
        k_antecedent = tf.cast(tf.nn.avg_pool3d( tf.cast(k_antecedent,tf.float32), strides=self.kq_downscale_stride,
                                ksize=self.kq_downscale_kernelshape, padding="SAME"), tf.float16)

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
        super(DeformableConvLayer, self ).__init__(
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
        #self._dtype = dtype
        
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
            dtype=self._compute_dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self._compute_dtype)

        # create offset conv layer
        offset_num = self.kernel_size[0] * self.kernel_size[1] * self.num_deformable_group
        self.offset_layer_kernel = self.add_weight(
            name='offset_layer_kernel',
            shape=self.kernel_size + (input_dim, offset_num * 2),  # 2 means x and y axis
            initializer=tf.zeros_initializer(),
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype=self._compute_dtype)
        self.offset_layer_bias = self.add_weight(
            name='offset_layer_bias',
            shape=(offset_num * 2,),
            initializer=tf.zeros_initializer(),
            # initializer=tf.random_uniform_initializer(-5, 5),
            regularizer=self.bias_regularizer,
            trainable=True,
            dtype=self._compute_dtype)
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
        #y, x = [tf.cast(i,self._compute_dtype)  for i in [y, x]]

        # add offset
        y, x = y + y_off, x + x_off
        y = tf.clip_by_value(y, 0, in_h - 1)
        x = tf.clip_by_value(x, 0, in_w - 1)

        # get four coordinates of points around (x, y)
        y0, x0 = [tf.cast(tf.floor(i), self._compute_dtype) for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1
        # clip
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]

        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [DeformableConvLayer._get_pixel_values_at_point(inputs, i, self._compute_dtype) for i in indices]

        # cast to float
        x0, x1, y0, y1 = [tf.cast(i,self._compute_dtype) for i in [x0, x1, y0, y1]]
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
        x, y = [ tf.cast(x, self._compute_dtype) , tf.cast(y, self._compute_dtype)]
        return y, x

    @staticmethod
    def _get_pixel_values_at_point(inputs, indices, dtype):
        """get pixel values
        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        batch, h, w, n = y.get_shape().as_list()[0: 4]

        batch_idx = tf.reshape(tf.range(0.0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))
        pixel_idx = tf.stack([ tf.cast(b,dtype), y, x], axis=-1)
        return tf.gather_nd(inputs, tf.cast(pixel_idx, tf.int32 ) )