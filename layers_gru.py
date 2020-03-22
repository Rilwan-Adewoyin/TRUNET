import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float16')
#tf.keras.backend.set_floatx('float32')


from tensorflow.python.ops import inplace_ops
from tensorflow.python.keras import backend as K

##New imports
from tensorflow.keras import activations
#from tensorflow.keras import backend as K
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op


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
from tensorflow.python.keras.layers.recurrent_v2 import is_sequence_right_padded, calculate_sequence_by_mask,  device, uuid
from tensorflow.python.keras.layers.recurrent import _caching_device

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import resource_variable_ops, gen_cudnn_rnn_ops, control_flow_ops, math_ops, resource_variable_ops, state_ops
from tensorflow.python.eager import function
from tensorflow.python.eager import context


# The following string constants are used by Defun approach for unified backend
# of LSTM and GRU.
_DEFUN_API_NAME_ATTRIBUTE = 'api_implements'
_DEFUN_DEVICE_ATTRIBUTE = 'api_preferred_device'
_CPU_DEVICE_NAME = 'CPU'
_GPU_DEVICE_NAME = 'GPU'

# The following number constants are used to represent the runtime of the defun
# backend function. Since the CPU/GPU implementation are mathematically same, we
# need some signal for the function to indicate which function is executed. This
# is for testing purpose to verify the correctness of swapping backend function.
_RUNTIME_UNKNOWN = 0
_RUNTIME_CPU = 1
_RUNTIME_GPU = 2

RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')


class GRU_LN(RNN):
  """Gated Recurrent Unit - Cho et al. 2014.
    There are two variants. The default one is based on 1406.1078v3 and
    has reset gate applied to hidden state before matrix multiplication. The
    other one is based on original 1406.1078v1 and has the order reversed.
    The second variant is compatible with CuDNNGRU (GPU-only) and allows
    inference on CPU. Thus it has separate biases for `kernel` and
    `recurrent_kernel`. Use `'reset_after'=True` and
    `recurrent_activation='sigmoid'`.
    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: hard sigmoid (`hard_sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      recurrent_regularizer: Regularizer function applied to
        the `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
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
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of
        smaller dot products and additions, whereas mode 2 will
        batch them into fewer, larger operations. These modes will
        have different performance profiles on different hardware and
        for different applications.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state
        in addition to the output.
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
      time_major: The shape format of the `inputs` and `outputs` tensors.
        If True, the inputs and outputs will be in shape
        `(timesteps, batch, ...)`, whereas in the False case, it will be
        `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
        efficient because it avoids transposes at the beginning and end of the
        RNN calculation. However, most TensorFlow data is batch-major, so by
        default this function accepts input and emits output in batch-major
        form.
      reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before" (default),
        True = "after" (CuDNN compatible).
    Call arguments:
      inputs: A 3D tensor.
      mask: Binary tensor of shape `(samples, timesteps)` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.
  """

  def __init__(self,
               units,
               layer_norm,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               reset_after=False,
               **kwargs):
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    layer_norm._dtype = "float32"
    #layer_norm._compute_dtype = "float16"
    cell = GRU_LN_Cell(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        layer_norm = layer_norm,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        reset_after=reset_after,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True))
    super(GRU_LN, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]
    self.layer_norm = layer_norm

  def call(self, inputs, mask=None, training=None, initial_state=None):
    self._maybe_reset_cell_dropout_mask(self.cell)
    return super(GRU_LN, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)
  #region properties
  @property
  def units(self):
    return self.cell.units

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

  @property
  def implementation(self):
    return self.cell.implementation
  
  @property
  def reset_after(self):
    return self.cell.reset_after
  #endregion

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'layer_norm':
          self.layer_norm,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation,
        'reset_after':
            self.reset_after
    }
    base_config = super(GRU_LN, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)

class GRU_LN_v2(DropoutRNNCellMixin, GRU_LN):
  """Gated Recurrent Unit - Cho et al. 2014.
    See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
    for details about the usage of RNN API.
    Based on available runtime hardware and constraints, this layer
    will choose different implementations (cuDNN-based or pure-TensorFlow)
    to maximize the performance. If a GPU is available and all
    the arguments to the layer meet the requirement of the CuDNN kernel
    (see below for details), the layer will use a fast cuDNN implementation.
    The requirements to use the cuDNN implementation are:
    1. `activation` == `tanh`
    2. `recurrent_activation` == `sigmoid`
    3. `recurrent_dropout` == 0
    4. `unroll` is `False`
    5. `use_bias` is `True`
    6. `reset_after` is `True`
    7. Inputs are not masked or strictly right padded.
    There are two variants of the GRU implementation. The default one is based on
    [v3](https://arxiv.org/abs/1406.1078v3) and has reset gate applied to hidden
    state before matrix multiplication. The other one is based on
    [original](https://arxiv.org/abs/1406.1078v1) and has the order reversed.
    The second variant is compatible with CuDNNGRU (GPU-only) and allows
    inference on CPU. Thus it has separate biases for `kernel` and
    `recurrent_kernel`. To use this variant, set `'reset_after'=True` and
    `recurrent_activation='sigmoid'`.
    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: sigmoid (`sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default:
        `glorot_uniform`.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent
        state. Default: `orthogonal`.
      bias_initializer: Initializer for the bias vector. Default: `zeros`.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_regularizer: Regularizer function applied to the bias vector. Default:
        `None`.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation"). Default: `None`.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_constraint: Constraint function applied to the `recurrent_kernel`
        weights matrix. Default: `None`.
      bias_constraint: Constraint function applied to the bias vector. Default:
        `None`.
      dropout: Float between 0 and 1. Fraction of the units to drop for the linear
        transformation of the inputs. Default: 0.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop for
        the linear transformation of the recurrent state. Default: 0.
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of
        smaller dot products and additions, whereas mode 2 will
        batch them into fewer, larger operations. These modes will
        have different performance profiles on different hardware and
        for different applications. Default: 2.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence. Default: `False`.
      return_state: Boolean. Whether to return the last state in addition to the
        output. Default: `False`.
      go_backwards: Boolean (default `False`).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.
      time_major: The shape format of the `inputs` and `outputs` tensors.
        If True, the inputs and outputs will be in shape
        `[timesteps, batch, feature]`, whereas in the False case, it will be
        `[batch, timesteps, feature]`. Using `time_major = True` is a bit more
        efficient because it avoids transposes at the beginning and end of the
        RNN calculation. However, most TensorFlow data is batch-major, so by
        default this function accepts input and emits output in batch-major
        form.
      reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before",
        True = "after" (default and CuDNN compatible).
    Call arguments:
      inputs: A 3D tensor, with shape `[batch, timesteps, feature]`.
      mask: Binary tensor of shape `[samples, timesteps]` indicating whether
        a given timestep should be masked  (optional, defaults to `None`).
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used  (optional, defaults to `None`).
      initial_state: List of initial state tensors to be passed to the first
        call of the cell  (optional, defaults to `None` which causes creation
        of zero-filled initial state tensors).
    Examples:
    ```python
    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    gru = tf.keras.layers.GRU(4)
    output = gru(inputs)  # The output has shape `[32, 4]`.
    gru = tf.keras.layers.GRU(4, return_sequences=True, return_state=True)
    # whole_sequence_output has shape `[32, 10, 4]`.
    # final_state has shape `[32, 4]`.
    whole_sequence_output, final_state = gru(inputs)
    ```
  """

  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=2,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               time_major=False,
               reset_after=True,
               **kwargs):
    # return_runtime is a flag for testing, which shows the real backend
    # implementation chosen by grappler in graph mode.
    self._return_runtime = kwargs.pop('return_runtime', False)

    super(GRU_LN_v2, self).__init__(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        time_major=time_major,
        reset_after=reset_after,
        **kwargs)
    # CuDNN uses following setting by default and not configurable.
    self.could_use_cudnn = (
        activation == 'tanh' and recurrent_activation == 'sigmoid' and
        recurrent_dropout == 0 and not unroll and use_bias and
        reset_after and ops.executing_eagerly_outside_functions())

  def build(self, input_shape):
    super(GRU_LN_v2, self).build(input_shape)

    if not all(isinstance(v, resource_variable_ops.ResourceVariable)
               for v in self.weights):
      # Non-resource variables, such as DistributedVariables and
      # AutoCastVariables, do not work properly with the implementation
      # selector, which is used when cuDNN is used. However, by chance, such
      # variables happen to work in LSTM, so this check is only needed for GRU.
      # TODO(b/136512020): Make non-resource variables work with the
      # implementation selector.
      self.could_use_cudnn = False
      if( self.could_use_cudnn  == True ):
        raise Exception("This layer normalization is currently only implmented for the non cudnn implementation")

  def call(self, inputs, mask=None, training=None, initial_state=None):
    # The input should be dense, padded with zeros. If a ragged input is fed
    # into the layer, it is padded and the row lengths are used for masking.
    inputs, row_lengths = K.convert_inputs_if_ragged(inputs)
    is_ragged_input = (row_lengths is not None)
    self._validate_args_if_ragged(is_ragged_input, mask)

    # GRU does not support constants. Ignore it during process.
    inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)

    if isinstance(mask, list):
      mask = mask[0]

    input_shape = K.int_shape(inputs)
    timesteps = input_shape[0] if self.time_major else input_shape[1]

    if not self.could_use_cudnn:
      kwargs = {'training': training}
      self._maybe_reset_cell_dropout_mask(self.cell)

      def step(cell_inputs, cell_states):
        return self.cell.call(cell_inputs, cell_states, **kwargs)

      last_output, outputs, states = K.rnn(
          step,
          inputs,
          initial_state,
          constants=None,
          go_backwards=self.go_backwards,
          mask=mask,
          unroll=self.unroll,
          input_length=row_lengths if row_lengths is not None else timesteps,
          time_major=self.time_major,
          zero_output_for_mask=self.zero_output_for_mask)
      # This is a dummy tensor for testing purpose.
      runtime = _runtime(_RUNTIME_UNKNOWN)
    else:
      last_output, outputs, runtime, states = self._defun_gru_call(
          inputs, initial_state, training, mask, row_lengths)

    if self.stateful:
      updates = [state_ops.assign(self.states[0], states[0])]
      self.add_update(updates)

    if self.return_sequences:
      output = K.maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)
    else:
      output = last_output

    if self.return_state:
      return [output] + list(states)
    elif self._return_runtime:
      return output, runtime
    else:
      return output

  def _defun_gru_call(self, inputs, initial_state, training, mask,
                      sequence_lengths):
    # Use the new defun approach for backend implementation swap.
    # Note that different implementations need to have same function
    # signature, eg, the tensor parameters need to have same shape and dtypes.

    self.reset_dropout_mask()
    dropout_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    if dropout_mask is not None:
      inputs = inputs * dropout_mask[0]

    cudnn_gru_kwargs = {
        'inputs': inputs,
        'init_h': initial_state[0],
        'kernel': self.cell.kernel,
        'recurrent_kernel': self.cell.recurrent_kernel,
        'bias': self.cell.bias,
        'mask': mask,
        'time_major': self.time_major,
        'go_backwards': self.go_backwards,
        'sequence_lengths': sequence_lengths
    }
    normal_gru_kwargs = cudnn_gru_kwargs.copy()
    normal_gru_kwargs.update({
        'activation': self.activation,
        'recurrent_activation': self.recurrent_activation
    })

    if context.executing_eagerly():
      device_type = _get_context_device_type()
      can_use_gpu = (
          # Either user specified GPU or unspecified but GPU is available.
          (device_type == _GPU_DEVICE_NAME
           or (device_type is None and context.num_gpus() > 0))
          and
          (mask is None or is_sequence_right_padded(mask, self.time_major)))
      # Under eager context, check the device placement and prefer the
      if can_use_gpu:
        last_output, outputs, new_h, runtime = cudnn_gru(**cudnn_gru_kwargs)
      else:
        last_output, outputs, new_h, runtime = standard_gru(**normal_gru_kwargs)
    else:
      last_output, outputs, new_h, runtime = gru_with_backend_selection(
          **normal_gru_kwargs)

    states = [new_h]
    return last_output, outputs, runtime, states

class GRU_LN_Cell(DropoutRNNCellMixin, Layer):
  """Cell class for the GRU layer.
    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      recurrent_activation: Activation function to use
        for the recurrent step.
        Default: hard sigmoid (`hard_sigmoid`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix,
        used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
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
        Fraction of the units to drop for the linear transformation of the inputs.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for
        the linear transformation of the recurrent state.
      implementation: Implementation mode, either 1 or 2.
        Mode 1 will structure its operations as a larger number of
        smaller dot products and additions, whereas mode 2 will
        batch them into fewer, larger operations. These modes will
        have different performance profiles on different hardware and
        for different applications.
      reset_after: GRU convention (whether to apply reset gate after or
        before matrix multiplication). False = "before" (default),
        True = "after" (CuDNN compatible).
    Call arguments:
      inputs: A 2D tensor.
      states: List of state tensors corresponding to the previous timestep.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.
  """

  def __init__(self,
               units,
               layer_norm,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               reset_after=False,
               **kwargs):
    self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    super(GRU_LN_Cell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)  
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias
    
    self.layer_norm = layer_norm
    if self.layer_norm == None:
      self.bool_ln = False
    else:
      self.bool_ln = True
      self.layer_norm._dtype =  "float32"

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    if self.recurrent_dropout != 0 and implementation != 1:
      logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
      self.implementation = 1
    else:
      self.implementation = implementation
    self.reset_after = reset_after
    self.state_size = self.units
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[-1]
    default_caching_device = _caching_device(self)
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 3),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 3),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)

    if self.use_bias:
      if not self.reset_after:
        bias_shape = (3 * self.units,)
      else:
        # separate biases for input and recurrent kernels
        # Note: the shape is intentionally different from CuDNNGRU biases
        # `(2 * 3 * self.units,)`, so that we can distinguish the classes
        # when loading and converting saved weights.
        bias_shape = (2, 3 * self.units)
      self.bias = self.add_weight(shape=bias_shape,
                                  name='bias',
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint,
                                  caching_device=default_caching_device)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, states, training=None):
    h_tm1 = states[0]  # previous memory

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=3)

    if self.use_bias:
      if not self.reset_after:
        input_bias, recurrent_bias = self.bias, None
      else:
        input_bias, recurrent_bias = array_ops.unstack(self.bias)

    if self.implementation == 1:
      if 0. < self.dropout < 1.:
        inputs_z = inputs * dp_mask[0]
        inputs_r = inputs * dp_mask[1]
        inputs_h = inputs * dp_mask[2]
      else:
        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

      x_z = K.dot(inputs_z, self.kernel[:, :self.units])
      x_r = K.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
      x_h = K.dot(inputs_h, self.kernel[:, self.units * 2:])

      if self.use_bias:
        x_z = K.bias_add(x_z, input_bias[:self.units])
        x_r = K.bias_add(x_r, input_bias[self.units: self.units * 2])
        x_h = K.bias_add(x_h, input_bias[self.units * 2:])

      if 0. < self.recurrent_dropout < 1.:
        h_tm1_z = h_tm1 * rec_dp_mask[0]
        h_tm1_r = h_tm1 * rec_dp_mask[1]
        h_tm1_h = h_tm1 * rec_dp_mask[2]
      else:
        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

      recurrent_z = K.dot(h_tm1_z, self.recurrent_kernel[:, :self.units])
      recurrent_r = K.dot(h_tm1_r,
                          self.recurrent_kernel[:, self.units:self.units * 2])
      if self.reset_after and self.use_bias:
        recurrent_z = K.bias_add(recurrent_z, recurrent_bias[:self.units])
        recurrent_r = K.bias_add(recurrent_r,
                                 recurrent_bias[self.units:self.units * 2])

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      # reset gate applied after/before matrix multiplication
      if self.reset_after:
        recurrent_h = K.dot(h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
        if self.use_bias:
          recurrent_h = K.bias_add(recurrent_h, recurrent_bias[self.units * 2:])
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = K.dot(r * h_tm1_h,
                            self.recurrent_kernel[:, self.units * 2:])

      hh = self.activation(x_h + recurrent_h)
    else:
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]

      # inputs projected by all gate matrices at once
      matrix_x = K.dot(inputs, self.kernel)
      if self.use_bias:
        # biases: bias_z_i, bias_r_i, bias_h_i
        matrix_x = K.bias_add(matrix_x, input_bias)

      x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=-1)

      if self.reset_after:
        # hidden state projected by all gate matrices at once
        matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
          matrix_inner = K.bias_add(matrix_inner, recurrent_bias)
      else:
        # hidden state projected separately for update/reset and new
        matrix_inner = K.dot(h_tm1, self.recurrent_kernel[:, :2 * self.units])

      recurrent_z, recurrent_r, recurrent_h = array_ops.split(
          matrix_inner, [self.units, self.units, -1], axis=-1)

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      if self.reset_after:
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = K.dot(r * h_tm1,
                            self.recurrent_kernel[:, 2 * self.units:])

      hh = self.activation(x_h + recurrent_h)
    # previous and candidate state mixed by update gate
    if self.bool_ln :
      h = z * h_tm1 + (1 - z) * tf.cast( self.layer_norm(hh), self._compute_dtype) #layer norm
      #h = z * h_tm1 + (1 - z) * tf.cast( self.layer_norm(hh), self._dtype) #layer norm
    else:
      h = z * h_tm1 + (1 - z) * hh
    

    return h, [h]

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'dropout': self.dropout,
        'recurrent_dropout': self.recurrent_dropout,
        'implementation': self.implementation,
        'reset_after': self.reset_after,
        'layer_norm':self.layer_norm,
        'bool_ln':self.bool_ln
    }
    base_config = super(GRU_LN_Cell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

def standard_gru(inputs, init_h, kernel, recurrent_kernel, bias, activation,
                 recurrent_activation, mask, time_major, go_backwards,
                 sequence_lengths):
  """GRU with standard kernel implementation.
  This implementation can be run on all types of hardware.
  This implementation lifts out all the layer weights and make them function
  parameters. It has same number of tensor input params as the CuDNN
  counterpart. The RNN step logic has been simplified, eg dropout and mask is
  removed since CuDNN implementation does not support that.
  Arguments:
    inputs: Input tensor of GRU layer.
    init_h: Initial state tensor for the cell output.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. The bias contains the
      combined input_bias and recurrent_bias.
    activation: Activation function to use for output.
    recurrent_activation: Activation function to use for hidden recurrent state.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked.
    time_major: Boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
  Returns:
    last_output: output tensor for the last timestep, which has shape
      [batch, units].
    outputs: output tensor for all timesteps, which has shape
      [batch, time, units].
    state_0: the cell output, which has same shape as init_h.
    runtime: constant string tensor which indicate real runtime hardware. This
      value is for testing purpose and should be used by user.
  """
  input_shape = K.int_shape(inputs)
  timesteps = input_shape[0] if time_major else input_shape[1]

  input_bias, recurrent_bias = array_ops.unstack(bias)

  def step(cell_inputs, cell_states):
    """Step function that will be used by Keras RNN backend."""
    h_tm1 = cell_states[0]

    # inputs projected by all gate matrices at once
    matrix_x = K.dot(cell_inputs, kernel)
    matrix_x = K.bias_add(matrix_x, input_bias)

    x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=1)

    # hidden state projected by all gate matrices at once
    matrix_inner = K.dot(h_tm1, recurrent_kernel)
    matrix_inner = K.bias_add(matrix_inner, recurrent_bias)

    recurrent_z, recurrent_r, recurrent_h = array_ops.split(matrix_inner, 3,
                                                            axis=1)
    z = recurrent_activation(x_z + recurrent_z)
    r = recurrent_activation(x_r + recurrent_r)
    hh = activation(x_h + r * recurrent_h)

    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    return h, [h]

  last_output, outputs, new_states = K.rnn(
      step,
      inputs, [init_h],
      constants=None,
      unroll=False,
      time_major=time_major,
      mask=mask,
      go_backwards=go_backwards,
      input_length=sequence_lengths
      if sequence_lengths is not None else timesteps)
  return last_output, outputs, new_states[0], _runtime(_RUNTIME_CPU)

def cudnn_gru(inputs, init_h, kernel, recurrent_kernel, bias, mask, time_major,
              go_backwards, sequence_lengths):
  """GRU with CuDNN implementation which is only available for GPU."""
  if not time_major and mask is None:
    inputs = array_ops.transpose(inputs, perm=(1, 0, 2))
    seq_axis, batch_axis = (0, 1)
  else:
    seq_axis, batch_axis = (0, 1) if time_major else (1, 0)
  # For init_h, cuDNN expects one more dim of num_layers before or after batch
  # dim for time major or batch major inputs respectively
  init_h = array_ops.expand_dims(init_h, axis=seq_axis)

  weights = array_ops.split(kernel, 3, axis=1)
  weights += array_ops.split(recurrent_kernel, 3, axis=1)
  # Note that the bias was initialized as shape (2, 3 * units), flat it into
  # (6 * units)
  bias = array_ops.split(K.flatten(bias), 6)
  # Note that the gate order for CuDNN is different from the canonical format.
  # canonical format is [z, r, h], whereas CuDNN is [r, z, h]. The swap need to
  # be done for kernel, recurrent_kernel, input_bias, recurrent_bias.
  # z is update gate weights.
  # r is reset gate weights.
  # h is output gate weights.
  weights[0], weights[1] = weights[1], weights[0]
  weights[3], weights[4] = weights[4], weights[3]
  bias[0], bias[1] = bias[1], bias[0]
  bias[3], bias[4] = bias[4], bias[3]

  params = _canonical_to_params(
      weights=weights,
      biases=bias,
      shape=constant_op.constant([-1]),
      transpose_weights=True)

  if mask is not None:
    sequence_lengths = calculate_sequence_by_mask(mask, time_major)

  if sequence_lengths is not None:
    if go_backwards:
      # Three reversals are required. E.g.,
      # normal input = [1, 2, 3, 0, 0]  # where 0 need to be masked
      # reversed_input_to_cudnn = [3, 2, 1, 0, 0]
      # output_from_cudnn = [6, 5, 4, 0, 0]
      # expected_output = [0, 0, 6, 5 ,4]
      inputs = array_ops.reverse_sequence_v2(
          inputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
    outputs, h, _, _, _ = gen_cudnn_rnn_ops.cudnn_rnnv3(
        inputs,
        input_h=init_h,
        input_c=0,
        params=params,
        is_training=True,
        rnn_mode='gru',
        sequence_lengths=sequence_lengths,
        time_major=time_major)
    if go_backwards:
      outputs = array_ops.reverse_sequence_v2(
          outputs, sequence_lengths, seq_axis=seq_axis, batch_axis=batch_axis)
      outputs = array_ops.reverse(outputs, axis=[seq_axis])
  else:
    if go_backwards:
      # Reverse axis 0 since the input is already convert to time major.
      inputs = array_ops.reverse(inputs, axis=[0])
    outputs, h, _, _ = gen_cudnn_rnn_ops.cudnn_rnn(
        inputs, input_h=init_h, input_c=0, params=params, is_training=True,
        rnn_mode='gru')

  last_output = outputs[-1]
  if not time_major and mask is None:
    outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
  h = array_ops.squeeze(h, axis=seq_axis)

  # In the case of variable length input, the cudnn kernel will fill zeros for
  # the output, whereas the default keras behavior is to bring over the previous
  # output for t-1, so that in the return_sequence=False case, user can quickly
  # get the final effect output instead just 0s at the last timestep.
  # In order to mimic the default keras behavior, we copy the final h state as
  # the last_output, since it is numerically same as the output.
  if mask is not None:
    last_output = h

  return last_output, outputs, h, _runtime(_RUNTIME_GPU)

def gru_with_backend_selection(inputs, init_h, kernel, recurrent_kernel, bias,
                               mask, time_major, go_backwards, activation,
                               recurrent_activation, sequence_lengths):
  """Call the GRU with optimized backend kernel selection.
  Under the hood, this function will create two TF function, one with the most
  generic kernel and can run on all device condition, and the second one with
  CuDNN specific kernel, which can only run on GPU.
  The first function will be called with normal_lstm_params, while the second
  function is not called, but only registered in the graph. The Grappler will
  do the proper graph rewrite and swap the optimized TF function based on the
  device placement.
  Args:
    inputs: Input tensor of GRU layer.
    init_h: Initial state tensor for the cell output.
    kernel: Weights for cell kernel.
    recurrent_kernel: Weights for cell recurrent kernel.
    bias: Weights for cell kernel bias and recurrent bias. Only recurrent bias
      is used in this case.
    mask: Boolean tensor for mask out the steps within sequence.
    time_major: Boolean, whether the inputs are in the format of
      [time, batch, feature] or [batch, time, feature].
    go_backwards: Boolean (default False). If True, process the input sequence
      backwards and return the reversed sequence.
    activation: Activation function to use for output.
    recurrent_activation: Activation function to use for hidden recurrent state.
    sequence_lengths: The lengths of all sequences coming from a variable length
      input, such as ragged tensors. If the input has a fixed timestep size,
      this should be None.
  Returns:
    List of output tensors, same as standard_gru.
  """
  params = {
      'inputs': inputs,
      'init_h': init_h,
      'kernel': kernel,
      'recurrent_kernel': recurrent_kernel,
      'bias': bias,
      'mask': mask,
      'time_major': time_major,
      'go_backwards': go_backwards,
      'activation': activation,
      'recurrent_activation': recurrent_activation,
      'sequence_lengths': sequence_lengths
  }

  def cudnn_gru_with_fallback(inputs, init_h, kernel, recurrent_kernel, bias,
                              mask, time_major, go_backwards, activation,
                              recurrent_activation, sequence_lengths):
    """Use CuDNN kernel when mask is none or strictly right padded."""
    if mask is None:
      return cudnn_gru(
          inputs=inputs,
          init_h=init_h,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          sequence_lengths=sequence_lengths)

    def input_right_padded():
      return cudnn_gru(
          inputs=inputs,
          init_h=init_h,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          sequence_lengths=sequence_lengths)

    def input_not_right_padded():
      return standard_gru(
          inputs=inputs,
          init_h=init_h,
          kernel=kernel,
          recurrent_kernel=recurrent_kernel,
          bias=bias,
          mask=mask,
          time_major=time_major,
          go_backwards=go_backwards,
          activation=activation,
          recurrent_activation=recurrent_activation,
          sequence_lengths=sequence_lengths)

    return control_flow_ops.cond(
        is_sequence_right_padded(mask, time_major),
        true_fn=input_right_padded,
        false_fn=input_not_right_padded)

  # Each time a `tf.function` is called, we will give it a unique
  # identifiable API name, so that Grappler won't get confused when it
  # sees multiple GRU layers added into same graph, and it will be able
  # to pair up the different implementations across them.
  api_name = 'gru_' + str(uuid.uuid4())
  defun_standard_gru = _generate_defun_backend(
      api_name, _CPU_DEVICE_NAME, standard_gru)
  defun_cudnn_gru = _generate_defun_backend(
      api_name, _GPU_DEVICE_NAME, cudnn_gru_with_fallback)

  # Call the normal GRU impl and register the CuDNN impl function. The
  # grappler will kick in during session execution to optimize the graph.
  last_output, outputs, new_h, runtime = defun_standard_gru(**params)
  function.register(defun_cudnn_gru, **params)
  return last_output, outputs, new_h, runtime

def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
  if inputs is not None:
    batch_size = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)

def _canonical_to_params(weights, biases, shape, transpose_weights=False):
  """Utility function convert variable to CuDNN compatible parameter.
  Note that Keras weights for kernels are different from the CuDNN format. Eg.:
  ```
    Keras                 CuDNN
    [[0, 1, 2],  <--->  [[0, 2, 4],
     [3, 4, 5]]          [1, 3, 5]]
  ```
  If the input weights need to be in a unified format, then set
  `transpose_weights=True` to convert the weights.
  Args:
    weights: list of weights for the individual kernels and recurrent kernels.
    biases: list of biases for individual gate.
    shape: the shape for the converted variables that will be feed to CuDNN.
    transpose_weights: boolean, whether to transpose the weights.
  Returns:
    The converted weights that can be feed to CuDNN ops as param.
  """
  def convert(w):
    return array_ops.transpose(w) if transpose_weights else w

  weights = [array_ops.reshape(convert(x), shape) for x in weights]
  biases = [array_ops.reshape(x, shape) for x in biases]
  return array_ops.concat(weights + biases, axis=0)

def _generate_defun_backend(unique_api_name, preferred_device, func):
  function_attributes = {
      _DEFUN_API_NAME_ATTRIBUTE: unique_api_name,
      _DEFUN_DEVICE_ATTRIBUTE: preferred_device,
  }
  return function.defun_with_attributes(func=func,
                                        attributes=function_attributes,
                                        autograph=False)

def _get_context_device_type():
  """Parse the current context and return the device type, eg CPU/GPU."""
  current_device = context.context().device_name
  if current_device is None:
    return None
  return device.DeviceSpec.from_string(current_device)

def _runtime(runtime_name):
  with ops.device('/cpu:0'):
    return constant_op.constant(
        runtime_name, dtype=dtypes.float32, name='runtime')

def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
  """Generate a zero filled tensor with shape [batch_size, state_size]."""
  if batch_size_tensor is None or dtype is None:
    raise ValueError(
        'batch_size and dtype cannot be None while constructing initial state: '
        'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

  def create_zeros(unnested_state_size):
    flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
    init_state_size = [batch_size_tensor] + flat_dims
    return array_ops.zeros(init_state_size, dtype=dtype)

  if nest.is_sequence(state_size):
    return nest.map_structure(create_zeros, state_size)
  else:
    return create_zeros(state_size)