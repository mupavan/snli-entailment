from keras import backend as K
from keras.engine import Layer
from keras.layers import Lambda, LSTM

import inspect
import sys
import types as python_types
import marshal


class HiddenStateLSTM(LSTM):
    """LSTM with input/output capabilities for its hidden state.
    This layer behaves just like an LSTM, except that it accepts further inputs
    to be used as its initial states, and returns additional outputs,
    representing the layer's final states.
    See Also:
        https://github.com/fchollet/keras/issues/2995
    """
    def build(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) > 1:
            # input_shape, *hidden_shapes = input_shape
            hidden_shapes = input_shape[1:]
            input_shape = input_shape[0]
            for shape in hidden_shapes:
                assert shape[0]  == input_shape[0]
                assert shape[-1] == self.output_dim
        super(HiddenStateLSTM, self).build(input_shape)

    def call(self, x, mask=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        input_shape = self.input_spec[0].shape
        if isinstance(x, (tuple, list)):
            # x, *custom_initial = x
            custom_initial = x[1:]
            x = x[0]
        else:
            custom_initial = None
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))
        if self.stateful and custom_initial:
            raise Exception(('Initial states should not be specified '
                             'for stateful LSTMs, since they would overwrite '
                             'the memorized states.'))
        elif custom_initial:
            initial_states = custom_initial
        elif self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        # only use the main input mask
        if isinstance(mask, list):
            mask = mask[0]

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=input_shape[1])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        if self.return_sequences:
            return [outputs] + states
        else:
            return [last_output] + states

    def get_output_shape_for(self, input_shape):
        if isinstance(input_shape, list) and len(input_shape) > 1:
            input_shape = input_shape[0]
        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], self.output_dim)
        else:
            output_shape = (input_shape[0], self.output_dim)
        state_output = (input_shape[0], self.output_dim)
        return [output_shape, state_output, state_output]

    def compute_mask(self, input, mask):
        if isinstance(mask, list) and len(mask) > 1:
            return mask
        elif self.return_sequences:
            return [mask, None, None]
        else:
            return [None] * 3


class MaskEatingLambda(Layer):
    '''Used for evaluating an arbitrary Theano / TensorFlow expression
    on the output of the previous layer.
    # Examples
    ```python
        # add a x -> x^2 layer
        model.add(Lambda(lambda x: x ** 2))
    ```
    ```python
        # add a layer that returns the concatenation
        # of the positive part of the input and
        # the opposite of the negative part
        def antirectifier(x):
            x -= K.mean(x, axis=1, keepdims=True)
            x = K.l2_normalize(x, axis=1)
            pos = K.relu(x)
            neg = K.relu(-x)
            return K.concatenate([pos, neg], axis=1)
        def antirectifier_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 2  # only valid for 2D tensors
            shape[-1] *= 2
            return tuple(shape)
        model.add(Lambda(antirectifier, output_shape=antirectifier_output_shape))
    ```
    # Arguments
        function: The function to be evaluated.
            Takes one argument: the output of previous layer
        output_shape: Expected output shape from function.
            Could be a tuple or a function of the shape of the input
        arguments: optional dictionary of keyword arguments to be passed
            to the function.
    # Input shape
        Arbitrary. Use the keyword argument input_shape
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Specified by `output_shape` argument.
    '''
    def __init__(self, function, output_shape=None, arguments={}, **kwargs):
        self.function = function
        self.arguments = arguments
        self.supports_masking = True
        if output_shape is None:
            self._output_shape = None
        elif type(output_shape) in {tuple, list}:
            self._output_shape = tuple(output_shape)
        else:
            if not hasattr(output_shape, '__call__'):
                raise Exception('In Lambda, `output_shape` '
                                'must be a list, a tuple, or a function.')
            self._output_shape = output_shape
        super(MaskEatingLambda, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self._output_shape is None:
            # if TensorFlow, we can infer the output shape directly:
            if K._BACKEND == 'tensorflow':
                if type(input_shape) is list:
                    xs = [K.placeholder(shape=shape) for shape in input_shape]
                    x = self.call(xs)
                else:
                    x = K.placeholder(shape=input_shape)
                    x = self.call(x)
                if type(x) is list:
                    return [K.int_shape(x_elem) for x_elem in x]
                else:
                    return K.int_shape(x)
            # otherwise, we default to the input shape
            return input_shape
        elif type(self._output_shape) in {tuple, list}:
            nb_samples = input_shape[0] if input_shape else None
            return (nb_samples,) + tuple(self._output_shape)
        else:
            shape = self._output_shape(input_shape)
            if type(shape) not in {list, tuple}:
                raise Exception('output_shape function must return a tuple')
            return tuple(shape)

    def call(self, x, mask=None):
        arguments = self.arguments
        arg_spec = inspect.getargspec(self.function)
        if 'mask' in arg_spec.args:
            arguments['mask'] = mask
        return self.function(x, **arguments)

    def get_config(self):
        py3 = sys.version_info[0] == 3

        if isinstance(self.function, python_types.LambdaType):
            if py3:
                function = marshal.dumps(self.function.__code__).decode('raw_unicode_escape')
            else:
                function = marshal.dumps(self.function.func_code).decode('raw_unicode_escape')
            function_type = 'lambda'
        else:
            function = self.function.__name__
            function_type = 'function'

        if isinstance(self._output_shape, python_types.LambdaType):
            if py3:
                output_shape = marshal.dumps(self._output_shape.__code__)
            else:
                output_shape = marshal.dumps(self._output_shape.func_code)
            output_shape_type = 'lambda'
        elif callable(self._output_shape):
            output_shape = self._output_shape.__name__
            output_shape_type = 'function'
        else:
            output_shape = self._output_shape
            output_shape_type = 'raw'

        config = {'function': function,
                  'function_type': function_type,
                  'output_shape': output_shape,
                  'output_shape_type': output_shape_type,
                  'arguments': self.arguments}
        base_config = super(Lambda, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self,x,input_mask=None):
        return None

    @classmethod
    def from_config(cls, config):
        function_type = config.pop('function_type')
        if function_type == 'function':
            function = globals()[config['function']]
        elif function_type == 'lambda':
            function = marshal.loads(config['function'].encode('raw_unicode_escape'))
            function = python_types.FunctionType(function, globals())
        else:
            raise Exception('Unknown function type: ' + function_type)

        output_shape_type = config.pop('output_shape_type')
        if output_shape_type == 'function':
            output_shape = globals()[config['output_shape']]
        elif output_shape_type == 'lambda':
            output_shape = marshal.loads(config['output_shape'])
            output_shape = python_types.FunctionType(output_shape, globals())
        else:
            output_shape = config['output_shape']

        config['function'] = function
        config['output_shape'] = output_shape
        return cls(**config)


# keras maskeating lambda functions
def lambda_mask_average(x,mask=None):
    denom = K.sum(K.cast(mask, dtype=K.floatx()), axis=-1, keepdims=True)
    denom = denom + 1e-16 # small epsilon to avoid nan
    return K.batch_dot(x,K.cast(mask, dtype=K.floatx()),axes=1) / denom


def lambda_mask_sum(x, mask=None):
    return K.batch_dot(x,K.cast(mask, dtype=K.floatx()),axes=1)


def lambda_mask_max(x, mask=None):
    mask = K.permute_dimensions(mask, ((0, 1, 'x')))
    return K.max(x * K.cast(mask, dtype=K.floatx()), axis=1)
