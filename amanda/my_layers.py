import sys
import os
import json
import numpy

from keras.engine import InputSpec
from keras.layers.convolutional import *
from keras.layers.core import *
from keras.layers import recurrent, Input, merge, Masking
from keras.layers import TimeDistributed
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop, Adam
import keras.backend as K


class ThreeDOut(Layer):
	'''
	Implementation of multi-factor attention
	'''
	def __init__(self, output_dim, nb_feature=4,
		     init='glorot_uniform', weights=None,
		     W_regularizer=None, b_regularizer=None, activity_regularizer=None,
		     W_constraint=None, b_constraint=None,
		     bias=True, input_dim=None, **kwargs):
		self.output_dim = output_dim
		self.nb_feature = nb_feature
		if self.nb_feature < 2:
			raise NotImplementedError
		self.init = initializations.get(init)
		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)

		self.W_constraint = constraints.get(W_constraint)
		self.b_constraint = constraints.get(b_constraint)

		self.bias = bias
		self.initial_weights = weights
		self.input_spec = [InputSpec(ndim=2)]

		self.input_dim = input_dim
		if self.input_dim:
			kwargs['input_shape'] = (self.input_dim,)
		super(ThreeDOut, self).__init__(**kwargs)

	def build(self, input_shape):
		input_dim = input_shape[1]
		self.input_spec = [InputSpec(dtype=K.floatx(),
					     shape=(None, input_dim))]

		self.W = self.init((self.nb_feature, input_dim, self.output_dim),
				   name='{}_W'.format(self.name))
		if self.bias:
			self.b = K.zeros((self.nb_feature, self.output_dim),
					 name='{}_b'.format(self.name))
			self.trainable_weights = [self.W, self.b]
		else:
			self.trainable_weights = [self.W]

		self.regularizers = []
		if self.W_regularizer:
			self.W_regularizer.set_param(self.W)
			self.regularizers.append(self.W_regularizer)

		if self.bias and self.b_regularizer:
			self.b_regularizer.set_param(self.b)
			self.regularizers.append(self.b_regularizer)

		if self.activity_regularizer:
			self.activity_regularizer.set_layer(self)
			self.regularizers.append(self.activity_regularizer)

		self.constraints = {}
		if self.W_constraint:
			self.constraints[self.W] = self.W_constraint
		if self.bias and self.b_constraint:
			self.constraints[self.b] = self.b_constraint

		if self.initial_weights is not None:
			self.set_weights(self.initial_weights)
			del self.initial_weights

	def get_output_shape_for(self, input_shape):
		assert input_shape and len(input_shape) == 2
		# return (input_shape[0], self.output_dim)
		return (input_shape[0], self.nb_feature, self.output_dim)

	def call(self, x, mask=None):
		# no activation, this layer is only linear.
		output = K.dot(x, self.W)
		if self.bias:
			output += self.b
		# output = K.max(output, axis=1)
		return output

	def get_config(self):
		config = {'output_dim': self.output_dim,
			  'init': self.init.__name__,
			  'nb_feature': self.nb_feature,
			  'W_regularizer': self.W_regularizer.get_config()
			  if self.W_regularizer else None,
			  'b_regularizer': self.b_regularizer.get_config()
			  if self.b_regularizer else None,
			  'activity_regularizer': self.activity_regularizer.get_config()
			  if self.activity_regularizer else None,
			  'W_constraint': self.W_constraint.get_config()
			  if self.W_constraint else None,
			  'b_constraint': self.b_constraint.get_config()
			  if self.b_constraint else None,
			  'bias': self.bias,
			  'input_dim': self.input_dim}
		base_config = super(ThreeDOut, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


def custom_softmax(x, axis=1):
	'''
	Apply softmax on different axis.
	'''
	e = K.exp(x - K.max(x, axis=axis, keepdims=True))
	s = K.sum(e, axis=1, keepdims=True)
	return e / s


class MeanOverTime(Layer):
	'''
	Mean over time layer
	'''
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(MeanOverTime, self).__init__(**kwargs)

	def call(self, x, mask=None):
		if mask is not None:
			return K.cast(x.sum(axis=1) / mask.sum(axis=1, keepdims=True), K.floatx())
		else:
			# return K.mean(x, axis=1)
			return K.cast(x.sum(axis=1) / K.ones_like(x).sum(axis=1), K.floatx())

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[2])

	def compute_mask(self, x, mask):
		return None

	def get_config(self):
		config = {}
		base_config = super(MeanOverTime, self).get_config()
		return dict(list(base_config.items()))


class AttentionWoW(Layer):
	'''
	    Attention layer with only the vector
	    without any bilinear term
	'''

	def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
		self.supports_masking = True
		assert op in {'attsum', 'attmean'}
		assert activation in {None, 'tanh'}
		self.op = op
		self.activation = activation
		self.init_stdev = init_stdev
		super(AttentionWoW, self).__init__(**kwargs)

	def build(self, input_shape):
		init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
		self.att_v = K.variable(init_val_v, name='att_v')
		self.trainable_weights = [self.att_v]

	def call(self, x, mask=None):
		if not self.activation:
			weights = K.theano.tensor.tensordot(self.att_v, x, axes=[0, 2])
		elif self.activation == 'tanh':
			weights = K.theano.tensor.tensordot(self.att_v, K.tanh(x), axes=[0, 2])
		weights = K.softmax(weights)
		out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
		if self.op == 'attsum':
			out = out.sum(axis=1)
		elif self.op == 'attmean':
			out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
		return K.cast(out, K.floatx())

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], input_shape[2])

	def compute_mask(self, x, mask):
		return None

	def get_config(self):
		config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
		base_config = super(AttentionWoW, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class Concat3D2D(Layer):
	'''
	Broadcasted concatenation
	'''

	def __init__(self, **kwargs):
		self.supports_masking = True
		super(Concat3D2D, self).__init__(**kwargs)

	def call(self, inputs, mask=None):
		repeated = K.repeat(inputs[1], inputs[0].shape[1])
		return K.concatenate([inputs[0], repeated], axis=-1)

	def get_output_shape_for(self, input_shapes):
		return (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2] + input_shapes[1][1])

	def compute_mask(self, inputs, mask):
		return mask[0]

def concat_3d2d(inputs):
	"""
	    Functional API for Concat3D2D
	"""
	concat_layer = Concat3D2D()
	return concat_layer(inputs)


class MaskedConcat(Layer):
	"""
	    Masked concats 2 3D tensors along -1 axis
	    Preserves masking
	"""
	def __init__(self, concat_axis, layers=None, **kwargs):
		self.concat_axis = concat_axis
		self.supports_masking = True
		super(MaskedConcat, self).__init__(**kwargs)

		if layers:
			node_indices = [0 for _ in range(len(layers))]
			self.built = True
			self.add_inbound_node(layers, node_indices, None)
		else:
			self.built = False

	def call(self, inputs, mask=None):
		assert isinstance(inputs, list)
		if mask:
			assert isinstance(mask, list) or isinstance(mask, tuple)
			if self.concat_axis == 2:
				assert K.equal(mask[0], mask[1])

		return K.concatenate(inputs, axis=self.concat_axis)

	def compute_mask(self, inputs, input_mask=None):
		if input_mask:
			assert (isinstance(input_mask, list) or isinstance(input_mask, tuple))
			if self.concat_axis == 2:
				assert (K.equal(input_mask[0], input_mask[1]))
				return input_mask[0]
			elif self.concat_axis == 1:
				return K.concatenate(input_mask, axis=1)
			else:
				return None
		else:
			return None

	def get_output_shape_for(self, input_shapes):
		if self.concat_axis == 2:
			in_shape = input_shapes[0]
			output_shape = (in_shape[0], in_shape[1], in_shape[2] * 2)
		elif self.concat_axis == 1:
			output_shape = (input_shapes[0][0],
					input_shapes[0][1] + input_shapes[1][1],
					input_shapes[0][2])
		return output_shape


def masked_concat(inputs, concat_axis=2):
	"""
	    Functional API for MaskedConcat
	"""
	concat_layer = MaskedConcat(concat_axis=concat_axis)
	return concat_layer(inputs)


class MaskedSumBroadCast(Layer):
	"""
	    Sums  3D and 2D tensors, accounts for mask
	"""

	def __init__(self, layers=None, **kwargs):
		self.supports_masking = True
		super(MaskedSumBroadCast, self).__init__(**kwargs)

		if layers:
			node_indices = [0 for _ in range(len(layers))]
			self.built = True
			self.add_inbound_node(layers, node_indices, None)
		else:
			self.built = False

	def call(self, inputs, mask=None):
		"""
		Mask 1 exists, while mask 2 does not
		"""
		# assert isinstance(mask, list)
		# assert mask[0] != None and mask[1] == None
		return inputs[0] + K.theano.tensor.addbroadcast(inputs[1], 1)

	def compute_mask(self, inputs, input_mask=None):
		return input_mask[0]

	def get_output_shape_for(self, input_shapes):
		return input_shapes[0]


def masked_sum_broadcast(inputs):
	masked_sum_layer = MaskedSumBroadCast()
	return masked_sum_layer(inputs)
