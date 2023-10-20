from functools import partial

from mindspore.nn import ReLU

from xt.model.impala.default_config import GAMMA, LR
from xt.model.ms_compat import ms, DTYPE_MAP, Adam, Conv2d, Flatten, Cell, Dense, TrainOneStepCell
import mindspore.ops as ops
import mindspore.nn as nn
import numpy as np
from xt.model.model_ms import XTModel_MS
from mindspore import Tensor, set_context, Parameter
from xt.model.ms_utils import MSVariables
import xt.model.impala.vtrace_ms as vtrace
from xt.model.atari_model import get_atari_filter
from xt.model.model_utils_ms import state_transform_ms, custom_norm_initializer_ms
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers
from absl import logging
from xt.model.impala.impala_cnn_ms import MyTrainOneStepCell

#set_context(mode=ms.PYNATIVE_MODE)
set_context(mode=ms.GRAPH_MODE)


@Registers.model
class ImpalaCnnOptMS(XTModel_MS):
	def __init__(self, model_info):
		model_config = model_info.get("model_config", dict())
		import_config(globals(), model_config)
		self.model_config = model_config
		self.dtype = DTYPE_MAP.get(model_info.get("default_dtype", "float32"))
		self.input_dtype = model_info.get("input_dtype", "float32")
		self.sta_mean = model_info.get("state_mean", 0.)
		self.sta_std = model_info.get("state_std", 255.)

		self._transform = partial(state_transform_ms,
								  mean=self.sta_mean,
								  std=self.sta_std,
								  input_dtype=self.input_dtype)

		self.state_dim = model_info["state_dim"]
		self.action_dim = model_info["action_dim"]
		self.filter_arch = get_atari_filter(self.state_dim)

		# lr schedule with linear_cosine_decay
		self.lr_schedule = model_config.get("lr_schedule", None)
		self.opt_type = model_config.get("opt_type", "adam")
		self.net = ImpalaCnnOptNet(state_dim=self.state_dim, action_dim=self.action_dim)
		self.lr = None

		self.ph_state = None
		self.ph_adv = None
		self.out_actions = None
		self.pi_logic_outs, self.baseline = None, None

		# placeholder for behavior policy logic outputs
		self.ph_bp_logic_outs = None
		self.ph_actions = None
		self.ph_dones = None
		self.ph_rewards = None
		self.loss, self.optimizer, self.train_op = None, None, None

		self.grad_norm_clip = model_config.get("grad_norm_clip", 40.0)

		self.saver = None
		self.explore_paras = None
		self.actor_var = None  # store weights for agent

		super().__init__(model_info)

	def create_model(self, model_info):
		# create learner and split

		loss_fn = ImpalaCnnOptMSLoss(self.model_config)
		
		global_step = ms.Parameter(0, name="global_step", requires_grad=False)
		if self.opt_type == "adam":
			if self.lr_schedule:
				learning_rate = self._get_lr(global_step)
			else:
				learning_rate = LR
			optimizer = Adam(params=self.net.trainable_params(), learning_rate=learning_rate)
		elif self.opt_type == "rmsprop":
			optimizer = nn.RMSProp(learning_rate=LR, decay=0.99, epsilon=0.1, centered=True)
		else:
			raise KeyError("invalid opt_type: {}".format(self.opt_type))

		loss_net = MyWithLossCell(self.net, loss_fn)
		model = TrainOneStepCell(loss_net, optimizer)
		self.model = model
		self.actor_var = MSVariables(self.net)
		return model

	def train(self, state, label):
		"""Train with sess.run."""
		bp_logic_outs, actions, dones, rewards = label
		ph_state = ms.Tensor(state, dtype=ms.float32)
		self.ph_state = ph_state
		ph_bp_logic_outs = ms.Tensor(bp_logic_outs, dtype=ms.float32)
		ph_actions = ms.Tensor(actions, dtype=ms.float32)
		ph_dones = ms.Tensor(dones, dtype=ms.float32)
		ph_rewards = ms.Tensor(rewards, dtype=ms.float32)
		loss = self.model(ph_state, ph_bp_logic_outs, ph_actions, ph_dones, ph_rewards)
		return loss.asnumpy()

	def predict(self, state):
		state = Tensor(state, dtype=ms.float32)
		pi_logic_outs, baseline, out_actions = self.net(state)
		pi_logic_outs = pi_logic_outs.asnumpy()
		baseline = baseline.asnumpy()
		out_actions = out_actions.asnumpy()
		return [pi_logic_outs, baseline, out_actions]

	def _get_lr(self, global_step, decay_step=20000.):
		"""Make decay learning rate."""
		lr_schedule = self.lr_schedule
		if len(lr_schedule) != 2:
			logging.warning("Need 2 elements in lr_schedule!\n, "
							"likes [[0, 0.01], [20000, 0.000001]]")
			logging.fatal("lr_schedule invalid: {}".format(lr_schedule))

		if lr_schedule[0][0] != 0:
			logging.info("lr_schedule[0][1] could been init learning rate")

		learning_rate = nn.CosineDecayLR(min_lr=lr_schedule[0][1], max_lr=lr_schedule[0][1], decay_steps=decay_step)(
			global_step)

		return learning_rate


def calc_baseline_loss(advantages):
	"""Calculate the baseline loss."""
	op = ms.ops.ReduceSum()
	return 0.5 * op(ms.ops.square(advantages))


def calc_entropy_loss(logic_outs):
	"""Calculate entropy loss."""
	op = ms.nn.Softmax()
	pi = op(logic_outs)
	op = ms.nn.LogSoftmax()
	log_pi = op(logic_outs)
	op = ms.ops.ReduceSum()
	entropy_per_step = op(-pi * log_pi, axis=-1)
	return -op(entropy_per_step)


def calc_pi_loss(logic_outs, actions, advantages):
	"""Calculate policy gradient loss."""
	op = ms.nn.SoftmaxCrossEntropyWithLogits()
	cross_entropy = op(logits=logic_outs, labels=actions)
	advantages = ms.ops.stop_gradient(advantages)
	pg_loss_per_step = cross_entropy * advantages
	op = ms.ops.ReduceSum()
	return op(pg_loss_per_step)


def vtrace_loss(
		bp_logic_outs, tp_logic_outs, actions,
		discounts, rewards, values, bootstrap_value):
	value_of_state, pg_advantages = vtrace.from_logic_outputs(
		behaviour_policy_logic_outputs=bp_logic_outs,
		target_policy_logic_outputs=tp_logic_outs,
		actions=actions,
		discounts=discounts,
		rewards=rewards,
		values=values,
		bootstrap_value=bootstrap_value,
	)

	pi_loss = calc_pi_loss(tp_logic_outs, actions, pg_advantages)
	val_loss = calc_baseline_loss(value_of_state - values)
	entropy_loss = calc_entropy_loss(tp_logic_outs)

	return pi_loss + 0.5 * val_loss + 0.01 * entropy_loss


class ImpalaCnnOptNet(Cell):
	def __init__(self, **descript):
		super(ImpalaCnnOptNet, self).__init__()
		self.state_dim = descript.get("state_dim")
		self.action_dim = descript.get("action_dim")
		self.filter_arch = get_atari_filter(self.state_dim)

		(out_size, kernel, stride) = self.filter_arch[-1]
		self.conv_layer2 = nn.Conv2d(
			in_channels=32,
			out_channels=out_size,
			kernel_size=(kernel, kernel),
			stride=(stride, stride),
			pad_mode="valid",
		)
		self.relu = ReLU()
		self.conv_layer3 = nn.Conv2d(
			in_channels=out_size,
			out_channels=self.action_dim,
			kernel_size=(1, 1),
			pad_mode="same",
		)
		self.flattenlayer = Flatten()

	def construct(self, state_input):
		last_layer = ms.ops.Transpose()(state_input, (0, 3, 1, 2))
		i = 0
		outSize = 0
		for (out_size, kernel, stride) in self.filter_arch[:-1]:
			if i == 0:
				inChannels = 4
			else:
				inChannels = outSize
			print("i:{}", i, " inChannels: {}", inChannels)

			last_layer = nn.Conv2d(
				in_channels=inChannels,
				out_channels=out_size,
				kernel_size=(kernel, kernel),
				stride=(stride, stride),
				pad_mode="same",
			)(last_layer)
			outSize = out_size
			print("-----last_layer.shape-----", last_layer.shape)
			last_layer = self.relu(last_layer)
			i += 1

		convolution_layer = self.conv_layer2(last_layer)
		convolution_layer = self.relu(convolution_layer)

		convolution_layer = self.conv_layer3(convolution_layer)
		squeeze = ops.Squeeze((2, 3))
		pi_logic_outs = squeeze(convolution_layer)
		baseline_flat = self.flattenlayer(convolution_layer)
		
		my_custom_norm_initializer_ms = custom_norm_initializer_ms(0.01)
		denselayer = nn.Dense(
			# in_channels=256,
			in_channels=9,
			out_channels=1,
			activation=None,
			weight_init=my_custom_norm_initializer_ms(shape=(1, 9))
		)
		baseline = denselayer(baseline_flat)
		squeeze2 = ops.Squeeze(1)
		baseline = squeeze2(baseline)

		multiform = ops.multinomial(pi_logic_outs, num_sample=1)
		out_actions = squeeze2(multiform)
		return (pi_logic_outs, baseline, out_actions)


class ImpalaCnnOptMSLoss(nn.LossBase):
	def __init__(self, model_config):
		super(ImpalaCnnOptMSLoss, self).__init__()
		self.sample_batch_steps = model_config.get("sample_batch_step", 50)

	def split_batches(self, tensor, drop_last=False):
		batch_step = self.sample_batch_steps
		batch_count = ops.shape(tensor)[0] // batch_step
		reshape_tensor = ops.Reshape()(
			tensor,
			ops.concat((ms.Tensor([batch_count, batch_step]), ms.Tensor(ops.shape(tensor)[1:])), axis=0),
		)

		# swap B and T axes
		s1 = ms.Tensor(ops.shape(tensor))
		res = ops.Transpose()(
			reshape_tensor,
			(1, 0) + tuple(range(2, 1 + int(ops.shape(s1)[0]))),
		)

		if drop_last:
			return res[:-1]
		return res

	def construct(self, input, out):
		pi_logic_outs, baseline, _ = out
		ph_bp_logic_outs, ph_actions, ph_dones, ph_rewards = input
		loss = vtrace_loss(
			bp_logic_outs=self.split_batches(ph_bp_logic_outs, drop_last=True),
			tp_logic_outs=self.split_batches(pi_logic_outs, drop_last=True),
			actions=self.split_batches(ph_actions, drop_last=True),
			discounts=self.split_batches(ops.cast(ph_dones, ms.float32) * GAMMA, drop_last=True),
			rewards=self.split_batches(ops.clip_by_value(ph_rewards, ms.Tensor(-1), ms.Tensor(1)), drop_last=True),
			values=self.split_batches(baseline, drop_last=True),
			bootstrap_value=self.split_batches(baseline)[-1],
		)
		return loss


class MyWithLossCell(Cell):
	def __init__(self, backbone, loss_fn):
		super(MyWithLossCell, self).__init__(auto_prefix=False)
		self._backbone = backbone
		self._loss_fn = loss_fn

	def construct(self, state_input, ph_bp_logic_outs, ph_actions, ph_dones, ph_rewards):
		label = (ph_bp_logic_outs, ph_actions, ph_dones, ph_rewards)
		out = self._backbone(state_input)
		return self._loss_fn(out, label)
