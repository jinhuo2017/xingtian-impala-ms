import mindspore.ops as ops
from xt.model.ms_compat import ms
from xt.model.ms_compat import Cell, Tensor, Dense, MSELoss, Adam, Conv2d, WithLossCell, ReLU, Flatten, \
    DynamicLossScaleUpdateCell, MultitypeFuncGraph, Cast
from xt.model.impala.default_config import ENTROPY_LOSS, LR
from xt.model.model_ms import XTModel_MS
from xt.model.ms_utils import MSVariables
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers


@Registers.model
class ImpalaCnnMS(XTModel_MS):
    """Create model for ImpalaNetworkCnn."""

    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.net = ImpalaCnnNet(state_dim=self.state_dim, action_dim=self.action_dim)
        super().__init__(model_info)

    def create_model(self, model_info):
        decay_value = 0.00000000512
        loss_fn = MSELoss()
        adam = Adam(params=self.net.trainable_params(), learning_rate=LR, use_amsgrad=True, weight_decay=decay_value)
        loss_net = WithLossCell(self.net, loss_fn)
        device_target = ms.get_context("device_target")
        if device_target == 'Ascend':
            manager = DynamicLossScaleUpdateCell(loss_scale_value=2 ** 12, scale_factor=2, scale_window=1000)
            model = MyTrainOneStepCell(loss_net, adam, manager, grad_clip=True, clipnorm=10.)
        else:
            model = MyTrainOneStepCell(loss_net, adam, grad_clip=True, clipnorm=10.)
        self.actor_var = MSVariables(self.net)
        return model

    def predict(self, state):
        state[0] = Tensor(state[0], dtype=ms.float32)
        state[1] = Tensor(state[1], dtype=ms.float32)
        return self.net(state).asnumpy()


class ImpalaCnnNet(Cell):
    def __init__(self, **descript):
        super(ImpalaCnnNet, self).__init__()
        self.state_dim = descript.get("state_dim")
        action_dim = descript.get("action_dim")
        self.convlayer1 = Conv2d(self.state_dim[2], 32, kernel_size=8, stride=4, pad_mode='valid',
                                 weight_init="xavier_uniform")
        self.convlayer2 = Conv2d(32, 64, kernel_size=4, stride=2, pad_mode='valid', weight_init="xavier_uniform")
        self.convlayer3 = Conv2d(64, 64, kernel_size=3, stride=1, pad_mode='valid', weight_init="xavier_uniform")
        self.relu = ReLU()
        self.flattenlayer = Flatten()
        _dim = (
                (((self.state_dim[0] - 4) // 4 - 2) // 2 - 2)
                * (((self.state_dim[1] - 4) // 4 - 2) // 2 - 2)
                * 64
        )
        self.denselayer1 = Dense(_dim, 256, activation='relu', weight_init="xavier_uniform")
        self.denselayer2 = Dense(256, action_dim, weight_init="xavier_uniform")
        self.denselayer3 = Dense(256, 1, weight_init="xavier_uniform")

    def construct(self, x):
        out = Cast()(x.transpose((0, 3, 1, 2)), ms.float32) / 255.  # 自定义输入
        out = self.convlayer1(out)
        out = self.relu(out)
        out = self.convlayer2(out)
        out = self.relu(out)
        out = self.convlayer3(out)
        out = self.relu(out)
        out = self.flattenlayer(out)
        out = self.denselayer1(out)
        out_actions = self.denselayer2(out)
        out_value = self.denselayer3(out)
        return out_actions, out_value


_grad_scale = MultitypeFuncGraph("grad_scale")


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(ops.Reciprocal()(scale), ops.dtype(grad))


class MyTrainOneStepCell(ms.nn.TrainOneStepWithLossScaleCell):
    def __init__(self, network, optimizer, scale_sense=1, grad_clip=False, clipnorm=1.):
        self.clipnorm = clipnorm
        if isinstance(scale_sense, (int, float)):
            scale_sense = Tensor(scale_sense, dtype=ms.float32)
        super(MyTrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.grad_clip = grad_clip

    def construct(self, state, label):
        weights = self.weights
        loss = self.network(state, label)
        scaling_sens = self.scale_sense
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        scaling_sens_filled = ops.ones_like(loss) * ops.cast(scaling_sens, ops.dtype(loss))
        grads = self.grad(self.network, weights)(state, label, scaling_sens_filled)
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads, self.clipnorm)
        grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        if not overflow:
            loss = ops.depend(loss, self.optimizer(grads))
        return loss
