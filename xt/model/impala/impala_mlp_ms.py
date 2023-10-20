from mindspore import set_context
from mindspore.nn import Loss
from mindspore.nn.loss.loss import _check_is_tensor
from mindspore.ops import functional as F
from xt.model.ms_compat import Dense, Adam, MSELoss, Cell, Model, DynamicLossScaleUpdateCell, TrainOneStepCell

from xt.model.ms_utils import MSVariables
from xt.model.impala.default_config import ENTROPY_LOSS, HIDDEN_SIZE, LR, NUM_LAYERS
from xt.model.model_ms import XTModel_MS
from zeus.common.util.common import import_config
from zeus.common.util.register import Registers
import mindspore as ms
import mindspore.nn as nn
import numpy as np
from xt.model.impala.impala_cnn_ms import MyTrainOneStepCell

set_context(mode=ms.GRAPH_MODE)


@Registers.model
class ImpalaMlpMS(XTModel_MS):

    def __init__(self, model_info):
        model_config = model_info.get('model_config', None)
        import_config(globals(), model_config)

        self.state_dim = model_info['state_dim']
        self.action_dim = model_info['action_dim']
        self.learning_rate = LR
        self.net = ImpalaMlpNet(state_dim=self.state_dim, action_dim=self.action_dim)
        self.adv = ms.Tensor((1,), dtype=ms.float32)
        self.loss_fn = ImpalaMlpMSLoss(self.adv)

        super().__init__(model_info)

    def create_model(self, model_info):
        """Create mindspore model."""

        self.loss_fn = ImpalaMlpMSLoss(adv=self.adv)

        opt = Adam(params=self.net.trainable_params(), learning_rate=self.learning_rate)
        loss_net = MyWithLossCell(self.net, self.loss_fn)
        model = nn.TrainOneStepCell(loss_net, opt)
        self.model = model
        self.actor_var = MSVariables(self.net)
        return model

    def train(self, state, label):
        state_input = ms.Tensor(state[0], dtype=ms.float32)
        adv = ms.Tensor(state[1], dtype=ms.float32)
        self.adv = adv
        state = [state_input, adv]

        out_actions = ms.Tensor(label[0], dtype=ms.float32)
        out_value = ms.Tensor(label[0], dtype=ms.float32)
        label = [out_actions, out_value]
        loss = self.model(state, label)
        return loss.asnumpy()

    def predict(self, state):
        self.infer_state = ms.Tensor(state[0], dtype=ms.float32)
        self.adv = ms.Tensor(state[1], dtype=ms.float32)
        infer_p, infer_v = self.net([self.infer_state, self.adv])
        infer_p = infer_p.asnumpy()
        infer_v = infer_v.asnumpy()
        return [infer_p, infer_v]


def impala_loss(advantage):
    """Compute loss for IMPALA."""
    def loss(y_true, y_pred):
        policy = y_pred
        log = ms.ops.Log()
        log_policy = log(policy + 1e-10)
        entropy = (-policy * log_policy)
        cross_entropy = (-y_true * log_policy)
        return ms.ops.mean(advantage * cross_entropy - ENTROPY_LOSS * entropy)

    return loss

class MyWithLossCell(Cell):
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, x, label):
        output = self._backbone(x)
        return self._loss_fn(output, label)


class ImpalaMlpNet(Cell):
    def __init__(self, **descript):
        super(ImpalaMlpNet, self).__init__()
        self.state_dim = descript.get("state_dim")
        self.action_dim = descript.get("action_dim")
        self.denselayer1 = Dense(self.state_dim[-1], HIDDEN_SIZE, activation='relu', weight_init='xavier_uniform')
        self.denselayer2 = Dense(HIDDEN_SIZE, HIDDEN_SIZE, activation='relu', weight_init='xavier_uniform')
        self.denselayer3 = Dense(HIDDEN_SIZE, self.action_dim, activation='softmax', weight_init='xavier_uniform')
        self.denselayer4 = Dense(HIDDEN_SIZE, 1, activation='softmax', weight_init='xavier_uniform')

    def construct(self, x):
        out = self.denselayer1(x[0].astype("float32"))
        for _ in range(NUM_LAYERS - 1):
            out = self.denselayer2(out)
        out_actions = self.denselayer3(out)
        out_value = self.denselayer4(out)
        return out_actions, out_value



class MyMSELoss(MSELoss):
    def __init__(self):
        super(MyMSELoss, self).__init__()

    def construct(self, logits, labels):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        x = F.square(logits - labels)
        return self.get_loss(x, weights=.5)


class ImpalaMlpMSLoss(nn.LossBase):
    def __init__(self, adv):
        super(ImpalaMlpMSLoss, self).__init__()
        self.adv = adv

    def construct(self, state, label):
        loss_fn1 = impala_loss(advantage=self.adv)
        output_actions = loss_fn1(state[0], label[0])
        loss_fn2 = MyMSELoss()
        output_value = loss_fn2(state[1], label[1])
        loss = output_actions + output_value
        return loss
