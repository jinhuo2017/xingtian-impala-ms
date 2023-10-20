from __future__ import absolute_import, division, print_function

from xt.model.ms_compat import ms

def assert_has_rank(self, rank):
    if (len(self.shape) != rank):
        raise ValueError("Shape %s must have rank %d" % (self.shape, rank))

def from_logic_outputs(behaviour_policy_logic_outputs,
                       target_policy_logic_outputs,
                       actions,
                       discounts,
                       rewards,
                       values,
                       bootstrap_value,
                       clip_importance_sampling_threshold=1.0,
                       clip_pg_importance_sampling_threshold=1.0):
    """
    Calculate vtrace with logic outputs.

    :param behaviour_policy_logic_outputs: behaviour_policy_logic_outputs
    :param target_policy_logic_outputs: target_policy_logic_outputs
    :param actions:
    :param discounts:
    :param rewards:
    :param values:
    :param bootstrap_value:
    :param clip_importance_sampling_threshold:
    :param clip_pg_importance_sampling_threshold:
    :return:
    """

    behaviour_policy_logic_outputs = ms.Tensor(behaviour_policy_logic_outputs, dtype=ms.float32)
    target_policy_logic_outputs = ms.Tensor(target_policy_logic_outputs, dtype=ms.float32)
    actions = ms.Tensor(actions, dtype=ms.int32)


    assert_has_rank(behaviour_policy_logic_outputs, 3)
    assert_has_rank(target_policy_logic_outputs, 3)
    assert_has_rank(actions, 3)

    target_log_prob = -ms.ops.SoftmaxCrossEntropyWithLogits(logits=target_policy_logic_outputs, labels=actions)

    behaviour_log_prob = -ms.ops.SoftmaxCrossEntropyWithLogits(logits=behaviour_policy_logic_outputs, labels=actions)

    # log importance sampling weight
    importance_sampling_weights = ms.ops.exp(target_log_prob - behaviour_log_prob)

    clipped_importance_sampling_weight = ms.ops.minimum(clip_importance_sampling_threshold, importance_sampling_weights)
    clipped_pg_importance_sampling_weight = ms.ops.minimum(clip_pg_importance_sampling_threshold, importance_sampling_weights)

    coefficient = ms.ops.minimum(1.0, importance_sampling_weights)
    
    next_values = ms.ops.concat([values[1:], ms.ops.expand_dims(bootstrap_value, 0)], axis=0)

    deltas = clipped_importance_sampling_weight * (rewards + discounts * next_values - values)
    sequences = (deltas, discounts, coefficient)

    # calculate Vtrace with tf.scan, and set reverse: True, back --> begin
    def scan_fn(cumulative_value, sequence_item):
        _delta, _discount, _coefficient = sequence_item
        return _delta + _discount * _coefficient * cumulative_value

    last_values = ms.ops.ZerosLike(bootstrap_value)
    
    temporal_difference = ms.ops.DynamicRNN(
        cell_fn=scan_fn
    )

    value_of_states = ms.ops.add(temporal_difference, values)


    # Advantage for policy gradient.
    value_of_next_state = ms.ops.concat([value_of_states[1:], ms.ops.ExpandDims(bootstrap_value, 0)], axis=0)
    pg_advantages = clipped_pg_importance_sampling_weight * (rewards + discounts * value_of_next_state - values)

    value_of_states = ms.ops.stop_gradient(value_of_states)
    pg_advantages = ms.ops.stop_gradient(pg_advantages)
    return value_of_states, pg_advantages
