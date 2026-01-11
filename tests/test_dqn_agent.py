import numpy as np

from rl.dqn import DQNAgent


def test_dqn_select_action_masks_unavailable():
    agent = DQNAgent(
        state_dim=4,
        num_actions=3,
        hidden_sizes=[16],
        epsilon_start=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        train_start=1,
        batch_size=1,
    )

    state = np.zeros(4, dtype=np.float32)
    available = [1]
    action = agent.select_action(state, available)
    assert action == 1


def test_dqn_train_step_runs():
    agent = DQNAgent(
        state_dim=4,
        num_actions=2,
        hidden_sizes=[16],
        epsilon_start=0.0,
        epsilon_min=0.0,
        epsilon_decay=1.0,
        buffer_size=10,
        batch_size=1,
        train_start=1,
        target_update_freq=1,
    )

    state = np.zeros(4, dtype=np.float32)
    next_state = np.ones(4, dtype=np.float32)
    agent.remember(state, 0, 1.0, next_state, False, [0, 1])
    loss = agent.train_step()
    assert loss is not None