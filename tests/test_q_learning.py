"""
Tests for Q-Learning agent.
"""
import pytest
import numpy as np
from pathlib import Path
from rl.q_learning import QLearningAgent


@pytest.fixture
def q_agent():
    """Create a Q-Learning agent for testing."""
    return QLearningAgent(num_actions=3)


@pytest.mark.unit
def test_q_learning_initialization(q_agent):
    """Test Q-Learning agent initialization."""
    assert q_agent.num_actions == 3
    assert q_agent.learning_rate > 0
    assert q_agent.discount_factor > 0
    assert q_agent.exploration_rate > 0
    assert isinstance(q_agent.q_table, dict)


@pytest.mark.unit
def test_q_learning_get_q_value(q_agent):
    """Test Q-value retrieval."""
    state = (0, 0, 0)
    action = 0
    
    # Initially should return 0.0
    q_value = q_agent.get_q_value(state, action)
    assert q_value == 0.0
    
    # After setting, should return the value
    q_agent.q_table[state] = np.array([1.0, 2.0, 3.0])
    q_value = q_agent.get_q_value(state, action)
    assert q_value == 1.0


@pytest.mark.unit
def test_q_learning_action_selection_exploitation(q_agent):
    """Test action selection in exploitation mode."""
    state = (0, 0, 0)
    
    # Set Q-values so action 1 is best
    q_agent.q_table[state] = np.array([1.0, 10.0, 2.0])
    q_agent.exploration_rate = 0.0  # No exploration
    
    # Should select action 1 (best Q-value)
    action = q_agent.select_action(state)
    assert action == 1


@pytest.mark.unit
def test_q_learning_update(q_agent):
    """Test Q-value update using Bellman equation."""
    state = (0, 0, 0)
    action = 0
    reward = 10.0
    next_state = (1, 0, 0)
    
    # Initialize Q-values
    q_agent.q_table[state] = np.array([0.0, 0.0, 0.0])
    q_agent.q_table[next_state] = np.array([5.0, 0.0, 0.0])
    
    # Update Q-value
    q_agent.update(state, action, reward, next_state)
    
    # Check that Q-value was updated
    new_q = q_agent.get_q_value(state, action)
    assert new_q > 0.0
    assert new_q != 0.0


@pytest.mark.unit
def test_q_learning_reward_calculation(q_agent):
    """Test reward calculation."""
    # Successful request with low latency
    reward = q_agent.calculate_reward(response_time_ms=50.0, success=True)
    assert reward < 0  # Negative because we want to minimize latency
    assert reward > -100  # Should be scaled
    
    # Failed request
    reward_fail = q_agent.calculate_reward(response_time_ms=0.0, success=False)
    assert reward_fail < -500  # Large penalty


@pytest.mark.unit
def test_q_learning_exploration_decay(q_agent):
    """Test exploration rate decay."""
    initial_epsilon = q_agent.exploration_rate
    
    # Decay multiple times
    for _ in range(10):
        q_agent.decay_exploration()
    
    assert q_agent.exploration_rate <= initial_epsilon
    assert q_agent.exploration_rate >= q_agent.min_exploration_rate
    assert q_agent.episode_count == 10


@pytest.mark.unit
def test_q_learning_save_load(tmp_path, q_agent):
    """Test saving and loading Q-table."""
    # Train the agent a bit
    state = (0, 0, 0)
    action = 0
    reward = 10.0
    next_state = (1, 0, 0)
    q_agent.update(state, action, reward, next_state)
    q_agent.episode_count = 5
    
    # Save
    filepath = tmp_path / "q_table.pkl"
    q_agent.save_q_table(filepath)
    assert filepath.exists()
    
    # Load
    loaded_agent = QLearningAgent.load_q_table(filepath)
    assert loaded_agent.num_actions == q_agent.num_actions
    assert loaded_agent.episode_count == 5
    assert state in loaded_agent.q_table
    assert np.allclose(
        loaded_agent.get_q_value(state, action),
        q_agent.get_q_value(state, action)
    )
