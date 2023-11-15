import jax
import jax.numpy as jnp


def stochastic_policy(q_function, states, actions, h):
    """
    Calculate the stochastic policy based on Q-values.

    Args:
    - q_function: A dictionary mapping state-action-horizon tuples to Q-values.
    - states: A list of all possible states.
    - actions: A dictionary mapping states to a list of possible actions.
    - h: The current horizon step.

    Returns:
    - A dictionary representing the stochastic policy. 
      The keys are states, and the values are dictionaries mapping actions to probabilities.
    """
    policy = {}

    for state in states:
        q_values = jnp.array([q_function[(state, action, h)] for action in actions[state]])
        probabilities = jax.nn.softmax(q_values)

        policy[state] = {action: prob for action, prob in zip(actions[state], probabilities)}

    return policy

# Example usage
# q_function = ...  # Q-function from the previous step
# states = ...  # List of all states
# actions = ...  # Dictionary mapping states to possible actions
# h = ...  # Current horizon step
# stochastic_policy = stochastic_policy(q_function, states, actions, h)
