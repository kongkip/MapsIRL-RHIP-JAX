import jax
import jax.numpy as jnp


def update_q_value_function(reward_function, value_function, states, actions, graph, h):
    """
    Update the Q-function and value function for each state and action.

    Args:
    - reward_function: A function that takes a state and action and returns the reward.
    - value_function: A dictionary mapping states to their value.
    - states: A list of all possible states (locations/intersections).
    - actions: A dictionary mapping states to a list of possible actions (directions/roads).
    - graph: The graph representation of the map.
    - h: The current horizon step.

    Returns:
    - Updated Q-function and value function.
    """
    q_function = {}
    new_value_function = {}

    for state in states:
        q_values = []
        for action in actions[state]:
            # Assuming the action leads to a directly connected node in the graph
            next_state = graph[state][action]['next_node']
            q_value = reward_function(state, action) + value_function[next_state]
            q_function[(state, action, h)] = q_value
            q_values.append(q_value)

        # Update the value function using log-sum-exp for numerical stability
        new_value_function[state] = jax.scipy.special.logsumexp(jnp.array(q_values))

    return q_function, new_value_function

# Example usage
# reward_function = ...  # Define your reward function
# value_function = ...  # Initial value function from the Dijkstra's algorithm
# states = ...  # List of all states (locations/intersections)
# actions = ...  # Dictionary mapping states to possible actions (directions/roads)
# graph = ...  # The graph representation of the map
# h = ...  # Current horizon step
# q_function, new_value_function = update_q_value_function(reward_function, value_function, states, actions, graph, h)
