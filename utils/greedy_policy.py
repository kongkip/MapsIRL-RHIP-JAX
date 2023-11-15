def greedy_policy(reward_function, value_function, state, actions, graph):
    """
    Compute the greedy deterministic policy.

    Args:
    - reward_function: A function that takes a state and action and returns the reward.
    - value_function: A dictionary mapping states to their value.
    - state: The current state (location/intersection).
    - actions: A list of possible actions (directions/roads) from the current state.
    - graph: The graph representation of the map.

    Returns:
    - The action that maximizes the reward plus the value of the next state.
    """
    best_action = None
    max_value = float('-inf')

    for action in actions:
        # Determine the next state based on the current state and action
        next_state = graph[state][action]['next_node']
        value = reward_function(state, action) + value_function[next_state]

        if value > max_value:
            max_value = value
            best_action = action

    return best_action

# Example usage
# reward_function = ...  # Define your reward function
# value_function = ...  # Define your value function mapping states to values
# state = ...  # Current state (location/intersection)
# actions = ...  # Possible actions (directions/roads) from the current state
# graph = ...  # The graph representation of the map
# best_action = greedy_policy(reward_function, value_function, state, actions, graph)
