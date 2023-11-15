def rollout(combined_policy, start_state, graph, max_steps=100):
    """
    Perform a roll-out using the combined policy to compute state-action visitation frequencies.

    Args:
    - combined_policy: A policy dict representing the combined policy.
    - start_state: The starting state for the roll-out.
    - graph: The graph representation of the map.
    - max_steps: Maximum number of steps in the roll-out.

    Returns:
    - A dictionary of state-action visitation frequencies.
    """
    visitation_frequencies = {}
    current_state = start_state

    for step in range(max_steps):
        actions = combined_policy.get(current_state, {})
        if not actions:
            break

        # Select an action based on the policy's probabilities
        action = np.random.choice(list(actions.keys()), p=list(actions.values()))
        visitation_frequencies[(current_state, action)] = visitation_frequencies.get((current_state, action), 0) + 1

        # Move to the next state
        current_state = graph[current_state][action]['next_node']

    return visitation_frequencies

# Example usage
# combined_policy = ...  # Combined policy from the previous step
# start_state = ...  # Starting state for the roll-out
# graph = ...  # The graph representation of the map
# visitation_frequencies = rollout(combined_policy, start_state, graph)
