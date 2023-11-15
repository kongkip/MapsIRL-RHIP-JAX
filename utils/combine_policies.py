def combine_policies(stochastic_policy, deterministic_policy, states, H):
    """
    Combine the stochastic and deterministic policies over the receding horizon.

    Args:
    - stochastic_policy: A policy dict for the stochastic policy.
    - deterministic_policy: A policy dict for the deterministic policy.
    - states: A list of all possible states.
    - H: Horizon for switching from stochastic to deterministic policy.

    Returns:
    - A combined policy dict.
    """
    combined_policy = {}

    for state in states:
        if state in stochastic_policy and state in deterministic_policy:
            # Use stochastic policy for the first H steps, then deterministic policy
            combined_policy[state] = {
                action: (stochastic_policy[state][action] if h < H else deterministic_policy[state][action])
                for action, h in enumerate(range(H + 1))}
        else:
            combined_policy[state] = deterministic_policy.get(state, stochastic_policy.get(state, {}))

    return combined_policy

# Example usage
# stochastic_policy = ...  # Stochastic policy from the previous step
# deterministic_policy = ...  # Deterministic policy from an earlier step
# states = ...  # List of all states
# H = ...  # Horizon for switching policies
# combined_policy = combine_policies(stochastic_policy, deterministic_policy, states, H)
