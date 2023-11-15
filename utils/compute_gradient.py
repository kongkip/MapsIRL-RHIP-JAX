def compute_gradient(desired_visitation_freq, obtained_visitation_freq, reward_function, learning_rate=0.01):
    """
    Compute the gradient for updating the reward function parameters.

    Args:
    - desired_visitation_freq: The desired state-action visitation frequencies (from demonstrations).
    - obtained_visitation_freq: The obtained state-action visitation frequencies (from roll-out).
    - reward_function: The reward function to update.
    - learning_rate: The learning rate for the update.

    Returns:
    - Updated reward function.
    """
    for state_action, freq in desired_visitation_freq.items():
        obtained_freq = obtained_visitation_freq.get(state_action, 0)
        gradient = freq - obtained_freq
        # Update the reward function parameters based on the gradient
        # This depends on how reward_function is represented and how its parameters are structured
        # For example:
        reward_function.params[state_action] += learning_rate * gradient

    return reward_function

# Example usage
# desired_visitation_freq = ...  # Desired visitation frequencies from the demonstration
# obtained_visitation_freq = ...  # Obtained visitation frequencies from the roll-out
# reward_function = ...  # The current reward function
# updated_reward_function = compute_gradient(desired_visitation_freq, obtained_visitation_freq, reward_function)
