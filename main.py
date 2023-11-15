from rhip import RHIP


def main():
    # Example graph (simplified representation of a map)
    graph = {
        'A': {'B': {'cost': 1, 'next_node': 'B'}},
        'B': {'C': {'cost': 1, 'next_node': 'C'}, 'D': {'cost': 2, 'next_node': 'D'}},
        'C': {'D': {'cost': 1, 'next_node': 'D'}},
        'D': {}
    }

    # States (intersections)
    states = ['A', 'B', 'C', 'D']

    # Actions (possible roads from each intersection)
    actions = {
        'A': ['B'],
        'B': ['C', 'D'],
        'C': ['D'],
        'D': []
    }

    # Start state and destination
    start_state = 'A'
    destination = 'D'

    # Expert demonstrations (simplified as a list of state-action pairs)
    demonstrations = [('A', 'B'), ('B', 'C'), ('C', 'D')]

    # Initial reward function (simplified as a constant value for each state-action pair)
    reward_function = lambda state, action: -graph[state][action]['cost']

    # Horizon for the RHIP algorithm
    H = 2

    # Run RHIP algorithm
    updated_reward_function = RHIP(graph, states, actions, start_state, destination, demonstrations, reward_function, H)

    print("Updated Reward Function:", updated_reward_function)


if __name__ == "__main__":
    main()
