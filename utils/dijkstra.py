from heapq import heappush, heappop


def dijkstra(graph, start, end, reward_function):
    """
    Compute the shortest path using Dijkstra's algorithm.
    
    Args:
    - graph: A representation of the graph where graph[u][v] is the cost from node u to v.
    - start: The starting node.
    - end: The destination node.
    - reward_function: A function that takes an edge and returns its reward.

    Returns:
    - The cost of the shortest path from start to end.
    """
    # Initialize distances and priority queue
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_vertex = heappop(priority_queue)

        # If reached the destination node
        if current_vertex == end:
            break

        # If a shorter path to a neighbor is found
        for neighbor, weight in graph[current_vertex].items():
            # Assuming 'current_vertex' represents the current state and 'weight' the action
            distance = current_distance + reward_function(current_vertex, weight)

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heappush(priority_queue, (distance, neighbor))

    return distances[end]

# Example of how to use the dijkstra function
# graph = ... # Define your graph
# start = ... # Starting node
# end = ... # Destination node
# reward_function = lambda weight: -weight # Example reward function
# shortest_path_cost = dijkstra(graph, start, end, reward_function)
