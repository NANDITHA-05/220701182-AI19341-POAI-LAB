def dfs_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()

    # Mark the current node as visited
    visited.add(node)
    print(node, end=" ")

    # Recur for all adjacent nodes
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

# Example graph as an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': [],
    'E': ['F'],
    'F': []
}

# Call the recursive DFS function
print("Recursive DFS traversal:")
dfs_recursive(graph, 'A')
