from heapq import heappop, heappush

class Node:
    def __init__(self, x, y, cost=0, heuristic=0):
        self.x = x
        self.y = y
        self.cost = cost
        self.heuristic = heuristic
        self.total_cost = cost + heuristic
        self.parent = None

    def __lt__(self, other):
        return self.total_cost < other.total_cost

def heuristic(a, b):
   
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, end):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heappush(open_set, Node(start[0], start[1], 0, heuristic(start, end)))
    closed_set = set()

    while open_set:
        current = heappop(open_set)
        if (current.x, current.y) == end:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1] 
        closed_set.add((current.x, current.y))

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  
            nx, ny = current.x + dx, current.y + dy

            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                if (nx, ny) in closed_set:
                    continue

                new_cost = current.cost + 1
                neighbor = Node(nx, ny, new_cost, heuristic((nx, ny), end))
                neighbor.parent = current
                heappush(open_set, neighbor)

    return None 


if __name__ == "__main__":
    grid = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0]
    ]
    start = (0, 0)
    end = (4, 4)
    path = a_star(grid, start, end)

    if path:
        print("Path found:", path)
    else:
        print("No path found.")
