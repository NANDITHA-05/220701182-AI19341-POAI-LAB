from collections import deque

# Function to check if the problem can be solved
def can_measure_water(x, y, z):
    if z > x + y:  # If the target amount is greater than the sum of jug capacities
        return False
    if z % gcd(x, y) != 0:  # If z is not a multiple of the GCD of x and y
        return False
    return True

# Function to calculate GCD (Greatest Common Divisor)
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Function to solve the water jug problem
def water_jug_problem(x, y, z):
    if not can_measure_water(x, y, z):
        return "Not possible to measure out the exact amount."

    # Set to keep track of visited states
    visited = set()

    # Queue for BFS: each element is a tuple (amount_in_jug1, amount_in_jug2)
    queue = deque([(0, 0)])
    steps = []

    while queue:
        a, b = queue.popleft()
        
        # If we reach the target amount in either jug, return True
        if a == z or b == z or a + b == z:
            steps.append((a, b))
            return steps

        # If the state has been visited, skip it
        if (a, b) in visited:
            continue

        # Mark the state as visited
        visited.add((a, b))
        steps.append((a, b))

        # Generate all possible states from the current state
        queue.append((x, b))  # Fill Jug1
        queue.append((a, y))  # Fill Jug2
        queue.append((0, b))  # Empty Jug1
        queue.append((a, 0))  # Empty Jug2
        queue.append((max(0, a - (y - b)), min(y, b + a)))  # Pour Jug1 -> Jug2
        queue.append((min(x, a + b), max(0, b - (x - a))))  # Pour Jug2 -> Jug1

    return "No solution found."

# Example Usage
x = 4  # Capacity of Jug1
y = 3  # Capacity of Jug2
z = 2  # Target amount

result = water_jug_problem(x, y, z)
if isinstance(result, list):
    print("Steps to measure the target amount:")
    for step in result:
        print(step)
else:
    print(result)
