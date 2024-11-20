# Function to check if a queen can be placed on board[row][col]
def is_safe(board, row, col, n):
    # Check this column on upper rows
    for i in range(row):
        if board[i][col] == 1:
            return False
    
    # Check upper left diagonal
    for i, j in zip(range(row, -1, -1), range(col, -1, -1)):
        if board[i][j] == 1:
            return False

    # Check upper right diagonal
    for i, j in zip(range(row, -1, -1), range(col, n)):
        if board[i][j] == 1:
            return False

    return True

# Recursive utility function to solve the problem
def solve_n_queens(board, row, n):
    # If all queens are placed, return True
    if row >= n:
        return True

    # Consider each column for the current row
    for col in range(n):
        if is_safe(board, row, col, n):
            # Place the queen
            board[row][col] = 1

            # Recur to place the rest of the queens
            if solve_n_queens(board, row + 1, n):
                return True

            # Backtrack if placing queen here doesn't lead to a solution
            board[row][col] = 0

    return False

# Function to print the solution
def print_solution(board, n):
    for row in board:
        print(" ".join("Q" if x == 1 else "." for x in row))
    print("\n")

# Main function to solve 8-Queens problem
def eight_queens():
    n = 8  # Size of the chessboard
    board = [[0 for _ in range(n)] for _ in range(n)]

    if solve_n_queens(board, 0, n):
        print("One of the possible solutions for the 8-Queens problem is:")
        print_solution(board, n)
    else:
        print("No solution exists.")

# Call the function
eight_queens()
