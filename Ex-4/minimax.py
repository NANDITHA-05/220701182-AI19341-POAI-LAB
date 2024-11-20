import math

# Function to evaluate the board state
def evaluate(board):
    # Check rows, columns, and diagonals for a winner
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != "_":
            return 1 if row[0] == "O" else -1

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != "_":
            return 1 if board[0][col] == "O" else -1

    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != "_":
        return 1 if board[0][0] == "O" else -1

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != "_":
        return 1 if board[0][2] == "O" else -1

    return 0  # No winner yet

# Check if there are moves left
def is_moves_left(board):
    for row in board:
        if "_" in row:
            return True
    return False

# Minimax algorithm
def minimax(board, depth, is_maximizing):
    score = evaluate(board)

    # If computer wins
    if score == 1:
        return score

    # If player wins
    if score == -1:
        return score

    # If no moves are left, it's a tie
    if not is_moves_left(board):
        return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == "_":
                    board[i][j] = "O"  # Computer's move
                    best_score = max(best_score, minimax(board, depth + 1, False))
                    board[i][j] = "_"  # Undo move
        return best_score
    else:
        best_score = math.inf
        for i in range(3):
            for j in range(3):
                if board[i][j] == "_":
                    board[i][j] = "X"  # Player's move
                    best_score = min(best_score, minimax(board, depth + 1, True))
                    board[i][j] = "_"  # Undo move
        return best_score

# Function to find the best move for the computer
def find_best_move(board):
    best_val = -math.inf
    best_move = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == "_":
                board[i][j] = "O"  # Try the move
                move_val = minimax(board, 0, False)
                board[i][j] = "_"  # Undo the move

                if move_val > best_val:
                    best_val = move_val
                    best_move = (i, j)

    return best_move

# Main game loop
def print_board(board):
    for row in board:
        print(" ".join(row))
    print()

def main():
    board = [["_"] * 3 for _ in range(3)]
    print("Welcome to Tic-Tac-Toe!")
    print_board(board)

    while is_moves_left(board):
        # Player's move
        x, y = map(int, input("Enter your move (row and column): ").split())
        if board[x][y] != "_":
            print("Invalid move. Try again.")
            continue
        board[x][y] = "X"

        # Check if the player won
        if evaluate(board) == -1:
            print("You win!")
            print_board(board)
            return

        # Computer's move
        if is_moves_left(board):
            print("Computer's turn...")
            move = find_best_move(board)
            board[move[0]][move[1]] = "O"

            # Check if the computer won
            if evaluate(board) == 1:
                print("Computer wins!")
                print_board(board)
                return

        print_board(board)

    print("It's a tie!")

# Start the game
main()
