"""
Tic Tac Toe Player
"""

import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # Count the number of Xs and Os on the board, and whichever is less is the next player
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return X if x_count <= o_count else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    return {(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY}


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    # Check if the action is valid. The move needs to be empty
    # and on the board
    if not (0 <= action[0] < 3) or not (0 <= action[1] < 3):
        raise ValueError("This action is out of bounds")

    if board[action[0]][action[1]] != EMPTY:
        raise ValueError("The action is invalid because cell already taken")
    
    # Use a try, in case there are other invalid actions
    try:
        # Make a deep copy of the board SO it doesn't change the original board
        new_board = copy.deepcopy(board)
        new_board[action[0]][action[1]] = player(new_board)
    except:
        raise ValueError("Invalid action")

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows - if the count of either rows is 3, then we have a winner
    for row in board:
        if row.count(X) == 3: return X
        if row.count(O) == 3: return O

    # Check columns - if the count of either columns is 3, then we have a winner
    for j in range(3): # Start on first column
        if [board[i][j] for i in range(3)].count(X) == 3: return X # Counts Xs in each column to see if any has 3
        if [board[i][j] for i in range(3)].count(O) == 3: return O

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != EMPTY:
        return board[0][0] # This will have X if X wins, and O if O wins
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != EMPTY:
        return board[0][2]
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Checks if there is a winner or if the board is full
    return (winner(board) != None or all(cell != EMPTY for row in board for cell in row))


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_player = winner(board)
    
    if winner_player == X: return 1
    elif winner_player == O: return -1
    else: return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.

    """

    if terminal(board): return None

    current_player = player(board)

    # Set current value to abs(2) to ensure the first action will replace it
    # Alpha holds the current highest value for X, and beta for O
    alpha = -2
    beta = 2

    for action in actions(board):
        # Alpha and beta is passed on to the next level, so pruning can be done
        value = simulate_game(result(board, action), alpha, beta)
        if (value > alpha and current_player == X):
            alpha = value
            rec_action = action
        elif (value < beta and current_player == O):
            beta = value
            rec_action = action

    return rec_action


def simulate_game(board, alpha, beta):
    """
    Returns the minimax value of a board for the current player.
    """
    if terminal(board): return utility(board)

    current_player = player(board)

    # Set current value to abs(2) to ensure any action will be better and update the choice
    top_value = -2 if current_player == X else 2 


    for action in actions(board):
        # Loop through this function until the game is completed
        value = simulate_game(result(board, action), alpha, beta)

        if (value >= top_value and current_player == X): 
            top_value = value
            if top_value > alpha: alpha = top_value
        elif (value <= top_value and current_player == O):
            top_value = value
            if top_value < beta: beta = top_value
        
        # If this branch has lower O values than the currently highest X value,
        # then it will never be chosen, so we can prune it
        if alpha >= beta: break
    return top_value













