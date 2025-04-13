from tictactoe import *


def print_board(board):
    for row in board:
        print(row)
    print()

def play_game():
    board = initial_state()
    
    while not terminal(board):
        
        print_board(board)
        if player(board) == X:
            print("Player X's turn")
            action = minimax(board)
        else:
            print("Player O's turn")
            action = minimax(board)
        
        board = result(board, action)
    
    print_board(board)
    winner_player = winner(board)
    if winner_player:
        print(f"Winner: {winner_player}")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    play_game()