"""
Tic Tac Toe Player
"""

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
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return O if x_count > o_count else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    return {(r, c) for r, row in enumerate(board) for c, cell in enumerate(row) if cell is EMPTY}


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    r, c = action
    if board[r][c] is EMPTY:
        new_board = [row.copy() for row in board]
        new_board[r][c] = player(board)
        return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for player in [X, O]:
        # Check rows
        for row in board:
            if all(cell == player for cell in row):
                return player
        # Check columns
        for c in range(3):
            if all(row[c] == player for row in board):
                return player
        # Check diagonals
        if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
            return player

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    return True if winner(board) is not None or not actions(board) else False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    player = winner(board)
    if player == X:
        return 1
    elif player == O:
        return -1
    else:
        return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    def max_value(board):
        if terminal(board):
            return utility(board), None
        v = -math.inf
        for action in actions(board):
            min_val, _ = min_value(result(board, action))
            if min_val > v:
                v = min_val
                best_action = action
        return v, best_action

    def min_value(board):
        if terminal(board):
            return utility(board), None
        v = math.inf
        for action in actions(board):
            max_val, _ = max_value(result(board, action))
            if max_val < v:
                v = max_val
                best_action = action
        return v, best_action

    _, action = max_value(board) if player(board) == X else min_value(board)

    return action
