from agent import Agent
from board import Board
import math

class AlphaBetaAgent(Agent):
    
    def __init__(self, player=1, max_depth=3):
        super().__init__(player)
        self.max_depth = max_depth
        
    def next_action(self, obs):
        action, _ = self.alpha_beta(obs, self.player, True, self.max_depth, -math.inf, math.inf)
        return action
    
    def heuristic_utility(self, board: Board):
        total_coins = board.grid.sum()
        sequences = self.count_sequences(board.grid)
        return total_coins + sequences

    def count_sequences(self, grid):
        sequences = 0
        for row in grid:
            in_sequence = False
            for cell in row:
                if cell == 1 and not in_sequence:
                    sequences += 1
                    in_sequence = True
                elif cell == 0:
                    in_sequence = False
        return sequences
    
    def alpha_beta(self, board: Board, player, is_maximizing, depth, alpha= -math.inf, beta= math.inf):
        if board.is_end(player)[0] or depth == 0:
            if board.is_end(player)[0]:
                if board.is_end(player)[1] == self.player:
                    return None, 1 #win
                else:
                    return None, -1 #loss
            return None, self.heuristic_utility(board)

        possible_actions = board.get_possible_actions()
        if is_maximizing:
            best_value = -math.inf
            best_action = None
            for action in possible_actions:
                new_board = board.clone()
                new_board.play(action)
                _, value = self.alpha_beta(new_board, (player % 2) + 1, False, depth - 1, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_action = action
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            return best_action, best_value
        else:
            best_value = math.inf
            best_action = None
            for action in possible_actions:
                new_board = board.clone()
                new_board.play(action)
                _, value = self.alpha_beta(new_board, (player % 2) + 1, True, depth - 1, alpha, beta)
                if value < best_value:
                    best_value = value
                    best_action = action
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
            return best_action, best_value
