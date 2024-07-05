from agent import Agent
from board import Board
import math

class ExpectimaxAgent(Agent):
    
    def __init__(self, player=1, max_depth=3):
        super().__init__(player)
        self.max_depth = max_depth

    def next_action(self, obs):
        action, _ = self.expectimax(obs, self.player, self.max_depth, True)
        return action

    def heuristic_utility(self, board: Board):
        total_coins = board.grid.sum()
        sequences = self.count_sequences(board.grid)
        return -total_coins + sequences

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

    def expectimax(self, board: Board, player, depth, is_maximizing):
        if board.is_end(player)[0] or depth == 0:
            if board.is_end(player)[1] == self.player:
                return None, 1  #win
            elif board.is_end(player)[1] != 0:
                return None, -1  #loss
            return None, self.heuristic_utility(board)

        possible_actions = board.get_possible_actions()
        if is_maximizing:
            best_value = -math.inf
            best_action = None
            for action in possible_actions:
                new_board = board.clone()
                new_board.play(action)
                _, value = self.expectimax(new_board, (player % 2) + 1, depth - 1, False)
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_action, best_value
        else:
            expected_value = 0
            for action in possible_actions:
                new_board = board.clone()
                new_board.play(action)
                _, value = self.expectimax(new_board, (player % 2) + 1, depth - 1, True)
                expected_value += value / len(possible_actions)
            return None, expected_value
