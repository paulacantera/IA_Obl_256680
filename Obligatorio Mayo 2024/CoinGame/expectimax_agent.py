from agent import Agent
from board import Board
import math

class ExpectimaxAgent(Agent):
    
    def __init__(self, player=1, max_depth=3):
        super().__init__(player)
        self.max_depth = max_depth
        
    def next_action(self, obs):
        action, _ = self.expectimax(obs, self.player, True, self.max_depth)
        return action
    
    def heuristic_utility(self, board: Board):
        # heuristica simple q se basa en la cant de coins restantes
        return board.grid.sum()
    
    def expectimax(self, board: Board, player, is_maximizing, depth):
        if depth == 0 or board.is_end(player)[0]:
            return None, self.heuristic_utility(board)
        
        possible_actions = board.get_possible_actions()
        
        if is_maximizing:
            best_value = -math.inf
            best_action = None
            for action in possible_actions:
                new_board = board.clone()
                new_board.play(action)
                _, value = self.expectimax(new_board, (player % 2) + 1, False, depth - 1)
                if value > best_value:
                    best_value = value
                    best_action = action
            return best_action, best_value
        else:
            expected_value = 0
            for action in possible_actions:
                new_board = board.clone()
                new_board.play(action)
                _, value = self.expectimax(new_board, (player % 2) + 1, True, depth - 1)
                expected_value += value / len(possible_actions)
            return None, expected_value