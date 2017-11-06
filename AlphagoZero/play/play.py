import os, glob
import json
import re
import numpy as np
from shutil import copy
from AlphagoZero.ai import MCTSPlayer
import AlphagoZero.go as go
from AlphagoZero.models.policy_value import PolicyValue
from AlphagoZero.util import flatten_idx, pprint_board

class Human(object):
    def __init__(self,board_size):
        self.is_human = True
        self.board_size = board_size

    def get_move(self, state):
        query = raw_input("Your move: ")
        if len(query)==0:
            return go.PASS_MOVE
        else:
            alphabet, number = re.match(r"([a-z]+)([0-9]+)", query, re.I).groups()
            y = ord(alphabet.upper()) - ord('A')
            x = self.board_size - int(number)
            return ((x,y))

def run_a_game(alphago_player, human_player, boardsize):
    '''Run num_games games to completion, keeping track of each position and move of the new_player.
    And return the win ratio

    '''

    board_size = boardsize
    state = go.GameState(size=board_size, komi=0)

    # Start all odd games with moves by 'old_player'. Even games will have 'new_player' black.
    human_color = np.random.choice([go.BLACK, go.WHITE])
    if human_color == go.BLACK:
        current = human_player
        other = alphago_player
        print("Your color is black.")
    else:
        current = alphago_player
        other = human_player
        print("Your color is white.")

    pprint_board(state.board)
    while not state.is_end_of_game:
        move = current.get_move(state)
        try:
            state.do_move(move)
        except:
            print("Illegal move!")
            continue
        if other == alphago_player:
            other.mcts.update_with_move(move)
        current, other = other, current

        pprint_board(state.board)
    winner = state.get_winner()
    if winner == human_color:
        print("You won.")
    elif winner == 0:
        print("Tie.")
    else:
        print("AlphagoZero won")

def run_play(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Play a game with the current best neural network checkpoint.')  # noqa: E501
    parser.add_argument("--model_json", help="Path to policy value model JSON.", default='network.json')
    parser.add_argument("--best_directory", help="Path to folder where the model params and metadata will be saved after each evaluation.", default='/../ckpt/best/weights.*.hdf5')  # noqa: E501/
    parser.add_argument("--optimized_directory", help="Path to folder where optimized weights are saved", default="/../ckpt/optimized/weights.*.hdf5"),
    parser.add_argument("--playout_depth", help="Playout depth", default=5, type=int)
    parser.add_argument("--n_playout", help="number of playout", default=5, type=int)
    # Baseline function (TODO) default lambda state: 0  (receives either file
    # paths to JSON and weights or None, in which case it uses default baseline 0)
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    best_model_json = os.path.dirname(__file__) + os.path.join(os.path.dirname(args.best_directory), args.model_json)

    args.optimized_directory = os.path.dirname(__file__) + args.optimized_directory
    args.best_directory = os.path.dirname(__file__) + args.best_directory

    while True:
        best_weight_list = glob.glob(args.best_directory)
        if len(best_weight_list) == 0:
            while True:
                optimized_weight_list = glob.glob(args.optimized_directory)
                if len(optimized_weight_list) != 0:
                    optimized_weight_list.sort()
                    candid_weight_path = optimized_weight_list[-1]
                    copy(candid_weight_path, os.path.dirname(args.best_directory))
                    copy(os.path.join(os.path.dirname(args.optimized_directory), args.model_json), os.path.dirname(args.best_directory))
                    print("The first neural network!")
                    best_weight_path = os.path.join(os.path.dirname(args.best_directory), os.path.basename(candid_weight_path))
                    with open(best_model_json, 'r') as f:
                        js = json.load(f)
                        js["weights_file"] = best_weight_path
                        f.close()
                    with open(best_model_json, 'w') as f:
                        json.dump(js,f)
                        f.close()
                    break
        else:
            break

    best_weight_list.sort()
    best_weight_path = best_weight_list[-1]


    print("Playing with weights {}".format(best_weight_path))

    # Set initial conditions
    policy = PolicyValue.load_model(best_model_json)
    policy.model.load_weights(best_weight_path)
    boardsize = policy.model.input_shape[-1]
    best_player = MCTSPlayer(policy.eval_value_state, policy.eval_policy_state, playout_depth=args.playout_depth, n_playout=args.n_playout, evaluating=True)
    human_player = Human(boardsize)
    run_a_game(best_player, human_player, boardsize)

if __name__ == '__main__':
    run_play()
