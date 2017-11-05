import os, glob
import json
import re
import numpy as np
from shutil import copy
from AlphagoZero.ai import MCTSPlayer
import AlphagoZero.go as go
from AlphagoZero.models.policy_value import PolicyValue
from AlphagoZero.util import flatten_idx, pprint_board


def run_a_game(new_player, old_player, i, boardsize, mock_state=[]):
    '''Run num_games games to completion, keeping track of each position and move of the new_player.
    And return the win ratio

    '''

    board_size = boardsize
    state = go.GameState(size=board_size, komi=0)

    # Allowing injection of a mock state object for testing purposes
    if mock_state:
        state = mock_state

    # Start all odd games with moves by 'old_player'. Even games will have 'new_player' black.
    new_player_color = go.BLACK if i % 2 == 0 else go.WHITE
    if new_player_color == go.BLACK:
        current = new_player
        other = old_player
    else:
        current = old_player
        other = new_player

    while not state.is_end_of_game:
        move = current.get_move(state,True)

        #print(move)
        childrens = current.mcts._root._children.items()
        actions, next_states = map(list, zip(*childrens))
        n_ = [next_state._n_visits for next_state in next_states]
        #print(zip(actions, n_))
        current.mcts.update_with_move(move)

        state.do_move(move)
        other.mcts.update_with_move(move)
        current, other = other, current

        pprint_board(state.board)
    #print(state.get_winner())
    return state.get_winner() == new_player_color

def run_evaluate(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate neural network checkpoint.')  # noqa: E501
    parser.add_argument("--model_json", help="Path to policy value model JSON.", default='network.json')
    parser.add_argument("--best_directory", help="Path to folder where the model params and metadata will be saved after each evaluation.", default='/../ckpt/best/weights.*.hdf5')  # noqa: E501/
    parser.add_argument("--optimized_directory", help="Path to folder where optimized weights are saved", default="/../ckpt/optimized/weights.*.hdf5"),
    parser.add_argument("--num_games", help="The number of games for evaluation", default=40, type=int),
    parser.add_argument("--resume", help="Load latest metadata", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=False, action="store_true")  # noqa: E501
    parser.add_argument("--playout_depth", help="Playout depth", default=5, type=int)
    parser.add_argument("--n_playout", help="number of playout", default=5, type=int)
    # Baseline function (TODO) default lambda state: 0  (receives either file
    # paths to JSON and weights or None, in which case it uses default baseline 0)
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    best_model_json = os.path.dirname(__file__) + os.path.join(os.path.dirname(args.best_directory), args.model_json)
    candid_model_json = os.path.dirname(__file__) + os.path.join(os.path.dirname(args.optimized_directory), args.model_json)

    args.optimized_directory = os.path.dirname(__file__) + args.optimized_directory
    args.best_directory = os.path.dirname(__file__) + args.best_directory


    if not args.resume:
        metadata = {
            "win_ratio": {}  # map from best_player to tuple of (candid_weight, win ratio)
        }
    else:
        with open(os.path.join(os.path.dirname(args.best_directory), "metadata.json"), "r") as f:
            metadata = json.load(f)

    # Append args of current run to history of full command args.
    metadata["cmd_line_args"] = metadata.get("cmd_line_args", [])
    metadata["cmd_line_args"].append(vars(args))

    def save_metadata():
        with open(os.path.join(os.path.dirname(args.best_directory), "metadata.json"), "w") as f:
            json.dump(metadata, f, sort_keys=True, indent=2)

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

    while True:
        optimized_weight_list = glob.glob(args.optimized_directory)
        optimized_weight_list.sort()
        candid_weight_path = optimized_weight_list[-1]
        if os.path.basename(best_weight_path) == os.path.basename(candid_weight_path):
            pass
        else:
            print("Evaluating with candid weights {} and best weights {}".format(candid_weight_path, best_weight_path))

            # Set initial conditions
            policy = PolicyValue.load_model(best_model_json)
            policy.model.load_weights(best_weight_path)
            boardsize = policy.model.input_shape[-1]
            # different opponents come from simply changing the weights of 'opponent.policy.model'. That
            # is, only 'opp_policy' needs to be changed, and 'opponent' will change.
            candid_policy = PolicyValue.load_model(candid_model_json)
            candid_policy.model.load_weights(candid_weight_path)

            game_history = np.zeros((args.num_games))
            for i in range(args.num_games):
                print(str(i) + "th evaluating game")
                best_player = MCTSPlayer(policy.eval_value_state, policy.eval_policy_state, playout_depth=args.playout_depth, n_playout=args.n_playout, evaluating=True)
                candid_player= MCTSPlayer(candid_policy.eval_value_state, candid_policy.eval_policy_state, playout_depth=args.playout_depth, n_playout=args.n_playout, evaluating=True)
                game_history[i] = run_a_game(candid_player, best_player, i, boardsize)
                del best_player
                del candid_player

            win_ratio = game_history.sum()/args.num_games
            metadata["win_ratio"][best_weight_path] = [candid_weight_path, win_ratio]
            save_metadata()
            print(win_ratio)
            if win_ratio > 0.55:
                print(candid_weight_path)
                copy(candid_weight_path, os.path.dirname(best_weight_path))
                print("The new best neural network!")
                best_weight_path = os.path.join(os.path.dirname(best_weight_path), os.path.basename(candid_weight_path))
                with open(best_model_json, 'r') as f:
                    js = json.load(f)
                    js["weights_file"] = best_weight_path
                    f.close()
                with open(best_model_json, 'w') as f:
                    json.dump(js,f)
                    f.close()

if __name__ == '__main__':
    run_evaluate()
