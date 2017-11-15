import os, glob, pickle
import json
import re
import numpy as np
from shutil import copy
from AlphagoZero.ai import MCTSPlayer
import AlphagoZero.go as go
from AlphagoZero.models.policy_value import PolicyValue
from AlphagoZero.util import flatten_idx, pprint_board, plot_board
from AlphagoZero.preprocessing import preprocessing


def self_play_and_save(player, opp_player, i, boardsize, weight_name, plot_dir, mock_state=[]):
    '''Run num_games games to completion, keeping track of each position and move of the new_player.
    And save the game data

    '''
    state_list = []
    pi_list = []
    player_list = []

    board_size = boardsize
    state = go.GameState(size=board_size, komi=0)

    # Allowing injection of a mock state object for testing purposes
    if mock_state:
        state = mock_state

    # Start all odd games with moves by 'old_player'. Even games will have 'new_player' black.
    player_color = go.BLACK if i % 2 == 0 else go.WHITE
    if player_color == go.BLACK:
        current = player
        other = opp_player
    else:
        current = opp_player
        other = player

    step = 0
    while not state.is_end_of_game:
        move = current.get_move(state, self_play=False)

        #print(move)
        childrens = current.mcts._root._children.items()
        #print(childrens)
        actions, next_states = map(list, zip(*childrens))
        _n_visits = [next_state._n_visits for next_state in next_states]
        if not move == go.PASS_MOVE:
            if step < 25: # temperature is considered to be 1
                distribution = np.divide(_n_visits, np.sum(_n_visits))
            else:
                max_visit_idx = np.argmax(_n_visits)
                distribution = np.zeros(np.shape(_n_visits))
                distribution[max_visit_idx] = 1.0
        else: # to prevent the model from overfitting to PASS_MOVE
            distribution = np.zeros(np.shape(_n_visits))
        pi = zip(actions, distribution)
        #print(zip(actions, _n_visits))
        state_list.append(state)
        pi_list.append(pi)

        current.mcts.update_with_move(move)
        state.do_move(move)
        other.mcts.update_with_move(move)
        current, other = other, current
        step += 1

        plot_board(state.board, state.history, plot_dir, weight_name+'-{0:03d}'.format(step)+".png")
        #pprint_board(state.board)

    winner = state.get_winner()
    #print(winner)
    if winner == go.BLACK:
        reward_list = [(-1.)**j for j in range(len(state_list))]
    else : # winner == go.WHITE:
        reward_list = [(-1.)**(j+1) for j in range(len(state_list))]
    return state_list, pi_list, reward_list

def run_self_play(cmd_line_args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Self-play and generate data for optimization using the best neural network checkpoint.')  # noqa: E501
    parser.add_argument("--model_json", help="Path to policy value model JSON.", default='network.json')
    parser.add_argument("--best_directory", help="Path to folder where the model params and metadata will be saved after each evaluation.", default='/../ckpt/best/weights.0000524.hdf5')  # noqa: E501/
    parser.add_argument("--plot_directory", help="Path to folder where data for optimization are saved", default="/./plot"),
    parser.add_argument("--num_games", help="The number of games for evaluation", default=500, type=int),
    parser.add_argument("--verbose", "-v", help="Turn on verbose mode", default=True, action="store_true")  # noqa: E501
    parser.add_argument("--n_playout", help="number of playout", default=7, type=int)

    # Baseline function (TODO) default lambda state: 0  (receives either file
    # paths to JSON and weights or None, in which case it uses default baseline 0)
    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)
    args.model_json = os.path.dirname(__file__) + os.path.join(os.path.dirname(args.best_directory), args.model_json)
    args.best_directory = os.path.dirname(__file__) + args.best_directory
    args.plot_directory = os.path.dirname(__file__) + args.plot_directory

    if not os.path.exists(args.plot_directory):
        os.makedirs(args.plot_directory)

    best_weight_path = args.best_directory
    weight_name = (os.path.basename(args.best_directory)).split('.')[1]
    if args.verbose:
        print("Self-playing with weights {}".format(best_weight_path))

    # Set initial conditions
    policy = PolicyValue.load_model(args.model_json)
    policy.model.load_weights(best_weight_path)

    boardsize = policy.model.input_shape[-1]
    # different opponents come from simply changing the weights of 'opponent.policy.model'. That
    # is, only 'opp_policy' needs to be changed, and 'opponent' will change.
    opp_policy = PolicyValue.load_model(args.model_json)
    opp_policy.model.load_weights(best_weight_path)


    print("Start self play with "+best_weight_path)
    player = MCTSPlayer(policy.eval_value_state, policy.eval_policy_state, n_playout=args.n_playout, evaluating=True, self_play=False)
    opp_player= MCTSPlayer(opp_policy.eval_value_state, opp_policy.eval_policy_state, n_playout=args.n_playout, evaluating=True, self_play=False)
    state_list, pi_list, reward_list = self_play_and_save(opp_player, player, 0, boardsize, weight_name, args.plot_directory)



if __name__ == '__main__':
    run_self_play()
