import os
import pickle
import argparse
import numpy as np

from datetime import datetime
from pathlib import Path
from game import State
from mcts import pv_mcts_scores
from dual_network import DualNetwork

# value of first player (1 : win / 0 : draw / -1 : lose)
def first_player_value(ended_state):
    if ended_state.is_lose():
        return -1 if ended_state.is_first_player() else 1
    return 0

# store the train data
def write_data(history):
    now = datetime.now()
    os.makedirs('./data/', exist_ok=True)
    path = f'./data/{now.year:04}{now.month:02}{now.day:02}{now.hour:02}{now.minute:02}{now.second:02}.history'

    with open(path, mode='wb') as f:
        pickle.dump(history, f)

# game play
def play(model, pv_eval_count, temperature):
    history = []
    state = State()

    while True:
        if state.is_done():
            break

        scores = pv_mcts_scores(model, state, pv_eval_count, temperature)

        policies = [0] * 9 # output_size
        for action, policy in zip(state.legal_actions(), scores):
            policies[action] = policy
        history.append([[state.pieces, state.enemy_pieces], policies, None])

        action = np.random.choice(state.legal_actions(), p=scores)

        state = state.next(action)

    value = first_player_value(state)
    for i in range(len(history)):
        history[i][2] = value
        value = -value
        
    return history

# self play
def self_play(args, net):
    history = []
    net.eval()
    for i in range(args.self_count):
        h = play(net, args.pv_eval_count, args.temperature)
        history.extend(h)
        print(f'\rSelfPlay {i+1}/{args.self_count}', end='')
    
    print('')

    write_data(history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare and store the data to play self")

    parser.add_argument('--pv_eval_count', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--num_residual_block', type=int, default=16)
    parser.add_argument('--num_filters', type=int, default=128)
    parser.add_argument('--self_count', type=int, default=500)

    args = parser.parse_args()
    net = DualNetwork(num_residual_block=args.num_residual_block, num_filters=args.num_filters)
    self_play(args, net)