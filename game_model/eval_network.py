import argparse
import torch

from shutil import copy

from game import State, random_action, alpha_beta_action, mcts_action
from dual_network import DualNetwork
from mcts import pv_mcts_action


parser = argparse.ArgumentParser(description="Reinforcement Train")

# model hyper parameter
parser.add_argument('--pv_eval_count', type=int, default=50)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--num_residual_block', type=int, default=16)
parser.add_argument('--num_filters', type=int, default=128)

# evaluate hyper parameter
parser.add_argument('--eval_epochs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)


def first_player_point(ended_state):
    # 1: win, 0: lose, 0.5: draw
    if ended_state.is_lose():
        return 0 if ended_state.is_first_player() else 1
    return 0.5

def play(next_actions):
    state = State()

    while True:
        if state.is_done():
            break

        next_action = next_actions[0] if state.is_first_player() else next_actions[1]
        action = next_action(state)

        state = state.next(action)

    return first_player_point(state)

def evaluate_algorithm(next_actions, eval_epochs):
    total_point = 0
    for eval_epoch in range(eval_epochs):
        if eval_epoch % 2 == 0:
            total_point += play(next_actions)
        else:
            total_point += 1 - play(list(reversed(next_actions)))
        print(f'\rEvaluate {eval_epoch+1}/{eval_epochs}', end='')
    print('')

    average_point = total_point / eval_epochs
    return average_point

def update_best_player():
    copy('./model/latest.pth', './model/best.pth')
    print('Change BestPlayer')

def evaluate_network(args, net):
    # Initialize the state
    state = State()

    # select action in mcts algorithm with latest & best model
    net.load_state_dict(torch.load('./model/latest.pth'))
    next_action_latest = pv_mcts_action(net, args.pv_eval_count, args.temperature)
    net.load_state_dict(torch.load('./model/best.pth'))
    next_action_best = pv_mcts_action(net, args.pv_eval_count, args.temperature)
    next_actions = (next_action_latest, next_action_best)

    # calculate the average point and change the model
    average_point = evaluate_algorithm(next_actions, args.eval_epochs)
    print(f'AveragePoint: {average_point}')
    if average_point > 0.5:
        update_best_player()
        return True
    else:
        return False


def evaluate_best_player(args, net):
    net.load_state_dict(torch.load('./model/best.pth'))
    state = State()

    # mcts action function
    next_pv_mcts_action = pv_mcts_action(net, pv_eval_count=args.pv_eval_count, temperature=0.0)

    # vs. random algorithm
    next_actions = (next_pv_mcts_action, random_action)
    average_point = evaluate_algorithm(next_actions, args.eval_epochs)
    print(f'VS_Random: {average_point}')

    # vs. alpha_beta algorithm
    next_actions = (next_pv_mcts_action, alpha_beta_action)
    average_point = evaluate_algorithm(next_actions, args.eval_epochs)
    print(f'VS_AlphaBeta: {average_point}')

    # vs. native mcts algorithm
    next_actions = (next_pv_mcts_action, mcts_action)
    average_point = evaluate_algorithm(next_actions, args.eval_epochs)
    print(f'VS_MCTS: {average_point}')

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = DualNetwork(num_residual_block=args.num_residual_block, num_filters=args.num_filters).to(device)

    # load the best.pth and calculate the win rate
    evaluate_best_player(args, net)