import torch
import argparse

from dual_network import DualNetwork
from self_play import self_play
from data import TicTacToeDataset
from train_network import train_network
from eval_network import evaluate_network, evaluate_best_player

parser = argparse.ArgumentParser(description="Reinforcement Train")

# model hyper parameter
parser.add_argument('--pv_eval_count', type=int, default=50)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--num_residual_block', type=int, default=16)
parser.add_argument('--num_filters', type=int, default=128)

# train hyper parameter
parser.add_argument('--self_count', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--train_epochs', type=int, default=100)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--num_workers', type=int, default=4)

# evaluate hyper parameter
parser.add_argument('--eval_epochs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)

def train(args):
    # Declare the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nDevice : ', device)

    # Define the model & make empty weights file named best.pth
    net = DualNetwork(num_residual_block=args.num_residual_block, num_filters=args.num_filters).to(device)
    torch.save(net.state_dict(), './model/best.pth')

    # train and evaluate cycle
    for epoch in range(args.epochs):
        print(f'\nTrain Cycle [{epoch} / {args.epochs}]')
        self_play(args, net)
        train_network(args, net, device)
        update_best_player = evaluate_network(args, net)
        if update_best_player:
            evaluate_best_player(args, net)

        
if __name__ == '__main__':
    args = parser.parse_args()
    train(args)