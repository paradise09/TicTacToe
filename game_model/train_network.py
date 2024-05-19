import os
import torch

from tqdm import tqdm
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from data import TicTacToeDataset

def train_network(args, net, device):
    # Dataset & DataLoader
    print('\n<Load Dataset>')
    train_dataset = TicTacToeDataset()
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Criterion, Optimizer
    policy_criterion = CrossEntropyLoss()
    value_criterion = MSELoss()
    optimizer = Adam(net.parameters(), lr=args.learning_rate)

    # Train
    print('\n<Train Model>')
    for train_epoch in range(args.train_epochs):
        # adjust learning rate
        if train_epoch >= 50:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0005
        if train_epoch >= 80:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.00025
        
        net.train()
        train_loss = 0.0
        best_loss = 100.0
        for state, y_policy, y_value in tqdm(train_loader):
            state = state.to(device)
            y_policy = y_policy.to(device)
            y_value = y_value.to(device)
            
            optimizer.zero_grad()

            policy_output, value_output = net(state)
            value_output = value_output.squeeze()

            policy_loss = policy_criterion(policy_output, y_policy)
            value_loss = value_criterion(value_output, y_value)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        if best_loss > train_loss:
            os.makedirs('./model/', exist_ok=True)
            torch.save(net.state_dict(), './model/latest.pth')

        print(f'{train_epoch+1}/{args.train_epochs}')
        print(f'    Loss: {train_loss:.4f}')