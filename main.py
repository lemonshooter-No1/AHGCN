import argparse
import os
import random
import statistics
from math import sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from Data.Feeder import Feeder
from Processing.data_processing import Data_Loader
from model import module


def parse_args():
    parser = argparse.ArgumentParser(description='AHGCN model training script')

    # Dataset related parameters
    parser.add_argument('--dataset', type=str, default='Kimore_ex2', help='Dataset name')
    parser.add_argument('--test_size', type=float, default=0.4, help='Test set ratio')
    parser.add_argument('--val_size', type=float, default=0.5, help='Validation set ratio relative to test set')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')

    # Model related parameters
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')

    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')

    # Dropout parameters
    parser.add_argument('--tcn_dropout', type=float, default=0.2, help='Dropout in TCN layers')
    parser.add_argument('--model_dropout', type=float, default=0.25, help='Dropout in main model')

    # Training related parameters
    parser.add_argument('--num_epochs', type=int, default=800, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr_decay_epochs', type=int, default=500, help='Learning rate decay cycle')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--loss_beta', type=float, default=0.1, help='Beta parameter for SmoothL1Loss')

    # Device related parameters
    parser.add_argument('--device', type=str, default='cuda', help='Training device')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='Whether to use deterministic algorithms')

    # Output related parameters
    parser.add_argument('--print_freq', type=int, default=50, help='Print frequency')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Directory for saving models')
    parser.add_argument('--model_name', type=str, default='best_model', help='Model save name')

    return parser.parse_args()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def adjust_learning_rate(optimizer, epoch, start_lr, decay_epochs, decay_factor):
    lr = start_lr * (decay_factor ** (epoch // decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def calculate_metrics(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape
    }


def train_model(args):
    seed_torch(args.random_state)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Model save directory: {args.save_dir}\n")

    print(f"\nDropout Configuration:")
    print(f"  TCN Dropout: {args.tcn_dropout}")
    print(f"  Model Dropout: {args.model_dropout}")

    print(f"Loading dataset: {args.dataset}")
    data_loader = Data_Loader(args.dataset)

    train_x, test_x, train_y, test_y = train_test_split(
        data_loader.scaled_x, data_loader.scaled_y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    val_x, test_x, val_y, test_y = train_test_split(
        test_x, test_y,
        test_size=args.val_size,
        random_state=args.random_state
    )

    train_dataset = Feeder(train_x, train_y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    print(f"Initializing model, dataset: {args.dataset}")
    model = module.Module(args.dataset, device=device,tcn_dropout=args.tcn_dropout,model_dropout=args.model_dropout).to(device)

    criterion = nn.SmoothL1Loss(beta=args.loss_beta).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    print(f"Starting training for {args.num_epochs} epochs")
    train_losses = []
    val_losses = []

    best_metrics = {
        'mae': float('inf'),
        'rmse': float('inf'),
        'mape': float('inf'),
    }
    best_epoch = 0
    best_predictions = None

    for epoch in tqdm(range(args.num_epochs)):

        lr = adjust_learning_rate(
            optimizer, epoch, args.learning_rate,
            args.lr_decay_epochs, args.lr_decay_factor
        )

        if epoch % 400 == 0:
            print(f"Epoch: {epoch}, Learning Rate: {lr:.2E}")


        model.train()
        train_loss = 0
        train_metrics = {
            'mae': 0,
            'rmse': 0,
            'mape': 0
        }

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.float().to(device), y.float().to(device)


            outputs = model(x)
            loss = criterion(outputs, y)


            y_np = y.cpu().detach().numpy()
            outputs_np = outputs.cpu().detach().numpy()
            batch_metrics = calculate_metrics(y_np, outputs_np)

            train_loss += loss.item()
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key]


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        num_batches = len(train_loader)
        train_loss /= num_batches
        for key in train_metrics:
            train_metrics[key] /= num_batches

        train_losses.append(train_loss)


        model.eval()
        with torch.no_grad():
            val_x_tensor = torch.as_tensor(val_x).float().to(device)
            val_y_tensor = torch.as_tensor(val_y).float().to(device)

            val_outputs = model(val_x_tensor)
            val_loss = criterion(val_outputs, val_y_tensor).item()

            val_outputs_np = val_outputs.cpu().numpy()
            val_y_np = val_y_tensor.cpu().numpy()
            val_metrics = calculate_metrics(val_y_np, val_outputs_np)

            val_losses.append(val_loss)


            if val_metrics['mae'] < best_metrics['mae']:
                best_metrics['mae'] = val_metrics['mae']

            if val_metrics['rmse'] < best_metrics['rmse']:
                best_metrics['rmse'] = val_metrics['rmse']
                best_epoch = epoch
                best_predictions = val_outputs_np

                torch.save(
                    model.state_dict(),
                    os.path.join(args.save_dir, f"{args.model_name}_best.pth")
                )

            if val_metrics['mape'] < best_metrics['mape']:
                best_metrics['mape'] = val_metrics['mape']



        if (epoch + 1) % args.print_freq == 0:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
            print(f"Learning Rate: {lr:.2E}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation MAE: {val_metrics['mae']:.4f}")
            print(f"Validation RMSE: {val_metrics['rmse']:.4f}")
            print(f"Validation MAPE: {val_metrics['mape']:.4f}")


    print("\nEvaluating best model on test set")
    model.load_state_dict(torch.load(os.path.join(args.save_dir, f"{args.model_name}_best.pth")))
    model.eval()

    with torch.no_grad():
        test_x_tensor = torch.as_tensor(test_x).float().to(device)
        test_y_tensor = torch.as_tensor(test_y).float().to(device)

        test_outputs = model(test_x_tensor)
        test_outputs_np = test_outputs.cpu().numpy()
        test_y_np = test_y_tensor.cpu().numpy()

        test_metrics = calculate_metrics(test_y_np, test_outputs_np)

        print(f"Test MAE: {test_metrics['mae']:.4f}")
        print(f"Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"Test MAPE: {test_metrics['mape']:.4f}")

    print("\nBest Metrics:")
    print(f"Epoch: {best_epoch + 1}")
    print(f"MAE: {best_metrics['mae']:.4f}")
    print(f"RMSE: {best_metrics['rmse']:.4f}")
    print(f"MAPE: {best_metrics['mape']:.4f}")

    return model, best_metrics


if __name__ == '__main__':
    args = parse_args()
    model, best_metrics = train_model(args)





