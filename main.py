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
from graph.Hypergraph import HypergraphCache
from Processing.data_processing import Data_Loader
from Processing.data_processing_UIPRMD import UI_Data_Loader
from Processing.data_processing_skate import Oly_Skate_Data_Loader
from model import module


def parse_args():
    parser = argparse.ArgumentParser(description='AHGCN model training script')

    # Dataset related parameters
    parser.add_argument('--dataset', type=str, default='Kimore_ex2', help='Dataset name')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set ratio')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation set ratio within the training pool')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')

    # Model related parameters
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')

    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading')

    # Dropout parameters
    parser.add_argument('--tcn_dropout', type=float, default=0.0, help='Dropout in TCN layers')
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

    parser.add_argument('--num_center', type=int, default=5)
    parser.add_argument('--sparse_lambda', type=float, default=0.01)
    parser.add_argument('--sparse_threshold', type=float, default=0.001)
    parser.add_argument('--sparse_iter', type=int, default=1000)
    parser.add_argument('--sparse_tol', type=float, default=1e-6)
    parser.add_argument('--temporal_k', type=int, default=4)
    parser.add_argument('--gru_norm', type=int, choices=[1, 2], default=1)
    parser.add_argument('--init_variant', choices=['original', 'spatial_open', 'temporal_identity', 'gru_memory', 'spatial_gru', 'all'], default='original')
    parser.add_argument('--topk_ratio', type=float, default=0.5)
    parser.add_argument('--use_topk', action='store_true')
    parser.add_argument('--hypergraph_cache', choices=['auto', 'cpu', 'cuda', 'none'], default='auto')

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
    is_skate = 'MIT_Skate' in args.dataset or 'MITSkate' in args.dataset
    if is_skate:
        data_loader = Oly_Skate_Data_Loader(args.dataset)
        train_indices, test_indices = train_test_split(
            np.arange(len(data_loader.scaled_y)), train_size=100,
            random_state=args.random_state
        )
        train_x, train_y = data_loader.scaled_x[train_indices], data_loader.scaled_y[train_indices]
        test_x, test_y = data_loader.scaled_x[test_indices], data_loader.scaled_y[test_indices]
        np.savez(os.path.join(args.save_dir, f'{args.model_name}_split.npz'),
                 train_indices=train_indices, test_indices=test_indices)
    else:
        if 'UI_PRMD' in args.dataset:
            data_loader = UI_Data_Loader(args.dataset)
            all_x, all_y = data_loader.all_data_x, data_loader.all_data_y
        else:
            data_loader = Data_Loader(args.dataset)
            all_x, all_y = data_loader.scaled_x, data_loader.scaled_y
        pool_indices, test_indices = train_test_split(
            np.arange(len(all_y)), test_size=args.test_size, random_state=args.random_state
        )
        train_indices, val_indices = train_test_split(
            pool_indices, test_size=args.val_size, random_state=args.random_state
        )
        train_x, train_y = all_x[train_indices], all_y[train_indices]
        val_x, val_y = all_x[val_indices], all_y[val_indices]
        test_x, test_y = all_x[test_indices], all_y[test_indices]
        np.savez(os.path.join(args.save_dir, f'{args.model_name}_split.npz'),
                 train_indices=train_indices, val_indices=val_indices, test_indices=test_indices)

    cache_mode = getattr(args, 'hypergraph_cache', 'auto')
    train_dataset = Feeder(train_x, train_y, return_index=cache_mode != 'none')
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    print(f"Initializing model, dataset: {args.dataset}")
    model = module.Module(args.dataset, device=device, tcn_dropout=args.tcn_dropout,
                          model_dropout=args.model_dropout, num_center=args.num_center,
                          sparse_lambda=args.sparse_lambda, sparse_threshold=args.sparse_threshold,
                          sparse_iter=args.sparse_iter, sparse_tol=args.sparse_tol,
                          temporal_k=args.temporal_k, topk_ratio=args.topk_ratio,
                          gru_norm=args.gru_norm, init_variant=args.init_variant,
                          use_topk=args.use_topk).to(device)
    model_config = dict(model.hypergraph_config, dataset_name=args.dataset,
                        tcn_dropout=args.tcn_dropout, model_dropout=args.model_dropout)
    data_sets = [train_x, test_x] if is_skate else [train_x, val_x, test_x]
    cache_device = HypergraphCache.select_device(cache_mode, device, data_sets, args.temporal_k)
    train_cache = val_cache = test_cache = None
    if cache_device is not None:
        print(f'Preparing hypergraph cache on {cache_device}')
        train_cache = HypergraphCache(model.structure, train_x, cache_device, args.batch_size, device)
        if not is_skate:
            val_cache = HypergraphCache(model.structure, val_x, cache_device, args.batch_size, device)
        test_cache = HypergraphCache(model.structure, test_x, cache_device, args.batch_size, device)
        print('Hypergraph cache ready')


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
    best_epochs = {}
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

        for batch_idx, batch in enumerate(train_loader):
            x, y = batch[:2]
            graphs = train_cache.batch(batch[2], device) if train_cache is not None else None
            x, y = x.float().to(device), y.float().to(device)


            outputs = model(x, hypergraphs=graphs)
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
        if is_skate:
            if (epoch + 1) % args.print_freq == 0:
                print(f'Epoch {epoch + 1}/{args.num_epochs}, Training Loss: {train_loss:.4f}')
            if epoch + 1 == args.num_epochs:
                torch.save({'state_dict': model.state_dict(), 'model_config': model_config}, os.path.join(args.save_dir, f'{args.model_name}_final.pth'))
            continue



        model.eval()
        with torch.no_grad():
            val_x_tensor = torch.as_tensor(val_x).float().to(device)
            val_y_tensor = torch.as_tensor(val_y).float().to(device)

            graphs = val_cache.batch(range(len(val_x)), device) if val_cache is not None else None
            val_outputs = model(val_x_tensor, hypergraphs=graphs)
            val_loss = criterion(val_outputs, val_y_tensor).item()

            val_outputs_np = val_outputs.cpu().numpy()
            val_y_np = val_y_tensor.cpu().numpy()
            val_metrics = calculate_metrics(val_y_np, val_outputs_np)

            val_losses.append(val_loss)


            for metric in best_metrics:
                if val_metrics[metric] < best_metrics[metric]:
                    best_metrics[metric] = val_metrics[metric]
                    best_epochs[metric] = epoch + 1
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'model_config': model_config,
                        'epoch': epoch + 1,
                        'selection_metric': metric,
                        'val_metrics': dict(val_metrics),
                    }
                    torch.save(checkpoint, os.path.join(args.save_dir, f"{args.model_name}_best_{metric}.pth"))
                    if metric == 'rmse':
                        torch.save(checkpoint, os.path.join(args.save_dir, f"{args.model_name}_best.pth"))

        if (epoch + 1) % args.print_freq == 0:
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
            print(f"Learning Rate: {lr:.2E}")
            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation MAE: {val_metrics['mae']:.4f}")
            print(f"Validation RMSE: {val_metrics['rmse']:.4f}")
            print(f"Validation MAPE: {val_metrics['mape']:.4f}")


    checkpoint_suffixes = ['final'] if is_skate else ['best_mae', 'best_rmse', 'best_mape']
    test_results = {}
    for checkpoint_suffix in checkpoint_suffixes:
        checkpoint_path = os.path.join(args.save_dir, f"{args.model_name}_{checkpoint_suffix}.pth")
        print(f"\nEvaluating {checkpoint_suffix} checkpoint on test set")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'epoch' in checkpoint:
            print(f"Epoch: {checkpoint['epoch']}")
            validation = checkpoint['val_metrics']
            print(f"Validation MAE: {validation['mae']:.4f}, RMSE: {validation['rmse']:.4f}, MAPE: {validation['mape']:.4f}")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        with torch.no_grad():
            test_x_tensor = torch.as_tensor(test_x).float().to(device)
            test_y_tensor = torch.as_tensor(test_y).float().to(device)

            if is_skate:
                test_outputs = torch.cat([
                    model(torch.as_tensor(test_x[start:start + args.batch_size], dtype=torch.float32, device=device),
                          hypergraphs=test_cache.batch(range(start, min(start + args.batch_size, len(test_x))), device)
                          if test_cache is not None else None)
                    for start in range(0, len(test_x), args.batch_size)
                ], dim=0)
            else:
                graphs = test_cache.batch(range(len(test_x)), device) if test_cache is not None else None
                test_outputs = model(test_x_tensor, hypergraphs=graphs)
            test_outputs_np = test_outputs.cpu().numpy()
            test_y_np = test_y_tensor.cpu().numpy()

            test_metrics = calculate_metrics(test_y_np, test_outputs_np)

            if is_skate:
                print(f"Test SRC: {spearmanr(test_y_np.ravel(), test_outputs_np.ravel())[0]:.4f}")

            print(f"Test MAE: {test_metrics['mae']:.4f}")
            print(f"Test RMSE: {test_metrics['rmse']:.4f}")
            print(f"Test MAPE: {test_metrics['mape']:.4f}")
        test_results[checkpoint_suffix] = test_metrics

    if is_skate:
        return model, test_results['final']

    print("\nBest validation metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric.upper()}: {value:.4f} (Epoch {best_epochs[metric]})")

    model.load_state_dict(torch.load(os.path.join(args.save_dir, f"{args.model_name}_best_rmse.pth"), map_location=device)['state_dict'])
    return model, best_metrics


if __name__ == '__main__':
    args = parse_args()
    model, best_metrics = train_model(args)





