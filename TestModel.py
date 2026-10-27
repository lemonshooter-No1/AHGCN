import argparse
import os
from math import sqrt

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from Processing.data_processing import Data_Loader
from Processing.data_processing_UIPRMD import UI_Data_Loader
from Processing.data_processing_skate import Oly_Skate_Data_Loader
from model import module
from graph.Hypergraph import HypergraphCache


def parse_args():
    parser = argparse.ArgumentParser(description='Model performance testing script')

    # Dataset related parameters
    parser.add_argument('--dataset', type=str, default='Kimore_ex2',
                        help='Dataset name (Kimore_ex2, UI_PRMD_ex1, etc.)')
    parser.add_argument('--model_path', type=str,required=True,
                        help='Path to model weights file')


    # Device related parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Testing device (cuda or cpu)')

    parser.add_argument('--split_path', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--hypergraph_cache', choices=['auto', 'cpu', 'cuda', 'none'], default='auto')

    return parser.parse_args()


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def test_model(args):

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    if 'Kimore' in args.dataset:
        print(f"Loading dataset: {args.dataset}")
        data_loader = Data_Loader(args.dataset)
        test_x = data_loader.scaled_x
        test_y = data_loader.scaled_y
        num_point = 25
    elif 'UI_PRMD' in args.dataset:
        print(f"Loading dataset: {args.dataset}")
        data_loader = UI_Data_Loader(args.dataset)
        test_x = data_loader.all_data_x
        test_y = data_loader.all_data_y
        num_point = 39
    elif 'MIT_Skate' in args.dataset or 'MITSkate' in args.dataset:
        print(f"Loading dataset: {args.dataset}")
        data_loader = Oly_Skate_Data_Loader(args.dataset)
        if not args.split_path:
            raise ValueError('--split_path is required for MIT-Skate testing.')
        with np.load(args.split_path, allow_pickle=False) as split:
            indices = split['test_indices']
            train_indices = split['train_indices']
        if (indices.ndim != 1 or train_indices.ndim != 1
                or not np.issubdtype(indices.dtype, np.integer)
                or not np.issubdtype(train_indices.dtype, np.integer)):
            raise ValueError('Split indices must be one-dimensional integer arrays.')
        all_indices = np.concatenate((train_indices, indices))
        if (len(train_indices) != 100 or len(all_indices) != len(data_loader.scaled_y)
                or not np.array_equal(np.sort(all_indices), np.arange(len(data_loader.scaled_y)))):
            raise ValueError('Split must partition all samples into 100 training samples and the remaining test samples.')
        test_x = data_loader.scaled_x[indices]
        test_y = data_loader.scaled_y[indices]
        num_point = 18
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


    if num_point != 18:
        if not args.split_path:
            raise ValueError('--split_path is required to restore the training-time test subset.')
        with np.load(args.split_path, allow_pickle=False) as split:
            indices = split['test_indices']
            train_indices = split['train_indices']
            val_indices = split['val_indices']
        if any(index.ndim != 1 or not np.issubdtype(index.dtype, np.integer)
               for index in (train_indices, val_indices, indices)):
            raise ValueError('Split indices must be one-dimensional integer arrays.')
        all_indices = np.concatenate((train_indices, val_indices, indices))
        if not np.array_equal(np.sort(all_indices), np.arange(len(test_y))):
            raise ValueError('Saved split must partition all input samples without overlap.')
        test_x, test_y = test_x[indices], test_y[indices]

    print(f"Initializing model, dataset: {args.dataset}, joints: {num_point}")
    print(f"Loading model weights: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_config' in checkpoint:
        config = checkpoint['model_config'].copy()
        if config.pop('dataset_name') != args.dataset:
            raise ValueError('Checkpoint dataset does not match --dataset.')
        model = module.Module(args.dataset, device=device, **config).to(device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError('This checkpoint predates the hypergraph update. Train the updated model to create a matching checkpoint.')
    model.eval()
    cache_device = HypergraphCache.select_device(getattr(args, 'hypergraph_cache', 'auto'),
                                                 device, [test_x], model.structure.temporal_k)
    cache = None
    if cache_device is not None:
        cache = HypergraphCache(model.structure, test_x, cache_device, args.batch_size, device)


    print("Starting testing...")
    with torch.no_grad():
        if args.batch_size < 1:
            raise ValueError('--batch_size must be positive.')
        output = torch.cat([
            model(torch.as_tensor(test_x[start:start + args.batch_size], dtype=torch.float32, device=device),
                  hypergraphs=cache.batch(range(start, min(start + args.batch_size, len(test_x))), device)
                  if cache is not None else None)
            for start in range(0, len(test_x), args.batch_size)
        ], dim=0)


    output = output.cpu().numpy()
    mae = mean_absolute_error(test_y, output)
    rms_dev = sqrt(mean_squared_error(output, test_y))
    mape = mean_absolute_percentage_error(test_y, output)


    print("\nModel Performance Evaluation Results:")
    if num_point == 18:
        print(f"SRC (Spearman rank correlation): {spearmanr(np.ravel(test_y), np.ravel(output))[0]:.4f}")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"RMSE (Root Mean Square Error): {rms_dev:.4f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.4f}%")



if __name__ == '__main__':
    args = parse_args()
    test_model(args)

