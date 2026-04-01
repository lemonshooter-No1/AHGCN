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

    return parser.parse_args()


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def test_model(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
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
        test_x = data_loader.test_x
        test_y = data_loader.test_y
        num_point = 39
    elif 'MIT_Skate' in args.dataset:
        print(f"Loading dataset: {args.dataset}")
        data_loader = Oly_Skate_Data_Loader(args.dataset)
        test_x = data_loader.test_x
        test_y = data_loader.test_y
        num_point = 18
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


    print(f"Initializing model, dataset: {args.dataset}, joints: {num_point}")
    model = module.Module(args.dataset, device=device).to(device)


    print(f"Loading model weights: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()


    print("Starting testing...")
    with torch.no_grad():
        output = model(test_x, False, False)


    output = output.cpu().numpy()
    mae = mean_absolute_error(test_y, output)
    rms_dev = sqrt(mean_squared_error(output, test_y))
    mape = mean_absolute_percentage_error(test_y, output)

    # 打印结果
    print("\nModel Performance Evaluation Results:")
    print(f"MAE (Mean Absolute Error): {mae:.4f}")
    print(f"RMSE (Root Mean Square Error): {rms_dev:.4f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.4f}%")



if __name__ == '__main__':
    args = parse_args()
    test_model(args)

