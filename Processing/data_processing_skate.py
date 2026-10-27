from pathlib import Path

import numpy as np
import pandas as pd


# Load pre-extracted MIT-Skate skeletons and scores.
class Oly_Skate_Data_Loader:
    def __init__(self, dir):
        self.dir = Path(dir)
        self.num_channel = 3
        self.num_joints = 18
        self.num_timestep = 4500
        self.scaled_x, self.scaled_y = self.import_dataset()
        self.train_x = self.scaled_x
        self.train_y = self.scaled_y
        self.batch_size = len(self.scaled_y)

    def import_dataset(self):
        coordinate_path = self.dir / 'train_x_4500.xlsx'
        score_path = self.dir / 'scores.xlsx'
        coordinates = pd.read_excel(coordinate_path, header=None).to_numpy(dtype=np.float32)
        scores = pd.read_excel(score_path, header=None).to_numpy(dtype=np.float32)
        if scores.ndim != 2 or scores.shape[1] != 1:
            raise ValueError('scores.xlsx must contain one score column, in video order.')
        expected_shape = (len(scores) * self.num_timestep, self.num_joints * self.num_channel)
        if coordinates.shape != expected_shape:
            raise ValueError(f'train_x_4500.xlsx must have shape {expected_shape}, got {coordinates.shape}.')
        if not np.isfinite(coordinates).all() or not np.isfinite(scores).all():
            raise ValueError('Prepared coordinates and scores must contain finite values.')
        coordinates = coordinates.reshape(len(scores), self.num_timestep, self.num_joints, self.num_channel)
        return coordinates, scores

    def preprocessing(self):
        return self.scaled_x, self.scaled_y
