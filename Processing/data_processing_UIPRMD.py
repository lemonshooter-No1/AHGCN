from pathlib import Path


import torch
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

Left_head_front = 0
Right_head_front = 1
Left_back_head = 2
Rright_back_head = 3
cervical_vertebra = 4
thoracic_vertabra = 5
Clavicle = 6
Sternum = 7
Right_back = 8
Left_shoulder = 9
Left_upper_arm = 10
Left_elbow = 11
Left_forearm = 12
Left_wrist_A = 13              #
Left_wrist_B = 14
Left_finger = 15
Right_shoulder = 16
Right_upper_arm=17
Right_elbow=18
Right_forearm=19
Right_wrist_A=20               #
Right_wrist_B=21
Right_finger=22
Left_ASIS=23
Right_ASIS=24
Left_PSIS=25
Right_PSIS=26
Left_thigh=27
Left_knee=28
Left_tibia=29
Left_ankle=30
Left_heel=31
Left_toe=32
Right_thigh=33
Right_knee=34
Right_tibia=35
Right_ankle=36
Right_heel=37
Right_toe=38

random.seed(42)

# Load and preprocess UI-PRMD exercise sequences.
class UI_Data_Loader():
    def __init__(self,dir):
        self.num_channel = 3
        self.dir = dir
        self.body_parts = self.body_parts()
        self.num_timestep = 240
        self.batch_size = 90
        self.num_joints = 39
        self.cor_x, self.cor_y,self.incor_x,self.incor_y = self.import_dataset()
        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        self.train_x, self.train_y, self.val_x,  self.val_y, self.test_x, self.test_y, self.all_data_x, self.all_data_y = self.preprocessing()
        pass

    def body_parts(self):

        body_parts = [Left_head_front, Right_head_front, Left_back_head, Rright_back_head, cervical_vertebra,
                      thoracic_vertabra, Clavicle, Sternum, Right_back, Left_shoulder, Left_upper_arm, Left_elbow,
                      Left_forearm, Left_wrist_A, Left_wrist_B, Left_finger, Right_shoulder, Right_upper_arm,
                      Right_elbow, Right_forearm,
                      Right_wrist_A, Right_wrist_B, Right_finger, Left_ASIS, Right_ASIS, Left_PSIS, Right_PSIS,
                      Left_thigh, Left_knee, Left_tibia, Left_ankle, Left_heel, Left_toe, Right_thigh, Right_knee,
                      Right_tibia, Right_ankle, Right_heel, Right_toe]
        return body_parts

    def import_dataset(self):
        dataset_path = Path(self.dir)
        cor_x = pd.read_csv(dataset_path / 'Data_Correct.csv', header=None).values
        cor_y = pd.read_csv(dataset_path / 'Labels_Correct.csv', header=None).values
        incor_x = pd.read_csv(dataset_path / 'Data_Incorrect.csv', header=None).values
        incor_y = pd.read_csv(dataset_path / 'Labels_Incorrect.csv', header=None).values
        return cor_x, cor_y, incor_x, incor_y

    def preprocessing(self):

        X_cor = np.zeros((self.batch_size, self.cor_x.shape[1], 117)).astype('float32')  # (90,240,117)
        for i in range(len(self.cor_x) // 117):
            X_cor[i, :, :] = np.transpose(self.cor_x[117 * i:117 * (i + 1), :])

        Xcor = X_cor.reshape(-1,117)

        y_cor = np.reshape(self.cor_y, (-1, 1))

        X_cor_ = np.zeros((self.batch_size, self.num_timestep, self.num_joints, self.num_channel))

        for batch in range(X_cor_.shape[0]):
            for timestep in range(X_cor_.shape[1]):
                for node in range(X_cor_.shape[2]):
                    for channel in range(X_cor_.shape[3]):
                        X_cor_[batch, timestep, node, channel] = Xcor[timestep + (batch * self.num_timestep),channel + (node * self.num_channel)]
                        #(90,240,39,3)

        X_cor = torch.tensor(X_cor_,dtype=torch.double)


        X_incor = np.zeros((self.batch_size, self.incor_x.shape[1], 117)).astype('float32')  # (90,240,117)
        for i in range(len(self.cor_x) // 117):
            X_incor[i, :, :] = np.transpose(self.incor_x[117 * i:117 * (i + 1), :])

        Xincor = X_incor.reshape(-1, 117)

        y_incor = np.reshape(self.incor_y, (-1, 1))

        X_incor_ = np.zeros((self.batch_size, self.num_timestep, self.num_joints, self.num_channel))

        for batch in range(X_incor_.shape[0]):
            for timestep in range(X_incor_.shape[1]):
                for node in range(X_incor_.shape[2]):
                    for channel in range(X_incor_.shape[3]):
                        X_incor_[batch, timestep, node, channel] = Xincor[
                            timestep + (batch * self.num_timestep), channel + (node * self.num_channel)]
                            # (90,240,39,3)

        X_incor = torch.tensor(X_incor_, dtype=torch.double)

        trainidx1 = random.sample(range(0, X_cor.shape[0]), int(90 * 0.6))
        trainidx2 = random.sample(range(0, X_incor.shape[0]), int(90 * 0.6))


        middle1 = np.setdiff1d(np.arange(0, 90, 1), trainidx1)
        middle2 = np.setdiff1d(np.arange(0, 90, 1), trainidx2)


        val1 = random.sample(range(0, middle1.shape[0]), int(36 * 0.5))
        val2 = random.sample(range(0, middle2.shape[0]), int(36 * 0.5))


        test1 = np.setdiff1d(np.arange(0, 36, 1), val1)
        test2 = np.setdiff1d(np.arange(0, 36, 1), val2)

        train_x = np.concatenate((X_cor[trainidx1, :, :], X_incor[trainidx2, :, :]))
        train_y = np.concatenate((np.squeeze(y_cor[trainidx1]), np.squeeze(y_incor[trainidx2])))

        val_x = np.concatenate((X_cor[val1, :, :], X_incor[val2, :, :]))
        val_y = np.concatenate((np.squeeze(y_cor[val1]), np.squeeze(y_incor[val2])))

        test_x = np.concatenate((X_cor[test1, :, :], X_incor[test2, :, :]))
        test_y = np.concatenate((np.squeeze(y_cor[test1]), np.squeeze(y_incor[test2])))


        train_y = train_y[:,None]
        val_y = val_y[:,None]
        test_y = test_y[:,None]


        all_data_x = np.concatenate((X_cor[:, :, :], X_incor[:, :, :]))
        all_data_y = np.concatenate((np.squeeze(y_cor), np.squeeze(y_incor)))
        all_data_y = all_data_y[:, None]

        return train_x, train_y, val_x, val_y, test_x, test_y, all_data_x, all_data_y

if __name__ == '__main__':
    DL = UI_Data_Loader("UI_PRMD_ex4")
    DL.preprocessing()
    #print("train_x",DL.train_x)
    print("train_y",DL.train_y)
    #print("val_x",DL.val_x)
    print("val_y",DL.val_y)
    #print("test_x",DL.test_x)
    print("test_y",DL.test_y)

    print(f"\n完整数据集形状:")
    print(f"all_data_x: {DL.all_data_x.shape}")
    print(f"all_data_y: {DL.all_data_y.shape}")

    print(f"\n各数据集形状汇总:")
    print(f"train_x: {DL.train_x.shape}")
    print(f"train_y: {DL.train_y.shape}")
    print(f"val_x: {DL.val_x.shape}")
    print(f"val_y: {DL.val_y.shape}")
    print(f"test_x: {DL.test_x.shape}")
    print(f"test_y: {DL.test_y.shape}")
