

import torch
import random
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

Left_head_front = 0            #左前头
Right_head_front = 1           #右前头
Left_back_head = 2             #左后头
Rright_back_head = 3           #右后头
cervical_vertebra = 4          #颈椎椎骨
thoracic_vertabra = 5          #胸椎
Clavicle = 6                   #锁骨
Sternum = 7                    #胸骨
Right_back = 8                 #右后背
Left_shoulder = 9              #左肩
Left_upper_arm = 10            #左大臂
Left_elbow = 11                #左肘
Left_forearm = 12              #左小臂
Left_wrist_A = 13              #
Left_wrist_B = 14
Left_finger = 15               #左手指
Right_shoulder = 16            #右肩
Right_upper_arm=17             #右大臂
Right_elbow=18                 #右肘
Right_forearm=19               #右小臂
Right_wrist_A=20               #
Right_wrist_B=21
Right_finger=22                #右手指
Left_ASIS=23             #left anterior superior lliac 左前上韧带
Right_ASIS=24
Left_PSIS=25             #left posterior superior lliac 左后上韧带
Right_PSIS=26
Left_thigh=27            #左大腿
Left_knee=28             #左膝盖
Left_tibia=29            #左胫骨
Left_ankle=30            #左脚踝
Left_heel=31             #左足跟
Left_toe=32
Right_thigh=33           #右大腿
Right_knee=34
Right_tibia=35
Right_ankle=36
Right_heel=37            #右足跟
Right_toe=38

random.seed(42)

class UI_Data_Loader():
    def __init__(self,dir):
        self.num_channel = 3
        self.dir = dir
        self.body_parts = self.body_parts()
        self.num_timestep = 240
        self.batch_size = 90
        self.num_joints = 39
        self.cor_x, self.cor_y,self.incor_x,self.incor_y = self.import_dataset()            #得到正确错误数据集的数据和标签
        self.sc1 = StandardScaler()  # 将数据归一化和标准化的类，计算训练集的平均值和标准差，以便测试数据集使用相同的变换
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
        base_path = r"C:\Deep learning\STGCN-rehab-main"
        #UI-PRMD
        # cor_x = pd.read_csv("./" + self.dir + "/Data_Correct.csv", header=None).iloc[:, :].values        #./是所在目录里 ../是上一级目录 所以命令行运行和直接运行的目录不同
        # cor_y = pd.read_csv("./" + self.dir + "/Labels_Correct.csv", header=None).iloc[:, :].values
        #
        # incor_x = pd.read_csv("./" + self.dir + "/Data_Incorrect.csv", header=None).iloc[:,:].values  # ./是所在目录里 ../是上一级目录 所以命令行运行和直接运行的目录不同
        # incor_y = pd.read_csv("./" + self.dir + "/Labels_Incorrect.csv", header=None).iloc[:, :].values

        cor_x = pd.read_csv(base_path + "/" + self.dir + "/Data_Correct.csv", header=None).iloc[:, :].values
        cor_y = pd.read_csv(base_path + "/" + self.dir + "/Labels_Correct.csv", header=None).iloc[:, :].values

        incor_x = pd.read_csv(base_path + "/" + self.dir + "/Data_Incorrect.csv", header=None).iloc[:, :].values
        incor_y = pd.read_csv(base_path + "/" + self.dir + "/Labels_Incorrect.csv", header=None).iloc[:, :].values

        return cor_x, cor_y, incor_x, incor_y
    def preprocessing(self):
        #处理正确的数据集
        X_cor = np.zeros((self.batch_size, self.cor_x.shape[1], 117)).astype('float32')  # (90,240,117)
        for i in range(len(self.cor_x) // 117):
            X_cor[i, :, :] = np.transpose(self.cor_x[117 * i:117 * (i + 1), :])  #把元数据进行转置 得到(90,240,117)

        Xcor = X_cor.reshape(-1,117)                                             #变成(90*240,117)的矩阵,下面的for循环才能赋值

        y_cor = np.reshape(self.cor_y, (-1, 1))

        X_cor_ = np.zeros((self.batch_size, self.num_timestep, self.num_joints, self.num_channel))

        for batch in range(X_cor_.shape[0]):
            for timestep in range(X_cor_.shape[1]):
                for node in range(X_cor_.shape[2]):
                    for channel in range(X_cor_.shape[3]):
                        X_cor_[batch, timestep, node, channel] = Xcor[timestep + (batch * self.num_timestep),channel + (node * self.num_channel)]
                        #(90,240,39,3)

        X_cor = torch.tensor(X_cor_,dtype=torch.double)

        #处理错误的数据集
        X_incor = np.zeros((self.batch_size, self.incor_x.shape[1], 117)).astype('float32')  # (90,240,117)
        for i in range(len(self.cor_x) // 117):
            X_incor[i, :, :] = np.transpose(self.incor_x[117 * i:117 * (i + 1), :])  # 把元数据进行转置 得到(90,240,117)

        Xincor = X_incor.reshape(-1, 117)  # 变成(90*240,117)的矩阵,下面的for循环才能赋值

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

        #中间一次的划分
        middle1 = np.setdiff1d(np.arange(0, 90, 1), trainidx1)
        middle2 = np.setdiff1d(np.arange(0, 90, 1), trainidx2)

        #验证集
        val1 = random.sample(range(0, middle1.shape[0]), int(36 * 0.5))
        val2 = random.sample(range(0, middle2.shape[0]), int(36 * 0.5))

        #测试集
        test1 = np.setdiff1d(np.arange(0, 36, 1), val1)
        test2 = np.setdiff1d(np.arange(0, 36, 1), val2)

        train_x = np.concatenate((X_cor[trainidx1, :, :], X_incor[trainidx2, :, :]))
        train_y = np.concatenate((np.squeeze(y_cor[trainidx1]), np.squeeze(y_incor[trainidx2])))

        val_x = np.concatenate((X_cor[val1, :, :], X_incor[val2, :, :]))
        val_y = np.concatenate((np.squeeze(y_cor[val1]), np.squeeze(y_incor[val2])))

        test_x = np.concatenate((X_cor[test1, :, :], X_incor[test2, :, :]))
        test_y = np.concatenate((np.squeeze(y_cor[test1]), np.squeeze(y_incor[test2])))

        #扩展最后一维
        train_y = train_y[:,None]
        val_y = val_y[:,None]
        test_y = test_y[:,None]

        # 新增：构建完整的数据集（所有正确和错误数据的合并）
        all_data_x = np.concatenate((X_cor[:, :, :], X_incor[:, :, :]))
        all_data_y = np.concatenate((np.squeeze(y_cor), np.squeeze(y_incor)))
        all_data_y = all_data_y[:, None]  # 扩展维度为 (180, 1)

        return train_x, train_y, val_x, val_y, test_x, test_y, all_data_x, all_data_y

if __name__ == '__main__':
    DL = UI_Data_Loader("UI_PRMD_ex4")    #这个类的作用是对UI_PRMD数据集处理,得到(90,240,39,3)的数据 表示90次重复动作,240帧,39个关节的三维坐标
    DL.preprocessing()
    #print("train_x",DL.train_x)
    print("train_y",DL.train_y)
    #print("val_x",DL.val_x)
    print("val_y",DL.val_y)
    #print("test_x",DL.test_x)
    print("test_y",DL.test_y)

    print(f"\n完整数据集形状:")
    print(f"all_data_x: {DL.all_data_x.shape}")  # 应该是 (180, 240, 39, 3)
    print(f"all_data_y: {DL.all_data_y.shape}")

    print(f"\n各数据集形状汇总:")
    print(f"train_x: {DL.train_x.shape}")
    print(f"train_y: {DL.train_y.shape}")
    print(f"val_x: {DL.val_x.shape}")
    print(f"val_y: {DL.val_y.shape}")
    print(f"test_x: {DL.test_x.shape}")
    print(f"test_y: {DL.test_y.shape}")


    #UI-PRMD数据集包含10530行，240列
    # 每一列是一帧，一共240帧，每一列是所有关节点的三维坐标
    #每117行是一个完整的动作，也就是240帧下39个关节点3维坐标
    #一共是10530行，共90个动作
    #UI-PRMD 数据集：包含 90 个三维人体动作，每个动作由 39 个关节点的 3D 坐标（共117维）在 240 帧下的运动序列构成，总数据规模为 10530 行 × 240 列。