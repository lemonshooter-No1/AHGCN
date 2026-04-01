from torch.utils.data import Dataset


class Feeder(Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data,label

    def __len__(self):
        return len(self.data)
