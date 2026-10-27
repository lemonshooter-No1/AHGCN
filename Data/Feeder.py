from torch.utils.data import Dataset


# Expose skeleton sequences and scores to the data loader.
class Feeder(Dataset):
    def __init__(self,data,label,return_index=False):
        self.data = data
        self.label = label
        self.return_index = return_index

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        if self.return_index:
            return data,label,index
        return data,label

    def __len__(self):
        return len(self.data)
