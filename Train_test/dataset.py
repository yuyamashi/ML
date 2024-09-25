import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image


class FashionDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.csv_path = csv_path
        self.img_path = '../Data/All_images/'+df['img']
        self.y = df['label']
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_path[index])
        if self.transform is not None:
            image = self.transform(image)
        label = self.y[index]

        return image, label
    
    def __len__(self):
        
        return len(self.img_path)

def get_data(batch_size):

    transform_train = transforms.Compose([
                                transforms.ToTensor()
                                ])
    transform_test = transforms.Compose([
                                transforms.ToTensor()
                                ])

    train_dataset = FashionDataset(
                        csv_path='../Data/train-majority-vote.csv',
                        transform = transform_train
                        )
    test_dataset = FashionDataset(
                        csv_path='../Data/test.csv',
                        transform = transform_test
                        )
                        
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    test_data = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)

    return train_data, test_data