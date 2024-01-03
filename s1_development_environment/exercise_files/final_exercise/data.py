import torch
from torch.utils.data import Dataset, TensorDataset
import glob
def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    train_images_list = glob.glob("../../../data/corruptmnist/train_images_*.pt")
    train_images_list = [torch.load(x).unsqueeze(1) for x in train_images_list]
    train_images = torch.cat(train_images_list, dim=0)
    print('Train set:', train_images.shape)

    train_target_list = glob.glob("../../../data/corruptmnist/train_target_*.pt")
    train_target_list = [torch.load(x) for x in train_target_list]
    train_target = torch.cat(train_target_list, dim=0)

    train = TensorDataset(train_images, train_target)

    test_images = torch.load("../../../data/corruptmnist/test_images.pt").unsqueeze(1)
    print('Test set:', test_images.shape)
    test_target = torch.load("../../../data/corruptmnist/test_target.pt")
    test = TensorDataset(test_images, test_target)

    return train, test
