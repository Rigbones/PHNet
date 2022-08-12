import random
from torch.utils.data import Dataset
from image import *
from torchvision import transforms
import torchvision.transforms.functional as F

# the path to the .h5 file
GT_PATH = "./dataset/CrowdFlow/density_map_init/"

# This program is called during the "for i,(img, target) in enumerate(train_loader)" in train.py

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, direct=False, batch_size=1, num_workers=4, gt_code=1):
        if train:
            root = root
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.direct=direct 
        self.gt_code = gt_code
    
    def __len__(self):
        # the i in "for i,(img, target) in enumerate(train_loader)" calls this function

        return self.nSamples

    def __getitem__(self, index):
        # the (img, target) in "for i,(img, target) in enumerate(train_loader)" calls this function
        # returns:
        #   img: a torch tensor of shape (rgb, frames, 720, 1280)
        #   target: a torch tensor of shape 
        assert index <= len(self), 'index range error'         
        img_path = self.lines[index]
        # calls load_data function in image.py
        img, target = load_data(img_path, GT_PATH, False, code=self.gt_code)
        img_r = load_data(img_path, GT_PATH, False, direct=True, code=self.gt_code)
        if self.direct:
            return img, target, img_r
        return img, target