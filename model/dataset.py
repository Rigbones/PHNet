import random # used for shuffling
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image # a comfortable pillow used to resize images

# the path to the processed ground truths file
GT_PATH = "./dataset/CrowdFlow/processed_density/"

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

def load_data(img_path, gt_path, train = True, direct = False, code = 1):
    # img_path: "./dataset/CrowdFlow/CrowdFlow_10_frames_mask/IM01_frame_0000"
    # gt_path = "./dataset/CrowdFlow/processed_density/"

    # define a function transform2
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    transform2 = transforms.ToTensor()

    # get the processed density path
    # processed_density_path looks sth like: "./dataset/CrowdFlow/processed_density/IM01_frame_0000.npy"
    processed_density_path = gt_path + img_path.split("/")[-1] + '.npy'
    processed_density = np.load(processed_density_path)

    for root, dir, filenames in os.walk(img_path):
        # root: "./dataset/CrowdFlow/CrowdFlow_10_frames_mask/IM01_frame_0000"
        # dir: []
        # filenames: ['IM01_frame_0000.png', ..., 'IM01_frame_0009.png']
        data_augmentation_mode = 0

        # loop through one frame by one frame...
        for i in range(0, len(filenames)):
            # if (i == 0), which is the first frame, then reshape image to (rgb, 1, 720, 1280)
            if (i == 0):
                img = np.array(Image.open(img_path + "/" + filenames[i]).convert('RGB'))
                if direct is True:
                    return transform2(img)

                # this line changes the image shape from (720, 1280, 3) to (3, 720, 1280), casts it to PyTorch tensor, and perhaps does sth else?
                img = transform(img)
                image = img

                # if data_augmentation_mode is 0, then add a 1 in dimension, turns it to (3, 1, 720, 1280)
                if data_augmentation_mode == 0:
                    image = image.unsqueeze(dim = 1)

                # if data_augmentation_mode is 1, flip the image left-right
                if data_augmentation_mode == 1:
                    image = img.transpose(Image.FLIP_LEFT_RIGHT)

                # if data_augmentation mode is 2, do what?
                if data_augmentation_mode == 2:
                    crop_size = (int(image.shape[1]/2),int(image.shape[2]/2))
                    if random.randint(0,9)<= 3:
                        dx = int(random.randint(0,1)*image.shape[1]*1./2)
                        dy = int(random.randint(0,1)*image.shape[2]*1./2)
                    else:
                        dx = int(random.random()*image.shape[1]*1./2)
                        dy = int(random.random()*image.shape[2]*1./2)
                    image = image[:,dx:crop_size[0]+dx,dy:crop_size[1]+dy]

            # if (i != 0), which is the 2nd, 3rd etc frame, then add the frame to first frame.
            else:
                new_img =  np.array(Image.open(img_path + "/" + filenames[i]).convert('RGB'))

                # this line changes the image shape from (720, 1280, 3) to (3, 720, 1280), casts it to PyTorch tensor, and perhaps does sth else?
                new_img = transform(new_img)
                new_image = new_img

                # if data_augmentation_mode is 0, then add a 1 in dimension, turns it to (3, 1, 720, 1280)
                if data_augmentation_mode == 0:
                    new_image = new_image.unsqueeze(dim = 1)

                # if data_augmentation_mode is 1, flip the image left-right
                if data_augmentation_mode == 1:
                    new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)

                # if data_augmentation mode is 2, do what?
                if data_augmentation_mode == 2:
                    new_image = new_image[:,dx:crop_size[0]+dx,dy:crop_size[1]+dy]
                    new_image = new_image.unsqueeze(dim = 1)

                # stick the (i)th frame to the (i-1)th frame
                image = torch.cat([image, new_image], axis = 1)

    # load the .h5 file and get the density array
    # 'density' is the gaussian filter result

    # if data_augmentation_mode is 1, flip the image left-right
    if data_augmentation_mode == 1:
        processed_density = np.fliplr(processed_density)

    # if data_augmentation mode is 2, do what?
    if data_augmentation_mode == 2:
        processed_density = processed_density[dx:crop_size[0]+dx,dy:crop_size[1]+dy]
    
    # resize the density map.  Originally (720, 1280) -> (180, 320) or (90, 160) for more MaxPool2d layers
    # the resizing is necessary because of the MaxPool2d layers in forward pass
    # image.shape = (rgb, frames, 720, 1280)
    dimension = (int(np.floor(image.shape[3]/8)), int(np.floor(image.shape[2]/8)))
    t = Image.fromarray(processed_density)
    t = np.array(t.resize(dimension, Image.BICUBIC), dtype = processed_density.dtype) * 64

    return image, t