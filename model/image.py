import random
import os
from PIL import Image
import numpy as np
import h5py
import cv2
import torch
from torchvision import transforms

def load_data(img_path, gt_path, train = True, direct = False, code = 1):
    # img_path: "./dataset/Venice/venice/ablation3\\4896_004140.jpg"
    # gt_path = "./dataset/Venice/venice/density_map_init/"

    # define a function transform2
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
    transform2 = transforms.ToTensor()

    # get the ground truth path, which points to the .h5 files.
    # hf (the .h5) file is a dictionary with keys 'density' and 'roi'
    # 'density' is the gaussian filter result
    # 'roi' is matrix of 0 and 1
    img_name = img_path.split("/")[-1].replace('jpg', '.h5').replace('.png', '.h5')
    gt_path = gt_path + img_name

    for root, dir, filenames in os.walk(img_path):
        # root: './dataset/Venice/venice/ablation3\\4896_004140.jpg'
        # dir: []
        # filenames: ['4896_004020.jpg', '4896_004080.jpg', '4896_004140.jpg']
        
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
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    # if data_augmentation_mode is 1, flip the image left-right
    if data_augmentation_mode == 1:
        target = np.fliplr(target)

    # if data_augmentation mode is 2, do what?
    if data_augmentation_mode == 2:
        target = target[dx:crop_size[0]+dx,dy:crop_size[1]+dy]
    
    # resize the gaussian filter result.  Originally (720, 1280) -> (90, 160)
    # image.shape = (rgb, frames, 720, 1280)
    target_dimension = (int(np.floor(image.shape[3]/8)), int(np.floor(image.shape[2]/8)))
    target = 64 * cv2.resize(target, target_dimension, interpolation = cv2.INTER_CUBIC)

    return image,target