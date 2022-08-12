# My imports
import os # used for all the path stuff
import re # used in searchFile function
import shutil # used in mycopyfile function
import numpy as np # used to load the npy files
import matplotlib.pyplot as plt # used for debugging and reading, writing png images
import json # used when writing to the json file
from fancy_progress_bar import Printer # a more sexy progress bar that has some bugs

# Run this program in folder "PHNet_pytorch".
# Use the command "python ./datasets/CrowdFlow/4.preprocessing.py"

DATASET_NAME = "CrowdFlow"
NUM_FRAMES = 10
MULTIPLY_MASK = False # If you have a mask and wish to multiply it, set to True

GROUND_TRUTH_DENSITY_PATH = './dataset/CrowdFlow/density' # ground truth density map dataset location, each map is in npy format
PROCESSED_DENSITY_PATH = './dataset/CrowdFlow/processed_density' # where to place the density maps multiplied by the mask, each map is in npy format
MASK_PATH = './dataset/CrowdFlow/mask' # mask matrix location, each map is in png format
IMAGES_PATH = './dataset/CrowdFlow/images' # where the training and testing png images are, each picture is in png format
PROCESSED_IMAGES_PATH = './dataset/CrowdFlow/processed_images' # where to place the images multiplied by the mask, each image is in png format
EXPORT_FRAMES_PATH = f'./dataset/CrowdFlow/{DATASET_NAME}_{NUM_FRAMES}_frames' + ('_mask' if MULTIPLY_MASK else '_no_mask') # where chronological frames are placed in one folder

# EXPORT_FRAMES_PATH would be sth like: './dataset/CrowdFlow/CrowdFlow_10_frames_mask' or './dataset/CrowdFlow/CrowdFlow_10_frames_no_mask'

def searchFile(pathname, filename):
    # searches the path (and all its subfolders) for files ending with a certain extension
    matchedFile =[]
    for root, dirs, files in os.walk(pathname):
        for file in files:
            if re.match(filename,file):
                matchedFile.append((root,file))
    return matchedFile

def mycopyfile(srcfile, dstfile):
    # Copies the file from path srcfile to path dstfile.
    # Creates the directory if it doesnt exist
    if not os.path.isfile(srcfile):
        print("not exist!"%(srcfile))
    else:
        fpath, fname=os.path.split(dstfile)   
        if not os.path.exists(fpath):
            os.makedirs(fpath)               
        shutil.copyfile(srcfile,dstfile)     
        # print("copy %s -> %s"%( srcfile,dstfile))

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
#  Load the density map (.npy), mask images (.png), actual image (.png)  #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

# searches the folder "density"
# creates a list of length 1600.  Each list_density[i] is a tuple
# eg list_density[0] is ('./dataset/CrowdFlow/density/IM01', 'IM01_frame_0000.npy')
list_density = searchFile(GROUND_TRUTH_DENSITY_PATH, '(.*).npy')
list_density.sort()

# searches the folder "mask" if MULTIPLY_MASK is True
# a list of length 1600.  Each list_mask[i] is a tuple
# eg list_mask[0] is ('./dataset/CrowdFlow/mask/IM01', 'IM01_frame_0000.png')
if (MULTIPLY_MASK):
    list_mask = searchFile(MASK_PATH, '(.*).png')
    list_mask.sort()

# searches the folder "images"
# creates a list of length 1600.  Each list_images[i] is a tuple
# eg list_images[0] is ('./dataset/CrowdFlow/images/IM01', 'IM01_frame_0000.png')
list_images = searchFile(IMAGES_PATH, '(.*).png')
list_images.sort()

total_images = len(list_density)


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
#  Multiply density * mask and save to "processed_density" folder  #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

progressBar1 = Printer(total_images, 'Multiply density * mask and save to "processed_density" folder')
for i in range(0, total_images):
    # i = 0, 1, ..., 1599

    # Update the fancy progress bar
    progressBar1.print(i)

    # creates the "processed_density" folder if it does not exist
    if (not os.path.exists(PROCESSED_DENSITY_PATH)):
        os.makedirs(PROCESSED_DENSITY_PATH)

    # Get the destination path for an processed density map (processed means multiplied by mask)
    # destination_density_path looks sth like: "./dataset/CrowdFlow/processed_density/IM01_frame_0000.npy"
    destination_density_path = os.path.join(PROCESSED_DENSITY_PATH, list_density[i][1])

    # Get the source path for an UNprocessed density map (processed means multiplied by mask)
    # source_density_path looks sth like: "./dataset/CrowdFlow/density/IM01/IM01_frame_0000.npy"
    source_density_path = os.path.join(list_density[i][0], list_density[i][1])
    original_density = np.load(source_density_path)

    # Get the path for the mask
    # mask_path looks sth like: "./dataset/CrowdFlow/mask/IM01/IM01_frame_0000.png"
    if (MULTIPLY_MASK):
        mask_path = os.path.join(list_mask[i][0], list_mask[i][1])
        mask = plt.imread(mask_path)

    # Multiply density * mask if (MULTIPLY_MASK == True)
    if (MULTIPLY_MASK):
        original_density = original_density * mask[:, :, 0] # numpy broadcasting: (720, 1280, 3) * (720, 1280, 1)
        
    # Save the processed density map to "processed_density" folder in both npy and png formats
    # the png format will not be used later in the program.  Just for visualisation.
    np.save( destination_density_path, original_density )
    plt.imsave( destination_density_path.replace('.npy', '.png'), original_density )

progressBar1.finish()


#-#-#-#-#-#-#-#-#-#-#-#-#-#
#  Multiply image * mask  #
#-#-#-#-#-#-#-#-#-#-#-#-#-#


progressBar2 = Printer(total_images, 'Multiply image * mask')
for i in range(0, total_images):
    # i = 0, 1, ..., 1599

    # Update the fancy progress bar
    progressBar2.print(i)

    # creates the "processed_images" folder if it does not exist
    if (not os.path.exists(PROCESSED_IMAGES_PATH)):
        os.makedirs(PROCESSED_IMAGES_PATH)

    # Get the path for the img
    # img_path looks sth like: "./dataset/CrowdFlow/images/IM01/IM01_frame_0000.png"
    img_path = os.path.join(list_images[i][0], list_images[i][1])
    img = plt.imread(img_path)

    # Get the path for the mask
    # mask_path looks sth like: "./dataset/CrowdFlow/mask/IM01/IM01_frame_0000.png"
    if (MULTIPLY_MASK):
        mask_path = os.path.join(list_mask[i][0], list_mask[i][1])
        mask = plt.imread(mask_path)

    # Get the path for the processed image
    # destination_img_path looks sth like: "./dataset/CrowdFlow/processed_images/IM01_frame_0000.png"
    destination_img_path = os.path.join(PROCESSED_IMAGES_PATH, list_images[i][1])

    # Multiply image * mask
    if (MULTIPLY_MASK):
        img = img * mask[:, :, 0:1] # numpy broadcasting: (720, 1280, 3) * (720, 1280, 1)

    # Save the processed image to "processed_images" folder in png format
    plt.imsave( destination_img_path, img )

progressBar2.finish()


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-
#  Stick the frames in 1 folder, save to EXPORT_FRAMES_PATH  #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

train_json = []
test_json = []

# searches the folder "processed_images"
# a list of length 1600.  Each processed_images[i] is a tuple
# eg processed_images[0] is ('./dataset/CrowdFlow/processed_images', 'IM01_frame_0000.png')
list_processed_images = searchFile(PROCESSED_IMAGES_PATH, '(.*).png')
list_processed_images.sort()

progressBar3 = Printer(total_images - NUM_FRAMES + 1, 'Stick the frames in 1 folder, save to EXPORT_FRAMES_PATH')
for i in range(0, total_images - NUM_FRAMES + 1):
    # i goes from 0, 1, ..., 1590 (not 1591!)

    # Update the fancy progress bar
    progressBar3.print(i)

    for frame_num in range(0, NUM_FRAMES):
        # frame_num goes from 0, 1, ..., 9

        # Get the path for the processed image
        # processed_img_path looks sth like: './dataset/CrowdFlow/processed_images/IM01_frame_0007.png'
        processed_img_path = os.path.join(list_processed_images[i][0], list_processed_images[i + frame_num][1])

        # Get the destination path
        # destination_path looks sth like: './dataset/CrowdFlow/CrowdFlow_10_frames_mask/IM01_frame_0000/IM01_frame_0007.png'
        destination_path = os.path.join(EXPORT_FRAMES_PATH, list_processed_images[i + NUM_FRAMES - 1][1][:-4], list_processed_images[i + frame_num][1])
        
        # copy the file over
        mycopyfile(processed_img_path, destination_path)
        
    # Fill up the .json file.
    # IM01, IM02, IM03 is used for training
    # IM04, IM05 is used for testing
    IM0X = list_processed_images[i + NUM_FRAMES - 1][1].split('_')[0] # get what IM0? number it is
    # path looks sth like: 
    path = os.path.join(EXPORT_FRAMES_PATH, list_processed_images[i + NUM_FRAMES - 1][1][:-4])

    if (IM0X == "IM01" or IM0X == "IM02" or IM0X == "IM03"):
        train_json.append(path)
    elif (IM0X == "IM04" or IM0X == "IM05"):
        test_json.append(path)
    else:
        print("gg nah")

progressBar3.finish()


#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
#  Copy EXPORT_FRAMES_PATH to a json file #
#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#
        
# Filename
train_filename = f'./jsons/{DATASET_NAME}_train_{NUM_FRAMES}_frames.json'
test_filename = f'./jsons/{DATASET_NAME}_test_{NUM_FRAMES}_frames.json'

# Creates a json file from the "train_json" and "test_json" python lists
with open(train_filename, 'w', encoding='utf-8') as json_file:
    json.dump(train_json, json_file, indent=1)
    print("Length of train_json: ", len(train_json))

with open(test_filename, 'w', encoding='utf-8') as json_file:
    json.dump(test_json, json_file, indent=1)
    print("Length of test_json: ", len(test_json))