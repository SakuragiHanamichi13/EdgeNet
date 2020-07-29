import numpy as np
import random
import os
import cv2

def random_crop(x_folder='./data/x_whole/', y_folder='./data/y_whole/', num_crops = 20000, x_save_folder='./data/x/', y_save_folder = './data/y/', crop_size = 64):
    for root, dir, files in os.walk(x_folder):
        img_names = files

    save_ind = 0
    for i in range(num_crops):
        rand_img_j = random.randint(0, len(img_names)-1)

        x = cv2.imread(x_folder + img_names[rand_img_j], 0)
        y = cv2.imread(y_folder + img_names[rand_img_j], 0)

        rand_row = random.randint(0, x.shape[0] - crop_size - 1)
        rand_col = random.randint(0, x.shape[1] - crop_size - 1)

        x_crop = x[rand_row:rand_row + crop_size, rand_col:rand_col + crop_size]
        y_crop = y[rand_row:rand_row + crop_size, rand_col:rand_col + crop_size]

        x_crop_save_name = x_save_folder + str(save_ind) + '.png'
        y_crop_save_name = y_save_folder + str(save_ind) + '.png'

        cv2.imwrite(x_crop_save_name, x_crop)
        cv2.imwrite(y_crop_save_name, y_crop)


        save_ind+=1



            
