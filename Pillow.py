# Author: Nguyen Duong

### Import and Load Packages

import os
import numpy as np
from os import listdir
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

### Set Directory

os.chdir('/Users/ngyduong/Desktop')

# load all images in the directory as a list of arrays

data = np.zeros(shape=(100,588,588))

i = 0
for filename in listdir('photo'):
    if filename == ".DS_Store":
        pass
    else:
        image = Image.open('photo/'+filename).convert('L')
        image = img_to_array(image)
        image = image.reshape((1,588,588))
        print(image.shape)
        print(type(image))
        data[i] = image
        i+=1


# folder = "/Users/ngyduong/Desktop/photo"
# onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
# print("Working with {0} images".format(len(onlyfiles)))
#
#
# train_files = []
# y_train = []
# i = 0
# for _file in onlyfiles:
#     train_files.append(_file)
#     label_in_file = _file.find(".")
#     y_train.append(_file[0:label_in_file])
# print("Files in train_files: %d" % len(train_files))

