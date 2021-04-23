# -*- coding: utf-8 -*-
"""debrisdetection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18p3NnsTJXP_U0nZV6Gm5Fz6DAY6nC3fX
"""

from google.colab import drive
drive.mount('/content/drive')

!cp -r '/content/drive/MyDrive/debris detection' '/content'

import os
try:
  os.mkdir('/content/train')
  os.mkdir('/content/val')
  os.mkdir('/content/test')
except OSError as error:
  print(error)

!cp '/content/debris detection/train.zip' '/content/train/train.zip'
!unzip '/content/train/train.zip' -d '/content/train'
!rm -f '/content/train/train.zip'

!cp '/content/debris detection/val.zip' '/content/val/val.zip'
!unzip '/content/val/val.zip' -d '/content/val'
!rm -f '/content/val/val.zip'

!cp '/content/debris detection/val.csv' '/content/val.csv'
!cp '/content/debris detection/train.csv' '/content/train.csv'

!git clone https://github.com/yhenon/pytorch-retinanet.git
!apt-get install tk-dev python-tk
!pip install pandas
!pip install pycocotools
!pip install opencv-python
!pip install requests

#!wget https://drive.google.com/open?id=1yLmjq3JtXi841yXWBxst0coAgR26MNBS
!gdown --id 1yLmjq3JtXi841yXWBxst0coAgR26MNBS

import torch
import torchvision.models as models

retinanet = models.resnet50(num_classes=3,)
retinanet.load_state_dict(torch.load('/content/coco_resnet_50_map_0_335_state_dict.pt'),strict=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests
import urllib
import os
from PIL import Image

from ast import literal_eval
df_train = pd.read_csv('/content/train.csv',index_col=False)
df_train['bboxes'] = df_train['bboxes'].apply(literal_eval)
data_train = df_train.to_numpy()
print(data_train[0:5])

from tqdm import tqdm
output_arr = []

for imageid,bboxes in tqdm(data_train):
    
    for vals in bboxes:
        image_path = '/content/train/'
        arr = [image_path+str(imageid)+'.jpg',vals[0],vals[2],vals[1],vals[3],'debris']
        output_arr.append(arr)
    


dataset = pd.DataFrame(output_arr,columns=['ImageID','xmin','ymin','xmax','ymax','class'])
print(dataset.head())

dataset.to_csv('./training_new.csv',index=False,header=None)

df_val = pd.read_csv('/content/val.csv',index_col=False)
df_val['bboxes'] = df_val['bboxes'].apply(literal_eval)
data_val = df_val.to_numpy()
print(data_val[0:5])

output_arr = []

for imageid,bboxes in tqdm(data_val):
    
    for vals in bboxes:
        image_path = '/content/val/'
        arr = [image_path+str(imageid)+'.jpg',vals[0],vals[2],vals[1],vals[3],'debris']
        output_arr.append(arr)
    


dataset = pd.DataFrame(output_arr,columns=['ImageID','xmin','ymin','xmax','ymax','class'])
print(dataset.head())
dataset.to_csv('./val_new.csv',index=False,header=None)

!head /content/training_new.csv

!head /content/val_new.csv

classes = ['debris']
with open('/content/classes.csv', 'w') as f:
  for i, class_name in enumerate(classes):
    f.write(f'{class_name},{i}\n')

!head /content/classes.csv

!python '/content/pytorch-retinanet/train.py' --dataset csv --csv_train '/content/training_new.csv' --csv_classes '/content/classes.csv' --csv_val '/content/val_new.csv' --epochs 5

!cp '/content/debris detection/test.zip' '/content/test/test.zip'
!unzip '/content/test/test.zip' -d '/content/test'
!rm -f '/content/test/test.zip'

2+2

# %matplotlib inline
# !python '/content/pytorch-retinanet/visualize_single_image.py' --image_dir '/content/test/0.jpg' --model_path '/content/csv_retinanet_4.pt' --class_list '/content/classes.csv'

!cp '/content/train/0.jpg' '/content/test/0.jpg'

!rm -rf '/content/test'
!ls -lrt '/content/test'

os.mkdir('/content/test')

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
!python '/content/pytorch-retinanet/dum.py' --image_dir '/content/test/' --model_path '/content/csv_retinanet_4.pt' --class_list '/content/classes.csv'