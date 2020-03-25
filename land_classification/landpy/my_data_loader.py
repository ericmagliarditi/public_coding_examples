from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import torch.utils.data as data
import torch
import glob
import os


class MyDataLoader(data.Dataset):
  def __init__(self, folder_path, label_size, image_transforms=None, mask_transforms=None):
    self.image_transforms = image_transforms
    self.mask_transforms = mask_transforms
    self.img_files = glob.glob(os.path.join(folder_path,'sat_images','*.jpg'))
    self.mask_files = []
    self.label_size = label_size
    for img_path in self.img_files:
      sat_name = img_path.split("/")[-1]
      sat_name = sat_name.split("_")[0]
      self.mask_files.append(os.path.join(folder_path,'masks',sat_name+"_mask.png"))

  def __getitem__(self, index):
    img_path = self.img_files[index]
    mask_path = self.mask_files[index]

    data = Image.open(img_path)
    label = Image.open(mask_path)

    if self.image_transforms and self.mask_transforms:
      data = self.image_transforms(data)
      label = self.mask_transforms(label)
      class_label = self.create_label_class(label)
      
      return data, label, class_label
    
    data = np.array(data)
    label = np.array(label)
    return torch.from_numpy(data), torch.from_numpy(label)

  def __len__(self):
    return len(self.img_files)

  def create_label_class(self, label):
    '''
    Takes in a single image and produces the class label
    as a single channel image
    '''
    
    label = torch.round(label)
    image = 4*label[0,:,:] + 2*label[1,:,:] + label[2,:,:]
    image = image.data.numpy()
    label_image = np.zeros((self.label_size,self.label_size), dtype=np.uint8)
    label_image[image==3] = 0 # (Cyan: 011) Urban land
    label_image[image==6] = 1 # (Yellow: 110) Agriculture land
    label_image[image==5] = 2 # (Purple: 101) Rangeland
    label_image[image==2] = 3 # (Green: 010) Forest land
    label_image[image==1] = 4 # (Blue: 001) Water
    label_image[image==7] = 5 # (White: 111) Barren land
    label_image[image==0] = 6 # (Black: 000) Unknown

    return torch.tensor(label_image)















