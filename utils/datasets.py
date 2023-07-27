import os
import torch
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

data_transforms = transforms.Compose([
                transforms.Resize((256, 256), Image.ANTIALIAS),
                transforms.ToTensor(),
                transforms.CenterCrop(224),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])]) # from imagenet
gt_transforms = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.CenterCrop(224)])

class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase, half=False):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1
        self.half = half
        
    def load_dataset(self):

        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        
        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0]*len(img_paths))
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"
        
        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        if self.half:
            img = img.half()
        if gt == 0:
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)
        
        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"

        return img, gt, label, os.path.basename(img_path[:-4]), img_type

# for calibration purposes
class RandomImageDataset(Dataset):
    def __init__(self, num_images, transform=None):
        self.num_images = num_images
        self.transform = transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Generate a random image
        image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)

        # Convert numpy array to PIL image
        image = transforms.ToPILImage()(image)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, 0, 0, 0, 0
    
    
class Own_Imagenet(Dataset):
    def __init__(self, transform, phase = 'val', root = r'/mnt/crucial/UNI/IIIT_Muen/MA/ImageNet/ILSVRC/Data/CLS-LOC'):
        img_paths_full = glob.glob(os.path.join(root, phase) + "/*.JPEG")
        self.img_paths = np.random.choice(img_paths_full, 5000, replace=False)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        return img, 0, 0, 0, 0
