import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import torch
from numpy.random import rand


def load_bbox(fn, shape):
    bbfile = open(fn)
    nbbs = int(bbfile.readline())
    mask = np.zeros((shape[0],shape[1]),dtype=np.uint8) # initialize mask
    for bb in range(nbbs):
        x1, y1,  x2,  y2 = bbfile.readline().split(' ')[:4]
        x1,  y1, x2, y2 = int(x1),  int(y1), int(x2), int(y2) 
        if x1 > x2:
            tmp = x2
            x2 = x1
            x1 = tmp
        if y1 > y2:
            tmp = y2
            y2 = y1
            y1 = tmp
        mask[y1:y2,x1:x2] = 1 # fill with white pixels 
    bbfile.close()
    return mask


class SegmentationBase(Dataset):
    def __init__(self,
                 data_csv, data_root, segmentation_root,
                 size=None, random_crop=False, interpolation="bicubic",
                 n_labels=1, shift_segmentation=False,
                 augmentation=False,
                 ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        self.data_csv = data_csv
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        with open(self.data_csv, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
            "segmentation_path_": [os.path.join(self.segmentation_root, l.replace(".jpg", ".txt"))
                                   for l in self.image_paths]
        }

        size = None if size is not None and size<=0 else size
        self.size = size

        if augmentation:
            self.augmentation = albumentations.OneOf(
                    [albumentations.VerticalFlip(p=0.25),
                    albumentations.HorizontalFlip(p=0.25),
                     albumentations.RandomRotate90(p=0.25)
                    ])
        else:
            self.augmentation = None

        if self.size is not None:
            self.interpolation = interpolation
            self.interpolation = {
                "nearest": cv2.INTER_NEAREST,
                "bilinear": cv2.INTER_LINEAR,
                "bicubic": cv2.INTER_CUBIC,
                "area": cv2.INTER_AREA,
                "lanczos": cv2.INTER_LANCZOS4}[self.interpolation]
            self.image_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                 interpolation=self.interpolation)
            self.segmentation_rescaler = albumentations.SmallestMaxSize(max_size=self.size,
                                                                        interpolation=cv2.INTER_NEAREST)
            self.center_crop = not random_crop
            if self.center_crop:
                self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size, width=self.size)
            self.preprocessor = self.cropper

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        segmentation = load_bbox(example["segmentation_path_"], image.shape)
        if self.size is not None:
            image = self.image_rescaler(image=image)["image"]


        if self.shift_segmentation:
            # used to support segmentations containing unlabeled==255 label
            segmentation = segmentation+1
        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]
        if self.size is not None:
            processed = self.preprocessor(image=image,
                                          mask=segmentation
                                          )
        else:
            processed = {"image": image,
                         "mask": segmentation
                         }

        if self.augmentation is not None:
            processed = self.augmentation(image=processed['image'], mask=processed['mask'])       

        example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        segmentation = processed["mask"]

        onehot = np.eye(self.n_labels)[segmentation].astype(np.float32)
        example["segmentation"] = onehot
        s_onehot = ((onehot/np.max(onehot))*2)-1
        example["imageseg"] = np.concatenate((example["image"], s_onehot), axis=2)
        example['aesegmentation'] = s_onehot


        return example


class ExamplesTrain(SegmentationBase):
    def __init__(self, 
                 size=None, 
                 data_csv="/data/LDPolyp/LDPolip_train.txt",
                 data_root="/data/LDPolyp/train/images",
                 segmentation_root="/data/LDPolyp/train/bbox",
                 random_crop=False, 
                 interpolation="bicubic", 
                 n_labels=2,
                augmentation=False):
        super().__init__(data_csv=data_csv,
                         data_root=data_root,
                         segmentation_root=segmentation_root,
                         n_labels=n_labels,
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         augmentation=augmentation)

class ExamplesTest(SegmentationBase):
    def __init__(self, 
                 size=None, 
                 data_csv="/data/LDPolyp/LDPolip_test.txt",
                 data_root="/data/LDPolyp/test/images",
                 segmentation_root="/data/LDPolyp/test/bbox",
                 random_crop=False, 
                 interpolation="bicubic", 
                 n_labels=2,
                 augmentation=False):
        super().__init__(size=size, 
                         data_csv=data_csv,
                         data_root=data_root,
                         segmentation_root=segmentation_root,
                         n_labels=n_labels,
                         random_crop=random_crop, 
                         interpolation=interpolation,
                         augmentation=augmentation)



class ExamplesDebug(SegmentationBase):
    def __init__(self, size=None, random_crop=False, 
                 interpolation="bicubic", n_labels=2):
        super().__init__(data_csv="/home/bejar/ssdstorage/Colon/Dataset/train.txt",
                         data_root="/home/bejar/ssdstorage/Colon/Dataset/train/images",
                         segmentation_root="/home/bejar/ssdstorage/Colon/Dataset/train/bbox",
                         n_labels=n_labels,
                         size=size, random_crop=random_crop, interpolation=interpolation)

class ExamplesVAE(Dataset):
    def __init__(self, model='AEKL', size=256, encoder='AllAEKL64', 
                 root='/gpfs/projects/bsc70/bsc70642/Data/LDPolyp', 
                 normalization=1.):
        base_path = f'{root}/{model}/{size}/{encoder}/'
        print(base_path)
        self.images = [os.path.join(base_path,  d)
                       for d in sorted(os.listdir(base_path))]
        self.normalization = normalization
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        example = {"image": torch.tensor(np.load(img_path)/(1.0 * self.normalization))}
        return example
    
class ExamplesVAE(Dataset):
    def __init__(self, model='AEKL', size=256, encoder='AllAEKL64', 
                 root='/gpfs/projects/bsc70/bsc70642/Data/LDPolyp', 
                 normalization=1., 
                 segmentation=False, 
                 augmentation=False,
                 augprob=0.5):
        base_path = f'{root}/{model}/{size}/{encoder}/'
        if augmentation:
            base_path += 'aug/'
            self.images = [os.path.join(os.path.join(base_path, 'images'),  d)
                       for d in sorted(os.listdir(os.path.join(base_path, 'images')))
                        if 'orig' in d]     
            if segmentation:
                self.segmentations = [os.path.join(os.path.join(base_path, 'masks'),  d)
                            for d in sorted(os.listdir(os.path.join(base_path, 'masks')))
                            if 'orig' in d]       
        else:
            self.images = [os.path.join(os.path.join(base_path, 'images'),  d)
                       for d in sorted(os.listdir(os.path.join(base_path, 'images')))] 
                    
            if segmentation:
                self.segmentations = [os.path.join(os.path.join(base_path, 'masks'),  d)
                            for d in sorted(os.listdir(os.path.join(base_path, 'masks')))]
                
        self.normalization = normalization
        self.seg = segmentation
        self.augprob = augprob
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        ntrans = 5
        augdiv = self.augprob/ntrans
        ltrans = ['vflip', 'hflip', 'rot90', 'rot180', 'rot270','orig']
        vtrans = []
        for i in range(ntrans):
            vtrans.append([augdiv *(i+1), ltrans[i]])
        vtrans.append([1, 'orig'])

        rep = 'orig'
        for i in range(ntrans+1):
            if rand(1)[0] < vtrans[i][0]:
                rep = vtrans[i][1]
                break

        img_path = self.images[idx].replace('orig', rep)
        # print(img_path)
        if self.seg:
            image = torch.tensor(np.load(img_path)/(1.0 * self.normalization))
            mask_path = self.segmentations[idx].replace('orig', rep)
            onehot = torch.tensor(np.load(mask_path))
            # s_onehot = ((onehot/onehot.max())*2)-1
            example = {"image": image ,
                    "segmentation": onehot,
                    # "imageseg": np.concatenate((image, s_onehot), axis=0)
                    }
        else:
            example = {"image": torch.tensor(np.load(img_path)/(1.0 * self.normalization))
                    }   
        return example



if __name__ == '__main__':
    from skimage.measure import block_reduce

    ex = ExamplesDebug(size=256, n_labels=2, interpolation='area')
    print(ex[1]['segmentation'].shape)
    # print(ex[0]['segmentation'].shape)  
    for i in range(10):
        print(i, ex[i]['segmentation'].shape)
        print(i, ex[i]['imageseg'].shape)
        print(ex[i]['segmentation'])
