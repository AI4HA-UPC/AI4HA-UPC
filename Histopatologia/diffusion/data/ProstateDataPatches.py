import os
import numpy as np
import cv2
import albumentations
from PIL import Image
from torch.utils.data import Dataset
import torch
from numpy.random import rand
from tqdm import tqdm 
from diffusion.normalization.reinhard import Normalizer


def calculate_gleason_score(mask):
    # Step 1: Define the Gleason patterns
    patterns = [2, 3, 4] 

    # Step 2: Compute the area occupied by each Gleason pattern
    area_percentage = {}
    total_area = mask.size
    for pattern in patterns:
        area_percentage[pattern] = (mask == pattern).sum() / total_area
    
    if sum(area_percentage.values()) == 0:
        return 0

    # Step 3: Determine the most frequent pattern
    most_frequent_pattern = max(area_percentage, key=area_percentage.get)
    
    # Step 4: Extract the highest Gleason value present in the mask 
    highest_pattern_present = []
    for pattern, percentage in area_percentage.items():
        if percentage > 0:
            highest_pattern_present.append(pattern)
    highest_pattern_present = max(highest_pattern_present)

    # Step 5: Determine the second most frequent pattern
    second_most_frequent_pattern = None
    second_most_frequent_percentage = 0
    for pattern in patterns:
        if pattern != most_frequent_pattern:
            if (area_percentage[pattern] >= 0.05 and area_percentage[pattern] > second_most_frequent_percentage) or pattern == highest_pattern_present:
                second_most_frequent_pattern = pattern
                second_most_frequent_percentage = area_percentage[pattern]

    # Step 6: Calculate the Gleason score
    if second_most_frequent_pattern is not None:
        most_frequent_pattern += 1
        second_most_frequent_pattern += 1
        return most_frequent_pattern + second_most_frequent_pattern
    else:
        most_frequent_pattern += 1
        return most_frequent_pattern * 2
    

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
                 data_csv,
                 data_root,
                 segmentation_root,
                 size=None,
                 random_crop=False,
                 interpolation="bicubic",
                 n_labels=1,
                 shift_segmentation=False,
                 augmentation=False,
                 normalization=False
                 ):
        self.n_labels = n_labels
        self.shift_segmentation = shift_segmentation
        self.data_csv = data_csv
        self.data_root = data_root
        self.segmentation_root = segmentation_root
        self.normalization = normalization
        
        self.image_ids = []
        self.gleason_list = []
        with open(os.path.join(self.data_root, self.data_csv), "r") as f:
            lines = f.read().splitlines()
            for line in lines[1:]:
                self.image_ids.append(line.split(',')[0])
                self.gleason_list.append(line.split(',')[1])
             
        self._length = len(self.image_ids)
        self.labels = {
            "image_name": self.image_ids,
            "relative_file_path_": [l+'.png' for l in self.image_ids],
            "file_path_": [os.path.join(self.data_root, l+'.png')
                           for l in self.image_ids],
            "segmentation_path_": [os.path.join(self.segmentation_root, l+'_mask.png')
                                   for l in self.image_ids],
            "gleason": self.gleason_list
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

        if self.normalization:
            self.normalizer = Normalizer()
            self.normalizer.fit(image_path='/gpfs/projects/bsc70/MN4/bsc70/bsc70174/PANDA_code/diffusion/normalization/reference_img.png')

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        segmentation = Image.open(example["segmentation_path_"])
        #if not segmentation.mode == "RGB":
        #    segmentation = segmentation.convert("RGB")
        segmentation = np.array(segmentation).astype(np.uint8)

        #example["gleason"] = calculate_gleason_score(segmentation)

        #segmentation = load_bbox(example["segmentation_path_"], image.shape)

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

        #example["image"] = (processed["image"]/127.5 - 1.0).astype(np.float32)
        if self.normalization:
            processed["image"] = self.normalizer.transform(processed["image"])
        example["image"] = (processed["image"]/255).astype(np.float32)

        example["mask"] = np.expand_dims(processed["mask"], -1)

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
                 data_csv="/media/aromero/Axel/Treball/bsc70174/Data/PANDA/patches_256/train.csv",
                 data_root="/media/aromero/Axel/Treball/bsc70174/Data/PANDA/patches_256/patches",
                 segmentation_root="/media/aromero/Axel/Treball/bsc70174/Data/PANDA/patches_256/masks",
                 random_crop=False, 
                 interpolation="bicubic", 
                 n_labels=2,
                augmentation=False,
                normalization=False):
        super().__init__(data_csv=data_csv,
                         data_root=data_root,
                         segmentation_root=segmentation_root,
                         n_labels=n_labels,
                         size=size, random_crop=random_crop, interpolation=interpolation,
                         augmentation=augmentation,
                         normalization=normalization)

class ExamplesTest(SegmentationBase):
    def __init__(self, 
                 size=None, 
                 data_csv="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/train.csv",
                 data_root="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/patches",
                 segmentation_root="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/masks",
                 random_crop=False, 
                 interpolation="bicubic", 
                 n_labels=2,
                 augmentation=False,
                 normalization=False):
        super().__init__(size=size, 
                         data_csv=data_csv,
                         data_root=data_root,
                         segmentation_root=segmentation_root,
                         n_labels=n_labels,
                         random_crop=random_crop, 
                         interpolation=interpolation,
                         augmentation=augmentation,
                         normalization=normalization)



class ExamplesDebug(SegmentationBase):
    def __init__(self, size=None, random_crop=False, 
                 interpolation="bicubic", n_labels=2):
        super().__init__(data_csv="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/train.csv",
                         data_root="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/patches",
                         segmentation_root="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/masks",
                         n_labels=n_labels,
                         size=size, random_crop=random_crop, interpolation=interpolation)
"""
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
    def __init__(self, model='sdxl-vae', size=256, 
                 root='/gpfs/projects/bsc70/bsc70174/Data/PANDAGen', 
                 normalization=1., 
                 segmentation=False, 
                 augmentation=False):
        base_path = f'{root}/{model}/{size}/'
        if augmentation:
            base_path += 'aug/'
            self.images = [os.path.join(os.path.join(base_path, 'images'),  d)
                       for d in sorted(os.listdir(os.path.join(base_path, 'images')))
                        if 'orig' in d]     
            if segmentation:
                self.segmentations = [os.path.join(os.path.join(base_path, 'masks'),  d)
                            for d in sorted(os.listdir(os.path.join(base_path, 'masks')))
                            if 'orig' in d]     
                self.segmentations_reduced = [os.path.join(os.path.join(base_path, 'masks_reduced'),  d)
                            for d in sorted(os.listdir(os.path.join(base_path, 'masks_reduced')))
                            if 'orig' in d]  
        else:
            self.images = [os.path.join(os.path.join(base_path, 'images'),  d)
                       for d in sorted(os.listdir(os.path.join(base_path, 'images')))] 
                    
            if segmentation:
                self.segmentations = [os.path.join(os.path.join(base_path, 'masks'),  d)
                            for d in sorted(os.listdir(os.path.join(base_path, 'masks')))]
                self.segmentations_reduced = [os.path.join(os.path.join(base_path, 'masks_reduced'),  d)
                            for d in sorted(os.listdir(os.path.join(base_path, 'masks_reduced')))]
                
        self.normalization = normalization
        self.seg = segmentation
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        r = rand(1)[0]
        if r <0.5:
            rep = 'orig'
        elif r<0.6:
            rep = 'vflip'
        elif r<0.7:
            rep = 'hflip'
        elif r<0.8:
            rep='rot90'
        elif r<0.9:
            rep = 'rot180'
        else:
            rep = 'rot270'

        img_path = self.images[idx].replace('orig', rep)
        if self.seg:
            image = torch.tensor(np.load(img_path)/(1.0 * self.normalization))
            mask_path = self.segmentations[idx].replace('orig', rep)
            mask_reduced_path = self.segmentations_reduced[idx].replace('orig', rep)
            #onehot = torch.tensor(np.load(mask_path))
            #s_onehot = ((onehot/onehot.max())*2)-1
            mask = torch.tensor(np.load(mask_path))
            mask_reduced = torch.tensor(np.load(mask_reduced_path))

            example = {"image": image,
                    "segmentation": mask,
                    "segmentation_reduced": mask_reduced
                    #"imageseg": np.concatenate((image, onehot), axis=0)
                    }
        else:
            example = {"image": torch.tensor(np.load(img_path)/(1.0 * self.normalization))
                    }   
        return example
""" 

class ExamplesVAE(Dataset):
    def __init__(self, model='sdxl-vae', size=256, 
                 root='/gpfs/projects/bsc70/bsc70174/Data/PANDAGen', 
                 data_csv="/gpfs/projects/bsc70/bsc70174/Data/PANDA/patches_256/train.csv",
                 normalization=1., 
                 segmentation=False, 
                 augmentation=False):
        
        base_path = f'{root}/{model}/{size}/'
        self.data_csv = data_csv

        self.image_ids = []
        self.gleason_list = []
        with open(self.data_csv, "r") as f:
            lines = f.read().splitlines()
            for line in lines[1:]:
                self.image_ids.append(line.split(',')[0])
                self.gleason_list.append(line.split(',')[1])

        self.image_ids_set = set(self.image_ids) 

        if augmentation:
            base_path += 'aug/'
            image_dir = os.path.join(base_path, 'images')

            #self.images = [
            #    os.path.join(image_dir, d)
            #    for d in sorted(os.listdir(image_dir))
            #    if 'orig' in d and '_'.join(os.path.basename(d).split('_')[:3]) in self.image_ids_set]
            self.images = [
                os.path.join(image_dir, d+'-orig.npy')
                for d in self.image_ids]
            
            if segmentation:
                segmentation_dir = os.path.join(base_path, 'masks')
                segmentation_reduced_dir = os.path.join(base_path, 'masks_reduced')

                #self.segmentations = [
                #        os.path.join(segmentation_dir, d+'_mask.png')
                #        for d in sorted(os.listdir(segmentation_dir))
                #        if 'orig' in d and '_'.join(os.path.basename(d).split('_')[:3]) in self.image_ids_set]
                self.segmentations = [
                        os.path.join(segmentation_dir, d+'-orig_mask.npy')
                        for d in self.image_ids]
                
                #self.segmentations_reduced = [
                #        os.path.join(segmentation_reduced_dir, d)
                #        for d in sorted(os.listdir(segmentation_reduced_dir))
                #        if 'orig' in d and '_'.join(os.path.basename(d).split('_')[:3]) in self.image_ids_set]
                self.segmentations_reduced = [
                        os.path.join(segmentation_reduced_dir, d+'-orig_mask.npy')
                        for d in self.image_ids]
        else:
            image_dir = os.path.join(base_path, 'images')

            self.images = [
                os.path.join(image_dir, d)
                for d in sorted(os.listdir(image_dir))
                if 'orig' in d and '_'.join(os.path.basename(d).split('_')[:3]) in self.image_ids_set]
                    
            if segmentation:
                segmentation_dir = os.path.join(base_path, 'masks')
                segmentation_reduced_dir = os.path.join(base_path, 'masks_reduced')

                self.segmentations = [
                        os.path.join(segmentation_dir, d)
                        for d in sorted(os.listdir(segmentation_dir))
                        if 'orig' in d and '_'.join(os.path.basename(d).split('_')[:3]) in self.image_ids_set]
                
                self.segmentations_reduced = [
                        os.path.join(segmentation_reduced_dir, d)
                        for d in sorted(os.listdir(segmentation_reduced_dir))
                        if 'orig' in d and '_'.join(os.path.basename(d).split('_')[:3]) in self.image_ids_set]
                
        self.normalization = normalization
        self.seg = segmentation

        image_ids_dict = {f"{image_id}-orig.npy": index for index, image_id in enumerate(self.image_ids)}
        #split_values = ['_'.join(im.split('/')[-1].split('_')[:3]) for im in self.images]
        self.final_gleason = [int(self.gleason_list[image_ids_dict[image_name.split('/')[-1]]]) for image_name in tqdm(self.images)]

        #self.final_gleason = [int(self.gleason_list[self.image_ids.index('_'.join(im.split('/')[-1].split('_')[:3]))]) for im in self.images]
        
    def __len__(self):
        return len(self.segmentations_reduced)

    def __getitem__(self, idx):
        r = rand(1)[0]
        if r <0.5:
            rep = 'orig'
        elif r<0.6:
            rep = 'vflip'
        elif r<0.7:
            rep = 'hflip'
        elif r<0.8:
            rep='rot90'
        elif r<0.9:
            rep = 'rot180'
        else:
            rep = 'rot270'

        img_path = self.images[idx].replace('orig', rep)
        if self.seg:
            image = torch.tensor(np.load(img_path)/(1.0 * self.normalization))
            mask_path = self.segmentations[idx].replace('orig', rep)
            mask_reduced_path = self.segmentations_reduced[idx].replace('orig', rep)
            #onehot = torch.tensor(np.load(mask_path))
            #s_onehot = ((onehot/onehot.max())*2)-1
            mask = torch.tensor(np.load(mask_path))
            mask_reduced = torch.tensor(np.load(mask_reduced_path))

            example = {"image": image,
                    "gleason": self.final_gleason[idx],
                    "segmentation": mask,
                    "segmentation_reduced": mask_reduced
                    #"imageseg": torch.tensor(np.concatenate((image, s_onehot), axis=0))
                    }
        else:
            example = {"image": torch.tensor(np.load(img_path)/(1.0 * self.normalization)),
                       "gleason": self.final_gleason[idx]} 
        return example

"""
if __name__ == '__main__':

    ex = ExamplesDebug(size=512, n_labels=5, interpolation='area')

    mask_alpha = 0.5  # Adjust the alpha (transparency) level as needed 
    for i in np.random.randint(0, len(ex), 50):
        example = ex[i]['imageseg']
        img = example[..., :3]
        mask = example[..., 3:]

        fig, axs = plt.subplots(2, 6)
        for j in range(6):
            if j == 0:
                axs[0, j].imshow(img)
                axs[0, j].set_title('Original Image', fontsize=6)
                axs[1, j].imshow(mask[..., 2:]/5, cmap='gray')
            else:
                mask_temp = mask[..., j-1]
                axs[0, j].imshow(mask_temp, cmap='gray')
                axs[0, j].set_title(f'Mask of the label {j-1}', fontsize=6)
                masked_image = np.copy(img)
                masked_image[mask_temp == 1] = masked_image[mask_temp == 1] * (1 - mask_alpha) + np.array([1, 0, 0]) * mask_alpha
                axs[1, j].imshow(masked_image, cmap='gray')
            axs[0, j].axis('off')
            axs[1, j].axis('off')
        fig.tight_layout()
        fig.savefig(f'/media/aromero/Axel/Treball/bsc70174_Code/diffusion/data/analysis_patches/patch_{i}.png', dpi=800)
        #plt.show()


if __name__ == '__main__':

    ex = ExamplesVAE(augmentation=True, segmentation=True) 

    for i in range(6):
        img = ex[i]['image']
        mask = ex[i]['segmentation'] 

        print(img.shape)
        print(mask.shape)
"""