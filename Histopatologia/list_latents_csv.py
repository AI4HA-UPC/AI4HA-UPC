import os 
import csv
import numpy as np 
import pandas as pd
from tqdm import tqdm


PATH = '/gpfs/projects/bsc70/bsc70174/Data/PANDAGen/sdxl-vae/512-256/aug'
IMAGES_PATH = os.path.join(PATH, 'images')

list_images = os.listdir(IMAGES_PATH) 

df_dict = {'Images_id': list_images}

df = pd.DataFrame.from_dict(df_dict)
df.to_csv(os.path.join(PATH, 'list_latents.csv'), index=False)
