# Generative Models for Histopathological Images    

Latent Diffusion Models for the generation of histopathological images in the field of prostate cancer.  

### Dataset     
The Prostate dataset is based on different publicly available datasets. It contains patches of histopathological images and it's corresponding label where: 
* 0: background/unknown  
* 1: stroma  
* 2: healthy epithelium  
* 3: Gleason 3+3  
* 4: Gleason 3+4  
* 5: Gleason 4+3  
* 6: Gleason 4+4  
* 7: Gleason 3+5  
* 8: Gleason 5+3  
* 9: Gleason 4+5  
* 10: Gleason 5+4   
* 11: Gleason 5+5    

The different public datasets used to build the final dataset are PANDA, Gleason Challenge 2019 and SICAPv2.   

### Experiments   
All the code used to perform the different experiments is presented in this repository. This list of experiments can be grouped in different branches of experimentation:   
* Training of unconditional latent diffusion models   
* Training of conditional latent diffusion models (with prompt or label embedding)  
* Generation of synthetic datasets   
* Training of diffusion transformer models   
* Fine-tune of state-of-the-art diffusion models using LoRA and other methods    
* Data augmentation using inpainting with the trained generative model   