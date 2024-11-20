import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from utils.util import find_max_epoch, print_size, training_loss, calc_diffusion_hyperparams, sampling
from utils.util import get_mask_mnr, get_mask_bm, get_mask_rm

from models.SSSDS4Imputer import SSSDS4Imputer
from preprocess.data_preprocess import data_preprocess
from SSSDS4Dataset import SSSDS4Dataset
from sklearn.metrics import mean_squared_error
from statistics import mean


def train(output_directory,
          ckpt_iter,
          n_iters,
          iters_per_ckpt,
          iters_per_logging,
          learning_rate,
          use_model,
          only_generate_missing,
          masking,
          missing_k):
    
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    # map diffusion hyperparameters to gpu
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].cuda()

    # predefine model
    net = SSSDS4Imputer(**model_config).cuda()
    print_size(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # load checkpoint
    if ckpt_iter == 'max':
        ckpt_iter = find_max_epoch(output_directory)
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            print('Successfully loaded model at iteration {}'.format(ckpt_iter))
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization try.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

        
        
    
    ### Custom data loading and reshaping ###  
    max_seq_len = 100
    X, T, _ , max_seq_len, padding_value, labels = data_preprocess(trainset_config['train_data_path'], max_seq_len)
    
    training_data = torch.from_numpy(X).float().cuda()
    dataset = SSSDS4Dataset(X, T, labels)
    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=128,
            shuffle=False
    )
    print('Data loaded')
    
    n_iter = ckpt_iter + 1
    while n_iter < n_iters + 1:
        for current_data, prev_data, T_mb, Y_mb in dataloader:
            batch = current_data
            
            if masking == 'rm':
                transposed_mask = get_mask_rm(batch[0], missing_k)
            elif masking == 'mnr':
                transposed_mask = get_mask_mnr(batch[0], missing_k)
            elif masking == 'bm':
                transposed_mask = get_mask_bm(batch[0], missing_k)
                
            mask = transposed_mask.permute(1, 0)
            mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
            loss_mask = ~mask.bool()
            
            
            batch = batch.permute(0, 2, 1)

            assert batch.size() == mask.size() == loss_mask.size()

            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask, Y_mb, prev_data
            loss = training_loss(net, nn.MSELoss(), X, label=Y_mb, prev_step=prev_data, diffusion_hyperparams=diffusion_hyperparams,only_generate_missing=only_generate_missing)

            loss.backward()
            optimizer.step()

            if n_iter % iters_per_logging == 0:
                print("iteration: {} \tloss: {}".format(n_iter, loss.item()))

            # save checkpoint
            """checkpoint_name = '{}.pkl'.format(n_iter)
            torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
            print('model at iteration %s is saved' % n_iter)"""
                
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                torch.save({'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(output_directory, checkpoint_name))
                print('model at iteration %s is saved' % n_iter)

            n_iter += 1
            
    all_mse = []
    first = True

    for i, (current_data, prev_data, T_mb, Y_mb) in enumerate(dataloader):   
    	batch = current_data 
    	num_samples = batch.shape[0]
    	
    	if masking == 'mnr':
    		mask_T = get_mask_mnr(batch[0], missing_k)
    		mask = mask_T.permute(1, 0)
    		mask = mask.repeat(batch.size()[0], 1, 1)
    		mask = mask.type(torch.float).cuda()
    		
    	elif masking == 'bm':
    		mask_T = get_mask_bm(batch[0], missing_k)
    		mask = mask_T.permute(1, 0)
    		mask = mask.repeat(batch.size()[0], 1, 1)
    		mask = mask.type(torch.float).cuda()
    	elif masking == 'rm':
    		mask_T = get_mask_rm(batch[0], missing_k)
    		mask = mask_T.permute(1, 0)
    		mask = mask.repeat(batch.size()[0], 1, 1).float().cuda()
    	
    	batch = batch.permute(0,2,1)
    	start = torch.cuda.Event(enable_timing=True)
    	end = torch.cuda.Event(enable_timing=True)
    	start.record()
    	
    	sample_length = batch.size(2)
    	sample_channels = batch.size(1)
    	generated_audio = sampling(net, (num_samples, sample_channels, sample_length),diffusion_hyperparams,cond=batch,mask=mask,label=Y_mb,prev_step=prev_data,only_generate_missing=only_generate_missing)
    	end.record()
    	torch.cuda.synchronize()
    	
    	print('generated {} utterances of random_digit at iteration {} in {}  seconds'.format(num_samples, ckpt_iter, int(start.elapsed_time(end) / 1000)))
    	
    	generated_audio = generated_audio.detach().cpu().numpy()
    	batch = batch.detach().cpu().numpy()
    	mask = mask.detach().cpu().numpy()
    	
    	print(output_directory)
    	ckpt_path = output_directory
    	#ckpt_path = os.path.join(ckpt_path, local_path)
    	
    	outfile = f'imputation{i}.npy'
    	new_out = os.path.join(ckpt_path, outfile)
    	np.save(new_out, generated_audio)
    	
    	outfile = f'original{i}.npy'
    	new_out = os.path.join(ckpt_path, outfile)
    	np.save(new_out, batch)
    	
    	outfile = f'mask{i}.npy'
    	new_out = os.path.join(ckpt_path, outfile)
    	np.save(new_out, mask)
    	
    	print('saved generated samples at iteration %s' % ckpt_iter)
    	
    	mse = mean_squared_error(generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)])
    	
    	all_mse.append(mse)
    
    print('Total MSE:', mean(all_mse))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)
    

    train_config = config["train_config"]  # training parameters
    gen_config = config['gen_config']

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config)  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config['use_model'] == 0:
        model_config = config['wavenet_config']
    elif train_config['use_model'] == 1:
        model_config = config['sashimi_config']
    elif train_config['use_model'] == 2:
        model_config = config['wavenet_config']

    train(**train_config)
