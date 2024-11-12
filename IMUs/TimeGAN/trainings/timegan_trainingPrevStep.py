import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.timegan_utils import  *
from utils.utils import *
import torch.optim as optim
from models.timegan_v4_PrevStep import TimeGAN
import argparse
import glob
import pandas as pd
import random
from data_preprocess import data_preprocess
from sklearn.model_selection import train_test_split
import time
from typing import Dict, Union
from tqdm import tqdm, trange
from TimeGANDataset import TimeGANDataset
from torch.utils.tensorboard import SummaryWriter
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class TimeGanTrainer:
    def __init__(self, device, model_cfg, learning_rate=0.001, n_epochs=1000, iters_per_ckpt=100):
        # self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.iters_per_ckpt = iters_per_ckpt

        # Inputs for the main function
        self.parser = argparse.ArgumentParser()

        # Experiment Arguments
        self.parser.add_argument(
            '--device',
            choices=['cuda', 'cpu'],
            default='cuda',
            type=str)
        self.parser.add_argument(
            '--exp',
            default='final',
            type=str)
        self.parser.add_argument(
            "--is_train",
            type=str2bool,
            default=True)
        self.parser.add_argument(
            '--seed',
            default=0,
            type=int)
        self.parser.add_argument(
            '--feat_pred_no',
            default=2,
            type=int)
        # Data Arguments
        self.parser.add_argument(
            '--train_rate',
            default=0.1,
            type=float)
        # Model Arguments   
        self.parser.add_argument(
            '--dis_thresh',
            default=0.15,
            type=float)
        self.parser.add_argument(
            '--learning_rate',
            default=1e-3,
            type=float)
        self.parser.add_argument(
            '--padding_value',
            default=-1,
            type=int)

        self.args = self.parser.parse_args()

        self.args.max_seq_len = model_cfg['max_seq_len']
        self.args.emb_epochs = model_cfg['emb_epochs']
        self.args.sup_epochs = model_cfg['sup_epochs']
        self.args.gan_epochs = model_cfg['gan_epochs']
        self.args.batch_size = model_cfg['batch_size']
        self.args.hidden_dim = model_cfg['hidden_dim']
        self.args.num_layers = model_cfg['num_layers']
        self.args.optimizer = model_cfg['optimizer']
        #self.args.learning_rate = model_cfg['learning_rate']


    def extract_numbers(filename):
        # Extract subject, condition, and run numbers from the filename
        parts = filename.split('_')
        subject_num = int(parts[1])
        condition_num = int(parts[3])
        run_num = int(parts[5])
        return (subject_num, condition_num, run_num)

    def embedding_trainer(
            self, model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            e_opt: torch.optim.Optimizer,
            r_opt: torch.optim.Optimizer,
            args: Dict,
            writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)] = None,
            gen_cfg: str = None
            
    ) -> None:
        """The training loop for the embedding and recovery functions
        """
        logger = trange(args.emb_epochs, desc=f"Epoch: 0, Loss: 0")
        emb_epoch_list = []
        emb_loss_list = []
        
        for epoch in logger:
            for current_data, prev_data, T_mb, Y_mb in dataloader:

                # Reset gradients
                model.zero_grad()

                # Forward Pass
                _, E_loss0, E_loss_T0 = model(X=current_data, prev_step=prev_data, T=T_mb, Z=None, obj="autoencoder", labels=Y_mb)
                loss = np.sqrt(E_loss_T0.item())

                # Backward Pass
                E_loss0.backward()

                # Update model parameters
                e_opt.step()
                r_opt.step()

            # Log loss for final batch of each epoch (29 iters)
            emb_epoch_list.append(epoch)
            emb_loss_list.append(loss)
            
            logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
            if writer:
                writer.add_scalar(
                    "Embedding/Loss:",
                    loss,
                    epoch
                )
                writer.flush()
        
        with open(f"{self.args.model_path}/emb_loss_{gen_cfg}.pickle", "wb") as fb:
            pickle.dump(emb_loss_list, fb)
            
        with open(f"{self.args.model_path}/emb_epochs_{gen_cfg}.pickle", "wb") as fb:
            pickle.dump(emb_epoch_list, fb)

    def supervisor_trainer(
            self, model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            s_opt: torch.optim.Optimizer,
            g_opt: torch.optim.Optimizer,
            args: Dict,
            writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)] = None,
            gen_cfg: str = None
    ) -> None:
        """The training loop for the supervisor function
        """
        logger = trange(args.sup_epochs, desc=f"Epoch: 0, Loss: 0")
        sup_loss_list = []
        sup_epoch_list = []
        
        for epoch in logger:
            for current_data, prev_data, T_mb, Y_mb in dataloader:
                # Reset gradients
                model.zero_grad()

                # Forward Pass
                S_loss = model(X=current_data, prev_step=prev_data, T=T_mb, Z=None, obj="supervisor", labels=Y_mb)

                # Backward Pass
                S_loss.backward()
                loss = np.sqrt(S_loss.item())

                # Update model parameters
                s_opt.step()

            # Log loss for final batch of each epoch (29 iters)
            sup_loss_list.append(loss)
            sup_epoch_list.append(epoch)
            
            logger.set_description(f"Epoch: {epoch}, Loss: {loss:.4f}")
            if writer:
                writer.add_scalar(
                    "Supervisor/Loss:",
                    loss,
                    epoch
                )
                writer.flush()
                
        with open(f"{self.args.model_path}/sup_loss_{gen_cfg}.pickle", "wb") as fb:
            pickle.dump(sup_loss_list, fb)
            
        with open(f"{self.args.model_path}/sup_epochs_{gen_cfg}.pickle", "wb") as fb:
            pickle.dump(sup_epoch_list, fb)
            
    def joint_trainer(
            self, model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            e_opt: torch.optim.Optimizer,
            r_opt: torch.optim.Optimizer,
            s_opt: torch.optim.Optimizer,
            g_opt: torch.optim.Optimizer,
            d_opt: torch.optim.Optimizer,
            args: Dict,
            writer: Union[torch.utils.tensorboard.SummaryWriter, type(None)] = None,
            gen_cfg: str = None
    ) -> None:
        """The training loop for training the model altogether
        """
        logger = trange(
            args.sup_epochs,
            desc=f"Epoch: 0, E_loss: 0, G_loss: 0, D_loss: 0"
        )
        G_loss_list = []
        E_loss_list = []
        D_loss_list = []
        epoch_list = []
            
        counter = 0
        for epoch in logger:
            for current_data, prev_data, T_mb, Y_mb in dataloader:
                ## Generator Training
                for _ in range(2):
                    # Random Generator
                    Z_mb = torch.rand((current_data.shape[0], args.max_seq_len, args.Z_dim))
    
                    # Forward Pass (Generator)
                    model.zero_grad()
                    G_loss = model(X=current_data, prev_step=prev_data, T=T_mb, Z=Z_mb, obj="generator", labels=Y_mb)
                    G_loss.backward()
                    G_loss = np.sqrt(G_loss.item())
    
                    # Update model parameters
                    g_opt.step()
                    s_opt.step()
    
                    # Forward Pass (Embedding)
                    model.zero_grad()
                    E_loss, _, E_loss_T0 = model(X=current_data, prev_step=prev_data, T=T_mb, Z=Z_mb, obj="autoencoder", labels=Y_mb)
                    E_loss.backward()
                    E_loss = np.sqrt(E_loss.item())
    
                    # Update model parameters
                    e_opt.step()
                    r_opt.step()

                # Random Generator
                Z_mb = torch.rand((current_data.shape[0], args.max_seq_len, args.Z_dim))

                ## Discriminator Training
                model.zero_grad()
                # Forward Pass
                D_loss = model(X=current_data, prev_step=prev_data, T=T_mb, Z=Z_mb, obj="discriminator", labels=Y_mb)

                # Check Discriminator loss
                if D_loss > args.dis_thresh:
                    # Backward Pass
                    D_loss.backward()

                    # Update model parameters
                    d_opt.step()
                D_loss = D_loss.item()

                counter += 1
                
            G_loss_list.append(G_loss)
            E_loss_list.append(E_loss)
            D_loss_list.append(D_loss)
            epoch_list.append(epoch)

            logger.set_description(
                f"Epoch: {epoch}, E: {E_loss:.4f}, G: {G_loss:.4f}, D: {D_loss:.4f}"
            )
            if writer:
                writer.add_scalar(
                    'Joint/Embedding_Loss:',
                    E_loss,
                    epoch
                )
                writer.add_scalar(
                    'Joint/Generator_Loss:',
                    G_loss,
                    epoch
                )
                writer.add_scalar(
                    'Joint/Discriminator_Loss:',
                    D_loss,
                    epoch
                )
                writer.flush() 
                
        with open(f"{self.args.model_path}/joint_G_loss_{gen_cfg}.pickle", "wb") as fb:
            pickle.dump(G_loss_list, fb)
            
        with open(f"{self.args.model_path}/joint_D_loss_{gen_cfg}.pickle", "wb") as fb:
            pickle.dump(D_loss_list, fb)
            
        with open(f"{self.args.model_path}/joint_E_loss_{gen_cfg}.pickle", "wb") as fb:
            pickle.dump(E_loss_list, fb)
            
        with open(f"{self.args.model_path}/joint_epochs_{gen_cfg}.pickle", "wb") as fb:
            pickle.dump(epoch_list, fb)
            
    def timegan_trainer(self, model, data, time, labels, args, generate_cfg): 
        """The training procedure for TimeGAN
        Args:
            - model (torch.nn.module): The model model that generates synthetic data
            - data (numpy.ndarray): The data for training the model
            - time (numpy.ndarray): The time for the model to be conditioned on
            - args (dict): The model/training configurations
        Returns:
            - generated_data (np.ndarray): The synthetic data generated by the model
        """

        # Initialize TimeGAN dataset and dataloader
        dataset = TimeGANDataset(data, time, labels)
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        model.to(args.device)

        # Initialize Optimizers
        e_opt = torch.optim.Adam(model.embedder.parameters(), lr=args.learning_rate)
        r_opt = torch.optim.Adam(model.recovery.parameters(), lr=args.learning_rate)
        s_opt = torch.optim.Adam(model.supervisor.parameters(), lr=args.learning_rate)
        g_opt = torch.optim.Adam(model.generator.parameters(), lr=args.learning_rate)
        d_opt = torch.optim.Adam(model.discriminator.parameters(), lr=args.learning_rate)

        crit_cls = nn.CrossEntropyLoss()

        # TensorBoard writer
        writer = SummaryWriter(os.path.join(f"tensorboard/{args.exp}"))

        print("\nStart Embedding Network Training")
        self.embedding_trainer(
            model=model,
            dataloader=dataloader,
            e_opt=e_opt,
            r_opt=r_opt,
            args=args,
            writer=writer,
            gen_cfg=generate_cfg['name']
        )

        print("\nStart Training with Supervised Loss Only")
        self.supervisor_trainer(
            model=model,
            dataloader=dataloader,
            s_opt=s_opt,
            g_opt=g_opt,
            args=args,
            writer=writer,
            gen_cfg=generate_cfg['name']
        )

        print("\nStart Joint Training")
        self.joint_trainer(
            model=model,
            dataloader=dataloader,
            e_opt=e_opt,
            r_opt=r_opt,
            s_opt=s_opt,
            g_opt=g_opt,
            d_opt=d_opt,
            args=args,
            writer=writer,
            gen_cfg=generate_cfg['name']
        )

        # Save model, args, and hyperparameters
        torch.save(args, f"{args.model_path}/args_{generate_cfg['name']}.pickle")
        torch.save(model.state_dict(), f"{args.model_path}/model_{generate_cfg['name']}.pt")
        print(f"\nSaved at path: {args.model_path}")

    def timegan_generator(self, model, T, labels, args, generate_cfg):
        """The inference procedure for TimeGAN
        Args:
            - model (torch.nn.module): The model model that generates synthetic data
            - T (List[int]): The time to be generated on
            - args (dict): The model/training configurations
        Returns:
            - generated_data (np.ndarray): The synthetic data generated by the model
        """
        # Load model for inference
        if not os.path.exists(args.model_path):
            raise ValueError(f"Model directory not found...")

        # Load arguments and model
        with open(f"{args.model_path}/args_{generate_cfg['name']}.pickle", "rb") as fb: 
            args = torch.load(fb)

        model.load_state_dict(torch.load(f"{args.model_path}/model_{generate_cfg['name']}.pt"))

        print("\nGenerating Data...")
        # Initialize model to evaluation mode and run without gradients
        model.to(args.device)
        model.eval()
        first = True
        generated_data_list = []
        generated_labels_list = []
        
        for i in range(2):
        	with torch.no_grad():
        		for step in range(len(T)):
        			# Generate fake data
        			Z = torch.rand((1, args.max_seq_len, args.Z_dim))
        			
        			labels_aux = torch.tensor(labels[step:step+1], dtype=torch.int64)
        			
        			if first or labels[step] != current_label:
        				prev_data = torch.rand((1, args.max_seq_len, args.Z_dim))
        				current_label = labels[step]
        				first = False
        			else:
        				prev_data = generated_data
        				
        			generated_data, generated_labels = model(X=None, T=T[step:step+1], Z=Z, obj="inference", labels=labels_aux, prev_step=prev_data)
        			
        			generated_data_list.append(generated_data)
        			generated_labels_list.append(labels_aux)
                
        generated_data_stacked = torch.stack(generated_data_list)
        generated_labels_stacked = torch.stack(generated_labels_list)
        
        return generated_data_stacked.numpy(), generated_labels_stacked.cpu().numpy()

    def train_model(self, rank, num_gpus, model_cfg, diffusion_cfg, dataset_cfg, train_cfg, opt, generate_cfg):        
        print("Version with outliers") 
        ## Runtime directory
        code_dir = os.path.abspath(".")
        if not os.path.exists(code_dir):
            raise ValueError(f"Code directory not found at {code_dir}.")

        ## Data directory
        data_path = os.path.abspath("./")
        if not os.path.exists(data_path):
            raise ValueError(f"Data file not found at {data_path}.")
        data_dir = os.path.dirname(data_path)
        data_file_name = os.path.basename(data_path)

        ## Output directories
        self.args.model_path = os.path.abspath(f"./output/{self.args.exp}/")
        out_dir = os.path.abspath(self.args.model_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # TensorBoard directory
        tensorboard_path = os.path.abspath("./tensorboard")
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path, exist_ok=True)

        print(f"\nCode directory:\t\t\t{code_dir}")
        print(f"Data directory:\t\t\t{data_path}")
        print(f"Output directory:\t\t{out_dir}")
        print(f"TensorBoard directory:\t\t{tensorboard_path}\n")

        ##############################################
        # Initialize random seed and CUDA
        ##############################################

        os.environ['PYTHONHASHSEED'] = str(self.args.seed)
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)

        if self.args.device == "cuda" and torch.cuda.is_available():
            print("Using CUDA\n")
            self.args.device = torch.device("cuda:0")
            # torch.cuda.manual_seed_all(args.seed)
            torch.cuda.manual_seed(self.args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            print("Using CPU\n")
            self.args.device = torch.device("cpu")

        #########################
        # Load and preprocess data for model
        #########################

        print(os.getcwd())
        
        real_data = f'{dataset_cfg["train_file"]}/train_data.pickle'
        with open(real_data, "rb") as real_file:
        	trainX = np.squeeze(pickle.load(real_file))
        	
        real_labels = f'{dataset_cfg["train_file"]}/train_labels.pickle'
        with open(real_labels, "rb") as real_file:
        	trainY = np.squeeze(pickle.load(real_file))
        	
        real_time = f'{dataset_cfg["train_file"]}/train_time.pickle'
        with open(real_time, "rb") as real_file:
        	trainT = np.squeeze(pickle.load(real_file))
    	

        print(f"Processed data: {trainX.shape} (Idx x MaxSeqLen x Features)\n")

        self.args.feature_dim = trainX.shape[-1]
        self.args.Z_dim = trainX.shape[-1]

        # Train-Test Split data and time
        """train_data, test_data, train_time, test_time, train_labels, test_labels = train_test_split(
            X, T, labels, test_size=self.args.train_rate, random_state=self.args.seed, shuffle=False
        )"""
        
        train_data = trainX
        train_labels = trainY

        print(f'Train data shape: {train_data.shape}')

        # #########################
        # # Initialize and Run model
        # #########################

        # Log start time
        start = time.time()

        model = TimeGAN(self.args) 
        if self.args.is_train == True:  
            self.timegan_trainer(model, train_data, trainT, train_labels, self.args, generate_cfg)
        generated_data, generated_labels = self.timegan_generator(model, trainT, train_labels, self.args, generate_cfg)
        #generated_time = train_time 

        # Log end time
        end = time.time()

        print(generated_data.shape)
        print(f"Generated data preview:\n{generated_data[:2, -10:, :2]}\n")
        print(f"Model Runtime: {(end - start) / 60} mins\n")
        print(f'Generated labels: {generated_labels}, {np.unique(generated_labels)}')

        # #########################
        # # Save train and generated data for visualization
        # #########################

        # Save splitted data and generated data
        with open(f"{self.args.model_path}/train_data_{generate_cfg['name']}.pickle", "wb") as fb:
            pickle.dump(train_data, fb)
        with open(f"{self.args.model_path}/train_labels_{generate_cfg['name']}.pickle", "wb") as fb:
            pickle.dump(train_labels, fb)
        with open(f"{self.args.model_path}/fake_data_{generate_cfg['name']}.pickle", "wb") as fb:
            pickle.dump(generated_data, fb)
        with open(f"{self.args.model_path}/fake_labels_{generate_cfg['name']}.pickle", "wb") as fb:  
            pickle.dump(generated_labels, fb) 


