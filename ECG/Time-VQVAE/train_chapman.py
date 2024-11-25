from Datasets import *
import torch
import wandb
import copy
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from experiments.exp_stage1 import ExpStage1
from experiments.exp_stage2 import ExpStage2
from experiments.exp_fidelity_enhancer import ExpFidelityEnhancer
from evaluation.evaluation import Evaluation
from utils import get_root_dir, load_yaml_param_settings, save_model, str2bool

def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config_chapman.yaml'))
    parser.add_argument('--dataset_name', help="e.g., PTBXL.", default='Chapman_100-combined')
    parser.add_argument('--gpus', default=1, type=int, help='Number of GPU devices to use.')
    parser.add_argument('--feature_extractor_type', type=str, default='rocket', help='supervised_fcn | rocket')
    parser.add_argument('--use_custom_dataset', type=str2bool, default=True, help='Using a custom dataset, then set it to True.')
    return parser.parse_args()

def train_stage1(config: dict,
                 dataset_name: str,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpus,
                 ):
    """
    :param do_validate: if True, validation is conducted during training with a test dataset.
    """
    project_name = config["exp_params"]["exp_name"]
    log_dir = os.path.join('models', project_name)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
            print('Created log dir ', log_dir)
        except:
            print('Rewriting log dir ', log_dir)
    else:
        print('Rewriting log dir ', log_dir)

    # fit
    _, in_channels, input_length = train_data_loader.dataset.X.shape
    train_exp = ExpStage1(in_channels, input_length, config)

    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb_logger = WandbLogger(project=project_name, name=None, config={**config, 'dataset_name': dataset_name, 'n_trainable_params:': n_trainable_params})

    # Check if GPU is available
    if not torch.cuda.is_available():
        print('GPU is not available.')
        # num_cpus = multiprocessing.cpu_count()
        num_cpus = 1
        print(f'using {num_cpus} CPUs..')
        device = num_cpus
        accelerator = 'cpu'
    else:
        accelerator = 'gpu'
        device = gpus

    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='step')],
                         max_steps=config['trainer_params']['max_steps']['stage1'],
                         devices=1,
                         accelerator=accelerator,
                         val_check_interval=config['trainer_params']['val_check_interval']['stage1'],
                         check_val_every_n_epoch=None,
                         )
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader
                )
    # test
    print('closing...')
    wandb.finish()

    print('saving the models...')
    save_model({'encoder_l': train_exp.encoder_l,
                'decoder_l': train_exp.decoder_l,
                'vq_model_l': train_exp.vq_model_l,
                'encoder_h': train_exp.encoder_h,
                'decoder_h': train_exp.decoder_h,
                'vq_model_h': train_exp.vq_model_h,
                }, log_dir=log_dir)

    print('saving checkpoint...')
    trainer.save_checkpoint(f'{log_dir}/stage1.ckpt')

def train_stage2(config: dict,
                 dataset_name: str,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpus,
                 feature_extractor_type:str,
                 use_custom_dataset:bool,
                 ):
    project_name = config["exp_params"]["exp_name"]
    log_dir = os.path.join('models', project_name)

    # fit
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    _, in_channels, input_length = train_data_loader.dataset.X.shape
    train_exp = ExpStage2(dataset_name, train_dataset, test_dataset, in_channels, input_length, config, n_classes, feature_extractor_type, use_custom_dataset)
    
    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb_logger = WandbLogger(project=project_name, name=None, 
                               config={**config, 'dataset_name': dataset_name, 'n_trainable_params': n_trainable_params, 'feature_extractor_type':feature_extractor_type})
    
    # Check if GPU is available
    if not torch.cuda.is_available():
        print('GPU is not available.')
        # num_cpus = multiprocessing.cpu_count()
        num_cpus = 1
        print(f'using {num_cpus} CPUs..')
        device = num_cpus
        accelerator = 'cpu'
    else:
        accelerator = 'gpu'
        device = gpus

    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='step')],
                         max_steps=config['trainer_params']['max_steps']['stage2'],
                         devices=device,
                         accelerator=accelerator,
                         val_check_interval=config['trainer_params']['val_check_interval']['stage2'],
                         check_val_every_n_epoch=None,
                         )
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader
                )

    wandb.finish()

    print('saving the models...')
    save_model({'maskgit': train_exp.maskgit}, log_dir=log_dir)
    print('saving checkpoint...')
    trainer.save_checkpoint(f'{log_dir}/stage2.ckpt')

    # test
    """
    print('evaluating...')
    eval_device = device[0] if accelerator == 'gpu' else 'cpu'
    evaluation = Evaluation(dataset_name, train_dataset, test_dataset, in_channels, input_length, 
            n_classes, eval_device, config, use_fidelity_enhancer=False, feature_extractor_type=
            feature_extractor_type, use_custom_dataset=use_custom_dataset).to(eval_device)
    min_num_gen_samples = config['evaluation']['min_num_gen_samples']  # large enough to capture the distribution
    (_, _, x_gen), _ = evaluation.sample(max(evaluation.X_test.shape[0], min_num_gen_samples), 'unconditional')
    # z_train = evaluation.z_train
    z_test = evaluation.z_test
    z_gen = evaluation.compute_z_gen(x_gen)

    # fid_train = evaluation.fid_score(z_test, z_gen)
    wandb.log({'FID': evaluation.fid_score(z_test, z_gen)})
    if not use_custom_dataset:
        IS_mean, IS_std = evaluation.inception_score(x_gen)
        wandb.log({'IS_mean': IS_mean, 'IS_std': IS_std})


    # evaluation.log_visual_inspection(evaluation.X_train, x_gen, 'X_train vs X_gen')
    evaluation.log_visual_inspection(evaluation.X_test, x_gen, 'X_test vs Xhat')
    # evaluation.log_visual_inspection(evaluation.X_train, evaluation.X_test, 'X_train vs X_test')

    # evaluation.log_pca([z_train, z_gen], ['z_train', 'z_gen'])
    evaluation.log_pca([z_test, z_gen], ['Z_test', 'Zhat'])
    # evaluation.log_pca([z_train, z_test], ['z_train', 'z_test'])
    """

def train_stage_fid_enhancer(config: dict,
                 dataset_name: str,
                 train_dataset: Dataset,
                 test_dataset: Dataset,
                 train_data_loader: DataLoader,
                 test_data_loader: DataLoader,
                 gpus,
                 feature_extractor_type:str,
                 use_custom_dataset:bool,
                 ):
    
    project_name = config["exp_params"]["exp_name"]
    log_dir = os.path.join('models', project_name)

    # fit
    n_classes = len(np.unique(train_data_loader.dataset.Y))
    _, in_channels, input_length = train_data_loader.dataset.X.shape
    train_exp = ExpFidelityEnhancer(dataset_name, train_dataset, test_dataset, in_channels, input_length, config, n_classes, feature_extractor_type, use_custom_dataset)

    n_trainable_params = sum(p.numel() for p in train_exp.parameters() if p.requires_grad)
    wandb_logger = WandbLogger(project=project_name, name=None, config={**config, 'dataset_name':dataset_name, 'n_trainable_params':n_trainable_params})

    # Check if GPU is available
    if not torch.cuda.is_available():
        print('GPU is not available.')
        # num_cpus = multiprocessing.cpu_count()
        num_cpus = 1
        print(f'using {num_cpus} CPUs..')
        device = num_cpus
        accelerator = 'cpu'
    else:
        accelerator = 'gpu'
        device = gpus
    
    eval_device = 'cuda:0' if accelerator == 'gpu' else 'cpu'
    train_exp.search_optimal_tau(X_train=train_data_loader.dataset.X, device=eval_device, wandb_logger=wandb_logger)

    trainer = pl.Trainer(logger=wandb_logger,
                         enable_checkpointing=False,
                         callbacks=[LearningRateMonitor(logging_interval='step')],
                         max_steps=config['trainer_params']['max_steps']['stage_fid_enhancer'],
                         devices=device,
                         accelerator=accelerator,
                         val_check_interval=config['trainer_params']['val_check_interval']['stage_fid_enhancer'],
                         check_val_every_n_epoch=None)
    trainer.fit(train_exp,
                train_dataloaders=train_data_loader,
                val_dataloaders=test_data_loader
                )

    print('saving the model...')
    save_model({'fidelity_enhancer': train_exp.fidelity_enhancer}, log_dir=log_dir)

    wandb.finish()

if __name__ == '__main__':
    # load config
    args = load_args()
    config = load_yaml_param_settings(args.config)
    wandb.init(project="Time_VQVAE", entity="zrfnz")
    wandb.config.update(config)
    torch.set_float32_matmul_precision('medium')

    train_dataset = multi_channel(filename='../../data/ecg/chapman_100-combined_train.csv')
    test_dataset = multi_channel(filename='../../data/ecg/chapman_100-combined_test.csv')
    # train
    print("stage: ",config["stage"])
    if config["stage"] == 1:
        train_loader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_sizes"]["stage1"], 
            num_workers=12, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config["dataset"]["batch_sizes"]["stage1"], 
            num_workers=12, shuffle=True, drop_last=True)
        train_stage1(config, args.dataset_name, train_loader, test_loader, args.gpus)
    elif config["stage"] == 2:
        train_loader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_sizes"]["stage2"], 
            num_workers=12, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config["dataset"]["batch_sizes"]["stage2"], 
            num_workers=12, shuffle=True, drop_last=True)
        train_stage2(config, args.dataset_name, train_dataset, test_dataset, train_loader, test_loader, 
                args.gpus, args.feature_extractor_type, args.use_custom_dataset)
    elif config["stage"] == 3:
        train_loader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_sizes"]["stage3"], 
            num_workers=12, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=config["dataset"]["batch_sizes"]["stage3"], 
            num_workers=12, shuffle=True, drop_last=True)
        train_stage_fid_enhancer(config, args.dataset_name, train_dataset, test_dataset, train_loader, 
                test_loader, args.gpus, args.feature_extractor_type, args.use_custom_dataset)
