import json
import importlib
import numpy as np
import torch
import yaml
import os
import pandas as pd
from glob import glob
from torch.utils.data import Dataset, DataLoader
import dask.dataframe as dd
import re
from functools import partial
from preprocess.data_preprocess_allDatasets import data_preprocess
from typing import Tuple, List
from tqdm import tqdm

class GaitDatasetChunked(Dataset):
    def __init__(self, yaml_dirs, csv_dirs, column_names, chunk_size=1000, fixed_len=64, sample_size=1000, max_seq_len=64, padding_value=0.0, impute_method="mode", scaling_method="minmax"):
        self.yaml_files = []
        self.csv_files = []
        self.labels = []
        self.column_names = column_names
        self.chunk_size = chunk_size
        self.pd_csic_identifier = "PD-CSIC"
        self.fixed_len =  fixed_len
        self.sample_size = sample_size
        self.max_seq_len = max_seq_len
        self.padding_value = padding_value
        self.impute_method = impute_method
        self.scaling_method = scaling_method
        
        for yaml_dir, csv_dir in zip(yaml_dirs, csv_dirs):
        	found_yaml_files = glob(os.path.join(yaml_dir, "*.yaml"))
        	found_csv_files = glob(os.path.join(csv_dir, "*.csv"))
        	
        	filtered_yaml_files = [file_ for file_ in found_yaml_files if "_gait_events" in os.path.basename(file_)]
        	if not filtered_yaml_files:
        		filtered_yaml_files = [file_ for file_ in found_yaml_files if "_gaitEvents" in os.path.basename(file_)]
        		
        	filtered_csv_files = [file_ for file_ in found_csv_files if "Accelerometer" in os.path.basename(file_)]
        	
        	self.yaml_files.extend(filtered_yaml_files)
        	self.csv_files.extend(filtered_csv_files)
        	
        	# Y labels
        	if "GaitDatabase" in yaml_dir: # For GaitDatabase data
        		filtered_yaml_files = [file_ for file_ in found_yaml_files if "Info" in os.path.basename(file_)]        		
        	elif "GePi_cond1" in yaml_dir:
        		filtered_yaml_files = [file_ for file_ in found_yaml_files if "Info" in os.path.basename(file_)]
        	elif "PD-CSIC" in yaml_dir:
        		filtered_yaml_files = [file_ for file_ in found_yaml_files if "info" in os.path.basename(file_)]        		
        		
        	self.labels.extend(filtered_yaml_files)   
        self.segments, self.labels_values = self._create_segments()
        self.max_length = self._calculate_mean_segment_size()
        self.max_length = 2 ** int(np.ceil(np.log2(self.max_length)))
        
        # Compute global scaling parameters
        self.global_min, self.global_max = self._compute_global_scaling_params()
        
    def _compute_global_scaling_params(self) -> Tuple[float, float]:
    	global_min = np.inf
    	global_max = -np.inf
    	
    	for csv_file in tqdm(self.csv_files):
    		try:
    			for chunk in pd.read_csv(csv_file, chunksize=self.chunk_size, usecols=self.column_names):
    				chunk_min = chunk.min().min()
    				chunk_max = chunk.max().max()
    				if chunk_min < global_min:
    					global_min = chunk_min
    				if chunk_max > global_max:
    					global_max = chunk_max
    		except Exception as e:
    			print(f"Error reading {csv_file} for scaling params: {e}")
    			continue
    			
    	print(f"Global min: {global_min}, Global max: {global_max}")
    	return global_min, global_max
        
        
    def _read_yaml(self, file_path):
        try:
            with open(file_path, 'r') as stream:
                return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None
            
    def _replace_and_check(self, yaml_file_name):
    	
    	pattern = re.compile(r'^(?P<prefix>.*?_)(cond_.*?_run_.*?gait[_]?events\.yaml)$', re.IGNORECASE)
    	
    	match = pattern.match(yaml_file_name)
    	if match:
    		prefix = match.group('prefix')
    		
    		
    	# Generate thw two possible replacements
    	replacement_info = f"{prefix}Info.yaml"
    	replacement_info_lower = f"{prefix}info.yaml"
    	
    	if replacement_info in self.labels:
    		return replacement_info
    	elif replacement_info_lower in self.labels:
    		return replacement_info_lower
    	else:
    		return None
        
    	

    def _find_matching_csv(self, yaml_file_path):    	
    	base_names = [yaml_file_path.replace('_gaitEvents.yaml', '_Accelerometer.csv'),
    	yaml_file_path.replace('_gait_events.yaml', '_Accelerometer.csv')]
    	    	
    	for csv_file in self.csv_files:
    		for base_name in base_names:
    			if base_name == csv_file:
    				if self._csv_has_required_columns(csv_file):
    					yaml_info_file = self._replace_and_check(yaml_file_path)
    					return csv_file, yaml_info_file   
    	
    	return None, None
        

    def _csv_has_required_columns(self, csv_file):
    	try:
    		df = pd.read_csv(csv_file, nrows=1)
    		return all(col in df.columns for col in self.column_names)
    	except Exception as e:
    		print(f"Error reading {csv_file}: {e}")
    		return False

    def _match_yaml_and_csv_files(self):
        matched_pairs = []
        for yaml_file in self.yaml_files:
            csv_file = self._find_matching_csv(yaml_file)
            if csv_file:
                matched_pairs.append((yaml_file, csv_file))
        return matched_pairs
        
    def _swap_pd_csic_columns(self, dataframe):
        x_columns = [col for col in dataframe.columns if 'x' in col]
        z_columns = [col for col in dataframe.columns if 'z' in col]
        
        if len(x_columns) == len(z_columns):
            for x_col, z_col in zip(x_columns, z_columns):
                dataframe[x_col], dataframe[z_col] = dataframe[z_col].copy(), dataframe[x_col].copy()
        
        return dataframe

    def _get_csv_data_between(self, csv_file, start, end, first):
        selected_data = []
        try:
        	#print(csv_file, start, end, first)
        	dtype_dict = {col: 'float64' for col in self.column_names}
        	
        	for chunk in pd.read_csv(csv_file, chunksize=self.chunk_size, dtype=dtype_dict):
        		if 'time' not in chunk.columns:
        			raise ValueError(f"'time' column not found in {csv_file}")
        			
        		if first:
        			mask = chunk['time'] <= end
        		else:
        			mask = (chunk['time'] > start) & (chunk['time'] <= end)
        		
        		filtered_chunk = chunk.loc[mask, self.column_names]
        		selected_data.append(filtered_chunk)
        		
        		if chunk['time'].max() > end:
        			break
        			
        	if selected_data:
        		filtered_data = pd.concat(selected_data, ignore_index=True)
        	else:
        		filtered_data = pd.DataFrame(columns=self.column_names)
        		
        	if "PD-CSIC" in csv_file:
        		filtered_data = self._swap_pd_csic_columns(filtered_data)
        	return filtered_data
        except Exception as e:
        	print(f"Error processing {csv_file} between {start} and {end}: {e}")
        	return pd.DataFrame(columns=self.column_names)

    def _create_segments(self):
    
    	segments = []
    	labels_per_segment = []
    	for yaml_file in self.yaml_files:
    		csv_file, yaml_labels_file = self._find_matching_csv(yaml_file)
    		if not csv_file:
    			continue
    			
    		gait_events = self._read_yaml(yaml_file)
    		if not gait_events:
    			continue
    		
    		if 'l_heel_strike' in gait_events:
    			l_heel_strike = gait_events['l_heel_strike']
    		elif 'IC' in gait_events:
    			l_heel_strike = gait_events['IC']
    		else:
    			raise ValueError(f"l_heel_strike nor IC found in {yaml_file}")
    			
	    	for i in range(len(l_heel_strike) - 1):
	    		start, end, ini = (l_heel_strike[i], l_heel_strike[i], True) if i == 0 else (l_heel_strike[i - 1], l_heel_strike[i], False)
	    		segments.append((yaml_file, csv_file, start, end, ini))
	    		labels = self._obtain_labels(yaml_labels_file)	
	    		labels_per_segment.append(labels)	
	    	
	    		
    	return segments, labels_per_segment
    	
    def _obtain_labels(self, yaml_labels):
    	labels = self._read_yaml(yaml_labels)
    	
    	if 'label' in labels:	
    		value = labels['label']
    	elif 'stage' in labels:
    		value = labels['stage'][0]
    	else:
    		value = 0
    	return value
    	
    def _calculate_mean_segment_size(self):
    	max_val = 0    	
    	segm_mean = []
    	
    	print("Calculating mean segment size...")
    	for i, (_, csv_file, start, end, ini) in enumerate(self.segments):    		
    		if i < self.sample_size:
    			csv_data = self._get_csv_data_between(csv_file, start, end, ini)
    			segm_mean.append(csv_data.shape[0])
    		else:
    			break
    			
    	if not segm_mean:
    		return self.max_seq_len
    			
    	mean_val = int(np.mean(segm_mean))
    	return mean_val

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        yaml_file, csv_file, start, end, ini = self.segments[idx]
        label = self.labels_values[idx]
        csv_data = self._get_csv_data_between(csv_file, start, end, ini)
        
        # Preprocess the data segment
        processed_segment, seq_len = data_preprocess(df=csv_data, max_seq_len=self.max_seq_len, resample_len=64, padding_value=self.padding_value, impute_method=self.impute_method, scaling_params=(self.global_min, self.global_max))
        
        data_tensor = torch.tensor(processed_segment, dtype=torch.float32)
        
        # Convert label to tensor
        label_tensor = torch.tensor(int(label),dtype=torch.long)
        
        return data_tensor, label_tensor

    @staticmethod    
    def custom_collate_fn(batch):
        data, labels = zip(*batch)
        data = torch.stack(data, dim=0)
        labels = torch.stack(labels, dim=0)
        return data, labels
