import os
import torch

rebel_path = ''
data_path = ''

project_root = os.path.join(rebel_path, 'models/rnd/happy_whale/')
data_folder = os.path.join(data_path, 'train_images')

exp = 'exp1.1'

sub_exp = exp.replace('exp', '') + '.6'

csv_folder = os.path.join(project_root, 'csv')
exp_root = os.path.join(project_root, exp)

logger_path = os.path.join(project_root, f'logs/scalar/{sub_exp}')
check_pointer_path = os.path.join(project_root, f'check_points/{sub_exp}')

AVAIL_GPUS = min(1, torch.cuda.device_count())

