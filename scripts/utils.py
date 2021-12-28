from datetime import datetime, date
import string
import os
import numpy as np
import shutil

def init_out_dir(out_dir):
#     out_dir = os.path.join('/home/gcgreen2/neurips_comp/out', date.today().strftime("%m-%d") + '-' \
#         ''.join(np.random.choice(list(string.ascii_lowercase), size=3, replace=True)))
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile('/home/gcgreen2/neurips_comp/cae/config.py', os.path.join(out_dir,'config.py'))

model_str = lambda par: 'models.' + par['model'] + '()'

def log(logfile, text, newline=False):
    start = '\n' if newline else ''
    with open(logfile, 'a') as fh:
        now = datetime.now().strftime("%m/%d %H:%M:%S")
        fh.write(start + now + ':   ' + text + '\n')
        
def logheader(logfile, par):
    log(logfile,"""
        output dir: {}
        n_pcs: {}
        batch size: {}
        learning rate: {}
        lambda_mod2: {}
        lambda_latent: {}
==============================================================\n""".format(
            par['out_dir'], par['n_pcs'], par['batch'], par['lr'], par['lambda_mod2'], par['lambda_latent']))
    
def print_epoch(epoch):
    return epoch<=5 or epoch%20==0

def errors_str(error_X, error_Y, error_Z, reg_Z):
    error = error_X + error_Y + error_Z + reg_Z
    return 'X {:.2f}.  Y {:.2f}.  Z {:.2f}. Reg {:.2f}.  Total {:.2f}'.format(
        error_X.item(), error_Y.item(), error_Z.item(), reg_Z.item(), error.item()) 