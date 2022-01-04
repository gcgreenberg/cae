from datetime import datetime, date
import string
import os
import numpy as np
import shutil
    
model_str = lambda par: 'models.' + par['model'] + '(**par)'

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
        \u03bb mod1: {}
        \u03bb mod2: {}
        \u03bb latent: {}
        \u03bb regularization: {}
==============================================================\n""".format(par['out_dir'], par['n_pcs'], par['batch'], par['lr'], par['lambda_mod1'], par['lambda_mod2'], par['lambda_latent'], par['lambda_reg']))
    
def print_epoch(epoch):
    return epoch<=5 or epoch%20==0

def errors_str(error_X, error_Y, error_Z, reg_Z):
    error = error_X + error_Y + error_Z + reg_Z
    return 'X {:.2f}.  Y {:.2f}.  Z {:.2f}. Reg {:.2f}.  Total {:.2f}'.format(
        error_X.item(), error_Y.item(), error_Z.item(), reg_Z.item(), error.item()) 