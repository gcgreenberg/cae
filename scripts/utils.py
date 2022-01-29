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
        \u03bb KL: {}
        \u03bb latent: {}
        \u03bb regularization: {}
==============================================================\n""".format(par['out_dir'], par['n_pcs'], par['batch'], par['lr'], par['lambda_mod1'], par['lambda_mod2'], par['lambda_kl'], par['lambda_latent'], par['lambda_reg']))
    
def print_epoch(epoch):
    return epoch<=5 or epoch%20==0

def errors_str(*errors):
    error_X, error_Y, error_Z, KL, reg_Z = [e.item() for e in errors]
    error = error_X + error_Y + error_Z + reg_Z
    return 'X {:.5f}.  Y {:.5f}.  Z {:.5f}. KL {:.5f}.  Reg {:.5f}.  Total {:.5f}'.format(
        error_X, error_Y, error_Z, KL, reg_Z, error) 