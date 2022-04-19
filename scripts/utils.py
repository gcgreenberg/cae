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
    log(logfile,str(par))
    
def print_epoch(epoch):
    return epoch<=5 or epoch%20==0

def errors_str(*errors):
    error_X, error_Y, error_Z, KL, reg_Z = [e.item() for e in errors]
    error = error_X + error_Y + error_Z + KL + reg_Z
    return 'X {:.5f}.  Y {:.5f}.  Z {:.5f}. KL {:.5f}.  Reg {:.5f}.  Total {:.5f}'.format(
        error_X, error_Y, error_Z, KL, reg_Z, error) 

def errors_str_vae(*errors):
    error_X, KL, reg_Z = [e.item() for e in errors]
    error = error_X + KL + reg_Z
    return 'X {:.5f}.  KL {:.5f}.  Reg {:.5f}.  Total {:.5f}'.format(
        error_X, KL, reg_Z, error) 