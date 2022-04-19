import sys, os

sys.path.append('/home/gcgreen2/neurips_comp/cae')
import config, config_vae
from scripts import init, train, train_vae

try: option = sys.argv[1]
except: option = '0'
print('option', option, ':', 'cvae' if option=='0' else 'vae' if option=='1' else 'invalid')

if option == '0':
    init.init(config)
    train.train(config)
elif option == '1':
    init.init(config_vae)
    train_vae.train_vae(config_vae)
    
# os.system('jupyter nbconvert --execute '+config.files['eval_cp'])
