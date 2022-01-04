import sys
sys.path.append('/home/gcgreen2/neurips_comp/cae')
import config
from scripts import init, train

init.init(config)
train.train(config)
