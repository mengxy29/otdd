import os
import matplotlib

from otdd.pytorch.datasets import load_torchvision_data
from otdd.pytorch.flows import OTDD_Gradient_Flow
from otdd.pytorch.flows import CallbackList, ImageGridCallback, TrajectoryDump

# Load datasets
loaders_src = load_torchvision_data('MNIST', valid_size=0, resize = 28, maxsize=2000)[0]
loaders_tgt = load_torchvision_data('USPS',  valid_size=0, resize = 28, maxsize=2000)[0]


outdir =  os.path.join('out', 'flows')
callbacks = CallbackList([
  ImageGridCallback(display_freq=2, animate=False, save_path = outdir + '/grid'),
])

flow = OTDD_Gradient_Flow(loaders_src['train'], loaders_tgt['train'],
                          ### Gradient Flow Args
                          # method = 'xonly-attached',                          
                          method = 'xonly',                          
                          use_torchoptim=True,
                          optim='adam',
                          steps=10,
                          step_size=1,
                          callback=callbacks,              
                          clustering_method='kmeans',                                      
                          ### OTDD Args                          
                          online_stats=True,
                          diagonal_cov = False,
                          device='cpu'
                          )
d,out = flow.flow()