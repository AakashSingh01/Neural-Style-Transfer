import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_weights = {'conv1_1': 1.,
                 'conv2_1': 0.75,
                 'conv3_1': 0.2,
                 'conv4_1': 0.2,
                 'conv5_1': 0.2}

layers = {'0': 'conv1_1',
          '5': 'conv2_1', 
          '10': 'conv3_1', 
          '19': 'conv4_1',
          '21': 'conv4_2',  ## content representation
          '28': 'conv5_1'}
content_weight = 1  # alpha
style_weight = 1e9  # beta
show_every = 200
steps = 5000  
