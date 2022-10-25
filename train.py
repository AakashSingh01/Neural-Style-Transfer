from helper.helper import Helper
import numpy as np
import matplotlib.pyplot as plt
from helper.config import *
import torch.optim as optim
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

helper = Helper()
vgg = helper.load_vgg(device)

content = helper.load_image('images/content.jpg').to(device)
style = helper.load_image('images/style.jpg', shape=content.shape[-2:]).to(device)

content_features =  helper.get_features(content, vgg, layers )
style_features =  helper.get_features(style, vgg, layers )

style_grams = {layer:  helper.gram_matrix(style_features[layer]) for layer in style_features}

target = content.clone().requires_grad_(True).to(device)
optimizer = optim.Adam([target], lr=0.003)

for ii in range(1, steps+1):
    
    target_features = helper.get_features(target, vgg, layers)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    style_loss = 0

    for layer in style_weights:

        target_feature = target_features[layer]
        target_gram = helper.gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (d * h * w)
        
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    # display intermediate images and print the loss
    if  ii % show_every == 0:
        print(ii, ' Total loss: ', total_loss.item())
        plt.imshow(helper.im_convert(target))
        name = "result/%05d.jpg" % (ii)
        plt.savefig(name)