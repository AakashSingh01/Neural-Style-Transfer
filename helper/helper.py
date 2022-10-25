from PIL import Image
from torchvision import transforms, models
import torch
import numpy as np

class Helper:

    def load_image(self, img_path, max_size=400, shape=None):
        ## making sure the image is <= 400 pixels
        ## large images will slow down processing

        image = Image.open(img_path).convert('RGB')
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)

        if shape is not None:
            size = shape
        ## (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) is the mean and std of Imagenet
        ## Using the mean and std of Imagenet is a common practice. 
        
        in_transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), 
                                                 (0.229, 0.224, 0.225))])

        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3,:,:].unsqueeze(0)

        return image

    def load_vgg(self, device):
        vgg = models.vgg19(pretrained=True).features
        # freeze all VGG parameters since we're only optimizing the target image
        for param in vgg.parameters():
            param.requires_grad_(False)
        vgg.to(device)
        return vgg
    
    def im_convert(self, tensor):
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1,2,0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)

        return image

    def get_features(self, image, model, layers=None):
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features

    def gram_matrix(self, tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())
        return gram


