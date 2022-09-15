import torch
import torch.nn.functional as F
import torchvision.models as models
from collections import namedtuple


# works only with vgg16 model
class VggPerceptualLossNetwork(torch.nn.Module):
    def __init__(self):
        super(VggPerceptualLossNetwork, self).__init__()
        self.vgg_model = models.vgg16(pretrained=True)
        if torch.cuda.is_available():
            self.vgg_model.cuda()
        self.vgg_layers = self.vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3",
            '22': "relu4_3"
        }

    def forward(self, x):
        FeatureMaps = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        feature_maps = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                feature_maps[self.layer_name_mapping[name]] = x
        return FeatureMaps(**feature_maps)

    def calc_loss(self, input, target):
        input_features = self.forward(input)
        target_features = self.forward(target)
        loss = F.mse_loss(input, target)
        for input_feature, target_feature in zip(input_features, target_features):
            loss += F.mse_loss(input_feature, target_feature, reduction='sum')
        return loss
