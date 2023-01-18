import torch
import torch.nn.functional as F
import torchvision.models as models
from fid_score import get_batch_as_rgb
from collections import namedtuple
import torchvision
import VAE_training

# works only with vgg16 model
class VggPerceptualLossNetwork(torch.nn.Module):
    def __init__(self,dataloader):
        super(VggPerceptualLossNetwork, self).__init__()
        self.vgg_model = models.vgg16(pretrained=True)
        # self.vgg_model = models.vgg19(pretrained=True)
        if torch.cuda.is_available():
            self.vgg_model.cuda()
        self.vgg_layers = self.vgg_model.features
        # # vgg16 layers mapping
        self.layer_name_mapping = {
             '1': "relu1_1",
             '6': "relu2_1",
             '11': "relu3_1",
             '18': "relu4_1",
             '25': "relu5_1",
         }
        # vgg19 layers mapping
        # self.layer_name_mapping = {
        #    '1': "relu1_1",
        #    '6': "relu2_1",
        #    '11': "relu3_1",
        #    '20': "relu4_1",
        #    '29': "relu5_1",
        # }
        self.dataset_mean, self.dataset_std = VAE_training.get_dataset_mean_and_std(dataloader)



    def forward(self, x):
        FeatureMaps = namedtuple("LossOutput", ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"])
        feature_maps = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                feature_maps[self.layer_name_mapping[name]] = x
        return FeatureMaps(**feature_maps)

    def normalize_batch(self, batch):
        transform = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        #transform = torchvision.transforms.Normalize(mean=self.dataset_mean, std=self.dataset_std)
        return transform(batch)

    def calc_loss(self, input, target):
        input = get_batch_as_rgb(input)
        target = get_batch_as_rgb(target)
        input = self.normalize_batch(input)
        target = self.normalize_batch(target)
        input_features = self.forward(input)
        target_features = self.forward(target)
        loss = F.mse_loss(input, target, reduction='sum')
        for input_feature, target_feature in zip(input_features, target_features):
            loss += F.mse_loss(input_feature, target_feature, reduction='sum')
        return loss


