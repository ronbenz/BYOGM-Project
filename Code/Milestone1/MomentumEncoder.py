import copy

import VAE_training
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class MomentumEncoder:
    def __init__(self, online, weights_factor, feature_maps_indices):
        self.encoder = copy.deepcopy(online)
        self.weights_factor = weights_factor
        self.feature_maps_indices = feature_maps_indices

    def update_target_weights(self, online_model):
        target_model = self.encoder
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_weight, online_weight = target_param.data, online_param.data
            target_param.data = target_weight*self.weights_factor + (1-self.weights_factor)*online_weight
        # for target_buffer, online_buffer in zip(target_model.buffers(), online_model.buffers()):
        #     target_buffer_data, online_buffer_data = target_buffer.data, online_buffer.data
        #     target_buffer.data = target_buffer_data*self.weights_factor + (1-self.weights_factor)*online_buffer_data

    def extract_feature_maps(self, x):
        feature_maps = []

        for idx, module in self.encoder.layers._modules.items():
            x = module(x)
            if int(idx) in self.feature_maps_indices:
                feature_maps.append(x)
        return feature_maps

    def calc_loss(self, input, target):
        loss = F.mse_loss(input, target, reduction='sum')
        input_features = self.extract_feature_maps(input)
        target_features = self.extract_feature_maps(target)
        for input_feature, target_feature in zip(input_features, target_features):
            loss += F.mse_loss(input_feature, target_feature, reduction='sum')
        return loss


# def main():
#     vae_encoder = VAE.VaeEncoder(3, 128)
#     for name, module in vae_encoder.layers._modules.items():
#         print(f"layer #{name}, module:", module)
#
#
# if __name__ == '__main__':
#     main()
