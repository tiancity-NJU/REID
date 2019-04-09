from __future__ import absolute_import
import torch.nn as nn
import torch
import math
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from .resnet import ResNet

class PCB(nn.Module):
    def __init__(self,num_class = 0, last_stride = 1,num_stripes = 6,local_conv_out_channel = 256):

        super(PCB,self).__init__()


        self.base = ResNet(last_stride=last_stride)

        self.num_stripes = num_stripes
        self.num_class = num_class

        self.local_conv_list=nn.ModuleList()
        for _ in range(num_stripes):
            self.local_conv_list.append(nn.Sequential(
                nn.Conv2d(2048,local_conv_out_channel,kernel_size=1),
                nn.BatchNorm2d(local_conv_out_channel),
                nn.ReLU(inplace=True)
            ))

        if self.num_class > 0:
            self.fc_list = nn.ModuleList()
            for _ in range(num_stripes):
                fc = nn.Linear(local_conv_out_channel,num_class)
                init.normal_(fc.weight,std=0.001)
                init.constant_(fc.bias,0)
                self.fc_list.append(fc)


    def forward(self, x):

        """
        :param x:
        :return:
            local_feat_list
            local_logits_list
        """

        feat=self.base(x)
        assert feat.size(2)%self.num_stripes == 0
        stripes_h = int(feat.size(2)/self.num_stripes)
        local_feat_list = []
        local_logits_list = []
        for i in range(self.num_stripes):
            local_feat = F.avg_pool2d(feat[:,:,i*stripes_h:(i+1)*stripes_h,:],(stripes_h,feat.size(-1)))
            local_feat = self.local_conv_list[i](local_feat)   # N*256*1*1
            local_feat = local_feat.view(local_feat.size(0),-1)
            local_feat_list.append(local_feat)

            local_logits = self.fc_list[i](local_feat)
            if self.training:
                local_logits_list.append(local_logits)

        embed_feat = torch.stack(local_feat_list)
        embed_feat = embed_feat.permute(1, 0, 2).contiguous()
        embed_feat = embed_feat.view(embed_feat.size(0), -1)

        if self.training:
            return [embed_feat],local_logits_list    # during training. we just need logits_list

        return embed_feat     # b * 1536 embedding feature

    def get_optim_policy(self):
        params = [
            {'params': self.base.parameters()},
            {'params': self.local_conv_list.parameters()},
            {'params': self.fc_list.parameters()},
        ]
        return params



