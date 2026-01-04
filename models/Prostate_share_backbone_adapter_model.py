import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import intel_extension_for_pytorch as ipex
except:
    pass

from typing import Union
from tqdm import tqdm
from .base_model import BaseModel
from . import networks3D
from .densenet import *
from monai.networks.layers.factories import Conv, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer


class ResidualDepthwiseConvAdapter(nn.Module):
    def __init__(self, spatial_dims, inplanes=128, planes=128, m=4,
                 act: Union[str, tuple] = ("relu", {"inplace": True}),
                 norm: Union[str, tuple] = "batch"):
        """Residual Adapter with adaptive kernels for convolutional features"""
        super(ResidualDepthwiseConvAdapter, self).__init__()
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        avg_pool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]
        self.conv1 = conv_type(inplanes*4, planes*2, kernel_size=1, bias=False)
        self.conv_reduce1 = conv_type(inplanes*4, planes, kernel_size=1, bias=False)
        self.conv2 = conv_type(inplanes*8, planes*2, kernel_size=1, bias=False)
        self.conv_reduce2 = conv_type(inplanes*8, planes, kernel_size=1, bias=False)
        
        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes)
        self.norm3 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes)
        self.norm4 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes)
        self.relu = get_act_layer(name=act)
        self.prelu = get_act_layer(name='prelu')
        self.conv_reduction1 = conv_type(planes*4, planes//m, kernel_size=1, bias=False)
        self.conv_reduction2 = conv_type(planes*2, planes//m, kernel_size=3, bias=False)
        self.conv_z = conv_type(5, 9, kernel_size=1, bias=False)
        self.conv_double = conv_type(planes//m, planes//m*2, kernel_size=1, bias=False)
        self.gate = conv_type(planes//m*2, planes//8, kernel_size=1)
        self.fuse = conv_type(planes//m*2, planes//8, kernel_size=3, stride=2)
        self.norm_fuse = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=planes//8)
        self.pool = avg_pool_type(1)
        self.flat = nn.Flatten(1)

    def forward(self, f1, f2, f3, f4):
        # f1: [4, 128, 24, 24, 24], f2: [4, 256, 12, 12, 12], f3: [4, 512, 6, 6, 6], f4: [4, 1024, 6, 6, 6]
        batch_size = f1.shape[0]
        feat1_half_chn = self.conv_reduce1(f3) # [4, 128, 6, 6, 6], the output channel of densenet ResBlock1 is 128 which is the input channel of this kernel
        feat1 = self.conv1(f3) # [30, 256, 6, 6, 6]
        feat2 = self.conv2(f4) # [30, 256, 6, 6, 6]
        feat2_half_chn = self.conv_reduce2(f4) # [30, 128, 6, 6, 6]
        f1 = torch.split(f1, 1, 0) # shape of f1: [30, 128, 24, 24, 24],
        f2 = torch.split(f2, 1, 0) # shape of f2: [30, 256, 12, 12, 12]
        feat1 = torch.split(feat1, 1, 0)
        feat2 = torch.split(feat2, 1, 0)
        feat1_half_chn = torch.split(feat1_half_chn, 1, 0)
        feat2_half_chn = torch.split(feat2_half_chn, 1, 0)
        out = []
        pad = (1,1,1,1,1,1)
        for i in range(batch_size):
            # 1*256*6*6*6 -> 256*1*6*6*6
            feat_single1 = feat1[i].permute(1, 0, 2, 3, 4).contiguous()
            kernel1 = feat_single1
            # 1*256*6*6*6 -> 256*1*6*6*6
            feat_single2 = feat2[i].permute(1, 0, 2, 3, 4).contiguous()
            kernel2 = feat_single2
            f2_one = F.pad(input=f2[i], pad=pad, mode='constant', value=0)
            # [1, 256, 9, 9, 9]
            feat2_k1 = F.conv3d(input=f2_one, weight=kernel1, stride=1, groups=f2[i].size(1))
            feat2_k2 = F.conv3d(input=f2_one, weight=kernel2, stride=1, groups=f2[i].size(1))
            feat2_k1 = self.relu(feat2_k1)
            feat2_k2 = self.relu(feat2_k2)
            # [1, 32, 9, 9, 9]
            out1 = self.conv_reduction1(torch.cat([feat2_k1, feat2_k2], dim=1))
            # 1*128*9*9*9 -> 128*1*9*9*9
            feat_half_chn_single1 = feat1_half_chn[i].permute(1, 0, 2, 3, 4).contiguous()
            feat_half_chn_single2 = feat2_half_chn[i].permute(1, 0, 2, 3, 4).contiguous()
            kernel3 = feat_half_chn_single1 # 128*1*9*9*9
            kernel4 = feat_half_chn_single2
            
            # f1_one: [1, 128, 26, 26, 26]
            f1_one = F.pad(input=f1[i], pad=pad, mode='constant', value=0)
            # [1, 128, 11, 11, 11]
            feat1_k3 = F.conv3d(input=f1_one, weight=kernel3, stride=2, groups=f1_one.size(1))
            feat1_k4 = F.conv3d(input=f1_one, weight=kernel4, stride=2, groups=f1_one.size(1))
            feat1_k3 = self.relu(feat1_k3)
            feat1_k4 = self.relu(feat1_k4)
            # [1, 32, 9, 9, 9]
            out2 = self.conv_reduction2(torch.cat([feat1_k3, feat1_k4], dim=1))
            # feature shape should match before concat, [1, 64, 9, 9, 9]
            if out1.size(-1) == out2.size(-1):
                out.append(torch.cat([out1, out2], dim=1))
            else:
                # [1,32,9,9,4] and [1,32,9,9,1] -> [1,32,9,9,5]
                out_tmp = torch.cat([out1, out2], dim=-1)
                # [1,5,9,9,32] -> [1,9,9,9,32]
                self.relu(out_tmp)
                out_tmp = self.conv_z(out_tmp.permute(0, 4, 2, 3, 1).contiguous())
                self.relu(out_tmp)
                out_tmp = self.conv_double(out_tmp.permute(0, 4, 2, 3, 1).contiguous())
                out.append(out_tmp)
        y = torch.cat(out, dim=0) # [4, 16, 9, 9, 9]
        # [4, 16, 1, 1, 1]
        scale = torch.sigmoid(self.gate(self.pool(y)))
        # fuse features: [4, 16, 4, 4, 4]
        y = self.fuse(y)
        y = self.relu(self.norm_fuse(y))
        y = scale * y
        return self.flat(y)
    

class ProstateShareBackboneAdapterModel(BaseModel):
    def name(self):
        return 'ProstateShareBackboneAdapterModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        return parser

    def initialize(self, opt):
        if opt.gpu_ids == 'xpu':
            use_dawn = True
        else:
            use_dawn = False
        if use_dawn:
            self.device = torch.device('xpu')
            self.gpu_ids = ['xpu']

        BaseModel.initialize(self, opt)
        self.class_num = 1
        self.K_neigs = opt.K_neigs
        self.beta = opt.beta
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.using_focalloss = opt.focal
        self.loss_names = ['cls']
        self.train_encoders = opt.train_encoders

        if self.using_focalloss:
            self.loss_names.append('focal')
        
        self.model_names = ['Encoder', 'Classifier', 'Adapter_DWI_T2', 'Adapter_CT_PET']
        dropout_prob = 0.
        self.netEncoder = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=dropout_prob), self.gpu_ids)
        
        self.num_graph_update = opt.num_graph_update
        self.weight_u = opt.weight_u # loss weight for unlabeled data
        self.act_func = 'relu'
        self.netClassifier = torch.nn.Linear(1024, self.class_num)
        self.netAdapter_DWI_T2 = ResidualDepthwiseConvAdapter(spatial_dims=3, inplanes=128, planes=128, act=self.act_func, m=4)
        self.netAdapter_CT_PET = ResidualDepthwiseConvAdapter(spatial_dims=3, inplanes=128, planes=128, act=self.act_func, m=4)
        if len(self.gpu_ids) > 0:
            self.netClassifier.to(self.gpu_ids)
            #self.netClassifier = torch.nn.DataParallel(self.netClassifier, self.gpu_ids)
            self.netAdapter_DWI_T2.to(self.device)
            # # self.netAdapter_MRI = torch.nn.DataParallel(self.netAdapter_MRI, self.gpu_ids)
            self.netAdapter_CT_PET.to(self.device)
        if self.class_num == 1:
            self.criterionCE = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterionCE = torch.nn.CrossEntropyLoss()
        
        self.netDecoder_HGIB = self.netClassifier
        params = [{'params': self.netEncoder.parameters()},
                    {'params': self.netClassifier.parameters()},# 'lr': 10*opt.lr},
                    {'params': self.netAdapter_CT_PET.parameters()},# 'lr': 10*opt.lr},
                    {'params': self.netAdapter_DWI_T2.parameters()}#, 'lr': 10*opt.lr}
                    ]
        # initialize optimizers
        if self.isTrain:
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            if use_dawn:
                self.train()
                self.netEncoder, self.optimizer = ipex.optimize(self.netEncoder, optimizer=self.optimizer, dtype=torch.float32)
                self.netClassifier, self.optimizer = ipex.optimize(self.netClassifier, optimizer=self.optimizer, dtype=torch.float32)
                self.netAdapter_DWI_T2, self.optimizer = ipex.optimize(self.netAdapter_DWI_T2, optimizer=self.optimizer, dtype=torch.float32)
                self.netAdapter_CT_PET, self.optimizer = ipex.optimize(self.netAdapter_CT_PET, optimizer=self.optimizer, dtype=torch.float32)
                
            self.optimizers = []
            self.optimizers.append(self.optimizer)

    def info(self, lens):
        self.len_train = lens[0]
        self.len_test = lens[1]

    def set_input(self, input):
        self.CT = input[0].to(self.device)
        self.PET = input[1].to(self.device)
        self.DWI = input[2].to(self.device)
        self.T2 = input[3].to(self.device)
        self.target = input[4].to(self.device)

    def set_HGinput(self, input=None):
        self.embedding = self.embedding.to(self.device)
        if input is not None:
            self.target = input.to(self.device)

    def ExtractFeatures(self, phase='test'):
        if self.CT.device.type != next(self.netEncoder.parameters()).device.type:
            self.netEncoder = self.netEncoder.to(self.device)
            self.netClassifier = self.netClassifier.to(self.device)
            self.netAdapter_CT_PET = self.netAdapter_CT_PET.to(self.device)
            self.netAdapter_DWI_T2 = self.netAdapter_DWI_T2.to(self.device)
        if phase == 'test':
            with torch.no_grad():
                self.embedding_CT, f1, f2, f3, f4 = self.netEncoder(self.CT, return_feat_map=True)
                feat = self.netAdapter_CT_PET(f1, f2, f3, f4)
                self.embedding_CT += feat
                self.embedding_PET, f1, f2, f3, f4 = self.netEncoder(self.PET, return_feat_map=True)
                feat = self.netAdapter_CT_PET(f1, f2, f3, f4)
                self.embedding_PET += feat
                self.embedding_DWI, f1, f2, f3, f4 = self.netEncoder(self.DWI, return_feat_map=True)
                feat = self.netAdapter_DWI_T2(f1, f2, f3, f4)
                self.embedding_DWI += feat
                self.embedding_T2, f1, f2, f3, f4 = self.netEncoder(self.T2, return_feat_map=True)
                feat = self.netAdapter_DWI_T2(f1, f2, f3, f4)
                self.embedding_T2 += feat
        else:
            self.embedding_CT, f1, f2, f3, f4 = self.netEncoder(self.CT, return_feat_map=True)
            feat = self.netAdapter_CT_PET(f1, f2, f3, f4)
            self.embedding_CT += feat
            self.embedding_PET, f1, f2, f3, f4 = self.netEncoder(self.PET, return_feat_map=True)
            feat = self.netAdapter_CT_PET(f1, f2, f3, f4)
            self.embedding_PET += feat
            self.embedding_DWI, f1, f2, f3, f4 = self.netEncoder(self.DWI, return_feat_map=True)
            feat = self.netAdapter_DWI_T2(f1, f2, f3, f4)
            self.embedding_DWI += feat
            self.embedding_T2, f1, f2, f3, f4 = self.netEncoder(self.T2, return_feat_map=True)
            feat = self.netAdapter_DWI_T2(f1, f2, f3, f4)
            self.embedding_T2 += feat
            
        return self.embedding_CT, self.embedding_PET, self.embedding_DWI, self.embedding_T2

    def HGconstruct(self, embedding_CT, embedding_PET, embedding_DWI, embedding_T2):
        self.embedding = torch.Tensor((embedding_CT + embedding_PET + embedding_DWI + embedding_T2)/4.0).to(self.device)
        
    def forward(self, phase='train', train_loader=None, test_loader=None, train_loader_u=None, epoch=None):
        if phase == 'train':
            prediction_all = []
            embedding_all = []
            targets_all = []
            assert train_loader is not None, 'train_loader is None, please provide train_loader for training'
            len_train_loader = len(train_loader)
            train_loader_x_iter = iter(train_loader)
            iteration = 0
            if train_loader_u is not None:
                len_train_loader = max(len(train_loader), len(train_loader_u))
                train_loader_u_iter = iter(train_loader_u)
            for i in range(len_train_loader): 
                try:
                    data = next(train_loader_x_iter)
                except StopIteration:
                    train_loader_x_iter = iter(train_loader)
                    data = next(train_loader_x_iter)
                
                if epoch is not None:
                    weight_u = self.weight_u * min(epoch / 80., 1.)
                else:
                    weight_u = self.weight_u
                self.set_input(data)
                self.ExtractFeatures(phase='train')
                embedding = (self.embedding_CT+self.embedding_PET+self.embedding_DWI+self.embedding_T2)/4.0

                prediction = self.netClassifier(embedding)
                if self.class_num == 1:
                    prediction, self.target = prediction.squeeze(), self.target.float()
                self.loss_cls = self.criterionCE(prediction, self.target)
                
                prediction_all.append(prediction)
                targets_all.append(self.target)
                embedding_all.append(embedding)
                
                # create hypergraph
                loss_x = torch.tensor(0.)
                if self.using_focalloss:
                    gamma = 0.5
                    alpha = 2
                    pt = torch.exp(-self.loss_cls)
                    self.loss_focal = (alpha * (1 - pt) ** gamma * self.loss_cls).mean()
                    self.loss = self.loss_cls + self.loss_focal
                else:
                    self.loss = self.loss_cls
                
                loss_u = torch.tensor(0.)
                self.loss = self.loss + weight_u * loss_x + weight_u * loss_u
                # self.loss = self.loss + weight_u * loss_x + weight_u * (loss_u + (self.f1_mmd + self.f2_mmd + self.f3_mmd)/3.0)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                iteration += 1
                
                if i % 100 == 0:
                    print('Iteration {}, total loss for encoders {:.5f}, loss_x {:.5f}, loss_u {:.5f} ({:.5f}*{:.5f})'.format(
                        i, self.loss.item(), loss_x.item(),  weight_u * loss_u.item(),  weight_u, loss_u.item()))
                    
            self.prediction_cur = torch.cat(prediction_all, dim=0)
            self.pred_encoder = torch.cat(prediction_all, dim=0)
            self.target_cur = torch.cat(targets_all, dim=0)
            if self.class_num == 1:
                self.accuracy = ((torch.sigmoid(self.prediction_cur) >= 0.5) == self.target_cur).float().sum().float() / float(self.target_cur.size(0))
                self.acc_encoder = ((torch.sigmoid(self.pred_encoder) >= 0.5) == self.target_cur).float().sum().float() / float(self.target_cur.size(0))
            else:
                self.accuracy = (torch.softmax(self.prediction_cur, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target_cur.size(0))
                self.acc_encoder = (torch.softmax(self.pred_encoder, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target_cur.size(0))
            
        elif phase == 'test':
            with torch.no_grad():
                CT, PET, DWI, T2, Label, length = self.get_features([train_loader, test_loader])
                # create hypergraph
                self.HGconstruct(CT, PET, DWI, T2)
                self.info(length)
                self.set_HGinput(Label)
                idx = torch.tensor(range(self.len_test)).to(self.device) + self.len_train
                prediction_CT = self.netClassifier(torch.tensor(CT).to(self.device))
                prediction_PET = self.netClassifier(torch.tensor(PET).to(self.device))
                prediction_DWI = self.netClassifier(torch.tensor(DWI).to(self.device))
                prediction_T2 = self.netClassifier(torch.tensor(T2).to(self.device))
                prediction_encoder = (prediction_CT + prediction_DWI + prediction_PET + prediction_T2) / 4.0
                
                self.prediction = prediction_encoder
                self.prediction_cur = self.prediction[idx]
                self.loss_cls = 0
                self.loss_kl = 0
                self.loss_focal = 0
                self.loss = self.loss_cls
                
                self.target_cur = self.target[idx]
                self.pred_encoder = prediction_encoder[idx]
                self.accuracy = (torch.softmax(self.prediction_cur, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target_cur.size(0))
                self.acc_encoder = (torch.softmax(self.pred_encoder, dim=1).argmax(dim=1) == self.target_cur).float().sum().float() / float(self.target_cur.size(0))
        else:
            print('Wrong in loss calculation')
            exit(-1)


    def optimize_parameters(self, train_loader, test_loader, train_loader_u=None, epoch=None):
        # forward pass is here
        self.netClassifier.train()
        self.netAdapter_CT_PET.train()
        self.netAdapter_DWI_T2.train()
        self.train()
        self.forward('train', train_loader, test_loader, train_loader_u=train_loader_u, epoch=epoch)

    def validation(self):
        self.netClassifier.eval()
        self.netAdapter_CT_PET.eval()
        self.netAdapter_DWI_T2.eval()
        self.eval()
        with torch.no_grad():
            self.forward('test', self.train_loader, self.test_loader)

    def get_pred_encoder(self):
        return self.pred_encoder
    
    def get_acc_encoder(self):
        return self.acc_encoder
    
    def get_features(self, loaders, phase='test'):
        # extract featrues from pre-trained model
        # stack them
        CT = None
        PET = None
        DWI = None
        T2 = None
        Label = None
        length = [0, 0]
        for idx, loader in enumerate(loaders):
            for i, data in enumerate(tqdm(loader)):
                self.set_input(data)
                i_CT, i_PET, i_DWI, i_T2 = self.ExtractFeatures(phase)
                if CT is None:
                    CT = i_CT
                    PET = i_PET
                    DWI = i_DWI
                    T2 = i_T2
                    Label = data[4]
                else:
                    CT = torch.cat([CT, i_CT], 0)
                    PET = torch.cat([PET, i_PET], 0)
                    DWI = torch.cat([DWI, i_DWI], 0)
                    T2 = torch.cat([T2, i_T2], 0)
                    Label = torch.cat([Label, data[4]], 0)
            length[idx] = CT.size(0)
        length[1] = length[1] - length[0]
        return CT.cpu().detach().numpy(), PET.cpu().detach().numpy(), DWI.cpu().detach().numpy(), T2.cpu().detach().numpy(), Label, length
