import numpy as np
import torch
try:
    import intel_extension_for_pytorch as ipex
except:
    pass

from tqdm import tqdm
from .base_model import BaseModel
from . import networks3D
from .densenet import *


class ProstateSingleModalModel(BaseModel):
    def name(self):
        return 'ProstateSingleModalModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
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
        
        self.model_names = ['Encoder', 'Classifier']
        dropout = 0.
        self.netEncoder = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=dropout), self.gpu_ids)
        
        self.num_graph_update = opt.num_graph_update
        self.weight_u = opt.weight_u # loss weight for unlabeled data
        self.netClassifier = torch.nn.Linear(1024, self.class_num)
        if len(self.gpu_ids) > 0:
            self.netClassifier.to(self.gpu_ids)
        if self.class_num == 1:
            self.criterionCE = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterionCE = torch.nn.CrossEntropyLoss()
        
        self.modality = 'CT'
        
        self.netDecoder_HGIB = self.netClassifier
        params = [{'params': self.netEncoder.parameters()},
                    {'params': self.netClassifier.parameters()},# 'lr': 10*opt.lr},
                    ]
        # initialize optimizers
        if self.isTrain:
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            if use_dawn:
                self.netEncoder.train()
                self.netClassifier.train()
                self.netEncoder, self.optimizer = ipex.optimize(self.netEncoder, optimizer=self.optimizer, dtype=torch.float32)
                self.netClassifier, self.optimizer = ipex.optimize(self.netClassifier, optimizer=self.optimizer, dtype=torch.float32)
                
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
        if self.modality == 'CT':
            self.data = self.CT
        elif self.modality == 'PET':
            self.data = self.PET
        elif self.modality == 'DWI':
            self.data = self.DWI
        else:
            self.data = self.T2
        self.target = input[4].to(self.device)

    def set_HGinput(self, input=None):
        self.embedding = self.embedding.to(self.device)
        if input is not None:
            self.target = input.to(self.device)

    def ExtractFeatures(self, phase='test'):
        if self.data.device.type != next(self.netEncoder.parameters()).device.type:
            self.netEncoder = self.netEncoder.to(self.device)
            self.netClassifier = self.netClassifier.to(self.device)
        if phase == 'test':
            with torch.no_grad():
                self.embedding, f1, f2, f3, f4 = self.netEncoder(self.data, return_feat_map=True)
        else:
            self.embedding, f1_MRI, f2_MRI, f3_MRI, f4_MRI = self.netEncoder(self.data, return_feat_map=True)

        return self.embedding, None, None, None

    def HGconstruct(self, embedding_CT, embedding_PET=None, embedding_DWI=None, embedding_T2=None):
        self.embedding = torch.Tensor(embedding_CT).to(self.device)

    def forward(self, phase='train', train_loader=None, test_loader=None, train_loader_u=None, epoch=None):
        if phase == 'train':
            prediction_all = []
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

                prediction = self.netClassifier(self.embedding)
                if self.class_num == 1:
                    prediction, self.target = prediction.squeeze(), self.target.float()
                prediction_all.append(prediction)
                targets_all.append(self.target)
                
                self.loss_cls = self.criterionCE(prediction, self.target)
                loss_x = torch.tensor(0.)
                if self.using_focalloss:
                    gamma = 0.5
                    alpha = 2
                    pt = torch.exp(-self.loss_cls)
                    self.loss_focal = (alpha * (1 - pt) ** gamma * self.loss_cls).mean()
                    self.loss = self.loss_cls + self.loss_focal
                else:
                    self.loss = self.loss_cls
                
                self.loss = self.loss + weight_u * loss_x
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                iteration += 1
                
                if i % 100 == 0:
                    print('Iteration {}, total loss for encoders {:.5f}, loss_x {:.5f}'.format(
                        i, self.loss.item(), loss_x.item()))
                    
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
                embedding, _, _, _, Label, length = self.get_features([train_loader, test_loader])
                # create hypergraph
                self.HGconstruct(embedding)
                self.info(length)
                self.set_HGinput(Label)
                num_graph_update = self.num_graph_update
                idx = torch.tensor(range(self.len_test)).to(self.device) + self.len_train
                prediction = self.netClassifier(torch.tensor(embedding).to(self.device))
                    
                prediction_encoder = prediction
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
        self.train()
        self.forward('train', train_loader, test_loader, train_loader_u=train_loader_u, epoch=epoch)

    def validation(self):
        self.netClassifier.eval()
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
        embedding = None
        Label = None
        length = [0, 0]
        for idx, loader in enumerate(loaders):
            for i, data in enumerate(tqdm(loader)):
                self.set_input(data)
                i_embedding, _, _, _ = self.ExtractFeatures(phase)
                if embedding is None:
                    embedding = i_embedding
                    Label = data[4]
                else:
                    embedding = torch.cat([embedding, i_embedding], 0)
                    Label = torch.cat([Label, data[4]], 0)
            length[idx] = embedding.size(0)
        length[1] = length[1] - length[0]
        return embedding.cpu().detach().numpy(), None, None, None, Label, length
