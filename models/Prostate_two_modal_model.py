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


class ProstateTwoModalModel(BaseModel):
    def name(self):
        return 'ProstateTwoModalModel'

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
        dropout_prob = 0.
        self.netEncoder = networks3D.init_net_update(DenseNet121(spatial_dims=3, in_channels=1, out_channels=1024, dropout_prob=dropout_prob), self.gpu_ids)
        
        self.num_graph_update = opt.num_graph_update
        self.weight_u = opt.weight_u # loss weight for unlabeled data
        self.netClassifier = torch.nn.Linear(1024, self.class_num)
        if len(self.gpu_ids) > 0:
            self.netClassifier.to(self.gpu_ids)
        if self.class_num == 1:
            self.criterionCE = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterionCE = torch.nn.CrossEntropyLoss()

        self.modality = 'DWI'
        if self.modality == 'DWI' or self.modality == 'T2':
            print('use modality of DWI and T2')
        else:
            print('use modality of CT and MRI')
        
        self.netDecoder_HGIB = self.netClassifier
        params = [{'params': self.netEncoder.parameters()},
                    {'params': self.netClassifier.parameters()},
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
        if self.modality == 'CT' or self.modality == 'PET':
            self.data = self.CT
            self.data2 = self.PET
        else:
            self.data = self.DWI
            self.data2 = self.T2
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
                self.embedding1, f1, f2, f3, f4 = self.netEncoder(self.data, return_feat_map=True)
                self.embedding2, f1, f2, f3, f4 = self.netEncoder(self.data2, return_feat_map=True)
        else:
            self.embedding1, f1_MRI, f2_MRI, f3_MRI, f4_MRI = self.netEncoder(self.data, return_feat_map=True)
            self.embedding2, f1_MRI, f2_MRI, f3_MRI, f4_MRI = self.netEncoder(self.data2, return_feat_map=True)
        return self.embedding1, self.embedding2, None, None

    def HGconstruct(self, embedding_CT, embedding_PET=None, embedding_DWI=None, embedding_T2=None):
        self.embedding = torch.Tensor((embedding_CT+embedding_PET)/2.0).to(self.device)

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
                embedding = (self.embedding1 + self.embedding2)/2.0
                prediction = self.netClassifier(embedding)
                if self.class_num == 1:
                    prediction, self.target = prediction.squeeze(), self.target.float()
                prediction_all.append(prediction)
                targets_all.append(self.target)
                
                self.loss_cls = self.criterionCE(prediction, self.target)
                
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
                embedding1, embedding2, _, _, Label, length = self.get_features([train_loader, test_loader])
                # create hypergraph
                self.HGconstruct(embedding1, embedding2)
                self.info(length)
                self.set_HGinput(Label)
                idx = torch.tensor(range(self.len_test)).to(self.device) + self.len_train
                prediction = self.netClassifier(torch.tensor((embedding1+embedding2)/2.0).to(self.device))
                    
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
        embedding1 = None
        embedding2 = None
        Label = None
        length = [0, 0]
        for idx, loader in enumerate(loaders):
            for i, data in enumerate(tqdm(loader)):
                self.set_input(data)
                i_embedding1, i_embedding2, _, _ = self.ExtractFeatures(phase)
                if embedding1 is None:
                    embedding1 = i_embedding1
                    embedding2 = i_embedding2
                    Label = data[4]
                else:
                    embedding1 = torch.cat([embedding1, i_embedding1], 0)
                    embedding2 = torch.cat([embedding2, i_embedding2], 0)
                    Label = torch.cat([Label, data[4]], 0)
            length[idx] = embedding1.size(0)
        length[1] = length[1] - length[0]
        return embedding1.cpu().detach().numpy(), embedding2.cpu().detach().numpy(), None, None, Label, length
