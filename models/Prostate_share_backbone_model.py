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


class ProstateShareBackboneModel(BaseModel):
    def name(self):
        return 'ProstateShareBackboneModel'

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
        self.target = input[4].to(self.device)

    def set_HGinput(self, input=None):
        self.embedding = self.embedding.to(self.device)
        if input is not None:
            self.target = input.to(self.device)

    def ExtractFeatures(self, phase='test'):
        if self.CT.device.type != next(self.netEncoder.parameters()).device.type:
            self.netEncoder = self.netEncoder.to(self.device)
            self.netClassifier = self.netClassifier.to(self.device)
            
        if phase == 'test':
            with torch.no_grad():
                self.embedding_CT, f1, f2, f3, f4 = self.netEncoder(self.CT, return_feat_map=True)
                self.embedding_PET, f1, f2, f3, f4 = self.netEncoder(self.PET, return_feat_map=True)
                self.embedding_DWI, f1, f2, f3, f4 = self.netEncoder(self.DWI, return_feat_map=True)
                self.embedding_T2, f1, f2, f3, f4 = self.netEncoder(self.T2, return_feat_map=True)
        else:
            self.embedding_CT, f1_MRI, f2_MRI, f3_MRI, f4_MRI = self.netEncoder(self.CT, return_feat_map=True)
            self.embedding_PET, f1_PET, f2_PET, f3_PET, f4_PET = self.netEncoder(self.PET, return_feat_map=True)
            self.embedding_DWI, f1, f2, f3, f4 = self.netEncoder(self.DWI, return_feat_map=True)
            self.embedding_T2, f1, f2, f3, f4 = self.netEncoder(self.T2, return_feat_map=True)
        return self.embedding_CT, self.embedding_PET, self.embedding_DWI, self.embedding_T2

    def HGconstruct(self, embedding_CT, embedding_PET, embedding_DWI, embedding_T2):
        self.embedding = torch.Tensor((embedding_CT + embedding_PET + embedding_DWI + embedding_T2)/4.0).to(self.device)

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
                embedding = (self.embedding_CT+self.embedding_PET+self.embedding_DWI+self.embedding_T2)/4.0
                prediction = self.netClassifier(embedding)
                if self.class_num == 1:
                    prediction, self.target = prediction.squeeze(), self.target.float()
                self.loss_cls = self.criterionCE(prediction, self.target)
                
                prediction_all.append(prediction)
                targets_all.append(self.target)
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
                num_graph_update = self.num_graph_update
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
