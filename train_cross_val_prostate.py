import torch
import monai
try:
    import intel_extension_for_pytorch as ipex
except:
    pass
from utils.NiftiDataset_cls_densenet_prostate import NifitDataSetProstate

import os
import numpy as np
import copy
from options.train_options import TrainOptions
import pandas as pd
from models import create_model
from utils.visualizer import Visualizer
from tqdm import tqdm
from monai.data import DataLoader
from torchmetrics import ConfusionMatrix, Accuracy
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassSpecificity, MulticlassRecall, MulticlassAUROC
from torchmetrics.classification import BinaryConfusionMatrix, BinaryF1Score, BinaryAccuracy, BinarySpecificity, BinaryRecall, BinaryAUROC

from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    ScaleIntensity,
    CenterSpatialCrop,
    RandFlip,
    Resize
)
from torch.utils.tensorboard import SummaryWriter
torch.multiprocessing.set_sharing_strategy('file_system')
# from torchvision.utils import save_image
from utils.utils import cal_metrics

import random


def validation(model, epoch, save_best_result=None, test=False, str_metric=None):
    visualizer = Visualizer(opt)
    total_steps = 0
    print("Validating model...")

    losses = []
    acc = []
    best_for_test = False
    if test:
        phase = 'Test'
    else:
        phase = 'Validation'

    visualizer.reset()
    total_steps += opt.batch_size

    model.validation()
    loss= model.get_current_losses(train=False)
    losses.append(loss)
    acc.append(model.get_current_acc())

    prediction = model.get_prediction_cur()
    target = model.get_target_cur()

    prediction = prediction.squeeze().cpu().detach()

    target = target.cpu().detach()
    
    pred_encoder = model.get_pred_encoder()
    pred_encoder = pred_encoder.squeeze().cpu().detach()
    metric_encoder = BinaryAccuracy()
    acc_pred_class_reduction = metric_encoder(pred_encoder, target)
    acc_pred = metric_encoder(pred_encoder, target)
    metric_encoder = BinaryF1Score()
    F1 = metric_encoder(pred_encoder, target)
    metric_encoder = BinaryAUROC()
    AUROC_encoder = metric_encoder(pred_encoder, target)

    metric_encoder = BinarySpecificity()
    Specificity = metric_encoder(pred_encoder, target)
    metric_encoder = BinaryRecall()
    recall = metric_encoder(pred_encoder, target)

    confmat_encoder = BinaryConfusionMatrix()
    CM = confmat_encoder(pred_encoder, target)
    PPV_encoder, NPV_encoder = cal_metrics(CM)
    print('['+phase+'] for encoders: ', '\t Accuracy:', acc_pred_class_reduction.item(),
        'AUC', AUROC_encoder.item(), 'Mean AUC:', AUROC_encoder.mean().item(),
        'F1:', F1.mean().item(), 'Spec.:', Specificity.mean().item(), 'Recall:', recall.mean().item()
        )

    if save_best_result is not None:
        save_results = 'Epoch {}, Accuracy {:.4f}, Per-class Acc {}, Mean Acc {:.4f}, AUC {}, Mean AUC {:.4f}, PPV {}, Mean PPV {:.4f}, NPV {}, Mean NPV {:.4f}, F1 {}, Mean F1 {:.4f}, Spec. {}, Mean Spec. {:.4f}, Recall {}, Mean Recall {:.4f}'.format(
            epoch, acc_pred_class_reduction, acc_pred, acc_pred.mean().item(), AUROC_encoder, AUROC_encoder.mean().item(), 
            PPV_encoder, PPV_encoder.mean(), NPV_encoder, NPV_encoder.mean(), F1, F1.mean().item(), Specificity, Specificity.mean().item(), recall, recall.mean().item())
        if test is False:
            str_metric = []
            if save_best_result['Enc_ROC'] < AUROC_encoder.mean().item():
                save_best_result['Enc_ROC'] = AUROC_encoder.mean().item()
                save_best_result['Enc_ROC_all'] = save_results
                best_for_test = True
                str_metric.append('ROC')
            if save_best_result['Enc_PPV'] < PPV_encoder.mean():
                save_best_result['Enc_PPV'] = PPV_encoder.mean()
                save_best_result['Enc_PPV_all'] = save_results
                best_for_test = True
                str_metric.append('PPV')
            if save_best_result['Enc_NPV'] < NPV_encoder.mean():
                save_best_result['Enc_NPV'] = NPV_encoder.mean()
                save_best_result['Enc_NPV_all'] = save_results
                best_for_test = True
                str_metric.append('NPV')
            if save_best_result['Enc_ACC'] < acc_pred_class_reduction.item():
                save_best_result['Enc_ACC'] = acc_pred_class_reduction.item()
                save_best_result['Enc_ACC_all'] = save_results
                best_for_test = True
                str_metric.append('ACC')
        else:
            assert str_metric is not None, 'str_metric parameter should not be None during testing phase'
            for metr in str_metric:
                save_best_result['Enc_' + metr + '_all'] = save_results
        
    if best_for_test:
        try:
            model.save_networks('best')
            print(f'save the best model at epoch {epoch}')
        except:
            print(f'save model failed at epoch {epoch}')
    return save_best_result, best_for_test, str_metric


if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()
    seed=opt.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    use_writer = False
    if use_writer:
        writer = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir, opt.name, 'tensorboard/logs'))
    else:
        writer = None
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    # load dataset
    trainTransforms1 = Compose([RandFlip(prob=0.5), ScaleIntensity(), EnsureChannelFirst(), Resize([200, 200, 200]), CenterSpatialCrop([192, 192, 192])])
    trainTransforms2 = Compose([RandFlip(prob=0.5), ScaleIntensity(), EnsureChannelFirst(), Resize([200, 200, 32]), CenterSpatialCrop([192, 192, 32])])
    train_set = NifitDataSetProstate(opt.root, transforms1=trainTransforms1, transforms2=trainTransforms2, shuffle_labels=False, phase='train', fold=10)

    print('length labeled train list:', len(train_set))
    drop_last = False
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, 
                              pin_memory=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=g)  # Here are then fed to the network with a defined batch size
    testTransforms1 = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize([200, 200, 200]), CenterSpatialCrop([192, 192, 192])])
    testTransforms2 = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize([200, 200, 32]), CenterSpatialCrop([192, 192, 32])])
    # TODO: use the path to the valdation set rather than the path to the training set
    val_set = NifitDataSetProstate(opt.root, transforms1=testTransforms1, transforms2=testTransforms2, shuffle_labels=False, phase='train', fold=10)
    print('length val list:', len(val_set))
    val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers,
                            pin_memory=True)
    # create model
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print("Creating model...")
    model = create_model(opt)
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.epoch_count)
    if opt.load_weight:
        if not opt.continue_train:
            model.load_pretrained_networks(opt.epoch_count)
        else:
            print('opt.continue_train is True, expecting resuming training from epoch {}, \
                  but the weights at that epoch are over-written by {}'.format(opt.epoch_count, opt.load_weight))

    visualizer = Visualizer(opt)
    total_steps = 0
    if opt.binary_class:
        num_classes = 1
    else:
        num_classes = 2

    model.train_loader, model.test_loader = train_loader, val_loader
    CT, PET, DWI, T2, Label, length = model.get_features([train_loader], model)
    # create hypergraph
    model.HGconstruct(CT, PET, DWI, T2)
    model.info(length)

    print("Training model...")
    save_best_result = {'Enc_ACC': 0, 'Enc_PPV': 0, 'Enc_NPV': 0, 'Enc_ROC': 0,
                        'Avg_ACC': 0, 'Avg_PPV': 0, 'Avg_NPV': 0, 'Avg_ROC': 0}
    save_test_result = copy.deepcopy(save_best_result)
    best_model = None
    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1), desc="Current epoch during training."):
        epoch_iter = 0
        losses = []
        acc = []
        prediction = None
        target = None

        visualizer.reset()
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        model.set_HGinput(Label)
        if epoch == 0 and (torch.cuda.is_available() or torch.xpu.is_available()):
            validation(model, epoch*len(train_loader), writer)
        model.optimize_parameters(model.train_loader, model.test_loader, epoch=epoch)

        loss = model.get_current_losses()
        losses.append(loss)
        acc.append(model.get_current_acc())
        if use_writer:
            writer.add_scalar("train_loss/CE", loss[0],  epoch )
            if opt.focal:
                writer.add_scalar("train_loss/focal", loss[1], epoch)
            writer.add_scalar("train_loss/acc", model.get_current_acc(), epoch)
        prediction = model.get_prediction_cur()
        target = model.get_target_cur()
        prediction = prediction.cpu().detach()

        # evaluation results
        target = target.cpu().detach()
        if opt.binary_class:
            acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro')
            f1_metric = MulticlassF1Score(num_classes=num_classes, average=None)
            auc_metric = MulticlassAUROC(num_classes=num_classes, average=None)
        else:
            acc_metric = BinaryAccuracy()
            f1_metric = BinaryF1Score()
            auc_metric = BinaryAUROC()
        acc = acc_metric(prediction, target)
        F1 = f1_metric(prediction, target)
        prediction = monai.utils.type_conversion.convert_to_tensor(prediction)
        AUROC = auc_metric(prediction, target)
        print('Train loss: ', np.sum(losses, 0) / len(losses), '\t\t Accuracy:', acc.item(), 'AUC', AUROC.item(), 'F1', F1.item())

        if opt.focal:
            visualizer.print_val_losses(epoch, np.sum(losses, 0) / len(losses), acc, 'Train')
        else:
            visualizer.print_val_losses(epoch, losses, acc, 'Train')

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                    (epoch, total_steps))
            model.save_networks('latest')

        if epoch % opt.save_epoch_freq == 0 and epoch != 0:
            print('saving the model at the end of epoch %d, iters %d' %
                (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d ' % (epoch, opt.niter + opt.niter_decay))
        model.update_learning_rate(model.get_current_acc())

        if (epoch)%10 == 0 and epoch != 0:
            CT, PET, DWI, T2,  Label, length = model.get_features([val_loader], model)
            model.HGconstruct(CT, PET, DWI, T2)
            model.info(length)
            model.set_HGinput(Label)
            save_best_result, best_for_test, str_metric = validation(model, epoch, save_best_result)
            df = pd.DataFrame(save_best_result, index=[0])
            df.T.to_csv('checkpoints/{}/Best_results_fold{}.csv'.format(opt.name, opt.fold), index=True, header=False)
            
        if use_writer:
            writer.close()
        
    