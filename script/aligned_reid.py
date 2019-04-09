# encoding: utf-8
import os
import sys
from os import path as osp
from pprint import pprint
import time
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from collections import defaultdict

sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..'))


from config import opt
from datasets import data_manager
from datasets.data_loader import ImageData
from datasets.samplers import RandomIdentitySampler
from models.AlignedReid import AlignedResNet50
from trainers.evaluator import AlignedEvaluator
from trainers.trainer import AlignedTrainer
from utils.loss import CrossEntropyLabelSmooth
from utils.AlignedTripletLoss import TripletLossAlignedReID, DeepSupervision, low_memory_local_dist
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TestTransform, TrainTransform
from utils.meters import AverageMeter



def train(**kwargs):
    opt._parse(kwargs)
    #opt.lr=0.00002
    opt.model_name='AlignedReid'
    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)
    os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')

    print('initializing dataset {}'.format(opt.dataset))
    dataset = data_manager.init_dataset(name=opt.dataset, mode=opt.mode)

    pin_memory = True if use_gpu else False

    summary_writer = SummaryWriter(osp.join(opt.save_dir, 'tensorboard_log'))

    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform(opt.datatype)),
        sampler=RandomIdentitySampler(dataset.train, opt.num_instances),
        batch_size=opt.train_batch, num_workers=opt.workers,
        pin_memory=pin_memory, drop_last=True
    )


    queryloader = DataLoader(
        ImageData(dataset.query, TestTransform(opt.datatype)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    galleryloader = DataLoader(
        ImageData(dataset.gallery, TestTransform(opt.datatype)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    queryFliploader = queryloader
    galleryFliploader = galleryloader
    # queryFliploader = DataLoader(
    #     ImageData(dataset.query, TestTransform(opt.datatype, True)),
    #     batch_size=opt.test_batch, num_workers=opt.workers,
    #     pin_memory=pin_memory
    # )
    #
    # galleryFliploader = DataLoader(
    #     ImageData(dataset.gallery, TestTransform(opt.datatype, True)),
    #     batch_size=opt.test_batch, num_workers=opt.workers,
    #     pin_memory=pin_memory
    # )

    print('initializing model ...')
    model = AlignedResNet50(num_classes=dataset.num_train_pids, loss={'softmax', 'metric'},
                              aligned=True, use_gpu=use_gpu)

    optim_policy = model.get_optim_policy()

    if opt.pretrained_model:
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        # state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

    if use_gpu:
        model = nn.DataParallel(model).cuda()
    reid_evaluator = AlignedEvaluator(model)

    if opt.evaluate:
        #rank1 = test(model, queryloader, galleryloader, use_gpu)
        reid_evaluator.evaluate(queryloader, galleryloader,
                                queryFliploader, galleryFliploader, re_ranking=opt.re_ranking, savefig=opt.savefig, test_distance='global')
        return

    # xent_criterion = nn.CrossEntropyLoss()
    xent_criterion = CrossEntropyLabelSmooth(dataset.num_train_pids,use_gpu=use_gpu)

    embedding_criterion = TripletLossAlignedReID(margin=opt.margin)

    # def criterion(triplet_y, softmax_y, labels):
    #     losses = [embedding_criterion(output, labels)[0] for output in triplet_y] + \
    #              [xent_criterion(output, labels) for output in softmax_y]
    #     loss = sum(losses)
    #     return loss


    def criterion(outputs, features, local_features, labels):
        if opt.htri_only:
            if isinstance(features, tuple):
                global_loss, local_loss = DeepSupervision(embedding_criterion, features, labels)
            else:
                global_loss, local_loss = embedding_criterion(features, labels, local_features)
        else:
            if isinstance(outputs, tuple):
                xent_loss = DeepSupervision(xent_criterion, outputs, labels)
            else:
                xent_loss = xent_criterion(outputs, labels)

            if isinstance(features, tuple):
                global_loss, local_loss = DeepSupervision(embedding_criterion, features, labels)
            else:
                global_loss, local_loss = embedding_criterion(features, labels, local_features)
        loss = xent_loss + global_loss + local_loss
        return loss



    # get optimizer
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(optim_policy, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(optim_policy, lr=opt.lr, weight_decay=opt.weight_decay)

    start_epoch = opt.start_epoch
    # get trainer and evaluator
    reid_trainer = AlignedTrainer(opt, model, optimizer, criterion, summary_writer)

    def adjust_lr(optimizer, ep):
        if ep < 50:
            lr = opt.lr * (ep // 5 + 1)
        elif ep < 200:
            lr = opt.lr*10
        elif ep < 300:
            lr = opt.lr
        else:
            lr = opt.lr*0.1
        for p in optimizer.param_groups:
            p['lr'] = lr

    # start training
    best_rank1 = opt.best_rank
    best_epoch = 0
    print('start train......')
    for epoch in range(start_epoch, opt.max_epoch):
        if opt.adjust_lr:
            adjust_lr(optimizer, epoch + 1)

        reid_trainer.train(epoch, trainloader)

        # skip if not save model
        if opt.eval_step > 0 and (epoch + 1) % opt.eval_step == 0 or (epoch + 1) == opt.max_epoch:

            #  just avoid out of memory during eval,and can't save the model
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({'state_dict': state_dict, 'epoch': epoch + 1},
                            is_best=0, save_dir=opt.save_dir,
                            filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

            if opt.mode == 'class':
                rank1 = test(model, queryloader)
            else:
                rank1 = reid_evaluator.evaluate(queryloader, galleryloader, queryFliploader, galleryFliploader)
                #rank1 = test(model, queryloader, galleryloader, use_gpu)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = epoch + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            if is_best:
                save_checkpoint({'state_dict': state_dict, 'epoch': epoch + 1},
                            is_best=is_best, save_dir=opt.save_dir,
                            filename='checkpoint_ep' + str(epoch + 1) + '.pth.tar')

    print('Best rank-1 {:.1%}, achived at epoch {}'.format(best_rank1, best_epoch))


def test(model, queryloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target, _ in queryloader:
            output = model(data).cpu()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    rank1 = 100. * correct / len(queryloader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(queryloader.dataset), rank1))
    return rank1



if __name__ == '__main__':
    import fire

    fire.Fire()
