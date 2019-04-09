# encoding: utf-8
import os
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

sys.path.insert(0,os.path.abspath(os.path.dirname(__file__)+os.path.sep+'..'))


from config import opt
from datasets import data_manager
from datasets.data_loader import ImageData
from datasets.samplers import RandomIdentitySampler
from models.PCB import PCB
from trainers.evaluator import ResNetEvaluator
from trainers.trainer import PCBTrainer
from utils.loss import CrossEntropyLabelSmooth, TripletLoss, Margin
from utils.SelfTrainingTripletLoss import SelfTraining_TripletLoss
from utils.LiftedStructure import LiftedStructureLoss
from utils.DistWeightDevianceLoss import DistWeightBinDevianceLoss
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TestTransform, TrainTransform
from utils.reranking import re_ranking
from sklearn.cluster import DBSCAN



def extract_pcb_features(model,dataloader):
    ori = model.training
    model.eval()
    features=[]
    labels=[]
    for i,(imgs,pids,camids) in enumerate(dataloader):
        imgs=imgs.cuda()
        outputs = model(imgs)
        outputs = outputs.data.cpu()
        features.append(outputs)
        labels.append(pids)

    model.training = ori
    return torch.cat(features,0), torch.cat(labels,0)





def train(**kwargs):
    opt._parse(kwargs)
    #opt.lr=0.00002
    opt.model_name='PCB'
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
    tgt_dataset = data_manager.init_dataset(name=opt.tgt_dataset,mode=opt.mode)

    pin_memory = True if use_gpu else False

    summary_writer = SummaryWriter(osp.join(opt.save_dir, 'tensorboard_log'))

    trainloader = DataLoader(
        ImageData(dataset.train, TrainTransform(opt.datatype)),
        batch_size=opt.train_batch, num_workers=opt.workers,
        pin_memory=pin_memory, drop_last=True
    )

    tgt_trainloader = DataLoader(
        ImageData(tgt_dataset.train, TrainTransform(opt.datatype)),
        batch_size=opt.train_batch,num_workers=opt.workers,
        pin_memory=pin_memory,drop_last=True
    )

    tgt_queryloader = DataLoader(
        ImageData(tgt_dataset.query, TestTransform(opt.datatype)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    tgt_galleryloader = DataLoader(
        ImageData(tgt_dataset.gallery, TestTransform(opt.datatype)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )
    tgt_queryFliploader = DataLoader(
        ImageData(tgt_dataset.query, TestTransform(opt.datatype, True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    tgt_galleryFliploader = DataLoader(
        ImageData(tgt_dataset.gallery, TestTransform(opt.datatype, True)),
        batch_size=opt.test_batch, num_workers=opt.workers,
        pin_memory=pin_memory
    )

    print('initializing model ...')
    model = PCB(dataset.num_train_pids)


    optim_policy = model.get_optim_policy()

    if opt.pretrained_model:
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        # state_dict = {k: v for k, v in state_dict.items() \
        #        if not ('reduction' in k or 'softmax' in k)}
        try:
            model.load_state_dict(state_dict, False)
            print('load pretrained model ' + opt.pretrained_model)
        except:
            RuntimeError('please keep the same size with source dataset..')
    else:
        raise RuntimeError('please load a pre-trained model...')

    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))


    if use_gpu:
        model = nn.DataParallel(model).cuda()
    reid_evaluator = ResNetEvaluator(model)

    if opt.evaluate:
        print('transfer directly....... ')
        reid_evaluator.evaluate(tgt_queryloader, tgt_galleryloader,
                                tgt_queryFliploader, tgt_galleryFliploader, re_ranking=opt.re_ranking, savefig=opt.savefig)
        return


    #xent_criterion = CrossEntropyLabelSmooth(dataset.num_train_pids)


    embedding_criterion = SelfTraining_TripletLoss(margin=0.5,num_instances=4)

    # def criterion(triplet_y, softmax_y, labels):
    #     losses = [embedding_criterion(output, labels)[0] for output in triplet_y] + \
    #              [xent_criterion(output, labels) for output in softmax_y]
    #     loss = sum(losses)
    #     return loss


    def criterion(triplet_y, softmax_y, labels):
        #losses = [torch.sum(torch.stack([xent_criterion(logits, labels) for logits in softmax_y]))]
        losses = [torch.sum(torch.stack([embedding_criterion(output,labels) for output in triplet_y]))]
        loss = sum(losses)
        return loss


    # get optimizer
    if opt.optim == "sgd":
        optimizer = torch.optim.SGD(optim_policy, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam(optim_policy, lr=opt.lr, weight_decay=opt.weight_decay)

    start_epoch = opt.start_epoch
    # get trainer and evaluator
    reid_trainer = PCBTrainer(opt, model, optimizer, criterion, summary_writer)

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


    print('transfer directly.....')
    reid_evaluator.evaluate(tgt_queryloader, tgt_galleryloader,
                            tgt_queryFliploader, tgt_galleryFliploader, re_ranking=opt.re_ranking, savefig=opt.savefig)


    for iter_n in range(start_epoch,opt.max_epoch):
        if opt.lambda_value == 0:
            source_features = 0
        else:
            # get source datas' feature
            print('Iteration {}: Extracting Source Dataset Features...'.format(iter_n + 1))
            source_features, _ = extract_pcb_features(model, trainloader)

        # extract training images' features
        print('Iteration {}: Extracting Target Dataset Features...'.format(iter_n + 1))
        target_features, _ = extract_pcb_features(model, tgt_trainloader)
        # synchronization feature order with dataset.train

        # calculate distance and rerank result
        print('Calculating feature distances...')
        target_features = target_features.numpy()
        rerank_dist = re_ranking(
            source_features, target_features, lambda_value=opt.lambda_value)
        if iter_n == 0:
            # DBSCAN cluster
            tri_mat = np.triu(rerank_dist, 1)  # tri_mat.dim=2    取上三角
            tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
            tri_mat = np.sort(tri_mat, axis=None)
            top_num = np.round(opt.rho * tri_mat.size).astype(int)
            eps = tri_mat[:top_num].mean()  # DBSCAN聚类半径
            print('eps in cluster: {:.3f}'.format(eps))
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8)

        # select & cluster images as training set of this epochs
        print('Clustering and labeling...')
        labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(labels)) - 1
        print('Iteration {} have {} training ids'.format(iter_n + 1, num_ids))
        # generate new dataset
        new_dataset = []
        for (fname, _, _), label in zip(tgt_dataset.train, labels):
            if label == -1:
                continue
            # dont need to change codes in trainer.py _parsing_input function and sampler function after add 0
            new_dataset.append((fname, label, 0))
        print('Iteration {} have {} training images'.format(iter_n + 1, len(new_dataset)))

        selftrain_loader = DataLoader(
            ImageData(new_dataset, TrainTransform(opt.datatype)),
            sampler=RandomIdentitySampler(new_dataset, opt.num_instances),
            batch_size=opt.train_batch, num_workers=opt.workers,
            pin_memory=pin_memory, drop_last=True
        )

        # train model with new generated dataset
        trainer = PCBTrainer(opt, model, optimizer, criterion, summary_writer)
        reid_evaluator = ResNetEvaluator(model)
        # Start training
        for epoch in range(opt.selftrain_iterations):
            trainer.train(epoch, selftrain_loader)


        # skip if not save model
        if opt.eval_step > 0 and (iter_n + 1) % opt.eval_step == 0 or (iter_n + 1) == opt.max_epoch:
            #  just avoid out of memory during eval,and can't save the model
            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            save_checkpoint({'state_dict': state_dict, 'epoch': iter_n + 1},
                            is_best=0, save_dir=opt.save_dir,
                            filename='checkpoint_ep' + str(iter_n + 1) + '.pth.tar')


            if opt.mode == 'class':
                rank1 = test(model, tgt_queryloader)
            else:
                rank1 = reid_evaluator.evaluate(tgt_queryloader, tgt_galleryloader, tgt_queryFliploader, tgt_galleryFliploader)
            is_best = rank1 > best_rank1
            if is_best:
                best_rank1 = rank1
                best_epoch = iter_n + 1

            if use_gpu:
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            if is_best:
                save_checkpoint({'state_dict': state_dict, 'epoch': iter_n + 1},
                            is_best=is_best, save_dir=opt.save_dir,
                            filename='checkpoint_ep' + str(iter_n + 1) + '.pth.tar')

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
