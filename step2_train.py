#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yuxian Jiang
@Contact: yuxianjiang@sjtu.edu.cn
@Article: Morphology Prior Enhanced Teeth Segmentation for High Resolution Oral Scans
@Journal: Journal of Biomedical and Health Informatics (2025)
"""


import argparse
import os
import random
from tqdm import tqdm
import numpy as np
import time
import json
import pickle
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tensorboardX import SummaryWriter
from TeethDataset import TeethDataset, MultiDiceLoss
from DADSN import DADSN
from util import IOStream, get_lr, weights_init, bn_momentum_adjust
import pandas as pd


torch.autograd.set_detect_anomaly(True)

def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    os.system('cp step2_train.py outputs'+'/'+args.exp_name+'/'+'train.py')
    os.system('cp DADSN.py outputs' + '/' + args.exp_name + '/' + 'DADSN.py')
    os.system('cp util.py outputs' + '/' + args.exp_name + '/' + 'util.py')
    os.system('cp TeethDataset.py outputs' + '/' + args.exp_name + '/' + 'TeethDataset.py')
    os.system('cp '+ args.data_root +' outputs' + '/' + args.exp_name + '/' + 'split_names.json')

def set_seed(seed=1):
    print('Using random seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser('Train point cloud semantic segmentation')
    parser.add_argument('--exp_name', type=str, default='debug', metavar='N',
                        help='Name of the experiment') ###
    parser.add_argument('--data_root', type=str, default='data/Teeth3DS.json',
                        help='path to data') ###
    parser.add_argument('--npoint', type=int, default=16000,    ####
                        help='Point Number [default: 16000]')
    parser.add_argument('--num_classes', type=int, default=17, ###
                        help='number of classes')
    parser.add_argument('--batch_size', type=int, default=2,   ###
                        help='Batch Size during training') 
    parser.add_argument('--pretrain', type=str, default=None,  ###
                        help='pretrained model path') 
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=32, metavar='N',  ##
                        help='Num of nearest neighbors to use')
    parser.add_argument('--epoch', default=200, type=int,
                        help='Epoch to run [default: 200]')
    parser.add_argument('--learning_rate', default=0.001, type=float,  ##
                        help='Initial learning rate [default: 0.001]')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Adam or SGD [default: Adam]')
    parser.add_argument('--decay_rate', type=float, default=1e-4,
                        help='weight decay [default: 1e-4]')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        help='Learning rate scheduler [default: cosine decay]')
    parser.add_argument('--step_size', type=int, default=10,
                        help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7,
                        help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed [default: 1]')

    return parser.parse_args()


def train(args, io):
    '''LOG'''
    io.cprint(str(args))

    exp_dir = 'outputs/' + args.exp_name
    NUM_CLASSES = args.num_classes
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    class_weight = [1] * NUM_CLASSES
    class_weight = np.array(class_weight, dtype=np.float32) / np.sum(class_weight)
    '''MODEL LOADING'''
    classifier = DADSN(
        num_classes=NUM_CLASSES, k=args.k, emb_dims=args.emb_dims, dropout=args.dropout).cuda()

    multiclass_dice_loss = MultiDiceLoss(num_classes=args.num_classes, weight=class_weight).cuda()
    cross_entropy_loss = nn.CrossEntropyLoss(weight=torch.Tensor(class_weight).cuda())

    if args.optimizer.lower() == 'Adam'.lower():
        optimizer = Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            weight_decay=args.decay_rate
        )
        io.cprint('Using Adam optimizer')
    else:
        optimizer = SGD(classifier.parameters(),
                        lr=args.learning_rate, momentum=0.9,
                        weight_decay=args.decay_rate)
        io.cprint('Using SGD optimizer')

    if args.lr_scheduler == 'cosine':
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epoch, eta_min=1e-5)
        io.cprint('Using Cosine LR decay')
    else:
        lr_scheduler = StepLR(
            optimizer, step_size=args.step_size, gamma=args.lr_decay)
        io.cprint('Using Step LR decay')

    if args.pretrain is not None:
        checkpoint = torch.load(args.pretrain)
        classifier.load_state_dict(checkpoint['state_dict'])
        io.cprint(f'Use pretrain model from {args.pretrain}')
        start_epoch = 0
    else:
        try:
            checkpoint = torch.load(exp_dir + '/models/best_model.pth')
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['state_dict'])
            io.cprint('Use existing model')
        except:
            io.cprint('No existing model, starting training from scratch...')
            start_epoch = 0
            classifier = classifier.apply(weights_init)

    '''DATA LOADING'''
    io.cprint('Loading data')

    with open(args.data_root, 'r') as f:
        names = json.load(f)

    train_names = names["train_names"]
    val_names = names["val_names"]


    print("start loading training data ...")
    train_dataset = TeethDataset(file_paths=train_names, num_classes=NUM_CLASSES, num_points=NUM_POINT)
    print("start loading test data ...")
    test_dataset = TeethDataset(file_paths=val_names, num_classes=NUM_CLASSES, num_points=NUM_POINT)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1,
                                                  pin_memory=True, drop_last=False,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1,
                                                 pin_memory=True, drop_last=False)
    weights = torch.Tensor(class_weight).cuda()

    io.cprint("The number of training data is: %d" % len(train_dataset))
    io.cprint("The number of test data is: %d" % len(test_dataset))

    '''CREATE DIR'''
    start_datetime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    logs_dir = "outputs/{}/".format(args.exp_name)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    log_writer = SummaryWriter(os.path.join(logs_dir, 'train_logs'))

    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    min_loss = float('inf')
    train_losses = []
    eval_losses = []

    for epoch in range(start_epoch, args.epoch + 1):
        '''Train on chopped scenes'''
        io.cprint('\n\n**** Epoch %d (%d/%s) ****' %
              (global_epoch, epoch, args.epoch))

        # BN momentum decay
        momentum = MOMENTUM_ORIGINAL * \
            (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        io.cprint('BN momentum updated to: %.4f' % momentum)
        classifier = classifier.apply(
            lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        num_batches = len(trainDataLoader)
        total_correct = 0.
        total_seen = 0.
        dice_loss_sum = 0.
        cross_loss_sum = 0.
        loss_sum = 0.
        loss_bd_sum = 0.
        save_model = (epoch % 20 == 0)
        for i, data in tqdm(enumerate(trainDataLoader),
                            total=len(trainDataLoader), smoothing=0.9):
            points, target = data
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            
            seg_pred = classifier(points)

            dice_loss = multiclass_dice_loss(seg_pred.permute(0, 2, 1), target)
            cross_loss = cross_entropy_loss(seg_pred.permute(0, 2, 1), target)

            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            loss = dice_loss * 10 + cross_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            cross_loss_sum += cross_loss.item()
            dice_loss_sum += dice_loss.item()
            loss_sum += loss.item()

        mean_dice_loss = dice_loss_sum / num_batches
        mean_cross_loss = cross_loss_sum / num_batches
        mean_loss = loss_sum / num_batches
        mean_loss_bd = loss_bd_sum / num_batches
        mean_acc = total_correct / float(total_seen)
        lr = get_lr(optimizer)
        io.cprint('Training mean dice loss: %.4f' % (mean_dice_loss))
        io.cprint('Training mean cross loss: %.4f' % (mean_cross_loss))
        io.cprint('Training mean loss: %.4f' % (mean_loss))
        io.cprint('Training mean boundary loss: %.4f' % (mean_loss_bd))
        io.cprint('Training accuracy: %.4f' % (mean_acc))
        io.cprint('Learning rate: %.6f' % (lr))
        log_writer.add_scalar('train/loss', mean_loss, epoch)
        log_writer.add_scalar('train/loss_bd', mean_loss_bd, epoch)
        log_writer.add_scalar('train/dice_loss', mean_dice_loss)
        log_writer.add_scalar('train/cross_loss', mean_cross_loss)
        log_writer.add_scalar('train/accuracy', mean_acc, epoch)
        log_writer.add_scalar('train/lr', lr, epoch)
        train_losses.append(mean_loss)

        if save_model:
            model_name = 'model_epoch{}.pth'.format(epoch)
            savepath = os.path.join(logs_dir, 'models', model_name)
            io.cprint('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'state_dict': classifier.state_dict(),
                'opt': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            torch.save(state, savepath)

        io.cprint('Evaluate on Epoch %d' % epoch)
        with torch.no_grad():
            num_batches = len(testDataLoader)
            classifier = classifier.eval()
            total_correct = 0.
            total_seen = 0.
            dice_loss_sum = 0.
            cross_loss_sum = 0.
            loss_sum = 0.

            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_class = [0 for _ in range(NUM_CLASSES)]

            for i, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points, target = data
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)
                seg_pred = classifier(points)

                dice_loss = multiclass_dice_loss(seg_pred.permute(0, 2, 1), target)
                cross_loss = cross_entropy_loss(seg_pred.permute(0, 2, 1), target)

                batch_label = target.cpu().data.numpy()
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                loss = dice_loss * 10 + cross_loss
                dice_loss_sum += dice_loss.item()
                cross_loss_sum += cross_loss.item()
                loss_sum += loss.item()

                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_class[l] += np.sum(((pred_val == l) | (batch_label == l)))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_class, dtype=np.float32) + 1e-6))
            eval_loss = loss_sum / float(num_batches)
            eval_losses.append(eval_loss)
            io.cprint('Eval mean dice loss: %.4f' % (dice_loss_sum / num_batches))
            io.cprint('Eval mean cross loss: %.4f' % (cross_loss_sum / num_batches))
            io.cprint('Eval mean loss: %.4f' % (eval_loss))
            io.cprint('Eval accuracy: %.4f' % (total_correct / float(total_seen)))
            io.cprint('Eval mIoU: %.4f' % mIoU)
            io.cprint('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float32) + 1e-6))))
            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    str(l), class_weight[l],
                    total_correct_class[l] / float(total_iou_class[l]))

            if eval_loss < min_loss:
                min_loss = eval_loss
                model_name = 'best_model.pth'
                savepath = os.path.join(logs_dir, 'models', model_name)
                io.cprint('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'state_dict': classifier.state_dict(),
                    'opt': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict()
                }
                torch.save(state, savepath)

        global_epoch += 1
        lr_scheduler.step()
        torch.cuda.empty_cache()
    df = pd.DataFrame({'train_loss': train_losses, 'eval_loss': eval_losses})
    df.to_csv(os.path.join(logs_dir, 'loss_curve.csv'), index=False)



if __name__ == '__main__':
    args = parse_args()
    _init_()
    io = IOStream('outputs/' + args.exp_name + '/run.log')
    
    set_seed(args.seed)
    cudnn.benchmark = True

    train(args, io)

