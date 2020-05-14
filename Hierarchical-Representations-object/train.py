from __future__ import  absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import ipdb
from tqdm import tqdm
from torch.utils import data as data_

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNN_mine
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.eval_tool import eval_detection_voc


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(individual, **kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False,
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNN_mine(individual)
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()

        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)

            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                ori_img_ = inverse_normalize(at.tonumpy(img[0]))

                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)

        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        best_path = None
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        individual.accuracy = best_map
