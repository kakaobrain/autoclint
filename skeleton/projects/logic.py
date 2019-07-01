# -*- coding: utf-8 -*-
from __future__ import absolute_import
import random

import tensorflow as tf
import torchvision as tv
import torch
import numpy as np

from .api import Model
from .others import *
import skeleton

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


LOGGER = get_logger(__name__)


class LogicModel(Model):
    def __init__(self, metadata, session=None):
        super(LogicModel, self).__init__(metadata)
        LOGGER.info('--------- Model.metadata ----------')
        LOGGER.info('path: %s', self.metadata.get_dataset_name())
        LOGGER.info('shape:  %s', self.metadata.get_tensor_size(0))
        LOGGER.info('size: %s', self.metadata.size())
        LOGGER.info('num_class:  %s', self.metadata.get_output_size())

        test_metadata_filename = self.metadata.get_dataset_name().replace('train', 'test') + '/metadata.textproto'
        self.num_test = [int(line.split(':')[1]) for line in open(test_metadata_filename, 'r').readlines() if 'sample_count' in line][0]
        LOGGER.info('num_test:  %d', self.num_test)

        self.timers = {
            'train': skeleton.utils.Timer(),
            'test': skeleton.utils.Timer()
        }
        self.info = {
            'dataset': {
                'path': self.metadata.get_dataset_name(),
                'shape': self.metadata.get_tensor_size(0),
                'size': self.metadata.size(),
                'num_class': self.metadata.get_output_size()
            },
            'loop': {
                'epoch': 0,
                'test': 0,
                'best_score': 0.0
            },
            'condition': {
                'first': {
                    'train': True,
                    'valid': True,
                    'test': True
                }
            },
            'terminate': False
        }

        # TODO: adaptive logic for hyper parameter
        self.hyper_params = {
            'dataset': {
                'train_info_sample': 256,
                'cv_valid_ratio': 0.1,
                'max_valid_count': 128,

                'max_size': 64,
                'base': 16,  # input size should be multipliers of 16

                'batch_size': 32,
                'steps_per_epoch': 20,
                'max_epoch': 100,  # initial value
                'batch_size_test': 256,
            },
            'checkpoints': {
                'keep': 30
            },
            'conditions': {
                'score_type': 'auc',
                'early_epoch': 1,
                'skip_valid_score_threshold': 0.90,  # if bigger then 1.0 is not use
                'skip_valid_after_test': min(10, max(3, int(self.info['dataset']['size'] // 1000))),
                'test_after_at_least_seconds': 1,
                'test_after_at_least_seconds_max': 90,
                'test_after_at_least_seconds_step': 2,

                'threshold_valid_score_diff': 0.001,
                'max_inner_loop_ratio': 0.2,
                'min_lr': 1e-6,
                'use_fast_auto_aug': True
            }
        }
        self.checkpoints = []
        LOGGER.info('[init] build')

        self.build()
        LOGGER.info('[init] session')

        self.dataloaders = {
            'train': None,
            'valid': None,
            'test': None
        }
        LOGGER.info('[init] done')

    def __repr__(self):
        return '\n---------[{0}]---------\ninfo:{1}\nparams:{2}\n---------- ---------'.format(
            self.__class__.__name__,
            self.info, self.hyper_params
        )

    def build(self):
        raise NotImplementedError

    def update_model(self):
        # call after to scan train sample
        pass

    def epoch_train(self, epoch, train):
        raise NotImplementedError

    def epoch_valid(self, epoch, valid):
        raise NotImplementedError

    def skip_valid(self, epoch):
        raise NotImplementedError

    def prediction(self, dataloader):
        raise NotImplementedError

    def adapt(self):
        raise NotImplementedError

    def is_multiclass(self):
        return self.info['dataset']['train']['is_multiclass']

    def build_or_get_train_dataloader(self, dataset):
        if not self.info['condition']['first']['train']:
            return self.build_or_get_dataloader('train')

        num_images = self.info['dataset']['size']

        # split train/valid
        num_valids = int(min(num_images * self.hyper_params['dataset']['cv_valid_ratio'], self.hyper_params['dataset']['max_valid_count']))
        num_trains = num_images - num_valids
        LOGGER.info('[cv_fold] num_trains:%d num_valids:%d', num_trains, num_valids)

        LOGGER.info('[%s] scan before', 'sample')
        num_samples = self.hyper_params['dataset']['train_info_sample']
        sample = dataset.take(num_samples).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        train = skeleton.data.TFDataset(self.session, sample, num_samples)
        self.info['dataset']['train'] = train.scan(samples=num_samples)
        del train
        del sample
        LOGGER.info('[%s] scan after', 'sample')

        # input_shape = [min(s, self.hyper_params['dataset']['max_size']) for s in self.info['dataset']['shape']]
        height, width, channels = self.info['dataset']['train']['example']['shape'][1:]
        aspect_ratio = width / height

        # fit image area to 64x64
        if aspect_ratio > 2 or 1. / aspect_ratio > 2:
            self.hyper_params['dataset']['max_size'] *= 2
        size = [min(s, self.hyper_params['dataset']['max_size']) for s in [height, width]]

        # keep aspect ratio
        if aspect_ratio > 1:
            size[0] = size[1] / aspect_ratio
        else:
            size[1] = size[0] * aspect_ratio

        # too small image use original image
        if width <= 32 and height <= 32:
            input_shape = [height, width, channels]
        else:
            size = list(map(lambda x: int(x / self.hyper_params['dataset']['base'] + 0.8) * self.hyper_params['dataset']['base'], size))
            input_shape = size + [channels]
        LOGGER.info('[input_shape] origin:%s aspect_ratio:%f target:%s', [height, width, channels], aspect_ratio, input_shape)

        self.hyper_params['dataset']['input'] = input_shape

        num_class = self.info['dataset']['num_class']
        batch_size = self.hyper_params['dataset']['batch_size']
        if num_class > batch_size / 2:
            self.hyper_params['dataset']['batch_size'] = batch_size * 2
        batch_size = self.hyper_params['dataset']['batch_size']

        enough_image = num_images > 5000
        if not enough_image:
            preprocessor1 = get_tf_resize(input_shape[0], input_shape[1])
            preprocessor2 = get_tf_to_tensor(is_random_flip=False)
            preprocessor = lambda *tensor: preprocessor2(preprocessor1(*tensor))

            batchsize = 128
            tf_dataset = dataset.apply(
                tf.data.experimental.map_and_batch(
                    map_func=lambda *x: (preprocessor(x[0]), x[1]),
                    batch_size=batchsize,
                    drop_remainder=False,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
            ).prefetch(buffer_size=3)
            dataset = skeleton.data.TFDataset(self.session, tf_dataset, num_images)

            LOGGER.info('[%s] scan before', 'train')
            self.info['dataset']['train'], tensors = dataset.scan(
                with_tensors=True, is_batch=True,
                device=self.device, half=self.is_half
            )
            tensors = [torch.cat(t, dim=0) for t in zip(*tensors)]
            LOGGER.info('[%s] scan after', 'train')

            del tf_dataset
            del dataset

            dataset = torch.utils.data.TensorDataset(*tensors)
            index = list(range(num_images))

            # StratifiedShuffleSplit
            labels = LabelEncoder().fit_transform([''.join(str(l)) for l in tensors[1]])
            unique, counts = np.unique(np.array(labels), return_counts=True)
            num_single = sum(counts == 1)
            if num_single > 0:
                target = unique[np.argmax(counts)]
                single = unique[counts == 1]
                for idx, l in enumerate(labels):
                    if l not in single:
                        continue
                    labels[idx] = target
            LOGGER.info('[StratifiedShuffleSplit] unique label counts: %d, single: %d', len(counts), num_single)

            sss = StratifiedShuffleSplit(n_splits=1, test_size=num_valids, random_state=None)
            sss = sss.split(index, labels)
            train_idx, valid_idx = next(sss)

            # random.shuffle(index)
            # train_idx = index[num_valids:]
            # valid_idx = index[:num_valids]

            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            valid_dataset = torch.utils.data.Subset(dataset, valid_idx)

            # Stratified Shuffle
            labels = [labels[idx] for idx in train_idx]
            sampler = skeleton.data.StratifiedSampler(labels)
            # sampler = skeleton.data.InfiniteSampler(train_dataset, shuffle=True)

            transform = tv.transforms.Compose([
                skeleton.data.RandomFlip(p=0.5),
            ])
            train_dataset = skeleton.data.TransformDataset(train_dataset, transform, index=0)

            transform = tv.transforms.Compose([
            ])
            valid_dataset = skeleton.data.TransformDataset(valid_dataset, transform, index=0)

            self.dataloaders['train'] = skeleton.data.FixedSizeDataLoader(
                train_dataset,
                steps=self.hyper_params['dataset']['steps_per_epoch'],
                batch_size=self.hyper_params['dataset']['batch_size'],
                shuffle=True, drop_last=True, num_workers=0, pin_memory=False,
                sampler=sampler
            )
            self.dataloaders['valid'] = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=self.hyper_params['dataset']['batch_size_test'],
                shuffle=False, drop_last=False, num_workers=0, pin_memory=False
            )
            self.info['condition']['first']['valid'] = False

            self.datasets = {
                'train': None,
                'valid': None,
                'num_trains': num_trains,
                'num_valids': num_valids
            }
        else:
            # dataset = dataset.shuffle(buffer_size=num_valids * 4, reshuffle_each_iteration=False)
            train = dataset.skip(num_valids)
            valid = dataset.take(num_valids)
            self.datasets = {
                'train': train,
                'valid': valid,
                'num_trains': num_trains,
                'num_valids': num_valids
            }
        return self.build_or_get_dataloader('train', self.datasets['train'], num_trains)

    def build_or_get_dataloader(self, mode, dataset=None, num_items=0):
        if mode in self.dataloaders and self.dataloaders[mode] is not None:
            return self.dataloaders[mode]

        LOGGER.debug('[dataloader] %s build start', mode)
        if mode == 'train':
            batch_size = self.hyper_params['dataset']['batch_size']
            input_shape = self.hyper_params['dataset']['input']

            preprocessor1 = get_tf_resize(input_shape[0], input_shape[1])
            preprocessor2 = get_tf_to_tensor(is_random_flip=True)
            preprocessor = lambda *tensor: preprocessor2(preprocessor1(*tensor))

            if num_items < 10000:
                dataset = dataset.map(
                    lambda *x: (preprocessor1(x[0]), x[1]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                )

                dataset = dataset.cache()

                dataset = dataset.apply(
                    tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size * 4)
                )

                dataset = dataset.map(
                    lambda *x: (preprocessor2(x[0]), x[1]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                ).prefetch(buffer_size=batch_size * 3)

            else:
                dataset = dataset.apply(
                    tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size * 4)
                )

                dataset = dataset.map(
                    lambda *x: (preprocessor(x[0]), x[1]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                ).prefetch(buffer_size=batch_size * 3)

            # dataset = dataset.apply(
            #     tf.data.experimental.map_and_batch(
            #         map_func=lambda *x: (preprocessor(x[0]), x[1]),
            #         batch_size=batch_size,
            #         drop_remainder=False,
            #         num_parallel_calls=tf.data.experimental.AUTOTUNE
            #     )
            # ).prefetch(tf.data.experimental.AUTOTUNE)
            # batch_size = None

            dataset = skeleton.data.TFDataset(self.session, dataset, num_items)

            transform = tv.transforms.Compose([
            ])
            dataset = skeleton.data.TransformDataset(dataset, transform, index=0)

            self.dataloaders['train'] = skeleton.data.FixedSizeDataLoader(
                dataset,
                steps=self.hyper_params['dataset']['steps_per_epoch'],
                batch_size=batch_size,
                shuffle=False, drop_last=True, num_workers=0, pin_memory=False
            )
        elif mode in ['valid', 'test']:
            batch_size = self.hyper_params['dataset']['batch_size_test']
            input_shape = self.hyper_params['dataset']['input']
            preprocessor1 = get_tf_resize(input_shape[0], input_shape[1])
            preprocessor2 = get_tf_to_tensor(is_random_flip=False)
            preprocessor = lambda *tensor: preprocessor2(preprocessor1(*tensor))

            # batch_size = 500
            tf_dataset = dataset.apply(
                tf.data.experimental.map_and_batch(
                    map_func=lambda *x: (preprocessor(x[0]), x[1]),
                    batch_size=batch_size,
                    drop_remainder=False,
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
            ).prefetch(buffer_size=3)

            dataset = skeleton.data.TFDataset(self.session, tf_dataset, num_items)

            LOGGER.info('[%s] scan before', mode)
            self.info['dataset'][mode], tensors = dataset.scan(
                with_tensors=True, is_batch=True,
                device=self.device, half=self.is_half
            )
            tensors = [torch.cat(t, dim=0) for t in zip(*tensors)]
            LOGGER.info('[%s] scan after', mode)

            del tf_dataset
            del dataset
            dataset = skeleton.data.prefetch_dataset(tensors)

            transform = tv.transforms.Compose([
            ])
            dataset = skeleton.data.TransformDataset(dataset, transform, index=0)
            self.dataloaders[mode] = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.hyper_params['dataset']['batch_size_test'],
                shuffle=False, drop_last=False, num_workers=0, pin_memory=False
            )
            self.info['condition']['first'][mode] = False
        # elif mode == 'test':
        #     batch_size = self.hyper_params['dataset']['batch_size_test']
        #     input_shape = self.hyper_params['dataset']['input']
        #     preprocessor1 = get_tf_resize(input_shape[0], input_shape[1])
        #     preprocessor2 = get_tf_to_tensor(is_random_flip=False)
        #     preprocessor = lambda *tensor: preprocessor2(preprocessor1(*tensor))
        #
        #     tf_dataset = dataset.apply(
        #         tf.data.experimental.map_and_batch(
        #             map_func=lambda *x: (preprocessor(x[0]), x[1]),
        #             batch_size=batch_size,
        #             drop_remainder=False,
        #             num_parallel_calls=tf.data.experimental.AUTOTUNE
        #         )
        #     ).cache().repeat().prefetch(buffer_size=2)
        #
        #     steps = num_items // batch_size + (1 if num_items % batch_size > 0 else 0)
        #     dataset = skeleton.data.TFDataset(self.session, tf_dataset, steps)
        #
        #     self.dataloaders[mode] = torch.utils.data.DataLoader(
        #         dataset,
        #         batch_size=1, #self.hyper_params['dataset']['batch_size_test'],
        #         shuffle=False, drop_last=False, num_workers=0, pin_memory=True
        #     )
        #     self.info['condition']['first'][mode] = False

        LOGGER.debug('[dataloader] %s build end', mode)
        return self.dataloaders[mode]

    def update_condition(self, metrics=None):
        self.info['condition']['first']['train'] = False
        self.info['loop']['epoch'] += 1

        metrics.update({'epoch': self.info['loop']['epoch']})
        self.checkpoints.append(metrics)

        indices = np.argsort(np.array([v['valid']['score'] for v in self.checkpoints] if len(self.checkpoints) > 0 else [0]))
        indices = sorted(indices[::-1][:self.hyper_params['checkpoints']['keep']])
        self.checkpoints = [self.checkpoints[i] for i in indices]

    def break_train_loop_condition(self, remaining_time_budget=None, inner_epoch=1):
        consume = inner_epoch * self.timers['train'].step_time

        best_idx = np.argmax(np.array([c['valid']['score'] for c in self.checkpoints]))
        best_epoch = self.checkpoints[best_idx]['epoch']
        best_loss = self.checkpoints[best_idx]['valid']['loss']
        best_score = self.checkpoints[best_idx]['valid']['score']
        lr = self.optimizer.get_learning_rate()
        LOGGER.debug('[CONDITION] best (epoch:%04d loss:%.2f score:%.2f) lr:%.8f time delta:%.2f',
                     best_epoch, best_loss, best_score, lr, consume)

        if self.info['loop']['epoch'] <= self.hyper_params['conditions']['early_epoch']:
            LOGGER.info('[BREAK] early %d epoch', self.hyper_params['conditions']['early_epoch'])
            return True

        if best_score > 0.995:
            LOGGER.info('[BREAK] achieve best score %f', best_score)
            return True

        if consume > self.hyper_params['conditions']['test_after_at_least_seconds'] and \
            self.checkpoints[best_idx]['epoch'] > self.info['loop']['epoch'] - inner_epoch and \
            best_score > self.info['loop']['best_score'] * 1.001:
            # increase hyper param
            self.hyper_params['conditions']['test_after_at_least_seconds'] = min(
                self.hyper_params['conditions']['test_after_at_least_seconds_max'],
                self.hyper_params['conditions']['test_after_at_least_seconds'] + self.hyper_params['conditions']['test_after_at_least_seconds_step']
            )

            self.info['loop']['best_score'] = best_score
            LOGGER.info('[BREAK] found best model (score:%f)', best_score)
            return True

        if lr < self.hyper_params['conditions']['min_lr']:
            LOGGER.info('[BREAK] too small lr (lr:%f < %f)', lr, self.hyper_params['conditions']['min_lr'])
            return True

        early_term_budget = 3 * 60
        expected_more_time = (self.timers['test'].step_time + (self.timers['train'].step_time * 2)) * 1.5
        if remaining_time_budget is not None and \
            remaining_time_budget - early_term_budget < expected_more_time:
            LOGGER.info('[BREAK] not enough time to train (remain:%f need:%f)', remaining_time_budget, expected_more_time)
            return True

        if self.info['loop']['epoch'] >= 20 and \
            inner_epoch > self.hyper_params['dataset']['max_epoch'] * self.hyper_params['conditions']['max_inner_loop_ratio']:
            LOGGER.info('[BREAK] cannot found best model in too many epoch')
            return True

        return False

    def terminate_train_loop_condition(self, remaining_time_budget=None, inner_epoch=0):
        best_idx = np.argmax(np.array([c['valid']['score'] for c in self.checkpoints]))
        best_score = self.checkpoints[best_idx]['valid']['score']
        if best_score > 0.995:
            LOGGER.info('[TERMINATE] achieve best score %f', best_score)
            self.info['terminate'] = True
            self.done_training = True
            return True

        early_term_budget = 3 * 60
        expected_more_time = (self.timers['test'].step_time + (self.timers['train'].step_time * 2)) * 1.5
        if remaining_time_budget is not None and \
            remaining_time_budget - early_term_budget < expected_more_time:
            LOGGER.info('[TERMINATE] not enough time to train (remain:%f need:%f)', remaining_time_budget,
                        expected_more_time)
            self.info['terminate'] = True
            self.done_training = True
            return True

        scores = [c['valid']['score'] for c in self.checkpoints]
        diff = (max(scores) - min(scores)) * (1 - max(scores))
        threshold = self.hyper_params['conditions']['threshold_valid_score_diff']
        if 1e-8 < diff and diff < threshold and \
            self.info['loop']['epoch'] >= 20:
            LOGGER.info('[TERMINATE] too small score change (diff:%f < %f)', diff, threshold)
            done = True if self.info['terminate'] else False
            self.info['terminate'] = True
            self.done_training = done
            return True

        if self.optimizer.get_learning_rate() < self.hyper_params['conditions']['min_lr']:
            LOGGER.info('[TERMINATE] lr=%f', self.optimizer.get_learning_rate())
            done = True if self.info['terminate'] else False
            self.info['terminate'] = True
            self.done_training = done
            return True

        if self.info['loop']['epoch'] >= 20 and \
            inner_epoch > self.hyper_params['dataset']['max_epoch'] * self.hyper_params['conditions']['max_inner_loop_ratio']:
            LOGGER.info('[TERMINATE] cannot found best model in too many epoch')
            done = True if self.info['terminate'] else False
            self.info['terminate'] = True
            self.done_training = done
            return True

        return False

    def get_total_time(self):
        return sum([self.timers[key].total_time for key in self.timers.keys()])

    def train(self, dataset, remaining_time_budget=None):
        LOGGER.debug(self)
        LOGGER.debug('[train] [%02d] budget:%f', self.info['loop']['epoch'], remaining_time_budget)
        self.timers['train']('outer_start', exclude_total=True, reset_step=True)

        train_dataloader = self.build_or_get_train_dataloader(dataset)
        if self.info['condition']['first']['train']:
            self.update_model()
            LOGGER.info(self)
        self.timers['train']('build_dataset')

        inner_epoch = 0
        while True:
            inner_epoch += 1
            remaining_time_budget -= self.timers['train'].step_time

            self.timers['train']('start', reset_step=True)
            train_metrics = self.epoch_train(self.info['loop']['epoch'], train_dataloader)
            self.timers['train']('train')

            train_score = np.min([c['train']['score'] for c in self.checkpoints[-20:] + [{'train': train_metrics}]])
            if train_score > self.hyper_params['conditions']['skip_valid_score_threshold'] or \
                self.info['loop']['test'] >= self.hyper_params['conditions']['skip_valid_after_test']:
                is_first = self.info['condition']['first']['valid']
                valid_dataloader = self.build_or_get_dataloader('valid', self.datasets['valid'], self.datasets['num_valids'])
                self.timers['train']('valid_dataset', exclude_step=is_first)

                valid_metrics = self.epoch_valid(self.info['loop']['epoch'], valid_dataloader)
                is_skip_valid = False
            else:
                valid_metrics = self.skip_valid(self.info['loop']['epoch'])
                is_skip_valid = True
            self.timers['train']('valid')

            metrics = {
                'epoch': self.info['loop']['epoch'],
                'model': self.get_model_state(),
                'train': train_metrics,
                'valid': valid_metrics,
            }

            self.update_condition(metrics)
            self.timers['train']('adapt', exclude_step=True)

            LOGGER.info(
                '[train] [%02d] time(budge:%.2f, total:%.2f, step:%.2f) loss:(train:%.3f valid:%.3f) score:(train:%.3f valid:%.3f) lr:%f',
                self.info['loop']['epoch'], remaining_time_budget, self.get_total_time(), self.timers['train'].step_time,
                metrics['train']['loss'], metrics['valid']['loss'], metrics['train']['score'], metrics['valid']['score'],
                self.optimizer.get_learning_rate()
            )
            LOGGER.debug('[train] [%02d] Timer:%s', self.info['loop']['epoch'], self.timers['train'])

            self.hyper_params['dataset']['max_epoch'] = self.info['loop']['epoch'] + remaining_time_budget // self.timers['train'].step_time
            LOGGER.info('[ESTIMATE] max_epoch: %d', self.hyper_params['dataset']['max_epoch'])

            if self.break_train_loop_condition(remaining_time_budget, inner_epoch):
                break

            self.timers['train']('end')

        # if 'test' in self.dataloaders and self.dataloaders['test'] is not None and \
        #     not is_skip_valid:
        #     self.prediction(self.dataloaders['test'])
        #     for step in range(3):
        #         test_metric = self.epoch_train(self.info['loop']['epoch'], self.dataloaders['test'])
        #         LOGGER.info(
        #             '[train] [%02d] [%02d/%02d] [finetune] loss:%.3f score:%.3f',
        #             self.info['loop']['epoch'], step, 3, test_metric['loss'], test_metric['score'],
        #         )
        # self.timers['train']('finetune')

        remaining_time_budget -= self.timers['train'].step_time
        self.terminate_train_loop_condition(remaining_time_budget, inner_epoch)

        if not self.done_training:
            self.adapt(remaining_time_budget)

        self.timers['train']('outer_end')
        LOGGER.info(
            '[train] [%02d] time(budge:%.2f, total:%.2f, step:%.2f) loss:(train:%.3f valid:%.3f) score:(train:%.3f valid:%.3f) lr:%f',
            self.info['loop']['epoch'], remaining_time_budget, self.get_total_time(), self.timers['train'].step_time,
            metrics['train']['loss'], metrics['valid']['loss'], metrics['train']['score'], metrics['valid']['score'],
            self.optimizer.get_learning_rate()
        )
        # LOGGER.info('[train] [%02d] Timer:%s', self.info['loop']['epoch'], self.timers['train'])

    def test(self, dataset, remaining_time_budget=None):
        self.timers['test']('start', exclude_total=True, reset_step=True)
        is_first = self.info['condition']['first']['test']
        self.info['loop']['test'] += 1

        dataloader = self.build_or_get_dataloader('test', dataset, self.num_test)
        self.timers['test']('build_dataset', reset_step=is_first)

        rv = self.prediction(dataloader)
        self.timers['test']('end')

        LOGGER.info(
            '[test ] [%02d] test:%02d time(budge:%.2f, total:%.2f, step:%.2f)',
            self.info['loop']['epoch'], self.info['loop']['test'], remaining_time_budget, self.get_total_time(), self.timers['test'].step_time,
        )
        # LOGGER.debug('[test ] [%02d] Timer:%s', self.info['loop']['epoch'], self.timers['test'])
        return rv
