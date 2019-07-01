# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import threading

import tensorflow as tf
import torch
import torchvision as tv
import numpy as np

import skeleton
from architectures.resnet import ResNet18
from skeleton.projects import LogicModel, get_logger
from skeleton.projects.others import NBAC, AUC


torch.backends.cudnn.benchmark = True
threads = [
    threading.Thread(target=lambda: torch.cuda.synchronize()),
    threading.Thread(target=lambda: tf.Session())
]
[t.start() for t in threads]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LOGGER = get_logger(__name__)


class Model(LogicModel):
    def __init__(self, metadata):
        super(Model, self).__init__(metadata)
        self.use_test_time_augmentation = False

    def build(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        in_channels = self.info['dataset']['shape'][-1]
        num_class = self.info['dataset']['num_class']
        # torch.cuda.synchronize()

        LOGGER.info('[init] session')
        [t.join() for t in threads]

        self.device = torch.device('cuda', 0)
        self.session = tf.Session()

        LOGGER.info('[init] Model')
        Network = ResNet18  # ResNet18  # BasicNet, SENet18, ResNet18
        self.model = Network(in_channels, num_class)
        self.model_pred = Network(in_channels, num_class).eval()
        # torch.cuda.synchronize()

        LOGGER.info('[init] weight initialize')
        if Network in [ResNet18]:
            model_path = os.path.join(base_dir, 'models')
            LOGGER.info('model path: %s', model_path)

            self.model.init(model_dir=model_path, gain=1.0)
        else:
            self.model.init(gain=1.0)
        # torch.cuda.synchronize()

        LOGGER.info('[init] copy to device')
        self.model = self.model.to(device=self.device).half()
        self.model_pred = self.model_pred.to(device=self.device).half()
        self.is_half = self.model._half
        # torch.cuda.synchronize()

        LOGGER.info('[init] done.')

    def update_model(self):
        num_class = self.info['dataset']['num_class']

        epsilon = min(0.1, max(0.001, 0.001 * pow(num_class / 10, 2)))
        if self.is_multiclass():
            self.model.loss_fn = torch.nn.BCEWithLogitsLoss(reduction='none')
            # self.model.loss_fn = skeleton.nn.BinaryCrossEntropyLabelSmooth(num_class, epsilon=epsilon, reduction='none')
            self.tau = 8.0
            LOGGER.info('[update_model] %s (tau:%f, epsilon:%f)', self.model.loss_fn.__class__.__name__, self.tau, epsilon)
        else:
            self.model.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
            # self.model.loss_fn = skeleton.nn.CrossEntropyLabelSmooth(num_class, epsilon=epsilon)
            self.tau = 8.0
            LOGGER.info('[update_model] %s (tau:%f, epsilon:%f)', self.model.loss_fn.__class__.__name__, self.tau, epsilon)
        self.model_pred.loss_fn = self.model.loss_fn

        self.init_opt()
        LOGGER.info('[update] done.')

    def init_opt(self):
        steps_per_epoch = self.hyper_params['dataset']['steps_per_epoch']
        batch_size = self.hyper_params['dataset']['batch_size']

        params = [p for p in self.model.parameters() if p.requires_grad]

        warmup_multiplier = 2.0
        lr_multiplier = max(1.0, batch_size / 32)
        scheduler_lr = skeleton.optim.gradual_warm_up(
            skeleton.optim.get_reduce_on_plateau_scheduler(
                0.025 * lr_multiplier / warmup_multiplier,
                patience=10, factor=.5, metric_name='train_loss'
            ),
            warm_up_epoch=5,
            multiplier=warmup_multiplier
        )
        self.optimizer = skeleton.optim.ScheduledOptimizer(
            params,
            torch.optim.SGD,
            # skeleton.optim.SGDW,
            steps_per_epoch=steps_per_epoch,
            clip_grad_max_norm=None,
            lr=scheduler_lr,
            momentum=0.9,
            weight_decay=0.001 * 1 / 4,
            nesterov=True
        )
        LOGGER.info('[optimizer] %s (batch_size:%d)', self.optimizer._optimizer.__class__.__name__, batch_size)

    def adapt(self, remaining_time_budget=None):
        epoch = self.info['loop']['epoch']
        input_shape = self.hyper_params['dataset']['input']
        height, width = input_shape[:2]
        batch_size = self.hyper_params['dataset']['batch_size']

        train_score = np.average([c['train']['score'] for c in self.checkpoints[-5:]])
        valid_score = np.average([c['valid']['score'] for c in self.checkpoints[-5:]])
        LOGGER.info('[adapt] [%04d/%04d] train:%.3f valid:%.3f',
                    epoch, self.hyper_params['dataset']['max_epoch'],
                    train_score, valid_score)

        self.use_test_time_augmentation = self.info['loop']['test'] > 1

        # Adapt Apply Fast auto aug
        if self.hyper_params['conditions']['use_fast_auto_aug'] and \
                (train_score > 0.995 or self.info['terminate']) and \
                remaining_time_budget > 120 and \
                self.dataloaders['valid'] is not None and \
                not hasattr(self, 'update_transforms'):
            LOGGER.info('[adapt] search fast auto aug policy')
            self.update_transforms = True
            self.info['terminate'] = True

            # reset optimizer pararms
            self.init_opt()
            self.hyper_params['conditions']['max_inner_loop_ratio'] *= 3
            self.hyper_params['conditions']['threshold_valid_score_diff'] = 0.00001
            self.hyper_params['conditions']['min_lr'] = 1e-8

            original_valid_policy = self.dataloaders['valid'].dataset.transform.transforms
            original_train_policy = self.dataloaders['train'].dataset.transform.transforms
            policy = skeleton.data.augmentations.autoaug_policy()

            num_policy_search = 100
            num_sub_policy = 3
            num_select_policy = 3
            searched_policy = []
            for policy_search in range(num_policy_search):
                selected_idx = np.random.choice(list(range(len(policy))), num_sub_policy)
                selected_policy = [policy[i] for i in selected_idx]
                self.dataloaders['valid'].dataset.transform.transforms = original_valid_policy + [
                    lambda t: t.cpu().float() if isinstance(t, torch.Tensor) else torch.Tensor(t),
                    tv.transforms.ToPILImage(),
                    skeleton.data.augmentations.Augmentation(
                        selected_policy
                    ),
                    tv.transforms.ToTensor(),
                    lambda t: t.to(device=self.device).half()
                ]

                metrics = []
                for policy_eval in range(num_sub_policy):
                    valid_dataloader = self.build_or_get_dataloader('valid', self.datasets['valid'], self.datasets['num_valids'])
                    # original_valid_batch_size = valid_dataloader.batch_sampler.batch_size
                    # valid_dataloader.batch_sampler.batch_size = batch_size

                    valid_metrics = self.epoch_valid(self.info['loop']['epoch'], valid_dataloader, reduction='max')

                    # valid_dataloader.batch_sampler.batch_size = original_valid_batch_size
                    metrics.append(valid_metrics)
                loss = np.max([m['loss'] for m in metrics])
                score = np.max([m['score'] for m in metrics])
                LOGGER.info('[adapt] [FAA] [%02d/%02d] score: %f, loss: %f, selected_policy: %s',
                            policy_search, num_policy_search, score, loss, selected_policy)

                searched_policy.append({
                    'loss': loss,
                    'score': score,
                    'policy': selected_policy
                })

            flatten = lambda l: [item for sublist in l for item in sublist]

            policy_sorted_index = np.argsort([p['score'] for p in searched_policy])[::-1][:num_select_policy]
            policy = flatten([searched_policy[idx]['policy'] for idx in policy_sorted_index])
            policy = skeleton.data.augmentations.remove_duplicates(policy)

            LOGGER.info('[adapt] [FAA] scores: %s',
                        [searched_policy[idx]['score'] for idx in policy_sorted_index])

            self.dataloaders['valid'].dataset.transform.transforms = original_valid_policy
            self.dataloaders['train'].dataset.transform.transforms = original_train_policy + [
                lambda t: t.cpu().float() if isinstance(t, torch.Tensor) else torch.Tensor(t),
                tv.transforms.ToPILImage(),
                skeleton.data.augmentations.Augmentation(
                    policy
                ),
                tv.transforms.ToTensor(),
                lambda t: t.to(device=self.device).half()
            ]

    def activation(self, logits):
        if self.is_multiclass():
            logits = torch.sigmoid(logits)
            prediction = (logits > 0.5).to(logits.dtype)
        else:
            logits = torch.softmax(logits, dim=-1)
            _, k = logits.max(-1)
            prediction = torch.zeros(logits.shape, dtype=logits.dtype, device=logits.device).scatter_(-1, k.view(-1, 1), 1.0)
        return logits, prediction

    def get_model_state(self):
        return self.model.state_dict()

    def epoch_train(self, epoch, train, model=None, optimizer=None):
        model = model if model is not None else self.model
        optimizer = optimizer if optimizer is not None else self.optimizer
        model.train()
        num_steps = len(train)
        metrics = []
        for step, (examples, labels) in enumerate(train):
            if examples.shape[0] == 1:
                examples = examples[0]
                labels = labels[0]
            original_labels = labels
            if not self.is_multiclass():
                labels = labels.argmax(dim=-1)

            # batch_size = examples.size(0)
            # examples = torch.cat([examples, torch.flip(examples, dims=[-1])], dim=0)
            # labels = torch.cat([labels, labels], dim=0)

            skeleton.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
            logits, loss = model(examples, labels, tau=self.tau)
            loss.backward()

            max_epoch = self.hyper_params['dataset']['max_epoch']
            optimizer.update(maximum_epoch=max_epoch)
            optimizer.step()
            model.zero_grad()

            # logits1, logits2 = torch.split(logits, batch_size, dim=0)
            # logits = (logits1 + logits2) / 2.0

            logits, prediction = self.activation(logits.float())
            tpr, tnr, nbac = NBAC(prediction, original_labels.float())
            auc = AUC(logits, original_labels.float())

            score = auc if self.hyper_params['conditions']['score_type'] == 'auc' else float(nbac.detach().float())
            metrics.append({
                'loss': loss.detach().float().cpu(),
                'score': score,
            })

            LOGGER.debug(
                '[train] [%02d] [%03d/%03d] loss:%.6f AUC:%.3f NBAC:%.3f tpr:%.3f tnr:%.3f, lr:%.8f',
                epoch, step, num_steps, loss, auc, nbac, tpr, tnr,
                optimizer.get_learning_rate()
            )

        train_loss = np.average([m['loss'] for m in metrics])
        train_score = np.average([m['score'] for m in metrics])
        optimizer.update(train_loss=train_loss)

        return {
            'loss': train_loss,
            'score': train_score,
        }

    def epoch_valid(self, epoch, valid, reduction='avg'):
        self.model.eval()
        num_steps = len(valid)
        metrics = []
        tau = self.tau

        for step, (examples, labels) in enumerate(valid):
            original_labels = labels
            if not self.is_multiclass():
                labels = labels.argmax(dim=-1)

            # skeleton.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
            logits, loss = self.model(examples, labels, tau=tau, reduction=reduction)

            logits, prediction = self.activation(logits.float())
            tpr, tnr, nbac = NBAC(prediction, original_labels.float())
            auc = AUC(logits, original_labels.float())

            score = auc if self.hyper_params['conditions']['score_type'] == 'auc' else float(nbac.detach().float())
            metrics.append({
                'loss': loss.detach().float().cpu(),
                'score': score,
            })

            LOGGER.debug(
                '[valid] [%02d] [%03d/%03d] loss:%.6f AUC:%.3f NBAC:%.3f tpr:%.3f tnr:%.3f, lr:%.8f',
                epoch, step, num_steps, loss, auc, nbac, tpr, tnr,
                self.optimizer.get_learning_rate()
            )
        if reduction == 'avg':
            valid_loss = np.average([m['loss'] for m in metrics])
            valid_score = np.average([m['score'] for m in metrics])
        elif reduction == 'max':
            valid_loss = np.max([m['loss'] for m in metrics])
            valid_score = np.max([m['score'] for m in metrics])
        elif reduction == 'min':
            valid_loss = np.min([m['loss'] for m in metrics])
            valid_score = np.min([m['score'] for m in metrics])
        else:
            raise Exception('not support reduction method: %s' % reduction)
        self.optimizer.update(valid_loss=np.average(valid_loss))

        return {
            'loss': valid_loss,
            'score': valid_score,
        }

    def skip_valid(self, epoch):
        LOGGER.debug('[valid] skip')
        return {
            'loss': 99.9,
            'score': epoch * 1e-4,
        }

    def prediction(self, dataloader):
        self.model_pred.eval()
        epoch = self.info['loop']['epoch']

        best_idx = np.argmax(np.array([c['valid']['score'] for c in self.checkpoints]))
        best_loss = self.checkpoints[best_idx]['valid']['loss']
        best_score = self.checkpoints[best_idx]['valid']['score']

        tau = self.tau

        states = self.checkpoints[best_idx]['model']
        self.model_pred.load_state_dict(states)
        LOGGER.info('best checkpoints at %d/%d (valid loss:%f score:%f) tau:%f',
                    best_idx + 1, len(self.checkpoints), best_loss, best_score, tau)

        predictions = []
        self.model_pred.eval()
        for step, (examples, labels) in enumerate(dataloader):
            # examples = examples[0]
            # skeleton.nn.MoveToHook.to((examples, labels), self.device, self.is_half)

            batch_size = examples.size(0)

            # Test-Time Augment flip
            if self.use_test_time_augmentation:
                examples = torch.cat([examples, torch.flip(examples, dims=[-1])], dim=0)

            # skeleton.nn.MoveToHook.to((examples, labels), self.device, self.is_half)
            logits = self.model_pred(examples, tau=tau)

            # avergae
            if self.use_test_time_augmentation:
                logits1, logits2 = torch.split(logits, batch_size, dim=0)
                logits = (logits1 + logits2) / 2.0

            logits, prediction = self.activation(logits)

            predictions.append(logits.detach().float().cpu().numpy())

        predictions = np.concatenate(predictions, axis=0).astype(np.float)
        return predictions
