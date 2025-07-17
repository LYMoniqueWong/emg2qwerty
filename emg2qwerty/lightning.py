# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
    RNNBlock,
    AttentionBlock,
)
from emg2qwerty.transforms import Transform
import logging


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        batch_norm: DictConfig,
        layer_norm: DictConfig,
        **kwarg,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]
        self.num_features = num_features

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
        )
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        if self.batch_norm:
            self.BN = nn.BatchNorm1d(num_features)
        if self.layer_norm:
            self.LN = nn.LayerNorm(num_features)

        self.fc_layer = nn.Linear(num_features, charset().num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def _tds(self, inputs: torch.Tensor):
        x = self.model(inputs)
        if self.batch_norm:
            x = x.permute(1, 2, 0)
            x = self.BN(x)
            x = x.permute(2, 0, 1)
        if self.layer_norm:
            x = self.LN(x)

        return x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.fc_layer(self._tds(inputs)))

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class TDSConvLSTMModule(TDSConvCTCModule):
    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        rnn_config: DictConfig,
        batch_norm: DictConfig,
        layer_norm: DictConfig,
        **kwarg,
    ):
        super().__init__(
            in_features=in_features,
            mlp_features=mlp_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            decoder=decoder,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
        )
        self.rnn = RNNBlock(
            input_size=self.num_features,
            hidden_size=self.num_features,
            num_layers=rnn_config.num_layers,
            dropout=rnn_config.dropout,
            bidirectional=rnn_config.bidirection,
            useLayerNorm=rnn_config.layer_norm,
        )
        self.proj_rnn = nn.Linear(
            2 * self.num_features if rnn_config.bidirection else self.num_features,
            self.num_features,  # Reduce to match input size
        )

        self.useBatchNorm = rnn_config.batch_norm
        if self.useBatchNorm:
            self.BN = nn.BatchNorm1d(self.num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._tds(inputs)
        x_rnn = self.rnn(x)
        x_rnn = self.proj_rnn(x_rnn)

        if self.useBatchNorm:
            x_rnn = x_rnn.permute(1, 2, 0)
            x_rnn = self.BN(x_rnn)
            x_rnn = x_rnn.permute(2, 0, 1)

        m = nn.Dropout(p=0.4)
        x_rnn = m(x_rnn)

        return self.softmax(self.fc_layer(x_rnn))


class TDSConvLSTMModuleV2(TDSConvCTCModule):
    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        rnn_config: DictConfig,
        batch_norm: DictConfig,
        layer_norm: DictConfig,
        **kwarg,
    ):
        super().__init__(
            in_features=in_features,
            mlp_features=mlp_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            decoder=decoder,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
        )
        self.rnn = RNNBlock(
            input_size=self.num_features,
            hidden_size=self.num_features,
            num_layers=rnn_config.num_layers,
            dropout=rnn_config.dropout,
            bidirectional=rnn_config.bidirection,
            useLayerNorm=rnn_config.layer_norm,
        )
        self.fc_layer = nn.Linear(
            2 * self.num_features if rnn_config.bidirection else self.num_features,
            charset().num_classes,  # Reduce to match input size
        )

        self.useBatchNorm = rnn_config.batch_norm
        if self.useBatchNorm:
            self.BN = nn.BatchNorm1d(self.num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._tds(inputs)
        x_rnn = self.rnn(x)
        m = nn.Dropout(p=0.4)
        x_rnn = m(x_rnn)

        if self.useBatchNorm:
            x_rnn = x_rnn.permute(1, 2, 0)
            x_rnn = self.BN(x_rnn)
            x_rnn = x_rnn.permute(2, 0, 1)

        x_rnn = m(x_rnn)

        return self.softmax(self.fc_layer(x_rnn))


class TDSConvLSTMModuleV3(TDSConvCTCModule):
    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        rnn_config: DictConfig,
        batch_norm: DictConfig,
        layer_norm: DictConfig,
        **kwarg,
    ):
        super().__init__(
            in_features=in_features,
            mlp_features=mlp_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            decoder=decoder,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
        )

        self.rnn_dim = rnn_config.rnn_dim
        self.rnn_dropout_p = rnn_config.dropout

        self.toRnn = nn.Linear(self.num_features, self.rnn_dim)

        self.rnn = RNNBlock(
            input_size=self.rnn_dim,
            hidden_size=self.rnn_dim,
            num_layers=rnn_config.num_layers,
            dropout=rnn_config.dropout,
            bidirectional=rnn_config.bidirection,
            useLayerNorm=rnn_config.layer_norm,
        )
        self.fc_layer = nn.Linear(
            2 * self.rnn_dim if rnn_config.bidirection else self.rnn_dim,
            charset().num_classes,  # Reduce to match input size
        )

        self.useBatchNorm = rnn_config.batch_norm
        if self.useBatchNorm:
            self.BN = nn.BatchNorm1d(self.num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._tds(inputs)
        x = self.toRnn(x)

        m = nn.Dropout(p=self.rnn_dropout_p)

        x = m(x)

        x_rnn = self.rnn(x)

        if self.useBatchNorm:
            x_rnn = x_rnn.permute(1, 2, 0)
            x_rnn = self.BN(x_rnn)
            x_rnn = x_rnn.permute(2, 0, 1)

        x_rnn = m(x_rnn)

        return self.softmax(self.fc_layer(x_rnn))


# class TDSConvAttentionModuleV1(TDSConvCTCModule):
#     def __init__(
#         self,
#         in_features: int,
#         mlp_features: Sequence[int],
#         block_channels: Sequence[int],
#         kernel_width: int,
#         optimizer: DictConfig,
#         lr_scheduler: DictConfig,
#         decoder: DictConfig,
#         # attn_config: DictConfig,
#         batch_norm: DictConfig,
#         layer_norm: DictConfig,
#         **kwargs,
#     ):
#         super().__init__(
#             in_features=in_features,
#             mlp_features=mlp_features,
#             block_channels=block_channels,
#             kernel_width=kernel_width,
#             optimizer=optimizer,
#             lr_scheduler=lr_scheduler,
#             decoder=decoder,
#             batch_norm=batch_norm,
#             layer_norm=layer_norm,
#         )

#         self.attn_dim = 128  # attn_config.attn_dim
#         self.attn_dropout_p = 0.6  # attn_config.dropout

#         self.toAttn = nn.Linear(self.num_features, self.attn_dim)

#         self.attn = AttentionBlock(
#             input_size=self.attn_dim,
#             num_heads=4,  # attn_config.num_heads,
#             dropout=self.attn_dropout_p,
#             useLayerNorm=True,  # attn_config.layer_norm,
#         )

#         self.fc_layer = nn.Linear(
#             self.attn_dim,
#             charset().num_classes,  # Reduce to match input size
#         )

#         self.useBatchNorm = False  # attn_config.batch_norm
#         if self.useBatchNorm:
#             self.BN = nn.BatchNorm1d(self.num_features)

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         x = self._tds(inputs)
#         x = self.toAttn(x)

#         m = nn.Dropout(p=self.attn_dropout_p)

#         x = m(x)

#         x_attn = self.attn(x)

#         if self.useBatchNorm:
#             x_attn = x_attn.permute(1, 2, 0)  # Convert to (N, C, T)
#             x_attn = self.BN(x_attn)
#             x_attn = x_attn.permute(2, 0, 1)  # Convert back to (T, N, C)

#         x_attn = m(x_attn)

#         return self.softmax(self.fc_layer(x_attn))
