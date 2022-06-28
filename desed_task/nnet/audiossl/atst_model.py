import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from desed_task.nnet.vanilla.RNN import BidirectionalGRU
from desed_task.nnet.vanilla.CRNN import CRNN


class ATSTModel(LightningModule):
    def __init__(self,
                 encoder,
                 dim=384,
                 nclass=10,
                 chunk_input=False,
                 n_RNN_cell=128,
                 n_layers_RNN=2,
                 dropout_recurrent=0,
                 dropout=0,
                 cls="RNN",
                 soft_attn=True,
                 **kwargs
                 ):
        super().__init__()
        self.encoder = encoder
        self.sigmoid = nn.Sigmoid()
        self.chunk_input = chunk_input
        self.cls_token = kwargs["ast"]["use_cls"]
        self.attention = soft_attn
        self.downstream = cls
        if cls == "RNN":
            self.rnn = BidirectionalGRU(
                    n_in=dim,
                    n_hidden=n_RNN_cell,
                    dropout=dropout_recurrent,
                    num_layers=n_layers_RNN,
                )
            self.dropout = nn.Dropout(dropout)
            self.dense = nn.Linear(n_RNN_cell * 2, nclass)
            self.sigmoid = nn.Sigmoid()

            if self.attention:
                print("Using soft attn for weak predictions")
                self.softmax = nn.Softmax(dim=-1)
                self.soft_dense = nn.Linear(n_RNN_cell * 2, nclass)
        elif cls == "CRNN":
            # self.tfm2crnn = nn.Linear(dim, 128) # Transform to spectrogram size
            self.crnn = CRNN(**kwargs["crnn"])
        elif cls == "MLP":
            self.mlp = nn.Linear(dim, nclass)
            self.softmax = nn.Softmax(dim=-1)
            self.soft_dense = nn.Linear(dim, nclass)
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(dropout)

    def forward(self, feature, temp=None):
        # Warp the batch into the shape of ATST required
        feature = feature.unsqueeze(1)
        # Generate a pseudo length vector
        lengths = torch.ones(len(feature)).to(feature) * feature.shape[-1]
        feats, _ = self.encoder(((feature, lengths), None))
        last_layer = sum(feats) / len(feats)
        # RNN decoder
        if self.downstream == "RNN":
            if self.attention:
                last_layer = last_layer[:, 1:, :]
            x = self.rnn(last_layer)
            x = self.dropout(x)
            # Concatenate cls token with the rest tokens
            # strong = x[:, 1:, :]
            # weak = x[:, 0, :].unsqueeze(1).expand(-1, strong.shape[1], -1)
            # v64
            # x = torch.cat([strong, weak], dim=-1)
            # v65
            # x = torch.cat([weak + strong, weak * strong], -1)
            strong_logits = self.dense(x)
            if temp is not None:
                strong_logits = strong_logits / temp
            # Generate weak and strong predictions
            pred_probs = self.sigmoid(strong_logits)
            if not self.attention:
                weak = pred_probs[:, 0, :]
                strong = pred_probs[:, 1:, :]
            else:
                strong = pred_probs
                soft_logits = self.soft_dense(x)
                soft_mask = self.softmax(soft_logits)
                # hard_mask = torch.gt(soft_mask, soft_mask.mean())
                # soft_mask = soft_mask * hard_mask
                soft_mask = torch.clamp(soft_mask, min=1e-7, max=1)
                weak = (strong * soft_mask).sum(1) / soft_mask.sum(1)

            if self.chunk_input:
                chunk_end = len(weak) // 2  # Half of bsz
                weak = (weak[: chunk_end, :] + weak[chunk_end:, :]) / 2
                strong = torch.cat([strong[: chunk_end, :, :], strong[chunk_end:, :, :]], dim=1)

            return strong.transpose(-2, -1), weak

        elif self.downstream == "CRNN":
            v_spectrogram = last_layer[:, 1:, :]
            # v_spectrogram = self.tfm2crnn(last_layer)
            strong, weak = self.crnn(v_spectrogram.transpose(1, 2))
            return strong, weak

        elif self.downstream == "MLP":
            x = self.dropout(last_layer[:, 1:, :])
            strong_logits = self.mlp(x)
            if temp is not None:
                strong_logits = strong_logits / temp
            strong = self.sigmoid(strong_logits)
            soft_logits = self.soft_dense(x)
            soft_mask = self.softmax(soft_logits)
            # hard_mask = torch.gt(soft_mask, soft_mask.mean())
            # soft_mask = soft_mask * hard_mask
            soft_mask = torch.clamp(soft_mask, min=1e-7, max=1)
            weak = (strong * soft_mask).sum(1) / soft_mask.sum(1)
            return strong.transpose(-2, -1), weak
