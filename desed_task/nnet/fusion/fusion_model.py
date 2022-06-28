import torch
import torch.nn as nn
import warnings
from pytorch_lightning import LightningModule
from desed_task.nnet.fusion.RNN import BidirectionalGRU
from desed_task.nnet.fusion.CNN import CNN


class ATSTModel(LightningModule):
    def __init__(self,
                 encoder,
                 atst_mode,
                 chunk_input=False,
                 n_in_channel=1,
                 nclass=10,
                 attention=True,
                 activation="glu",
                 dropout=0.5,
                 train_cnn=True,
                 rnn_type="BGRU",
                 n_RNN_cell=128,
                 n_layers_RNN=2,
                 dropout_recurrent=0,
                 cnn_integration=False,
                 freeze_bn=False,
                 atst_res_block=False,
                 **kwargs,
                 ):
        super().__init__()
        self.encoder = encoder
        self.sigmoid = nn.Sigmoid()
        self.chunk_input = chunk_input
        if self.chunk_input:
            print("Using chunk input")

        # Copy from CRNN:
        self.n_in_channel = n_in_channel
        self.attention = attention
        self.cnn_integration = cnn_integration
        self.freeze_bn = freeze_bn

        n_in_cnn = n_in_channel

        if cnn_integration:
            n_in_cnn = 1

        self.cnn = CNN(
            n_in_channel=n_in_cnn,
            activation=activation,
            conv_dropout=dropout,
            atst_res_block=atst_res_block,
            **kwargs
        )

        self.train_cnn = train_cnn
        if not train_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        if rnn_type == "BGRU":
            nb_in = self.cnn.nb_filters[-1] * 7 if atst_mode == "base" else self.cnn.nb_filters[-1] * 4
            if self.cnn_integration:
                # self.fc = nn.Linear(nb_in * n_in_channel, nb_in)
                nb_in = nb_in * n_in_channel

            self.rnn = BidirectionalGRU(
                n_in=nb_in,
                n_hidden=n_RNN_cell,
                dropout=dropout_recurrent,
                num_layers=n_layers_RNN,
            )
        else:
            NotImplementedError("Only BGRU supported for CRNN for now")

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(n_RNN_cell * 2, nclass)
        self.sigmoid = nn.Sigmoid()

        if self.attention:
            self.dense_softmax = nn.Linear(n_RNN_cell * 2, nclass)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature_crnn, feature_atst, temp=None):
        # ATST feature processing - transformer encoder
        # Warp the batch into the shape of ATST required
        feature_atst = feature_atst.unsqueeze(1)
        # Generate a pseudo length vector
        lengths = torch.ones(len(feature_atst)).to(feature_atst) * feature_atst.shape[-1]
        feats, _ = self.encoder(((feature_atst, lengths), None))
        bsz, timestamp, feat_dim = feats[0].shape

        # ATST feature processing - input for RNN block
        # input level
        # v_spectrogram = torch.concat([feats[0].reshape(bsz, timestamp, 32, -1), feats[1].reshape(bsz, timestamp, 32, -1)], dim=-1)
        # hidden level
        v_spectrogram = sum(feats) / len(feats)
        # Drop cls token
        # input level
        # v_spectrogram = v_spectrogram.permute(0, 3, 1, 2)[:, :, 1:, :]
        # hidden level
        v_spectrogram = v_spectrogram[:, 1:, :]

        # CNN processing
        r_spectrogram = feature_crnn.transpose(1, 2).unsqueeze(1)
        # input size : (batch_size, n_channels, n_frames, n_freq)

        # conv features
        # input level
        # x = self.cnn(r_spectrogram, v_spectrogram)
        # hidden level
        x = self.cnn(r_spectrogram)
        x = torch.concat([x, v_spectrogram.transpose(1, 2).unsqueeze(-1)], dim=1)
        bs, chan, frames, freq = x.size()

        if freq != 1:
            warnings.warn(
                f"Output shape is: {(bs, frames, chan * freq)}, from {freq} staying freq"
            )
            x = x.permute(0, 2, 1, 3)
            x = x.contiguous().view(bs, frames, chan * freq)
        else:
            x = x.squeeze(-1)
            x = x.permute(0, 2, 1)  # [bs, frames, chan]

        # rnn features
        x = self.rnn(x)

        x = self.dropout(x)
        strong_logits = self.dense(x)  # [bs, frames, nclass]
        if temp is not None:
            strong_logits = strong_logits / temp
        strong = self.sigmoid(strong_logits)
        if self.attention:
            sof = self.dense_softmax(x)  # [bs, frames, nclass]
            sof = self.softmax(sof)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)  # [bs, nclass]
        else:
            weak = strong.mean(1)

        if self.chunk_input:
            chunk_end = len(weak) // 2  # Half of bsz
            weak = (weak[: chunk_end, :] + weak[chunk_end:, :]) / 2
            strong = torch.cat([strong[: chunk_end, :, :], strong[chunk_end:, :, :]], dim=1)
        return strong.transpose(1, 2), weak
