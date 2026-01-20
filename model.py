import torch
import torch.nn as nn

class CNN_LSTM_Hybrid(nn.Module):
    def __init__(self, bert_dim=768, temporal_dim=4, lstm_hidden=256, cnn_out=128, fusion_hidden=128, dropout=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=bert_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.cnn = nn.Conv1d(
            in_channels=2 * lstm_hidden,
            out_channels=cnn_out,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.temporal_fc = nn.Sequential(
            nn.Linear(temporal_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(cnn_out + 32, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 1)
        )

    def forward(self, bert_seq, temporal_feat):
        lstm_out, _ = self.lstm(bert_seq)
        lstm_out = lstm_out.transpose(1, 2)

        cnn_out = self.cnn(lstm_out)
        cnn_out = self.pool(cnn_out).squeeze(-1)

        temp_out = self.temporal_fc(temporal_feat)

        fused = torch.cat([cnn_out, temp_out], dim=1)
        out = self.classifier(fused)

        return out
