import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer

# ===== Hyperparams =====
INPUT_DIM   = 10
EMB_SIZE    = 32
NHEAD       = 2
FFN_HID_DIM = 64
NUM_LAYERS  = 2
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Positional Encoding (batch_first: x shape = [B, T, C]) =====
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, maxlen=100):
        super().__init__()
        pe = torch.zeros(maxlen, emb_size)                   # [T, C]
        position = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)                                 # [1, T, C]
        self.register_buffer("pe", pe)

    def forward(self, x):  # x: [B, T, C]
        T = x.size(1)
        return x + self.pe[:, :T, :]

# ===== Mini Transformer (Encoder–Decoder) =====
class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_emb = nn.Embedding(INPUT_DIM, EMB_SIZE)
        self.tgt_emb = nn.Embedding(INPUT_DIM, EMB_SIZE)
        self.pos_enc = PositionalEncoding(EMB_SIZE)

        self.transformer = Transformer(
            d_model=EMB_SIZE, nhead=NHEAD,
            num_encoder_layers=NUM_LAYERS,
            num_decoder_layers=NUM_LAYERS,
            dim_feedforward=FFN_HID_DIM,
            dropout=0.1,
            batch_first=True,          # <— dùng batch_first để thuận tiện
            norm_first=True
        )
        self.fc_out = nn.Linear(EMB_SIZE, INPUT_DIM)

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device):
        # mask cho decoder: không nhìn tương lai
        # 0 trên/ dưới đường chéo chính? Transformer dùng -inf cho vị trí cấm.
        mask = torch.full((sz, sz), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(self, src, tgt):
        # src, tgt: [B, T]
        src = self.pos_enc(self.src_emb(src))  # [B, Ts, C]
        tgt = self.pos_enc(self.tgt_emb(tgt))  # [B, Tt, C]

        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), src.device)  # [Tt, Tt]
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)  # [B, Tt, C]
        return self.fc_out(out)  # [B, Tt, V]

# ===== Toy data =====
src    = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=DEVICE)  # [B=1, Ts=4]
tgt_in = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=DEVICE)  # decoder input
tgt_out= torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=DEVICE)  # next tokens

# ===== Train =====
model = MiniTransformer().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    logits = model(src, tgt_in)             # [B, Tt, V]
    B, Tt, V = logits.shape
    loss = criterion(logits.reshape(B*Tt, V), tgt_out.reshape(B*Tt))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:02d} | Loss = {loss.item():.4f}")
