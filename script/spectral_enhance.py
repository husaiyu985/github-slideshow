import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.nn import functional as F
import pickle
import h5py
from typing import Optional, List


# =========================
# Utils: device and seeds
# =========================
def get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return torch.device("cuda")
        except RuntimeError:
            return torch.device("cpu")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Dataset
# =========================
class H5IQSignalDataset(Dataset):
    def __init__(self, h5_path: str, target_snr_db: Optional[float] = None,
                 phase_rotate: bool = False, cfo_ppm: Optional[float] = None):
        self.target_snr_db = target_snr_db
        self.phase_rotate = phase_rotate
        self.cfo_ppm = cfo_ppm
        with h5py.File(h5_path, 'r') as f:
            self.iq_data = f['X'][:].transpose(2, 1, 0).astype(np.float32)  # [N,32,2]
            self.labels = f['y'][:].T.astype(np.float32)  # [N,16]

    def __len__(self):
        return len(self.iq_data)

    def __getitem__(self, idx):
        sample = self.iq_data[idx].T.copy()  # [2,32]
        label = self.labels[idx]

        # Optional SNR augmentation
        if self.target_snr_db is not None:
            power = np.mean(sample**2)
            noise_power = power / (10**(self.target_snr_db/10))
            noise = np.random.normal(0, np.sqrt(noise_power/2), sample.shape).astype(np.float32)
            sample += noise

        # Optional random phase rotation
        if self.phase_rotate:
            theta = np.random.uniform(0.0, 2*np.pi)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]], dtype=np.float32)
            sample = rot @ sample

        # Optional slight CFO (carrier frequency offset)
        if self.cfo_ppm is not None and self.cfo_ppm != 0:
            ppm = np.random.uniform(-self.cfo_ppm, self.cfo_ppm)
            L = sample.shape[-1]
            n = np.arange(L, dtype=np.float32)
            # Assume nominal fs; ppm only changes discrete-time phase slope
            phi = 2*np.pi*(ppm*1e-6)*n
            c = np.stack([np.cos(phi), np.sin(phi)], axis=0).astype(np.float32)
            # complex multiply: (I + jQ) * exp(jphi)
            I, Q = sample[0], sample[1]
            I2 = I*c[0] - Q*c[1]
            Q2 = I*c[1] + Q*c[0]
            sample = np.stack([I2, Q2], axis=0)

        return torch.from_numpy(sample), torch.from_numpy(label)


# =========================
# FFT/PSD helpers (Hann + fftshift + nfft)
# =========================
def _get_hann_window(length: int, device: torch.device) -> torch.Tensor:
    return torch.hann_window(length, device=device)


def fft_with_window(x: torch.Tensor, nfft: int) -> torch.Tensor:
    # x: [B,2,L]
    B, _, L = x.shape
    device = x.device
    w = _get_hann_window(L, device)
    z = torch.complex(x[:, 0] * w, x[:, 1] * w)  # [B,L]
    X = torch.fft.fftshift(torch.fft.fft(z, n=nfft), dim=-1)  # [B,nfft]
    return X


def ifft_with_trim(X: torch.Tensor, L: int) -> torch.Tensor:
    # X: [B,nfft] (shifted)
    Y = torch.fft.ifftshift(X, dim=-1)
    y = torch.fft.ifft(Y, n=X.shape[-1])  # [B,nfft]
    y = y[..., :L]
    return torch.stack([y.real, y.imag], dim=1)  # [B,2,L]


def compute_channel_psd(x: torch.Tensor, num_channels: int = 16, nfft: int = 256,
                        nperseg: int = 32) -> torch.Tensor:
    # x: [B,2,L]
    X = fft_with_window(x, nfft=nfft)
    psd = (X.abs() ** 2) / float(nperseg)
    band_size = nfft // num_channels
    # average per band using unfold
    bands = psd.unfold(-1, band_size, band_size).mean(-1)  # [B, n_bands]
    if bands.shape[-1] > num_channels:
        bands = bands[..., :num_channels]
    return bands


# =========================
# Loss: band-gain margin + smoothing + power
# =========================
def band_gain_margin_loss(enhanced: torch.Tensor, original: torch.Tensor, labels: torch.Tensor,
                          nfft: int = 256, pos_db: float = 3.0, neg_db: float = 3.0,
                          smooth_lambda: float = 0.05, power_lambda: float = 0.01,
                          eps: float = 1e-12) -> torch.Tensor:
    def bands_db(x: torch.Tensor) -> torch.Tensor:
        bands = compute_channel_psd(x, num_channels=16, nfft=nfft, nperseg=32)
        return 10.0 * torch.log10(bands + eps)

    enh_db = bands_db(enhanced)
    orig_db = bands_db(original)
    gain_db = enh_db - orig_db  # [B,16]

    pos = (labels > 0.5).float()
    neg = 1.0 - pos

    loss_pos = F.relu(pos_db - gain_db) * pos
    loss_neg = F.relu(gain_db + neg_db) * neg

    # smoothing across adjacent bands
    smooth = (gain_db[:, 1:] - gain_db[:, :-1]).pow(2).mean()

    # power conservation (optional, small weight)
    power_diff = (enhanced.pow(2).sum(dim=(1, 2)) - original.pow(2).sum(dim=(1, 2))).abs().mean()

    main = (loss_pos.sum() / (pos.sum() + 1e-6) + loss_neg.sum() / (neg.sum() + 1e-6))
    return main + smooth_lambda * smooth + power_lambda * power_diff


# =========================
# Model: Band mask enhancer
# =========================
class BandMaskEnhancer(nn.Module):
    def __init__(self, num_bands: int = 16, nfft: int = 256, hidden: int = 64):
        super().__init__()
        self.num_bands = num_bands
        self.nfft = nfft
        self.band_mlp = nn.Sequential(
            nn.Linear(num_bands * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_bands)
        )

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B,2,L]
        B, _, L = x.shape
        X = fft_with_window(x, nfft=self.nfft)  # [B,nfft]
        mag = X.abs()

        band_size = self.nfft // self.num_bands
        bands = mag.unfold(-1, band_size, band_size).mean(-1)  # [B, >=num_bands]
        if bands.shape[-1] > self.num_bands:
            bands = bands[:, :self.num_bands]

        feats = torch.cat([bands, torch.log(bands + 1e-8)], dim=-1)  # [B, 2*num_bands]
        logits = self.band_mlp(feats)  # [B,num_bands]

        if labels is not None:
            g_pos = 1.0 + F.softplus(logits)      # > 1
            g_neg = torch.sigmoid(logits)         # (0,1)
            gains = labels * g_pos + (1.0 - labels) * g_neg
        else:
            gains = 0.5 + 1.5 * torch.sigmoid(logits)  # (0.5, 2.0)

        mask = gains.repeat_interleave(band_size, dim=-1)[..., :self.nfft]  # [B,nfft]
        Y = mask * X
        y = ifft_with_trim(Y, L=L)  # [B,2,L]
        return y


# =========================
# Train / Eval
# =========================
def train_model(train_h5_path: str, val_h5_path: str, output_dir: str,
                epochs: int = 50, batch_size: int = 512, learning_rate: float = 1e-4,
                nfft: int = 256, snr_db: Optional[float] = None, phase_rotate: bool = True,
                cfo_ppm: Optional[float] = 5.0):
    os.makedirs(output_dir, exist_ok=True)
    device = get_device()

    train_dataset = H5IQSignalDataset(train_h5_path, target_snr_db=snr_db,
                                      phase_rotate=phase_rotate, cfo_ppm=cfo_ppm)
    val_dataset = H5IQSignalDataset(val_h5_path, target_snr_db=None,
                                    phase_rotate=False, cfo_ppm=None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = BandMaskEnhancer(num_bands=16, nfft=nfft, hidden=64).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float('inf')
    train_losses, val_losses = [] , []

    for epoch in range(epochs):
        # warmup first 5 epochs
        warmup_epochs = 5
        if epoch < warmup_epochs:
            lr_scale = min(1.0, float(epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = learning_rate * lr_scale

        model.train()
        epoch_train = 0.0
        with tqdm(train_loader, desc=f"Train {epoch+1}/{epochs}", unit="batch") as pbar:
            for signals, labels in pbar:
                signals = signals.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    enhanced = model(signals, labels)
                    loss = band_gain_margin_loss(enhanced, signals, labels, nfft=nfft)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                epoch_train += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_train = epoch_train / len(train_loader)
        train_losses.append(avg_train)

        # validation
        model.eval()
        epoch_val = 0.0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                enhanced = model(signals, labels)
                loss = band_gain_margin_loss(enhanced, signals, labels, nfft=nfft)
                epoch_val += loss.item()
        avg_val = epoch_val / len(val_loader)
        val_losses.append(avg_val)
        scheduler.step(avg_val)

        print(f"Epoch {epoch+1}/{epochs}  train={avg_train:.4f}  val={avg_val:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train,
                'val_loss': avg_val,
                'lr': optimizer.param_groups[0]['lr'],
            }, os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))

    return model


def test_model(model: nn.Module, test_h5_path: str, output_dir: str, nfft: int = 256):
    device = get_device()
    test_dataset = H5IQSignalDataset(test_h5_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)
    h5_dir = os.path.join(output_dir, 'h5_files')
    os.makedirs(h5_dir, exist_ok=True)

    results = []
    model.eval()
    with torch.no_grad():
        for i, (signals, labels) in enumerate(tqdm(test_loader, desc='Test')):
            signals = signals.to(device)
            labels_tensor = labels.to(device)
            labels_np = labels.cpu().numpy()[0]

            enhanced = model(signals, labels_tensor)

            sig_np = signals.squeeze(0).cpu().numpy()
            enh_np = enhanced.squeeze(0).cpu().numpy()

            # Save enhanced signal [2,32]
            base = f"sample_{i}"
            h5_path = os.path.join(h5_dir, f"{base}.h5")
            with h5py.File(h5_path, 'w') as hf:
                dset = hf.create_dataset('signal', (2, sig_np.shape[-1]), dtype='f4')
                dset[0] = enh_np[0]
                dset[1] = enh_np[1]

            # compute per-band PSDs
            def psd_bands_np(arr: np.ndarray) -> List[float]:
                t = torch.from_numpy(arr[None]).to(device)  # [1,2,L]
                bands = compute_channel_psd(t, num_channels=16, nfft=nfft, nperseg=arr.shape[-1])
                return bands.squeeze(0).detach().cpu().numpy().tolist()

            orig_psds = psd_bands_np(sig_np)
            enh_psds = psd_bands_np(enh_np)

            results.append({
                'index': i,
                'label': labels_np,
                'original_psds': orig_psds,
                'enhanced_psds': enh_psds,
                'h5_path': h5_path,
            })

    with open(os.path.join(output_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_h5', type=str, required=True)
    parser.add_argument('--val_h5', type=str, required=True)
    parser.add_argument('--test_h5', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--nfft', type=int, default=256)
    parser.add_argument('--snr_db', type=float, default=None)
    parser.add_argument('--phase_rotate', action='store_true')
    parser.add_argument('--cfo_ppm', type=float, default=5.0)
    args = parser.parse_args()

    set_seed(42)
    print('Device:', get_device())

    model = train_model(
        train_h5_path=args.train_h5,
        val_h5_path=args.val_h5,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        nfft=args.nfft,
        snr_db=args.snr_db,
        phase_rotate=args.phase_rotate,
        cfo_ppm=args.cfo_ppm,
    )

    _ = test_model(model, test_h5_path=args.test_h5, output_dir=args.out, nfft=args.nfft)


if __name__ == '__main__':
    main()

