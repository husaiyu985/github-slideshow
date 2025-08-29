import os
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import h5py
import scipy.signal as signalcomp


# 设置中文字体（可选）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# =========================
# 数据集（增加标准化与数据增强）
# =========================
class IQSignalDataset(Dataset):
    """IQ信号数据集类，增加了每样本标准化与可选增强（相位旋转、微频偏、加噪）。"""

    def __init__(
        self,
        data_dir: str,
        chunk_size: int = 128,
        stride: int = 12,
        samples_per_file: int = 10000,
        target_snr_db: float | None = 10,
        normalize: bool = True,
        augment: bool = False,
        freq_jitter_max_ppm: float = 30.0,
        snr_range_db: tuple[float, float] = (15.0, 35.0),
    ) -> None:
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.stride = stride
        self.samples_per_file = samples_per_file
        self.target_snr_db = target_snr_db
        self.normalize = normalize
        self.augment = augment
        self.freq_jitter_max_ppm = freq_jitter_max_ppm
        self.snr_range_db = snr_range_db

        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.bin')]
        self.samples: list[np.ndarray] = []
        self.labels: list[list[int]] = []
        self._preload_data()

    def _preload_data(self) -> None:
        print("预加载数据集...")
        for file_name in tqdm(self.file_list):
            # 从文件名解析标签 (例如 "0010_xxx.bin" -> [0,0,1,0])
            label_str = os.path.splitext(file_name)[0].split('_')[0]
            label = [int(bit) for bit in label_str]
            file_path = os.path.join(self.data_dir, file_name)

            with open(file_path, 'rb') as f:
                data = np.fromfile(f, dtype=np.complex64)

            real = np.real(data).astype(np.float32)
            imag = np.imag(data).astype(np.float32)
            iq_data = np.stack([real, imag], axis=1)  # [N, 2]

            num_chunks = max(0, (len(iq_data) - self.chunk_size) // self.stride + 1)
            if num_chunks <= 0:
                continue
            indices = np.random.choice(num_chunks, min(self.samples_per_file, num_chunks), replace=False)
            for idx in indices:
                start = idx * self.stride
                end = start + self.chunk_size
                chunk = iq_data[start:end]  # [chunk_size, 2]
                self.samples.append(chunk)
                self.labels.append(label)

        print(f"数据集加载完成，共 {len(self.samples)} 个样本")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _to_complex(x_2c: torch.Tensor) -> torch.Tensor:
        # x_2c: [2, L] -> complex [L]
        return torch.complex(x_2c[0], x_2c[1])

    @staticmethod
    def _from_complex(z: torch.Tensor) -> torch.Tensor:
        # z: complex [L] -> [2, L]
        return torch.stack([z.real, z.imag], dim=0)

    def _per_sample_normalize(self, x_2c: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        # x_2c: [2, L]
        z = self._to_complex(x_2c)
        z = z - z.mean()  # 去直流
        power = torch.mean(torch.abs(z) ** 2)
        scale = torch.sqrt(power + eps)
        z = z / scale
        return self._from_complex(z)

    def _apply_augment(self, x_2c: torch.Tensor) -> torch.Tensor:
        # 相位随机旋转
        L = x_2c.shape[-1]
        z = self._to_complex(x_2c)

        theta = torch.empty(1).uniform_(-math.pi, math.pi).item()
        z = z * torch.exp(1j * torch.tensor(theta))

        # 微小频偏 (ppm 相对带宽)
        if self.freq_jitter_max_ppm > 0:
            ppm = torch.empty(1).uniform_(-self.freq_jitter_max_ppm, self.freq_jitter_max_ppm).item()
            # 简化：把ppm映射为相位随时间线性增长的微偏移
            n = torch.arange(L, dtype=torch.float32)
            freq_off = ppm * 1e-6  # 归一化频偏
            z = z * torch.exp(1j * 2 * math.pi * freq_off * n)

        # 加性白噪声（随机SNR）
        if self.snr_range_db is not None:
            snr_db = float(torch.empty(1).uniform_(self.snr_range_db[0], self.snr_range_db[1]).item())
            snr_lin = 10 ** (snr_db / 10.0)
            sig_pow = torch.mean(torch.abs(z) ** 2).item()
            noise_pow = sig_pow / snr_lin
            noise = (np.random.randn(L) + 1j * np.random.randn(L)).astype(np.complex64)
            noise = torch.from_numpy(noise) * math.sqrt(noise_pow / 2.0)
            z = z + noise

        return self._from_complex(z)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]  # [L, 2]
        label = self.labels[idx]
        x = torch.tensor(sample, dtype=torch.float32).permute(1, 0)  # [2, L]

        if self.normalize:
            x = self._per_sample_normalize(x)
        if self.augment:
            x = self._apply_augment(x)

        return x, torch.tensor(label, dtype=torch.float32)


# =========================
# 频域基础模块
# =========================
class ComplexFFT1D(nn.Module):
    def __init__(self, norm: str = 'ortho') -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        return torch.fft.fft(x, norm=self.norm)


class ComplexIFFT1D(nn.Module):
    def __init__(self, norm: str = 'ortho') -> None:
        super().__init__()
        self.norm = norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        return torch.fft.ifft(x, norm=self.norm)


class FrequencyBandSplit1D(nn.Module):
    """一维频带分割，若不能整除则零填充到能整除"""

    def __init__(self, num_bands: int = 4) -> None:
        super().__init__()
        self.num_bands = num_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] 复数频谱
        B, C, L = x.shape
        if L % self.num_bands != 0:
            pad_len = self.num_bands - (L % self.num_bands)
            pad = torch.zeros(B, C, pad_len, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)
            L = x.shape[-1]
        band_size = L // self.num_bands
        bands = []
        for i in range(self.num_bands):
            start = i * band_size
            end = (i + 1) * band_size
            bands.append(x[:, :, start:end])
        return torch.cat(bands, dim=1)  # [B, C*num_bands, band_size]


# =========================
# 改进的频带注意力
# =========================
class SpectralBandingSelfAttention1D_v2(nn.Module):
    """
    频带注意力（保留实部与虚部的交互）并显式还原 band 拼接
    """

    def __init__(self, in_channels: int, num_heads: int = 4, num_bands: int = 2, reduce_channels: int = 16) -> None:
        super().__init__()
        assert in_channels > 0, "in_channels must > 0"
        assert reduce_channels % num_heads == 0, "reduce_channels must be divisible by num_heads"
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_bands = num_bands
        self.reduce_channels = reduce_channels
        self.head_dim = reduce_channels // num_heads

        self.band_split = FrequencyBandSplit1D(num_bands=num_bands)
        self.channel_reduce = nn.Conv1d(2 * in_channels * num_bands, reduce_channels, kernel_size=1)
        self.to_qkv = nn.Conv1d(reduce_channels, reduce_channels * 3, kernel_size=1)
        self.channel_expand = nn.Conv1d(reduce_channels, 2 * in_channels * num_bands, kernel_size=1)
        self.proj_out = nn.Conv1d(in_channels, in_channels, kernel_size=1)

    def _inverse_band_concat(self, banded_complex: torch.Tensor, original_len: int) -> torch.Tensor:
        """
        将 [B, C*num_bands, band_len] 的复数频谱恢复为 [B, C, L_pad]，再裁剪至 original_len。
        按照 band_split 的顺序：通道维按 [band0的C通道, band1的C通道, ...] 排列。
        还原时，将每个通道的 num_bands 个片段沿频率轴拼接回去。
        """
        B, CN, band_len = banded_complex.shape
        C = self.in_channels
        assert CN == C * self.num_bands, "通道数与期望不匹配"

        # 变形为 [B, num_bands, C, band_len]
        banded = banded_complex.view(B, self.num_bands, C, band_len)
        # 沿频率轴拼接 -> [B, C, num_bands * band_len]
        restored_full = banded.permute(0, 2, 1, 3).contiguous().view(B, C, self.num_bands * band_len)
        # 裁剪至原始长度
        return restored_full[:, :, :original_len]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        fft = ComplexFFT1D()(x)  # [B, C, L] complex
        banded = self.band_split(fft)  # [B, C*num_bands, band_len] complex

        real_imag = torch.cat([banded.real, banded.imag], dim=1)  # [B, 2*C*num_bands, T_band]
        reduced = self.channel_reduce(real_imag)
        qkv = self.to_qkv(reduced)
        q, k, v = qkv.chunk(3, dim=1)

        T = q.shape[-1]
        q = q.view(B, self.num_heads, self.head_dim, T)
        k = k.view(B, self.num_heads, self.head_dim, T)
        v = v.view(B, self.num_heads, self.head_dim, T)

        attn_scores = torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5)
        attn = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn, v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(B, self.reduce_channels, T)

        expanded = self.channel_expand(out)  # [B, 2*C*num_bands, band_len]
        band_len = banded.shape[-1]
        expanded = expanded[:, :, :band_len]
        mid = expanded.shape[1] // 2
        real_exp = expanded[:, :mid, :]
        imag_exp = expanded[:, mid:, :]
        complex_exp = torch.complex(real_exp, imag_exp)  # [B, C*num_bands, band_len]

        restored = self._inverse_band_concat(complex_exp, original_len=L)  # [B, C, L]
        ifft = ComplexIFFT1D()(restored)
        out_time_real = ifft.real
        out_proj = self.proj_out(out_time_real)
        return out_proj + x


# =========================
# 复数1x1卷积 & 频谱前馈
# =========================
class ComplexConv1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight_real = nn.Parameter(torch.randn(channels, channels, 1))
        self.weight_imag = nn.Parameter(torch.randn(channels, channels, 1))
        self.bias_real = nn.Parameter(torch.zeros(channels, 1))
        self.bias_imag = nn.Parameter(torch.zeros(channels, 1))
        nn.init.kaiming_normal_(self.weight_real)
        nn.init.kaiming_normal_(self.weight_imag)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_real = (
            F.conv1d(real, self.weight_real) -
            F.conv1d(imag, self.weight_imag) +
            self.bias_real
        )
        out_imag = (
            F.conv1d(real, self.weight_imag) +
            F.conv1d(imag, self.weight_real) +
            self.bias_imag
        )
        return out_real, out_imag


class SpectralEnhancedFFN(nn.Module):
    def __init__(self, in_channels: int, expansion: int = 2, dilation: int = 2) -> None:
        super().__init__()
        hidden_dim = in_channels * expansion
        self.expand_conv = nn.Conv1d(in_channels, hidden_dim, 1, bias=True)
        self.dw_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=True)
        self.dilated_dw_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation, groups=hidden_dim, bias=True)
        self.complex_conv1 = ComplexConv1d(hidden_dim)
        self.complex_conv2 = ComplexConv1d(hidden_dim)
        self.compress_conv = nn.Conv1d(hidden_dim, in_channels, 1, bias=True)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.expand_conv(x)
        x = self.act(x)
        branch1 = self.dw_conv(x)
        branch2 = self.dilated_dw_conv(x)
        spatial_out = branch1 + branch2
        x_freq = torch.fft.rfft(spatial_out, dim=-1)
        real, imag = x_freq.real, x_freq.imag
        real1, imag1 = self.complex_conv1(real, imag)
        real1, imag1 = self.act(real1), self.act(imag1)
        real2, imag2 = self.complex_conv2(real1, imag1)
        freq_out = torch.complex(real2, imag2)
        x_out = torch.fft.irfft(freq_out, n=spatial_out.size(-1), dim=-1)
        x_out = self.compress_conv(x_out)
        return x_out


# =========================
# Transformer块
# =========================
class SpectralDomainBlock(nn.Module):
    def __init__(self, in_channels: int, num_heads: int = 4, num_bands: int = 2, expansion: int = 2, reduce_channels: int = 16) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_channels)
        self.sbsa = nn.Sequential(
            SpectralBandingSelfAttention1D_v2(
                in_channels, num_heads=num_heads, num_bands=num_bands, reduce_channels=reduce_channels
            ),
            nn.Conv1d(in_channels, in_channels, 3, padding=1),
            nn.SiLU(),
        )
        self.norm2 = nn.LayerNorm(in_channels)
        self.seff = nn.Sequential(
            SpectralEnhancedFFN(in_channels, expansion=expansion),
            nn.Conv1d(in_channels, in_channels, 3, padding=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.permute(0, 2, 1)
        x_norm = self.norm1(x_perm).permute(0, 2, 1)
        x = x + self.sbsa(x_norm)
        x_perm = x.permute(0, 2, 1)
        x_norm = self.norm2(x_perm).permute(0, 2, 1)
        x = x + self.seff(x_norm)
        return x


# =========================
# 频域正则化模块（保持）
# =========================
class SpectralRegularization(nn.Module):
    def __init__(self, channels: int, window_size: int = 5, suppression_strength: float = 0.8, adaptive_threshold: bool = True) -> None:
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.suppression_strength = suppression_strength
        self.adaptive_threshold = adaptive_threshold

        self.freq_filter = nn.Sequential(
            nn.Conv1d(channels, channels, window_size, padding=window_size // 2, groups=channels),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
            nn.SiLU(),
        )

        self.leakage_detector = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 3, padding=1),
            nn.BatchNorm1d(channels // 2),
            nn.SiLU(),
            nn.Conv1d(channels // 2, channels, 1),
            nn.Sigmoid(),
        )

        self.signal_quality = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(channels // 4, channels, 1),
            nn.Sigmoid(),
        )

        if adaptive_threshold:
            self.threshold_learner = nn.Parameter(torch.tensor(0.1))
            self.threshold_momentum = nn.Parameter(torch.tensor(0.9))

        self.freq_enhancer = nn.Sequential(
            nn.Conv1d(channels, channels * 2, 1),
            nn.SiLU(),
            nn.Conv1d(channels * 2, channels, 1),
            nn.Tanh(),
        )

        self.output_proj = nn.Conv1d(channels, channels, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def detect_spectral_leakage(self, psd: torch.Tensor, adaptive_threshold: bool = True) -> torch.Tensor:
        B, C, L = psd.shape
        if adaptive_threshold:
            kernel_size = min(11, L // 4)
            if kernel_size % 2 == 0:
                kernel_size += 1
            padding = kernel_size // 2
            kernel = torch.ones(1, 1, kernel_size, device=psd.device) / kernel_size
            local_mean = F.conv1d(psd, kernel, padding=padding, groups=1)
            local_var = F.conv1d(psd ** 2, kernel, padding=padding, groups=1) - local_mean ** 2
            local_std = torch.sqrt(torch.clamp(local_var, min=1e-8))
            k_factor = torch.clamp(self.threshold_learner, min=0.01, max=5.0)
            threshold = local_mean + k_factor * local_std
        else:
            global_max = torch.max(psd, dim=-1, keepdim=True)[0]
            threshold = 0.1 * global_max

        leakage_mask = (psd > threshold).float()
        kernel = torch.ones(1, 1, 3, device=psd.device)
        dilated_mask = F.conv1d(leakage_mask, kernel, padding=1, groups=1)
        dilated_mask = (dilated_mask > 0).float()
        eroded_mask = F.conv1d(dilated_mask, kernel, padding=1, groups=1)
        eroded_mask = (eroded_mask > 0).float()
        return eroded_mask

    def apply_spectral_suppression(self, psd: torch.Tensor, leakage_mask: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        B, C, L = psd.shape
        suppression_factor = 1.0 - self.suppression_strength
        suppressed_psd = psd * (1 - leakage_mask * suppression_factor)
        filtered_psd = self.freq_filter(psd)
        alpha = leakage_mask * 0.7
        final_psd = (1 - alpha) * suppressed_psd + alpha * filtered_psd
        final_psd = torch.clamp(final_psd, min=1e-12)
        return final_psd

    def enhance_signal_quality(self, psd: torch.Tensor, signal_mask: torch.Tensor) -> torch.Tensor:
        quality_score = self.signal_quality(psd)
        enhanced_features = self.freq_enhancer(psd)
        enhanced_psd = psd + signal_mask * enhanced_features * 0.3 * quality_score
        return enhanced_psd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        fft_data = torch.fft.fft(x, dim=-1)
        psd = torch.abs(fft_data) ** 2 / max(1, L)
        phase = torch.angle(fft_data)
        psd = torch.fft.fftshift(psd, dim=-1)
        phase = torch.fft.fftshift(phase, dim=-1)
        leakage_mask = self.detect_spectral_leakage(psd, self.adaptive_threshold)
        suppressed_psd = self.apply_spectral_suppression(psd, leakage_mask, phase)
        signal_mask = 1.0 - leakage_mask
        enhanced_psd = self.enhance_signal_quality(suppressed_psd, signal_mask)
        enhanced_psd = torch.fft.ifftshift(enhanced_psd, dim=-1)
        phase = torch.fft.ifftshift(phase, dim=-1)
        enhanced_complex = torch.sqrt(enhanced_psd + 1e-12) * torch.exp(1j * phase)
        enhanced_signal = torch.fft.ifft(enhanced_complex, dim=-1)
        output = self.output_proj(enhanced_signal.real)
        return output + x


# =========================
# 主模型（修复freq_enhancement通道错误 + 残差缩放门）
# =========================
class EnhancedSDTBModel(nn.Module):
    def __init__(
        self,
        input_size: int = 128,
        num_blocks: int = 6,
        num_heads: int = 8,
        num_bands: int = 4,
        expansion: int = 2,
        reduce_channels: int = 32,
        use_spectral_reg: bool = True,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_blocks = num_blocks
        self.use_spectral_reg = use_spectral_reg

        self.init_conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
        )

        if use_spectral_reg:
            self.spectral_reg = SpectralRegularization(channels=64, suppression_strength=0.8, adaptive_threshold=True)

        self.transformer_blocks = nn.ModuleList([
            SpectralDomainBlock(
                in_channels=64,
                num_heads=num_heads,
                num_bands=num_bands,
                expansion=expansion,
                reduce_channels=reduce_channels,
            )
            for _ in range(num_blocks)
        ])

        # 修复第二个 conv 的通道数 (128 -> 64)
        self.freq_enhancement = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Conv1d(128, 64, 1),
            nn.SiLU(),
        )

        self.output_conv = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(16, 2, kernel_size=3, padding=1),
        )

        # 残差缩放门，初始较小
        self.res_scale = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, iq_data: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(iq_data)
        if self.use_spectral_reg:
            x = self.spectral_reg(x)
        residual = x
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if (i + 1) % 3 == 0:
                x = x + residual
                residual = x
        x = x + self.freq_enhancement(x)
        enhanced = self.output_conv(x)
        gate = torch.tanh(self.res_scale)
        return iq_data + gate * enhanced

    def get_adaptive_weights(self) -> dict:
        return {
            'leakage_weight': torch.tensor(1.0, device=self.res_scale.device),
            'enhancement_weight': torch.tensor(1.0, device=self.res_scale.device),
        }


# =========================
# 工具：PSD计算
# =========================
def compute_psd(signal: np.ndarray, fs: float = 20e6, nperseg: int = 128):
    freqs, psd = signalcomp.welch(
        signal, fs=fs, nperseg=nperseg, noverlap=nperseg - 12, return_onesided=False, scaling='density'
    )
    psd = np.fft.fftshift(psd)
    freqs = np.fft.fftshift(freqs)
    return freqs, psd


def compute_channel_psd(tensor: torch.Tensor, num_channels: int = 4, fs: float = 20e6, nperseg: int = 128) -> torch.Tensor:
    batch_size, _, seq_len = tensor.shape
    avg_psd = torch.zeros(batch_size, num_channels, device=tensor.device)
    for b in range(batch_size):
        real = tensor[b, 0].to(torch.float32)
        imag = tensor[b, 1].to(torch.float32)
        complex_signal = torch.complex(real, imag)
        fft_data = torch.fft.fft(complex_signal, n=nperseg)
        fft_data = torch.fft.fftshift(fft_data)
        psd = torch.abs(fft_data) ** 2 / nperseg
        splits = torch.chunk(psd, num_channels, dim=0)
        for j in range(num_channels):
            avg_psd[b, j] = torch.mean(splits[j])
    return avg_psd


# =========================
# 损失：改进PSD + 带间距边际(dB)
# =========================
def enhanced_psd_loss_v3(
    enhanced_signals: torch.Tensor,
    original_signals: torch.Tensor,
    labels: torch.Tensor,
    enhance_factor: float = 1.5,
    suppress_factor: float = 0.1,
    fidelity_weight: float = 0.5,
    eps: float = 1e-12,
) -> torch.Tensor:
    B, C, L = enhanced_signals.shape
    num_channels = labels.shape[1]

    def calc_psd(x: torch.Tensor) -> torch.Tensor:
        psd_list: list[torch.Tensor] = []
        for b in range(B):
            real = x[b, 0].float()
            imag = x[b, 1].float()
            sig = torch.complex(real, imag)
            fft_sig = torch.fft.fft(sig)
            fft_sig = torch.fft.fftshift(fft_sig)
            psd = (torch.abs(fft_sig) ** 2) / max(1, L)
            chunks = torch.chunk(psd, num_channels, dim=0)
            psd_list.append(torch.stack(chunks))
        return torch.stack(psd_list) + eps  # [B, num_channels, ~]

    enhanced_psd = calc_psd(enhanced_signals)
    original_psd = calc_psd(original_signals)
    labels_exp = labels.unsqueeze(-1)
    target_psd = labels_exp * (original_psd * enhance_factor) + (1.0 - labels_exp) * (original_psd * suppress_factor)
    psd_loss = F.mse_loss(enhanced_psd, target_psd)
    fidelity = F.mse_loss(enhanced_signals, original_signals) + F.l1_loss(enhanced_signals, original_signals)
    total_loss = psd_loss + fidelity_weight * fidelity
    return total_loss


def band_margin_loss(
    predicted_signal: torch.Tensor,
    original_signal: torch.Tensor,
    labels: torch.Tensor,
    margin_pos_db: float = 2.0,
    margin_neg_db: float = 2.0,
    num_bands: int = 4,
    eps: float = 1e-12,
) -> torch.Tensor:
    """在每个频带上以dB为单位设置间距边际，增强正类、压制负类。"""
    def band_avg_psd_db(x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        out = []
        for b in range(B):
            z = torch.complex(x[b, 0].float(), x[b, 1].float())
            psd = torch.abs(torch.fft.fftshift(torch.fft.fft(z))) ** 2 / max(1, L)
            bands = torch.chunk(psd, num_bands, dim=0)
            band_means = torch.stack([torch.mean(band) for band in bands])  # [num_bands]
            out.append(10.0 * torch.log10(band_means + eps))
        return torch.stack(out)  # [B, num_bands]

    pred_db = band_avg_psd_db(predicted_signal)
    orig_db = band_avg_psd_db(original_signal)

    labels = labels.float()
    pos = labels
    neg = 1.0 - labels

    # hinge 风格：正类希望 >= 原始 + margin_pos_db
    loss_pos = F.relu((orig_db + margin_pos_db) - pred_db) * pos
    # 负类希望 <= 原始 - margin_neg_db
    loss_neg = F.relu(pred_db - (orig_db - margin_neg_db)) * neg
    loss = (loss_pos + loss_neg).mean()
    return loss


# =========================
# 训练：Cosine Warmup LR + EMA
# =========================
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, device: str | torch.device | None = None) -> None:
        self.ema = self._clone_model(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        if device is not None:
            self.ema.to(device)

    @staticmethod
    def _clone_model(model: nn.Module) -> nn.Module:
        import copy
        ema = copy.deepcopy(model)
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            if esd[k].dtype.is_floating_point:
                esd[k].mul_(d).add_(msd[k], alpha=(1.0 - d))
            else:
                esd[k] = msd[k]


def build_cosine_warmup_scheduler(optimizer: optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model_enhanced(
    data_dir: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
) -> nn.Module:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 单独构造训练/验证数据集，以便启用/关闭增强
    train_dataset = IQSignalDataset(data_dir, chunk_size=128, stride=12, samples_per_file=5000, normalize=True, augment=True)
    val_dataset = IQSignalDataset(data_dir, chunk_size=128, stride=12, samples_per_file=2000, normalize=True, augment=False)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("数据集为空，请检查 data_dir 是否正确，或bin文件是否有效。")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = EnhancedSDTBModel(
        input_size=128,
        num_blocks=4,
        num_heads=4,
        num_bands=4,
        expansion=2,
        reduce_channels=32,
        use_spectral_reg=True,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    total_steps = epochs * max(1, len(train_dataloader))
    warmup_steps = max(10, int(0.05 * total_steps))
    scheduler = build_cosine_warmup_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    ema = ModelEMA(model, decay=0.999, device=device)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        with tqdm(train_dataloader, desc=f"训练 Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
            for signals, labels in pbar:
                signals = signals.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    enhanced_signals = model(signals, labels)
                    loss_psd = enhanced_psd_loss_v3(
                        enhanced_signals, signals, labels, enhance_factor=1.5, suppress_factor=0.1, fidelity_weight=0.4
                    )
                    loss_margin = band_margin_loss(enhanced_signals, signals, labels, margin_pos_db=2.0, margin_neg_db=2.0, num_bands=labels.shape[1])
                    loss = loss_psd + 0.5 * loss_margin

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                # EMA 与调度
                ema.update(model)
                scheduler.step()
                global_step += 1

                epoch_train_loss += loss.item()
                pbar.set_postfix(loss=float(loss.item()), lr=optimizer.param_groups[0]['lr'])

        avg_train_loss = epoch_train_loss / max(1, len(train_dataloader))
        train_losses.append(avg_train_loss)

        # 验证（使用EMA权重）
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for signals, labels in val_dataloader:
                signals = signals.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                enhanced_signals = ema.ema(signals, labels)
                loss_psd = enhanced_psd_loss_v3(
                    enhanced_signals, signals, labels, enhance_factor=1.5, suppress_factor=0.1, fidelity_weight=0.4
                )
                loss_margin = band_margin_loss(enhanced_signals, signals, labels, margin_pos_db=2.0, margin_neg_db=2.0, num_bands=labels.shape[1])
                loss = loss_psd + 0.5 * loss_margin
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / max(1, len(val_dataloader))
        val_losses.append(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{epochs}, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, LR: {current_lr:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, "best_model.pth")
            torch.save(ema.ema.state_dict(), best_model_path)
            print(f"最佳模型已保存(EMA)，验证损失: {best_val_loss:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_state_dict': ema.ema.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'lr': current_lr,
                },
                checkpoint_path,
            )

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失曲线')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_validation_loss.png'))
    plt.close()

    return ema.ema


# =========================
# 测试与可视化
# =========================
def test_model(model: nn.Module, data_dir: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    test_dataset = IQSignalDataset(data_dir, chunk_size=128, stride=128, samples_per_file=50, target_snr_db=None, normalize=True, augment=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    os.makedirs(output_dir, exist_ok=True)
    enhanced_dir = os.path.join(output_dir, "enhanced_signals")
    os.makedirs(enhanced_dir, exist_ok=True)
    h5_dir = os.path.join(output_dir, "h5_files")
    os.makedirs(h5_dir, exist_ok=True)

    results = []
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for i, (signals, labels) in enumerate(tqdm(test_dataloader, desc="测试模型")):
            signals = signals.to(device)
            labels_tensor = labels.to(device)
            labels_np = labels.cpu().numpy()[0]

            enhanced_signals = model(signals, labels_tensor)
            signals_np = signals.squeeze(0).cpu().numpy().T
            enhanced_np = enhanced_signals.squeeze(0).cpu().numpy().T

            orig_complex = signals_np[:, 0] + 1j * signals_np[:, 1]
            enhanced_complex = enhanced_np[:, 0] + 1j * enhanced_np[:, 1]

            orig_freqs, orig_psd = compute_psd(orig_complex)
            enhanced_freqs, enhanced_psd = compute_psd(enhanced_complex)

            num_channels = 4
            band_size = len(orig_psd) // num_channels
            orig_psds, enhanced_psds = [], []
            for j in range(num_channels):
                start = j * band_size
                end = (j + 1) * band_size
                orig_psds.append(np.mean(orig_psd[start:end]))
                enhanced_psds.append(np.mean(enhanced_psd[start:end]))

            base_name = f"sample_{i}_label_{''.join(str(int(x)) for x in labels_np)}"
            h5_path = os.path.join(h5_dir, f"{base_name}.h5")
            with h5py.File(h5_path, 'w') as hf:
                dset = hf.create_dataset("signal", (128, 2), dtype='f4')
                dset[:, 0] = enhanced_np[:, 0]
                dset[:, 1] = enhanced_np[:, 1]

            results.append(
                {
                    'index': i,
                    'label': labels_np,
                    'original_psds': orig_psds,
                    'enhanced_psds': enhanced_psds,
                    'h5_path': h5_path,
                }
            )

            if i % 100 == 0:
                visualize_results(
                    i,
                    signals_np,
                    enhanced_np,
                    labels_np,
                    orig_psds,
                    enhanced_psds,
                    orig_psd,
                    enhanced_psd,
                    output_dir,
                )

    results_path = os.path.join(output_dir, "test_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"测试结果已保存至: {results_path}")
    print(f"HDF5文件已保存至: {h5_dir}")
    return results


def visualize_results(
    idx: int,
    orig_signal: np.ndarray,
    enhanced_signal: np.ndarray,
    label: np.ndarray,
    orig_psds: list[float],
    enhanced_psds: list[float],
    orig_psd_full: np.ndarray,
    enhanced_psd_full: np.ndarray,
    output_dir: str,
):
    fig, axes = plt.subplots(4, 1, figsize=(15, 24))
    time = np.arange(128) / 20e6 * 1e6  # μs

    ax1 = axes[0]
    ax1.plot(time, orig_signal[:, 0], label='实部')
    ax1.plot(time, orig_signal[:, 1], label='虚部')
    ax1.set_xlabel('时间 (μs)')
    ax1.set_ylabel('幅度')
    ax1.set_title(f'原始信号时域波形 (标签: {label})')
    ax1.legend()
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(time, enhanced_signal[:, 0], label='实部')
    ax2.plot(time, enhanced_signal[:, 1], label='虚部')
    ax2.set_xlabel('时间 (μs)')
    ax2.set_ylabel('幅度')
    ax2.set_title('增强信号时域波形')
    ax2.legend()
    ax2.grid(True)

    ax3 = axes[2]
    channels = range(1, 5)
    width = 0.35
    ax3.bar(np.array(list(channels)) - width / 2, orig_psds, width, label='原始PSD')
    ax3.bar(np.array(list(channels)) + width / 2, enhanced_psds, width, label='处理后PSD')
    ax3.set_xlabel('信道编号')
    ax3.set_ylabel('平均PSD (线性值)')
    ax3.set_title('信道平均PSD对比')
    ax3.set_xticks(list(channels))
    ax3.legend()
    ax3.grid(True)

    ax4 = axes[3]
    freqs = np.linspace(-10, 10, len(orig_psd_full))  # MHz
    ax4.plot(freqs, 10 * np.log10(orig_psd_full + 1e-12), label='原始信号')
    ax4.plot(freqs, 10 * np.log10(enhanced_psd_full + 1e-12), label='增强信号')
    for i in range(4):
        start_freq = -10 + i * 5
        end_freq = -10 + (i + 1) * 5
        color = 'green' if label[i] == 1 else 'red'
        ax4.axvspan(start_freq, end_freq, alpha=0.2, color=color)
        ax4.text((start_freq + end_freq) / 2, ax4.get_ylim()[1] * 0.9, f"信道 {i + 1}\n({label[i]})", ha='center', fontsize=10)
    ax4.set_xlabel('频率 (MHz)')
    ax4.set_ylabel('PSD (dB)')
    ax4.set_title('频谱对比 (绿色:需增强, 红色:需抑制)')
    ax4.legend()
    ax4.grid(True)
    ax4.set_xlim([-10, 10])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'result_{idx}.png'), dpi=150)
    plt.close(fig)


# =========================
# 抽样处理
# =========================
def process_sampled_files(model: nn.Module, data_dir: str, output_dir: str, samples_per_channel: int = 500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.bin')]

    for file_name in tqdm(file_list, desc="处理采样文件"):
        file_path = os.path.join(data_dir, file_name)
        label_str = os.path.splitext(file_name)[0].split('_')[0]
        label = [int(bit) for bit in label_str]
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0).to(device)

        file_size = os.path.getsize(file_path)
        num_samples = file_size // 8
        chunk_size = 128
        stride = 12
        num_chunks = max(0, (num_samples - chunk_size) // stride + 1)
        if num_chunks <= 0:
            print(f"{file_name} 不足一个chunk，跳过。")
            continue

        selected_indices = np.random.choice(num_chunks, min(samples_per_channel * 2, num_chunks), replace=False)

        output_file = os.path.join(output_dir, f"enhanced_{file_name}")
        with open(output_file, 'wb') as out_f:
            for chunk_idx in tqdm(selected_indices, desc=f"处理 {file_name}", leave=False):
                start_idx = chunk_idx * stride
                end_idx = start_idx + chunk_size
                with open(file_path, 'rb') as in_f:
                    in_f.seek(start_idx * 8)
                    data = np.fromfile(in_f, dtype=np.complex64, count=chunk_size)
                real_part = np.real(data).astype(np.float32)
                imag_part = np.imag(data).astype(np.float32)
                iq = np.stack([real_part, imag_part], axis=0)
                # 标准化
                z = (iq[0] + 1j * iq[1]).astype(np.complex64)
                z = z - np.mean(z)
                p = np.mean(np.abs(z) ** 2) + 1e-6
                z = z / np.sqrt(p)
                iq_norm = np.stack([np.real(z).astype(np.float32), np.imag(z).astype(np.float32)], axis=0)

                iq_tensor = torch.tensor(iq_norm, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    enhanced_tensor = model(iq_tensor, label_tensor)
                enhanced_np = enhanced_tensor.squeeze(0).cpu().numpy().T
                complex_signal = enhanced_np[:, 0] + 1j * enhanced_np[:, 1]
                complex_signal.astype(np.complex64).tofile(out_f)
        print(f"文件处理完成: {file_name} -> enhanced_{file_name} (样本数: {len(selected_indices)})")


# =========================
# 主入口（参数）
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='包含 .bin 的数据目录')
    parser.add_argument('--output_dir', type=str, default='/workspace/outputs', help='输出目录')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    print("开始训练模型...")
    model = train_model_enhanced(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print("\n测试模型性能...")
    test_results = test_model(model, args.data_dir, args.output_dir)
    print("\n处理采样文件...")
    process_sampled_files(model, args.data_dir, args.output_dir, samples_per_channel=500)
    print("\n所有处理完成!")


if __name__ == '__main__':
    main()

