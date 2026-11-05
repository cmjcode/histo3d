//! U-Net kecil untuk segmentasi semantik tile H&E. Hanya memakai operator
//! yang pasti didukung tract 0.23 saat diekspor (Conv, Relu, MaxPool,
//! ConvTranspose, Concat); BatchNorm dilipat ke Conv saat ekspor.
//!
//! Masukan `[B, 3, S, S]` f32 ternormalisasi (mean/std per kanal), keluaran
//! logit `[B, C, S, S]`. `S` harus kelipatan `2^depth`.

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig};
use burn::nn::pool::{MaxPool2d, MaxPool2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d, Relu};
use burn::tensor::{Device, Tensor};

/// Konfigurasi U-Net.
#[derive(Config, Debug)]
pub struct UNetConfig {
    /// Kanal masukan.
    #[config(default = 3)]
    pub in_channels: usize,
    /// Jumlah kelas keluaran.
    #[config(default = 3)]
    pub n_classes: usize,
    /// Kanal tingkat pertama; tingkat berikutnya ×2.
    #[config(default = 32)]
    pub base: usize,
    /// Jumlah tingkat penurunan (pool).
    #[config(default = 4)]
    pub depth: usize,
}

/// Dua Conv3×3 + BatchNorm + ReLU.
#[derive(Module, Debug)]
pub struct DoubleConv {
    /// Konvolusi pertama.
    pub conv1: Conv2d,
    /// Normalisasi pertama.
    pub bn1: BatchNorm,
    /// Konvolusi kedua.
    pub conv2: Conv2d,
    /// Normalisasi kedua.
    pub bn2: BatchNorm,
    /// Aktivasi.
    pub relu: Relu,
}

impl DoubleConv {
    fn new(cin: usize, cout: usize, device: &Device) -> Self {
        let conv = |ci: usize, co: usize| {
            Conv2dConfig::new([ci, co], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
                .init(device)
        };
        Self {
            conv1: conv(cin, cout),
            bn1: BatchNormConfig::new(cout).init(device),
            conv2: conv(cout, cout),
            bn2: BatchNormConfig::new(cout).init(device),
            relu: Relu::new(),
        }
    }

    /// Maju.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.relu.forward(self.bn1.forward(self.conv1.forward(x)));
        self.relu.forward(self.bn2.forward(self.conv2.forward(x)))
    }
}

/// U-Net.
#[derive(Module, Debug)]
pub struct UNet {
    /// Blok enkoder per tingkat (0 = resolusi penuh).
    pub enc: Vec<DoubleConv>,
    /// Pool 2×2.
    pub pool: MaxPool2d,
    /// Blok terdalam.
    pub bottleneck: DoubleConv,
    /// `up[i]`: kanal tingkat `i+1` → tingkat `i`, naik 2×.
    pub up: Vec<ConvTranspose2d>,
    /// Blok dekoder tingkat `i` (masukan: up ⊕ skip).
    pub dec: Vec<DoubleConv>,
    /// Conv1×1 ke logit kelas.
    pub head: Conv2d,
}

impl UNetConfig {
    /// Validasi konfigurasi.
    pub fn validate(&self) -> Result<(), String> {
        if self.in_channels == 0 || self.n_classes == 0 || self.base == 0 {
            return Err("in_channels, n_classes, base harus > 0".into());
        }
        if self.depth == 0 || self.depth > 6 {
            return Err("depth harus 1..=6".into());
        }
        Ok(())
    }

    /// Bangun model dengan bobot awal acak dari benih `device.seed`.
    pub fn init(&self, device: &Device) -> UNet {
        let ch: Vec<usize> = (0..=self.depth).map(|i| self.base << i).collect();
        let mut enc = Vec::with_capacity(self.depth);
        let mut cin = self.in_channels;
        for &c in ch.iter().take(self.depth) {
            enc.push(DoubleConv::new(cin, c, device));
            cin = c;
        }
        let bottleneck = DoubleConv::new(ch[self.depth - 1], ch[self.depth], device);
        let mut up = Vec::with_capacity(self.depth);
        let mut dec = Vec::with_capacity(self.depth);
        for i in 0..self.depth {
            up.push(
                ConvTranspose2dConfig::new([ch[i + 1], ch[i]], [2, 2])
                    .with_stride([2, 2])
                    .init(device),
            );
            dec.push(DoubleConv::new(ch[i] * 2, ch[i], device));
        }
        let head = Conv2dConfig::new([ch[0], self.n_classes], [1, 1]).init(device);
        UNet {
            enc,
            pool: MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            bottleneck,
            up,
            dec,
            head,
        }
    }
}

impl UNet {
    /// Jumlah tingkat.
    pub fn depth(&self) -> usize {
        self.enc.len()
    }

    /// Maju: `[B, 3, S, S]` → logit `[B, C, S, S]`.
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let mut skips = Vec::with_capacity(self.enc.len());
        let mut x = input;
        for e in &self.enc {
            x = e.forward(x);
            skips.push(x.clone());
            x = self.pool.forward(x);
        }
        x = self.bottleneck.forward(x);
        for i in (0..self.enc.len()).rev() {
            x = self.up[i].forward(x);
            x = Tensor::cat(vec![x, skips[i].clone()], 1);
            x = self.dec[i].forward(x);
        }
        self.head.forward(x)
    }
}
