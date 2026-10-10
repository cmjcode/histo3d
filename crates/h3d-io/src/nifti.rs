//! Ekspor NIfTI-1 (`.nii.gz`) untuk validasi silang di 3D Slicer (8.7.3).
//!
//! Header 348 byte ditulis tangan (lebih sederhana daripada API penulis
//! crate `nifti`, dan tanpa dependensi tambahan di pohon rilis), lalu
//! dimampatkan gzip dengan `flate2`.
//!
//! - Volume label: `u8` pada kisi tumpukan, `pixdim` = (`s_xy`, `s_xy`,
//!   `s_z`) dalam **mm**.
//! - Volume atribut dan medan jarak: `f32` (`NaN` = NODATA).
//! - NIfTI butuh `s_z` seragam: bila jarak lapis tidak rata, volume
//!   diresampel ke `s_z` median dan catatannya dikembalikan untuk laporan.
//! - Orientasi: `sform` diagonal dengan sumbu y dibalik (ruang W tangan-kiri
//!   → dunia tangan-kanan, sama dengan konvensi render 5.2).

use std::io::{Read, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use h3d_volume::{resample_uniform_z, AttrVolume, StackGrid, VoxelMask};

use crate::error::IoError;

/// Panjang header NIfTI-1.
pub const NIFTI_HEADER_LEN: usize = 348;
/// Offset data voxel (header + 4 byte ekstensi kosong).
pub const NIFTI_VOX_OFFSET: usize = 352;
/// Kode tipe data `uint8`.
pub const DT_UINT8: i16 = 2;
/// Kode tipe data `float32`.
pub const DT_FLOAT32: i16 = 16;
/// Toleransi keseragaman jarak lapis (µm).
pub const UNIFORM_Z_TOL_UM: f64 = 1e-6;

/// Berkas NIfTI siap tulis beserta catatan untuk laporan.
#[derive(Clone, Debug, PartialEq)]
pub struct NiftiOutput {
    /// Isi `.nii.gz`.
    pub gz: Vec<u8>,
    /// Catatan (mis. resampling ke `s_z` median), bahasa Indonesia.
    pub notes: Vec<String>,
}

/// Field header yang relevan, hasil baca balik.
#[derive(Clone, Debug, PartialEq)]
pub struct NiftiHeader {
    /// `dim[0..8]`.
    pub dim: [i16; 8],
    /// `pixdim[0..8]`.
    pub pixdim: [f32; 8],
    /// Kode tipe data.
    pub datatype: i16,
    /// Bit per voxel.
    pub bitpix: i16,
    /// Offset data.
    pub vox_offset: f32,
    /// `xyzt_units`.
    pub xyzt_units: u8,
    /// `sform_code`.
    pub sform_code: i16,
    /// Baris affine `srow_x`, `srow_y`, `srow_z`.
    pub srow: [[f32; 4]; 3],
}

fn resample_note(dz: f64) -> String {
    format!(
        "Jarak irisan tidak seragam; volume NIfTI diresampel ke s_z median {dz:.3} µm \
         (tetangga terdekat)."
    )
}

fn check_grid(grid: &StackGrid, len: usize) -> Result<(), IoError> {
    let [nx, ny, nz] = grid.dims();
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(IoError::InvalidArgument("kisi volume kosong".into()));
    }
    if nx > i16::MAX as usize || ny > i16::MAX as usize || nz > i16::MAX as usize {
        return Err(IoError::InvalidArgument(
            "kisi terlalu besar untuk NIfTI-1 (dimensi > 32767)".into(),
        ));
    }
    if grid.voxel_count() != len {
        return Err(IoError::InvalidArgument(format!(
            "panjang data {len} tidak cocok dengan kisi {nx}×{ny}×{nz}"
        )));
    }
    if !(grid.s_xy.is_finite() && grid.s_xy > 0.0) {
        return Err(IoError::InvalidArgument("s_xy kisi tidak sah".into()));
    }
    Ok(())
}

/// Header + data mentah (`.nii` tak termampat) untuk kisi ber-`s_z` seragam.
fn encode_raw(grid: &StackGrid, datatype: i16, bitpix: i16, data: &[u8]) -> Vec<u8> {
    let [nx, ny, nz] = grid.dims();
    let sx = (grid.s_xy / 1000.0) as f32;
    let sz = (grid.median_dz() / 1000.0) as f32;
    let ox = ((grid.origin_um[0] + 0.5 * grid.s_xy) / 1000.0) as f32;
    let oy = (-(grid.origin_um[1] + 0.5 * grid.s_xy) / 1000.0) as f32;
    let oz = (grid.z_um.first().copied().unwrap_or(0.0) / 1000.0) as f32;

    let mut h = vec![0u8; NIFTI_VOX_OFFSET];
    let put_i32 = |h: &mut [u8], o: usize, v: i32| h[o..o + 4].copy_from_slice(&v.to_le_bytes());
    let put_i16 = |h: &mut [u8], o: usize, v: i16| h[o..o + 2].copy_from_slice(&v.to_le_bytes());
    let put_f32 = |h: &mut [u8], o: usize, v: f32| h[o..o + 4].copy_from_slice(&v.to_le_bytes());

    put_i32(&mut h, 0, NIFTI_HEADER_LEN as i32);
    h[38] = b'r';
    let dim: [i16; 8] = [3, nx as i16, ny as i16, nz as i16, 1, 1, 1, 1];
    for (i, d) in dim.iter().enumerate() {
        put_i16(&mut h, 40 + i * 2, *d);
    }
    put_i16(&mut h, 70, datatype);
    put_i16(&mut h, 72, bitpix);
    let pixdim: [f32; 8] = [1.0, sx, sx, sz, 0.0, 0.0, 0.0, 0.0];
    for (i, p) in pixdim.iter().enumerate() {
        put_f32(&mut h, 76 + i * 4, *p);
    }
    put_f32(&mut h, 108, NIFTI_VOX_OFFSET as f32);
    put_f32(&mut h, 112, 1.0); // scl_slope
    put_f32(&mut h, 116, 0.0); // scl_inter
    h[123] = 2; // xyzt_units: NIFTI_UNITS_MM
    let descrip = b"HISTO3D; hanya untuk riset";
    h[148..148 + descrip.len()].copy_from_slice(descrip);
    put_i16(&mut h, 252, 0); // qform_code: tidak dipakai
    put_i16(&mut h, 254, 1); // sform_code: NIFTI_XFORM_SCANNER_ANAT
    let srow: [[f32; 4]; 3] = [[sx, 0.0, 0.0, ox], [0.0, -sx, 0.0, oy], [0.0, 0.0, sz, oz]];
    for (r, row) in srow.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            put_f32(&mut h, 280 + r * 16 + c * 4, *v);
        }
    }
    h[344..348].copy_from_slice(b"n+1\0");
    h.extend_from_slice(data);
    h
}

fn gzip(raw: &[u8]) -> Result<Vec<u8>, IoError> {
    let p = std::path::Path::new("volume.nii.gz");
    // `GzEncoder::new` menulis mtime = 0 sehingga keluaran deterministik.
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(raw).map_err(|e| IoError::io(p, e))?;
    enc.finish().map_err(|e| IoError::io(p, e))
}

/// Volume label `u8`: nilai voxel = id label. `masks` = `(id label, masker)`
/// pada kisi yang sama; bila tumpang tindih, entri yang lebih akhir menang.
pub fn encode_label_nifti(masks: &[(u8, &VoxelMask)]) -> Result<NiftiOutput, IoError> {
    let Some((_, first)) = masks.first() else {
        return Err(IoError::InvalidArgument(
            "tidak ada masker voxel untuk ditulis".into(),
        ));
    };
    for (_, m) in masks {
        if m.grid != first.grid {
            return Err(IoError::InvalidArgument(
                "masker voxel berada pada kisi berbeda".into(),
            ));
        }
        check_grid(&m.grid, m.data.len())?;
    }
    let mut notes = Vec::new();
    let uniform = first.grid.is_uniform_z(UNIFORM_Z_TOL_UM);
    let resampled: Vec<(u8, VoxelMask)> = if uniform {
        Vec::new()
    } else {
        masks
            .iter()
            .map(|(id, m)| (*id, resample_uniform_z(m)))
            .collect()
    };
    let view: Vec<(u8, &VoxelMask)> = if uniform {
        masks.to_vec()
    } else {
        resampled.iter().map(|(id, m)| (*id, m)).collect()
    };
    let grid = &view[0].1.grid;
    if !uniform {
        notes.push(resample_note(grid.median_dz()));
    }
    let mut data = vec![0u8; grid.voxel_count()];
    for (id, m) in &view {
        check_grid(&m.grid, m.data.len())?;
        for (d, &v) in data.iter_mut().zip(m.data.iter()) {
            if v != 0 {
                *d = *id;
            }
        }
    }
    let raw = encode_raw(grid, DT_UINT8, 8, &data);
    Ok(NiftiOutput {
        gz: gzip(&raw)?,
        notes,
    })
}

/// Volume `f32` (atribut atau medan jarak) pada `grid`; `NaN` = NODATA.
pub fn encode_f32_nifti(grid: &StackGrid, values: &[f32]) -> Result<NiftiOutput, IoError> {
    check_grid(grid, values.len())?;
    let mut notes = Vec::new();
    let owned;
    let (grid, values) = if grid.is_uniform_z(UNIFORM_Z_TOL_UM) {
        (grid, values)
    } else {
        owned = AttrVolume {
            grid: grid.clone(),
            data: values.to_vec(),
        }
        .resample_uniform_z();
        notes.push(resample_note(owned.grid.median_dz()));
        (&owned.grid, owned.data.as_slice())
    };
    check_grid(grid, values.len())?;
    let mut data = Vec::with_capacity(values.len() * 4);
    for v in values {
        data.extend_from_slice(&v.to_le_bytes());
    }
    let raw = encode_raw(grid, DT_FLOAT32, 32, &data);
    Ok(NiftiOutput {
        gz: gzip(&raw)?,
        notes,
    })
}

/// Buka gzip `.nii.gz` menjadi byte `.nii`.
pub fn gunzip(gz: &[u8]) -> Result<Vec<u8>, IoError> {
    let p = std::path::Path::new("volume.nii.gz");
    let mut out = Vec::new();
    GzDecoder::new(gz)
        .read_to_end(&mut out)
        .map_err(|e| IoError::io(p, e))?;
    Ok(out)
}

/// Baca balik header dan data voxel dari byte `.nii` (tak termampat).
/// Berkas terpotong atau magic salah → galat.
pub fn decode_nifti(raw: &[u8]) -> Result<(NiftiHeader, &[u8]), IoError> {
    let p = std::path::Path::new("volume.nii");
    if raw.len() < NIFTI_VOX_OFFSET {
        return Err(IoError::invalid(p, "berkas terpotong"));
    }
    let i32_at = |o: usize| i32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);
    let i16_at = |o: usize| i16::from_le_bytes([raw[o], raw[o + 1]]);
    let f32_at = |o: usize| f32::from_le_bytes([raw[o], raw[o + 1], raw[o + 2], raw[o + 3]]);
    if i32_at(0) != NIFTI_HEADER_LEN as i32 || &raw[344..348] != b"n+1\0" {
        return Err(IoError::invalid(p, "bukan NIfTI-1 berkas tunggal"));
    }
    let mut dim = [0i16; 8];
    for (i, d) in dim.iter_mut().enumerate() {
        *d = i16_at(40 + i * 2);
    }
    let mut pixdim = [0f32; 8];
    for (i, v) in pixdim.iter_mut().enumerate() {
        *v = f32_at(76 + i * 4);
    }
    let mut srow = [[0f32; 4]; 3];
    for (r, row) in srow.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = f32_at(280 + r * 16 + c * 4);
        }
    }
    let header = NiftiHeader {
        dim,
        pixdim,
        datatype: i16_at(70),
        bitpix: i16_at(72),
        vox_offset: f32_at(108),
        xyzt_units: raw[123],
        sform_code: i16_at(254),
        srow,
    };
    let n_vox: usize = dim[1..=3]
        .iter()
        .map(|d| usize::try_from(*d).unwrap_or(0))
        .product();
    let bytes = n_vox
        .checked_mul(usize::try_from(header.bitpix / 8).unwrap_or(0))
        .ok_or_else(|| IoError::invalid(p, "ukuran volume tidak masuk akal"))?;
    let data = raw
        .get(NIFTI_VOX_OFFSET..NIFTI_VOX_OFFSET + bytes)
        .ok_or_else(|| IoError::invalid(p, "data voxel terpotong"))?;
    Ok((header, data))
}
