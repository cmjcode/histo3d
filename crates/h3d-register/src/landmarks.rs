//! Affine dari landmark manual (histo3d.md 8.4.5).

use h3d_core::Affine2;

use crate::error::RegError;

/// Pasangan `(moving, fixed)` dalam µm.
type PointPairs = Vec<([f64; 2], [f64; 2])>;

fn pairs_of(moving_um: &[[f64; 2]], fixed_um: &[[f64; 2]]) -> Result<PointPairs, RegError> {
    if moving_um.len() != fixed_um.len() {
        return Err(RegError::InvalidInput(format!(
            "jumlah landmark moving ({}) ≠ fixed ({})",
            moving_um.len(),
            fixed_um.len()
        )));
    }
    if let Some(p) = moving_um
        .iter()
        .chain(fixed_um.iter())
        .find(|p| !p.iter().all(|v| v.is_finite()))
    {
        return Err(RegError::InvalidInput(format!(
            "landmark tidak finit: {p:?}"
        )));
    }
    Ok(moving_um
        .iter()
        .zip(fixed_um.iter())
        .map(|(m, f)| (*m, *f))
        .collect())
}

/// `P_{moving→fixed}` (µm → µm) dari pasangan landmark: ≥ 3 pasang → affine
/// kuadrat terkecil; tepat 2 pasang → similarity (tanpa cermin); < 2 → galat.
pub fn from_landmarks(
    pairs_moving_um: &[[f64; 2]],
    pairs_fixed_um: &[[f64; 2]],
) -> Result<Affine2, RegError> {
    let pairs = pairs_of(pairs_moving_um, pairs_fixed_um)?;
    match pairs.len() {
        0 | 1 => Err(RegError::InvalidInput(format!(
            "butuh minimal 2 pasang landmark, diberikan {}",
            pairs.len()
        ))),
        2 => Ok(Affine2::fit_similarity(&pairs)?),
        _ => Ok(Affine2::fit_affine(&pairs)?),
    }
}

/// Galat registrasi sasaran (TRE) RMS dalam µm: `|transform(moving) − fixed|`
/// dengan `transform` = `P_{moving→fixed}`. Pasangan tidak sebanding → 0.
pub fn tre(transform: &Affine2, pairs_moving_um: &[[f64; 2]], pairs_fixed_um: &[[f64; 2]]) -> f64 {
    match pairs_of(pairs_moving_um, pairs_fixed_um) {
        Ok(pairs) => transform.rms_error(&pairs),
        Err(_) => 0.0,
    }
}
