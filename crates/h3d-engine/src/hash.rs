//! FNV-1a 64-bit: hash stabil lintas platform dan lintas jalan untuk kunci
//! cache dan `params_hash` (tidak memakai `DefaultHasher` yang berbenih acak).

/// Penghitung FNV-1a 64-bit.
#[derive(Clone, Copy, Debug)]
pub struct Fnv1a64(u64);

impl Default for Fnv1a64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv1a64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    /// Hash baru.
    pub fn new() -> Self {
        Fnv1a64(Self::OFFSET)
    }

    /// Tambahkan byte.
    pub fn bytes(&mut self, bytes: &[u8]) -> &mut Self {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
        self
    }

    /// Tambahkan byte berawalan panjang (agar batas antar field tidak ambigu).
    pub fn chunk(&mut self, bytes: &[u8]) -> &mut Self {
        self.u64(bytes.len() as u64);
        self.bytes(bytes)
    }

    /// Tambahkan `u64`.
    pub fn u64(&mut self, v: u64) -> &mut Self {
        self.bytes(&v.to_le_bytes())
    }

    /// Tambahkan `f64` lewat bit-nya.
    pub fn f64(&mut self, v: f64) -> &mut Self {
        self.u64(v.to_bits())
    }

    /// Tambahkan teks.
    pub fn str(&mut self, s: &str) -> &mut Self {
        self.chunk(s.as_bytes())
    }

    /// Nilai hash.
    pub fn finish(&self) -> u64 {
        self.0
    }
}

/// JSON ringkas sebuah nilai serde, untuk dimasukkan ke hash kunci cache.
pub(crate) fn json_bytes<T: serde::Serialize>(v: &T) -> Result<Vec<u8>, crate::error::EngineError> {
    serde_json::to_vec(v).map_err(|e| {
        crate::error::EngineError::Io(h3d_io::IoError::json(
            std::path::Path::new("kunci-cache"),
            e,
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vector() {
        assert_eq!(Fnv1a64::new().bytes(b"a").finish(), 0xaf63_dc4c_8601_ec8c);
    }
}
