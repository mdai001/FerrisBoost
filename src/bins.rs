//! Bin storage width per column block.
//!
//! Most features fit in 8-bit bins (`types::Bin = u8`): at most 255 real bins
//! for a numeric feature (bin 255 is the missing sentinel) and category codes
//! `0..=253` for a categorical feature (bin 254 is the unseen-category bin).
//! A feature that needs more bins (numeric `max_bin` above 255, or more
//! categories) is stored at 16 bits instead. The width is a pure function of
//! the feature's cuts ([`BinWidth::for_feature`]), and a column block is
//! homogeneous: it holds only features of one width
//! ([`crate::columns::partition_features_by_width`]). Every other block keeps
//! the 8-bit layout byte for byte; nothing widens the whole matrix.
//!
//! Kernels are written once over [`BinCode`] and dispatched per block or per
//! column from a [`BinSlice`], so the inner loops stay monomorphic.

/// Bytes per stored bin of a column block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BinWidth {
    U8,
    U16,
}

impl BinWidth {
    /// Bytes per cell.
    pub const fn bytes(self) -> usize {
        match self {
            BinWidth::U8 => 1,
            BinWidth::U16 => 2,
        }
    }

    /// The missing-value sentinel of this width (the largest code).
    pub const fn missing(self) -> u32 {
        match self {
            BinWidth::U8 => u8::MAX as u32,
            BinWidth::U16 => u16::MAX as u32,
        }
    }

    /// The bin of a category unseen in training (one below missing).
    pub const fn unseen_cat(self) -> u32 {
        self.missing() - 1
    }

    /// Most real bins a numeric feature of this width can hold.
    pub const fn max_numeric_bins(self) -> usize {
        self.missing() as usize
    }

    /// Largest category code a categorical feature of this width can hold.
    pub const fn max_cat_code(self) -> u32 {
        self.missing() - 2
    }

    /// The narrowest width that holds a feature with `n_bins` bins, or `None`
    /// when no supported width does. Categorical features also reserve the
    /// unseen-category bin.
    pub fn for_feature(n_bins: usize, categorical: bool) -> Option<BinWidth> {
        [BinWidth::U8, BinWidth::U16].into_iter().find(|w| {
            if categorical {
                n_bins <= w.max_cat_code() as usize + 1
            } else {
                n_bins <= w.max_numeric_bins()
            }
        })
    }

    /// The width with `bytes` bytes per cell (cache manifests).
    pub fn from_bytes(bytes: u8) -> Option<BinWidth> {
        match bytes {
            1 => Some(BinWidth::U8),
            2 => Some(BinWidth::U16),
            _ => None,
        }
    }
}

/// Largest `max_bin` the widest supported storage accepts.
pub const MAX_BIN_LIMIT_WIDE: u32 = u16::MAX as u32;

/// A stored bin code: `u8` or `u16`.
pub trait BinCode: Copy + Eq + Ord + Default + Send + Sync + std::fmt::Debug + 'static {
    const WIDTH: BinWidth;
    const MISSING: Self;
    /// The bin as a histogram slot index.
    fn idx(self) -> usize;
    /// Narrow a bin computed as `u32` to this width. The caller guarantees the
    /// range (the block width comes from the same cuts that produced the bin).
    fn from_bin(bin: u32) -> Self;
    fn into_data(v: Vec<Self>) -> BinData;
}

impl BinCode for u8 {
    const WIDTH: BinWidth = BinWidth::U8;
    const MISSING: Self = u8::MAX;
    #[inline(always)]
    fn idx(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn from_bin(bin: u32) -> Self {
        debug_assert!(bin <= u8::MAX as u32, "bin {bin} does not fit 8 bits");
        bin as u8
    }
    fn into_data(v: Vec<Self>) -> BinData {
        BinData::U8(v)
    }
}

impl BinCode for u16 {
    const WIDTH: BinWidth = BinWidth::U16;
    const MISSING: Self = u16::MAX;
    #[inline(always)]
    fn idx(self) -> usize {
        self as usize
    }
    #[inline(always)]
    fn from_bin(bin: u32) -> Self {
        debug_assert!(bin <= u16::MAX as u32, "bin {bin} does not fit 16 bits");
        bin as u16
    }
    fn into_data(v: Vec<Self>) -> BinData {
        BinData::U16(v)
    }
}

/// Owned bins of one width (a block's payload or one column).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BinData {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl Default for BinData {
    fn default() -> Self {
        BinData::U8(Vec::new())
    }
}

impl From<Vec<u8>> for BinData {
    fn from(v: Vec<u8>) -> Self {
        BinData::U8(v)
    }
}

impl From<Vec<u16>> for BinData {
    fn from(v: Vec<u16>) -> Self {
        BinData::U16(v)
    }
}

impl BinData {
    /// `len` cells of `width`, all set to the missing sentinel.
    pub fn missing(width: BinWidth, len: usize) -> Self {
        match width {
            BinWidth::U8 => BinData::U8(vec![u8::MISSING; len]),
            BinWidth::U16 => BinData::U16(vec![u16::MISSING; len]),
        }
    }

    pub fn width(&self) -> BinWidth {
        self.as_slice().width()
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Payload size in bytes.
    pub fn byte_len(&self) -> usize {
        self.len() * self.width().bytes()
    }

    pub fn as_slice(&self) -> BinSlice<'_> {
        match self {
            BinData::U8(v) => BinSlice::U8(v),
            BinData::U16(v) => BinSlice::U16(v),
        }
    }

    /// The 8-bit payload, if this is 8-bit data.
    pub fn as_u8(&self) -> Option<&[u8]> {
        match self {
            BinData::U8(v) => Some(v),
            BinData::U16(_) => None,
        }
    }

    /// The 16-bit payload, if this is 16-bit data.
    pub fn as_u16(&self) -> Option<&[u16]> {
        match self {
            BinData::U16(v) => Some(v),
            BinData::U8(_) => None,
        }
    }

    /// The little-endian byte image (the on-disk cache format). For 8-bit data
    /// this is the vector itself.
    pub fn to_le_bytes(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            BinData::U8(v) => std::borrow::Cow::Borrowed(v),
            BinData::U16(v) => {
                std::borrow::Cow::Owned(v.iter().flat_map(|c| c.to_le_bytes()).collect())
            }
        }
    }

    /// The payload as little-endian bytes, borrowed (the host is little-endian,
    /// so 16-bit cells already are their on-disk / device byte image).
    pub fn as_le_bytes(&self) -> &[u8] {
        self.as_slice().as_le_bytes()
    }

    /// The payload as mutable little-endian bytes (for reading a cache block
    /// straight into its typed storage, without a byte-to-cell conversion copy).
    pub fn as_le_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            BinData::U8(v) => v,
            // SAFETY: u16 has no padding and every byte pattern is a valid u16;
            // the view covers exactly the vector's initialised cells.
            BinData::U16(v) => unsafe {
                std::slice::from_raw_parts_mut(
                    v.as_mut_ptr().cast::<u8>(),
                    std::mem::size_of_val(v.as_slice()),
                )
            },
        }
    }

    /// Inverse of [`Self::to_le_bytes`].
    // Keep the portable forms until the crate's supported Rust version is explicit.
    #[allow(clippy::manual_is_multiple_of, clippy::chunks_exact_to_as_chunks)]
    pub fn from_le_bytes(width: BinWidth, bytes: Vec<u8>) -> anyhow::Result<Self> {
        match width {
            BinWidth::U8 => Ok(BinData::U8(bytes)),
            BinWidth::U16 => {
                anyhow::ensure!(
                    bytes.len() % 2 == 0,
                    "16-bit bin payload has an odd byte length {}",
                    bytes.len()
                );
                Ok(BinData::U16(
                    bytes
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect(),
                ))
            }
        }
    }
}

/// Borrowed bins of one width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinSlice<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
}

impl<'a> BinSlice<'a> {
    pub fn width(&self) -> BinWidth {
        match self {
            BinSlice::U8(_) => BinWidth::U8,
            BinSlice::U16(_) => BinWidth::U16,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            BinSlice::U8(s) => s.len(),
            BinSlice::U16(s) => s.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cell `i` widened to `u32` (the missing sentinel stays the width's own).
    #[inline]
    pub fn get(&self, i: usize) -> u32 {
        match self {
            BinSlice::U8(s) => s[i] as u32,
            BinSlice::U16(s) => s[i] as u32,
        }
    }

    pub fn slice(&self, range: std::ops::Range<usize>) -> BinSlice<'a> {
        match *self {
            BinSlice::U8(s) => BinSlice::U8(&s[range]),
            BinSlice::U16(s) => BinSlice::U16(&s[range]),
        }
    }

    pub fn to_owned(&self) -> BinData {
        match *self {
            BinSlice::U8(s) => BinData::U8(s.to_vec()),
            BinSlice::U16(s) => BinData::U16(s.to_vec()),
        }
    }

    /// The cells as little-endian bytes, borrowed (no copy: the host is
    /// little-endian). This is the cache file and device upload image.
    pub fn as_le_bytes(&self) -> &'a [u8] {
        match *self {
            BinSlice::U8(s) => s,
            // SAFETY: u16 has no padding and any byte pattern is a valid u8; the
            // view covers exactly the same cells for the same lifetime.
            BinSlice::U16(s) => unsafe {
                std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s))
            },
        }
    }

    /// The 8-bit payload, if this is an 8-bit slice.
    pub fn as_u8(&self) -> Option<&'a [u8]> {
        match *self {
            BinSlice::U8(s) => Some(s),
            BinSlice::U16(_) => None,
        }
    }

    /// The 8-bit payload, or a clear error for a 16-bit block. Used by paths
    /// that have no 16-bit implementation yet (the CUDA backend), so a wide
    /// block can never be misread as bytes.
    pub fn require_u8(&self, what: &str) -> anyhow::Result<&'a [u8]> {
        match *self {
            BinSlice::U8(s) => Ok(s),
            BinSlice::U16(_) => Err(crate::error::input_error(format!(
                "{what} does not support 16-bit bin columns yet (features with more than \
                 255 bins, i.e. max_bin > 255); train on the CPU or use max_bin <= 255"
            ))),
        }
    }
}

#[cfg(not(target_endian = "little"))]
compile_error!("16-bit bins are stored and uploaded as little-endian bytes");

/// Run `$body` with `$s` bound to the typed slice of a [`BinSlice`].
#[macro_export]
macro_rules! with_bins {
    ($slice:expr, |$s:ident| $body:expr) => {
        match $slice {
            $crate::bins::BinSlice::U8($s) => $body,
            $crate::bins::BinSlice::U16($s) => $body,
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widths_follow_the_sentinels() {
        assert_eq!(BinWidth::for_feature(255, false), Some(BinWidth::U8));
        assert_eq!(BinWidth::for_feature(256, false), Some(BinWidth::U16));
        assert_eq!(BinWidth::for_feature(65_535, false), Some(BinWidth::U16));
        assert_eq!(BinWidth::for_feature(65_536, false), None);
        // Categorical keeps the unseen bin free: codes 0..=253 in 8 bits.
        assert_eq!(BinWidth::for_feature(254, true), Some(BinWidth::U8));
        assert_eq!(BinWidth::for_feature(255, true), Some(BinWidth::U16));
        assert_eq!(BinWidth::for_feature(65_534, true), Some(BinWidth::U16));
        assert_eq!(BinWidth::for_feature(65_535, true), None);
        assert_eq!(BinWidth::U8.missing(), crate::types::MISSING_BIN as u32);
        assert_eq!(
            BinWidth::U8.unseen_cat(),
            crate::categorical::UNSEEN_CAT_BIN as u32
        );
        assert_eq!(
            BinWidth::U8.max_cat_code(),
            crate::categorical::MAX_CAT_CODE
        );
    }

    #[test]
    fn le_bytes_round_trip() {
        let narrow = BinData::U8(vec![0, 7, 255]);
        assert_eq!(&*narrow.to_le_bytes(), &[0, 7, 255]);
        let wide = BinData::U16(vec![0, 300, u16::MAX]);
        let bytes = wide.to_le_bytes().into_owned();
        assert_eq!(bytes, vec![0, 0, 44, 1, 255, 255]);
        assert_eq!(BinData::from_le_bytes(BinWidth::U16, bytes).unwrap(), wide);
        assert!(BinData::from_le_bytes(BinWidth::U16, vec![1, 2, 3]).is_err());
        assert_eq!(wide.byte_len(), 6);
        assert_eq!(wide.as_le_bytes(), &[0, 0, 44, 1, 255, 255]);
        let mut read = BinData::missing(BinWidth::U16, 3);
        read.as_le_bytes_mut()
            .copy_from_slice(&[0, 0, 44, 1, 255, 255]);
        assert_eq!(read, wide);
    }
}
