//! Categorical feature support: category sets and their decision rule.
//!
//! Semantics follow XGBoost 3.4.1 (`common/categorical.h`):
//!
//! * a categorical split stores the set of categories that go RIGHT; every
//!   other category goes left (`Decision` returns `!set.contains(cat)`),
//! * missing values (NaN) follow the node's default direction,
//! * an invalid category (negative, `>= 2^24`) at prediction time is "not in
//!   the set", so it goes left,
//! * training bins for a categorical feature are the category codes
//!   themselves, so `n_bins = max_code + 1`, including codes that never occur.
//!
//! Training codes are limited per storage width: an 8-bit categorical column
//! holds codes `0..=MAX_CAT_CODE` (253): bin 255 is the missing sentinel and
//! bin 254 (`UNSEEN_CAT_BIN`) is kept free so that a category unseen in
//! training always has a bin that is in no category set. A feature with a
//! larger code is stored at 16 bits (`crate::bins::BinWidth`), where the same
//! two sentinels sit at 65535 / 65534, so training accepts codes up to
//! `MAX_CAT_CODE_WIDE` (65533). Loaded XGBoost models may hold larger category
//! ids; the prediction paths use arbitrary-width `u32` bitsets for those.

use crate::types::MISSING_BIN;

/// Largest category code an 8-bit training bin can represent (`MISSING_BIN - 2`).
pub const MAX_CAT_CODE: u32 = MISSING_BIN as u32 - 2;

/// Largest category code training accepts at all (16-bit storage).
pub const MAX_CAT_CODE_WIDE: u32 = crate::bins::BinWidth::U16.max_cat_code();

/// The 8-bit bin a category unseen in training maps to. It is above every
/// trained code, so it is in no category set and goes left at every
/// categorical node, exactly as prediction from the float value does (XGBoost
/// semantics). It is never a missing value, so the node's default direction
/// never applies. 16-bit columns use `BinWidth::U16.unseen_cat()`.
pub const UNSEEN_CAT_BIN: u8 = (MISSING_BIN as u32 - 1) as u8;

/// XGBoost `OutOfRangeCat()`: category values at or above this are invalid.
const OUT_OF_RANGE_CAT: f32 = 16_777_216.0;

/// A set of training category codes (bins) as a bitset (training side).
///
/// Sets over an 8-bit categorical feature (codes `0..256`) live inline, so the
/// common case never allocates; a set over a 16-bit feature is sized by the
/// feature's bins ([`Self::for_bins`]). Equality compares the members only.
#[derive(Clone, Debug)]
pub enum CatBits {
    /// Codes `0..256`.
    Small([u64; 4]),
    /// Any size (64 codes per word).
    Large(Box<[u64]>),
}

impl Default for CatBits {
    fn default() -> Self {
        Self::empty()
    }
}

impl PartialEq for CatBits {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (trim64(self.words64()), trim64(other.words64()));
        a == b
    }
}

impl Eq for CatBits {}

fn trim64(w: &[u64]) -> &[u64] {
    let n = w.iter().rposition(|&x| x != 0).map_or(0, |i| i + 1);
    &w[..n]
}

/// Whether `cat` is in the 64-bit word bitset `words` (codes past the end are not).
#[inline(always)]
pub fn contains_in(words: &[u64], cat: u32) -> bool {
    match words.get((cat >> 6) as usize) {
        Some(&w) => w >> (cat & 63) & 1 != 0,
        None => false,
    }
}

impl CatBits {
    /// The empty set over codes `0..256`.
    pub const fn empty() -> Self {
        Self::Small([0; 4])
    }

    /// The empty set sized for a feature with `n_bins` codes.
    pub fn for_bins(n_bins: usize) -> Self {
        if n_bins <= 256 {
            Self::empty()
        } else {
            Self::Large(vec![0u64; n_bins.div_ceil(64)].into_boxed_slice())
        }
    }

    /// The members as 64-bit words (code `c` is bit `c % 64` of word `c / 64`).
    #[inline(always)]
    pub fn words64(&self) -> &[u64] {
        match self {
            Self::Small(w) => w,
            Self::Large(w) => w,
        }
    }

    pub fn set(&mut self, cat: u32) {
        let word = (cat >> 6) as usize;
        if word >= self.words64().len() {
            // Grow (rare: callers size the set with `for_bins`).
            let mut w = self.words64().to_vec();
            w.resize(word + 1, 0);
            *self = Self::Large(w.into_boxed_slice());
        }
        let w = match self {
            Self::Small(w) => &mut w[word],
            Self::Large(w) => &mut w[word],
        };
        *w |= 1u64 << (cat & 63);
    }

    #[inline]
    pub fn contains(&self, cat: u32) -> bool {
        contains_in(self.words64(), cat)
    }

    pub fn count(&self) -> u32 {
        self.words64().iter().map(|w| w.count_ones()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words64().iter().all(|&w| w == 0)
    }

    /// Members in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.words64().iter().enumerate().flat_map(|(i, &w)| {
            let mut w = w;
            std::iter::from_fn(move || {
                if w == 0 {
                    return None;
                }
                let b = w.trailing_zeros();
                w &= w - 1;
                Some(i as u32 * 64 + b)
            })
        })
    }

    /// The set built from little-endian 32-bit words (XGBoost's
    /// `CatBitField` layout, as stored in a model).
    pub fn from_words(words: &[u32]) -> Self {
        let n64 = words.len().div_ceil(2);
        let mut out = Self::for_bins(n64 * 64);
        let w = match &mut out {
            Self::Small(w) => &mut w[..],
            Self::Large(w) => &mut w[..],
        };
        for (i, &x) in words.iter().enumerate() {
            w[i / 2] |= (x as u64) << (32 * (i % 2));
        }
        out
    }

    /// The same set as little-endian 32-bit words (XGBoost's `CatBitField`
    /// layout), trimmed to the highest populated word.
    pub fn to_words(&self) -> Vec<u32> {
        let w64 = self.words64();
        let mut words: Vec<u32> = (0..2 * w64.len())
            .map(|i| (w64[i / 2] >> (32 * (i % 2))) as u32)
            .collect();
        while words.last() == Some(&0) {
            words.pop();
        }
        words
    }

    /// Whether every member is below 256 (fits the device's 8-word sets).
    pub fn fits_8_words(&self) -> bool {
        self.words64().iter().skip(4).all(|&w| w == 0)
    }

    /// The 8 words of the set, untrimmed, for device kernels. Device training
    /// only runs on 8-bit categorical columns (codes `0..=253`); a 16-bit
    /// categorical column is rejected before training starts.
    pub fn words8(&self) -> [u32; 8] {
        debug_assert!(self.fits_8_words(), "category set wider than 256 codes");
        let w64 = self.words64();
        std::array::from_fn(|i| w64.get(i / 2).map_or(0, |w| (w >> (32 * (i % 2))) as u32))
    }
}

/// Build the 32-bit word bitset for an ascending-or-not list of categories.
pub fn words_from_categories(cats: &[u32]) -> Vec<u32> {
    let max = cats.iter().copied().max().unwrap_or(0) as usize;
    let mut words = vec![0u32; max / 32 + 1];
    for &c in cats {
        words[(c / 32) as usize] |= 1u32 << (c % 32);
    }
    words
}

/// Members of a word bitset in ascending order.
pub fn categories_from_words(words: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for (i, &w) in words.iter().enumerate() {
        for b in 0..32 {
            if w >> b & 1 != 0 {
                out.push(i as u32 * 32 + b);
            }
        }
    }
    out
}

/// XGBoost `Decision`: whether a non-missing raw value goes LEFT at a
/// categorical node whose right-going set is `words`.
///
/// NaN is never passed here by callers that honour the default direction; it
/// is answered "right" so that a default-right node stays consistent when a
/// caller skips the missing check.
#[inline]
pub fn goes_left(words: &[u32], value: f32) -> bool {
    if value.is_nan() {
        return false;
    }
    if value < 0.0 || value >= OUT_OF_RANGE_CAT {
        return true;
    }
    let cat = value as u32; // truncation toward zero, as the C++ cast
    let word = (cat / 32) as usize;
    match words.get(word) {
        None => true,
        Some(&w) => w >> (cat % 32) & 1 == 0,
    }
}

/// The bin of a categorical value under the training cuts of a feature with
/// `n_bins` bins (`n_bins - 1` is the largest known code).
///
/// A fractional value is truncated toward zero, as prediction does (XGBoost
/// `AsCat`), so validation and prediction always pick the same category.
/// Codes outside the trained range (negative, or at or above `n_bins`) are
/// "unseen": they go left at every categorical node, so they map to
/// `UNSEEN_CAT_BIN`, which no trained category set contains (training codes
/// stop at `MAX_CAT_CODE`).
/// Only validation sets quantized with the training cuts can reach this:
/// training data is validated while the cuts are built.
///
/// The unseen bin is the one of the feature's storage width
/// (`BinWidth::unseen_cat`), so it is never a trained code.
#[inline]
pub fn code_to_bin(value: f32, n_bins: usize) -> u32 {
    let width = crate::bins::BinWidth::for_feature(n_bins, true)
        .expect("categorical feature wider than the widest bin storage");
    if value >= 0.0 && value < n_bins as f32 {
        return value as u32; // truncation toward zero, like `goes_left`
    }
    width.unseen_cat()
}

/// Whether a non-missing training category value is not a valid code.
/// NaN (missing) is not an error and must be filtered by the caller.
/// Codes above `MAX_CAT_CODE` are valid; they store the feature at 16 bits.
pub fn invalid_code(value: f32) -> bool {
    value < 0.0 || value > MAX_CAT_CODE_WIDE as f32 || value != value.trunc()
}

/// The error text for an invalid training category.
pub fn invalid_code_message(feature: &str, value: f32) -> String {
    format!(
        "categorical feature {feature} contains the value {value}: categories must be integer \
         codes in 0..={MAX_CAT_CODE_WIDE} (the two largest 16-bit bins are reserved for missing \
         values and unseen categories), so at most {} distinct categories are supported per \
         feature. Missing values must be NaN or null",
        MAX_CAT_CODE_WIDE + 1
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitset_round_trip() {
        let mut b = CatBits::empty();
        for c in [0, 3, 31, 32, 64, 200, 254] {
            b.set(c);
        }
        assert_eq!(b.count(), 7);
        assert_eq!(
            b.iter().collect::<Vec<_>>(),
            vec![0, 3, 31, 32, 64, 200, 254]
        );
        let words = b.to_words();
        assert_eq!(categories_from_words(&words), b.iter().collect::<Vec<_>>());
        assert_eq!(words_from_categories(&[0, 3, 31, 32, 64, 200, 254]), words);
        assert!(!b.contains(255) && !b.contains(300));
    }

    #[test]
    fn wide_sets_are_sized_by_the_feature() {
        let mut b = CatBits::for_bins(20_000);
        assert!(matches!(b, CatBits::Large(_)));
        for c in [0, 255, 256, 1000, 19_999] {
            b.set(c);
        }
        assert_eq!(
            b.iter().collect::<Vec<_>>(),
            vec![0, 255, 256, 1000, 19_999]
        );
        assert!(b.contains(19_999) && !b.contains(20_000) && !b.contains(65_533));
        assert!(!b.fits_8_words());
        let words = b.to_words();
        assert_eq!(words.len(), 19_999 / 32 + 1);
        assert_eq!(categories_from_words(&words), b.iter().collect::<Vec<_>>());
        assert_eq!(CatBits::from_words(&words), b);
        // A small set grows on demand and compares equal regardless of storage.
        let mut s = CatBits::empty();
        s.set(3);
        let mut l = CatBits::for_bins(5000);
        l.set(3);
        assert_eq!(s, l);
        s.set(4000);
        assert!(s.contains(4000) && s.contains(3) && s.count() == 2);
        // 8-bit sets keep the device layout.
        let mut d = CatBits::for_bins(254);
        d.set(253);
        assert!(matches!(d, CatBits::Small(_)));
        assert_eq!(d.words8()[7], 1 << 29);
        assert_eq!(CatBits::from_words(&d.to_words()), d);
    }

    #[test]
    fn decision_matches_xgboost() {
        let words = words_from_categories(&[1, 4, 33]);
        // In the set: right. Not in the set: left.
        assert!(!goes_left(&words, 1.0));
        assert!(!goes_left(&words, 4.0));
        assert!(!goes_left(&words, 33.0));
        assert!(goes_left(&words, 0.0));
        assert!(goes_left(&words, 2.0));
        // Truncation toward zero, like the C++ cast.
        assert!(!goes_left(&words, 4.9));
        // Beyond the bitset, negative, and out-of-range values are left.
        assert!(goes_left(&words, 1000.0));
        assert!(goes_left(&words, -1.0));
        assert!(goes_left(&words, -0.5));
        assert!(goes_left(&words, 16_777_216.0));
        assert!(goes_left(&words, f32::INFINITY));
        assert!(goes_left(&words, f32::NEG_INFINITY));
    }

    #[test]
    fn unseen_codes_use_the_spare_bin() {
        assert_eq!(code_to_bin(3.0, 6), 3);
        assert_eq!(code_to_bin(6.0, 6), 254);
        assert_eq!(code_to_bin(-1.0, 6), 254);
        assert_eq!(code_to_bin(2.5, 6), 2);
        assert_eq!(code_to_bin(5.9, 6), 5);
        assert_eq!(code_to_bin(-0.5, 6), UNSEEN_CAT_BIN as u32);
        assert_eq!(code_to_bin(6.0, 6), UNSEEN_CAT_BIN as u32);
        assert_eq!(code_to_bin(300.0, 254), UNSEEN_CAT_BIN as u32);
        assert_eq!(code_to_bin(253.0, 254), 253);
        assert_eq!(code_to_bin(254.0, 254), UNSEEN_CAT_BIN as u32);
        // 254 is a valid training code: it stores the feature at 16 bits.
        assert!(!invalid_code(254.0));
        assert!(!invalid_code(MAX_CAT_CODE_WIDE as f32));
        assert!(invalid_code(MAX_CAT_CODE_WIDE as f32 + 1.0));
        // Validation binning and prediction agree on every value class.
        let words = [0b0100u32]; // category 2 goes right
        for v in [2.0f32, 2.5, 2.99, 1.5, -0.5, 7.0, 300.0] {
            let bin = code_to_bin(v, 6);
            let bin_left = bin == UNSEEN_CAT_BIN as u32 || words[0] >> bin & 1 == 0;
            assert_eq!(bin_left, goes_left(&words, v), "value {v}");
        }
        assert!(!invalid_code(253.0));
    }
}
