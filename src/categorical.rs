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
//! Training codes are limited to `0..=MAX_CAT_CODE` (253): bin 255 is the
//! missing sentinel and bin 254 (`UNSEEN_CAT_BIN`) is kept free so that a
//! category unseen in training always has a bin that is in no category set. Loaded XGBoost models may hold larger category ids; the
//! prediction paths use arbitrary-width `u32` bitsets for those.

use crate::types::MISSING_BIN;

/// Largest category code a training bin can represent (`MISSING_BIN - 2`).
pub const MAX_CAT_CODE: u32 = MISSING_BIN as u32 - 2;

/// The bin a category unseen in training maps to. It is above every trained
/// code, so it is in no category set and goes left at every categorical node,
/// exactly as prediction from the float value does (XGBoost semantics). It is
/// never a missing value, so the node's default direction never applies.
pub const UNSEEN_CAT_BIN: u8 = (MISSING_BIN as u32 - 1) as u8;

/// XGBoost `OutOfRangeCat()`: category values at or above this are invalid.
const OUT_OF_RANGE_CAT: f32 = 16_777_216.0;

/// A set of category codes `0..=255` as a 256-bit bitset (training side).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CatBits(pub [u64; 4]);

impl CatBits {
    pub const fn empty() -> Self {
        Self([0; 4])
    }

    pub fn set(&mut self, cat: u32) {
        debug_assert!(cat < 256);
        self.0[(cat >> 6) as usize & 3] |= 1u64 << (cat & 63);
    }

    pub fn contains(&self, cat: u32) -> bool {
        cat < 256 && self.0[(cat >> 6) as usize] >> (cat & 63) & 1 != 0
    }

    pub fn count(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    /// Members in ascending order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        (0..256u32).filter(|&c| self.contains(c))
    }

    /// The same set as little-endian 32-bit words (XGBoost's `CatBitField`
    /// layout), trimmed to the highest populated word.
    pub fn to_words(&self) -> Vec<u32> {
        let mut words: Vec<u32> = (0..8)
            .map(|i| (self.0[i / 2] >> (32 * (i % 2))) as u32)
            .collect();
        while words.last() == Some(&0) {
            words.pop();
        }
        words
    }

    /// The 8 words of the set, untrimmed, for device kernels.
    pub fn words8(&self) -> [u32; 8] {
        std::array::from_fn(|i| (self.0[i / 2] >> (32 * (i % 2))) as u32)
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
/// Codes outside the trained range (or invalid values) are "unseen": they go
/// left at every categorical node, so they map to `UNSEEN_CAT_BIN`, which no
/// trained category set contains (training codes stop at `MAX_CAT_CODE`).
/// Only validation sets quantized with the training cuts can reach this:
/// training data is validated while the cuts are built.
#[inline]
pub fn code_to_bin(value: f32, n_bins: usize) -> u8 {
    debug_assert!(n_bins <= MAX_CAT_CODE as usize + 1);
    if value >= 0.0 && value < n_bins as f32 && value == value.trunc() {
        return value as u8;
    }
    UNSEEN_CAT_BIN
}

/// Whether a non-missing training category value is not a valid code.
/// NaN (missing) is not an error and must be filtered by the caller.
pub fn invalid_code(value: f32) -> bool {
    value < 0.0 || value > MAX_CAT_CODE as f32 || value != value.trunc()
}

/// The error text for an invalid training category.
pub fn invalid_code_message(feature: &str, value: f32) -> String {
    format!(
        "categorical feature {feature} contains the value {value}: categories must be integer \
         codes in 0..={MAX_CAT_CODE} (the two largest bins are reserved for missing values and unseen categories), \
         so at most {} distinct categories are supported per feature. Missing values \
         must be NaN or null",
        MAX_CAT_CODE + 1
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
        assert_eq!(code_to_bin(2.5, 6), 254);
        assert_eq!(code_to_bin(300.0, 254), UNSEEN_CAT_BIN);
        assert_eq!(code_to_bin(253.0, 254), 253);
        assert_eq!(code_to_bin(254.0, 254), UNSEEN_CAT_BIN);
        assert!(invalid_code(254.0));
        assert!(!invalid_code(253.0));
    }
}
