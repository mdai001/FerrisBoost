//! Category names: the values behind categorical codes.
//!
//! A categorical feature trains on integer codes `0..n`. When the input names
//! its categories (a pandas `category` column, an Arrow / Parquet string
//! dictionary), code `i` stands for the `i`-th entry of the feature's category
//! list. The model stores these lists (XGBoost 3.x `cats` in the model JSON)
//! and prediction re-codes named input by value, so a DataFrame or a Parquet
//! file whose dictionary lists the same categories in another order scores
//! exactly like the training data.
//!
//! Values are strings or integers, as in XGBoost (floating point and boolean
//! categories are rejected there too). Integer categories keep XGBoost's
//! array-interface type code (`9..=16` = int8, uint8, int16, uint16, int32,
//! uint32, int64, uint64) so a saved model round-trips the declared dtype;
//! values must fit `i64` (XGBoost refuses larger ones).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Most categories one feature can hold: codes `0..=MAX_CAT_CODE_WIDE`.
pub const MAX_CATEGORIES: usize = crate::categorical::MAX_CAT_CODE_WIDE as usize + 1;

/// XGBoost array-interface type codes accepted for integer categories.
pub const INT_TYPE_CODES: std::ops::RangeInclusive<u8> = 9..=16;

/// The categories of one feature, in code order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CategoryValues {
    Str(Vec<String>),
    Int { type_code: u8, values: Vec<i64> },
}

impl CategoryValues {
    pub fn len(&self) -> usize {
        match self {
            Self::Str(v) => v.len(),
            Self::Int { values, .. } => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// "string" / "integer", for messages.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Str(_) => "string",
            Self::Int { .. } => "integer",
        }
    }

    /// Code order sorted by value, as XGBoost writes `sorted_idx`: strings
    /// compare as signed bytes (XGBoost stores them as `int8`), shorter prefix
    /// first; integers numerically.
    pub fn sorted_idx(&self) -> Vec<u32> {
        let mut idx: Vec<u32> = (0..self.len() as u32).collect();
        match self {
            Self::Str(v) => {
                let signed = |s: &str| -> Vec<i8> { s.bytes().map(|b| b as i8).collect() };
                let keys: Vec<Vec<i8>> = v.iter().map(|s| signed(s)).collect();
                idx.sort_by(|&a, &b| keys[a as usize].cmp(&keys[b as usize]));
            }
            Self::Int { values, .. } => idx.sort_by_key(|&i| values[i as usize]),
        }
        idx
    }
}

#[derive(Clone, Debug)]
enum Lookup {
    Str(HashMap<String, u32>),
    Int(HashMap<i64, u32>),
}

/// One feature's categories plus a value → code index.
#[derive(Clone, Debug)]
pub struct FeatureCategories {
    values: CategoryValues,
    lookup: Lookup,
}

impl PartialEq for FeatureCategories {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values
    }
}

impl FeatureCategories {
    /// Validates the values (unique, at most [`MAX_CATEGORIES`], a known
    /// integer type code) and builds the lookup.
    pub fn new(values: CategoryValues) -> anyhow::Result<Self> {
        crate::ensure_input!(
            values.len() <= MAX_CATEGORIES,
            "{} categories exceed the limit of {MAX_CATEGORIES} per feature",
            values.len()
        );
        let lookup = match &values {
            CategoryValues::Str(v) => {
                let mut map = HashMap::with_capacity(v.len());
                for (i, s) in v.iter().enumerate() {
                    if map.insert(s.clone(), i as u32).is_some() {
                        crate::bail_input!("duplicate category {s:?}");
                    }
                }
                Lookup::Str(map)
            }
            CategoryValues::Int { type_code, values } => {
                crate::ensure_input!(
                    INT_TYPE_CODES.contains(type_code),
                    "unsupported integer category type code {type_code} (expected 9..=16)"
                );
                let mut map = HashMap::with_capacity(values.len());
                for (i, &x) in values.iter().enumerate() {
                    if map.insert(x, i as u32).is_some() {
                        crate::bail_input!("duplicate category {x}");
                    }
                }
                Lookup::Int(map)
            }
        };
        Ok(Self { values, lookup })
    }

    pub fn values(&self) -> &CategoryValues {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Code of a string category (`None` for an unknown one or an integer list).
    pub fn code_of_str(&self, s: &str) -> Option<u32> {
        match &self.lookup {
            Lookup::Str(m) => m.get(s).copied(),
            Lookup::Int(_) => None,
        }
    }

    /// Code of an integer category (`None` for an unknown one or a string list).
    pub fn code_of_int(&self, x: i64) -> Option<u32> {
        match &self.lookup {
            Lookup::Int(m) => m.get(&x).copied(),
            Lookup::Str(_) => None,
        }
    }

    /// Category `code` for messages.
    pub fn describe(&self, code: usize) -> String {
        match &self.values {
            CategoryValues::Str(v) => format!("{:?}", v[code]),
            CategoryValues::Int { values, .. } => values[code].to_string(),
        }
    }
}

/// Category lists of every feature of a model (`None` = numeric, or a
/// categorical feature trained on bare integer codes).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct CategoryTable {
    features: Vec<Option<FeatureCategories>>,
}

impl CategoryTable {
    pub fn new(features: Vec<Option<FeatureCategories>>) -> Self {
        Self { features }
    }

    pub fn n_features(&self) -> usize {
        self.features.len()
    }

    pub fn get(&self, feature: usize) -> Option<&FeatureCategories> {
        self.features.get(feature).and_then(Option::as_ref)
    }

    pub fn features(&self) -> &[Option<FeatureCategories>] {
        &self.features
    }

    /// Whether no feature has a category list.
    pub fn is_empty(&self) -> bool {
        self.features.iter().all(Option::is_none)
    }

    /// `None` when no feature has a category list.
    pub fn non_empty(self) -> Option<Self> {
        (!self.is_empty()).then_some(self)
    }

    /// Every feature with a category list must be categorical (`flags`, empty =
    /// all numeric), and the table must cover `n_features`.
    pub fn check_against(&self, n_features: usize, flags: &[bool]) -> anyhow::Result<()> {
        crate::ensure_input!(
            self.features.len() == n_features,
            "category lists cover {} features, the model has {n_features}",
            self.features.len()
        );
        for (f, c) in self.features.iter().enumerate() {
            if c.is_some() {
                crate::ensure_input!(
                    flags.get(f).copied().unwrap_or(false),
                    "feature {f} has category names but is not categorical"
                );
            }
        }
        Ok(())
    }

    /// Mixed into the quantized cache hash (only called when categories exist).
    pub(crate) fn hash_into(&self, mut hash: u64) -> u64 {
        let mut mix = |bytes: &[u8]| {
            for &b in bytes {
                hash = (hash ^ b as u64).wrapping_mul(0x100000001b3);
            }
        };
        for (f, c) in self.features.iter().enumerate() {
            let Some(c) = c else { continue };
            mix(&(f as u32).to_le_bytes());
            match &c.values {
                CategoryValues::Str(v) => {
                    mix(b"s");
                    for s in v {
                        mix(&(s.len() as u64).to_le_bytes());
                        mix(s.as_bytes());
                    }
                }
                CategoryValues::Int { type_code, values } => {
                    mix(&[b'i', *type_code]);
                    for x in values {
                        mix(&x.to_le_bytes());
                    }
                }
            }
        }
        hash
    }

    /// XGBoost 3.x `cats` object (`gradient_booster.model.cats`).
    pub(crate) fn to_xgb(&self) -> XgbCats {
        let mut enc = Vec::with_capacity(self.features.len());
        let mut feature_segments = Vec::with_capacity(self.features.len() + 1);
        let mut sorted_idx = Vec::new();
        feature_segments.push(0i64);
        let mut total = 0i64;
        for c in &self.features {
            match c {
                None => enc.push(XgbEnc::Str {
                    offsets: Vec::new(),
                    values: Vec::new(),
                }),
                Some(c) => {
                    match &c.values {
                        CategoryValues::Str(v) => {
                            let mut offsets = Vec::with_capacity(v.len() + 1);
                            let mut values = Vec::new();
                            offsets.push(0i64);
                            for s in v {
                                values.extend(s.bytes().map(|b| b as i8 as i64));
                                offsets.push(values.len() as i64);
                            }
                            enc.push(XgbEnc::Str { offsets, values });
                        }
                        CategoryValues::Int { type_code, values } => enc.push(XgbEnc::Int {
                            type_code: *type_code,
                            values: values.clone(),
                        }),
                    }
                    sorted_idx.extend(c.values.sorted_idx().into_iter().map(i64::from));
                    total += c.len() as i64;
                }
            }
            feature_segments.push(total);
        }
        XgbCats {
            enc,
            feature_segments,
            sorted_idx,
        }
    }

    /// Parse an XGBoost `cats` object. `None` when it holds no category list
    /// (XGBoost writes empty arrays for a model trained without names).
    pub(crate) fn from_xgb(cats: &XgbCats, n_features: usize) -> anyhow::Result<Option<Self>> {
        if cats.enc.is_empty() {
            return Ok(None);
        }
        crate::ensure_input!(
            cats.enc.len() == n_features,
            "cats.enc has {} entries, the model has {n_features} features",
            cats.enc.len()
        );
        crate::ensure_input!(
            cats.feature_segments.len() == n_features + 1,
            "cats.feature_segments has {} entries, expected {}",
            cats.feature_segments.len(),
            n_features + 1
        );
        let mut features = Vec::with_capacity(n_features);
        let mut total = 0i64;
        for (f, e) in cats.enc.iter().enumerate() {
            let values = match e {
                XgbEnc::Str { offsets, values } if offsets.is_empty() && values.is_empty() => None,
                XgbEnc::Str { offsets, values } => {
                    crate::ensure_input!(
                        offsets.first() == Some(&0)
                            && offsets.last().copied() == Some(values.len() as i64)
                            && offsets.windows(2).all(|w| w[0] <= w[1]),
                        "cats.enc[{f}] has corrupt string offsets"
                    );
                    let mut bytes = Vec::with_capacity(values.len());
                    for &b in values {
                        crate::ensure_input!(
                            (-128..=255).contains(&b),
                            "cats.enc[{f}] has a string byte out of range: {b}"
                        );
                        bytes.push(b as u8);
                    }
                    let mut strings = Vec::with_capacity(offsets.len() - 1);
                    for w in offsets.windows(2) {
                        let s = std::str::from_utf8(&bytes[w[0] as usize..w[1] as usize]).map_err(
                            |_| {
                                crate::input_err!(
                                    "cats.enc[{f}] holds a category that is not valid UTF-8"
                                )
                            },
                        )?;
                        strings.push(s.to_string());
                    }
                    Some(CategoryValues::Str(strings))
                }
                XgbEnc::Int { type_code, values } => Some(CategoryValues::Int {
                    type_code: *type_code,
                    values: values.clone(),
                }),
            };
            let n = values.as_ref().map_or(0, CategoryValues::len) as i64;
            total += n;
            crate::ensure_input!(
                cats.feature_segments[f + 1] - cats.feature_segments[f] == n,
                "cats.feature_segments does not match the category count of feature {f}"
            );
            features.push(
                values
                    .map(FeatureCategories::new)
                    .transpose()
                    .map_err(|e| crate::input_err!("cats.enc[{f}]: {e}"))?,
            );
        }
        crate::ensure_input!(
            cats.sorted_idx.len() as i64 == total,
            "cats.sorted_idx has {} entries, the model has {total} categories",
            cats.sorted_idx.len()
        );
        Ok(Self::new(features).non_empty())
    }
}

/// XGBoost `cats` JSON (`enc` per feature, `feature_segments`, `sorted_idx`).
#[derive(Serialize, Deserialize)]
#[doc(hidden)]
pub struct XgbCats {
    #[serde(default)]
    pub enc: Vec<XgbEnc>,
    #[serde(default)]
    pub feature_segments: Vec<i64>,
    #[serde(default)]
    pub sorted_idx: Vec<i64>,
}

/// One feature's encoding: integer categories carry an array-interface `type`;
/// strings are Arrow-style `offsets` into signed `values` bytes. A numeric
/// feature is a string entry with empty arrays.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
#[doc(hidden)]
pub enum XgbEnc {
    Int {
        #[serde(rename = "type")]
        type_code: u8,
        values: Vec<i64>,
    },
    Str {
        offsets: Vec<i64>,
        values: Vec<i64>,
    },
}

/// Learns one feature's category list in first-seen order (training pass 1).
#[derive(Default)]
pub struct CategoryBuilder {
    values: Vec<String>,
    lookup: HashMap<String, u32>,
}

impl CategoryBuilder {
    /// Code of `s`, appending it when new.
    pub fn intern(&mut self, s: &str, feature: &str) -> anyhow::Result<u32> {
        if let Some(&c) = self.lookup.get(s) {
            return Ok(c);
        }
        crate::ensure_input!(
            self.values.len() < MAX_CATEGORIES,
            "feature {feature:?} has more than {MAX_CATEGORIES} distinct categories (the limit per feature)"
        );
        let code = self.values.len() as u32;
        self.values.push(s.to_string());
        self.lookup.insert(s.to_string(), code);
        Ok(code)
    }

    /// Code of `s` if it has one already.
    pub fn get(&self, s: &str) -> Option<u32> {
        self.lookup.get(s).copied()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn finish(self) -> FeatureCategories {
        FeatureCategories {
            values: CategoryValues::Str(self.values),
            lookup: Lookup::Str(self.lookup),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strs(v: &[&str]) -> CategoryValues {
        CategoryValues::Str(v.iter().map(|s| s.to_string()).collect())
    }

    #[test]
    fn sorted_idx_matches_xgboost_signed_byte_order() {
        // XGBoost 3.4.1: ['', 'Z', 'a', 'aa', 'ab', 'aé', 'b\x7f', 'é'] -> [0,7,1,2,5,3,4,6]
        let v = strs(&["", "Z", "a", "aa", "ab", "aé", "b\x7f", "é"]);
        assert_eq!(v.sorted_idx(), vec![0, 7, 1, 2, 5, 3, 4, 6]);
        let i = CategoryValues::Int {
            type_code: 15,
            values: vec![3, -5, 100, -1],
        };
        assert_eq!(i.sorted_idx(), vec![1, 3, 0, 2]);
    }

    #[test]
    fn xgb_round_trip_and_layout() {
        let t = CategoryTable::new(vec![
            None,
            Some(FeatureCategories::new(strs(&["d", "b", "a", "é"])).unwrap()),
            Some(
                FeatureCategories::new(CategoryValues::Int {
                    type_code: 15,
                    values: vec![10, 3, 7],
                })
                .unwrap(),
            ),
        ]);
        let x = t.to_xgb();
        let json = serde_json::to_string(&x).unwrap();
        assert_eq!(
            json,
            r#"{"enc":[{"offsets":[],"values":[]},{"offsets":[0,1,2,3,5],"values":[100,98,97,-61,-87]},{"type":15,"values":[10,3,7]}],"feature_segments":[0,0,4,7],"sorted_idx":[3,2,1,0,1,2,0]}"#
        );
        let back: XgbCats = serde_json::from_str(&json).unwrap();
        assert_eq!(CategoryTable::from_xgb(&back, 3).unwrap(), Some(t));
    }

    #[test]
    fn duplicates_and_limits_are_rejected() {
        assert!(FeatureCategories::new(strs(&["a", "a"])).is_err());
        assert!(FeatureCategories::new(CategoryValues::Int {
            type_code: 3,
            values: vec![1]
        })
        .is_err());
        let mut b = CategoryBuilder::default();
        for i in 0..MAX_CATEGORIES {
            b.intern(&i.to_string(), "f").unwrap();
        }
        assert!(b.intern("one more", "f").is_err());
        assert_eq!(b.intern("7", "f").unwrap(), 7);
    }
}
