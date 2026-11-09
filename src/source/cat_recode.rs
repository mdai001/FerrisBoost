//! Re-coding named categories in Arrow batches.
//!
//! String dictionary columns (and plain string columns declared categorical)
//! are replaced by `Float32` category codes before any other decoding, so
//! the sketch, the quantizers and prediction keep reading numeric columns
//! through the single decode point (`arrow_source::visit_column_f32`).
//!
//! * Training pass 1 **learns** each feature's category list in the order the
//!   values first appear in the rows (files, then rows in order). Dictionary
//!   order and unused dictionary entries do not matter: a reader may split
//!   row groups into batches and re-encode dictionaries differently (Arrow's
//!   Parquet reader does when a batch spans row groups), so only row order
//!   is a stable definition. Per-batch and per-file dictionaries in any order
//!   therefore map to one consistent code per value, whatever the thread
//!   count or batching. This is `pandas.unique` order of the column.
//! * Every later read (pass 2, validation sets, quantized-cache rebuilds,
//!   prediction) uses a **fixed** table: values are looked up, and a row that
//!   holds a category outside the table is an error (XGBoost rejects such a
//!   category too). Unused dictionary entries are ignored.
//!
//! With a fixed table, integer dictionaries and plain integer columns are
//! re-coded by value when the feature's categories are integers (a model
//! trained from a pandas frame with integer categories). Without a category
//! list a numeric dictionary keeps its values as the codes, as before named
//! categories existed.
//!
//! CSV has no dictionaries: a categorical CSV column read as text (see
//! `csv_source::cat_reads`) is a plain string column here. A declared
//! categorical CSV column without a known list is read as text in training
//! pass 1 and *decided* there from every field ([`Coder::Decide`]): integer
//! codes when every non-empty field is a number, category names otherwise.

use std::sync::Arc;

use anyhow::Result;
use arrow::array::{
    Array, ArrayData, ArrayRef, AsArray, Float32Array, Int64Array, LargeStringArray, StringArray,
    StringViewArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::category_names::{CategoryBuilder, CategoryTable, CategoryValues, FeatureCategories};

const MISSING: u32 = u32::MAX;
const UNSEEN: u32 = u32::MAX - 1;
/// A dictionary entry a learning coder has not given a code yet: codes are
/// assigned when a row first uses the value, so the list does not depend on
/// how a reader splits row groups into batches and dictionaries.
const UNASSIGNED: u32 = u32::MAX - 2;

fn is_string(t: &DataType) -> bool {
    matches!(t, DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View)
}

/// Whether this column type carries named categories (a string dictionary, a
/// plain string column) and so must go through a recoder.
pub(crate) fn is_named(t: &DataType) -> bool {
    match t {
        DataType::Dictionary(_, v) => is_string(v),
        other => is_string(other),
    }
}

enum Strs<'a> {
    S(&'a StringArray),
    L(&'a LargeStringArray),
    V(&'a StringViewArray),
}

impl<'a> Strs<'a> {
    fn of(a: &'a dyn Array) -> Option<Self> {
        if let Some(s) = a.as_string_opt::<i32>() {
            return Some(Self::S(s));
        }
        if let Some(s) = a.as_string_opt::<i64>() {
            return Some(Self::L(s));
        }
        a.as_string_view_opt().map(Self::V)
    }

    fn get(&self, i: usize) -> Option<&'a str> {
        match self {
            Self::S(a) => (!a.is_null(i)).then(|| a.value(i)),
            Self::L(a) => (!a.is_null(i)).then(|| a.value(i)),
            Self::V(a) => (!a.is_null(i)).then(|| a.value(i)),
        }
    }
}

/// Integer dictionary values as `Option<Option<i64>>`: outer `None` = null
/// entry, inner `None` = a value outside `i64` (never a category).
fn int_values(values: &ArrayRef) -> Result<Vec<Option<Option<i64>>>> {
    if let Some(u) = values.as_any().downcast_ref::<UInt64Array>() {
        return Ok((0..u.len())
            .map(|i| (!u.is_null(i)).then(|| i64::try_from(u.value(i)).ok()))
            .collect());
    }
    let cast = arrow::compute::cast(values, &DataType::Int64)?;
    let a = cast
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("cast to Int64");
    Ok((0..a.len())
        .map(|i| (!a.is_null(i)).then(|| Some(a.value(i))))
        .collect())
}

/// How one feature column is coded.
enum Coder<'t> {
    /// Learning a new list (training pass 1).
    Learn(CategoryBuilder),
    /// A known list (a trained model, or a continued model's list).
    Fixed(&'t FeatureCategories),
    /// Training pass 1 of a declared categorical CSV column without a known
    /// list, read as text: codes or names is decided from every field.
    Decide(Box<Decide>),
}

/// A CSV field as the CSV type inference reads it: `Some(value)` for a
/// number (an integer or a decimal), `None` for anything else (text).
///
/// The rules are Arrow's CSV inference patterns (`-?\d+`, and
/// `-?((\d*\.\d+|\d+\.\d*)([eE][-+]?\d+)?|\d+([eE][-+]?\d+))`; an integer
/// of 19+ characters outside `i64` is text), with ASCII digits only. An
/// integer is converted like an `Int64` column (`i64 as f32`), a decimal like
/// a `Float64` one, so a column decided as numbers yields the values the
/// sampled reader used to give it.
pub(crate) fn csv_number(s: &str) -> Option<f32> {
    let b = s.as_bytes();
    let body = b.strip_prefix(b"-").unwrap_or(b);
    let digits = |from: usize| {
        body[from..]
            .iter()
            .take_while(|c| c.is_ascii_digit())
            .count()
    };
    let int = digits(0);
    if int == body.len() {
        if int == 0 {
            return None;
        }
        return s.parse::<i64>().ok().map(|v| v as f32);
    }
    let mut at = int;
    let dot = body[at] == b'.';
    if dot {
        let frac = digits(at + 1);
        at += 1 + frac;
        if int + frac == 0 {
            return None;
        }
    } else if int == 0 {
        return None;
    }
    if at < body.len() {
        if !matches!(body[at], b'e' | b'E') {
            return None;
        }
        at += 1;
        if at < body.len() && matches!(body[at], b'+' | b'-') {
            at += 1;
        }
        let exp = digits(at);
        if exp == 0 || at + exp != body.len() {
            return None;
        }
    } else if !dot {
        return None;
    }
    use arrow::compute::kernels::cast_utils::Parser;
    arrow::datatypes::Float64Type::parse(s).map(|v| v as f32)
}

/// Both readings of a [`Coder::Decide`] column, kept side by side over the
/// whole pass: the names in first-appearance order, and the code scan the
/// categorical sketch would have made of the numbers.
pub(crate) struct Decide {
    names: CategoryBuilder,
    /// The too-many-categories error, raised only if the column is names.
    overflow: Option<anyhow::Error>,
    /// Every non-empty field so far is a number.
    numeric: bool,
    /// The `missing` sentinel: a number equal to it is missing.
    missing: f32,
    /// Largest valid code, `-1` while none (as the categorical sketch).
    max_code: i32,
    /// The first number that is not a valid code.
    bad: Option<f32>,
}

/// What pass 1 decided for a [`Coder::Decide`] column.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Decided {
    /// Every field a number: integer codes, no names. The categorical sketch
    /// state of those codes.
    Codes { max_code: i32, bad: Option<f32> },
    /// Category names (codes `0..n` in first-appearance order).
    Names { n: usize },
}

impl Decide {
    fn new(missing: f32) -> Self {
        Self {
            names: CategoryBuilder::default(),
            overflow: None,
            numeric: true,
            missing,
            max_code: -1,
            bad: None,
        }
    }

    fn push(&mut self, s: &str, name: &str) -> Result<()> {
        if self.overflow.is_none() {
            if let Err(e) = self.names.intern(s, name) {
                self.overflow = Some(e);
            }
        }
        if self.numeric {
            match csv_number(s) {
                Some(v) if v == self.missing => {}
                Some(v) if crate::categorical::invalid_code(v) => {
                    self.bad.get_or_insert(v);
                }
                Some(v) => self.max_code = self.max_code.max(v as i32),
                None => self.numeric = false,
            }
        }
        match (self.numeric, self.overflow.take()) {
            (false, Some(e)) => Err(e),
            (true, overflow) => {
                self.overflow = overflow;
                Ok(())
            }
            (false, None) => Ok(()),
        }
    }

    fn decided(&self) -> Decided {
        if self.numeric {
            Decided::Codes {
                max_code: self.max_code,
                bad: self.bad,
            }
        } else {
            Decided::Names {
                n: self.names.len(),
            }
        }
    }
}

struct ColState<'t> {
    coder: Coder<'t>,
    /// Last dictionary seen and its per-entry codes: batches of one Parquet
    /// column chunk share the dictionary, so it is mapped once.
    cache: Option<(ArrayData, Vec<u32>)>,
}

impl<'t> ColState<'t> {
    fn code_str(&mut self, s: &str, name: &str) -> Result<u32> {
        match &mut self.coder {
            Coder::Learn(b) => b.intern(s, name),
            Coder::Fixed(c) => Ok(c.code_of_str(s).unwrap_or(UNSEEN)),
            Coder::Decide(_) => unreachable!("decided columns are coded by `recode` directly"),
        }
    }

    /// Codes of every dictionary entry (`MISSING` for a null entry, `UNSEEN`
    /// for a value outside a fixed list, `UNASSIGNED` for a value a learning
    /// coder has not seen in a row yet).
    fn dictionary_codes(&mut self, values: &ArrayRef, name: &str) -> Result<()> {
        let data = values.to_data();
        let hit = matches!(&self.cache, Some((d, _)) if d.ptr_eq(&data));
        if !hit {
            let mut codes = Vec::with_capacity(values.len());
            if let Some(s) = Strs::of(values.as_ref()) {
                if let Coder::Fixed(c) = &self.coder {
                    crate::ensure_input!(
                        matches!(c.values(), CategoryValues::Str(_)),
                        "feature column {name:?} holds string categories, the model's categories \
                         for it are integers; the category data type must match the training data"
                    );
                }
                for i in 0..values.len() {
                    codes.push(match (s.get(i), &self.coder) {
                        (None, _) => MISSING,
                        (Some(v), Coder::Learn(b)) => b.get(v).unwrap_or(UNASSIGNED),
                        (Some(v), Coder::Fixed(c)) => c.code_of_str(v).unwrap_or(UNSEEN),
                        (Some(_), Coder::Decide(_)) => {
                            unreachable!("CSV columns have no dictionaries")
                        }
                    });
                }
            } else {
                let Coder::Fixed(c) = &self.coder else {
                    unreachable!("only string dictionaries are learned")
                };
                crate::ensure_input!(
                    matches!(c.values(), CategoryValues::Int { .. }),
                    "feature column {name:?} holds integer categories, the model's categories for \
                     it are strings; the category data type must match the training data"
                );
                for v in int_values(values)? {
                    codes.push(match v {
                        None => MISSING,
                        Some(None) => UNSEEN,
                        Some(Some(x)) => c.code_of_int(x).unwrap_or(UNSEEN),
                    });
                }
            }
            self.cache = Some((data, codes));
        }
        Ok(())
    }

    /// Codes of a dictionary column's rows.
    fn code_dictionary(&mut self, column: &ArrayRef, name: &str) -> Result<Vec<f32>> {
        let dict = column.as_any_dictionary();
        let keys = dict.normalized_keys();
        let values = dict.values();
        self.dictionary_codes(values, name)?;
        let map = &mut self.cache.as_mut().expect("filled above").1;
        let mut out = Vec::with_capacity(keys.len());
        for (i, &k) in keys.iter().enumerate() {
            if column.is_null(i) {
                out.push(f32::NAN);
                continue;
            }
            let code = match map[k] {
                UNASSIGNED => {
                    let Coder::Learn(b) = &mut self.coder else {
                        unreachable!("only learning coders leave entries unassigned")
                    };
                    let s = Strs::of(values.as_ref()).expect("learned dictionaries hold strings");
                    let code = b.intern(s.get(k).expect("null entries are MISSING"), name)?;
                    map[k] = code;
                    code
                }
                code => code,
            };
            out.push(match code {
                MISSING => f32::NAN,
                UNSEEN => {
                    let shown = match Strs::of(values.as_ref()) {
                        Some(s) => format!("{:?}", s.get(k).unwrap_or("")),
                        None => arrow::util::display::array_value_to_string(values, k)
                            .unwrap_or_else(|_| "?".into()),
                    };
                    return Err(unseen_error(name, &shown));
                }
                code => code as f32,
            });
        }
        Ok(out)
    }
}

fn unseen_error(name: &str, value: &str) -> anyhow::Error {
    crate::input_err!(
        "feature column {name:?} holds the category {value}, which is not in the model's \
         categories for this feature (categories unseen in training are rejected, as in XGBoost)"
    )
}

/// Replaces named-category columns of batches with `Float32` codes.
pub(crate) struct Recoder<'t> {
    /// `features[f]` = how feature `f` is coded (`None` = left alone).
    features: Vec<Option<ColState<'t>>>,
    any: bool,
}

impl<'t> Recoder<'t> {
    /// A fixed recoder over `table` (no-op without a table).
    pub(crate) fn fixed(table: Option<&'t CategoryTable>) -> Self {
        let features: Vec<Option<ColState<'t>>> = match table {
            None => Vec::new(),
            Some(t) => t
                .features()
                .iter()
                .map(|c| {
                    c.as_ref().map(|c| ColState {
                        coder: Coder::Fixed(c),
                        cache: None,
                    })
                })
                .collect(),
        };
        let any = features.iter().any(Option::is_some);
        Self { features, any }
    }

    /// A learning recoder for training pass 1: categorical features (`flags`)
    /// with a list in `seed` (a continued model) are coded with it; the other
    /// categorical features learn their list from string columns.
    #[cfg(test)]
    pub(crate) fn learning(flags: &[bool], seed: Option<&'t CategoryTable>) -> Self {
        Self::learning_deciding(flags, &[], seed, f32::NAN)
    }

    /// [`Self::learning`] where the features marked in `decide` (declared
    /// categorical CSV columns without a seeded list, read as text) are
    /// decided from every field: integer codes when all non-empty fields are
    /// numbers, else names. Their columns become all-missing for the sketch;
    /// the caller sets the sketch from [`Self::decided`] after the pass.
    /// `missing` is the feature sentinel, which a number equal to it is.
    pub(crate) fn learning_deciding(
        flags: &[bool],
        decide: &[bool],
        seed: Option<&'t CategoryTable>,
        missing: f32,
    ) -> Self {
        let features: Vec<Option<ColState<'t>>> = flags
            .iter()
            .enumerate()
            .map(|(f, &cat)| {
                if !cat {
                    return None;
                }
                Some(ColState {
                    coder: match seed.and_then(|t| t.get(f)) {
                        Some(c) => Coder::Fixed(c),
                        None if decide.get(f).copied().unwrap_or(false) => {
                            Coder::Decide(Box::new(Decide::new(missing)))
                        }
                        None => Coder::Learn(CategoryBuilder::default()),
                    },
                    cache: None,
                })
            })
            .collect();
        Self {
            any: true,
            features,
        }
    }

    /// What pass 1 decided for each decided feature, by feature id.
    pub(crate) fn decided(&self) -> Vec<(usize, Decided)> {
        self.features
            .iter()
            .enumerate()
            .filter_map(|(f, s)| match s {
                Some(ColState {
                    coder: Coder::Decide(d),
                    ..
                }) => Some((f, d.decided())),
                _ => None,
            })
            .collect()
    }

    /// The learned table: seeded lists plus learned non-empty lists (a
    /// decided feature has its list only when decided as names). `None` when
    /// no feature has a list.
    pub(crate) fn into_table(self, n_features: usize) -> Option<CategoryTable> {
        let mut out: Vec<Option<FeatureCategories>> = Vec::with_capacity(n_features);
        let mut states = self.features.into_iter();
        for _ in 0..n_features {
            out.push(match states.next().flatten() {
                None => None,
                Some(ColState {
                    coder: Coder::Fixed(c),
                    ..
                }) => Some(c.clone()),
                Some(ColState {
                    coder: Coder::Learn(b),
                    ..
                }) => (!b.is_empty()).then(|| b.finish()),
                Some(ColState {
                    coder: Coder::Decide(d),
                    ..
                }) => (!d.numeric).then(|| d.names.finish()),
            });
        }
        CategoryTable::new(out).non_empty()
    }

    /// Re-code the named columns of `batch`. Column `c` holds feature
    /// `feature_of(c)` (`None` = not a feature, e.g. a weight column).
    pub(crate) fn recode(
        &mut self,
        batch: RecordBatch,
        feature_of: impl Fn(usize) -> Option<usize>,
    ) -> Result<RecordBatch> {
        if !self.any {
            return Ok(batch);
        }
        let schema = batch.schema();
        let mut replaced: Vec<(usize, ArrayRef)> = Vec::new();
        for (c, field) in schema.fields().iter().enumerate() {
            let Some(f) = feature_of(c) else { continue };
            let Some(Some(state)) = self.features.get_mut(f) else {
                continue;
            };
            let dt = field.data_type();
            let column = batch.column(c);
            let name = field.name().as_str();
            if let Coder::Decide(d) = &mut state.coder {
                let s = Strs::of(column.as_ref()).ok_or_else(|| {
                    anyhow::anyhow!("CSV categorical column {name:?} was not read as text")
                })?;
                for i in 0..column.len() {
                    if let Some(v) = s.get(i) {
                        d.push(v, name)?;
                    }
                }
                replaced.push((c, Arc::new(Float32Array::new_null(column.len()))));
                continue;
            }
            let codes: Option<Vec<f32>> = match dt {
                DataType::Dictionary(_, v) if is_string(v) || v.is_integer() => {
                    if v.is_integer() && matches!(state.coder, Coder::Learn(_)) {
                        // A numeric dictionary without a list: its values are the codes.
                        None
                    } else {
                        let out = state.code_dictionary(column, name)?;
                        Some(out)
                    }
                }
                DataType::Dictionary(_, v) => {
                    if let Coder::Fixed(c) = state.coder {
                        crate::bail_input!(
                            "feature column {name:?} is a dictionary of {v:?} values; the model's \
                             categories for it are {}s (categories must be strings or integers)",
                            c.values().kind()
                        );
                    }
                    None
                }
                t if is_string(t) => {
                    if let Coder::Fixed(c) = state.coder {
                        crate::ensure_input!(
                            matches!(c.values(), CategoryValues::Str(_)),
                            "feature column {name:?} holds string categories, the model's \
                             categories for it are integers; the category data type must match \
                             the training data"
                        );
                    }
                    let s = Strs::of(column.as_ref()).expect("string column");
                    let mut out = Vec::with_capacity(column.len());
                    for i in 0..column.len() {
                        out.push(match s.get(i) {
                            None => f32::NAN,
                            Some(v) => match state.code_str(v, name)? {
                                UNSEEN => return Err(unseen_error(name, &format!("{v:?}"))),
                                code => code as f32,
                            },
                        });
                    }
                    Some(out)
                }
                // Plain integers of a feature with integer categories are
                // category values (a CSV column, a plain Parquet column), coded
                // by value like an integer dictionary. Without an integer list
                // they stay the codes.
                t if t.is_integer() => match state.coder {
                    Coder::Fixed(c) if matches!(c.values(), CategoryValues::Int { .. }) => {
                        let mut out = Vec::with_capacity(column.len());
                        for (i, v) in int_values(column)?.into_iter().enumerate() {
                            out.push(match (v, v.flatten().and_then(|x| c.code_of_int(x))) {
                                (None, _) => f32::NAN,
                                (_, Some(code)) => code as f32,
                                _ => {
                                    let shown =
                                        arrow::util::display::array_value_to_string(column, i)
                                            .unwrap_or_else(|_| "?".into());
                                    return Err(unseen_error(name, &shown));
                                }
                            });
                        }
                        Some(out)
                    }
                    _ => None,
                },
                _ => None,
            };
            if let Some(codes) = codes {
                replaced.push((c, Arc::new(Float32Array::from(codes))));
            }
        }
        if replaced.is_empty() {
            return Ok(batch);
        }
        let mut fields: Vec<Arc<Field>> = schema.fields().iter().cloned().collect();
        let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
        for (c, array) in replaced {
            fields[c] = Arc::new(Field::new(fields[c].name(), DataType::Float32, true));
            columns[c] = array;
        }
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
            columns,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{DictionaryArray, Float64Array};
    use arrow::datatypes::Int32Type;

    fn dict(values: &[&str], keys: &[Option<i32>]) -> ArrayRef {
        let keys = arrow::array::Int32Array::from(keys.to_vec());
        Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                keys,
                Arc::new(StringArray::from(values.to_vec())),
            )
            .unwrap(),
        )
    }

    fn batch(cols: Vec<(&str, ArrayRef)>) -> RecordBatch {
        RecordBatch::try_from_iter(cols).unwrap()
    }

    fn codes(b: &RecordBatch, c: usize) -> Vec<f32> {
        b.column(c)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    #[test]
    fn learning_is_first_seen_across_dictionaries_and_fixed_recodes_by_value() {
        let mut r = Recoder::learning(&[true, false], None);
        let b1 = batch(vec![
            ("c", dict(&["x", "y", "unused"], &[Some(1), None, Some(0)])),
            ("n", Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]))),
        ]);
        let b2 = batch(vec![
            ("c", dict(&["z", "y", "x"], &[Some(0), Some(2), Some(1)])),
            ("n", Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]))),
        ]);
        let o1 = r.recode(b1, Some).unwrap();
        let o2 = r.recode(b2, Some).unwrap();
        // First appearance in rows: y, x, then z ("unused" is never used).
        assert_eq!(codes(&o1, 0)[0], 0.0);
        assert!(codes(&o1, 0)[1].is_nan());
        assert_eq!(codes(&o1, 0)[2], 1.0);
        assert_eq!(codes(&o2, 0), vec![2.0, 1.0, 0.0]);
        assert_eq!(o2.schema().field(1).data_type(), &DataType::Float64);
        let t = r.into_table(2).unwrap();
        assert_eq!(
            t.get(0).unwrap().values(),
            &CategoryValues::Str(vec!["y".into(), "x".into(), "z".into()])
        );
        assert!(t.get(1).is_none());

        let mut fixed = Recoder::fixed(Some(&t));
        let b3 = batch(vec![
            ("c", dict(&["new", "z", "x"], &[Some(1), Some(2)])),
            ("n", Arc::new(Float64Array::from(vec![1.0, 2.0]))),
        ]);
        assert_eq!(codes(&fixed.recode(b3, Some).unwrap(), 0), vec![2.0, 1.0]);
        let b4 = batch(vec![
            ("c", dict(&["new", "z"], &[Some(1), Some(0)])),
            ("n", Arc::new(Float64Array::from(vec![1.0, 2.0]))),
        ]);
        let err = fixed.recode(b4, Some).unwrap_err().to_string();
        assert!(
            err.contains("\"new\"") && err.contains("not in the model's categories"),
            "{err}"
        );
        // Plain string column.
        let b5 = batch(vec![
            ("c", Arc::new(StringArray::from(vec![Some("z"), None]))),
            ("n", Arc::new(Float64Array::from(vec![1.0, 2.0]))),
        ]);
        let o5 = fixed.recode(b5, Some).unwrap();
        assert_eq!(codes(&o5, 0)[0], 2.0);
        assert!(codes(&o5, 0)[1].is_nan());
    }

    #[test]
    fn integer_dictionaries_keep_values_without_a_list_and_recode_with_one() {
        let keys = arrow::array::Int32Array::from(vec![0, 1, 2]);
        let d: ArrayRef = Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                keys,
                Arc::new(arrow::array::Int64Array::from(vec![10i64, 3, 7])),
            )
            .unwrap(),
        );
        let mut learn = Recoder::learning(&[true], None);
        let out = learn.recode(batch(vec![("k", d.clone())]), Some).unwrap();
        assert!(matches!(
            out.schema().field(0).data_type(),
            DataType::Dictionary(..)
        ));
        assert!(learn.into_table(1).is_none());
        let t = CategoryTable::new(vec![Some(
            FeatureCategories::new(CategoryValues::Int {
                type_code: 15,
                values: vec![3, 7, 10],
            })
            .unwrap(),
        )]);
        let mut fixed = Recoder::fixed(Some(&t));
        assert_eq!(
            codes(&fixed.recode(batch(vec![("k", d)]), Some).unwrap(), 0),
            vec![2.0, 0.0, 1.0]
        );
        // Plain integers (a CSV or plain Parquet column) are values too.
        let plain: ArrayRef =
            Arc::new(arrow::array::Int64Array::from(vec![Some(7), None, Some(3)]));
        let out = fixed.recode(batch(vec![("k", plain)]), Some).unwrap();
        assert_eq!(codes(&out, 0)[0], 1.0);
        assert!(codes(&out, 0)[1].is_nan());
        assert_eq!(codes(&out, 0)[2], 0.0);
        let unseen: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![4i64]));
        let err = fixed
            .recode(batch(vec![("k", unseen)]), Some)
            .unwrap_err()
            .to_string();
        assert!(err.contains("\"k\"") && err.contains("category 4"), "{err}");
        // Without a list they stay the codes.
        let mut learn = Recoder::learning(&[true], None);
        let plain: ArrayRef = Arc::new(arrow::array::Int64Array::from(vec![7i64]));
        let out = learn.recode(batch(vec![("k", plain)]), Some).unwrap();
        assert_eq!(out.schema().field(0).data_type(), &DataType::Int64);
        let mut mismatch = Recoder::fixed(Some(&t));
        let err = mismatch
            .recode(batch(vec![("k", dict(&["a"], &[Some(0)]))]), Some)
            .unwrap_err()
            .to_string();
        assert!(err.contains("must match"), "{err}");
    }

    #[test]
    fn csv_numbers_follow_the_csv_inference_patterns() {
        for (s, v) in [
            ("0", 0.0),
            ("-7", -7.0),
            ("0012", 12.0),
            ("1.5", 1.5),
            ("2.", 2.0),
            (".25", 0.25),
            ("-.5", -0.5),
            ("1e3", 1000.0),
            ("1.5E-1", 0.15),
            ("2e+2", 200.0),
        ] {
            assert_eq!(csv_number(s), Some(v as f32), "{s}");
        }
        for s in [
            "",
            "-",
            ".",
            "+1",
            " 1",
            "1 ",
            "1e",
            "e3",
            "1.5e",
            "1x",
            "x",
            "NaN",
            "inf",
            "true",
            "1,000",
            "1.2.3",
            "--1",
            "\u{0661}",             // Arabic-Indic one: ASCII digits only
            "99999999999999999999", // outside i64: text, as the inference reads it
        ] {
            assert_eq!(csv_number(s), None, "{s:?}");
        }
        assert_eq!(csv_number("9223372036854775807"), Some(i64::MAX as f32));
    }

    fn text(values: &[Option<&str>]) -> ArrayRef {
        Arc::new(StringArray::from(values.to_vec()))
    }

    #[test]
    fn decided_columns_learn_names_and_codes_side_by_side() {
        // Feature 0 is numbers in every batch, feature 1 turns to text in the
        // last batch: codes for 0 (with the code scan), names for 1 in order of
        // first appearance, the numbers before the text included.
        let mut r = Recoder::learning_deciding(&[true, true], &[true, true], None, -1.0);
        let batches = [
            (
                vec![Some("3"), None, Some("-1")],
                vec![Some("2"), Some("10"), None],
            ),
            (
                vec![Some("7"), Some("1.0"), Some("4")],
                vec![Some("x"), Some("2"), Some("10")],
            ),
        ];
        for (a, b) in batches {
            let out = r
                .recode(batch(vec![("a", text(&a)), ("b", text(&b))]), Some)
                .unwrap();
            for c in 0..2 {
                assert_eq!(out.column(c).data_type(), &DataType::Float32);
                assert_eq!(out.column(c).null_count(), out.num_rows());
            }
        }
        assert_eq!(
            r.decided(),
            vec![
                (
                    0,
                    Decided::Codes {
                        max_code: 7,
                        bad: None
                    }
                ),
                (1, Decided::Names { n: 3 })
            ]
        );
        let t = r.into_table(2).unwrap();
        assert!(t.get(0).is_none());
        assert_eq!(
            t.get(1).unwrap().values(),
            &CategoryValues::Str(vec!["2".into(), "10".into(), "x".into()])
        );
        // An invalid code is reported only if the column stays numbers.
        let mut r = Recoder::learning_deciding(&[true, true], &[true, true], None, f32::NAN);
        let a = [Some("1.5"), Some("2")];
        let b = [Some("-3"), Some("y")];
        r.recode(batch(vec![("a", text(&a)), ("b", text(&b))]), Some)
            .unwrap();
        assert_eq!(
            r.decided(),
            vec![
                (
                    0,
                    Decided::Codes {
                        max_code: 2,
                        bad: Some(1.5)
                    }
                ),
                (1, Decided::Names { n: 2 })
            ]
        );
    }

    #[test]
    fn decided_columns_raise_the_category_limit_only_as_names() {
        let n = crate::category_names::MAX_CATEGORIES + 1;
        let codes: Vec<String> = (0..n).map(|i| format!("{:06}", i % 60_000)).collect();
        let more: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        let mut r = Recoder::learning_deciding(&[true], &[true], None, f32::NAN);
        // Distinct spellings past the limit: fine while every field is a number.
        for values in [&codes, &more] {
            let column: Vec<Option<&str>> = values.iter().map(|s| Some(s.as_str())).collect();
            r.recode(batch(vec![("k", text(&column))]), Some).unwrap();
        }
        assert!(matches!(r.decided()[0].1, Decided::Codes { .. }));
        let err = r
            .recode(batch(vec![("k", text(&[Some("z")]))]), Some)
            .unwrap_err()
            .to_string();
        assert!(err.contains("more than") && err.contains("\"k\""), "{err}");
    }
}
