use arrow::{
    array::{ArrayRef, Float32Array, Float64Array, Int32Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use ferrisboost::{
    source::parquet_source::{cuts_only_with_nthread, open_many_with_nthread, Dataset},
    BlockSource,
};
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};
use std::sync::{
    atomic::{AtomicU64, Ordering},
    Arc,
};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "fb-setup-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn write(
        &self,
        name: &str,
        first: usize,
        rows: usize,
        reordered: bool,
        group_rows: usize,
    ) -> std::path::PathBuf {
        let fields = [
            Field::new("f0", DataType::Float32, true),
            Field::new("f1", DataType::Int32, true),
            Field::new("f2", DataType::Float64, true),
            Field::new("target", DataType::Float32, false),
        ];
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(Float32Array::from(
                (first..first + rows)
                    .map(|i| {
                        if i % 17 == 0 {
                            None
                        } else {
                            Some(if i % 29 == 0 {
                                f32::NAN
                            } else {
                                (i % 113) as f32 / 7.
                            })
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                (first..first + rows)
                    .map(|i| {
                        if i % 19 == 0 {
                            None
                        } else {
                            Some((i % 97) as i32 - 48)
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                (first..first + rows)
                    .map(|i| {
                        if i % 23 == 0 {
                            None
                        } else {
                            Some((i % 139) as f64 / 9.)
                        }
                    })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (first..first + rows).map(|i| i as f32).collect::<Vec<_>>(),
            )),
        ];
        let order = if reordered {
            vec![3, 2, 0, 1]
        } else {
            vec![0, 1, 2, 3]
        };
        let schema = Arc::new(Schema::new(
            order.iter().map(|&i| fields[i].clone()).collect::<Vec<_>>(),
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            order.iter().map(|&i| arrays[i].clone()).collect(),
        )
        .unwrap();
        let path = self.0.join(name);
        let props = WriterProperties::builder()
            .set_max_row_group_size(group_rows)
            .build();
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, Some(props))
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        path
    }

    fn write_wide_nullable(
        &self,
        name: &str,
        rows: usize,
        group_rows: usize,
    ) -> std::path::PathBuf {
        const FEATURES: usize = 256;
        let fields: Vec<_> = (0..FEATURES)
            .map(|feature| Field::new(format!("feature_{feature:03}"), DataType::Float32, true))
            .chain(std::iter::once(Field::new(
                "target",
                DataType::Float32,
                false,
            )))
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let arrays: Vec<ArrayRef> = (0..FEATURES)
            .map(|feature| {
                Arc::new(Float32Array::from(
                    (0..rows)
                        .map(|row| {
                            if (row + feature * 7) % 13 == 0 {
                                None
                            } else {
                                Some(((row * 11 + feature * 7) % 97) as f32 / 9.0)
                            }
                        })
                        .collect::<Vec<_>>(),
                )) as ArrayRef
            })
            .chain(std::iter::once(Arc::new(Float32Array::from(
                (0..rows)
                    .map(|row| row as f32 * 2.0 + 0.25)
                    .collect::<Vec<_>>(),
            )) as ArrayRef))
            .collect();
        let batch = RecordBatch::try_new(schema.clone(), arrays).unwrap();
        let path = self.0.join(name);
        let props = WriterProperties::builder()
            .set_max_row_group_size(group_rows)
            .build();
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, Some(props))
                .unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        if let Ok(entries) = std::fs::read_dir(&self.0) {
            for entry in entries.flatten() {
                let _ = std::fs::remove_file(entry.path());
            }
        }
        let _ = std::fs::remove_dir(&self.0);
    }
}

fn assert_same(a: &Dataset, b: &Dataset) {
    assert_eq!(a.feature_names, b.feature_names);
    assert_eq!(a.labels, b.labels);
    assert_eq!(a.source.cuts().offsets, b.source.cuts().offsets);
    assert_eq!(
        a.source
            .cuts()
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        b.source
            .cuts()
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(a.source.n_blocks(), b.source.n_blocks());
    for i in 0..a.source.n_blocks() {
        a.source
            .with_block(i, &mut |left| {
                b.source
                    .with_block(i, &mut |right| {
                        assert_eq!(left.feat_ids, right.feat_ids);
                        assert_eq!(left.n_rows, right.n_rows);
                        assert_eq!(left.data, right.data);
                    })
                    .unwrap();
            })
            .unwrap();
    }
}

#[test]
fn setup_preserves_nullable_cuts_bins_labels_and_reordered_schema() {
    let fixture = Fixture::new();
    let paths = vec![
        fixture.write("a.parquet", 0, 9007, false, 3000),
        fixture.write("b.parquet", 9007, 8201, true, 1733),
    ];
    let reference = open_many_with_nthread(&paths, "target", None, 32, 2, 1).unwrap();
    assert_eq!(reference.feature_names, vec!["f0", "f1", "f2"]);
    assert_eq!(
        reference.labels,
        (0..17208).map(|i| i as f32).collect::<Vec<_>>()
    );
    for threads in [2, 3, 8, 0] {
        let parallel = open_many_with_nthread(&paths, "target", None, 32, 2, threads).unwrap();
        assert_same(&reference, &parallel);
        let supplied = open_many_with_nthread(
            &paths,
            "target",
            Some(reference.source.cuts().clone()),
            32,
            2,
            threads,
        )
        .unwrap();
        assert_same(&reference, &supplied);
    }
    for i in [0, 17, 29, 9007, 17207] {
        let value = if i % 17 == 0 || i % 29 == 0 {
            f32::NAN
        } else {
            (i % 113) as f32 / 7.
        };
        let expected = if value.is_nan() {
            255
        } else {
            reference.source.cuts().find_bin(0, value)
        };
        assert_eq!(reference.source.bin_at(i, 0), expected);
    }
}

#[test]
fn setup_handles_empty_files_single_groups_and_missing_labels() {
    let fixture = Fixture::new();
    let empty = fixture.write("empty.parquet", 0, 0, false, 10);
    let one = fixture.write("one.parquet", 0, 7, false, 10);
    let two = fixture.write("two.parquet", 7, 5, true, 10);
    let paths = vec![empty, one, two];
    let reference = open_many_with_nthread(&paths, "target", None, 16, 2, 1).unwrap();
    for threads in [2, 8, 0] {
        assert_same(
            &reference,
            &open_many_with_nthread(&paths, "target", None, 16, 2, threads).unwrap(),
        );
    }
    assert_eq!(
        reference.labels,
        (0..12).map(|i| i as f32).collect::<Vec<_>>()
    );
    assert!(open_many_with_nthread(&paths, "missing", None, 16, 2, 8)
        .err()
        .unwrap()
        .to_string()
        .contains("missing"));
}

#[test]
fn coalesced_wide_projection_preserves_nullable_bins_with_parallel_row_groups() {
    const FEATURES: usize = 256;
    const ROWS: usize = 37;
    let fixture = Fixture::new();
    let path = fixture.write_wide_nullable("wide.parquet", ROWS, 7);
    let paths = vec![path.clone()];

    let reference = open_many_with_nthread(&paths, "target", None, 32, 32, 1).unwrap();
    assert_eq!(reference.feature_names.len(), FEATURES);
    assert_eq!(reference.feature_names[0], "feature_000");
    assert_eq!(reference.feature_names[FEATURES - 1], "feature_255");
    let expected_labels: Vec<f32> = (0..ROWS).map(|row| row as f32 * 2.0 + 0.25).collect();
    assert_eq!(reference.labels, expected_labels);

    // The projection has exactly 256 of 257 flat roots, every row group has
    // at most seven rows, and the omitted label is under 5% of the file's
    // physical data size: it satisfies coalesced_projection's routing gates.
    let shared_cuts = cuts_only_with_nthread(&path, "target", 32, 1).unwrap();
    assert_eq!(shared_cuts.offsets, reference.source.cuts().offsets);
    assert_eq!(
        shared_cuts
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        reference
            .source
            .cuts()
            .values
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );

    for threads in [1, 2, 8] {
        let threaded_cuts = cuts_only_with_nthread(&path, "target", 32, threads).unwrap();
        assert_eq!(shared_cuts.offsets, threaded_cuts.offsets);
        assert_eq!(
            shared_cuts
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            threaded_cuts
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );

        let computed = open_many_with_nthread(&paths, "target", None, 32, 32, threads).unwrap();
        assert_same(&reference, &computed);
        let supplied =
            open_many_with_nthread(&paths, "target", Some(shared_cuts.clone()), 32, 32, threads)
                .unwrap();
        assert_same(&reference, &supplied);
    }

    for row in 0..ROWS {
        for feature in [0, 1, 12, 127, 255] {
            let expected = if (row + feature * 7) % 13 == 0 {
                255
            } else {
                let value = ((row * 11 + feature * 7) % 97) as f32 / 9.0;
                reference.source.cuts().find_bin(feature as u32, value)
            };
            assert_eq!(
                reference.source.bin_at(row, feature as u32),
                expected,
                "row={row}, feature={feature}"
            );
        }
    }
}
