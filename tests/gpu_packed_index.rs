#![cfg(feature = "cuda")]

use ferrisboost::{
    backend::cuda::{GpuRowSpan, GpuTrainCtx, SharedHistogramConfig},
    hist,
    types::{GradPairFixed, RowId, MISSING_BIN},
};

const N_ROWS: usize = 2049;
const N_FEATURES: usize = 4;
const FULL_OFFSETS: [u32; N_FEATURES + 1] = [0, 3, 8, 12, 14];
const SELECTED_PLANS: [&[usize]; 5] = [&[0, 2, 3], &[1, 3], &[0, 1, 3], &[0, 2], &[0, 2, 3]];
const NODE_COUNTS: [usize; 5] = [32, 2, 16, 1, 32];

fn fixture() -> (Vec<u8>, Vec<GradPairFixed>) {
    let bins = (0..N_FEATURES)
        .flat_map(|feat| {
            let width = (FULL_OFFSETS[feat + 1] - FULL_OFFSETS[feat]) as usize;
            (0..N_ROWS).map(move |row| {
                if (row * 13 + feat * 7) % 41 == 0 {
                    MISSING_BIN
                } else {
                    ((row * 11 + feat * 5 + row / 17) % width) as u8
                }
            })
        })
        .collect();
    let gpair = (0..N_ROWS)
        .map(|row| GradPairFixed {
            grad: match row % 13 {
                0 => -257,
                1 => 193,
                2 => -1,
                _ => (row as i64 % 29) - 14,
            },
            hess: match row % 11 {
                0 => -3,
                1 => 0,
                _ => (row as i64 % 17) + 1,
            },
        })
        .collect();
    (bins, gpair)
}

fn compact_offsets(selected: &[usize]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(selected.len() + 1);
    offsets.push(0);
    for &feat in selected {
        offsets.push(offsets.last().unwrap() + FULL_OFFSETS[feat + 1] - FULL_OFFSETS[feat]);
    }
    offsets
}

fn spans_for(n_nodes: usize) -> Vec<GpuRowSpan> {
    (0..n_nodes)
        .map(|node| match node {
            // Exact 4 × 512 and its one-row tail exercise both loop shapes.
            0 => GpuRowSpan {
                offset: 0,
                len: 2048,
            },
            1 => GpuRowSpan {
                offset: 2048,
                len: 1,
            },
            2 => GpuRowSpan {
                offset: 513,
                len: 511,
            },
            3 => GpuRowSpan {
                offset: 1024,
                len: 0,
            },
            _ => {
                let len = match node % 4 {
                    0 => 0,
                    1 => 1,
                    2 => 127,
                    _ => 512,
                };
                let offset = (node * 37) % (N_ROWS - len + 1);
                GpuRowSpan { offset, len }
            }
        })
        .collect()
}

fn expected_histogram(
    bins: &[u8],
    gpair: &[GradPairFixed],
    selected: &[usize],
    compact_column: bool,
    span: GpuRowSpan,
) -> Vec<GradPairFixed> {
    let rows: Vec<RowId> = (span.offset..span.offset + span.len)
        .map(|row| row as RowId)
        .collect();
    if compact_column {
        let compact_bins: Vec<u8> = selected
            .iter()
            .flat_map(|&feat| bins[feat * N_ROWS..(feat + 1) * N_ROWS].iter().copied())
            .collect();
        let offsets = compact_offsets(selected);
        let mut expected = vec![GradPairFixed::default(); *offsets.last().unwrap() as usize];
        hist::build(
            &compact_bins,
            N_ROWS,
            selected.len(),
            &offsets,
            gpair,
            Some(&rows),
            &mut expected,
        );
        expected
    } else {
        let mut expected = vec![GradPairFixed::default(); FULL_OFFSETS[N_FEATURES] as usize];
        for &feat in selected {
            let feature_bins = &bins[feat * N_ROWS..(feat + 1) * N_ROWS];
            let offsets = [0, FULL_OFFSETS[feat + 1] - FULL_OFFSETS[feat]];
            let begin = FULL_OFFSETS[feat] as usize;
            let end = FULL_OFFSETS[feat + 1] as usize;
            hist::build(
                feature_bins,
                N_ROWS,
                1,
                &offsets,
                gpair,
                Some(&rows),
                &mut expected[begin..end],
            );
        }
        expected
    }
}

fn make_context(resident_blocks: usize, gpu_split_cache: bool) -> GpuTrainCtx {
    let config = SharedHistogramConfig {
        runtime_log: false,
        threads_per_block: 512,
        hist_streams: Some(1),
        hist_nodes_per_batch: Some(32),
        resident_blocks: Some(resident_blocks),
        gpu_split_cache,
        ..SharedHistogramConfig::default()
    };
    GpuTrainCtx::new(
        0,
        N_ROWS,
        N_FEATURES,
        FULL_OFFSETS[N_FEATURES] as usize,
        1,
        128,
        config,
        false,
    )
    .unwrap()
}

fn assert_hist_matches(got: &[GradPairFixed], expected: &[GradPairFixed], context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: histogram length");
    if let Some(index) = got
        .iter()
        .zip(expected)
        .position(|(got, expected)| got != expected)
    {
        panic!(
            "{context}: first mismatch at histogram index {index}: got {:?}, expected {:?}",
            got[index], expected[index]
        );
    }
}

fn run_plan(
    ctx: &GpuTrainCtx,
    bins: &[u8],
    selected: &[usize],
    n_nodes: usize,
    resident: bool,
    compact_column: bool,
    gpair: &[GradPairFixed],
) {
    let spans = spans_for(n_nodes);
    let offsets = if compact_column {
        compact_offsets(selected)
    } else {
        FULL_OFFSETS.to_vec()
    };
    let mut outputs: Vec<Vec<GradPairFixed>> = (0..n_nodes)
        .map(|_| {
            vec![
                GradPairFixed {
                    grad: 91,
                    hess: -37
                };
                19
            ]
        })
        .collect();

    let mut session = if resident {
        ctx.block_session_resident(0, N_ROWS, N_FEATURES, &offsets, Some(selected))
            .unwrap()
            .expect("the seeded resident block should be available")
    } else {
        ctx.block_session(0, bins, N_ROWS, N_FEATURES, &offsets, Some(selected))
            .unwrap()
    };
    assert!(
        session.requires_batched_hist(),
        "selected-feature plans must use the packed batched ABI"
    );
    let mut output_refs: Vec<_> = outputs.iter_mut().collect();
    session
        .histogram_device_batch(&spans, &mut output_refs)
        .unwrap();
    drop(output_refs);

    for (node, (span, got)) in spans.iter().zip(&outputs).enumerate() {
        let expected = expected_histogram(bins, gpair, selected, compact_column, *span);
        assert_hist_matches(
            got,
            &expected,
            &format!(
                "packed-index histogram mismatch: resident={resident}, selected={selected:?}, nodes={n_nodes}, node={node}, span={span:?}"
            ),
        );
    }

    // Follow populated output with an all-empty batch at both valid boundaries.
    // Reuse the same slot and start host output vectors with stale sentinel data.
    let empty_spans = [
        GpuRowSpan { offset: 0, len: 0 },
        GpuRowSpan {
            offset: N_ROWS,
            len: 0,
        },
    ];
    let mut empty_outputs = vec![
        vec![
            GradPairFixed {
                grad: 91,
                hess: -37
            };
            19
        ];
        2
    ];
    let mut empty_refs: Vec<_> = empty_outputs.iter_mut().collect();
    session
        .histogram_device_batch(&empty_spans, &mut empty_refs)
        .unwrap();
    drop(empty_refs);
    let empty_len = if compact_column {
        *compact_offsets(selected).last().unwrap() as usize
    } else {
        FULL_OFFSETS[N_FEATURES] as usize
    };
    let zero_hist = vec![GradPairFixed::default(); empty_len];
    for (index, (span, got)) in empty_spans.iter().zip(&empty_outputs).enumerate() {
        assert_hist_matches(
            got,
            &zero_hist,
            &format!(
                "all-empty batch retained prior data: resident={resident}, selected={selected:?}, nodes={n_nodes}, node={index}, span={span:?}"
            ),
        );
    }

    // An empty request has no output vectors to resize or download.
    let mut no_outputs: [&mut Vec<GradPairFixed>; 0] = [];
    session
        .histogram_device_batch(&[], &mut no_outputs)
        .unwrap();
}

#[test]
fn packed_index_batch_histograms_match_cpu_with_compact_selection_and_reused_slots() {
    let (bins, gpair) = fixture();

    let probe = make_context(0, false);
    let path = probe.path_summary();
    let compact_column = probe.supports_compact_column_histogram();
    assert!(
        compact_column,
        "expected the production compact column-major ABI: {path}"
    );
    assert!(path.contains("layout_gpu=col_major"), "{path}");
    drop(probe);

    // Streaming reuses its single temporary slot across all node counts and
    // logical feature maps.
    let streaming = make_context(0, false);
    streaming.upload_gpair(&gpair).unwrap();
    streaming.begin_tree_rows().unwrap();
    for (plan, n_nodes) in SELECTED_PLANS.iter().zip(NODE_COUNTS) {
        run_plan(
            &streaming,
            &bins,
            plan,
            n_nodes,
            false,
            compact_column,
            &gpair,
        );
    }

    // Warm one resident physical block, then repeatedly refresh only its
    // packed metadata while the same resident payload and context stay alive.
    let resident = make_context(1, true);
    resident.upload_gpair(&gpair).unwrap();
    resident.begin_tree_rows().unwrap();
    {
        let mut seed = resident
            .block_session(0, &bins, N_ROWS, N_FEATURES, &FULL_OFFSETS, None)
            .unwrap();
        let spans = [GpuRowSpan { offset: 0, len: 1 }];
        let mut output = Vec::new();
        let mut outputs = [&mut output];
        if seed.requires_batched_hist() {
            seed.histogram_device_batch(&spans, &mut outputs).unwrap();
        } else {
            seed.histogram_device(spans[0], &mut output).unwrap();
        }
    }
    for (plan, n_nodes) in SELECTED_PLANS.iter().zip(NODE_COUNTS) {
        run_plan(
            &resident,
            &bins,
            plan,
            n_nodes,
            true,
            compact_column,
            &gpair,
        );
    }
}
