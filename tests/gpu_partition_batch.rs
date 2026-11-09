#![cfg(feature = "cuda")]

use ferrisboost::{
    backend::cuda::{GpuRowSpan, GpuTrainCtx, SharedHistogramConfig},
    types::{GradPairFixed, RowId, MISSING_BIN},
};

const N_ROWS: usize = 2051;
const LARGE_ROWS: usize = 256 * 1024 + 2;
const N_FEATURES: usize = 4;
const MAX_LEVEL_NODES: usize = 8192;
const NODE_COUNTS: [usize; 5] = [32, 2, 16, 1, 32];
const FEATURES: [&[usize]; 5] = [&[0, 2, 1, 3], &[1, 0], &[3, 1, 0, 2], &[2], &[1, 3, 0, 2]];

type PartitionRequest = (GpuRowSpan, usize, u32, bool);

static GPU_PARTITION_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn bins_for(n_rows: usize, n_features: usize) -> Vec<u8> {
    (0..n_features)
        .flat_map(|feat| {
            (0..n_rows).map(move |row| {
                if (row * 17 + feat * 29 + row / 7) % 19 == 0 {
                    MISSING_BIN
                } else {
                    ((row * 11 + feat * 3 + row / 13) % 4) as u8
                }
            })
        })
        .collect()
}

fn make_context(
    n_rows: usize,
    max_features: usize,
    n_blocks: usize,
    resident_blocks: usize,
) -> GpuTrainCtx {
    let config = SharedHistogramConfig {
        runtime_log: false,
        threads_per_block: 128,
        hist_streams: Some(1),
        hist_nodes_per_batch: Some(1),
        resident_blocks: Some(resident_blocks),
        gpu_split_cache: false,
        ..SharedHistogramConfig::default()
    };
    GpuTrainCtx::new(
        0,
        n_rows,
        max_features,
        max_features * 4,
        n_blocks,
        256,
        config,
        false,
    )
    .unwrap()
}

fn assert_partition_scratch_budget(ctx: &GpuTrainCtx, n_rows: usize) {
    let threads = 256usize;
    let blocks = n_rows.min(n_rows.div_ceil(threads) + 32 - 1);
    let offsets = blocks + 32;
    let descriptors = 32 * 11;
    let expected = n_rows
        + (blocks + offsets * 2 + descriptors) * std::mem::size_of::<u32>()
        + MAX_LEVEL_NODES * 2 * std::mem::size_of::<u32>();
    assert_eq!(
        ctx.budget().partition_scratch_bytes,
        expected,
        "partition budget must include the exact bounded batch descriptor and segmented scratch allocations"
    );
}

fn seed_resident(ctx: &GpuTrainCtx, block: usize, n_rows: usize, n_features: usize, bins: &[u8]) {
    let offsets: Vec<u32> = (0..=n_features).map(|feat| (feat * 4) as u32).collect();
    let gpair = vec![GradPairFixed { grad: 1, hess: 1 }; n_rows];
    ctx.upload_gpair(&gpair).unwrap();
    ctx.begin_tree_rows().unwrap();
    {
        let mut session = ctx
            .block_session(block, bins, n_rows, n_features, &offsets, None)
            .unwrap();
        let mut histogram = Vec::new();
        let span = GpuRowSpan {
            offset: 0,
            len: n_rows,
        };
        if session.requires_batched_hist() {
            let spans = [span];
            let mut outputs = [&mut histogram];
            session
                .histogram_device_batch(&spans, &mut outputs)
                .unwrap();
        } else {
            session.histogram_device(span, &mut histogram).unwrap();
        }
    }
}

// Keep the parity expression explicit in this deterministic span fixture.
#[allow(clippy::manual_is_multiple_of)]
fn span_for(index: usize) -> GpuRowSpan {
    match index {
        0 => GpuRowSpan {
            offset: 0,
            len: 513,
        },
        1 => GpuRowSpan {
            offset: 514,
            len: 511,
        },
        2 => GpuRowSpan {
            offset: 1026,
            len: 513,
        },
        3 => GpuRowSpan {
            offset: 1540,
            len: 511,
        },
        _ => GpuRowSpan {
            offset: if index % 2 == 0 { 513 } else { N_ROWS },
            len: 0,
        },
    }
}

fn requests_for(n_nodes: usize, features: &[usize]) -> Vec<PartitionRequest> {
    (0..n_nodes)
        .map(|index| {
            let (split_bin, missing_left) = match index {
                // Every valid and missing bin goes left, then every row goes right.
                0 => (u32::MAX, true),
                1 => (0, false),
                _ => ((index % 5) as u32, index % 2 == 0),
            };
            (
                span_for(index),
                features[index % features.len()],
                split_bin,
                missing_left,
            )
        })
        .collect()
}

fn request_for_span(span: GpuRowSpan, index: usize) -> PartitionRequest {
    let (split_bin, missing_left) = match index % 4 {
        0 => (u32::MAX, true),
        1 => (0, false),
        2 => (2, true),
        _ => (3, false),
    };
    (span, index % N_FEATURES, split_bin, missing_left)
}

fn segmented_32_requests() -> Vec<PartitionRequest> {
    let mut spans: Vec<_> = (0..31)
        .map(|fragment| GpuRowSpan {
            offset: fragment * 2,
            len: 1,
        })
        .collect();
    spans.push(GpuRowSpan {
        offset: 61,
        len: N_ROWS - 61,
    });
    // Reverse spatial order so request order differs from input-row order.
    spans.reverse();
    spans
        .into_iter()
        .enumerate()
        .map(|(index, span)| request_for_span(span, index))
        .collect()
}

fn empty_interspersed_32_requests() -> Vec<PartitionRequest> {
    let mut requests = vec![
        (GpuRowSpan { offset: 0, len: 0 }, 0, 1, false),
        (GpuRowSpan { offset: 0, len: 0 }, 1, 3, true),
    ];
    for row in 0..27 {
        if row == 16 {
            // Duplicate empty spans at a block boundary must not consume or
            // alias the neighboring nonempty segments' output reservations.
            requests.push((
                GpuRowSpan {
                    offset: 256,
                    len: 0,
                },
                row % N_FEATURES,
                2,
                false,
            ));
            requests.push((
                GpuRowSpan {
                    offset: 256,
                    len: 0,
                },
                (row + 1) % N_FEATURES,
                3,
                true,
            ));
        }
        requests.push(request_for_span(
            GpuRowSpan {
                offset: 240 + row,
                len: 1,
            },
            row,
        ));
    }
    requests.push((
        GpuRowSpan {
            offset: N_ROWS,
            len: 0,
        },
        2,
        0,
        false,
    ));
    assert_eq!(requests.len(), 32);
    requests
}

fn goes_left(bin: u8, split_bin: u32, missing_left: bool) -> bool {
    if bin == MISSING_BIN {
        missing_left
    } else {
        u32::from(bin) < split_bin
    }
}

fn assert_rows_equal(got: &[RowId], expected: &[RowId], context: &str) {
    assert_eq!(got.len(), expected.len(), "{context}: row count");
    if let Some(index) = got.iter().zip(expected).position(|(got, want)| got != want) {
        panic!(
            "{context}: first row mismatch at index {index}: got {}, expected {}",
            got[index], expected[index]
        );
    }
}

fn run_batch(
    ctx: &GpuTrainCtx,
    bins: &[u8],
    n_rows: usize,
    block: usize,
    requests: &[PartitionRequest],
) -> Vec<(GpuRowSpan, GpuRowSpan)> {
    let source_rows: Vec<Vec<RowId>> = requests
        .iter()
        .map(|(span, _, _, _)| ctx.download_rows(*span).unwrap())
        .collect();
    let mut session = ctx
        .partition_session_resident(block, 0)
        .unwrap()
        .expect("resident block zero was seeded before partitioning");
    let tokens = session.partition_resident_batch(requests).unwrap();
    assert_eq!(tokens.len(), requests.len(), "one token per request");
    drop(session);

    let resolved = ctx.resolve_partitions().unwrap();
    assert_eq!(
        resolved.len(),
        requests.len(),
        "one resolved pair per token"
    );
    ctx.finish_partition_level().unwrap();
    for (request_index, ((span, feature, split, missing_left), token)) in
        requests.iter().zip(&tokens).enumerate()
    {
        assert_eq!(
            *token, request_index,
            "partition tokens preserve request order"
        );
        let (left_span, right_span) = resolved[*token];
        let mut expected_left = Vec::new();
        let mut expected_right = Vec::new();
        for &row in &source_rows[request_index] {
            let bin = bins[*feature * n_rows + row as usize];
            if goes_left(bin, *split, *missing_left) {
                expected_left.push(row);
            } else {
                expected_right.push(row);
            }
        }

        let got_left = ctx.download_rows(left_span).unwrap();
        let got_right = ctx.download_rows(right_span).unwrap();
        assert_rows_equal(
            &got_left,
            &expected_left,
            &format!("request {request_index}, span={span:?}, feature={feature}, split={split}, missing_left={missing_left}, left"),
        );
        assert_rows_equal(
            &got_right,
            &expected_right,
            &format!("request {request_index}, span={span:?}, feature={feature}, split={split}, missing_left={missing_left}, right"),
        );
    }
    resolved
}

#[test]
fn resident_partition_batch_matches_stable_cpu_rows_and_rejects_invalid_levels() {
    let _gpu_guard = GPU_PARTITION_TEST_LOCK.lock().unwrap();
    // The harness runs this GPU test in isolation; the training layout is
    // always column-major.
    let bins = bins_for(N_ROWS, N_FEATURES);
    let ctx = make_context(N_ROWS, N_FEATURES, 1, 1);
    assert_partition_scratch_budget(&ctx, N_ROWS);
    seed_resident(&ctx, 0, N_ROWS, N_FEATURES, &bins);

    for (features, n_nodes) in FEATURES.iter().zip(NODE_COUNTS) {
        ctx.begin_tree_rows().unwrap();
        let requests = requests_for(n_nodes, features);
        let resolved = run_batch(&ctx, &bins, N_ROWS, 0, &requests);
        assert_eq!(resolved.len(), n_nodes);
    }

    // Exercise nearly the maximum segmented block count for 32 disjoint
    // requests: 31 one-row fragments plus a long tail, submitted in reverse
    // spatial order. For N=2051 and T=256 this needs 39 blocks of the planned
    // 40-block capacity.
    ctx.begin_tree_rows().unwrap();
    let segmented = segmented_32_requests();
    assert_eq!(segmented.len(), 32);
    let blocks_needed: usize = segmented
        .iter()
        .map(|(span, _, _, _)| span.len.div_ceil(256))
        .sum();
    assert_eq!(blocks_needed, 39);
    let segmented_output = run_batch(&ctx, &bins, N_ROWS, 0, &segmented);
    assert_eq!(segmented_output.len(), 32);

    // Leading and interspersed zero spans (including duplicate block-boundary
    // offsets) preserve one token per request while neighboring rows remain
    // stable.
    ctx.begin_tree_rows().unwrap();
    let with_empties = empty_interspersed_32_requests();
    let empty_mixed_output = run_batch(&ctx, &bins, N_ROWS, 0, &with_empties);
    assert_eq!(empty_mixed_output.len(), 32);

    // Follow the first split with a second batched level over its real child
    // spans. This verifies that finish_partition_level swaps arenas and that
    // later requests consume the children written by the prior level.
    ctx.begin_tree_rows().unwrap();
    let root = [(
        GpuRowSpan {
            offset: 0,
            len: N_ROWS,
        },
        0,
        2,
        true,
    )];
    let root_children = run_batch(&ctx, &bins, N_ROWS, 0, &root);
    let second_level = [
        (root_children[0].0, 1, 1, false),
        (root_children[0].1, 3, 3, true),
    ];
    let second_children = run_batch(&ctx, &bins, N_ROWS, 0, &second_level);
    assert_eq!(second_children.len(), 2);

    // An all-empty request list returns no tokens and launches no kernels.
    ctx.begin_tree_rows().unwrap();
    {
        let mut session = ctx.partition_session_resident(0, 0).unwrap().unwrap();
        let no_requests: [PartitionRequest; 0] = [];
        assert!(session
            .partition_resident_batch(&no_requests)
            .unwrap()
            .is_empty());
        assert_eq!(session.timing().kernel_calls, 0);
    }
    assert!(ctx.resolve_partitions().unwrap().is_empty());
    ctx.finish_partition_level().unwrap();

    // Empty spans at the first row, an interior gap, and the end produce tokens
    // without kernels and resolve to empty children.
    ctx.begin_tree_rows().unwrap();
    let empty_requests = [
        (GpuRowSpan { offset: 0, len: 0 }, 0, 1, false),
        (
            GpuRowSpan {
                offset: 513,
                len: 0,
            },
            2,
            3,
            true,
        ),
        (
            GpuRowSpan {
                offset: N_ROWS,
                len: 0,
            },
            1,
            0,
            false,
        ),
    ];
    {
        let mut session = ctx.partition_session_resident(0, 0).unwrap().unwrap();
        let tokens = session.partition_resident_batch(&empty_requests).unwrap();
        assert_eq!(tokens, [0, 1, 2]);
        assert_eq!(session.timing().kernel_calls, 0);
    }
    let empty_outputs = ctx.resolve_partitions().unwrap();
    assert_eq!(empty_outputs.len(), empty_requests.len());
    assert!(empty_outputs
        .iter()
        .all(|(left, right)| left.len == 0 && right.len == 0));
    ctx.finish_partition_level().unwrap();

    // Each invalid call must leave both the output cursor and pending-token
    // vector unchanged. The valid singleton must therefore still get token 0.
    ctx.begin_tree_rows().unwrap();
    {
        let mut session = ctx.partition_session_resident(0, 0).unwrap().unwrap();
        let too_many = vec![(GpuRowSpan { offset: 0, len: 0 }, 0, 1, false); 33];
        assert!(session.partition_resident_batch(&too_many).is_err());
        let invalid_empty_end = [(
            GpuRowSpan {
                offset: N_ROWS + 1,
                len: 0,
            },
            0,
            1,
            false,
        )];
        assert!(session
            .partition_resident_batch(&invalid_empty_end)
            .is_err());
        let invalid_span = [(
            GpuRowSpan {
                offset: N_ROWS,
                len: 1,
            },
            0,
            1,
            false,
        )];
        assert!(session.partition_resident_batch(&invalid_span).is_err());
        let invalid_feature = [(GpuRowSpan { offset: 0, len: 1 }, N_FEATURES, 1, false)];
        assert!(session.partition_resident_batch(&invalid_feature).is_err());
        let overlapping = [
            (GpuRowSpan { offset: 0, len: 20 }, 0, 1, false),
            (
                GpuRowSpan {
                    offset: 10,
                    len: 20,
                },
                1,
                1,
                false,
            ),
        ];
        assert!(session.partition_resident_batch(&overlapping).is_err());
        let valid = [(GpuRowSpan { offset: 0, len: 20 }, 2, 2, true)];
        assert_eq!(session.partition_resident_batch(&valid).unwrap(), [0]);
    }
    let after_invalid = ctx.resolve_partitions().unwrap();
    assert_eq!(after_invalid.len(), 1);
    assert_eq!(
        after_invalid[0].0.offset, 0,
        "invalid requests advanced the output cursor"
    );
    assert_eq!(after_invalid[0].0.len + after_invalid[0].1.len, 20);
    ctx.finish_partition_level().unwrap();

    // Accumulating N-1 output rows leaves room for exactly one row. A failed
    // two-row request must leave that final row available to a valid request.
    ctx.begin_tree_rows().unwrap();
    {
        let mut session = ctx.partition_session_resident(0, 0).unwrap().unwrap();
        let first = [(
            GpuRowSpan {
                offset: 0,
                len: N_ROWS - 1,
            },
            0,
            2,
            true,
        )];
        assert_eq!(session.partition_resident_batch(&first).unwrap(), [0]);
        let overflow = [(GpuRowSpan { offset: 0, len: 2 }, 1, 2, false)];
        assert!(session.partition_resident_batch(&overflow).is_err());
        let final_row = [(
            GpuRowSpan {
                offset: N_ROWS - 1,
                len: 1,
            },
            1,
            2,
            false,
        )];
        assert_eq!(session.partition_resident_batch(&final_row).unwrap(), [1]);
    }
    let after_overflow = ctx.resolve_partitions().unwrap();
    assert_eq!(
        after_overflow.len(),
        2,
        "overflow request changed pending slots"
    );
    assert_eq!(after_overflow[0].0.offset, 0);
    assert_eq!(
        after_overflow[0].0.len + after_overflow[0].1.len,
        N_ROWS - 1
    );
    assert_eq!(
        after_overflow[1].0.offset,
        N_ROWS - 1,
        "overflow request advanced the output cursor"
    );
    assert_eq!(after_overflow[1].0.len + after_overflow[1].1.len, 1);
    ctx.finish_partition_level().unwrap();

    // Fill the bounded pending table with empty entries only (no kernels),
    // then ensure the next request is rejected without changing its size.
    ctx.begin_tree_rows().unwrap();
    {
        let mut session = ctx.partition_session_resident(0, 0).unwrap().unwrap();
        let empty_chunk = [(GpuRowSpan { offset: 0, len: 0 }, 0, 1, false); 32];
        for chunk in 0..MAX_LEVEL_NODES / empty_chunk.len() {
            let tokens = session.partition_resident_batch(&empty_chunk).unwrap();
            assert_eq!(tokens.first().copied(), Some(chunk * empty_chunk.len()));
            assert_eq!(
                tokens.last().copied(),
                Some((chunk + 1) * empty_chunk.len() - 1)
            );
        }
        assert!(session
            .partition_resident_batch(&[(GpuRowSpan { offset: 0, len: 0 }, 0, 1, false)])
            .is_err());
        assert_eq!(session.timing().kernel_calls, 0);
    }
    assert_eq!(ctx.resolve_partitions().unwrap().len(), MAX_LEVEL_NODES);
    ctx.finish_partition_level().unwrap();

    // A narrow final resident block is validated against its actual width (2),
    // not the context's maximum block width (4).
    let narrow_ctx = make_context(N_ROWS, N_FEATURES, 2, 2);
    assert_partition_scratch_budget(&narrow_ctx, N_ROWS);
    let narrow_bins = bins_for(N_ROWS, 2);
    seed_resident(&narrow_ctx, 1, N_ROWS, 2, &narrow_bins);
    narrow_ctx.begin_tree_rows().unwrap();
    {
        let mut session = narrow_ctx
            .partition_session_resident(1, 0)
            .unwrap()
            .unwrap();
        let invalid = [(GpuRowSpan { offset: 0, len: 1 }, 2, 1, false)];
        assert!(session.partition_resident_batch(&invalid).is_err());
    }
    let narrow_output = run_batch(
        &narrow_ctx,
        &narrow_bins,
        N_ROWS,
        1,
        &[(
            GpuRowSpan {
                offset: 0,
                len: 513,
            },
            1,
            3,
            true,
        )],
    );
    assert_eq!(narrow_output.len(), 1);
}

#[test]
fn resident_partition_batch_accumulates_32_plus_1_tokens_before_one_resolve() {
    let _gpu_guard = GPU_PARTITION_TEST_LOCK.lock().unwrap();
    let bins = bins_for(N_ROWS, N_FEATURES);
    let ctx = make_context(N_ROWS, N_FEATURES, 1, 1);
    seed_resident(&ctx, 0, N_ROWS, N_FEATURES, &bins);
    ctx.begin_tree_rows().unwrap();

    let requests: Vec<PartitionRequest> = (0..33)
        .map(|index| {
            (
                GpuRowSpan {
                    offset: index * 16,
                    len: 16,
                },
                index % N_FEATURES,
                2,
                false,
            )
        })
        .collect();
    let source_rows: Vec<Vec<RowId>> = requests
        .iter()
        .map(|(span, _, _, _)| ctx.download_rows(*span).unwrap())
        .collect();

    // Keep one session open across the API's 32-request batch limit. The tail
    // call must append to the same pending-token/output-scratch transaction;
    // resolving or finishing between calls would miss cross-call accounting.
    let mut session = ctx.partition_session_resident(0, 0).unwrap().unwrap();
    let first_tokens = session.partition_resident_batch(&requests[..32]).unwrap();
    assert_eq!(first_tokens, (0..32).collect::<Vec<_>>());
    let tail_tokens = session.partition_resident_batch(&requests[32..]).unwrap();
    assert_eq!(tail_tokens, [32]);
    drop(session);

    let resolved = ctx.resolve_partitions().unwrap();
    assert_eq!(
        resolved.len(),
        33,
        "one resolve must return every call's tokens"
    );
    ctx.finish_partition_level().unwrap();

    for (index, (span, feature, split_bin, missing_left)) in requests.iter().enumerate() {
        let (left_span, right_span) = resolved[index];
        let mut expected_left = Vec::new();
        let mut expected_right = Vec::new();
        for &row in &source_rows[index] {
            let bin = bins[*feature * N_ROWS + row as usize];
            if goes_left(bin, *split_bin, *missing_left) {
                expected_left.push(row);
            } else {
                expected_right.push(row);
            }
        }

        let got_left = ctx.download_rows(left_span).unwrap();
        let got_right = ctx.download_rows(right_span).unwrap();
        assert_rows_equal(
            &got_left,
            &expected_left,
            &format!("split 32+1 request {index}, span={span:?}, left"),
        );
        assert_rows_equal(
            &got_right,
            &expected_right,
            &format!("split 32+1 request {index}, span={span:?}, right"),
        );
        assert_eq!(
            got_left.len() + got_right.len(),
            span.len,
            "left and right child lengths must account for every source row in request {index}"
        );
    }
}

#[test]
fn resident_partition_batch_scans_more_than_1024_segments() {
    let _gpu_guard = GPU_PARTITION_TEST_LOCK.lock().unwrap();
    let bins = bins_for(LARGE_ROWS, 2);
    let ctx = make_context(LARGE_ROWS, N_FEATURES, 1, 1);
    assert_partition_scratch_budget(&ctx, LARGE_ROWS);
    seed_resident(&ctx, 0, LARGE_ROWS, 2, &bins);

    let first_len = 256 * 1024 + 1;
    let first = [
        (
            GpuRowSpan {
                offset: 0,
                len: first_len,
            },
            0,
            2,
            true,
        ),
        (
            GpuRowSpan {
                offset: first_len,
                len: 1,
            },
            1,
            0,
            false,
        ),
    ];
    let first_blocks: usize = first
        .iter()
        .map(|(span, _, _, _)| span.len.div_ceil(256))
        .sum();
    assert!(first_blocks > 1024);
    ctx.begin_tree_rows().unwrap();
    let first_result = run_batch(&ctx, &bins, LARGE_ROWS, 0, &first);
    assert_eq!(first_result.len(), 2);

    let first_len = 256 * 1024 - 1;
    let second = [
        (
            GpuRowSpan {
                offset: 0,
                len: first_len,
            },
            1,
            3,
            false,
        ),
        (
            GpuRowSpan {
                offset: first_len,
                len: 3,
            },
            0,
            1,
            true,
        ),
    ];
    let second_blocks: usize = second
        .iter()
        .map(|(span, _, _, _)| span.len.div_ceil(256))
        .sum();
    assert!(second_blocks > 1024);
    ctx.begin_tree_rows().unwrap();
    let second_result = run_batch(&ctx, &bins, LARGE_ROWS, 0, &second);
    assert_eq!(second_result.len(), 2);

    // The resident-only batch API keeps its empty-input no-op available to a
    // streaming fallback session, while rejecting nonempty requests there.
    let streaming = make_context(N_ROWS, N_FEATURES, 1, 0);
    let streaming_bins = bins_for(N_ROWS, N_FEATURES);
    let mut fallback = streaming
        .partition_session(&streaming_bins[..N_ROWS])
        .unwrap();
    let none: [PartitionRequest; 0] = [];
    assert!(fallback.partition_resident_batch(&none).unwrap().is_empty());
    let nonempty = [(GpuRowSpan { offset: 0, len: 1 }, 0, 1, false)];
    assert!(fallback.partition_resident_batch(&nonempty).is_err());
}

#[test]
fn resident_partition_batch_after_arena_swap_uses_narrow_tail_block() {
    let _gpu_guard = GPU_PARTITION_TEST_LOCK.lock().unwrap();
    let ctx = make_context(N_ROWS, N_FEATURES, 2, 2);
    let full_bins = bins_for(N_ROWS, N_FEATURES);
    let tail_width = 2;
    let tail_bins = bins_for(N_ROWS, tail_width);
    seed_resident(&ctx, 0, N_ROWS, N_FEATURES, &full_bins);
    seed_resident(&ctx, 1, N_ROWS, tail_width, &tail_bins);

    // Produce a real child arena with the first resident block, then consume
    // both children through a batched partition on the narrow physical tail
    // block. This covers the producer-stream readiness edge for resident bins
    // and row-arena reuse after finish_partition_level, while checking that
    // the tail block's actual width and physical identity are respected.
    ctx.begin_tree_rows().unwrap();
    let root = [(
        GpuRowSpan {
            offset: 0,
            len: N_ROWS,
        },
        0,
        2,
        true,
    )];
    let root_children = run_batch(&ctx, &full_bins, N_ROWS, 0, &root);
    assert_eq!(root_children.len(), 1);

    let second_level = [
        (root_children[0].0, 0, 1, false),
        (root_children[0].1, 1, 3, true),
    ];
    let second_children = run_batch(&ctx, &tail_bins, N_ROWS, 1, &second_level);
    assert_eq!(second_children.len(), 2);
}

#[test]
fn resident_partition_multi_batches_nodes_of_several_blocks_in_one_level() {
    let _gpu_guard = GPU_PARTITION_TEST_LOCK.lock().unwrap();
    // Two resident blocks of different widths (4 and 2 columns) with
    // different contents; one level-wide batch interleaves their nodes.
    let block0 = bins_for(N_ROWS, N_FEATURES);
    let block1: Vec<u8> = bins_for(N_ROWS, N_FEATURES + 3)[3 * N_ROWS..5 * N_ROWS].to_vec();
    let ctx = make_context(N_ROWS, N_FEATURES, 2, 2);
    seed_resident(&ctx, 0, N_ROWS, N_FEATURES, &block0);
    seed_resident(&ctx, 1, N_ROWS, 2, &block1);
    ctx.begin_tree_rows().unwrap();

    let spans: Vec<GpuRowSpan> = (0..6)
        .map(|i| {
            let len = if i == 5 { N_ROWS - 5 * 300 } else { 300 };
            GpuRowSpan {
                offset: i * 300,
                len,
            }
        })
        .collect();
    // (span, block, local feature, split bin, missing left)
    let requests: Vec<(GpuRowSpan, usize, usize, u32, bool)> = vec![
        (spans[0], 0, 2, 2, true),
        (spans[1], 1, 1, 1, false),
        (spans[2], 0, 0, 3, false),
        (spans[3], 1, 0, 2, true),
        (spans[4], 1, 1, 0, true),
        (spans[5], 0, 3, 1, false),
    ];
    let source_rows: Vec<Vec<RowId>> = requests
        .iter()
        .map(|r| ctx.download_rows(r.0).unwrap())
        .collect();
    let mut session = ctx
        .partition_session_resident_blocks(&[0, 1])
        .unwrap()
        .expect("both blocks are resident");
    let tokens = session.partition_resident_multi(&requests).unwrap();
    assert_eq!(tokens, (0..requests.len()).collect::<Vec<_>>());
    // A single request on the non-primary block takes the single-node route.
    let single = session
        .partition_resident_multi(&[(GpuRowSpan { offset: 0, len: 0 }, 1, 1, 1, false)])
        .unwrap();
    assert_eq!(single, [requests.len()]);
    assert!(session
        .partition_resident_multi(&[(spans[0], 2, 0, 1, false)])
        .is_err());
    assert!(session
        .partition_resident_multi(&[(spans[0], 1, 2, 1, false), (spans[1], 0, 0, 1, false)])
        .is_err());
    drop(session);
    let resolved = ctx.resolve_partitions().unwrap();
    ctx.finish_partition_level().unwrap();
    for (k, &(_, block, feat, split, missing_left)) in requests.iter().enumerate() {
        let bins = if block == 0 { &block0 } else { &block1 };
        let (mut want_left, mut want_right) = (Vec::new(), Vec::new());
        for &row in &source_rows[k] {
            if goes_left(bins[feat * N_ROWS + row as usize], split, missing_left) {
                want_left.push(row);
            } else {
                want_right.push(row);
            }
        }
        let (left, right) = resolved[tokens[k]];
        assert_rows_equal(
            &ctx.download_rows(left).unwrap(),
            &want_left,
            &format!("request {k} left"),
        );
        assert_rows_equal(
            &ctx.download_rows(right).unwrap(),
            &want_right,
            &format!("request {k} right"),
        );
    }
}
