use super::*;
use crate::{
    backend::cuda::{GpuRowSpan, GpuTrainCtx, SharedHistogramConfig},
    hist::{self, Histogram},
    split::{best_split_in_block, SplitCandidate},
    types::{FeatId, GradPair, GradPairFixed, GradQuantizer, RowId, TrainParams, MISSING_BIN},
};

const N_ROWS: usize = 96;
const N_FEATURES: usize = 3;
const HIST_OFFSETS: [u32; N_FEATURES + 1] = [0, 2, 4, 6];

fn context() -> GpuTrainCtx {
    GpuTrainCtx::new(
        0,
        N_ROWS,
        N_FEATURES,
        HIST_OFFSETS[N_FEATURES] as usize,
        1,
        64,
        SharedHistogramConfig {
            runtime_log: false,
            threads_per_block: 128,
            hist_streams: Some(1),
            hist_nodes_per_batch: Some(3),
            resident_blocks: Some(1),
            gpu_split_cache: true,
            ..Default::default()
        },
        false,
    )
    .unwrap()
}

fn fixture() -> (Vec<u8>, Vec<GradPair>) {
    let bins = (0..N_FEATURES)
        .flat_map(|feat| {
            (0..N_ROWS).map(move |row| {
                if (row + feat * 7) % 29 == 0 {
                    MISSING_BIN
                } else {
                    ((row >> feat) & 1) as u8
                }
            })
        })
        .collect();
    let gpair = (0..N_ROWS)
        .map(|row| GradPair {
            grad: if (row >> 2) & 1 == 0 { -3.0 } else { 3.0 },
            hess: 1.0,
        })
        .collect();
    (bins, gpair)
}

fn row_sum(gpair: &[GradPairFixed], span: GpuRowSpan) -> GradPairFixed {
    gpair[span.offset..span.offset + span.len]
        .iter()
        .copied()
        .fold(GradPairFixed::default(), |mut sum, pair| {
            sum += pair;
            sum
        })
}

fn cpu_histogram(bins: &[u8], gpair: &[GradPairFixed], span: GpuRowSpan) -> Histogram {
    let rows: Vec<RowId> = (span.offset..span.offset + span.len)
        .map(|row| row as RowId)
        .collect();
    let mut expected = vec![GradPairFixed::default(); HIST_OFFSETS[N_FEATURES] as usize];
    hist::build(
        bins,
        N_ROWS,
        N_FEATURES,
        &HIST_OFFSETS,
        gpair,
        Some(&rows),
        &mut expected,
    );
    Histogram { bins: expected }
}

fn assert_candidate_eq(
    got: Option<SplitCandidate>,
    expected: Option<SplitCandidate>,
    context: &str,
) {
    match (got, expected) {
        (None, None) => {}
        (Some(got), Some(expected)) => {
            assert_eq!(got.feat, expected.feat, "{context}: feature");
            assert_eq!(got.bin, expected.bin, "{context}: bin");
            assert_eq!(
                got.gain.to_bits(),
                expected.gain.to_bits(),
                "{context}: gain"
            );
            assert_eq!(got.left, expected.left, "{context}: left sum");
            assert_eq!(got.right, expected.right, "{context}: right sum");
            assert_eq!(
                got.missing_left, expected.missing_left,
                "{context}: missing direction"
            );
        }
        (got, expected) => panic!("{context}: candidate presence differs: {got:?} vs {expected:?}"),
    }
}

fn expected_candidate(
    bins: &[u8],
    gpair: &[GradPairFixed],
    span: GpuRowSpan,
    q: &GradQuantizer,
    params: &TrainParams,
) -> Option<SplitCandidate> {
    expected_candidate_sampled(bins, gpair, span, q, params, None)
}

fn expected_candidate_sampled(
    bins: &[u8],
    gpair: &[GradPairFixed],
    span: GpuRowSpan,
    q: &GradQuantizer,
    params: &TrainParams,
    sampled: Option<&[FeatId]>,
) -> Option<SplitCandidate> {
    best_split_in_block(
        &cpu_histogram(bins, gpair, span),
        &[0, 1, 2],
        &HIST_OFFSETS,
        row_sum(gpair, span),
        q,
        params,
        sampled,
    )
}

fn assert_hist_eq(got: &[GradPairFixed], expected: &[GradPairFixed], context: &str) {
    assert_eq!(
        got, expected,
        "{context}: histogram differs from CPU reference"
    );
}

fn tied_feature_fixture() -> (Vec<u8>, Vec<GradPair>) {
    let bins = (0..N_FEATURES)
        .flat_map(|_| (0..N_ROWS).map(|row| u8::from(row % 16 >= 8)))
        .collect();
    let gpair = (0..N_ROWS)
        .map(|row| GradPair {
            grad: if row % 16 < 8 { -3.0 } else { 3.0 },
            hess: 1.0,
        })
        .collect();
    (bins, gpair)
}

fn run_cached_feature_tie_case(sampled: Option<&[FeatId]>, expected_feature: FeatId) {
    let ctx = context();
    let (bins, raw_gpair) = tied_feature_fixture();
    let feats: Vec<FeatId> = (0..N_FEATURES as FeatId).collect();
    let q = GradQuantizer::from_bounds(3.0, 1.0, N_ROWS);
    let gpair: Vec<GradPairFixed> = raw_gpair.iter().map(|pair| q.to_fixed(*pair)).collect();
    let params = TrainParams {
        lambda: 1.0,
        gamma: 0.0,
        min_child_weight: 1.0,
        hist_nodes_per_batch: Some(3),
        ..Default::default()
    };

    // Verify the fixture itself has an exact positive tie for every feature.
    // The per-feature slice uses the local [0, 2) offsets expected by the CPU
    // reference, while retaining each feature's identical two-bin histogram.
    let tie_span = GpuRowSpan { offset: 0, len: 16 };
    let tie_hist = cpu_histogram(&bins, &gpair, tie_span);
    let tie_sum = row_sum(&gpair, tie_span);
    let mut per_feature = Vec::with_capacity(N_FEATURES);
    for feat in 0..N_FEATURES as FeatId {
        let lo = HIST_OFFSETS[feat as usize] as usize;
        let hi = HIST_OFFSETS[feat as usize + 1] as usize;
        let one_feature_hist = Histogram {
            bins: tie_hist.bins[lo..hi].to_vec(),
        };
        let candidate = best_split_in_block(
            &one_feature_hist,
            &[feat],
            &[0, 2],
            tie_sum,
            &q,
            &params,
            None,
        )
        .expect("each identical feature must have a valid split");
        assert!(candidate.gain > 0.0, "fixture tie must have positive gain");
        assert_eq!(
            candidate.bin, 1,
            "fixture tie must choose the forward bin-1 split"
        );
        assert!(
            !candidate.missing_left,
            "fixture has no missing-bin alternative"
        );
        per_feature.push(candidate);
    }
    for candidate in &per_feature[1..] {
        assert_eq!(candidate.gain.to_bits(), per_feature[0].gain.to_bits());
        assert_eq!(candidate.bin, per_feature[0].bin);
        assert_eq!(candidate.left, per_feature[0].left);
        assert_eq!(candidate.right, per_feature[0].right);
        assert_eq!(candidate.missing_left, per_feature[0].missing_left);
    }

    ctx.upload_gpair(&gpair).unwrap();
    ctx.begin_tree_rows().unwrap();
    let mut session = ctx
        .block_session(0, &bins, N_ROWS, N_FEATURES, &HIST_OFFSETS, None)
        .unwrap();
    let parent_spans = [
        GpuRowSpan { offset: 0, len: 32 },
        GpuRowSpan {
            offset: 32,
            len: 32,
        },
        GpuRowSpan {
            offset: 64,
            len: 32,
        },
    ];
    let roots: Vec<_> = parent_spans
        .iter()
        .enumerate()
        .map(|(parent, &span)| GpuSplitGroup {
            small: parent,
            span,
            sum: row_sum(&gpair, span),
            bounds: crate::split::NodeBounds::UNBOUNDED,
            large: None,
        })
        .collect();

    // Keep the selected-feature set fixed for this whole tree. A fresh ctx,
    // session, and parent-cache seed are created for every invocation.
    session.begin_split_level(0).unwrap();
    let roots_result = session
        .split_cached_batch(&roots, &feats, sampled, None, None, &q, &params, true)
        .unwrap();
    assert_eq!(roots_result.len(), parent_spans.len());
    for (parent, span) in parent_spans.iter().copied().enumerate() {
        assert_candidate_eq(
            roots_result[parent].1.clone(),
            expected_candidate_sampled(&bins, &gpair, span, &q, &params, sampled),
            &format!("tied root {parent}"),
        );
        let winner = roots_result[parent]
            .1
            .clone()
            .expect("tied parent must produce a split");
        assert_eq!(
            winner.feat, expected_feature,
            "feature ties keep the first selected feature"
        );
        assert_eq!(winner.bin, 1, "forward scan keeps bin 1");
        assert!(
            !winner.missing_left,
            "missing direction remains right/default"
        );
    }
    session.finish_split_level(0);

    session.begin_split_level(1).unwrap();
    let mut groups = Vec::with_capacity(parent_spans.len());
    let mut child_spans = Vec::with_capacity(parent_spans.len() * 2);
    for (parent, parent_span) in parent_spans.iter().copied().enumerate() {
        let small_span = GpuRowSpan {
            offset: parent_span.offset,
            len: 16,
        };
        let large_span = GpuRowSpan {
            offset: parent_span.offset + 16,
            len: 16,
        };
        let parent_hist = cpu_histogram(&bins, &gpair, parent_span);
        let small_hist = cpu_histogram(&bins, &gpair, small_span);
        let large_hist = cpu_histogram(&bins, &gpair, large_span);
        let subtracted_large: Vec<_> = parent_hist
            .bins
            .iter()
            .zip(&small_hist.bins)
            .map(|(parent, small)| *parent - *small)
            .collect();
        assert_hist_eq(
            &subtracted_large,
            &large_hist.bins,
            &format!("tied fixture sibling subtraction for parent {parent}"),
        );
        let parent_slot = session
            .cached_parent_slot(parent)
            .expect("each tied parent histogram must be cached");
        let small_id = 10 + parent;
        let large_id = 20 + parent;
        groups.push(GpuSplitGroup {
            small: small_id,
            span: small_span,
            sum: row_sum(&gpair, small_span),
            bounds: crate::split::NodeBounds::UNBOUNDED,
            large: Some((
                large_id,
                parent_slot,
                row_sum(&gpair, large_span),
                crate::split::NodeBounds::UNBOUNDED,
            )),
        });
        child_spans.push((small_id, small_span));
        child_spans.push((large_id, large_span));
    }

    let before = session.timing;
    let children = session
        .split_cached_batch(&groups, &feats, sampled, None, None, &q, &params, false)
        .unwrap();
    assert_eq!(
        session.timing.split_d2h_calls - before.split_d2h_calls,
        2,
        "three paired groups at cap three require two packet-window collections"
    );
    let expected_ids: Vec<_> = child_spans.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        children.iter().map(|(id, _, _)| *id).collect::<Vec<_>>(),
        expected_ids,
        "packet decode preserves stable small/large child order"
    );
    for (index, ((_, span), (_, candidate, hist))) in child_spans.iter().zip(&children).enumerate()
    {
        assert!(
            hist.is_none(),
            "child split stage must not return a fallback histogram"
        );
        assert_candidate_eq(
            candidate.clone(),
            expected_candidate_sampled(&bins, &gpair, *span, &q, &params, sampled),
            &format!("tied cached child {index}"),
        );
        let winner = candidate
            .clone()
            .expect("each tied child must produce a split");
        assert_eq!(
            winner.feat, expected_feature,
            "feature ties keep the first selected feature"
        );
        assert_eq!(winner.bin, 1, "forward scan keeps bin 1");
        assert!(
            !winner.missing_left,
            "missing direction remains right/default"
        );
    }
}

#[test]
fn cached_gpu_split_preserves_feature_ties_for_all_and_sampled_features() {
    let sampled = [1, 2];
    for _ in 0..2 {
        run_cached_feature_tie_case(None, 0);
        run_cached_feature_tie_case(Some(&sampled), 1);
    }
}

#[test]
fn split_packet_window_collects_two_batches_then_tail_in_stable_order() {
    let ctx = context();
    let (bins, raw_gpair) = fixture();
    let feats: Vec<FeatId> = (0..N_FEATURES as FeatId).collect();
    let q = GradQuantizer::from_bounds(3.0, 1.0, N_ROWS);
    let gpair: Vec<GradPairFixed> = raw_gpair.iter().map(|pair| q.to_fixed(*pair)).collect();
    let params = TrainParams {
        lambda: 1.0,
        gamma: 0.0,
        min_child_weight: 1.0,
        hist_nodes_per_batch: Some(3),
        ..Default::default()
    };

    ctx.upload_gpair(&gpair).unwrap();
    ctx.begin_tree_rows().unwrap();
    let mut session = ctx
        .block_session(0, &bins, N_ROWS, N_FEATURES, &HIST_OFFSETS, None)
        .unwrap();

    // Seed three independent cached parent histograms. This direct session
    // setup intentionally gives the cache three roots so the next level has
    // three paired groups, even though a normal tree has one root.
    session.begin_split_level(0).unwrap();
    let parent_spans = [
        GpuRowSpan { offset: 0, len: 32 },
        GpuRowSpan {
            offset: 32,
            len: 32,
        },
        GpuRowSpan {
            offset: 64,
            len: 32,
        },
    ];
    let roots: Vec<_> = parent_spans
        .iter()
        .enumerate()
        .map(|(parent, &span)| GpuSplitGroup {
            small: parent,
            span,
            sum: row_sum(&gpair, span),
            bounds: crate::split::NodeBounds::UNBOUNDED,
            large: None,
        })
        .collect();
    let root_results = session
        .split_cached_batch(&roots, &feats, None, None, None, &q, &params, true)
        .unwrap();
    assert_eq!(root_results.len(), 3);
    session.finish_split_level(0);

    session.begin_split_level(1).unwrap();
    let mut groups = Vec::with_capacity(3);
    let mut child_spans = Vec::with_capacity(6);
    for (parent, parent_span) in parent_spans.iter().copied().enumerate() {
        let small_span = GpuRowSpan {
            offset: parent_span.offset,
            len: 16,
        };
        let large_span = GpuRowSpan {
            offset: parent_span.offset + 16,
            len: 16,
        };
        let parent_hist = cpu_histogram(&bins, &gpair, parent_span);
        let small_hist = cpu_histogram(&bins, &gpair, small_span);
        let large_hist = cpu_histogram(&bins, &gpair, large_span);
        let subtracted_large: Vec<_> = parent_hist
            .bins
            .iter()
            .zip(&small_hist.bins)
            .map(|(parent, small)| *parent - *small)
            .collect();
        assert_hist_eq(
            &subtracted_large,
            &large_hist.bins,
            &format!("manual sibling subtraction for parent {parent}"),
        );
        let parent_slot = session
            .cached_parent_slot(parent)
            .expect("every seeded parent histogram should be resident");
        let small_id = 10 + parent;
        let large_id = 20 + parent;
        groups.push(GpuSplitGroup {
            small: small_id,
            span: small_span,
            sum: row_sum(&gpair, small_span),
            bounds: crate::split::NodeBounds::UNBOUNDED,
            large: Some((
                large_id,
                parent_slot,
                row_sum(&gpair, large_span),
                crate::split::NodeBounds::UNBOUNDED,
            )),
        });
        child_spans.push((small_id, small_span));
        child_spans.push((large_id, large_span));
    }

    let before_empty = session.timing;
    let empty_before = session
        .split_cached_batch(&[], &feats, None, None, None, &q, &params, false)
        .unwrap();
    assert!(empty_before.is_empty());
    assert_eq!(session.timing.d2h_calls, before_empty.d2h_calls);
    assert_eq!(session.timing.d2h_bytes, before_empty.d2h_bytes);
    assert_eq!(session.timing.split_d2h_calls, before_empty.split_d2h_calls);
    assert_eq!(session.timing.split_d2h_bytes, before_empty.split_d2h_bytes);

    // Fill the two-batch packet window with a sentinel. Three cap-three paired
    // groups submit three two-node batches: the first two occupy four packet
    // slots, and the final tail reuses the first two. The final two slots must
    // remain untouched after collection.
    let packet_sentinel = vec![0xA5; 3 * 64 * 2];
    let packet_stream = session.slot.as_ref().unwrap().stream.clone();
    {
        let cache = session.slot.as_mut().unwrap().split_cache.as_mut().unwrap();
        packet_stream
            .memcpy_htod(&packet_sentinel, &mut cache.packets.slice_mut(..))
            .unwrap();
    }
    packet_stream.synchronize().unwrap();

    let before = session.timing;
    let results = session
        .split_cached_batch(&groups, &feats, None, None, None, &q, &params, false)
        .unwrap();
    assert_eq!(
        session.timing.split_d2h_calls - before.split_d2h_calls,
        2,
        "three producer batches must use one full two-batch packet collection and one tail collection"
    );
    assert_eq!(
        session.timing.split_d2h_bytes - before.split_d2h_bytes,
        6 * 64,
        "two packet collections transfer exactly six 64-byte candidates"
    );
    assert_eq!(
        session.timing.d2h_calls - before.d2h_calls,
        2,
        "last-level packet collection must not download full histograms"
    );

    let expected_ids: Vec<_> = child_spans.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        results.iter().map(|(id, _, _)| *id).collect::<Vec<_>>(),
        expected_ids,
        "packet decode must preserve producer and child order across the reused window tail"
    );
    assert_eq!(
        results
            .iter()
            .filter(|(_, candidate, _)| candidate.is_some())
            .count(),
        6,
        "the deterministic signal must produce a real split packet for every child"
    );
    for (index, ((_, span), (_, candidate, hist))) in child_spans.iter().zip(&results).enumerate() {
        assert!(
            hist.is_none(),
            "last-level child must not return a fallback histogram"
        );
        assert_candidate_eq(
            candidate.clone(),
            expected_candidate(&bins, &gpair, *span, &q, &params),
            &format!("child result {index}"),
        );
    }
    let before_empty_after = session.timing;
    let empty_after = session
        .split_cached_batch(&[], &feats, None, None, None, &q, &params, false)
        .unwrap();
    assert!(empty_after.is_empty());
    assert_eq!(session.timing.d2h_calls, before_empty_after.d2h_calls);
    assert_eq!(session.timing.d2h_bytes, before_empty_after.d2h_bytes);
    assert_eq!(
        session.timing.split_d2h_calls,
        before_empty_after.split_d2h_calls
    );
    assert_eq!(
        session.timing.split_d2h_bytes,
        before_empty_after.split_d2h_bytes
    );

    // This observer copy happens after counter assertions, so it cannot affect
    // the collection accounting above.
    let mut packet_bytes = vec![0u8; packet_sentinel.len()];
    let cache = session.slot.as_ref().unwrap().split_cache.as_ref().unwrap();
    packet_stream
        .memcpy_dtoh(
            &cache.packets.slice(..packet_bytes.len()),
            packet_bytes.as_mut_slice(),
        )
        .unwrap();
    packet_stream.synchronize().unwrap();
    assert!(
        packet_bytes[..4 * 64].iter().any(|byte| *byte != 0xA5),
        "the full window and final tail must write packet slots"
    );
    assert!(
        packet_bytes[4 * 64..].iter().all(|byte| *byte == 0xA5),
        "packet slots beyond the four-slot full window must preserve the sentinel"
    );

    // Re-enter the level with next-generation caching enabled. With capacity
    // three, group one admits two children, group two spills one, and group
    // three spills both. Each spill must collect the previous producer before
    // hist_out is reused, returning exact CPU histograms for uncached nodes.
    session.begin_split_level(1).unwrap();
    let before_spill = session.timing;
    let spilled = session
        .split_cached_batch(&groups, &feats, None, None, None, &q, &params, true)
        .unwrap();
    assert_eq!(spilled.len(), 6);
    assert_eq!(
        session.timing.split_d2h_calls - before_spill.split_d2h_calls,
        3,
        "capacity overflow must flush before each later producer overwrites hist_out"
    );
    assert_eq!(
        session.timing.d2h_calls - before_spill.d2h_calls,
        6,
        "three packet collections plus three uncached fallback histograms are required"
    );
    assert_eq!(
        spilled.iter().filter(|(_, _, hist)| hist.is_some()).count(),
        3,
        "capacity three admits three child histograms and returns three exact host fallbacks"
    );
    assert_eq!(
        spilled.iter().map(|(id, _, _)| *id).collect::<Vec<_>>(),
        expected_ids,
        "spill collection must preserve stable child indices"
    );
    for (index, ((_, span), (_, candidate, hist))) in child_spans.iter().zip(&spilled).enumerate() {
        assert_candidate_eq(
            candidate.clone(),
            expected_candidate(&bins, &gpair, *span, &q, &params),
            &format!("spilled child result {index}"),
        );
        if let Some(hist) = hist {
            assert_hist_eq(
                hist,
                &cpu_histogram(&bins, &gpair, *span).bins,
                &format!("fallback child {index}"),
            );
        }
    }
}

/// Monotone branch of `split_cached_features` vs the Rust constrained evaluator:
/// every constraint pattern × node bounds (unbounded, loose, binding), all
/// candidates bit-identical including "no valid split" outcomes.
#[test]
fn constrained_kernel_matches_cpu_evaluator() {
    use crate::split::{best_split_in_block_constrained, Monotone, NodeBounds};
    let (bins, raw_gpair) = fixture();
    let feats: Vec<FeatId> = (0..N_FEATURES as FeatId).collect();
    let q = GradQuantizer::from_bounds(3.0, 1.0, N_ROWS);
    let gpair: Vec<GradPairFixed> = raw_gpair.iter().map(|pair| q.to_fixed(*pair)).collect();
    let spans = [
        GpuRowSpan { offset: 0, len: 32 },
        GpuRowSpan {
            offset: 32,
            len: 32,
        },
        GpuRowSpan {
            offset: 64,
            len: 32,
        },
    ];
    let bounds_cases = [
        NodeBounds::UNBOUNDED,
        NodeBounds {
            lower: -0.5,
            upper: 0.5,
        },
        NodeBounds {
            lower: -0.05,
            upper: 0.02,
        },
        NodeBounds {
            lower: 0.1,
            upper: 0.1,
        },
    ];
    let mut constrained_differs = false;
    for constraints in [
        vec![1i8, 1, 1],
        vec![-1, -1, -1],
        vec![1, -1, 0],
        vec![0, 0, -1],
    ] {
        for bounds in bounds_cases {
            let params = TrainParams {
                lambda: 1.0,
                gamma: 0.0,
                min_child_weight: 1.0,
                hist_nodes_per_batch: Some(3),
                monotone_constraints: constraints.clone(),
                ..Default::default()
            };
            let ctx = context();
            ctx.upload_gpair(&gpair).unwrap();
            ctx.begin_tree_rows().unwrap();
            let mut session = ctx
                .block_session(0, &bins, N_ROWS, N_FEATURES, &HIST_OFFSETS, None)
                .unwrap();
            let groups: Vec<_> = spans
                .iter()
                .enumerate()
                .map(|(node, &span)| GpuSplitGroup {
                    small: node,
                    span,
                    sum: row_sum(&gpair, span),
                    bounds,
                    large: None,
                })
                .collect();
            session.begin_split_level(0).unwrap();
            let got = session
                .split_cached_batch(&groups, &feats, None, None, None, &q, &params, false)
                .unwrap();
            session.finish_split_level(0);
            for (node, span) in spans.iter().copied().enumerate() {
                let hist = cpu_histogram(&bins, &gpair, span);
                let sum = row_sum(&gpair, span);
                let expected = best_split_in_block_constrained(
                    &hist,
                    &feats,
                    &HIST_OFFSETS,
                    sum,
                    &q,
                    &params,
                    None,
                    Some(Monotone {
                        constraints: &constraints,
                        bounds,
                    }),
                );
                let unconstrained =
                    best_split_in_block(&hist, &feats, &HIST_OFFSETS, sum, &q, &params, None);
                if expected.as_ref().map(|c| (c.feat, c.bin, c.gain.to_bits()))
                    != unconstrained
                        .as_ref()
                        .map(|c| (c.feat, c.bin, c.gain.to_bits()))
                {
                    constrained_differs = true;
                }
                assert_candidate_eq(
                    got[node].1.clone(),
                    expected.clone(),
                    &format!("constraints={constraints:?} bounds={bounds:?} node {node}"),
                );
            }
        }
    }
    assert!(
        constrained_differs,
        "fixture must exercise constraints that change the chosen split"
    );
}
