use ferrisboost::gpu_mem_plan::{
    default_safety_reserve, resolve_gpu_streaming_plan, StreamingMemModel,
};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

fn model(rows: u64, streams: u64) -> StreamingMemModel {
    StreamingMemModel::new(rows, 255, streams, 16, true, true)
}

#[test]
fn column_major_peak_matches_streaming_and_resident_allocations() {
    let model = model(10_500_000, 2).with_resident_split_cache(255, 16);
    let (cols, streams, resident) = (8, 2, 7);
    let expected = model.fixed_bytes
        + cols * model.slot_col_bytes * streams
        + (cols * (model.slot_col_bytes + model.resident_extra_col_bytes)
            + model.resident_extra_fixed_bytes)
            * resident;

    assert_eq!(model.peak_for_layout(cols, streams, resident), expected);
}

#[test]
fn cached_resident_capacity_respects_its_column_major_budget() {
    let model = model(1_000_000, 2).with_resident_split_cache(255, 16);
    let cols = 8;
    let total_blocks = 12;
    let budgets = [
        model.peak_for_layout(cols, 2, 0),
        model.peak_for_layout(cols, 2, 2),
        model.peak_for_layout(cols, 2, 6),
        model.peak_for_layout(cols, 2, total_blocks),
    ];
    let capacities: Vec<_> = budgets
        .iter()
        .map(|&budget| model.resident_capacity(budget, cols, 2, total_blocks))
        .collect();

    assert_eq!(capacities, [0, 2, 6, total_blocks]);
    assert!(capacities.windows(2).all(|pair| pair[0] <= pair[1]));
    for (&budget, &resident) in budgets.iter().zip(&capacities) {
        assert!(model.peak_for_layout(cols, 2, resident) <= budget);
        if resident < total_blocks {
            assert!(model.peak_for_layout(cols, 2, resident + 1) > budget);
        }
    }
}

#[test]
fn resolved_higgs_plans_match_the_column_major_estimate() {
    let rows = 10_500_000;
    let features = 28;
    let free_vram = 1_152 * MIB;
    let total_vram = 6 * GIB;
    let safe_budget = free_vram - default_safety_reserve(free_vram);

    for fast_math in [false, true] {
        for cache_enabled in [false, true] {
            let make_model = |streams| {
                let model = StreamingMemModel::new(rows, 255, streams, 16, true, fast_math);
                if cache_enabled {
                    model.with_resident_split_cache(255, 16)
                } else {
                    model
                }
            };
            let plan = resolve_gpu_streaming_plan(
                Some(1),
                None,
                None,
                features,
                free_vram,
                total_vram,
                make_model,
            )
            .unwrap();
            let selected_model = make_model(plan.resolved_hist_streams);
            let peak = selected_model.peak_for_layout(
                plan.resolved_cols_per_block,
                plan.resolved_hist_streams,
                plan.resident_blocks,
            );

            assert_eq!(plan.estimated_peak, peak, "{}", plan.report());
            assert!(
                peak <= safe_budget,
                "fast_math={fast_math}, cache_enabled={cache_enabled}: plan exceeds safe budget: {}",
                plan.report(),
            );
        }
    }
}
