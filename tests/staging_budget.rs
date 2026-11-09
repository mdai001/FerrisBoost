use ferrisboost::gpu_mem_plan::{
    default_safety_reserve, resolve_gpu_streaming_plan, StreamingMemModel,
};
const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

fn expected_peak(model: &StreamingMemModel, cols: u64, streams: u64, resident: u64) -> u64 {
    model.fixed_bytes
        + cols * model.slot_col_bytes * streams
        + (cols * (model.slot_col_bytes + model.resident_extra_col_bytes)
            + model.resident_extra_fixed_bytes)
            * resident
}

#[test]
fn column_major_budget_prices_only_actual_slots_at_every_width() {
    for rows in [1024, 1_000_000, 10_500_000] {
        for cols in [1, 8, 32, 33, 64] {
            for (streams, resident) in [(1, 0), (2, 3), (2, 7), (2, 18)] {
                for cache in [false, true] {
                    let model = StreamingMemModel::new(rows, 255, streams, 16, true, true);
                    let model = if cache {
                        model.with_resident_split_cache(255, 16)
                    } else {
                        model
                    };
                    assert_eq!(
                        model.peak_for_layout(cols, streams, resident),
                        expected_peak(&model, cols, streams, resident)
                    );
                }
            }
        }
    }
}

#[test]
fn automatic_column_major_stream_policy_stays_narrow_two_wide_one() {
    for (cols, expected_streams) in [(8, 2), (32, 2), (33, 1), (64, 1)] {
        let plan = resolve_gpu_streaming_plan(
            Some(cols),
            None,
            Some(0),
            128,
            5 * GIB,
            6 * GIB,
            |streams| StreamingMemModel::new(100_000, 255, streams, 16, true, false),
        )
        .unwrap();
        assert_eq!(plan.resolved_hist_streams, expected_streams);
        assert_eq!(plan.resolved_cols_per_block, cols as u64);
    }
}

#[test]
fn resolved_higgs_plans_stay_inside_the_safe_budget_for_math_and_cache_modes() {
    let rows = 10_500_000;
    let features = 28;
    let free_vram = 1_152 * MIB;
    let total_vram = 6 * GIB;
    let safe_budget = free_vram - default_safety_reserve(free_vram);

    for fast_math in [false, true] {
        for cache_enabled in [false, true] {
            let plan = resolve_gpu_streaming_plan(
                Some(1),
                None,
                None,
                features,
                free_vram,
                total_vram,
                |streams| {
                    let model = StreamingMemModel::new(rows, 255, streams, 16, true, fast_math);
                    if cache_enabled {
                        model.with_resident_split_cache(255, 16)
                    } else {
                        model
                    }
                },
            )
            .unwrap();

            let selected_model =
                StreamingMemModel::new(rows, 255, plan.resolved_hist_streams, 16, true, fast_math);
            let selected_model = if cache_enabled {
                selected_model.with_resident_split_cache(255, 16)
            } else {
                selected_model
            };
            let corrected_peak = expected_peak(
                &selected_model,
                plan.resolved_cols_per_block,
                plan.resolved_hist_streams,
                plan.resident_blocks,
            );

            assert_eq!(plan.estimated_peak, corrected_peak, "{}", plan.report());
            assert!(
                corrected_peak <= safe_budget,
                "fast_math={fast_math}, cache_enabled={cache_enabled}: plan exceeds safe budget: {}",
                plan.report(),
            );
        }
    }
}
