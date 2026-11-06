use ferrisboost::gpu_mem_plan::{
    default_safety_reserve, resolve_gpu_streaming_plan, StreamingMemModel,
};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;
const RM_STAGE_BUDGET: u64 = 32 * MIB;
const RM_STAGE_MIN_PER_SLOT: u64 = 4 * MIB;

fn model(rows: u64, row_major: bool, streams: u64) -> StreamingMemModel {
    StreamingMemModel::new(
        rows,
        255,
        streams,
        16,
        true,
        true,
        row_major,
        RM_STAGE_BUDGET,
    )
}

fn layout_without_row_major_staging(
    model: &StreamingMemModel,
    cols: u64,
    streams: u64,
    resident: u64,
) -> u64 {
    model.fixed_bytes - model.staging_budget
        + cols * model.slot_col_bytes * streams
        + (cols * (model.slot_col_bytes + model.resident_extra_col_bytes)
            + model.resident_extra_fixed_bytes)
            * resident
}

#[test]
fn row_major_staging_preserves_the_aggregate_budget_until_the_per_slot_floor_applies() {
    let rows = 10_500_000;
    let cols = 8;
    let streams = 2;

    // Eight allocation slots split the 32 MiB aggregate evenly. At nine slots,
    // the 4 MiB floor makes the real aggregate 36 MiB.
    for (resident, expected_stage_bytes) in [(6, 32 * MIB), (7, 36 * MIB)] {
        let model = model(rows, true, streams);
        let peak = model.peak_for_layout(cols, streams, resident);
        let non_staging = layout_without_row_major_staging(&model, cols, streams, resident);
        assert_eq!(
            peak - non_staging,
            expected_stage_bytes,
            "{} total slots should account for {} MiB of row-major staging",
            streams + resident,
            expected_stage_bytes / MIB,
        );
    }
}

#[test]
fn row_major_staging_is_capped_by_the_rows_in_each_slot() {
    let rows = 1_024;
    let cols = 8;
    let streams = 2;
    let resident = 7;
    let model = model(rows, true, streams);
    let peak = model.peak_for_layout(cols, streams, resident);
    let non_staging = layout_without_row_major_staging(&model, cols, streams, resident);

    // Runtime uses at most two staging buffers per slot. Each buffer is capped
    // by rows × active columns, and the slot budget follows the aggregate/floor
    // rule. The model may conservatively retain the 32 MiB nominal allowance.
    let slots = streams + resident;
    let per_slot_budget = (RM_STAGE_BUDGET / slots).max(RM_STAGE_MIN_PER_SLOT);
    let actual_staging_upper_bound = slots * (2 * rows * cols).min(per_slot_budget);
    assert!(
        peak - non_staging >= actual_staging_upper_bound,
        "the budget model must not undercount staging capped by small row counts"
    );
    assert!(peak - non_staging >= RM_STAGE_BUDGET);
}

#[test]
fn pinned_two_buffer_staging_is_charged_when_small_rows_hit_the_per_slot_floor() {
    let rows = 1_000_000;
    let cols = 1;
    let streams = 2;
    let resident = 18;
    let slots = streams + resident;
    let per_slot_budget = (RM_STAGE_BUDGET / slots).max(RM_STAGE_MIN_PER_SLOT);

    let stage_for_buffers = |buffers: u64| {
        let rows_per_buffer = (per_slot_budget / buffers / cols).max(1).min(rows);
        rows_per_buffer * cols * buffers * slots
    };
    let pageable_stage = stage_for_buffers(1);
    let pinned_stage = stage_for_buffers(2);
    assert_eq!(pageable_stage, 20_000_000);
    assert_eq!(pinned_stage, 40_000_000);

    let model = model(rows, true, streams);
    let peak = model.peak_for_layout(cols, streams, resident);
    let non_staging = layout_without_row_major_staging(&model, cols, streams, resident);
    assert_eq!(
        peak - non_staging,
        pinned_stage,
        "planner must budget the larger pinned two-buffer allocation"
    );
}

#[test]
fn column_major_layout_has_no_row_major_staging_charge() {
    let model = model(10_500_000, false, 2);
    let peak = model.peak_for_layout(8, 2, 7);
    let non_staging = layout_without_row_major_staging(&model, 8, 2, 7);
    assert_eq!(peak, non_staging);
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
                    let model = StreamingMemModel::new(
                        rows,
                        255,
                        streams,
                        16,
                        true,
                        fast_math,
                        true,
                        RM_STAGE_BUDGET,
                    );
                    if cache_enabled {
                        model.with_resident_split_cache(255, 16)
                    } else {
                        model
                    }
                },
            )
            .unwrap();

            let selected_model = StreamingMemModel::new(
                rows,
                255,
                plan.resolved_hist_streams,
                16,
                true,
                fast_math,
                true,
                RM_STAGE_BUDGET,
            );
            let selected_model = if cache_enabled {
                selected_model.with_resident_split_cache(255, 16)
            } else {
                selected_model
            };
            let slots = plan.resolved_hist_streams + plan.resident_blocks;
            let per_slot_budget = (RM_STAGE_BUDGET / slots.max(1)).max(RM_STAGE_MIN_PER_SLOT);
            let stage_rows = (per_slot_budget / plan.resolved_cols_per_block.max(1))
                .max(1)
                .min(rows.max(1));
            let actual_stage_bytes = if plan.resolved_cols_per_block <= 32 {
                stage_rows * plan.resolved_cols_per_block * slots
            } else {
                0
            };
            let corrected_peak = layout_without_row_major_staging(
                &selected_model,
                plan.resolved_cols_per_block,
                plan.resolved_hist_streams,
                plan.resident_blocks,
            ) + actual_stage_bytes;

            assert_eq!(plan.estimated_peak, corrected_peak, "{}", plan.report());
            assert!(
                corrected_peak <= safe_budget,
                "fast_math={fast_math}, cache_enabled={cache_enabled}: plan exceeds safe budget: {}",
                plan.report(),
            );
        }
    }
}
