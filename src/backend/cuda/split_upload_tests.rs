//! CPU-only coverage for the host-side split input upload layout.
//!
//! These tests deliberately compare the raw native bytes: metadata is viewed
//! by CUDA as `u32`, so converting it numerically to `i64` would change the
//! representation and lose the intended packed layout.

use super::*;

fn expected_native_bytes(sums: &[i64], metadata: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(sums.len() * 8 + metadata.len() * 4);
    for &sum in sums {
        bytes.extend_from_slice(&sum.to_ne_bytes());
    }
    for &value in metadata {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn actual_native_bytes(packed: &[i64]) -> Vec<u8> {
    packed
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn check_layout(sums: Vec<i64>, metadata: &[u32]) {
    assert_eq!(sums.len() % 2, 0, "two sums per node");
    assert_eq!(
        metadata.len(),
        sums.len() * 2,
        "four metadata words per node"
    );

    let nodes = sums.len() / 2;
    let mut packed = vec![0i64; nodes * 4];
    for node in 0..nodes {
        write_split_input(
            &mut packed,
            node,
            crate::types::GradPairFixed {
                grad: sums[node * 2],
                hess: sums[node * 2 + 1],
            },
            metadata[node * 4..node * 4 + 4].try_into().unwrap(),
        );
    }
    assert_eq!(
        actual_native_bytes(&packed),
        expected_native_bytes(&sums, metadata),
        "packed bytes must preserve sums followed by native metadata bytes"
    );
}

#[test]
fn write_split_input_preserves_empty_one_two_three_and_sixty_four_node_layouts() {
    for nodes in [0usize, 1, 2, 3, 64] {
        let sums = (0..nodes * 2)
            .map(|i| match i % 4 {
                0 => i64::MIN,
                1 => i64::MAX,
                2 => -(i as i64) - 1,
                _ => i as i64,
            })
            .collect();
        let metadata = (0..nodes * 4)
            .map(|i| match i % 5 {
                0 => u32::MAX,
                1 => 0x8000_0001,
                2 => 0x7fff_fffe,
                _ => (i as u32).wrapping_mul(0x0101_0101),
            })
            .collect::<Vec<_>>();

        check_layout(sums, &metadata);
    }
}

#[test]
fn write_split_input_keeps_metadata_order_and_unsigned_sentinels() {
    let sums = vec![i64::MIN, -1, 0, i64::MAX];
    let metadata = vec![
        0x0102_0304,
        u32::MAX,
        0x8000_0000,
        0x7fff_ffff,
        0,
        1,
        0xdead_beef,
        0x1234_5678,
    ];

    let mut packed = vec![0i64; 2 * 4];
    for node in 0..2 {
        write_split_input(
            &mut packed,
            node,
            crate::types::GradPairFixed {
                grad: sums[node * 2],
                hess: sums[node * 2 + 1],
            },
            metadata[node * 4..node * 4 + 4].try_into().unwrap(),
        );
    }
    let bytes = actual_native_bytes(&packed);
    let metadata_bytes = bytes[4 * 8..]
        .chunks_exact(4)
        .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(metadata_bytes, metadata);
}

#[test]
fn write_split_input_handles_consecutive_decreasing_batches() {
    for nodes in [64usize, 23, 7, 2, 1] {
        let sums = (0..nodes * 2)
            .map(|i| (i as i64).wrapping_mul(-0x0102_0304_0506_0708))
            .collect();
        let metadata = (0..nodes * 4)
            .map(|i| u32::MAX.wrapping_sub(i as u32 * 17))
            .collect::<Vec<_>>();

        check_layout(sums, &metadata);
    }
}
