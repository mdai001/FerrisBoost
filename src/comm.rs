//! 通信抽象。
//!
//! 这个 trait 是「切分维度 = 并行维度」得以成立的地方:训练循环
//! 只依赖这个接口,后端可以是单线程(空操作)、共享内存、NCCL、
//! 或者将来的 MPI/gloo。阶段 1 只需要 Local。

use crate::types::{GradPair, RowId};
use crate::split::SplitCandidate;

pub trait Comm: Send + Sync {
    fn rank(&self) -> usize;
    fn size(&self) -> usize;

    /// 汇总各方的最优分裂候选,所有 rank 拿到同一个结果。
    fn allreduce_best_split(&self, local: Option<SplitCandidate>) -> Option<SplitCandidate>;

    /// 广播行分区结果。选中特征后只有持有该列的一方知道每行去哪边。
    ///
    /// 这是主要通信开销 —— 每行 1 bit,一亿行 12.5MB,每个节点一次。
    /// 阶段 4 上多卡前,先单独 benchmark 这一项占总时间的比例。
    fn broadcast_partition(&self, root: usize, bits: &mut [u8]);

    fn allreduce_gpair(&self, local: &mut [GradPair]);
}

/// 单进程实现:所有操作都是空的。
pub struct Local;

impl Comm for Local {
    fn rank(&self) -> usize { 0 }
    fn size(&self) -> usize { 1 }
    fn allreduce_best_split(&self, local: Option<SplitCandidate>) -> Option<SplitCandidate> { local }
    fn broadcast_partition(&self, _root: usize, _bits: &mut [u8]) {}
    fn allreduce_gpair(&self, _local: &mut [GradPair]) {}
}
