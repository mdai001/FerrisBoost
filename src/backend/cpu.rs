//! CPU 后端。rayon 并行,按列块切分。
//!
//! 注意:列分块在多 socket 上是天然的 NUMA 亲和策略 —— 每个 socket
//! 负责一组列,直方图在本地内存累加,跨 socket 只交换汇总结果。
//! 现在 XGBoost 按行切给线程,所有线程往共享直方图数组里写,
//! remote access 和 false sharing 都很明显。
