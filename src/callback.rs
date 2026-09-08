//! 训练回调。
//!
//! 刻意做成单方法的 trait,不照抄 XGBoost 的 TrainingCallback
//! 类体系 —— 那个形状是为了支持 Dask、Spark、学习率调度等一堆
//! 场景才长成那样。这里 early stopping 和 verbose 都是它的实现,
//! Python 侧再包一层让用户能传自己的函数。

use std::sync::{Arc, Mutex};

/// 一轮训练后的评估结果。
///
/// 结构对应 XGBoost 的 evals_result:eval_name -> metric_name -> value
#[derive(Clone, Debug)]
pub struct RoundMetrics {
    pub round: usize,
    /// (eval_name, metric_name, value)
    pub entries: Vec<(String, String, f32)>,
}

impl RoundMetrics {
    pub fn get(&self, eval_name: &str, metric_name: &str) -> Option<f32> {
        self.entries
            .iter()
            .find(|(e, m, _)| e == eval_name && m == metric_name)
            .map(|(_, _, v)| *v)
    }

    /// 最后一个 eval 的**第一个** metric —— XGBoost 的 early stopping
    /// 默认盯这个。
    ///
    /// 别拿 `entries.last()` 顶替:entries 的排法是「eval 外层、metric
    /// 内层」,`last()` 拿到的是最后一个 eval 的**最后一个** metric。
    /// 只配一个 metric 时两者恰好相同,配了两个就悄悄盯错了指标 ——
    /// 而且是那种"训得出来、只是停错地方"的错法。
    pub fn watched(&self) -> Option<&(String, String, f32)> {
        let (last_eval, _, _) = self.entries.last()?;
        self.entries.iter().find(|(e, _, _)| e == last_eval)
    }

    /// 最后一条记录。日志用;early stopping 要用 `watched()`。
    pub fn last(&self) -> Option<&(String, String, f32)> {
        self.entries.last()
    }
}

pub trait Callback: Send {
    /// 返回 false 表示停止训练。
    fn after_iteration(&mut self, m: &RoundMetrics) -> bool;

    /// 训练结束时调用,用于收尾。
    fn after_training(&mut self) {}

    /// 早停选中的轮次和分数。训练循环结束后来问一遍,写进模型。
    /// 不是早停类的回调返回 None。
    fn best_iteration(&self) -> Option<(usize, f32)> {
        None
    }
}

/// 连续 rounds 轮没有改善就停。
///
/// 语义对齐 XGBoost:盯最后一个 eval 的第一个 metric。
pub struct EarlyStopping {
    pub rounds: usize,
    pub lower_is_better: bool,
    best: Option<f32>,
    pub best_iteration: usize,
    since_improve: usize,
}

impl EarlyStopping {
    pub fn new(rounds: usize, lower_is_better: bool) -> Self {
        Self {
            rounds,
            lower_is_better,
            best: None,
            best_iteration: 0,
            since_improve: 0,
        }
    }

    fn improved(&self, v: f32) -> bool {
        match self.best {
            None => true,
            Some(b) => {
                if self.lower_is_better {
                    v < b
                } else {
                    v > b
                }
            }
        }
    }
}

impl Callback for EarlyStopping {
    fn after_iteration(&mut self, m: &RoundMetrics) -> bool {
        let Some((_, _, v)) = m.watched() else {
            return true; // 没有验证集就不早停
        };
        if self.improved(*v) {
            self.best = Some(*v);
            self.best_iteration = m.round;
            self.since_improve = 0;
        } else {
            self.since_improve += 1;
        }
        self.since_improve < self.rounds
    }

    fn best_iteration(&self) -> Option<(usize, f32)> {
        self.best.map(|s| (self.best_iteration, s))
    }
}

/// 每 period 轮打印一次。训几小时的任务看不到进度是不能忍的。
pub struct VerboseEval {
    pub period: usize,
}

impl Callback for VerboseEval {
    fn after_iteration(&mut self, m: &RoundMetrics) -> bool {
        if self.period > 0 && m.round % self.period == 0 {
            let line: Vec<String> = m
                .entries
                .iter()
                .map(|(e, name, v)| format!("{e}-{name}:{v:.5}"))
                .collect();
            println!("[{}]\t{}", m.round, line.join("\t"));
        }
        true
    }
}

/// 把每轮结果攒起来,对应 XGBoost 的 evals_result。
///
/// 存的是共享句柄,不是裸 Vec:`train()` 收的是 `Box<dyn Callback>`,
/// 训练结束后调用方拿不回盒子里的东西,更没法从 `dyn Callback` downcast。
/// 拿 `handle()` 先把句柄留在手上,训完直接读 —— 阶段 2 的 Python 绑定
/// 要的 evals_result 也是这个形状。
#[derive(Default, Clone)]
pub struct RecordHistory {
    history: Arc<Mutex<Vec<RoundMetrics>>>,
}

impl RecordHistory {
    pub fn new() -> Self {
        Self::default()
    }

    /// 共享句柄。训练结束后 `handle.lock().unwrap()` 就能读到每轮结果。
    pub fn handle(&self) -> Arc<Mutex<Vec<RoundMetrics>>> {
        self.history.clone()
    }
}

impl Callback for RecordHistory {
    fn after_iteration(&mut self, m: &RoundMetrics) -> bool {
        self.history.lock().expect("evals_result 的锁被毒化了").push(m.clone());
        true
    }
}
