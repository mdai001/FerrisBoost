//! 树结构和模型序列化。
//!
//! 输出格式对齐 XGBoost 的 JSON model schema —— 这是让这个项目从
//! 玩具变成工具的关键决定:训练用 ferrisboost,推理、SHAP、ONNX
//! 转换、现成流水线全部照旧能用。

use std::collections::BTreeMap;

use serde::de::Error as _;
use serde::{Deserialize, Serialize};

use crate::columns::BinCuts;
use crate::types::{FeatId, Objective};

/// 叶子的左右孩子下标。和 XGBoost 的编码一致。
const NO_CHILD: i32 = -1;

/// XGBoost 用它表示根节点没有父亲(uint32 的 -1 落到 int32 上)。
const NO_PARENT: i32 = 2147483647;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub feat: FeatId,
    pub split_cond: f32,
    pub left: i32,
    pub right: i32,
    /// 缺失值走左还是右。每个分裂独立学出来的,见 `split.rs`。
    pub default_left: bool,
    /// **已经乘过 learning_rate**。
    ///
    /// 收缩放在建树时做,不放在预测时 —— XGBoost 存下来的就是收缩后的
    /// 值,预测端不知道 eta 是多少。留到预测时再乘,导出的模型给
    /// XGBoost 加载就会整体偏大 eta 分之一倍。
    pub leaf_value: f32,
    pub is_leaf: bool,
    /// 分裂增益。只为导出服务:对应 XGBoost 的 `loss_changes`,
    /// 也就是 `get_score(importance_type="gain")` 的来源。
    pub gain: f32,
    /// 节点的二阶梯度和。同样只为导出服务:对应 `sum_hessian`,
    /// 也就是 importance 里的 "cover"。预测完全不看这两项。
    pub sum_hess: f32,
}

impl Node {
    pub fn leaf(value: f32, sum_hess: f32) -> Self {
        Self {
            feat: 0,
            split_cond: 0.0,
            left: NO_CHILD,
            right: NO_CHILD,
            default_left: false,
            leaf_value: value,
            is_leaf: true,
            gain: 0.0,
            sum_hess,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// 加一个叶子,返回它的下标。根节点也走这条 —— 树从「一个叶子」
    /// 开始长,分裂时再就地改成内部节点。
    pub fn push_leaf(&mut self, value: f32, sum_hess: f32) -> usize {
        self.nodes.push(Node::leaf(value, sum_hess));
        self.nodes.len() - 1
    }

    /// 把一个叶子就地变成内部节点,并挂上两个新叶子。返回两个孩子的
    /// 下标。
    ///
    /// 孩子的下标一定大于父亲(只往数组尾部追加),`predict` 沿树
    /// 下降就不会绕圈 —— 这是个很便宜的不变量,别为了「原地复用被
    /// 删掉的节点」把它破坏掉。
    #[allow(clippy::too_many_arguments)]
    pub fn split_leaf(
        &mut self,
        node: usize,
        feat: FeatId,
        split_cond: f32,
        default_left: bool,
        gain: f32,
        left_value: f32,
        left_hess: f32,
        right_value: f32,
        right_hess: f32,
    ) -> (usize, usize) {
        debug_assert!(self.nodes[node].is_leaf, "只能分裂叶子");
        let left = self.push_leaf(left_value, left_hess);
        let right = self.push_leaf(right_value, right_hess);

        let n = &mut self.nodes[node];
        n.feat = feat;
        n.split_cond = split_cond;
        n.left = left as i32;
        n.right = right as i32;
        n.default_left = default_left;
        n.leaf_value = 0.0;
        n.is_leaf = false;
        n.gain = gain;

        (left, right)
    }

    /// 沿树下降,返回叶子值(已含 learning_rate)。
    ///
    /// 缺失用 NaN 表示,走 default 方向。**不能**直接写 `v < cond`
    /// 指望 NaN 自然落到右边:那样 default_left 就白学了,而且和
    /// XGBoost 的行为差一个方向。
    pub fn leaf_value(&self, features: &[f32]) -> f32 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        let mut i = 0usize;
        loop {
            let n = &self.nodes[i];
            if n.is_leaf {
                return n.leaf_value;
            }
            let v = features[n.feat as usize];
            let go_left = if v.is_nan() {
                n.default_left
            } else {
                // XGBoost 的判据就是 v < split_cond 走左,不是 <=。
                v < n.split_cond
            };
            i = if go_left { n.left as usize } else { n.right as usize };
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub trees: Vec<Tree>,
    pub base_score: f32,
    pub objective: Objective,
    pub n_features: usize,
    /// 早停选中的轮次(0 起)。**预测默认不用它** —— XGBoost 3.4.1 的
    /// `Booster.predict()` 也是拿全部树算的,best_iteration 只是记在
    /// attributes 里的元信息。要截断到最佳轮次得显式调
    /// `predict_margin_upto`。
    ///
    /// (sklearn 那层包装的行为不一样,但我们对齐的是 Booster。)
    pub best_iteration: Option<usize>,
    /// 早停时最佳轮次的指标值,和 best_iteration 一起存进 attributes。
    pub best_score: Option<f32>,
    /// 训练时的**规范特征名与顺序**。空 = 位置模式(无表头 CSV / numpy)。
    ///
    /// 这是文件预测能按列名绑定的**唯一**依据:预测文件的列顺序可以和
    /// 训练时不同,Rust 按这份名单重排。空着就退回按位置绑定,并只能
    /// 校验列数。
    ///
    /// ⚠️ **标签列不在里面。** 标签是训练期的元信息,预测时不该被要求
    /// 提供 —— 把它混进特征名单会让「预测必须带标签列」这种错误契约
    /// 变得无法察觉。
    pub feature_names: Vec<String>,
    /// 绑定模式。`None` = **老模型**,没有记录过这件事。
    ///
    /// ⚠️ **空的 `feature_names` 有两种完全不同的来源**:numpy / 无表头 CSV
    /// 训出来的(真·位置模式),和本次改动之前从命名数据训出来的
    /// (其实是命名模式,只是没存名字)。两者在 JSON 里长得一模一样,
    /// 靠它们**猜不出**该怎么绑 —— 按位置绑一个本该按名字绑的模型,
    /// 遇到列序不同的预测文件就会静默算错。
    ///
    /// 所以这里显式记一笔:`None` 的模型做文件预测直接报错,让用户重训
    /// 或改用 numpy 预测,而不是替他赌一把。
    pub schema_mode: Option<SchemaMode>,
}

/// 特征怎么和输入列对上。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SchemaMode {
    /// 按列名绑;`feature_names` 就是规范顺序。输入列序可以不同。
    Named,
    /// 按位置绑;只能校验列数。无表头 CSV 和 numpy 走这条。
    Positional,
}

impl SchemaMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Named => "named",
            Self::Positional => "positional",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "named" => Some(Self::Named),
            "positional" => Some(Self::Positional),
            _ => None,
        }
    }
}

impl Model {
    /// 原始分数(链接函数**之前**)。训练循环累加的就是这个空间的值。
    ///
    /// 用**全部**树,即使早停记了 best_iteration —— 和 XGBoost 的
    /// `Booster.predict()` 一致(实测 3.4.1 默认就是全部树)。
    pub fn predict_margin(&self, features: &[f32]) -> f32 {
        self.predict_margin_upto(features, self.trees.len())
    }

    /// 只用前 `n_trees` 棵树算原始分数。对应 XGBoost 的
    /// `iteration_range=(0, n_trees)`。
    ///
    /// 早停之后想要"最佳轮次的模型",传 `best_iteration + 1`。
    pub fn predict_margin_upto(&self, features: &[f32], n_trees: usize) -> f32 {
        let n = n_trees.min(self.trees.len());
        let mut acc = crate::train::base_margin(self.objective, self.base_score);
        for t in &self.trees[..n] {
            acc += t.leaf_value(features);
        }
        acc
    }

    /// 和 XGBoost 的 `Booster.predict()` 同义:二分类返回概率,
    /// 回归返回预测值本身。要原始分数用 `predict_margin`。
    pub fn predict(&self, features: &[f32]) -> f32 {
        let m = self.predict_margin(features);
        match self.objective {
            Objective::SquaredError => m,
            Objective::Logistic => 1.0 / (1.0 + (-m).exp()),
        }
    }

    /// 导出成 XGBoost 兼容的 JSON。
    pub fn to_xgboost_json(&self) -> Result<String, serde_json::Error> {
        let trees: Vec<xgb::Tree> = self
            .trees
            .iter()
            .enumerate()
            .map(|(id, t)| xgb::Tree::from_tree(id as i32, t, self.n_features))
            .collect();

        let file = xgb::File {
            version: XGB_VERSION,
            learner: xgb::Learner {
                // XGBoost 把早停结果放在 attributes 里,值是**字符串**
                attributes: {
                    let mut a = BTreeMap::new();
                    if let Some(b) = self.best_iteration {
                        a.insert("best_iteration".to_string(), b.to_string());
                    }
                    if let Some(s) = self.best_score {
                        a.insert("best_score".to_string(), s.to_string());
                    }
                    // 绑定模式存在 attributes 里:XGBoost 的 JSON 本来就有这个
                    // 自由字段,不需要为它改格式,别的实现读到也只是多一个
                    // 不认识的属性,不会因此读不了模型。
                    if let Some(m) = self.schema_mode {
                        a.insert("ferrisboost_schema_mode".to_string(), m.as_str().to_string());
                    }
                    a
                },
                feature_names: self.feature_names.clone(),
                // 类型目前统一是数值;留空表示"未声明",和 XGBoost 一致。
                feature_types: Vec::new(),
                gradient_booster: xgb::GradientBooster {
                    name: "gbtree".into(),
                    model: xgb::GbTreeModel {
                        gbtree_model_param: xgb::GbTreeModelParam {
                            num_trees: trees.len().to_string(),
                            num_parallel_tree: "1".into(),
                        },
                        iteration_indptr: (0..=trees.len() as i32).collect(),
                        tree_info: vec![0; trees.len()],
                        trees,
                    },
                },
                learner_model_param: xgb::LearnerModelParam {
                    // base_score 存的是**变换前**空间的值(二分类是概率),
                    // 和 TrainConfig.base_score 同一个空间。XGBoost 自己
                    // 在加载时调 ProbToMargin,这里不能先转 logit。
                    base_score: self.base_score.to_string(),
                    boost_from_average: "1".into(),
                    num_class: "0".into(),
                    num_feature: self.n_features.to_string(),
                    num_target: "1".into(),
                },
                objective: xgb::ObjectiveJson {
                    name: objective_name(self.objective).into(),
                    rest: reg_loss_param(),
                },
            },
        };
        serde_json::to_string(&file)
    }

    /// 加载 XGBoost 训好的模型。主要用途是预测对拍:同一个模型两边
    /// 各跑一遍,预测值必须逐个对得上。
    pub fn from_xgboost_json(s: &str) -> Result<Self, serde_json::Error> {
        let file: xgb::File = serde_json::from_str(s)?;
        let learner = file.learner;
        let booster = learner.gradient_booster;

        if booster.name != "gbtree" {
            return Err(unsupported(format_args!(
                "只支持 gbtree,这个模型是 {}",
                booster.name
            )));
        }
        let mp = &learner.learner_model_param;
        if parse_param::<i32>("num_class", &mp.num_class)? > 1 {
            return Err(unsupported(format_args!(
                "多分类不在第一版范围里(num_class = {})",
                mp.num_class
            )));
        }
        if parse_param::<i32>("num_parallel_tree", &booster.model.gbtree_model_param.num_parallel_tree)? != 1 {
            return Err(unsupported(format_args!(
                "num_parallel_tree = {}(random forest 模式)不支持",
                booster.model.gbtree_model_param.num_parallel_tree
            )));
        }

        let objective = match learner.objective.name.as_str() {
            "reg:squarederror" | "reg:linear" => Objective::SquaredError,
            "binary:logistic" => Objective::Logistic,
            other => {
                return Err(unsupported(format_args!("objective {other} 不在第一版范围里")))
            }
        };

        let n_features = parse_param::<usize>("num_feature", &mp.num_feature)?;
        let base_score = parse_base_score(&mp.base_score)?;

        let trees = booster
            .model
            .trees
            .iter()
            .map(|t| t.to_tree())
            .collect::<Result<Vec<_>, _>>()?;

        // 闭包没法泛型,两个字段的目标类型不同,分开写
        let attrs = &learner.attributes;
        Ok(Self {
            trees,
            base_score,
            objective,
            n_features,
            // 存进去什么就读回什么,save → load → predict 必须保持同一套
            // 绑定语义。属性缺失 = 老模型,`None`,文件预测会明确拒绝。
            feature_names: learner.feature_names.clone(),
            schema_mode: attrs
                .get("ferrisboost_schema_mode")
                .and_then(|s| SchemaMode::parse(s)),
            best_iteration: attrs
                .get("best_iteration")
                .and_then(|v| v.trim().parse::<usize>().ok()),
            best_score: attrs.get("best_score").and_then(|v| v.trim().parse::<f32>().ok()),
        })
    }
}

/// 把分裂候选的 bin 落成节点上的阈值。
///
/// `bin` 是右边界(`bin_of(v) < bin` 归左),而 `find_bin` 的语义是
/// 「cut 是上界」,于是两边接上就是 `v < cuts[bin - 1]` —— 正好是
/// XGBoost 的判据,不用再做 ±1 的换算。
///
/// 两个端点没有对应的切分点,是「按是否缺失分裂」:
/// - `bin == 0`:非缺失行全走右,阈值取一个不大于任何取值的数
/// - `bin == n_bins`:非缺失行全走左,阈值取一个大于任何取值的数
///
/// 用 `f32::MIN` / `f32::MAX` 而不是 ±inf:JSON 里没有无穷,
/// serde 会把它写成 `null`,XGBoost 读回来直接炸。
///
/// 有限哨兵的代价只有一个:特征值**恰好等于** `f32::MAX` 时
/// `v < MAX` 为假,这一行会走右边而不是训练时的左边。没有任何有限
/// f32 比 `f32::MAX` 更大,换个数也躲不掉;真出现这种取值,分箱和
/// 梯度早就先出问题了。
pub fn split_threshold(cuts: &BinCuts, feat: FeatId, bin: u32) -> f32 {
    let c = cuts.cuts_for(feat);
    if bin == 0 {
        f32::MIN
    } else if bin as usize > c.len() {
        f32::MAX
    } else {
        c[bin as usize - 1]
    }
}

/// `split_threshold` 的逆:把节点上的浮点阈值换回 bin 右边界。
///
/// 训练集和验证集都是量化过的,沿树下降时手里只有 bin 没有原值,
/// 所以要把阈值换算回去比。**不能**直接用 `find_bin(split_cond)`:
/// 右端哨兵 `f32::MAX` 会被算成 `n_bins - 1`,顶上那个 bin 的行就
/// 走错边 —— 训练时它在左边,预测时跑到右边,而且只错这一个 bin,
/// 指标上几乎看不出来。
pub fn split_bin(cuts: &BinCuts, feat: FeatId, split_cond: f32) -> u32 {
    if split_cond == f32::MAX {
        cuts.n_bins(feat) as u32
    } else {
        cuts.find_bin(feat, split_cond) as u32
    }
}

/// 写出去的版本号。字段布局照的是 2.x 的 schema
/// (`iteration_indptr`、`boost_from_average`、`size_leaf_vector = 1`
/// 都是 2.0 才有的),所以版本号也要跟着报 2.x,不然 XGBoost 会走
/// 老版本的兼容分支去找不存在的字段。
const XGB_VERSION: [i32; 3] = [2, 0, 0];

fn objective_name(obj: Objective) -> &'static str {
    match obj {
        Objective::SquaredError => "reg:squarederror",
        Objective::Logistic => "binary:logistic",
    }
}

/// 两个 objective 都挂在 reg_loss_param 下面,XGBoost 加载时会找它。
fn reg_loss_param() -> BTreeMap<String, serde_json::Value> {
    let mut inner = serde_json::Map::new();
    inner.insert("scale_pos_weight".into(), serde_json::Value::String("1".into()));
    let mut rest = BTreeMap::new();
    rest.insert("reg_loss_param".to_string(), serde_json::Value::Object(inner));
    rest
}

fn unsupported(args: std::fmt::Arguments) -> serde_json::Error {
    serde_json::Error::custom(args)
}

/// base_score 有两种写法,都得吃下:
/// - 1.x / 2.x:裸标量 `"5E-1"`
/// - 3.x:**带方括号的向量** `"[5E-1]"`(多目标留下的形状)
///
/// 实测 3.4.1 加载我们写的裸标量没问题,所以导出仍然写裸标量 ——
/// 往回兼容老版本更划算。
fn parse_base_score(raw: &str) -> Result<f32, serde_json::Error> {
    let t = raw.trim();
    let inner = t
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or(t);

    let mut parts = inner.split(',').map(str::trim).filter(|s| !s.is_empty());
    let first = parts
        .next()
        .ok_or_else(|| unsupported(format_args!("base_score 是空的:{raw:?}")))?;
    if parts.next().is_some() {
        return Err(unsupported(format_args!(
            "base_score 是向量({raw:?}),多目标模型不在第一版范围里"
        )));
    }
    parse_param("base_score", first)
}

/// XGBoost 的参数**全部是字符串**,这是它自己的约定不是笔误。
/// 而且写出来的浮点是 "5E-1" 这种科学计数法,解析要能吃下。
fn parse_param<T: std::str::FromStr>(name: &str, raw: &str) -> Result<T, serde_json::Error> {
    raw.trim()
        .parse::<T>()
        .map_err(|_| unsupported(format_args!("参数 {name} 的值 {raw:?} 解析不了")))
}

/// XGBoost JSON schema 的镜像。
///
/// 刻意写成一组死板的 struct 而不是操作 `serde_json::Value`:字段名
/// 拼错在这里是编译期错误,在 Value 里是运行时才发现的 None。
mod xgb {
    use super::*;

    #[derive(Serialize, Deserialize)]
    pub struct File {
        pub version: [i32; 3],
        pub learner: Learner,
    }

    #[derive(Serialize, Deserialize)]
    pub struct Learner {
        #[serde(default)]
        pub attributes: BTreeMap<String, String>,
        #[serde(default)]
        pub feature_names: Vec<String>,
        #[serde(default)]
        pub feature_types: Vec<String>,
        pub gradient_booster: GradientBooster,
        pub learner_model_param: LearnerModelParam,
        pub objective: ObjectiveJson,
    }

    #[derive(Serialize, Deserialize)]
    pub struct GradientBooster {
        pub name: String,
        pub model: GbTreeModel,
    }

    #[derive(Serialize, Deserialize)]
    pub struct GbTreeModel {
        pub gbtree_model_param: GbTreeModelParam,
        /// 2.0 才有:每轮的树区间。单目标每轮一棵,就是 0..=n。
        #[serde(default)]
        pub iteration_indptr: Vec<i32>,
        pub tree_info: Vec<i32>,
        pub trees: Vec<Tree>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct GbTreeModelParam {
        pub num_trees: String,
        pub num_parallel_tree: String,
    }

    #[derive(Serialize, Deserialize)]
    pub struct LearnerModelParam {
        pub base_score: String,
        #[serde(default)]
        pub boost_from_average: String,
        pub num_class: String,
        pub num_feature: String,
        #[serde(default)]
        pub num_target: String,
    }

    #[derive(Serialize, Deserialize)]
    pub struct ObjectiveJson {
        pub name: String,
        /// reg_loss_param 之类的东西原样带着走,不解释也不丢。
        #[serde(flatten)]
        pub rest: BTreeMap<String, serde_json::Value>,
    }

    /// 一棵树。XGBoost 存的是**结构数组**(每个字段一个数组,按节点
    /// 下标对齐),不是节点数组 —— 别想当然按 `Vec<Node>` 去读。
    #[derive(Serialize, Deserialize)]
    pub struct Tree {
        pub base_weights: Vec<f32>,
        #[serde(default)]
        pub categories: Vec<i32>,
        #[serde(default)]
        pub categories_nodes: Vec<i32>,
        #[serde(default)]
        pub categories_segments: Vec<i32>,
        #[serde(default)]
        pub categories_sizes: Vec<i32>,
        pub default_left: Vec<u8>,
        pub id: i32,
        pub left_children: Vec<i32>,
        pub loss_changes: Vec<f32>,
        pub parents: Vec<i32>,
        pub right_children: Vec<i32>,
        pub split_conditions: Vec<f32>,
        pub split_indices: Vec<u32>,
        #[serde(default)]
        pub split_type: Vec<u8>,
        pub sum_hessian: Vec<f32>,
        pub tree_param: TreeParam,
    }

    #[derive(Serialize, Deserialize)]
    pub struct TreeParam {
        pub num_deleted: String,
        pub num_feature: String,
        pub num_nodes: String,
        pub size_leaf_vector: String,
    }

    impl Tree {
        pub fn from_tree(id: i32, t: &super::Tree, n_features: usize) -> Self {
            let n = t.nodes.len();
            let mut out = Tree {
                base_weights: Vec::with_capacity(n),
                categories: Vec::new(),
                categories_nodes: Vec::new(),
                categories_segments: Vec::new(),
                categories_sizes: Vec::new(),
                default_left: Vec::with_capacity(n),
                id,
                left_children: Vec::with_capacity(n),
                loss_changes: Vec::with_capacity(n),
                parents: vec![NO_PARENT; n],
                right_children: Vec::with_capacity(n),
                split_conditions: Vec::with_capacity(n),
                split_indices: Vec::with_capacity(n),
                // 0 = 数值分裂。类别分裂(1)第一版不产出。
                split_type: vec![0; n],
                sum_hessian: Vec::with_capacity(n),
                tree_param: TreeParam {
                    num_deleted: "0".into(),
                    num_feature: n_features.to_string(),
                    num_nodes: n.to_string(),
                    // 标量叶子。2.x 写的是 "1",1.x 写 "0"。
                    size_leaf_vector: "1".into(),
                },
            };

            for (i, node) in t.nodes.iter().enumerate() {
                out.left_children.push(node.left);
                out.right_children.push(node.right);
                out.default_left.push(node.default_left as u8);
                out.loss_changes.push(node.gain);
                out.sum_hessian.push(node.sum_hess);
                if node.is_leaf {
                    // 叶子把值存在 split_conditions 里,这是 XGBoost 的
                    // 存法(节点结构体是个 union),不是偷懒。
                    out.split_conditions.push(node.leaf_value);
                    out.split_indices.push(0);
                    out.base_weights.push(node.leaf_value);
                } else {
                    out.split_conditions.push(node.split_cond);
                    out.split_indices.push(node.feat);
                    out.base_weights.push(0.0);
                    out.parents[node.left as usize] = i as i32;
                    out.parents[node.right as usize] = i as i32;
                }
            }
            out
        }

        pub fn to_tree(&self) -> Result<super::Tree, serde_json::Error> {
            let n = self.left_children.len();
            let n_features = parse_param::<usize>("tree_param.num_feature", &self.tree_param.num_feature)?;
            if n == 0 {
                return Err(unsupported(format_args!("树 {} 没有根节点", self.id)));
            }
            for (name, len) in [
                ("right_children", self.right_children.len()),
                ("split_conditions", self.split_conditions.len()),
                ("split_indices", self.split_indices.len()),
                ("default_left", self.default_left.len()),
            ] {
                if len != n {
                    return Err(unsupported(format_args!(
                        "树 {} 的 {name} 长度 {len} 和节点数 {n} 对不上",
                        self.id
                    )));
                }
            }
            if !self.categories_sizes.is_empty() || self.split_type.iter().any(|&t| t != 0) {
                return Err(unsupported(format_args!(
                    "树 {} 有类别型分裂,第一版只支持数值分裂",
                    self.id
                )));
            }

            let mut nodes = Vec::with_capacity(n);
            let mut parents = vec![0u8; n];
            for i in 0..n {
                let (left, right) = (self.left_children[i], self.right_children[i]);
                if left == NO_CHILD {
                    if right != NO_CHILD {
                        return Err(unsupported(format_args!(
                            "树 {} 的节点 {i} 只有一个孩子:{left} / {right}",
                            self.id
                        )));
                    }
                    nodes.push(super::Node {
                        leaf_value: self.split_conditions[i],
                        sum_hess: self.sum_hessian.get(i).copied().unwrap_or(0.0),
                        ..super::Node::leaf(self.split_conditions[i], 0.0)
                    });
                    continue;
                }
                if left < 0
                    || right < 0
                    || left as usize >= n
                    || right as usize >= n
                    || left == right
                {
                    return Err(unsupported(format_args!(
                        "树 {} 的节点 {i} 孩子下标非法:{left} / {right}",
                        self.id
                    )));
                }
                for child in [left as usize, right as usize] {
                    parents[child] = parents[child].saturating_add(1);
                    if parents[child] > 1 {
                        return Err(unsupported(format_args!(
                            "树 {} 的节点 {child} 有多个父节点",
                            self.id
                        )));
                    }
                }
                if self.split_indices[i] as usize >= n_features {
                    return Err(unsupported(format_args!(
                        "树 {} 的节点 {i} 特征 {} 越界,共 {n_features} 个",
                        self.id, self.split_indices[i]
                    )));
                }
                nodes.push(super::Node {
                    feat: self.split_indices[i],
                    split_cond: self.split_conditions[i],
                    left,
                    right,
                    default_left: self.default_left[i] != 0,
                    leaf_value: 0.0,
                    is_leaf: false,
                    gain: self.loss_changes.get(i).copied().unwrap_or(0.0),
                    sum_hess: self.sum_hessian.get(i).copied().unwrap_or(0.0),
                });
            }
            if parents[0] != 0 {
                return Err(unsupported(format_args!("树 {} 的根节点被别的节点引用", self.id)));
            }
            let mut seen = vec![false; n];
            let mut stack = vec![0usize];
            while let Some(i) = stack.pop() {
                if std::mem::replace(&mut seen[i], true) {
                    return Err(unsupported(format_args!("树 {} 含环", self.id)));
                }
                if !nodes[i].is_leaf {
                    stack.push(nodes[i].left as usize);
                    stack.push(nodes[i].right as usize);
                }
            }
            if let Some(i) = seen.iter().position(|&reachable| !reachable) {
                return Err(unsupported(format_args!("树 {} 的节点 {i} 从根不可达", self.id)));
            }
            Ok(super::Tree { nodes })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::columns::BinCuts;

    /// 一棵手工的两层树,feat 0 分两次:
    ///
    /// ```text
    ///          x0 < 1.5 ?  (缺失走左)
    ///         /          \
    ///   x1 < 0.5 ?        leaf 3.0
    ///   (缺失走右)
    ///   /       \
    /// leaf 1.0  leaf 2.0
    /// ```
    fn fixture_tree() -> Tree {
        let mut t = Tree::default();
        let root = t.push_leaf(0.0, 10.0);
        let (l, _r) = t.split_leaf(root, 0, 1.5, true, 4.0, 0.0, 6.0, 3.0, 4.0);
        t.split_leaf(l, 1, 0.5, false, 2.0, 1.0, 3.0, 2.0, 3.0);
        t
    }

    fn fixture_model() -> Model {
        Model {
            feature_names: Vec::new(),
            schema_mode: None,
            trees: vec![fixture_tree()],
            base_score: 0.5,
            objective: Objective::SquaredError,
            n_features: 2,
            best_iteration: None,
            best_score: None,
        }
    }

    #[test]
    fn descends_by_strict_less_than() {
        let t = fixture_tree();
        // 边界上的值:1.5 不小于 1.5,走右
        assert_eq!(t.leaf_value(&[1.5, 0.0]), 3.0);
        assert_eq!(t.leaf_value(&[1.49, 0.0]), 1.0);
        assert_eq!(t.leaf_value(&[0.0, 0.5]), 2.0, "0.5 不小于 0.5,走右");
        assert_eq!(t.leaf_value(&[0.0, 0.49]), 1.0);
    }

    #[test]
    fn missing_follows_the_learned_direction() {
        let t = fixture_tree();
        // 根节点 default_left = true:缺失走左,再进第二层
        assert_eq!(t.leaf_value(&[f32::NAN, 0.0]), 1.0);
        // 第二层 default_left = false:缺失走右
        assert_eq!(t.leaf_value(&[f32::NAN, f32::NAN]), 2.0);
        assert_eq!(t.leaf_value(&[0.0, f32::NAN]), 2.0);
    }

    #[test]
    fn empty_tree_contributes_nothing() {
        assert_eq!(Tree::default().leaf_value(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn children_always_come_after_their_parent() {
        // 下降不绕圈就是靠这个不变量。
        let t = fixture_tree();
        for (i, n) in t.nodes.iter().enumerate() {
            if !n.is_leaf {
                assert!(n.left as usize > i, "节点 {i} 的左孩子在前面");
                assert!(n.right as usize > i, "节点 {i} 的右孩子在前面");
            }
        }
    }

    #[test]
    fn margin_sums_base_and_trees() {
        let mut m = fixture_model();
        m.trees.push(fixture_tree());
        // 回归:base_margin = base_score,两棵一样的树各给 3.0
        assert_eq!(m.predict_margin(&[9.0, 0.0]), 0.5 + 3.0 + 3.0);
        // 回归的 predict 不做变换
        assert_eq!(m.predict(&[9.0, 0.0]), 6.5);
    }

    #[test]
    fn logistic_predict_is_a_probability() {
        let m = Model {
            objective: Objective::Logistic,
            base_score: 0.5, // logit(0.5) = 0 —— 起点是 0 不是 0.5
            ..fixture_model()
        };
        let margin = m.predict_margin(&[9.0, 0.0]);
        assert_eq!(margin, 3.0, "base_score 0.5 转 raw margin 应该是 0");
        let p = m.predict(&[9.0, 0.0]);
        assert!((p - 1.0 / (1.0 + (-3.0f32).exp())).abs() < 1e-7);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn json_round_trips_through_our_own_reader() {
        let m = fixture_model();
        let back = Model::from_xgboost_json(&m.to_xgboost_json().unwrap()).unwrap();

        assert_eq!(back.n_features, m.n_features);
        assert_eq!(back.base_score, m.base_score);
        assert_eq!(back.objective, m.objective);
        assert_eq!(back.trees.len(), 1);
        for row in [[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [f32::NAN, 0.5]] {
            assert_eq!(back.predict(&row), m.predict(&row), "行 {row:?}");
        }
    }

    #[test]
    fn logistic_model_round_trips() {
        let m = Model { objective: Objective::Logistic, base_score: 0.25, ..fixture_model() };
        let back = Model::from_xgboost_json(&m.to_xgboost_json().unwrap()).unwrap();
        assert_eq!(back.objective, Objective::Logistic);
        // base_score 存的是概率空间的值,不能在导出时就转成 logit ——
        // 转了的话往返一次就变成 logit(logit(0.25)),悄悄地错。
        assert!((back.base_score - 0.25).abs() < 1e-7);
        assert!((back.predict(&[9.0, 0.0]) - m.predict(&[9.0, 0.0])).abs() < 1e-7);
    }

    #[test]
    fn dump_has_the_shape_xgboost_expects() {
        // 这条锁的是文件结构本身。XGBoost 的加载器按名字取字段,
        // 拼错一个键它就报「找不到」,而我们自己的 reader 是对称的,
        // 光靠 round trip 发现不了。
        let v: serde_json::Value =
            serde_json::from_str(&fixture_model().to_xgboost_json().unwrap()).unwrap();

        assert_eq!(v["version"][0], 2);
        let learner = &v["learner"];
        assert_eq!(learner["gradient_booster"]["name"], "gbtree");
        assert_eq!(learner["objective"]["name"], "reg:squarederror");
        assert_eq!(learner["objective"]["reg_loss_param"]["scale_pos_weight"], "1");
        // 所有 param 都是字符串,不是数字
        assert_eq!(learner["learner_model_param"]["num_feature"], "2");
        assert_eq!(learner["learner_model_param"]["num_class"], "0");
        assert_eq!(learner["learner_model_param"]["base_score"], "0.5");

        let model = &learner["gradient_booster"]["model"];
        assert_eq!(model["gbtree_model_param"]["num_trees"], "1");
        assert_eq!(model["tree_info"], serde_json::json!([0]));
        assert_eq!(model["iteration_indptr"], serde_json::json!([0, 1]));

        let t = &model["trees"][0];
        assert_eq!(t["tree_param"]["num_nodes"], "5");
        assert_eq!(t["left_children"], serde_json::json!([1, 3, -1, -1, -1]));
        assert_eq!(t["right_children"], serde_json::json!([2, 4, -1, -1, -1]));
        // 根的父亲是哨兵,不是 -1
        assert_eq!(t["parents"], serde_json::json!([NO_PARENT, 0, 0, 1, 1]));
        assert_eq!(t["default_left"], serde_json::json!([1, 0, 0, 0, 0]));
        assert_eq!(t["split_indices"], serde_json::json!([0, 1, 0, 0, 0]));
        // 叶子的值存在 split_conditions 里
        assert_eq!(t["split_conditions"], serde_json::json!([1.5, 0.5, 3.0, 1.0, 2.0]));
        assert_eq!(t["sum_hessian"], serde_json::json!([10.0, 6.0, 4.0, 3.0, 3.0]));
        assert_eq!(t["loss_changes"], serde_json::json!([4.0, 2.0, 0.0, 0.0, 0.0]));
        // 数值分裂,不带类别
        assert_eq!(t["split_type"], serde_json::json!([0, 0, 0, 0, 0]));
        assert_eq!(t["categories"], serde_json::json!([]));
    }

    /// 一个 XGBoost 真实 dump 的样子:参数是字符串、浮点是科学计数法、
    /// 还有我们不认识的字段。都得能吃下去。
    const XGB_DUMP: &str = r#"{
      "version": [2, 0, 3],
      "learner": {
        "attributes": {"best_iteration": "3"},
        "feature_names": [],
        "feature_types": [],
        "gradient_booster": {
          "model": {
            "gbtree_model_param": {"num_parallel_tree": "1", "num_trees": "1"},
            "iteration_indptr": [0, 1],
            "tree_info": [0],
            "trees": [{
              "base_weights": [0E0, -1E0, 5E-1],
              "categories": [], "categories_nodes": [], "categories_segments": [],
              "categories_sizes": [],
              "default_left": [1, 0, 0],
              "id": 0,
              "left_children": [1, -1, -1],
              "loss_changes": [1.25E1, 0E0, 0E0],
              "parents": [2147483647, 0, 0],
              "right_children": [2, -1, -1],
              "split_conditions": [2.5E0, -3E-1, 1.5E-1],
              "split_indices": [1, 0, 0],
              "split_type": [0, 0, 0],
              "sum_hessian": [8E0, 5E0, 3E0],
              "tree_param": {"num_deleted": "0", "num_feature": "3",
                             "num_nodes": "3", "size_leaf_vector": "1"}
            }]
          },
          "name": "gbtree"
        },
        "learner_model_param": {"base_score": "5E-1", "boost_from_average": "1",
                                "num_class": "0", "num_feature": "3", "num_target": "1"},
        "objective": {"name": "binary:logistic", "reg_loss_param": {"scale_pos_weight": "1"}}
      }
    }"#;

    #[test]
    fn reads_a_real_looking_xgboost_dump() {
        let m = Model::from_xgboost_json(XGB_DUMP).unwrap();
        assert_eq!(m.objective, Objective::Logistic);
        assert_eq!(m.n_features, 3);
        assert_eq!(m.base_score, 0.5, "5E-1 要能解析成 0.5");
        assert_eq!(m.trees.len(), 1);

        let t = &m.trees[0];
        assert!(!t.nodes[0].is_leaf);
        assert_eq!(t.nodes[0].feat, 1, "split_indices 才是特征号");
        assert_eq!(t.nodes[0].split_cond, 2.5);
        assert!(t.nodes[0].default_left);
        assert_eq!(t.nodes[0].sum_hess, 8.0);
        assert_eq!(t.nodes[0].gain, 12.5);
        assert!(t.nodes[1].is_leaf);
        assert_eq!(t.nodes[1].leaf_value, -0.3, "叶子值在 split_conditions 里");

        // x1 = 2.0 < 2.5 走左;base_score 0.5 → margin 0
        assert_eq!(m.predict_margin(&[0.0, 2.0, 0.0]), -0.3);
        // 缺失走左(default_left)
        assert_eq!(m.predict_margin(&[0.0, f32::NAN, 0.0]), -0.3);
        assert_eq!(m.predict_margin(&[0.0, 9.0, 0.0]), 0.15);
    }

    #[test]
    fn rejects_what_it_cannot_represent() {
        let cases = [
            ("\"num_class\": \"0\"", "\"num_class\": \"3\"", "多分类"),
            ("\"binary:logistic\"", "\"rank:pairwise\"", "objective"),
            ("\"num_parallel_tree\": \"1\"", "\"num_parallel_tree\": \"4\"", "森林"),
            ("\"split_type\": [0, 0, 0]", "\"split_type\": [1, 0, 0]", "类别分裂"),
            ("\"left_children\": [1, -1, -1]", "\"left_children\": [0, -1, -1]", "环"),
            ("\"right_children\": [2, -1, -1]", "\"right_children\": [-1, -1, -1]", "单边孩子"),
            ("\"split_indices\": [1, 0, 0]", "\"split_indices\": [3, 0, 0]", "越界特征"),
        ];
        for (from, to, what) in cases {
            let broken = XGB_DUMP.replace(from, to);
            assert_ne!(broken, XGB_DUMP, "替换没生效:{from}");
            let err = Model::from_xgboost_json(&broken)
                .expect_err(&format!("{what} 应该被拒绝"));
            // 报错要说人话,不能是 "invalid type at line 1"
            assert!(!err.to_string().is_empty());
        }
    }

    #[test]
    fn base_score_parses_both_spellings() {
        // 3.x 写成向量,1.x / 2.x 写成标量
        assert_eq!(parse_base_score("[5E-1]").unwrap(), 0.5);
        assert_eq!(parse_base_score("5E-1").unwrap(), 0.5);
        assert_eq!(parse_base_score(" [ -1.5 ] ").unwrap(), -1.5);
        // 多目标:宁可报错,也不能只取第一个悄悄往下跑
        assert!(parse_base_score("[0.1, 0.2]").is_err());
        assert!(parse_base_score("[]").is_err());
        assert!(parse_base_score("abc").is_err());
    }

    #[test]
    fn threshold_matches_find_bin_on_every_boundary() {
        // 这条把 columns.rs 的分箱和 tree.rs 的阈值钉在一起:
        // 对任何值 v 和任何右边界 bin,
        //     find_bin(v) < bin   ⟺   v < split_threshold(bin)
        // 错开一位的话训练和预测就走不同的分支。
        let cuts = BinCuts::new(vec![-2.5, 0.0, 1.5, 7.25], vec![0, 4]);
        let n_bins = cuts.n_bins(0) as u32;
        assert_eq!(n_bins, 5);

        // f32::MAX 不在探针里:见 split_threshold 的文档,右端哨兵是
        // 有限值,v == f32::MAX 是唯一对不上的取值。下面单独断言它。
        let probes = [
            f32::MIN, -1e9, -3.0, -2.5, -2.49, -0.1, 0.0, 0.1, 1.4, 1.5,
            1.6, 7.0, 7.25, 7.26, 1e9, f32::MAX / 2.0,
        ];
        for bin in 0..=n_bins {
            let thr = split_threshold(&cuts, 0, bin);
            for v in probes {
                let by_bin = (cuts.find_bin(0, v) as u32) < bin;
                let by_thr = v < thr;
                assert_eq!(by_bin, by_thr, "bin {bin}, v {v}, 阈值 {thr}");
            }
        }
    }

    #[test]
    fn split_bin_inverts_split_threshold() {
        // 两个方向必须严丝合缝:训练时按 bin 分区,预测时按阈值换回
        // 的 bin 走,错开一位就是"训练和预测走不同的分支"。
        let cuts = BinCuts::new(vec![-2.5, 0.0, 1.5, 7.25], vec![0, 4]);
        for bin in 0..=cuts.n_bins(0) as u32 {
            let thr = split_threshold(&cuts, 0, bin);
            assert_eq!(split_bin(&cuts, 0, thr), bin, "bin {bin} 的阈值是 {thr}");
        }
    }

    #[test]
    fn endpoint_thresholds_send_everyone_one_way() {
        let cuts = BinCuts::new(vec![0.0, 1.0], vec![0, 2]);
        // bin == 0:非缺失全走右
        assert_eq!(split_threshold(&cuts, 0, 0), f32::MIN);
        // bin == n_bins:非缺失全走左
        assert_eq!(split_threshold(&cuts, 0, 3), f32::MAX);
        // 有限值,不是 inf —— inf 会被 serde 写成 null,XGBoost 读不了
        assert!(split_threshold(&cuts, 0, 0).is_finite());
        assert!(split_threshold(&cuts, 0, 3).is_finite());

        // 有限哨兵唯一对不上的地方,钉在这里免得以后被当成 bug 改坏:
        // 判据是 v < thr 走左,而 f32::MAX 不小于 f32::MAX,所以这一行
        // 走右 —— 和训练时的左边不一致。
        assert!(f32::MAX >= split_threshold(&cuts, 0, 3));
    }
}
