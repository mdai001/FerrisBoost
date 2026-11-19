use super::{
    bounded_node_gain, bounded_split_gain, xgb_weight, NodeBounds, SplitCandidate, MIN_GAIN_EPS,
};
use crate::{
    categorical::CatBits,
    hist::Histogram,
    types::{FeatId, GradPairFixed, GradQuantizer, TrainParams},
};

pub(crate) struct LevelSplit {
    pub score: f64,
    pub candidates: Vec<SplitCandidate>,
}

impl LevelSplit {
    pub fn beats(&self, other: &Self) -> bool {
        self.score > other.score
            || (self.score == other.score && self.candidates[0].feat < other.candidates[0].feat)
    }
}

struct Parent {
    sum: GradPairFixed,
    bounds: NodeBounds,
    gain: f32,
}

struct ScoredSplit {
    at: usize,
    missing_left: bool,
    score: f64,
    children: Vec<(GradPairFixed, GradPairFixed, f32)>,
}

struct Scan<'a> {
    parents: &'a [Parent],
    quantizer: &'a GradQuantizer,
    params: &'a TrainParams,
    constraint: i8,
    best: ScoredSplit,
    scratch: Vec<(GradPairFixed, GradPairFixed, f32)>,
}

impl Scan<'_> {
    fn consider(
        &mut self,
        at: usize,
        missing_left: bool,
        part_is_left: bool,
        parts: impl Iterator<Item = GradPairFixed>,
    ) {
        self.scratch.clear();
        let mut score = 0.0;
        for (parent, part) in self.parents.iter().zip(parts) {
            let rest = parent.sum - part;
            let (left, right) = if part_is_left {
                (part, rest)
            } else {
                (rest, part)
            };
            let lf = self.quantizer.to_f64(left);
            let rf = self.quantizer.to_f64(right);
            let Some(improvement) = bounded_split_gain(
                parent.gain,
                lf,
                rf,
                self.constraint,
                parent.bounds,
                self.params,
            ) else {
                return;
            };
            score += f64::from(improvement);
            self.scratch.push((left, right, improvement));
        }
        debug_assert_eq!(self.scratch.len(), self.parents.len());
        // An individual leaf can lose score. All child pairs must be valid,
        // and the total must pay gamma for each new split in the level.
        if score.is_finite() && score > self.best.score {
            self.best.at = at;
            self.best.missing_left = missing_left;
            self.best.score = score;
            std::mem::swap(&mut self.scratch, &mut self.best.children);
        }
    }

    fn numeric(&mut self, bins: &[&[GradPairFixed]], prefix: &mut [GradPairFixed]) {
        prefix.fill(GradPairFixed::default());
        for i in 0..bins[0].len() {
            for (partial, node) in prefix.iter_mut().zip(bins) {
                *partial += node[i];
            }
            self.consider(i + 1, false, true, prefix.iter().copied());
        }
        if prefix
            .iter()
            .zip(self.parents)
            .all(|(&sum, parent)| sum == parent.sum)
        {
            return;
        }
        prefix.fill(GradPairFixed::default());
        for i in (0..bins[0].len()).rev() {
            for (partial, node) in prefix.iter_mut().zip(bins) {
                *partial += node[i];
            }
            self.consider(i, true, false, prefix.iter().copied());
        }
    }

    fn one_hot(&mut self, bins: &[&[GradPairFixed]]) {
        let missing: Vec<_> = self
            .parents
            .iter()
            .zip(bins)
            .map(|(parent, bins)| {
                let mut observed = GradPairFixed::default();
                for &bin in *bins {
                    observed += bin;
                }
                parent.sum - observed
            })
            .collect();
        for cat in 0..bins[0].len() {
            self.consider(cat, true, false, bins.iter().map(|node| node[cat]));
            self.consider(
                cat,
                false,
                false,
                bins.iter().zip(&missing).map(|(node, &missing)| {
                    let mut right = node[cat];
                    right += missing;
                    right
                }),
            );
        }
    }

    /// Use one category order for the level, based on the pooled statistics.
    /// Score every shared prefix independently on each leaf before reducing.
    fn partition(&mut self, bins: &[&[GradPairFixed]], prefix: &mut [GradPairFixed]) -> Vec<usize> {
        let n = bins[0].len();
        let limit = n.min(self.params.max_cat_threshold as usize);
        if limit < 2 {
            return Vec::new();
        }
        let weights: Vec<_> = (0..n)
            .map(|cat| {
                let mut sum = GradPairFixed::default();
                for node in bins {
                    sum += node[cat];
                }
                let (g, h) = self.quantizer.to_f64(sum);
                xgb_weight(g, h, self.params)
            })
            .collect();
        let mut order: Vec<_> = (0..n).collect();
        order.sort_by(|&a, &b| {
            weights[a]
                .partial_cmp(&weights[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        prefix.fill(GradPairFixed::default());
        for (i, &cat) in order[..limit - 1].iter().enumerate() {
            for (partial, node) in prefix.iter_mut().zip(bins) {
                *partial += node[cat];
            }
            self.consider(i + 1, true, false, prefix.iter().copied());
        }

        prefix.fill(GradPairFixed::default());
        for j in (n - (limit - 1)..n).rev() {
            for (partial, node) in prefix.iter_mut().zip(bins) {
                *partial += node[order[j]];
            }
            self.consider(j, false, true, prefix.iter().copied());
        }
        order
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn best_level_split_in_block(
    histograms: &[Histogram],
    feat_ids: &[FeatId],
    offsets: &[u32],
    nodes: &[(GradPairFixed, NodeBounds)],
    q: &GradQuantizer,
    p: &TrainParams,
    feature_allowed: impl Fn(FeatId) -> bool,
    categorical: Option<&[bool]>,
) -> Option<LevelSplit> {
    debug_assert!(!nodes.is_empty());
    debug_assert_eq!(histograms.len(), nodes.len());
    let parents: Vec<_> = nodes
        .iter()
        .map(|&(sum, bounds)| Parent {
            sum,
            bounds,
            gain: bounded_node_gain(q.to_f64(sum), p, bounds),
        })
        .collect();
    let mut scan = Scan {
        parents: &parents,
        quantizer: q,
        params: p,
        constraint: 0,
        best: ScoredSplit {
            at: 0,
            missing_left: false,
            score: f64::from(MIN_GAIN_EPS) * nodes.len() as f64,
            children: Vec::with_capacity(nodes.len()),
        },
        scratch: Vec::with_capacity(nodes.len()),
    };
    let mut best = None;
    let mut prefix = vec![GradPairFixed::default(); nodes.len()];
    let mut bins = Vec::with_capacity(nodes.len());
    for (local, &feat) in feat_ids.iter().enumerate() {
        if !feature_allowed(feat) {
            continue;
        }
        scan.constraint = p.monotone_of(feat);
        let range = offsets[local] as usize..offsets[local + 1] as usize;
        bins.clear();
        bins.extend(
            histograms
                .iter()
                .map(|histogram| &histogram.bins[range.clone()]),
        );
        let previous = scan.best.score;
        let is_cat = categorical.is_some_and(|flags| flags.get(feat as usize) == Some(&true));
        let one_hot = is_cat && (range.len() as u64) < u64::from(p.max_cat_to_onehot);
        let order = if one_hot {
            scan.one_hot(&bins);
            Vec::new()
        } else if is_cat {
            scan.partition(&bins, &mut prefix)
        } else {
            scan.numeric(&bins, &mut prefix);
            Vec::new()
        };
        if scan.best.score == previous {
            continue;
        }

        let cats = is_cat.then(|| {
            let mut set = CatBits::for_bins(range.len());
            if one_hot {
                set.set(scan.best.at as u32);
            } else {
                for &cat in &order[..scan.best.at] {
                    set.set(cat as u32);
                }
            }
            set
        });
        best = Some(LevelSplit {
            score: scan.best.score,
            candidates: scan
                .best
                .children
                .iter()
                .map(|&(left, right, gain)| SplitCandidate {
                    feat,
                    bin: if is_cat { 0 } else { scan.best.at as u32 },
                    gain,
                    left,
                    right,
                    missing_left: scan.best.missing_left,
                    cats: cats.clone(),
                })
                .collect(),
        });
    }
    best
}
