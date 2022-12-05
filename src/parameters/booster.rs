use std::default::Default;

use super::{linear, tree};

/// Type of booster to use when training a [Booster](../struct.Booster.html) model.
#[derive(Clone)]
pub enum BoosterType {
    /// Use a tree booster with given parameters when training.
    ///
    /// Construct parameters using
    /// [TreeBoosterParametersBuilder](tree/struct.TreeBoosterParametersBuilder.html).
    Tree(tree::TreeBoosterParameters),

    /// Use a linear booster with given parameters when training.
    ///
    /// Construct parameters using
    /// [LinearBoosterParametersBuilder](linear/struct.LinearBoosterParametersBuilder.html).
    Linear(linear::LinearBoosterParameters),
}

impl Default for BoosterType {
    fn default() -> Self {
        BoosterType::Tree(tree::TreeBoosterParameters::default())
    }
}

impl BoosterType {
    pub(crate) fn as_string_pairs(&self) -> Vec<(String, String)> {
        match *self {
            BoosterType::Tree(ref p) => p.as_string_pairs(),
            BoosterType::Linear(ref p) => p.as_string_pairs(),
        }
    }
}
