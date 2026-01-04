use std::fmt;

use crate::base::burnrl_model::RLModel;
use crate::base::ElemType;
use burn::module::AutodiffModule;
use burn::tensor::backend::{AutodiffBackend, Backend};

pub trait DQNModel<B: AutodiffBackend>: RLModel<B> + fmt::Debug + AutodiffModule<B> {
    fn soft_update(this: Self, that: &Self, tau: ElemType) -> Self;
}
