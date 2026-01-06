use std::fmt;

use burn::module::AutodiffModule;
use burn::tensor::backend::AutodiffBackend;
use evorl_core::model::DrlModel;

pub trait DQNModel<B: AutodiffBackend, const R: usize>:
    DrlModel<B, R> + fmt::Debug + AutodiffModule<B>
{
    fn soft_update(this: Self, that: &Self, tau: f64) -> Self;
}
