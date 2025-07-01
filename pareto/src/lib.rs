#![feature(adt_const_params)]
mod pareto_front;

pub use pareto_front::{
    Dominance, HasObjectives, MoSolution, Objectives, ParetoFront, Random, RandomCollection, Sense,
};
