#![feature(adt_const_params)]
#![feature(linked_list_retain)]
#![feature(linked_list_cursors)]

// mod nd_tree_old;
mod nd_tree;
mod nd_tree_pareto_front;
use nd_tree::Solution;
use nd_tree_pareto_front::NdTreeParetoFront;
use pareto::ParetoFront;

fn main() {
    // fix D=3 objectives, leaf cap N=4, exact children C=2
    const D: usize = 2;
    const N: usize = 4;
    const C: usize = 2;

    let mut tree = NdTreeParetoFront::<N, D, C>::new("test");

    let samples3d = vec![
        Solution::new([10, 20, 30]),
        Solution::new([9, 25, 28]),
        Solution::new([8, 22, 35]),
        Solution::new([12, 18, 33]),
        Solution::new([7, 30, 25]),
        Solution::new([11, 19, 31]),
        Solution::new([6, 28, 27]),
        Solution::new([13, 21, 29]),
        Solution::new([5, 26, 32]),
        Solution::new([14, 24, 34]),
        Solution::new([4, 23, 30]),
        Solution::new([15, 17, 36]),
        Solution::new([3, 27, 26]),
        Solution::new([16, 20, 33]),
        Solution::new([2, 25, 28]),
    ];

    let samples2d = vec![
        Solution::new([10, 20]),
        Solution::new([9, 25]),
        Solution::new([8, 22]),
        Solution::new([12, 18]),
        Solution::new([7, 30]),
        Solution::new([11, 19]),
        Solution::new([6, 28]),
        Solution::new([13, 21]),
        Solution::new([5, 26]),
        Solution::new([14, 24]),
        Solution::new([4, 23]),
        Solution::new([15, 17]),
        Solution::new([3, 27]),
        Solution::new([16, 20]),
        Solution::new([2, 25]),
    ];

    for sol in samples2d {
        tree.try_insert(&sol);
    }

    println!("Tree:");
    print!("{:?}", tree);
}
