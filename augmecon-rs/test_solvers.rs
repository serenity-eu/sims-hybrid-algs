// Test script to check available solvers in good_lp
use good_lp::*;

fn main() {
    println!("Testing good_lp solvers...");
    
    // Test what solvers are available
    let vars = variables!();
    let x = vars.add(variable().min(0).max(10));
    
    // Try to see what's available
    let problem = vars.minimise(x).using(default_solver);
    println!("Default solver works");
    
    // Try coin_cbc
    let problem_cbc = vars.minimise(x).using(coin_cbc);
    println!("Coin CBC works");
    
    // Let's see if there are other solvers available
    // Common good_lp solvers include: highs, glpk, coin_cbc, etc.
}
