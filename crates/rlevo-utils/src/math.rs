// number of combinations
// usage: combinations(54, 6) // returns 25,827,165
pub fn combinations(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let mut result = 1;
    for i in 1..=k {
        result = result * (n - i + 1) / i;
    }
    result
}
