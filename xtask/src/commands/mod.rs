pub mod test_fast;
pub mod test_heavy;

/// Print `labels` as a JSON array on stdout, for a GitHub Actions matrix built
/// with `fromJSON`.
///
/// Every label a tier produces is a package or test-target name, so no JSON
/// string needs escaping. xtask's log goes to stderr, leaving stdout clean for
/// `$(cargo xtask <tier> --list)`.
pub fn print_json_list(labels: impl IntoIterator<Item = String>) {
    let quoted: Vec<String> = labels
        .into_iter()
        .map(|label| format!("\"{label}\""))
        .collect();
    println!("[{}]", quoted.join(","));
}
