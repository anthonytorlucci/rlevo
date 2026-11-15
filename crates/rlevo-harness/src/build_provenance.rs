//! The rules the build script uses to find the commit the harness is built
//! from. Included by `build.rs` with `#[path]`, and compiled into the library
//! only for its tests, so the rules are tested where they are written.
//!
//! The harness's build script runs in the crate's own directory. In rlevo's
//! checkout that directory is `crates/rlevo-harness/`. A packaged copy (from
//! crates.io, or a probe's stage) sits under a cargo registry or a `vendor/`
//! directory, which an unrelated repository may enclose, a home directory
//! kept under git for instance. Asking git there would record that
//! repository's commit. So:
//!
//! 1. a packaged crate has `.cargo_vcs_info.json`, written by `cargo package`
//!    with the commit it was packaged from: use it, and run no git;
//! 2. otherwise run git only when the crate directory is rlevo's own
//!    checkout, which `git rev-parse --show-prefix` reports as
//!    `crates/rlevo-harness/`;
//! 3. otherwise record nothing.

/// Where the harness sits in rlevo's own repository.
pub const OWN_PREFIX: &str = "crates/rlevo-harness/";

/// The commit and dirty flag `cargo package` recorded in
/// `.cargo_vcs_info.json`, or `None` when the file names no commit.
///
/// The file is JSON written by cargo: `{"git": {"sha1": "<hex>", "dirty":
/// true}, "path_in_vcs": "..."}`, with `dirty` present only when the tree was
/// dirty. Read without a JSON parser, since a build script has no
/// dependencies here: the first `"sha1"` member's string value, which must be
/// hexadecimal, and whether a `"dirty": true` member appears.
pub fn parse_vcs_info(text: &str) -> Option<(String, bool)> {
    let after_key = &text[text.find("\"sha1\"")? + "\"sha1\"".len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?.trim_start();
    let value = after_colon.strip_prefix('"')?;
    let sha = &value[..value.find('"')?];
    if sha.is_empty() || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    Some((sha.to_owned(), compact.contains("\"dirty\":true")))
}

/// Whether `git rev-parse --show-prefix`, run in the crate's directory,
/// says the crate is rlevo's own checkout.
pub fn is_own_checkout(show_prefix: &str) -> bool {
    show_prefix.trim() == OWN_PREFIX
}

#[cfg(test)]
mod tests {
    use super::{is_own_checkout, parse_vcs_info};

    #[test]
    fn a_packaged_crate_reports_its_commit() {
        let clean = r#"{
  "git": {
    "sha1": "804f473c0123456789abcdef0123456789abcdef"
  },
  "path_in_vcs": "crates/rlevo-harness"
}"#;
        assert_eq!(
            parse_vcs_info(clean),
            Some(("804f473c0123456789abcdef0123456789abcdef".to_owned(), false))
        );
        let dirty =
            r#"{"git":{"sha1":"abc123","dirty":true},"path_in_vcs":"crates/rlevo-harness"}"#;
        assert_eq!(parse_vcs_info(dirty), Some(("abc123".to_owned(), true)));
    }

    #[test]
    fn a_file_with_no_usable_commit_reports_none() {
        for text in [
            "",
            "{}",
            r#"{"git":{"sha1":""}}"#,
            r#"{"git":{"sha1":"not hex"}}"#,
            r#"{"git":{"sha1":12}}"#,
        ] {
            assert_eq!(parse_vcs_info(text), None, "{text}");
        }
    }

    /// Only rlevo's own checkout counts: a repository enclosing a registry or
    /// a vendored copy reports some other prefix, or none.
    #[test]
    fn only_rlevos_own_checkout_runs_git() {
        assert!(is_own_checkout("crates/rlevo-harness/\n"));
        for other in [
            "",
            "\n",
            ".cargo/registry/src/index.crates.io-1949cf8c6b5b557f/rlevo-harness-0.7.0/\n",
            "vendor/rlevo-harness/\n",
            "crates/rlevo-harness/src/\n",
        ] {
            assert!(!is_own_checkout(other), "{other:?}");
        }
    }
}
