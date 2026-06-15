use std::io::{self, Read, Write};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// mdBook preprocessor protocol (https://rust-lang.github.io/mdBook/for_developers/preprocessors.html):
//
// 1. When called as `mdbook-wikilinks supports <renderer>`, exit 0 if we
//    support that renderer, non-zero otherwise.
// 2. Otherwise, read [PreprocessorContext, Book] as JSON from stdin, process
//    the Book, and write the modified Book to stdout.
//
// Configuration in book.toml:
//
//   [preprocessor.wikilinks]
//   command = "mdbook-wikilinks"
//   base_path = "decisions"   # optional: path prefix for resolved links

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() >= 3 && args[1] == "supports" {
        let renderer = &args[2];
        std::process::exit(if renderer == "html" { 0 } else { 1 });
    }

    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("failed to read stdin");

    let [ctx, mut book]: [Value; 2] =
        serde_json::from_str(&input).expect("invalid preprocessor input");

    let base_path = ctx
        .pointer("/config/preprocessor/wikilinks/base_path")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim_end_matches('/');

    let pattern = Regex::new(r"\[\[([^\[\]]+?)\]\]").expect("invalid regex");

    process_sections(book["sections"].as_array_mut().unwrap(), &pattern, base_path);

    io::stdout()
        .write_all(
            serde_json::to_string(&book)
                .expect("failed to serialise book")
                .as_bytes(),
        )
        .expect("failed to write stdout");
}

fn process_sections(sections: &mut Vec<Value>, pattern: &Regex, base_path: &str) {
    for section in sections.iter_mut() {
        if let Some(chapter) = section.get_mut("Chapter") {
            rewrite_chapter(chapter, pattern, base_path);
        }
    }
}

fn rewrite_chapter(chapter: &mut Value, pattern: &Regex, base_path: &str) {
    if let Some(content) = chapter["content"].as_str() {
        let rewritten = pattern.replace_all(content, |caps: &regex::Captures| {
            let label = &caps[1];
            let slug = label
                .to_lowercase()
                .replace(' ', "-")
                .replace(|c: char| !c.is_alphanumeric() && c != '-', "");
            if base_path.is_empty() {
                format!("[{label}]({slug}.md)")
            } else {
                format!("[{label}]({base_path}/{slug}.md)")
            }
        });
        chapter["content"] = Value::String(rewritten.into_owned());
    }

    if let Some(sub_items) = chapter["sub_items"].as_array_mut() {
        for item in sub_items.iter_mut() {
            if let Some(sub_chapter) = item.get_mut("Chapter") {
                rewrite_chapter(sub_chapter, pattern, base_path);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;

    fn pattern() -> Regex {
        Regex::new(r"\[\[([^\[\]]+?)\]\]").unwrap()
    }

    #[test]
    fn resolves_simple_wikilink() {
        let p = pattern();
        let result = p.replace_all("See [[ADR 0001]] for details.", |caps: &regex::Captures| {
            let label = &caps[1];
            let slug = label
                .to_lowercase()
                .replace(' ', "-")
                .replace(|c: char| !c.is_alphanumeric() && c != '-', "");
            format!("[{label}](decisions/{slug}.md)")
        });
        assert_eq!(result, "See [ADR 0001](decisions/adr-0001.md) for details.");
    }

    #[test]
    fn resolves_multiple_wikilinks() {
        let p = pattern();
        let input = "See [[ADR 0001]] and [[ADR 0002]].";
        let result = p.replace_all(input, |caps: &regex::Captures| {
            let label = &caps[1];
            let slug = label
                .to_lowercase()
                .replace(' ', "-")
                .replace(|c: char| !c.is_alphanumeric() && c != '-', "");
            format!("[{label}](decisions/{slug}.md)")
        });
        assert_eq!(
            result,
            "See [ADR 0001](decisions/adr-0001.md) and [ADR 0002](decisions/adr-0002.md)."
        );
    }

    #[test]
    fn no_false_positives_on_normal_links() {
        let p = pattern();
        let input = "[normal link](some/path.md) stays unchanged.";
        let result = p.replace_all(input, |caps: &regex::Captures| caps[0].to_owned());
        assert_eq!(result, input);
    }
}
