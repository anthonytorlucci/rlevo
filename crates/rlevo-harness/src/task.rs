//! Task identity: whether two runs faced the same task.
//!
//! A [`TaskFingerprint`] is computed from the config an environment (or a
//! landscape, or any other unit of trials) was built from, with the per-trial
//! seed left out, so every trial of one task carries the same fingerprint and
//! a change to any other config field changes it.
//! [`Suite::with_task`](crate::suite::Suite::with_task) and
//! [`TrialUnit::with_task`](crate::suite::TrialUnit::with_task) compute it,
//! and every trial report of the unit names it.
//!
//! The fingerprint is a hash of a canonical text of the config, computed with
//! a named, stable algorithm and tagged with it ([`TASK_FINGERPRINT_ALGORITHM`]),
//! so a fingerprint taken by one build compares with one taken by another. The
//! canonical text is written here, by walking the config with serde, not by a
//! JSON library whose output depends on the features some other crate in the
//! build turns on. It identifies a task; it is not a defence against a config
//! made to collide.

/// The algorithm every [`TaskFingerprint`] this build computes is tagged
/// with: 64-bit FNV-1a over the canonical text described at
/// [`task_fingerprint`], version 1. A change to that text or to the hash is a
/// new tag.
pub const TASK_FINGERPRINT_ALGORITHM: &str = "fnv1a64-canon-v1";

/// The identity of a task: its config, seed left out.
///
/// Two trials faced the same task when their fingerprints have the same
/// `algorithm` and `hash`. `config` is the canonical text the hash covers,
/// kept so a reader can see what the task was.
///
/// `#[non_exhaustive]`: built by [`task_fingerprint`], so a later field is not
/// a breaking change.
#[cfg_attr(feature = "json", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct TaskFingerprint {
    /// The algorithm the hash was computed with:
    /// [`TASK_FINGERPRINT_ALGORITHM`] for one this build computed.
    pub algorithm: String,
    /// The hash, 16 lowercase hexadecimal digits.
    pub hash: String,
    /// The canonical text of the config the hash covers, with its top-level
    /// `seed` field left out.
    pub config: String,
}

/// Why a config could not be fingerprinted.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TaskFingerprintError {
    /// The config's `Serialize` implementation returned an error.
    #[error("the task config failed to serialize: {0}")]
    Serialize(String),
}

#[cfg(feature = "json")]
impl serde::ser::Error for TaskFingerprintError {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        Self::Serialize(msg.to_string())
    }
}

/// Fingerprints the task `config` describes.
///
/// The config is walked with serde and written as a canonical text:
///
/// - **Structure.** JSON-like: `null`, `true`/`false`, `[..]`, `{..}`, with
///   no whitespace. A struct's or map's entries are sorted by key at every
///   depth, so field order and a map's iteration order do not count. An enum
///   variant is `"Variant"`, or `{"Variant":..}` when it carries data. An
///   `Option` is `null` or its value.
/// - **The top-level type's name** prefixes the text (`FrozenLakeConfig{..}`),
///   so two environments whose configs happen to have the same fields are
///   different tasks; renaming the config type changes the fingerprint.
/// - **The seed.** A top-level field named `seed` is left out: the harness
///   gives every trial its own seed, so a fingerprint that covered it would
///   make every trial a different task. Every other field counts, a nested
///   `seed` included.
/// - **Numbers.** An integer is written in decimal. A float is widened to
///   `f64` (an `f32` exactly, so `0.1_f32` is `0.10000000149011612`) and
///   written as Rust's `{:?}` writes it (`1.0`, `-0.0`, `1e-7`), so `-0.0`
///   and `0.0` differ. Non-finite values are `Infinity`, `-Infinity` and
///   `NaN`, distinct from each other and from `null`.
/// - **Strings** are quoted, with `"`, `\` and control characters escaped.
///
/// The hash is 64-bit FNV-1a over that text. The id the caller registers the
/// unit under is not part of it: two suites that name the same environment
/// and config differently faced the same task.
///
/// # Errors
///
/// [`TaskFingerprintError::Serialize`] when the config's `Serialize`
/// implementation returns an error.
///
/// # Examples
///
/// ```
/// use rlevo_harness::task::{TASK_FINGERPRINT_ALGORITHM, task_fingerprint};
///
/// #[derive(serde::Serialize)]
/// struct LakeConfig { size: u32, slippery: bool, seed: u64 }
///
/// let a = task_fingerprint(&LakeConfig { size: 4, slippery: true, seed: 1 }).unwrap();
/// let b = task_fingerprint(&LakeConfig { size: 4, slippery: true, seed: 2 }).unwrap();
/// let c = task_fingerprint(&LakeConfig { size: 4, slippery: false, seed: 1 }).unwrap();
/// assert_eq!(a.hash, b.hash, "the seed is not part of the task");
/// assert_ne!(a.hash, c.hash);
/// assert_eq!(a.algorithm, TASK_FINGERPRINT_ALGORITHM);
/// assert_eq!(a.config, r#"LakeConfig{"size":4,"slippery":true}"#);
/// ```
#[cfg(feature = "json")]
pub fn task_fingerprint<C>(config: &C) -> Result<TaskFingerprint, TaskFingerprintError>
where
    C: serde::Serialize + ?Sized,
{
    let tree = config.serialize(canon::Serializer)?;
    let text = canon::top_level_text(tree);
    Ok(TaskFingerprint {
        algorithm: TASK_FINGERPRINT_ALGORITHM.to_owned(),
        hash: format!("{:016x}", fnv1a64(text.as_bytes())),
        config: text,
    })
}

/// 64-bit FNV-1a.
#[cfg(feature = "json")]
fn fnv1a64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET_BASIS, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(PRIME)
    })
}

/// The canonical form: a serde `Serializer` that builds a tree, and the
/// writer that prints it.
#[cfg(feature = "json")]
mod canon {
    use std::fmt::Write as _;

    use serde::ser::{self, Serialize};

    use super::TaskFingerprintError as Error;

    /// A serialized value, before it is printed.
    pub(super) enum Canon {
        Null,
        Bool(bool),
        Int(i128),
        UInt(u128),
        Float(f64),
        Str(String),
        Seq(Vec<Canon>),
        Map(Vec<(Canon, Canon)>),
        /// A struct: its type name (printed only at the top level) and fields.
        Struct(&'static str, Vec<(&'static str, Canon)>),
        /// An enum variant carrying data.
        Variant(&'static str, Box<Canon>),
    }

    /// Prints `tree` as the top-level text: the type's name, if it is a
    /// struct, then its body, with a top-level `seed` left out.
    pub(super) fn top_level_text(tree: Canon) -> String {
        let mut out = String::new();
        match tree {
            Canon::Struct(name, fields) => {
                out.push_str(name);
                let fields = fields.into_iter().filter(|(k, _)| *k != "seed").collect();
                write(&Canon::Struct(name, fields), &mut out);
            }
            Canon::Map(entries) => {
                let entries = entries
                    .into_iter()
                    .filter(|(k, _)| !matches!(k, Canon::Str(s) if s == "seed"))
                    .collect();
                write(&Canon::Map(entries), &mut out);
            }
            other => write(&other, &mut out),
        }
        out
    }

    fn write(tree: &Canon, out: &mut String) {
        match tree {
            Canon::Null => out.push_str("null"),
            Canon::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Canon::Int(i) => {
                let _ = write!(out, "{i}");
            }
            Canon::UInt(u) => {
                let _ = write!(out, "{u}");
            }
            Canon::Float(f) => write_float(*f, out),
            Canon::Str(s) => write_str(s, out),
            Canon::Seq(items) => {
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write(item, out);
                }
                out.push(']');
            }
            Canon::Map(entries) => {
                let printed = entries.iter().map(|(k, v)| (text(k), text(v))).collect();
                write_sorted(printed, out);
            }
            Canon::Struct(_, fields) => {
                let printed = fields
                    .iter()
                    .map(|(k, v)| (text(&Canon::Str((*k).to_owned())), text(v)))
                    .collect();
                write_sorted(printed, out);
            }
            Canon::Variant(name, inner) => {
                out.push('{');
                write_str(name, out);
                out.push(':');
                write(inner, out);
                out.push('}');
            }
        }
    }

    fn text(tree: &Canon) -> String {
        let mut out = String::new();
        write(tree, &mut out);
        out
    }

    /// `{k:v,..}` with the entries in byte order of their key text.
    fn write_sorted(mut entries: Vec<(String, String)>, out: &mut String) {
        entries.sort();
        out.push('{');
        for (i, (k, v)) in entries.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(k);
            out.push(':');
            out.push_str(v);
        }
        out.push('}');
    }

    fn write_float(f: f64, out: &mut String) {
        if f.is_nan() {
            out.push_str("NaN");
        } else if f.is_infinite() {
            out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
        } else {
            let _ = write!(out, "{f:?}");
        }
    }

    fn write_str(s: &str, out: &mut String) {
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                c if u32::from(c) < 0x20 => {
                    let _ = write!(out, "\\u{:04x}", u32::from(c));
                }
                c => out.push(c),
            }
        }
        out.push('"');
    }

    /// Builds a [`Canon`] from any `Serialize` value.
    pub(super) struct Serializer;

    impl ser::Serializer for Serializer {
        type Ok = Canon;
        type Error = Error;
        type SerializeSeq = Seq;
        type SerializeTuple = Seq;
        type SerializeTupleStruct = Seq;
        type SerializeTupleVariant = TupleVariant;
        type SerializeMap = Map;
        type SerializeStruct = Struct;
        type SerializeStructVariant = StructVariant;

        fn serialize_bool(self, v: bool) -> Result<Canon, Error> {
            Ok(Canon::Bool(v))
        }
        fn serialize_i8(self, v: i8) -> Result<Canon, Error> {
            Ok(Canon::Int(v.into()))
        }
        fn serialize_i16(self, v: i16) -> Result<Canon, Error> {
            Ok(Canon::Int(v.into()))
        }
        fn serialize_i32(self, v: i32) -> Result<Canon, Error> {
            Ok(Canon::Int(v.into()))
        }
        fn serialize_i64(self, v: i64) -> Result<Canon, Error> {
            Ok(Canon::Int(v.into()))
        }
        fn serialize_i128(self, v: i128) -> Result<Canon, Error> {
            Ok(Canon::Int(v))
        }
        fn serialize_u8(self, v: u8) -> Result<Canon, Error> {
            Ok(Canon::UInt(v.into()))
        }
        fn serialize_u16(self, v: u16) -> Result<Canon, Error> {
            Ok(Canon::UInt(v.into()))
        }
        fn serialize_u32(self, v: u32) -> Result<Canon, Error> {
            Ok(Canon::UInt(v.into()))
        }
        fn serialize_u64(self, v: u64) -> Result<Canon, Error> {
            Ok(Canon::UInt(v.into()))
        }
        fn serialize_u128(self, v: u128) -> Result<Canon, Error> {
            Ok(Canon::UInt(v))
        }
        fn serialize_f32(self, v: f32) -> Result<Canon, Error> {
            Ok(Canon::Float(v.into()))
        }
        fn serialize_f64(self, v: f64) -> Result<Canon, Error> {
            Ok(Canon::Float(v))
        }
        fn serialize_char(self, v: char) -> Result<Canon, Error> {
            Ok(Canon::Str(v.to_string()))
        }
        fn serialize_str(self, v: &str) -> Result<Canon, Error> {
            Ok(Canon::Str(v.to_owned()))
        }
        fn serialize_bytes(self, v: &[u8]) -> Result<Canon, Error> {
            Ok(Canon::Seq(
                v.iter().map(|b| Canon::UInt((*b).into())).collect(),
            ))
        }
        fn serialize_none(self) -> Result<Canon, Error> {
            Ok(Canon::Null)
        }
        fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<Canon, Error> {
            value.serialize(self)
        }
        fn serialize_unit(self) -> Result<Canon, Error> {
            Ok(Canon::Null)
        }
        fn serialize_unit_struct(self, _name: &'static str) -> Result<Canon, Error> {
            Ok(Canon::Null)
        }
        fn serialize_unit_variant(
            self,
            _name: &'static str,
            _index: u32,
            variant: &'static str,
        ) -> Result<Canon, Error> {
            Ok(Canon::Str(variant.to_owned()))
        }
        fn serialize_newtype_struct<T: ?Sized + Serialize>(
            self,
            _name: &'static str,
            value: &T,
        ) -> Result<Canon, Error> {
            value.serialize(self)
        }
        fn serialize_newtype_variant<T: ?Sized + Serialize>(
            self,
            _name: &'static str,
            _index: u32,
            variant: &'static str,
            value: &T,
        ) -> Result<Canon, Error> {
            Ok(Canon::Variant(variant, Box::new(value.serialize(self)?)))
        }
        fn serialize_seq(self, len: Option<usize>) -> Result<Seq, Error> {
            Ok(Seq(Vec::with_capacity(len.unwrap_or(0))))
        }
        fn serialize_tuple(self, len: usize) -> Result<Seq, Error> {
            Ok(Seq(Vec::with_capacity(len)))
        }
        fn serialize_tuple_struct(self, _name: &'static str, len: usize) -> Result<Seq, Error> {
            Ok(Seq(Vec::with_capacity(len)))
        }
        fn serialize_tuple_variant(
            self,
            _name: &'static str,
            _index: u32,
            variant: &'static str,
            len: usize,
        ) -> Result<TupleVariant, Error> {
            Ok(TupleVariant(variant, Vec::with_capacity(len)))
        }
        fn serialize_map(self, len: Option<usize>) -> Result<Map, Error> {
            Ok(Map {
                entries: Vec::with_capacity(len.unwrap_or(0)),
                key: None,
            })
        }
        fn serialize_struct(self, name: &'static str, len: usize) -> Result<Struct, Error> {
            Ok(Struct(name, Vec::with_capacity(len)))
        }
        fn serialize_struct_variant(
            self,
            _name: &'static str,
            _index: u32,
            variant: &'static str,
            len: usize,
        ) -> Result<StructVariant, Error> {
            Ok(StructVariant(variant, Vec::with_capacity(len)))
        }
    }

    pub(super) struct Seq(Vec<Canon>);

    impl ser::SerializeSeq for Seq {
        type Ok = Canon;
        type Error = Error;
        fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
            self.0.push(value.serialize(Serializer)?);
            Ok(())
        }
        fn end(self) -> Result<Canon, Error> {
            Ok(Canon::Seq(self.0))
        }
    }

    impl ser::SerializeTuple for Seq {
        type Ok = Canon;
        type Error = Error;
        fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
            ser::SerializeSeq::serialize_element(self, value)
        }
        fn end(self) -> Result<Canon, Error> {
            ser::SerializeSeq::end(self)
        }
    }

    impl ser::SerializeTupleStruct for Seq {
        type Ok = Canon;
        type Error = Error;
        fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
            ser::SerializeSeq::serialize_element(self, value)
        }
        fn end(self) -> Result<Canon, Error> {
            ser::SerializeSeq::end(self)
        }
    }

    pub(super) struct TupleVariant(&'static str, Vec<Canon>);

    impl ser::SerializeTupleVariant for TupleVariant {
        type Ok = Canon;
        type Error = Error;
        fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
            self.1.push(value.serialize(Serializer)?);
            Ok(())
        }
        fn end(self) -> Result<Canon, Error> {
            Ok(Canon::Variant(self.0, Box::new(Canon::Seq(self.1))))
        }
    }

    pub(super) struct Map {
        entries: Vec<(Canon, Canon)>,
        key: Option<Canon>,
    }

    impl ser::SerializeMap for Map {
        type Ok = Canon;
        type Error = Error;
        fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), Error> {
            self.key = Some(key.serialize(Serializer)?);
            Ok(())
        }
        fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), Error> {
            let key = self
                .key
                .take()
                .ok_or_else(|| <Error as ser::Error>::custom("a map value came before its key"))?;
            self.entries.push((key, value.serialize(Serializer)?));
            Ok(())
        }
        fn end(self) -> Result<Canon, Error> {
            Ok(Canon::Map(self.entries))
        }
    }

    pub(super) struct Struct(&'static str, Vec<(&'static str, Canon)>);

    impl ser::SerializeStruct for Struct {
        type Ok = Canon;
        type Error = Error;
        fn serialize_field<T: ?Sized + Serialize>(
            &mut self,
            key: &'static str,
            value: &T,
        ) -> Result<(), Error> {
            self.1.push((key, value.serialize(Serializer)?));
            Ok(())
        }
        fn end(self) -> Result<Canon, Error> {
            Ok(Canon::Struct(self.0, self.1))
        }
    }

    pub(super) struct StructVariant(&'static str, Vec<(&'static str, Canon)>);

    impl ser::SerializeStructVariant for StructVariant {
        type Ok = Canon;
        type Error = Error;
        fn serialize_field<T: ?Sized + Serialize>(
            &mut self,
            key: &'static str,
            value: &T,
        ) -> Result<(), Error> {
            self.1.push((key, value.serialize(Serializer)?));
            Ok(())
        }
        fn end(self) -> Result<Canon, Error> {
            Ok(Canon::Variant(self.0, Box::new(Canon::Struct("", self.1))))
        }
    }
}

#[cfg(all(test, feature = "json"))]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use super::{TASK_FINGERPRINT_ALGORITHM, fnv1a64, task_fingerprint};

    #[derive(serde::Serialize)]
    struct Lake {
        seed: u64,
        rows: u32,
        slippery: bool,
        success_rate: f32,
        map: Option<Vec<String>>,
        nested: Inner,
        holes: Holes,
    }

    #[derive(serde::Serialize)]
    struct Inner {
        seed: u64,
        frozen_prob: f64,
    }

    #[derive(serde::Serialize)]
    enum Holes {
        None,
        Fixed(u32),
        Random { prob: f32 },
    }

    fn lake() -> Lake {
        Lake {
            seed: 7,
            rows: 4,
            slippery: true,
            success_rate: 1.0 / 3.0,
            map: None,
            nested: Inner {
                seed: 1,
                frozen_prob: 0.8,
            },
            holes: Holes::None,
        }
    }

    /// FNV-1a 64's published test vectors.
    #[test]
    fn fnv1a64_matches_the_reference_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    /// The whole canonical text of one config, pinned, and its hash, computed
    /// independently of this crate (a three-line FNV-1a in Python over the same
    /// bytes). A change to either is a new algorithm tag.
    #[test]
    fn the_canonical_text_and_hash_are_pinned() {
        let f = task_fingerprint(&lake()).unwrap();
        assert_eq!(f.algorithm, TASK_FINGERPRINT_ALGORITHM);
        assert_eq!(
            f.config,
            r#"Lake{"holes":"None","map":null,"nested":{"frozen_prob":0.8,"seed":1},"rows":4,"slippery":true,"success_rate":0.3333333432674408}"#
        );
        assert_eq!(f.hash, "726bc40e9ee538b7");
    }

    #[test]
    fn the_top_level_seed_is_left_out_and_nothing_else() {
        let a = task_fingerprint(&lake()).unwrap();
        let b = task_fingerprint(&Lake { seed: 99, ..lake() }).unwrap();
        assert_eq!(a, b, "a different seed is the same task");
        assert!(
            a.config.contains("\"seed\":1"),
            "a nested `seed` is a config field like any other: {}",
            a.config
        );
    }

    /// Every non-seed field, at any depth, changes the fingerprint, and so
    /// does the type's name.
    #[test]
    fn every_other_field_and_the_type_name_count() {
        #[derive(serde::Serialize)]
        struct Other {
            rows: u32,
        }
        #[derive(serde::Serialize)]
        struct Lake2 {
            rows: u32,
        }
        let base = task_fingerprint(&lake()).unwrap().hash;
        let changed = [
            task_fingerprint(&Lake { rows: 8, ..lake() }),
            task_fingerprint(&Lake {
                slippery: false,
                ..lake()
            }),
            task_fingerprint(&Lake {
                success_rate: 0.5,
                ..lake()
            }),
            task_fingerprint(&Lake {
                map: Some(vec!["SFFG".into()]),
                ..lake()
            }),
            task_fingerprint(&Lake {
                nested: Inner {
                    seed: 1,
                    frozen_prob: 0.9,
                },
                ..lake()
            }),
            task_fingerprint(&Lake {
                nested: Inner {
                    seed: 2,
                    frozen_prob: 0.8,
                },
                ..lake()
            }),
            task_fingerprint(&Lake {
                holes: Holes::Fixed(3),
                ..lake()
            }),
            task_fingerprint(&Lake {
                holes: Holes::Random { prob: 0.1 },
                ..lake()
            }),
        ];
        for (i, fingerprint) in changed.into_iter().enumerate() {
            assert_ne!(fingerprint.unwrap().hash, base, "change {i} kept the hash");
        }
        assert_ne!(
            task_fingerprint(&Other { rows: 4 }).unwrap().hash,
            task_fingerprint(&Lake2 { rows: 4 }).unwrap().hash,
            "two config types with the same fields are different tasks"
        );
    }

    /// Non-finite floats are written distinctly from each other and from
    /// `None`, and `-0.0` from `0.0`.
    #[test]
    fn special_floats_are_distinct() {
        #[derive(serde::Serialize)]
        struct C {
            x: Option<f32>,
        }
        let texts: Vec<String> = [
            None,
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
            Some(f32::NAN),
            Some(0.0),
            Some(-0.0),
        ]
        .into_iter()
        .map(|x| task_fingerprint(&C { x }).unwrap().config)
        .collect();
        assert_eq!(
            texts,
            [
                r#"C{"x":null}"#,
                r#"C{"x":Infinity}"#,
                r#"C{"x":-Infinity}"#,
                r#"C{"x":NaN}"#,
                r#"C{"x":0.0}"#,
                r#"C{"x":-0.0}"#,
            ]
        );
    }

    /// A map's entries are sorted, so a `HashMap`'s iteration order does not
    /// count, and keys and strings are escaped.
    #[test]
    fn maps_are_sorted_and_strings_escaped() {
        let mut hashed = HashMap::new();
        let mut ordered = BTreeMap::new();
        for (k, v) in [("b", 2), ("a", 1), ("c\"\n", 3)] {
            hashed.insert(k, v);
            ordered.insert(k, v);
        }
        let h = task_fingerprint(&hashed).unwrap();
        assert_eq!(h, task_fingerprint(&ordered).unwrap());
        assert_eq!(h.config, r#"{"a":1,"b":2,"c\"\u000a":3}"#);
    }

    #[test]
    fn a_serialize_error_is_reported() {
        struct Fails;
        impl serde::Serialize for Fails {
            fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("refused"))
            }
        }
        let err = task_fingerprint(&Fails).unwrap_err();
        assert!(err.to_string().contains("refused"), "{err}");
    }
}
