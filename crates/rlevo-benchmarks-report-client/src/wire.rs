//! Bincode-compatible mirror of [`rlevo_benchmarks::record::schema`].
//!
//! The native record schema is gated behind `feature = "record"` in
//! `rlevo-benchmarks`, which itself doesn't compile to `wasm32` because
//! `rand`/`getrandom` need the `js` feature there. Rather than restructure
//! the workspace, the client redeclares the wire types here. The
//! cross-crate compatibility test in `rlevo-benchmarks` round-trips a
//! populated `EpisodeRecord` between the two representations on every
//! `cargo test` to guarantee they stay byte-identical.
//!
//! **Sync contract:** these struct/enum definitions MUST stay in
//! lockstep with `rlevo-benchmarks/src/record/schema.rs`. Field order,
//! field types, and `#[non_exhaustive]` markers all participate in the
//! bincode wire format. Bumping `FORMAT_VERSION` is the canonical
//! escape hatch when the contract changes.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---- Mirror of rlevo_core::render::styled (kept thin; the cross-crate
// compat test in rlevo-benchmarks guards every field). -----------------

/// Wire mirror of `rlevo_core::render::StyledFrame` — one complete rendered frame.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyledFrame {
    /// Ordered rows of the rendered frame, top to bottom.
    pub lines: Vec<StyledLine>,
}

/// Wire mirror of `rlevo_core::render::StyledLine` — one row of styled spans.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyledLine {
    /// Consecutive styled text segments that make up the row.
    pub spans: Vec<StyledSpan>,
}

/// Wire mirror of `rlevo_core::render::StyledSpan` — a run of identically-styled text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StyledSpan {
    /// The rendered text content of this span.
    pub text: String,
    /// Visual style (foreground, background, modifier bits) to apply.
    pub style: SpanStyle,
}

/// Wire mirror of `rlevo_core::render::SpanStyle` — the visual attributes for one span.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanStyle {
    /// Optional foreground colour; `None` means inherit the terminal default.
    pub fg: Option<Color>,
    /// Optional background colour; `None` means inherit the terminal default.
    pub bg: Option<Color>,
    /// Bit-packed text modifier flags (bold, dim, italic, underlined, reversed).
    pub modifier: Modifier,
}

/// Wire mirror of `rlevo_core::render::Color` — the named terminal palette plus an indexed escape.
///
/// `Reset` and `Indexed(_)` map to no CSS class in the web renderer (see
/// [`crate::styled::color_class`]).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Color {
    Reset,
    Black,
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    Gray,
    DarkGray,
    LightRed,
    LightGreen,
    LightYellow,
    LightBlue,
    LightMagenta,
    LightCyan,
    White,
    /// 256-colour indexed terminal colour; rendered unstyled in the web report.
    Indexed(u8),
}

/// Wire mirror of `rlevo_core::render::Modifier` — bit-packed text decoration flags.
///
/// Bit layout: BOLD=1, DIM=2, ITALIC=4, UNDERLINED=8, REVERSED=16.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Modifier(pub u8);

// ---- /styled mirror ---------------------------------------------------

// ---- Mirror of rlevo_core::render::payload (rich per-family payloads). --

/// A 2-D point in world coordinates, shared by all rich family payloads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

impl Point2 {
    /// Creates a new [`Point2`] with the given coordinates.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Semantic role of a [`RigidBody2D`] within a Box2D scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BodyKind {
    /// Main agent body (e.g. lander hull, walker torso).
    Hull,
    /// Circular wheel attached to a vehicle or walker.
    Wheel,
    /// Limb segment (bipedal walker leg).
    Leg,
    /// Wing surface (not yet used; reserved for future flying agents).
    Wing,
    /// Static terrain or floor polygon.
    Ground,
    /// Landing pad or target region.
    Goal,
    /// Any body that doesn't fit a named category.
    Other,
}

/// Wire mirror of a single rigid body in a Box2D scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RigidBody2D {
    /// Local-frame polygon vertices, ordered for rendering.
    pub vertices: Vec<Point2>,
    /// World-space centre of the body at this frame.
    pub position: Point2,
    /// Counter-clockwise rotation from the body's rest orientation, in radians.
    pub rotation_rad: f32,
    /// Semantic role used to select the CSS class and legend entry.
    pub kind: BodyKind,
}

/// Per-frame snapshot for a 2-D landscape-search environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Landscape2DPayload {
    /// `(lo, hi)` search domain bounds along the x-axis.
    pub bounds_x: (f32, f32),
    /// `(lo, hi)` search domain bounds along the y-axis.
    pub bounds_y: (f32, f32),
    /// Position of the current candidate solution.
    pub current: Point2,
    /// Best candidate found so far, if available.
    pub best: Option<Point2>,
    /// Recent candidate positions, oldest first (used for the trail polyline).
    pub trail: Vec<Point2>,
    /// Human-readable landscape name (`"sphere"`, `"ackley"`, `"rastrigin"`, …).
    pub label: String,
}

/// Per-frame snapshot for a Box2D physics environment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Box2dPayload {
    /// Axis-aligned bounding box of the visible world, as `(min, max)`.
    pub world_bounds: (Point2, Point2),
    /// All rigid bodies in the scene at this frame.
    pub bodies: Vec<RigidBody2D>,
    /// Active contact points between bodies (rendered as small open rings).
    pub contacts: Vec<Point2>,
}

/// Per-frame snapshot for a 2-D locomotion environment (sagittal-plane view).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Locomotion2DPayload {
    /// World-space positions of every skeleton joint.
    pub joints: Vec<Point2>,
    /// Pairs of joint indices defining bones; index into [`joints`](Self::joints).
    pub bones: Vec<(u32, u32)>,
    /// World-space y-coordinate of the ground plane.
    pub ground_y: f32,
    /// Centre-of-mass position, if computed and available.
    pub com: Option<Point2>,
    /// Active contact points with the ground (rendered as open rings).
    pub contacts: Vec<Point2>,
}

/// Cardinal facing of the grid agent (mirror of `rlevo-core` `GridDir`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GridDir {
    East,
    South,
    West,
    North,
}

/// The six Minigrid colours (mirror of `rlevo-core` `GridColor`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GridColor {
    Red,
    Green,
    Blue,
    Purple,
    Yellow,
    Grey,
}

/// Door state (mirror of `rlevo-core` `GridDoorState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GridDoorState {
    Open,
    Closed,
    Locked,
}

/// One grid cell's contents (mirror of `rlevo-core` `GridTile`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GridTile {
    Empty,
    Floor,
    Wall,
    Goal,
    Lava,
    Door(GridColor, GridDoorState),
    Key(GridColor),
    Ball(GridColor),
    Box(GridColor),
}

/// Agent marker: cell position, facing, carried item (mirror of
/// `rlevo-core` `GridAgentMarker`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridAgentMarker {
    pub x: u16,
    pub y: u16,
    pub dir: GridDir,
    pub carrying: Option<GridTile>,
}

/// Structured tile grid for `grids` envs (mirror of
/// `rlevo-benchmarks::record::GridPayload`). `tiles` is row-major,
/// `len == width * height`; cell `(x, y)` is `tiles[y * width + x]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GridPayload {
    pub width: u16,
    pub height: u16,
    pub tiles: Vec<GridTile>,
    pub agent: GridAgentMarker,
}

/// Background class of a tabular grid cell (mirror of `rlevo-core`
/// `TabularCell`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TabularCell {
    Empty,
    Frozen,
    Start,
    Goal,
    Hazard,
}

/// Marker class overlaid on a tabular grid cell (mirror of `rlevo-core`
/// `TabularMarkerKind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TabularMarkerKind {
    Agent,
    Passenger,
    Destination,
    Location,
}

/// A point-of-interest overlaid on a tabular grid cell (mirror of
/// `rlevo-core` `TabularMarker`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TabularMarker {
    pub x: u16,
    pub y: u16,
    pub kind: TabularMarkerKind,
}

/// Grid layout for grid-shaped toy-text envs (mirror of `rlevo-core`
/// `TabularGrid`). `cells` is row-major, `len == width * height`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TabularGrid {
    pub width: u16,
    pub height: u16,
    pub cells: Vec<TabularCell>,
    pub markers: Vec<TabularMarker>,
}

/// Card-table layout for Blackjack (mirror of `rlevo-core` `CardTable`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CardTable {
    pub player_cards: Vec<u8>,
    pub player_total: u8,
    pub usable_ace: bool,
    pub dealer_cards: Vec<u8>,
    pub dealer_showing: u8,
}

/// Layout discriminant (mirror of `rlevo-core` `TabularLayout`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum TabularLayout {
    Grid(TabularGrid),
    Cards(CardTable),
}

/// Structured toy-text layout (mirror of `rlevo-benchmarks::record::TabularPayload`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TabularPayload {
    pub layout: TabularLayout,
}

/// Semantic role of a classic-control body (mirror of `rlevo-core`
/// `Classic2DRole`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Classic2DRole {
    Track,
    Cart,
    Pole,
    Link,
    Car,
    Hinge,
}

/// One world-space body of a classic-control mechanism (mirror of
/// `rlevo-core` `Classic2DBody`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Classic2DBody {
    pub points: Vec<Point2>,
    pub role: Classic2DRole,
    pub closed: bool,
}

/// Structured 2-D line-art for classic-control envs (mirror of
/// `rlevo-benchmarks::record::Classic2DPayload`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Classic2DPayload {
    pub bodies: Vec<Classic2DBody>,
    pub bounds: (Point2, Point2),
}

// ---- /payload mirror --------------------------------------------------

/// Current wire-format version this client crate writes/expects.
// Mirror of `rlevo-benchmarks::record::FORMAT_VERSION`.  Keep in sync;
// the const assertions in rlevo-benchmarks/tests/wire_format_compat.rs
// catch drift at compile time.
pub const FORMAT_VERSION: u16 = 6;

/// Oldest on-disk version this client accepts. Equal to
/// [`FORMAT_VERSION`] — no backward compatibility before first release.
// Mirror of `rlevo-benchmarks::record::MIN_SUPPORTED_VERSION`.
pub const MIN_SUPPORTED_VERSION: u16 = 6;

/// Returns the standard bincode configuration used for all record encode/decode operations.
#[must_use]
pub fn bincode_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

/// Opaque string identifier for a training run, unique within a workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunId(pub String);

/// Discriminant for the environment family recorded in a run.
///
/// `#[non_exhaustive]` — new families may be added in future wire-format
/// versions without breaking existing clients; match arms should include a
/// wildcard that routes to [`crate::adapters::fallback`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EnvFamily {
    /// CartPole, MountainCar, Pendulum, Acrobot.
    Classic,
    /// Minigrid-style grid worlds (empty, four-rooms, door-key, …).
    Grids,
    /// FrozenLake, CliffWalking, Taxi, Blackjack.
    ToyText,
    /// LunarLander, BipedalWalker, CarRacing (Box2D physics).
    Box2d,
    /// Ant, HalfCheetah, Hopper, Walker2D (MuJoCo / locomotion).
    Locomotion,
    /// Sphere, Ackley, Rastrigin (black-box optimisation landscapes).
    Landscapes,
}

/// Family-specific per-frame data appended to each [`FrameRecord`].
///
/// `Ascii` is the fallback when the environment has no rich payload.
/// `#[non_exhaustive]` — new variants may arrive in future format versions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FamilyPayload {
    /// No rich payload; the frame carries only ASCII / styled text.
    Ascii,
    /// 2-D landscape-search snapshot (Sphere, Ackley, Rastrigin, …).
    Landscape2D(Landscape2DPayload),
    /// Box2D rigid-body scene (LunarLander, BipedalWalker, …).
    Box2dBodies(Box2dPayload),
    /// Sagittal-plane locomotion skeleton (Ant, HalfCheetah, …).
    Locomotion2D(Locomotion2DPayload),
    /// Structured tile grid (empty, four-rooms, door-key, …). Added in
    /// `FORMAT_VERSION = 5`.
    Grid(GridPayload),
    /// Structured toy-text layout (FrozenLake/CliffWalking/Taxi grid, or
    /// Blackjack cards). Added in `FORMAT_VERSION = 5`.
    TabularText(TabularPayload),
    /// Structured 2-D line-art for classic physics envs. Added in
    /// `FORMAT_VERSION = 5`.
    Classic2D(Classic2DPayload),
}

/// Trial provenance. Mirror of
/// `rlevo_benchmarks::record::schema::TrialRef`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrialRef {
    /// Zero-based environment index within the suite.
    pub env_index: u32,
    /// Zero-based seed-repetition index for that environment.
    pub trial_index: u32,
}

/// Whether an episode was a training or evaluation rollout. Added in
/// `FORMAT_VERSION = 6`.
// Mirror of `rlevo-benchmarks::record::EpisodeKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EpisodeKind {
    /// Exploration rollout produced while the policy is being updated.
    #[default]
    Training,
    /// Deterministic rollout produced to measure the current policy.
    Evaluation,
}

/// Why a [`CheckpointRef`] was written. Added in `FORMAT_VERSION = 6`.
// Mirror of `rlevo-benchmarks::record::CheckpointKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CheckpointKind {
    /// Saved on a periodic step interval during training.
    Periodic,
    /// Saved because it is the best-performing checkpoint so far.
    Best,
    /// Saved at the end of the run.
    Final,
}

/// Burn recorder format of a saved learner checkpoint. Added in
/// `FORMAT_VERSION = 6`.
// Mirror of `rlevo-benchmarks::record::CheckpointFormat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CheckpointFormat {
    /// `NamedMpkFileRecorder` (`.mpk`).
    NamedMpk,
    /// `NamedMpkGzFileRecorder` (`.mpk.gz`).
    NamedMpkGz,
    /// `BinFileRecorder` (`.bin`).
    Bin,
    /// `PrettyJsonFileRecorder` (`.json`).
    Json,
    /// Any other recorder.
    Other,
}

/// Reference to one Burn-saved learner checkpoint; the record references
/// the file and never embeds weights. Added in `FORMAT_VERSION = 6`.
// Mirror of `rlevo-benchmarks::record::CheckpointRef`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointRef {
    /// Global training step at which the checkpoint was written.
    pub step: u32,
    /// Why this checkpoint was written.
    pub kind: CheckpointKind,
    /// Burn recorder format used for the artifact.
    pub format: CheckpointFormat,
    /// Path to the artifact, relative to the run directory.
    pub path: String,
    /// Metric (e.g. eval return) at save time, if known.
    pub metric: Option<f64>,
    /// 128-bit content digest.
    pub digest: Option<[u8; 16]>,
}

/// Fixed-size preamble written at the start of every `.rec` file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeRecordHeader {
    /// Wire-format version; must equal [`FORMAT_VERSION`] to be decoded.
    pub format_version: u16,
    /// Identifier of the training run that produced this file.
    pub run_id: RunId,
    /// RNG seed used for the run (for reproducibility).
    pub seed: u64,
    /// Environment family; controls which family adapter renders the frames.
    pub env_family: EnvFamily,
    /// Unix timestamp (seconds) when the episode file was created.
    pub created_at: i64,
    /// Trial that produced this episode, or `None` for non-harness
    /// producers. Added in `FORMAT_VERSION = 4`.
    pub trial: Option<TrialRef>,
    /// Whether this episode was a training or evaluation rollout. Added in
    /// `FORMAT_VERSION = 6`.
    pub kind: EpisodeKind,
}

/// One recorded simulation step, stored as a [`RecordChunk::Frame`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameRecord {
    /// Simulation step counter within the episode.
    pub step: u32,
    /// Raw action bytes serialised by the agent (interpretation is family-specific).
    pub action: Vec<u8>,
    /// Scalar reward received at this step.
    pub reward: f32,
    /// Plain-text ASCII render produced by the library tier, if available.
    pub ascii: Option<String>,
    /// Colour-annotated styled render produced by the library tier, if available.
    pub styled: Option<StyledFrame>,
    /// Family-specific rich payload (bodies, joints, landscape position, …).
    pub family_payload: FamilyPayload,
}

/// A single named scalar measurement emitted during training, stored as [`RecordChunk::Metrics`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSample {
    /// Training step at which the metric was recorded.
    pub step: u32,
    /// Metric key (e.g. `"policy_loss"`, `"entropy"`, `"best_fitness"`).
    pub name: String,
    /// Scalar value of the metric at `step`.
    pub value: f64,
}

/// One per-generation snapshot of an EA population, stored as [`RecordChunk::Population`].
///
/// Mirror of `rlevo_benchmarks::record::PopulationSample`; present in v3+ records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopulationSample {
    /// EA generation index; used as the x-axis in population panels.
    pub generation: u32,
    /// Raw fitness score for every individual in the population, in population order.
    pub fitnesses: Vec<f32>,
    /// Optional scalar diversity metric (e.g. mean pairwise genome distance).
    pub diversity: Option<f32>,
    /// Index into `fitnesses` of the best individual this generation.
    pub best_index: u32,
    /// 128-bit content digest of the best genome (for reproducibility); `None` when not recorded.
    pub best_genome_digest: Option<[u8; 16]>,
    /// Content digests of the parents that produced the best genome.
    pub parents_of_best: Vec<[u8; 16]>,
    /// Per-individual RL episode returns when the outer EA wraps an inner RL loop.
    pub inner_rl_returns: Option<Vec<f32>>,
}

/// Length-prefixed wire-format chunk written by the on-disk record writer.
///
/// **Variant ordering is wire-format-stable** — new variants append at
/// the end so existing bincode tags keep decoding. `Population` is at tag 2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecordChunk {
    Frame(FrameRecord),
    Metrics(Vec<MetricSample>),
    Population(PopulationSample),
}

/// In-memory aggregate of one fully decoded `.rec` file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpisodeRecord {
    /// Fixed-size preamble with run identity and format metadata.
    pub header: EpisodeRecordHeader,
    /// Ordered sequence of recorded simulation steps.
    pub frames: Vec<FrameRecord>,
    /// All metric samples emitted during the episode, in emission order.
    pub metrics: Vec<MetricSample>,
    /// Per-generation EA population snapshots; empty for RL-only runs.
    #[serde(default)]
    pub population_samples: Vec<PopulationSample>,
}

/// Key-value map of algorithm hyperparameters recorded at run start.
pub type Hyperparameters = BTreeMap<String, String>;

/// JSON manifest embedded in `index.html` describing the overall run.
///
/// `Eq` is intentionally not derived: `success_threshold` and the metrics
/// carried by [`CheckpointRef`] are `f64`, which is not `Eq`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    /// Identifier of the training run.
    pub run_id: RunId,
    /// RNG seed used for the run.
    pub seed: u64,
    /// Environment family for all episodes in the run.
    pub env_family: EnvFamily,
    /// Unix timestamp (seconds) when the run started.
    pub created_at: i64,
    /// Unix timestamp (seconds) when the run finished.
    pub finished_at: i64,
    /// Number of episodes recorded (may differ from `index.html` episode count on partial runs).
    pub episode_count: u32,
    /// Only every `frame_stride`-th simulation step was written to disk.
    pub frame_stride: u16,
    /// Wire-format version the emitter used; should match [`FORMAT_VERSION`].
    pub format_version: u16,
    /// Algorithm hyperparameters logged at run start; empty when not provided.
    #[serde(default)]
    pub hyperparameters: Hyperparameters,
    /// Algorithm identity (e.g. `"ppo"`). Added in `FORMAT_VERSION = 6`.
    #[serde(default)]
    pub algorithm: Option<String>,
    /// `rlevo` crate version. Added in v6.
    #[serde(default)]
    pub rlevo_version: Option<String>,
    /// Rust toolchain version string. Added in v6.
    #[serde(default)]
    pub rustc_version: Option<String>,
    /// Resolved `burn` dependency version. Added in v6.
    #[serde(default)]
    pub burn_version: Option<String>,
    /// `OS`-`ARCH` platform string. Added in v6.
    #[serde(default)]
    pub platform: Option<String>,
    /// Git commit hash of the build, if known. Added in v6.
    #[serde(default)]
    pub git_commit: Option<String>,
    /// Whether the working tree was dirty at build time. Added in v6.
    #[serde(default)]
    pub git_dirty: Option<bool>,
    /// Backend device descriptor. Added in v6.
    #[serde(default)]
    pub device: Option<String>,
    /// Distinct seed count across the trial suite. Added in v6.
    #[serde(default)]
    pub num_seeds: Option<u32>,
    /// Success threshold that produced `success_rate`. Added in v6.
    #[serde(default)]
    pub success_threshold: Option<f64>,
    /// Deep-RL learner checkpoints (Burn-`Recorder` files referenced, never
    /// embedded). Empty for EA and un-wired RL. Added in v6.
    #[serde(default)]
    pub checkpoints: Vec<CheckpointRef>,
}

/// Decode the raw bytes of a single `episode_*.rec` file produced by
/// the on-disk record writer. Tolerates truncated tails by stopping
/// cleanly at the last whole chunk.
///
/// # Errors
///
/// Returns [`DecodeError::Truncated`] if the preamble is shorter than
/// 16 bytes, [`DecodeError::VersionMismatch`] if the format version in
/// the preamble does not match [`FORMAT_VERSION`], or
/// [`DecodeError::Bincode`] if a length-prefixed chunk fails to
/// deserialise.
pub fn decode_episode_record(bytes: &[u8]) -> Result<EpisodeRecord, DecodeError> {
    if bytes.len() < 16 {
        return Err(DecodeError::Truncated("preamble"));
    }
    let version = u16::from_le_bytes([bytes[0], bytes[1]]);
    if version != FORMAT_VERSION {
        return Err(DecodeError::VersionMismatch {
            file: version,
            client: FORMAT_VERSION,
        });
    }
    let mut cursor = 16;
    let header: EpisodeRecordHeader = read_chunk(bytes, &mut cursor)?.ok_or(DecodeError::Truncated("header"))?;

    let mut frames = Vec::new();
    let mut metrics = Vec::new();
    let mut population_samples = Vec::new();
    while let Some(chunk) = read_chunk::<RecordChunk>(bytes, &mut cursor)? {
        match chunk {
            RecordChunk::Frame(fr) => frames.push(fr),
            RecordChunk::Metrics(ms) => metrics.extend(ms),
            RecordChunk::Population(ps) => population_samples.push(ps),
        }
    }
    Ok(EpisodeRecord {
        header,
        frames,
        metrics,
        population_samples,
    })
}

/// Reads one length-prefixed bincode chunk from `bytes` at `*cursor`.
///
/// Advances `*cursor` past the 4-byte length prefix and the payload.
/// Returns `Ok(None)` on a truncated length prefix or partial payload so
/// the caller can stop cleanly without treating a truncated tail as an error.
fn read_chunk<T: for<'de> Deserialize<'de>>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<Option<T>, DecodeError> {
    if *cursor >= bytes.len() {
        return Ok(None);
    }
    if bytes.len() - *cursor < 4 {
        // Partial length prefix — truncated tail.
        return Ok(None);
    }
    let len = u32::from_le_bytes([
        bytes[*cursor],
        bytes[*cursor + 1],
        bytes[*cursor + 2],
        bytes[*cursor + 3],
    ]) as usize;
    *cursor += 4;
    let available = bytes.len() - *cursor;
    if available < len {
        // Partial payload — truncated tail.
        return Ok(None);
    }
    let payload = &bytes[*cursor..*cursor + len];
    *cursor += len;
    let (value, _): (T, usize) = bincode::serde::decode_from_slice(payload, bincode_config())
        .map_err(|e| DecodeError::Bincode(e.to_string()))?;
    Ok(Some(value))
}

/// Errors that can occur while decoding a `.rec` binary file.
#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("record file truncated at {0}")]
    Truncated(&'static str),
    #[error("format version mismatch: file={file} client={client}")]
    VersionMismatch { file: u16, client: u16 },
    #[error("bincode decode failed: {0}")]
    Bincode(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_header() {
        let h = EpisodeRecordHeader {
            format_version: FORMAT_VERSION,
            run_id: RunId("xyz".into()),
            seed: 42,
            env_family: EnvFamily::Classic,
            created_at: 1700,
            trial: None,
            kind: EpisodeKind::Evaluation,
        };
        let bytes = bincode::serde::encode_to_vec(&h, bincode_config()).unwrap();
        let (decoded, _): (EpisodeRecordHeader, usize) =
            bincode::serde::decode_from_slice(&bytes, bincode_config()).unwrap();
        assert_eq!(h, decoded);
        assert_eq!(decoded.kind, EpisodeKind::Evaluation);
    }

    #[test]
    fn decode_rejects_version_mismatch() {
        let mut bytes = vec![0u8; 16];
        bytes[0..2].copy_from_slice(&999u16.to_le_bytes());
        let err = decode_episode_record(&bytes).unwrap_err();
        match err {
            DecodeError::VersionMismatch { file: 999, client } if client == FORMAT_VERSION => {}
            other => panic!("expected version mismatch, got {other:?}"),
        }
    }
}
