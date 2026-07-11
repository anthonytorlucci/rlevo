//! Blackjack-v1 environment.
//!
//! Implements a single-player Blackjack game against a fixed dealer strategy using an infinite
//! deck (draws with replacement). Episodes always terminate — the environment is never truncated.
//!
//! ## Observation space
//!
//! `(player_sum, dealer_showing, usable_ace)` — a 3-element tuple represented by
//! [`BlackjackObservation`].
//!
//! ## Action space
//!
//! Two discrete actions via [`BlackjackAction`]: `Stick` (0) or `Hit` (1).
//!
//! ## Reward
//!
//! | Outcome | Reward |
//! |---------|--------|
//! | Win     | +1.0   |
//! | Push    |  0.0   |
//! | Lose    | -1.0   |
//! | Natural (Standard, bonus on) | +1.5 |
//!
//! ## Rule variants
//!
//! Configure via [`BlackjackVariant`]: standard casino rules or the Sutton & Barto Example 5.1
//! formulation.
//!
//! ## Episode lifecycle
//!
//! Stepping after the episode has ended returns
//! [`EnvironmentError::StepAfterEpisodeEnd`]; call `reset()` to start a new episode.
//!
//! ## RNG behaviour
//!
//! The RNG advances continuously across episodes — `reset()` does **not** reseed. Two instances
//! with the same seed produce identical trajectories.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Action, Observation, State};
use rlevo_core::config::{ConfigError, Validate};
use rlevo_core::environment::{
    ConstructableEnv, Environment, EnvironmentError, Snapshot, SnapshotBase,
};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};

use crate::episode::EpisodeGuard;

// ── card helpers ──────────────────────────────────────────────────────────────

fn draw_card(rng: &mut StdRng) -> u8 {
    rng.random_range(1u8..=13).min(10)
}

/// Scores `hand`, returning `(total, usable_ace)`.
///
/// An ace counts as 11 whenever that keeps the hand at or under 21 (`usable_ace`
/// is then `true`); otherwise every card counts at face value.
///
/// # Saturation invariant
///
/// The raw pip total is accumulated in a `u16` and saturated to [`u8::MAX`] on
/// the way out, so an arbitrarily long hand can never overflow the accumulator
/// (a `u8` sum overflows at ~26 ten-valued cards: a panic in debug, a silent
/// wraparound into the "valid" range in release). Saturation is sound for every
/// consumer of this value: `255 > 21`, so a saturated total is still classified
/// as a bust, which is the only meaningful reading of a hand that large. The
/// usable-ace branch is unaffected — it only adds 10 when the total is at most
/// 11, far below the saturation point.
fn hand_value(hand: &[u8]) -> (u8, bool) {
    let sum: u16 = hand.iter().map(|&card| u16::from(card)).sum();
    let sum = u8::try_from(sum).unwrap_or(u8::MAX);
    let has_ace = hand.contains(&1);
    if has_ace && sum.saturating_add(10) <= 21 {
        (sum + 10, true)
    } else {
        (sum, false)
    }
}

fn is_natural(hand: &[u8]) -> bool {
    hand.len() == 2 && hand_value(hand).0 == 21
}

// ── config ────────────────────────────────────────────────────────────────────

/// Rule variant controlling Blackjack reward and terminal logic.
///
/// Selects between standard casino rules and the academic formulation from
/// Sutton & Barto, *Reinforcement Learning: An Introduction*, Example 5.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlackjackVariant {
    /// Standard casino rules.
    ///
    /// When `natural_pays_bonus` is `true`, a player natural (two-card 21) that beats a
    /// non-natural dealer hand pays 1.5 instead of 1.0.
    Standard { natural_pays_bonus: bool },
    /// Strict Sutton & Barto (S&B) Example 5.1 rules.
    ///
    /// A player natural against a non-natural dealer ends the episode immediately with reward
    /// +1.0; a dealer natural against a non-natural player ends it with reward −1.0.
    SuttonBarto,
}

impl Default for BlackjackVariant {
    fn default() -> Self {
        BlackjackVariant::Standard {
            natural_pays_bonus: false,
        }
    }
}

/// Configuration for the [`Blackjack`] environment.
///
/// Construct with [`BlackjackConfig::default`] or via the builder returned by
/// [`BlackjackConfig::builder`].
///
/// # Examples
///
/// ```rust
/// use rlevo_environments::toy_text::blackjack::{BlackjackConfig, BlackjackVariant};
///
/// let cfg = BlackjackConfig::builder()
///     .variant(BlackjackVariant::Standard { natural_pays_bonus: true })
///     .seed(42)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct BlackjackConfig {
    /// Rule variant governing reward and natural-hand handling.
    pub variant: BlackjackVariant,
    /// Seed used to initialise the RNG when the environment is created. Default: `0`.
    pub seed: u64,
}

impl Default for BlackjackConfig {
    fn default() -> Self {
        Self {
            variant: BlackjackVariant::Standard {
                natural_pays_bonus: false,
            },
            seed: 0,
        }
    }
}

impl BlackjackConfig {
    /// Returns a builder for constructing a `BlackjackConfig`.
    pub fn builder() -> BlackjackConfigBuilder {
        BlackjackConfigBuilder::default()
    }
}

impl Validate for BlackjackConfig {
    /// [`BlackjackConfig`] carries only a rule variant and a seed, so it has no
    /// numeric invariant to check; validation always succeeds.
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }
}

/// Builder for [`BlackjackConfig`].
#[derive(Default)]
pub struct BlackjackConfigBuilder {
    variant: BlackjackVariant,
    seed: u64,
}

impl BlackjackConfigBuilder {
    /// Sets the rule variant.
    pub fn variant(mut self, v: BlackjackVariant) -> Self {
        self.variant = v;
        self
    }

    /// Sets the RNG seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Builds the [`BlackjackConfig`].
    pub fn build(self) -> BlackjackConfig {
        BlackjackConfig {
            variant: self.variant,
            seed: self.seed,
        }
    }
}

// ── state ─────────────────────────────────────────────────────────────────────

/// Full internal state of a Blackjack game.
///
/// The player and dealer hands are stored privately; only the observable
/// summary fields are exposed to agents via [`BlackjackObservation`].
#[derive(Debug, Clone)]
pub struct BlackjackState {
    /// Current point total for the player's hand (may include a usable-ace bonus of 10).
    pub player_sum: u8,
    /// The dealer's single face-up card value, in `[1, 10]`.
    pub dealer_showing: u8,
    /// `true` when the player holds an ace counted as 11 without busting.
    pub usable_ace: bool,
    player_hand: Vec<u8>,
    dealer_hand: Vec<u8>,
}

impl State<1> for BlackjackState {
    type Observation = BlackjackObservation;

    fn shape() -> [usize; 1] {
        [3]
    }

    fn observe(&self) -> BlackjackObservation {
        BlackjackObservation {
            player_sum: self.player_sum,
            dealer_showing: self.dealer_showing,
            usable_ace: u8::from(self.usable_ace),
        }
    }

    fn is_valid(&self) -> bool {
        (4..=32).contains(&self.player_sum) && (1..=10).contains(&self.dealer_showing)
    }
}

// ── observation ───────────────────────────────────────────────────────────────

/// Agent-visible observation: `(player_sum, dealer_showing, usable_ace)`.
///
/// The usable-ace field is encoded as `u8` (`1` = usable, `0` = not usable) for
/// compatibility with numeric pipelines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackjackObservation {
    /// Player's current hand total.
    pub player_sum: u8,
    /// Dealer's single face-up card, in `[1, 10]`.
    pub dealer_showing: u8,
    /// `1` if the player holds a usable ace, `0` otherwise.
    pub usable_ace: u8,
}

impl Observation<1> for BlackjackObservation {
    fn shape() -> [usize; 1] {
        [3]
    }
}

// ── action ────────────────────────────────────────────────────────────────────

/// Two-action Blackjack space: hold the current hand or draw another card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlackjackAction {
    /// Hold the current hand and trigger the dealer's draw sequence.
    Stick = 0,
    /// Draw one more card from the infinite deck.
    Hit = 1,
}

impl Action<1> for BlackjackAction {
    fn shape() -> [usize; 1] {
        [1]
    }
    fn is_valid(&self) -> bool {
        true
    }
}

impl DiscreteAction<1> for BlackjackAction {
    const ACTION_COUNT: usize = 2;

    fn from_index(index: usize) -> Self {
        match index {
            0 => BlackjackAction::Stick,
            1 => BlackjackAction::Hit,
            _ => panic!("BlackjackAction index {index} out of range [0, 2)"),
        }
    }

    fn to_index(&self) -> usize {
        *self as usize
    }
}

// ── environment ───────────────────────────────────────────────────────────────

/// Blackjack-v1 environment (infinite deck, single-player vs dealer).
///
/// The RNG advances continuously across episodes — `reset()` does **not**
/// reseed from `config.seed`. Two `Blackjack` instances created with the same
/// seed will produce identical trajectories.
///
/// A finished episode is not resumable: once `step()` has emitted a terminal
/// snapshot, further calls return [`EnvironmentError::StepAfterEpisodeEnd`]
/// until `reset()` is called.
#[derive(Debug)]
pub struct Blackjack {
    state: BlackjackState,
    config: BlackjackConfig,
    rng: StdRng,
    /// Rejects a `step()` taken after the episode has ended.
    guard: EpisodeGuard,
}

impl Blackjack {
    /// Create a Blackjack environment with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] if `config` fails [`Validate`].
    /// [`BlackjackConfig`] currently has no numeric invariant, so this never
    /// fails in practice; the fallible signature keeps the construction
    /// contract uniform across environments.
    pub fn with_config(config: BlackjackConfig) -> Result<Self, ConfigError> {
        config.validate()?;
        let rng = StdRng::seed_from_u64(config.seed);
        Ok(Self {
            state: BlackjackState {
                player_sum: 0,
                dealer_showing: 1,
                usable_ace: false,
                player_hand: Vec::new(),
                dealer_hand: Vec::new(),
            },
            config,
            rng,
            guard: EpisodeGuard::new(),
        })
    }

    fn deal_initial(&mut self) {
        let p1 = draw_card(&mut self.rng);
        let p2 = draw_card(&mut self.rng);
        let d1 = draw_card(&mut self.rng);
        let d2 = draw_card(&mut self.rng);
        let player_hand = vec![p1, p2];
        let (player_sum, usable_ace) = hand_value(&player_hand);
        self.state = BlackjackState {
            player_sum,
            dealer_showing: d1,
            usable_ace,
            player_hand,
            dealer_hand: vec![d1, d2],
        };
    }

    fn apply_sab_override(&self, reward: &mut f32, done: &mut bool) {
        if let BlackjackVariant::SuttonBarto = self.config.variant {
            let p_nat = is_natural(&self.state.player_hand);
            let d_nat = is_natural(&self.state.dealer_hand);
            if p_nat && !d_nat {
                *reward = 1.0;
                *done = true;
            } else if d_nat && !p_nat {
                *reward = -1.0;
                *done = true;
            }
        }
    }
}

impl ConstructableEnv for Blackjack {
    fn new(_render: bool) -> Self {
        Self::with_config(BlackjackConfig::default()).expect("default config must validate")
    }
}

impl Environment<1, 1, 1> for Blackjack {
    type StateType = BlackjackState;
    type ObservationType = BlackjackObservation;
    type ActionType = BlackjackAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, BlackjackObservation, ScalarReward>;

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.guard.reset();
        self.deal_initial();
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward(0.0),
        ))
    }

    /// Applies one action and settles the hand.
    ///
    /// # Errors
    ///
    /// Returns [`EnvironmentError::StepAfterEpisodeEnd`] if the episode has
    /// already ended. The check runs before any card is drawn, so a rejected
    /// call leaves both the hands and the RNG stream untouched.
    fn step(&mut self, action: BlackjackAction) -> Result<Self::SnapshotType, EnvironmentError> {
        // Before any mutation *and* before any RNG draw: a rejected step must not
        // advance the stream, or determinism would depend on the caller's mistakes.
        self.guard.check()?;

        let mut reward;
        let mut done;

        match action {
            BlackjackAction::Hit => {
                let card = draw_card(&mut self.rng);
                self.state.player_hand.push(card);
                let (sum, ace) = hand_value(&self.state.player_hand);
                self.state.player_sum = sum;
                self.state.usable_ace = ace;
                if sum > 21 {
                    reward = -1.0_f32;
                    done = true;
                } else {
                    reward = 0.0_f32;
                    done = false;
                }
            }
            BlackjackAction::Stick => {
                while hand_value(&self.state.dealer_hand).0 < 17 {
                    let card = draw_card(&mut self.rng);
                    self.state.dealer_hand.push(card);
                }
                let dealer_sum = hand_value(&self.state.dealer_hand).0;
                let player_sum = self.state.player_sum;
                reward = if dealer_sum > 21 || player_sum > dealer_sum {
                    1.0
                } else if player_sum < dealer_sum {
                    -1.0
                } else {
                    0.0
                };
                // Natural 3:2 bonus.
                let pays_natural = matches!(
                    self.config.variant,
                    BlackjackVariant::Standard {
                        natural_pays_bonus: true
                    }
                );
                if pays_natural
                    && (reward - 1.0).abs() < 1e-6
                    && is_natural(&self.state.player_hand)
                    && !is_natural(&self.state.dealer_hand)
                {
                    reward = 1.5;
                }
                done = true;
            }
        }

        self.apply_sab_override(&mut reward, &mut done);

        let obs = self.state.observe();
        // Build the snapshot once and record *its* status, so the guard can never
        // disagree with the snapshot the caller was handed.
        let snapshot = if done {
            SnapshotBase::terminated(obs, ScalarReward(reward))
        } else {
            SnapshotBase::running(obs, ScalarReward(reward))
        };
        self.guard.record(snapshot.status());
        Ok(snapshot)
    }
}

// ---------------------------------------------------------------------------
// ASCII renderer
// ---------------------------------------------------------------------------

impl crate::render::AsciiRenderable for Blackjack {
    fn render_ascii(&self) -> String {
        let ace = if self.state.usable_ace { "A" } else { "" };
        format!(
            "Blackjack  player={}{ace}  dealer_showing={}",
            self.state.player_sum, self.state.dealer_showing
        )
    }

    fn render_styled(&self) -> crate::render::StyledFrame {
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER};
        use crate::render::{SpanStyle, StyledFrame, StyledLine, StyledSpan};

        const LABEL: &str = "Blackjack";
        let line = self.render_ascii();
        let label_style = SpanStyle::default()
            .fg(AGENT_FG)
            .with_modifier(AGENT_MODIFIER);
        let styled_line = if let Some(rest) = line.strip_prefix(LABEL) {
            StyledLine::from_spans(vec![
                StyledSpan::new(LABEL, label_style),
                StyledSpan::raw(rest.to_string()),
            ])
        } else {
            StyledLine::unstyled(line)
        };
        StyledFrame {
            lines: vec![styled_line],
        }
    }
}

impl rlevo_core::render::payload::TabularPayloadSource for Blackjack {
    fn tabular_snapshot(&self) -> rlevo_core::render::payload::TabularSnapshot {
        use rlevo_core::render::payload::{CardTable, TabularLayout, TabularSnapshot};
        TabularSnapshot {
            layout: TabularLayout::Cards(CardTable {
                player_cards: self.state.player_hand.clone(),
                player_total: self.state.player_sum,
                usable_ace: self.state.usable_ace,
                dealer_cards: self.state.dealer_hand.clone(),
                dealer_showing: self.state.dealer_showing,
            }),
        }
    }
}

#[cfg(test)]
/// Unit tests for [`Blackjack`], covering actions, observations, rewards, and RNG determinism.
mod tests {
    use super::*;
    use crate::episode::assert_rejects_post_terminal_step;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::base::Observation;
    use rlevo_core::environment::EpisodeStatus;

    fn make_env() -> Blackjack {
        Blackjack::with_config(BlackjackConfig::default()).expect("valid config")
    }

    #[test]
    fn default_config_validates() {
        assert!(BlackjackConfig::default().validate().is_ok());
    }

    impl Blackjack {
        /// Force-sets both hands for deterministic reward testing.
        fn set_hands(&mut self, player_hand: Vec<u8>, dealer_hand: Vec<u8>) {
            let (player_sum, usable_ace) = hand_value(&player_hand);
            let dealer_showing = dealer_hand[0];
            self.state = BlackjackState {
                player_sum,
                dealer_showing,
                usable_ace,
                player_hand,
                dealer_hand,
            };
        }
    }

    #[test]
    /// Verifies the discrete action count matches the Blackjack action space size.
    fn action_count() {
        assert_eq!(BlackjackAction::ACTION_COUNT, 2);
    }

    #[test]
    /// Verifies `from_index` and `to_index` are inverses for all valid action indices.
    fn action_roundtrip() {
        for i in 0..BlackjackAction::ACTION_COUNT {
            assert_eq!(BlackjackAction::from_index(i).to_index(), i);
        }
    }

    #[test]
    /// Verifies the observation shape matches the 3-element tuple specification.
    fn obs_shape() {
        assert_eq!(BlackjackObservation::shape(), [3]);
    }

    #[test]
    /// Verifies observation fields are stored and retrieved correctly.
    fn obs_encoding() {
        let obs = BlackjackObservation {
            player_sum: 18,
            dealer_showing: 10,
            usable_ace: 0,
        };
        assert_eq!(obs.player_sum, 18);
        assert_eq!(obs.dealer_showing, 10);
        assert_eq!(obs.usable_ace, 0);
    }

    #[test]
    /// Verifies that a player bust on `Hit` yields reward −1 and terminates the episode.
    fn bust_on_hit_returns_negative_one() {
        // Player at 20 (two 10s), any hit ≥ 2 busts. Try seeds until we get a bust.
        for seed in 0u64..200 {
            let mut env = make_env();
            env.reset().unwrap();
            env.set_hands(vec![10, 10], vec![6, 5]);
            env.rng = StdRng::seed_from_u64(seed);
            let snap = env.step(BlackjackAction::Hit).unwrap();
            let r: f32 = (*snap.reward()).into();
            if r == -1.0 {
                assert!(snap.is_done());
                return;
            }
        }
        panic!("could not find a busting seed in 200 tries");
    }

    #[test]
    /// Verifies that a dealer bust on `Stick` yields reward +1 and terminates the episode.
    fn dealer_bust_returns_positive_one() {
        // Player at 18 (9+9), dealer at 16 (10+6) must draw. Find seed where dealer busts.
        for seed in 0u64..200 {
            let mut env = make_env();
            env.reset().unwrap();
            env.set_hands(vec![9, 9], vec![10, 6]);
            env.rng = StdRng::seed_from_u64(seed);
            let snap = env.step(BlackjackAction::Stick).unwrap();
            let r: f32 = (*snap.reward()).into();
            if r == 1.0 {
                assert!(snap.is_done());
                return;
            }
        }
        panic!("could not find a dealer-bust seed in 200 tries");
    }

    #[test]
    /// Verifies that equal player and dealer sums produce a push (reward 0).
    fn push_on_equal_sums() {
        let mut env = make_env();
        env.reset().unwrap();
        // Player 18, dealer 18 (10+8) → no draw needed (18 ≥ 17).
        env.set_hands(vec![9, 9], vec![10, 8]);
        let snap = env.step(BlackjackAction::Stick).unwrap();
        let r: f32 = (*snap.reward()).into();
        assert_eq!(r, 0.0, "equal sums must push (reward 0), got {r}");
        assert!(snap.is_done());
    }

    #[test]
    /// Verifies the 3:2 natural payout when `natural_pays_bonus` is enabled.
    fn natural_pays_1_5_when_flag_set() {
        let cfg = BlackjackConfig::builder()
            .variant(BlackjackVariant::Standard {
                natural_pays_bonus: true,
            })
            .seed(0)
            .build();
        let mut env = Blackjack::with_config(cfg).expect("valid config");
        env.reset().unwrap();
        // Natural player [1,10]=21, dealer [8,9]=17 (not natural, no draw).
        env.set_hands(vec![1, 10], vec![8, 9]);
        let snap = env.step(BlackjackAction::Stick).unwrap();
        let r: f32 = (*snap.reward()).into();
        assert!((r - 1.5).abs() < 1e-6, "natural pays 1.5, got {r}");
        assert!(snap.is_done());
    }

    #[test]
    /// Verifies the S&B variant pays +1.0 on a player natural vs non-natural dealer.
    fn sab_player_natural_wins_one() {
        let cfg = BlackjackConfig::builder()
            .variant(BlackjackVariant::SuttonBarto)
            .seed(0)
            .build();
        let mut env = Blackjack::with_config(cfg).expect("valid config");
        env.reset().unwrap();
        // SAB: player natural [1,10], dealer non-natural [8,9] → reward = 1.0.
        env.set_hands(vec![1, 10], vec![8, 9]);
        let snap = env.step(BlackjackAction::Stick).unwrap();
        let r: f32 = (*snap.reward()).into();
        assert!(
            (r - 1.0).abs() < 1e-6,
            "SAB player natural must pay 1.0, got {r}"
        );
        assert!(snap.is_done());
    }

    #[test]
    /// Verifies the S&B variant charges −1.0 on a dealer natural vs non-natural player.
    fn sab_dealer_natural_costs_one() {
        let cfg = BlackjackConfig::builder()
            .variant(BlackjackVariant::SuttonBarto)
            .seed(0)
            .build();
        let mut env = Blackjack::with_config(cfg).expect("valid config");
        env.reset().unwrap();
        // SAB: dealer natural [1,10], player non-natural [9,7] → reward = -1.0.
        env.set_hands(vec![9, 7], vec![1, 10]);
        let snap = env.step(BlackjackAction::Stick).unwrap();
        let r: f32 = (*snap.reward()).into();
        assert!(
            (r - (-1.0)).abs() < 1e-6,
            "SAB dealer natural costs -1.0, got {r}"
        );
        assert!(snap.is_done());
    }

    #[test]
    /// Verifies that two environments seeded identically produce the same cumulative reward.
    fn determinism() {
        let cfg = BlackjackConfig {
            variant: BlackjackVariant::default(),
            seed: 99,
        };
        let run = || {
            let mut env = Blackjack::with_config(cfg.clone()).expect("valid config");
            let mut total = 0.0_f32;
            for _ in 0..10 {
                env.reset().unwrap();
                loop {
                    let a = if env.state.player_sum < 18 {
                        BlackjackAction::Hit
                    } else {
                        BlackjackAction::Stick
                    };
                    let snap = env.step(a).unwrap();
                    let r: f32 = (*snap.reward()).into();
                    total += r;
                    if snap.is_done() {
                        break;
                    }
                }
            }
            total
        };
        let a = run();
        let b = run();
        assert!(
            (a - b).abs() < 1e-5,
            "same seed must give same rewards; got {a} vs {b}"
        );
    }

    #[test]
    fn render_styled_matches_ascii() {
        use crate::render::AsciiRenderable;

        let mut env = Blackjack::with_config(BlackjackConfig::default()).expect("valid config");
        env.reset().unwrap();
        let plain = env.render_ascii();
        let styled = env.render_styled();
        assert_eq!(styled.lines.len(), 1);
        assert_eq!(styled.plain_text(), plain);
    }

    #[test]
    fn render_styled_uses_palette_consts() {
        use crate::render::AsciiRenderable;
        use crate::render::palette::{AGENT_FG, AGENT_MODIFIER};

        let mut env = Blackjack::with_config(BlackjackConfig::default()).expect("valid config");
        env.reset().unwrap();
        let styled = env.render_styled();
        let label = styled.lines[0]
            .spans
            .iter()
            .find(|s| s.text == "Blackjack")
            .expect("Blackjack label span present");
        assert_eq!(label.style.fg, Some(AGENT_FG));
        assert!(label.style.modifier.contains(AGENT_MODIFIER));
    }

    #[test]
    fn render_ascii_within_width_budget() {
        use crate::render::AsciiRenderable;

        let mut env = Blackjack::with_config(BlackjackConfig::default()).expect("valid config");
        env.reset().unwrap();
        for line in env.render_ascii().lines() {
            assert!(
                line.chars().count() <= 80,
                "line exceeds 80 cols: {line:?} ({} chars)",
                line.chars().count()
            );
        }
    }

    // ── post-terminal step guard ─────────────────────────────────────────────
    //
    // A player sitting on a hard 21 busts on *any* card: the total lands in
    // [22, 31], and an ace drawn there counts as 1 (11 would put it at 32+), so
    // these fixtures terminate the episode without searching for a seed.

    /// Drives `env` to a bust on `Hit`, returning the terminal snapshot.
    fn bust_on_hit(env: &mut Blackjack) -> SnapshotBase<1, BlackjackObservation, ScalarReward> {
        env.reset().expect("reset must succeed");
        env.set_hands(vec![10, 7, 4], vec![6, 5]); // hard 21 — any card busts it
        env.step(BlackjackAction::Hit)
            .expect("the first hit must be accepted")
    }

    #[test]
    /// Verifies a `Hit` replayed after a bust is rejected with `StepAfterEpisodeEnd`.
    fn test_blackjack_step_rejects_post_terminal_hit_after_bust() {
        let mut env = make_env();
        assert_rejects_post_terminal_step(&mut env, bust_on_hit, BlackjackAction::Hit);
    }

    #[test]
    /// Verifies a rejected post-terminal `Hit` draws no card and leaves the hand intact.
    fn test_blackjack_step_leaves_hand_untouched_when_rejected() {
        let mut env = make_env();
        let snapshot = bust_on_hit(&mut env);
        assert!(
            snapshot.is_done(),
            "hitting a hard 21 must bust and terminate the episode"
        );

        let hand_len = env.state.player_hand.len();
        let player_sum = env.state.player_sum;

        let err = env
            .step(BlackjackAction::Hit)
            .expect_err("a Hit after the bust must be rejected, not dealt");
        assert!(
            matches!(
                err,
                EnvironmentError::StepAfterEpisodeEnd {
                    status: EpisodeStatus::Terminated
                }
            ),
            "a post-terminal Hit must fail with StepAfterEpisodeEnd{{Terminated}}, got {err:?}"
        );
        assert_eq!(
            env.state.player_hand.len(),
            hand_len,
            "a rejected Hit must not push a card onto the player's hand"
        );
        assert_eq!(
            env.state.player_sum, player_sum,
            "a rejected Hit must not change the player's total"
        );
    }

    #[test]
    /// Verifies a `Stick` replayed after the hand settled is rejected rather than re-adjudicated.
    fn test_blackjack_step_rejects_post_terminal_stick_after_settlement() {
        let mut env = make_env();
        assert_rejects_post_terminal_step(
            &mut env,
            |env| {
                env.reset().expect("reset must succeed");
                // Player 18, dealer 18 (10+8, no draw needed) → settles as a push.
                env.set_hands(vec![9, 9], vec![10, 8]);
                env.step(BlackjackAction::Stick)
                    .expect("the first stick must settle the hand")
            },
            BlackjackAction::Stick,
        );
    }

    #[test]
    /// Verifies `reset()` re-opens an environment whose episode had already terminated.
    fn test_blackjack_reset_reopens_terminated_episode() {
        let mut env = make_env();
        let snapshot = bust_on_hit(&mut env);
        assert!(snapshot.is_done(), "the episode must have terminated");
        assert!(
            env.step(BlackjackAction::Hit).is_err(),
            "the guard must be closed before reset()"
        );

        env.reset()
            .expect("reset must succeed after a terminal episode");
        env.step(BlackjackAction::Hit)
            .expect("reset() must re-open the environment for a new episode");
        assert_eq!(
            env.state.player_hand.len(),
            3,
            "the re-opened episode deals two cards and the accepted Hit adds a third"
        );
    }

    /// One entry per emitted snapshot: `(player_hand, dealer_hand, reward_bits, done)`.
    ///
    /// The reward is kept as raw bits so the comparison is exact — this trace is asserted to be
    /// bit-identical, not approximately equal.
    type BlackjackTrace = Vec<(Vec<u8>, Vec<u8>, u32, bool)>;

    /// Replays a fixed "hit below 18" policy over several episodes, recording everything the RNG
    /// touches: both hands (every card dealt, in order), the reward, and the terminal flag.
    ///
    /// Every card in the trace comes from `env.rng`, so two environments produce the same trace
    /// **iff** their RNG streams are at the same offset. That is what makes this a stream probe
    /// rather than a state probe.
    fn replay_trace(env: &mut Blackjack) -> BlackjackTrace {
        let mut trace = BlackjackTrace::new();
        for _ in 0..8 {
            env.reset().expect("reset must succeed");
            loop {
                // A pure function of the observable state: the policy itself draws no randomness,
                // so any divergence between two traces comes from the RNG stream alone.
                let action = if env.state.player_sum < 18 {
                    BlackjackAction::Hit
                } else {
                    BlackjackAction::Stick
                };
                let snapshot = env
                    .step(action)
                    .expect("a running episode must accept a step");
                let reward: f32 = (*snapshot.reward()).into();
                trace.push((
                    env.state.player_hand.clone(),
                    env.state.dealer_hand.clone(),
                    reward.to_bits(),
                    snapshot.is_done(),
                ));
                if snapshot.is_done() {
                    break;
                }
            }
        }
        trace
    }

    #[test]
    /// Verifies a rejected post-terminal step draws no card, leaving the RNG stream untouched.
    ///
    /// `Hit` calls `draw_card(&mut self.rng)` as its very first act, so a guard checked *after*
    /// the draw would still leave the hand intact (the card is never pushed) while silently
    /// advancing the shared stream. Determinism would then depend on how many illegal steps a
    /// caller made, breaking seed→trajectory reproducibility (ADR 0029, `rules.md` §8). Hand-level
    /// assertions cannot see that; only replaying the stream can.
    ///
    /// `StdRng` is not `Clone`, so the baseline is a second environment built from the same seed
    /// and driven identically — never probed.
    fn test_blackjack_rejected_post_terminal_step_does_not_advance_rng() {
        // Same seed, same drive: both streams sit at the same offset (4 cards from the deal,
        // 1 from the busting Hit).
        let mut probed = make_env();
        let mut clean = make_env();

        for env in [&mut probed, &mut clean] {
            let terminal = bust_on_hit(env);
            assert!(
                terminal.is_done(),
                "hitting a hard 21 must bust and terminate the episode"
            );
        }

        // `probed` alone attempts (and is denied) several post-terminal `Hit`s — the action whose
        // first statement draws a card.
        for attempt in 0..3 {
            let err = probed
                .step(BlackjackAction::Hit)
                .expect_err("a post-terminal Hit must be rejected, not dealt");
            assert!(
                matches!(
                    err,
                    EnvironmentError::StepAfterEpisodeEnd {
                        status: EpisodeStatus::Terminated
                    }
                ),
                "post-terminal Hit #{attempt} must fail with StepAfterEpisodeEnd{{Terminated}}, got {err:?}"
            );
        }

        assert_eq!(
            replay_trace(&mut probed),
            replay_trace(&mut clean),
            "a rejected step must draw no card; a probed env must replay a bit-identical \
             trajectory (same cards, same rewards, same terminals) to one that never saw the step"
        );
    }

    #[test]
    /// Verifies `hand_value` saturates instead of overflowing on a hand whose pip total exceeds
    /// `u8::MAX`, and still classifies it as a bust.
    fn test_hand_value_saturates_on_overflowing_hand() {
        // 30 ten-valued cards sum to 300 — an overflow of the old `u8` accumulator
        // (panic in debug; a wraparound to 44 in release).
        let hand = vec![10_u8; 30];
        let (sum, usable_ace) = hand_value(&hand);
        assert_eq!(
            sum,
            u8::MAX,
            "a hand totalling more than u8::MAX must saturate, not wrap"
        );
        assert!(
            sum > 21,
            "a saturated total must still be classified as a bust, got {sum}"
        );
        assert!(
            !usable_ace,
            "an aceless hand can never report a usable ace, saturated or not"
        );
    }

    #[test]
    /// Verifies the usable-ace rule survives the widened accumulator for ordinary hands.
    fn test_hand_value_counts_usable_ace_as_eleven() {
        assert_eq!(
            hand_value(&[1, 6]),
            (17, true),
            "an ace with 6 counts as 11 (soft 17)"
        );
        assert_eq!(
            hand_value(&[1, 6, 10]),
            (17, false),
            "the same ace drops to 1 once counting it as 11 would bust the hand"
        );
        assert_eq!(
            hand_value(&[10, 7, 4]),
            (21, false),
            "an aceless hand totals at face value"
        );
    }
}
