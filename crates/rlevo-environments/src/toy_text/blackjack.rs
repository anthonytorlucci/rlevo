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
//! ## RNG behaviour
//!
//! The RNG advances continuously across episodes — `reset()` does **not** reseed. Two instances
//! with the same seed produce identical trajectories.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rlevo_core::action::DiscreteAction;
use rlevo_core::base::{Action, Observation, State};
use rlevo_core::environment::{Environment, EnvironmentError, SnapshotBase};
use rlevo_core::reward::ScalarReward;
use serde::{Deserialize, Serialize};

// ── card helpers ──────────────────────────────────────────────────────────────

fn draw_card(rng: &mut StdRng) -> u8 {
    rng.random_range(1u8..=13).min(10)
}

fn hand_value(hand: &[u8]) -> (u8, bool) {
    let sum: u8 = hand.iter().sum();
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
#[derive(Debug)]
pub struct Blackjack {
    state: BlackjackState,
    config: BlackjackConfig,
    rng: StdRng,
}

impl Blackjack {
    /// Create a Blackjack environment with the given configuration.
    pub fn with_config(config: BlackjackConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.seed);
        Self {
            state: BlackjackState {
                player_sum: 0,
                dealer_showing: 1,
                usable_ace: false,
                player_hand: Vec::new(),
                dealer_hand: Vec::new(),
            },
            config,
            rng,
        }
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

impl Environment<1, 1, 1> for Blackjack {
    type StateType = BlackjackState;
    type ObservationType = BlackjackObservation;
    type ActionType = BlackjackAction;
    type RewardType = ScalarReward;
    type SnapshotType = SnapshotBase<1, BlackjackObservation, ScalarReward>;

    fn new(_render: bool) -> Self {
        Self::with_config(BlackjackConfig::default())
    }

    fn reset(&mut self) -> Result<Self::SnapshotType, EnvironmentError> {
        self.deal_initial();
        Ok(SnapshotBase::running(
            self.state.observe(),
            ScalarReward(0.0),
        ))
    }

    fn step(&mut self, action: BlackjackAction) -> Result<Self::SnapshotType, EnvironmentError> {
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
        if done {
            Ok(SnapshotBase::terminated(obs, ScalarReward(reward)))
        } else {
            Ok(SnapshotBase::running(obs, ScalarReward(reward)))
        }
    }
}

#[cfg(test)]
/// Unit tests for [`Blackjack`], covering actions, observations, rewards, and RNG determinism.
mod tests {
    use super::*;
    use rlevo_core::action::DiscreteAction;
    use rlevo_core::base::Observation;
    use rlevo_core::environment::Snapshot;

    fn make_env() -> Blackjack {
        Blackjack::with_config(BlackjackConfig::default())
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
        let mut env = Blackjack::with_config(cfg);
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
        let mut env = Blackjack::with_config(cfg);
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
        let mut env = Blackjack::with_config(cfg);
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
            let mut env = Blackjack::with_config(cfg.clone());
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
}
