//! Step 6: does the learner train correctly, and beat a random baseline?
//!
//! Three checks on one step line. The trial passes every step's status to
//! the learner unchanged, and the training run holds both endings. The
//! learner's evaluation return beats a uniform random agent's under the same
//! configuration. A second run repeats the first bit for bit.

use std::sync::{Arc, Mutex, PoisonError};

use rlevo::core::fitness::Metric;
use rlevo::harness::evaluator::Evaluator;
use rlevo::harness::metrics::core::{RETURN_MEAN, STEPS_PER_SEC, WALL_CLOCK_SECONDS};
use rlevo::harness::report::{BenchmarkReport, TrialReport};
use rlevo::harness::reporter::logging::LoggingReporter;
use rlevo::harness::suite::Suite;
use rlevo::prelude::{
    Agent, DiscreteAction, Environment, EnvironmentError, EpisodeStatus, Rng, RngExt, Snapshot,
    Transition,
};

use crate::frozen_lake::{CAP, ENV_ID, lake, learner};
use crate::learn::{TRAIN_STEPS, config, run_learner};

/// Training episodes of each ending the run must hold, so the status check
/// is not one episode from vacuous.
const MIN_EACH: usize = 5;

/// What the two wrappers saw, in the order it happened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Event {
    /// The environment reset.
    Reset,
    /// The environment returned a step with this status.
    Stepped(EpisodeStatus),
    /// The agent was shown a transition with this status.
    Shown(EpisodeStatus),
    /// The agent was told its episode ended with this status.
    Ended(EpisodeStatus),
    /// The agent was told it is training (`true`) or measured.
    Training(bool),
}

type Log = Arc<Mutex<Vec<Event>>>;

fn push(log: &Log, event: Event) {
    log.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(event);
}

/// The environment, outermost in the factory, logging what it returns. An
/// `Environment` by delegation, as an author's own wrapper would be.
struct StatusLog<E> {
    inner: E,
    log: Log,
}

impl<E: Environment<1, 1>> Environment<1, 1> for StatusLog<E> {
    type ObservationType = E::ObservationType;
    type ActionType = E::ActionType;

    fn reset(&mut self) -> Result<Snapshot<Self::ObservationType>, EnvironmentError> {
        let snapshot = self.inner.reset()?;
        push(&self.log, Event::Reset);
        Ok(snapshot)
    }

    fn step(
        &mut self,
        action: Self::ActionType,
    ) -> Result<Snapshot<Self::ObservationType>, EnvironmentError> {
        let snapshot = self.inner.step(action)?;
        push(&self.log, Event::Stepped(snapshot.status));
        Ok(snapshot)
    }
}

/// The agent, logging every hook the trial calls before passing it on.
struct Watched<G> {
    inner: G,
    log: Log,
}

impl<O, A, G: Agent<O, A>> Agent<O, A> for Watched<G> {
    fn act(&mut self, obs: &O, rng: &mut dyn Rng) -> A {
        self.inner.act(obs, rng)
    }

    fn on_transition(&mut self, t: Transition<&O, &A>, rng: &mut dyn Rng) {
        push(&self.log, Event::Shown(t.status));
        self.inner.on_transition(t, rng);
    }

    fn on_episode_end(&mut self, status: EpisodeStatus) {
        push(&self.log, Event::Ended(status));
        self.inner.on_episode_end(status);
    }

    fn set_training(&mut self, training: bool) {
        push(&self.log, Event::Training(training));
        self.inner.set_training(training);
    }

    fn emit_metrics(&self) -> Vec<Metric> {
        self.inner.emit_metrics()
    }
}

/// The baseline: a uniform choice among the actions, from the trial's RNG.
/// Its hooks are the defaults, so its training phase learns nothing.
struct Uniform;

impl<O, A: DiscreteAction<1>> Agent<O, A> for Uniform {
    fn act(&mut self, _obs: &O, rng: &mut dyn Rng) -> A {
        A::from_index(rng.random_range(0..A::ACTION_COUNT))
    }
}

/// Run step 6 and report it.
pub fn run(report: impl Fn(&str, Result<String, String>)) {
    report("baseline", baseline());
}

fn baseline() -> Result<String, String> {
    let endings = statuses()?;
    let first = run_learner();
    let learned = return_mean(&first)?;
    let suite = Suite::new("byoa-1a").with_env(ENV_ID, lake);
    let random = return_mean(&Evaluator::new(config()).run_suite(
        &suite,
        |_seed| Uniform,
        &mut LoggingReporter::new(),
    ))?;
    if learned <= random {
        return Err(format!(
            "the learner's return/mean {learned} does not beat the random agent's {random} \
             under one configuration"
        ));
    }
    let compared = repeats(only_trial(&first)?, only_trial(&run_learner())?)?;
    Ok(format!(
        "every status reached the learner unchanged ({} transitions; training episodes \
         ended {} Terminated, {} Truncated at the cap of {CAP}); return/mean {learned:.3} \
         against the random agent's {random:.3}; a second run repeats {compared} values bit \
         for bit, the wall clock aside",
        endings.transitions, endings.terminated, endings.truncated
    ))
}

/// The run's one trial, which must not have errored.
fn only_trial(report: &BenchmarkReport) -> Result<&TrialReport, String> {
    let [trial] = report.trials.as_slice() else {
        return Err(format!(
            "expected 1 trial, the report holds {}",
            report.trials.len()
        ));
    };
    if trial.errored {
        return Err(format!(
            "the trial errored: {}",
            trial.error_message.as_deref().unwrap_or("no message")
        ));
    }
    Ok(trial)
}

fn return_mean(report: &BenchmarkReport) -> Result<f64, String> {
    only_trial(report)?
        .scalars
        .get(RETURN_MEAN)
        .copied()
        .ok_or_else(|| format!("the trial reports no `{RETURN_MEAN}`"))
}

/// What the training phase showed the learner, read from the log.
struct Endings {
    transitions: usize,
    terminated: usize,
    truncated: usize,
}

/// The learner, wrapped, through the same run: then the log is read back.
fn statuses() -> Result<Endings, String> {
    let log = Log::default();
    let env_log = Arc::clone(&log);
    let suite = Suite::new("byoa-1a").with_env(ENV_ID, move |seed| StatusLog {
        inner: lake(seed),
        log: Arc::clone(&env_log),
    });
    let agent_log = Arc::clone(&log);
    let report = Evaluator::new(config()).run_suite(
        &suite,
        move |_seed| Watched {
            inner: learner(),
            log: Arc::clone(&agent_log),
        },
        &mut LoggingReporter::new(),
    );
    only_trial(&report)?;
    let events = log.lock().unwrap_or_else(PoisonError::into_inner).clone();
    read_statuses(&events, config().max_steps)
}

/// Check the order of what the environment returned and what the learner
/// was told.
///
/// Each step the environment returns is shown to the learner next, with the
/// same status. An episode the environment ended is ended for the learner
/// with that status, before anything else happens. An episode that ends
/// while its last step is `Running` is the trial's cut at `max_steps`, and
/// ends `Truncated`. `set_training(true)` comes first, and
/// `set_training(false)` after exactly the training budget's transitions.
fn read_statuses(events: &[Event], max_steps: usize) -> Result<Endings, String> {
    let mut endings = Endings {
        transitions: 0,
        terminated: 0,
        truncated: 0,
    };
    let mut training: Option<bool> = None;
    let mut trained = 0;
    // A step the environment returned that the learner has not been shown.
    let mut unshown: Option<EpisodeStatus> = None;
    // The status of the open episode's last transition.
    let mut last: Option<EpisodeStatus> = None;
    let mut length = 0;
    for (at, event) in events.iter().enumerate() {
        let owed = last.filter(|status| status.is_done());
        if let (Some(status), false) = (owed, matches!(event, Event::Ended(_))) {
            return Err(format!(
                "event {at}: the episode ended {status:?} and the learner was not told before \
                 {event:?}"
            ));
        }
        if unshown.is_some() && !matches!(event, Event::Shown(_)) {
            return Err(format!(
                "event {at}: a step the environment returned was not shown to the learner \
                 before {event:?}"
            ));
        }
        match *event {
            Event::Training(true) => {
                if training.is_some() {
                    return Err(format!(
                        "event {at}: set_training(true) after the run began"
                    ));
                }
                training = Some(true);
            }
            Event::Training(false) => {
                if training != Some(true) || trained != TRAIN_STEPS {
                    return Err(format!(
                        "event {at}: set_training(false) after {trained} training \
                         transitions, not {TRAIN_STEPS}"
                    ));
                }
                training = Some(false);
                // The budget may leave its last episode open.
                last = None;
            }
            Event::Reset => {
                if last.is_some() {
                    return Err(format!("event {at}: a reset before the episode ended"));
                }
                length = 0;
            }
            Event::Stepped(status) => unshown = Some(status),
            Event::Shown(status) => {
                let Some(returned) = unshown.take() else {
                    return Err(format!(
                        "event {at}: the learner was shown a step the environment did not return"
                    ));
                };
                if status != returned {
                    return Err(format!(
                        "event {at}: the environment returned {returned:?} and the learner was \
                         shown {status:?}"
                    ));
                }
                match training {
                    Some(true) => trained += 1,
                    Some(false) => {}
                    None => return Err(format!("event {at}: a transition before training")),
                }
                endings.transitions += 1;
                length += 1;
                last = Some(status);
            }
            Event::Ended(status) => {
                let Some(final_step) = last.take() else {
                    return Err(format!(
                        "event {at}: the learner was told an episode ended {status:?} with \
                         no step since the last end"
                    ));
                };
                let due = if final_step.is_done() {
                    final_step
                } else if length == max_steps {
                    EpisodeStatus::Truncated
                } else {
                    return Err(format!(
                        "event {at}: the learner was told its episode ended {status:?} after \
                         a Running step, {length} steps in, short of the trial's {max_steps}"
                    ));
                };
                if status != due {
                    return Err(format!(
                        "event {at}: the episode's last step was {final_step:?} and the \
                         learner was told it ended {status:?}"
                    ));
                }
                if training == Some(true) {
                    match status {
                        EpisodeStatus::Terminated => endings.terminated += 1,
                        EpisodeStatus::Truncated => endings.truncated += 1,
                        EpisodeStatus::Running => {}
                    }
                }
            }
        }
    }
    if training != Some(false) {
        return Err("the trial never switched the learner to evaluation".to_string());
    }
    if last.is_some() || unshown.is_some() {
        return Err("the run ended inside an episode".to_string());
    }
    if endings.terminated < MIN_EACH || endings.truncated < MIN_EACH {
        return Err(format!(
            "training ended {} episodes Terminated and {} Truncated; the check needs {MIN_EACH} \
             of each, so the TimeLimit cap of {CAP} must change",
            endings.terminated, endings.truncated
        ));
    }
    Ok(endings)
}

/// Two runs of the learner, value by value: every scalar but the two the
/// wall clock sets, every counter, and every evaluation episode.
fn repeats(first: &TrialReport, second: &TrialReport) -> Result<usize, String> {
    let untimed =
        |key: &&String| key.as_str() != WALL_CLOCK_SECONDS && key.as_str() != STEPS_PER_SEC;
    let keys = |trial: &TrialReport| {
        trial
            .scalars
            .keys()
            .filter(untimed)
            .cloned()
            .collect::<Vec<_>>()
    };
    if keys(first) != keys(second) {
        return Err("the second run reports other scalars".to_string());
    }
    let mut compared = 0;
    for key in keys(first) {
        let (a, b) = (first.scalars[&key], second.scalars[&key]);
        if a.to_bits() != b.to_bits() {
            return Err(format!("`{key}` is {a} and then {b}"));
        }
        compared += 1;
    }
    if first.counters != second.counters {
        return Err(format!(
            "the counters differ: {:?} and then {:?}",
            first.counters, second.counters
        ));
    }
    compared += first.counters.len();
    if first.episodes.len() != second.episodes.len() {
        return Err("the second run has another number of episodes".to_string());
    }
    for (a, b) in first.episodes.iter().zip(&second.episodes) {
        if a.return_value.to_bits() != b.return_value.to_bits() || a.length != b.length {
            return Err(format!(
                "evaluation episode {} returned {} in {} steps, and then {} in {}",
                a.episode_idx, a.return_value, a.length, b.return_value, b.length
            ));
        }
        compared += 2;
    }
    Ok(compared)
}
