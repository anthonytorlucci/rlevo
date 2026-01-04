use crate::action::{Action, ActionTensorConvertible};
use crate::state::{State, StateTensorConvertible};
use std::fmt::Debug;

pub trait Environment<const S: usize, const A: usize> {
    type StateType: State + StateTensorConvertible<S> + Debug + Clone;
    type ActionType: Action + ActionTensorConvertible<A> + Debug + Clone;
    type RewardType: Into<f32> + Debug + Clone;

    /// Create a new environment (render: whether to display the environment).
    fn new(render: bool) -> Self;

    /// Reset the environment and return the initial state snapshot.
    fn reset(&mut self) -> Result<Snapshot<Self, S, A>, EnvironmentError>;

    /// Take an action and return the next state, reward, and done flag.
    fn step(&mut self, action: Self::ActionType) -> Result<Snapshot<Self, S, A>, EnvironmentError>;
}

pub struct Snapshot<E, const S: usize, const A: usize>
where
    E: Environment<S, A>,
{
    pub state: E::StateType,
    pub reward: E::RewardType,
    pub done: bool,
}

impl<E, const S: usize, const A: usize> Snapshot<E, S, A>
where
    E: Environment<S, A>,
{
    pub fn new(state: E::StateType, reward: E::RewardType, done: bool) -> Self {
        Self {
            state,
            reward,
            done,
        }
    }

    pub fn state(&self) -> &E::StateType {
        &self.state
    }

    pub fn reward(&self) -> &E::RewardType {
        &self.reward
    }

    pub fn done(&self) -> bool {
        self.done
    }
}

// Custom error type for environments:
#[derive(Debug)]
pub enum EnvironmentError {
    InvalidAction(String),
    RenderFailed(String),
    // ... other errors
    IoError(std::io::Error), // Wrap a standard library error
}

// The basic implementation only requires the trait to be implemented.
// The `source` method can be used to expose the underlying cause of an error.
// When the error variant is `IoError`, we return `Some(io_err)` which provides a reference to the underlying `std::io:Error`.
// For other error variants, we return `None` as they don't wrap other errors.
impl std::error::Error for EnvironmentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            EnvironmentError::IoError(io_err) => Some(io_err),
            _ => None,
        }
    }
}

// This trait provides a user-friendly description of the error.
// I've used a match expression to format each variant of the enum appropriately.
// Each variant extracts its contained string and formats it with a descriptive prefix.
//
// This implementation allows your error to be used with standard error handling patterns like:
// fn example_usage() -> Result<(), Box<dyn std::error::Error>> {
//     // Some code that might return your error
//     Err(BurnrlEnvironmentError::InvalidAction("Action out of bounds".to_string()))?;
//
//     Ok(())
// }
//
// The `IoError` variant uses the `Display` implementation of `io_err` to format the specific IO error message.
// We add context with "IO operation failed: " to make it clear what category of error occured.
// The `source` method is particularly useful for error chaining:
// fn example_io_operation() -> Result<(), BurnrlEnvironmentError> {
//     // This might fail with an io::Error which we wrap
//     std::fs::File::open("nonexistent_file.txt")
//         .map_err(|e| BurnRLEnvironmentError::IoError(e))
// }

// fn handle_error() {
//     match example_io_operation() {
//         Err(e) => {
//             println!("Error: {}", e); // Displays: "IO operation failed: No such file or directory (os error 2)"

//             // Using source to get the underlying error
//             let source = e.source().unwrap();
//             println!("Source: {}", source); // Displays: "No such file or directory (os error 2)"
//         }
//         Ok(_) => println!("Operation succeeded"),
//     }
// }
//
// This approach provides a nice separation between the high-level context (what operation failed) and the low-level details (the specific reason it failed).
impl std::fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvironmentError::InvalidAction(action_error) => {
                write!(f, "Invalid action: {}", action_error)
            }
            EnvironmentError::RenderFailed(render_error) => {
                write!(f, "Render failed: {}", render_error)
            }
            EnvironmentError::IoError(io_err) => {
                // Write the specific error context
                write!(f, "IO operation failed: {}", io_err)
            }
        }
    }
}

// Now you can use the `?` operator directly:
// fn example_with_from() -> Result<(), BurnrlEnvironmentError> {
//     std::fs::File::open("nonexistent_file.txt")?; // Automatically converts io::Error
//     Ok(())
// }
impl From<std::io::Error> for EnvironmentError {
    fn from(error: std::io::Error) -> Self {
        EnvironmentError::IoError(error)
    }
}
