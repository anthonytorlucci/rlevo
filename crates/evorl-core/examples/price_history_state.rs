use evorl_core::state::{FlattenedState, State, StateError, TemporalState};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PriceHistory {
    prices: Vec<u32>, // Last N price observations
}

impl State for PriceHistory {
    fn numel(&self) -> usize {
        self.prices.len()
    }

    fn shape(&self) -> Vec<usize> {
        vec![self.prices.len()]
    }
}

impl FlattenedState for PriceHistory {
    fn flatten(&self) -> Vec<f32> {
        self.prices.iter().map(|&p| p as f32).collect()
    }

    fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
        Ok(PriceHistory {
            prices: data.iter().map(|&p| p as u32).collect(),
        })
    }
}

impl TemporalState for PriceHistory {
    fn sequence_length(&self) -> usize {
        self.prices.len()
    }

    fn latest(&self) -> &[f32] {
        // Return the most recent price as a slice
        // (In practice, you'd maintain a separate f32 buffer)
        &[]
    }

    fn push_pop(&self, new_observation: &[f32]) -> Result<Self, StateError> {
        if new_observation.len() != 1 {
            return Err(StateError::InvalidSize {
                expected: 1,
                got: new_observation.len(),
            });
        }
        let mut new_prices = self.prices.clone();
        new_prices.remove(0); // Remove oldest
        new_prices.push(new_observation[0] as u32); // Add newest
        Ok(PriceHistory { prices: new_prices })
    }
}

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1.  Create an initial price history (e.g., three recent prices)
    let history = PriceHistory {
        prices: vec![10, 12, 13],
    };
    println!("Initial history: {:?}", history.prices);

    // 2.  Flatten the state to a Vec<f32>
    let flattened = history.flatten();
    println!("Flattened representation: {:?}", flattened);

    // 3.  Push a new observation (e.g., price = 15)
    let new_price: Vec<f32> = vec![15.0];
    let updated_history = history.push_pop(&new_price)?;
    println!(
        "After pushing a new price, the history is: {:?}",
        updated_history.prices
    );

    // 4.  Verify the sequence length (should still be 3)
    println!(
        "Sequence length after update: {}",
        updated_history.sequence_length()
    );

    // 5.  (Optional) Re‑create the object from its flattened form
    let recovered: PriceHistory = FlattenedState::from_flattened(flattened)?;
    println!("Recovered from flattened: {:?}", recovered.prices);

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
