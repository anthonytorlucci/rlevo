use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use evorl_core::state::{FlattenedState, State, StateError, StateTensorConvertible};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GridPosition {
    x: i32,
    y: i32,
    max_x: i32,
    max_y: i32,
}

impl State for GridPosition {
    fn is_valid(&self) -> bool {
        self.x >= 0 && self.y >= 0 && self.x < self.max_x && self.y < self.max_y
    }

    fn numel(&self) -> usize {
        2 // x and y coordinates
    }

    fn shape(&self) -> Vec<usize> {
        vec![2] // flat 1D representation
    }
}

impl FlattenedState for GridPosition {
    fn flatten(&self) -> Vec<f32> {
        vec![
            self.x as f32,
            self.y as f32,
            self.max_x as f32,
            self.max_y as f32,
        ]
    }

    fn from_flattened(data: Vec<f32>) -> Result<Self, StateError> {
        if data.len() != 4 {
            return Err(StateError::InvalidSize {
                expected: 4,
                got: data.len(),
            });
        }
        Ok(GridPosition {
            x: data[0] as i32,
            y: data[1] as i32,
            max_x: data[2] as i32,
            max_y: data[3] as i32,
        })
    }
}

impl<const R: usize> StateTensorConvertible<R> for GridPosition {
    fn to_tensor<B: Backend>(&self, device: &B::Device) -> Tensor<B, R> {
        let data = self.flatten();
        Tensor::from_floats(data.as_slice(), device)
    }

    fn from_tensor<B: Backend>(tensor: &Tensor<B, R>) -> Result<Self, StateError> {
        let data = tensor.to_data().to_vec::<f32>().unwrap();
        Self::from_flattened(data)
    }
}

// --------------------------------------------------------------------------
// Example usage
// --------------------------------------------------------------------------
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // -----------------------------------------------------------------------
    // 1. Create a few valid GridPosition instances
    // -----------------------------------------------------------------------
    let pos1 = GridPosition {
        x: 3,
        y: 7,
        max_x: 10,
        max_y: 10,
    };
    let pos2 = GridPosition {
        x: 0,
        y: 0,
        max_x: 10,
        max_y: 10,
    };
    let pos3 = GridPosition {
        x: 9,
        y: 9,
        max_x: 10,
        max_y: 10,
    };

    // -----------------------------------------------------------------------
    // 2. Validate them using the `State` implementation
    // -----------------------------------------------------------------------
    println!("pos1 is valid? {}", pos1.is_valid()); // true
    println!("pos2 is valid? {}", pos2.is_valid()); // true
    println!("pos3 is valid? {}", pos3.is_valid()); // true

    // -----------------------------------------------------------------------
    // 3. Flatten the positions
    // -----------------------------------------------------------------------
    let flat1 = pos1.flatten();
    let flat2 = pos2.flatten();
    let flat3 = pos3.flatten();

    println!("flattened pos1: {:?}", flat1); // [3.0, 7.0, 10.0, 10.0]
    println!("flattened pos2: {:?}", flat2); // [0.0, 0.0, 10.0, 10.0]
    println!("flattened pos3: {:?}", flat3); // [9.0, 9.0, 10.0, 10.0]

    // -----------------------------------------------------------------------
    // 4. Re‑construct from the flattened representation
    // -----------------------------------------------------------------------
    // Successful reconstruction
    let recovered1 = GridPosition::from_flattened(flat1.clone()).unwrap();
    let recovered2 = GridPosition::from_flattened(flat2.clone()).unwrap();
    let recovered3 = GridPosition::from_flattened(flat3.clone()).unwrap();

    println!("recovered pos1: {:?}", recovered1);
    println!("recovered pos2: {:?}", recovered2);
    println!("recovered pos3: {:?}", recovered3);

    // -----------------------------------------------------------------------
    // 5. Demonstrate error handling when the flattened data is malformed
    // -----------------------------------------------------------------------
    let malformed = vec![1.0]; // only one element instead of four
    match GridPosition::from_flattened(malformed) {
        Ok(_) => println!("unexpected success"),
        Err(StateError::InvalidSize { expected, got }) => {
            println!(
                "failed to reconstruct: expected {} elements, got {}",
                expected, got
            );
        }
        Err(e) => println!("unexpected error: {:?}", e),
    }

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
