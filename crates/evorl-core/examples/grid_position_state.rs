use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use evorl_core::base::TensorConvertible;
use evorl_core::state::{State, StateError};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GridPosition {
    x: i32,
    y: i32,
    max_x: i32,
    max_y: i32,
}

// impl State for GridPosition {
//     fn is_valid(&self) -> bool {
//         self.x >= 0 && self.y >= 0 && self.x < self.max_x && self.y < self.max_y
//     }

//     fn numel(&self) -> usize {
//         2 // x and y coordinates
//     }

//     fn shape(&self) -> Vec<usize> {
//         vec![2] // flat 1D representation
//     }
// }

impl<const R: usize, B: Backend> TensorConvertible<R, B> for GridPosition {
    fn to_tensor(&self, device: &B::Device) -> Tensor<B, R> {
        todo!("Implement conversion to tensor")
    }

    // fn from_tensor(tensor: &Tensor<B, R>) -> Result<Self, StateError> {
    //     todo!("Implement from tensor conversion")
    // }
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
    // println!("pos1 is valid? {}", pos1.is_valid()); // true
    // println!("pos2 is valid? {}", pos2.is_valid()); // true
    // println!("pos3 is valid? {}", pos3.is_valid()); // true

    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║                   Example Complete                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    Ok(())
}
