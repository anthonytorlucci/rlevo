use evorl_core::state::State;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BoundedPosition {
    x: i32,
    y: i32,
    max_x: i32,
    max_y: i32,
}

impl State for BoundedPosition {
    fn is_valid(&self) -> bool {
        self.x >= 0 && self.x < self.max_x && self.y >= 0 && self.y < self.max_y
    }

    fn numel(&self) -> usize {
        2
    }
    fn shape(&self) -> Vec<usize> {
        vec![2]
    }
}

fn main() {
    //
    let valid = BoundedPosition {
        x: 5,
        y: 3,
        max_x: 10,
        max_y: 10,
    };
    assert!(valid.is_valid());
    //
    let invalid = BoundedPosition {
        x: 15,
        y: 3,
        max_x: 10,
        max_y: 10,
    };
    assert!(!invalid.is_valid());
}
