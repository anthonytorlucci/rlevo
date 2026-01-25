//! Chess move representation and action encoding modeled after AlphaZero Chess.
//!
//! This module defines the core data structures and traits for representing chess moves
//! in a way suitable for reinforcement learning agents. It provides a dense, efficient
//! encoding scheme that maps moves to discrete action indices for use with neural networks.
//!
//! AlphaZero represents the action space of chess as an $8 \times 8 \times 73$ tensor,
//! totaling 4,672 possible move slots.
//!
//! Because a neural network requires a fixed-size output, AlphaZero must provide a value
//! for every possible move a piece could technically make on an $8 \times 8$ board, even
//! if that move is illegal in the current position.
//!
//! # The Action Space ($8 \times 8 \times 73$)
//! The output is structured such that the first two dimensions ($8 \times 8$) represent the
//! square from which a piece is moved. The third dimension (73 planes) represents the type of
//! move being made.
//!
//! **The 73 "Move Type" Planes**:
//! - **56 Queen-like Moves**: These covers moves in 8 directions (N, NE, E, SE, S, SW, W, NW).
//!   For each direction, there are 7 possible distances (1 to 7 squares).
//!   - _Note_: If a pawn moves to the black rank and promotes to a Queen, it is represented in
//!     these planes.
//! - **8 Knight Moves**: These represent the 8 "L" shapes a knight can jump.
//! - **9 Underpromotion Planes**: When a pawn reaches the 8th rank, it can promote to a Knight,
//!   Bishop, or Rook. These 9 planes cover the 3 possible piece types $\times 3$ possible "exit"
//!   directions (capture left, move straight, capture right).
//!
//! **How illegal moves are handled**: The network outputs probabilites for all 4,672 moves.
//! However, before the move is actually selected, AlphaZero applies a mask. It sets the
//! probability of all illegal moves to zero and renormalizes the remaining legal moves so they
//! sum to 1.
//!
//! # How the Network Evaluates a Position
//! AlphaZero uses a "dual-head" architecture. After the input (the $8 \times 8 \times 119$
//! tensor) passes through the main body of the residual layers, the network splits into two
//! distinct output heads:
//!
//! **The Policy Head ($\pi$)**
//! The policy head outputs the $8 \times 8 \times 73$ tensor described above. This is a
//! probability distribution over all possible moves.
//! - **Purpose**: It acts as the "intuition". It tells the system which moves look the most
//!   promising to investigate further.
//! - **Training**: It is trained to match the move distributions found by the search tree during
//!   self-play.
//!
//! **The Value Head ($v$)**
//! The value head outputs a single scalar value between -1 and 1.
//! - **Interpretation**:
//!   - $+1$: High confidence of a win.
//!   - $0$: Prediction of a draw.
//!   - $-1$: High confidence of a loss.
//! - **Purpose**: It replaces the traditional "evaluation function" (which might count points for
//!   pieces). Instead of calculating a score like $+1.5$, it predicts the expected outcome of the
//!   game from the current position.
//!
//! # The Synergy: Evaluation + Search
//! The magic of AlphaZero is how these two outputs work together inside the Monte Carlo Tree
//! Search (MCTS):
//! 1. **Prioritization**: When MCTS explores a new branch, it uses the Policy ($\pi$) to decide
//!   which moves to try first. This prevents the engine from wasting time on "human-obvious"
//!   blunders.
//! 2. **Leaf Evaluation**: In old-school MCTS, you would play "random games" until the end to see
//!   who won. AlphaZero doesn't do that. As soon as it hits a new position in its search tree, it
//!   asks the Value Head ($v$) for an estimate.
//! 3. **Backpropagation**: This value estimate is "rippled" back up the tree, updating the
//!   quality score of every move that led to that position.
//!
//!
//! # Action Space Structure
//!
//! The `ChessMove` struct implements both the `Action`, `MultiDiscreteAction`, and
//! `ActionTensorConvertible`  traits, enabling integration with the evorl-core framework:
//!
//! - **Validation**: Moves are validated by ensuring both source and destination squares
//!   are valid (0-63). Legal move validation is delegated to the environment.
//! - **Discretization**: Each move can be converted to/from the indicies in (8,8,73), enabling
//!   efficient neural network output layers and policy representations.
//!
//! # Example
//!
//! ```ignore
//! use evorl_envs::games::chess::{ChessMove, Square};
//!
//! // Create a move from e2 to e4
//! todo!
//!
//! // Convert to discrete action indices
//! let indices = move_e2_e4.to_indices();
//! assert!(todo!);  // assert the number of dimensions in indices == 3 i.e., the rank of the action space.
//!
//! // Reconstruct from indices
//! let reconstructed = ChessMove::from_indices(indices);
//! assert_eq!(move_e2_e4, reconstructed);
//! ```
//!
//! # Implementation Notes
//!
//! - All moves are immutable and copyable (`Copy` trait), suitable for rapid iteration
//!   during agent training.
//! - The fixed action space (8, 8, 73) simplifies policy networks, which output a single
//!   tensor rather than variable-length action lists.
//! - Illegal moves (e.g., moving into check) are encoded but marked as invalid by the
//!   environment. Agents learn to avoid them through reward signals.

use crate::games::chess::board::Square;
use evorl_core::action::MultiDiscreteAction;

/// Promotion piece types for pawn promotion moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromotionPiece {
    /// Promote to Queen (covered in queen-like moves for queen promotion)
    Queen,
    /// Promote to Rook (underpromotion)
    Rook,
    /// Promote to Bishop (underpromotion)
    Bishop,
    /// Promote to Knight (underpromotion)
    Knight,
}

/// Chess move representation compatible with AlphaZero's action space.
///
/// Encodes moves in an 8×8×73 tensor format where:
/// - First dimension: source rank (0-7)
/// - Second dimension: source file (0-7)
/// - Third dimension: move type plane (0-72)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChessMove {
    /// Source square (0-63, where 0 is a1 and 63 is h8)
    pub from: Square,
    /// Destination square (0-63)
    pub to: Square,
    /// Optional promotion piece (None for non-promotion moves, Some for promotions)
    pub promotion: Option<PromotionPiece>,
}

impl ChessMove {
    /// Creates a new chess move from source to destination.
    pub fn new(from: Square, to: Square) -> Self {
        Self {
            from,
            to,
            promotion: None,
        }
    }

    /// Creates a new chess move with a promotion.
    pub fn new_with_promotion(from: Square, to: Square, promotion: PromotionPiece) -> Self {
        Self {
            from,
            to,
            promotion: Some(promotion),
        }
    }

    /// Returns the source rank (0-7).
    #[inline]
    pub fn from_rank(&self) -> u8 {
        self.from.rank()
    }

    /// Returns the source file (0-7).
    #[inline]
    pub fn from_file(&self) -> u8 {
        self.from.file()
    }

    /// Returns the destination rank (0-7).
    #[inline]
    pub fn to_rank(&self) -> u8 {
        self.to.rank()
    }

    /// Returns the destination file (0-7).
    #[inline]
    pub fn to_file(&self) -> u8 {
        self.to.file()
    }

    /// Computes the move plane (0-72) for the AlphaZero action space.
    ///
    /// # Move Plane Encoding
    /// - Planes 0-55: Queen-like moves (8 directions × 7 distances)
    ///   - Directions: N, NE, E, SE, S, SW, W, NW (in that order)
    ///   - For each direction: distance 1-7
    /// - Planes 56-63: Knight moves (8 L-shapes)
    /// - Planes 64-72: Underpromotions (3 pieces × 3 directions)
    ///   - Pieces: Knight, Bishop, Rook
    ///   - Directions: left-capture, forward, right-capture
    fn compute_move_plane(&self) -> usize {
        let from_rank = self.from_rank() as i8;
        let from_file = self.from_file() as i8;
        let to_rank = self.to_rank() as i8;
        let to_file = self.to_file() as i8;

        let delta_rank = to_rank - from_rank;
        let delta_file = to_file - from_file;

        // Check for knight moves (planes 56-63)
        let knight_moves = [
            (2, 1),   // Plane 56
            (1, 2),   // Plane 57
            (-1, 2),  // Plane 58
            (-2, 1),  // Plane 59
            (-2, -1), // Plane 60
            (-1, -2), // Plane 61
            (1, -2),  // Plane 62
            (2, -1),  // Plane 63
        ];

        for (i, &(dr, df)) in knight_moves.iter().enumerate() {
            if delta_rank == dr && delta_file == df {
                return 56 + i;
            }
        }

        // Check for underpromotions (planes 64-72)
        if let Some(promo) = self.promotion {
            // Underpromotions only apply to pawns reaching rank 7 (for white) or rank 0 (for black)
            // We only encode underpromotions (not queen promotions, which use queen-like moves)
            if matches!(
                promo,
                PromotionPiece::Knight | PromotionPiece::Bishop | PromotionPiece::Rook
            ) {
                let piece_offset = match promo {
                    PromotionPiece::Knight => 0,
                    PromotionPiece::Bishop => 3,
                    PromotionPiece::Rook => 6,
                    _ => unreachable!(),
                };

                let direction = match delta_file {
                    -1 => 0, // capture left
                    0 => 1,  // move straight
                    1 => 2,  // capture right
                    _ => panic!("Invalid promotion move"),
                };

                return 64 + piece_offset + direction;
            }
        }

        // Queen-like moves (planes 0-55): 8 directions × 7 distances
        // Direction encoding: N, NE, E, SE, S, SW, W, NW
        let direction_index = if delta_rank > 0 && delta_file == 0 {
            0 // North
        } else if delta_rank > 0 && delta_file > 0 && delta_rank == delta_file {
            1 // Northeast
        } else if delta_rank == 0 && delta_file > 0 {
            2 // East
        } else if delta_rank < 0 && delta_file > 0 && -delta_rank == delta_file {
            3 // Southeast
        } else if delta_rank < 0 && delta_file == 0 {
            4 // South
        } else if delta_rank < 0 && delta_file < 0 && delta_rank == delta_file {
            5 // Southwest
        } else if delta_rank == 0 && delta_file < 0 {
            6 // West
        } else if delta_rank > 0 && delta_file < 0 && delta_rank == -delta_file {
            7 // Northwest
        } else {
            panic!(
                "Invalid queen-like move: delta_rank={}, delta_file={}",
                delta_rank, delta_file
            );
        };

        let distance = delta_rank.abs().max(delta_file.abs()) as usize;
        assert!(
            distance >= 1 && distance <= 7,
            "Invalid distance: {}",
            distance
        );

        direction_index * 7 + (distance - 1)
    }

    /// Decodes a move plane (0-72) and source square to a destination square and promotion.
    fn decode_move_plane(from: Square, plane: usize) -> (Square, Option<PromotionPiece>) {
        let from_rank = (from.rank()) as i8;
        let from_file = (from.file()) as i8;

        let (delta_rank, delta_file, promotion) = if plane < 56 {
            // Queen-like moves (planes 0-55)
            let direction = plane / 7;
            let distance = (plane % 7 + 1) as i8;

            let (dr, df) = match direction {
                0 => (1, 0),   // North
                1 => (1, 1),   // Northeast
                2 => (0, 1),   // East
                3 => (-1, 1),  // Southeast
                4 => (-1, 0),  // South
                5 => (-1, -1), // Southwest
                6 => (0, -1),  // West
                7 => (1, -1),  // Northwest
                _ => unreachable!(),
            };

            (dr * distance, df * distance, None)
        } else if plane < 64 {
            // Knight moves (planes 56-63)
            let knight_index = plane - 56;
            let (dr, df) = match knight_index {
                0 => (2, 1),
                1 => (1, 2),
                2 => (-1, 2),
                3 => (-2, 1),
                4 => (-2, -1),
                5 => (-1, -2),
                6 => (1, -2),
                7 => (2, -1),
                _ => unreachable!(),
            };
            (dr, df, None)
        } else {
            // Underpromotions (planes 64-72)
            let promo_index = plane - 64;
            let piece = match promo_index / 3 {
                0 => PromotionPiece::Knight,
                1 => PromotionPiece::Bishop,
                2 => PromotionPiece::Rook,
                _ => unreachable!(),
            };
            let direction = promo_index % 3;
            let df = match direction {
                0 => -1, // capture left
                1 => 0,  // move straight
                2 => 1,  // capture right
                _ => unreachable!(),
            };
            // Assume white pawn promotion (moving north)
            (1, df, Some(piece))
        };

        let to_rank = from_rank + delta_rank;
        let to_file = from_file + delta_file;

        // Clamp to board boundaries
        let to_rank = to_rank.clamp(0, 7) as u8;
        let to_file = to_file.clamp(0, 7) as u8;
        let to = to_rank * 8 + to_file;

        (Square(to), promotion)
    }
}

// impl Action for ChessMove {
//     /// Returns the shape of the multi-dimensional action space for chess moves.
//     ///
//     /// The action space is represented as a 3-dimensional tensor with dimensions `[8, 8, 73]`,
//     /// following the AlphaZero chess encoding scheme. This representation allows neural networks
//     /// to output move probabilities as a tensor that can be directly interpreted as chess moves.
//     ///
//     /// # Action Space Dimensions
//     ///
//     /// - **Dimension 0 (8)**: Source rank (0-7), where 0 represents rank 1 (white's back rank)
//     ///   and 7 represents rank 8 (black's back rank)
//     /// - **Dimension 1 (8)**: Source file (0-7), where 0 represents file 'a' and 7 represents file 'h'
//     /// - **Dimension 2 (73)**: Move plane encoding the destination square and promotion piece type
//     ///
//     /// # Move Plane Encoding (73 planes)
//     ///
//     /// The 73 move planes encode different types of moves:
//     /// - **Planes 0-55**: Queen-like moves (8 directions × 7 distances)
//     /// - **Planes 56-63**: Knight moves (8 possible knight jumps)
//     /// - **Planes 64-72**: Underpromotion moves (3 piece types × 3 directions)
//     ///
//     /// # Total Action Space Size
//     ///
//     /// The total number of possible actions is `8 × 8 × 73 = 4,672`, which represents all
//     /// legal chess moves including special cases like promotions, captures, and castling
//     /// (castling is represented as a king move to the destination square).
//     ///
//     /// # Examples
//     ///
//     /// ```rust
//     /// use evorl_envs::games::chess::moves::ChessMove;
//     /// use evorl_core::action::MultiDiscreteAction;
//     ///
//     /// let space = ChessMove::action_space();
//     /// assert_eq!(space, [8, 8, 73]);
//     /// ```
//     ///
//     /// # References
//     ///
//     /// This encoding follows the approach described in the AlphaZero paper:
//     /// "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
//     /// (Silver et al., 2017).
//     fn shape() -> [usize; 3] {
//         [8, 8, 73]
//     }

//     fn is_valid(&self) -> bool {
//         // Basic validity: both squares must be on the board (0-63)
//         self.from.0 <= 63 && self.to.0 <= 63 && self.from != self.to
//     }
// }

// impl MultiDiscreteAction<3> for ChessMove {
//     /// Constructs a `ChessMove` from multi-dimensional action space indices.
//     ///
//     /// This method converts tensor indices `[rank, file, plane]` into a concrete chess move
//     /// by decoding the AlphaZero-style encoding. It is the inverse of [`to_indices()`].
//     ///
//     /// # Arguments
//     ///
//     /// * `indices` - A 3-element array `[rank, file, plane]` where:
//     ///   - `indices[0]` (0-7): Source rank (0 = rank 1, 7 = rank 8)
//     ///   - `indices[1]` (0-7): Source file (0 = file 'a', 7 = file 'h')
//     ///   - `indices[2]` (0-72): Move plane encoding destination and promotion
//     ///
//     /// # Returns
//     ///
//     /// A `ChessMove` struct representing the decoded move with:
//     /// - `from`: Source square (0-63)
//     /// - `to`: Destination square (0-63)
//     /// - `promotion`: Optional promotion piece for pawn promotions
//     ///
//     /// # Implementation Details
//     ///
//     /// 1. Calculates the source square as `rank × 8 + file`
//     /// 2. Uses [`decode_move_plane()`] to decode the move plane into destination square
//     ///    and promotion piece
//     /// 3. Handles all move types: queen-like slides, knight jumps, and underpromotions
//     ///
//     /// # Panics
//     ///
//     /// Panics if the indices are out of bounds:
//     /// - `indices[0]` ≥ 8 (invalid rank)
//     /// - `indices[1]` ≥ 8 (invalid file)
//     /// - `indices[2]` ≥ 73 (invalid plane)
//     ///
//     /// # Examples
//     ///
//     /// ```rust
//     /// use evorl_envs::games::chess::moves::{ChessMove, PromotionPiece};
//     /// use evorl_core::action::MultiDiscreteAction;
//     ///
//     /// // e2 to e4 (center pawn advance on rank 2)
//     /// // Source: rank 1, file 4 (e2 = square 12)
//     /// // Plane: encodes forward move of distance 2
//     /// let mv = ChessMove::from_indices([1, 4, 1]); // rank 2 (index 1), file e (index 4)
//     ///
//     /// // a7 to a8 with queen promotion
//     /// let promo_move = ChessMove::from_indices([6, 0, 64]); // rank 7, file a, underpromotion plane
//     /// ```
//     ///
//     /// # See Also
//     ///
//     /// - [`to_indices()`] - Converts a move back to tensor indices
//     /// - [`decode_move_plane()`] - Decodes the move plane encoding
//     fn from_indices(indices: [usize; 3]) -> Self {
//         let from_rank = indices[0] as u8;
//         let from_file = indices[1] as u8;
//         let plane = indices[2];

//         let from = from_rank * 8 + from_file;
//         let (to, promotion) = Self::decode_move_plane(Square(from), plane);

//         Self {
//             from: Square(from),
//             to,
//             promotion,
//         }
//     }

//     /// Converts a `ChessMove` into multi-dimensional action space indices.
//     ///
//     /// This method encodes a concrete chess move into the `[rank, file, plane]` tensor
//     /// representation used by neural networks. It is the inverse of [`from_indices()`].
//     ///
//     /// # Returns
//     ///
//     /// A 3-element array `[rank, file, plane]` where:
//     /// - `rank` (0-7): Source rank of the move
//     /// - `file` (0-7): Source file of the move
//     /// - `plane` (0-72): Encoded move plane representation
//     ///
//     /// # Implementation Details
//     ///
//     /// 1. Extracts rank and file from the source square using [`from_rank()`] and [`from_file()`]
//     /// 2. Uses [`compute_move_plane()`] to determine the move plane encoding based on:
//     ///    - Move direction and distance (for queen-like and knight moves)
//     ///    - Promotion piece type (for pawn promotions)
//     ///    - Capture vs. non-capture moves for underpromotions
//     ///
//     /// # Examples
//     ///
//     /// ```rust
//     /// use evorl_envs::games::chess::moves::ChessMove;
//     /// use evorl_core::action::MultiDiscreteAction;
//     ///
//     /// // Create a move from e2 to e4
//     /// let mv = ChessMove::new(12, 28); // e2 = 12, e4 = 28
//     /// let indices = mv.to_indices();
//     /// assert_eq!(indices[0], 1); // rank 2 (index 1)
//     /// assert_eq!(indices[1], 4); // file e (index 4)
//     /// // indices[2] will be the plane encoding this forward move
//     ///
//     /// // Verify round-trip: from_indices(to_indices(move)) == move
//     /// let mv2 = ChessMove::from_indices(mv.to_indices());
//     /// assert_eq!(mv.from, mv2.from);
//     /// assert_eq!(mv.to, mv2.to);
//     /// assert_eq!(mv.promotion, mv2.promotion);
//     /// ```
//     ///
//     /// # See Also
//     ///
//     /// - [`from_indices()`] - Constructs a move from tensor indices
//     /// - [`compute_move_plane()`] - Computes the move plane encoding
//     fn to_indices(&self) -> [usize; 3] {
//         let plane = self.compute_move_plane();
//         [self.from_rank() as usize, self.from_file() as usize, plane]
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_indices() {
        // Test a1 (square 0)
        let mv = ChessMove::new(Square(0), Square(16)); // a1 to a3
        assert_eq!(mv.from_rank(), 0);
        assert_eq!(mv.from_file(), 0);

        // Test h8 (square 63)
        let mv = ChessMove::new(Square(63), Square(47)); // h8 to h6
        assert_eq!(mv.from_rank(), 7);
        assert_eq!(mv.from_file(), 7);

        // Test e2 (square 12)
        let mv = ChessMove::new(Square(12), Square(28)); // e2 to e4
        assert_eq!(mv.from_rank(), 1);
        assert_eq!(mv.from_file(), 4);
    }

    #[test]
    fn test_knight_move_encoding() {
        // Knight move from e4 (28) to f6 (45): +2 rank, +1 file
        let mv = ChessMove::new(Square(28), Square(45));
        // let indices = mv.to_indices();
        // assert_eq!(indices[0], 3); // rank 3 (e4)
        // assert_eq!(indices[1], 4); // file 4 (e-file)
        // assert_eq!(indices[2], 56); // First knight move plane

        // Reconstruct
        // let reconstructed = ChessMove::from_indices(indices);
        // assert_eq!(mv.from, reconstructed.from);
        // assert_eq!(mv.to, reconstructed.to);
    }

    #[test]
    fn test_queen_move_encoding() {
        // North move: e2 to e4 (2 squares north)
        let mv = ChessMove::new(Square(12), Square(28)); // e2 to e4
                                                         // let indices = mv.to_indices();
                                                         // assert_eq!(indices[0], 1); // rank 1
                                                         // assert_eq!(indices[1], 4); // file 4
                                                         // assert_eq!(indices[2], 1); // North direction, distance 2 (plane 0 + 1)

        // Reconstruct
        // let reconstructed = ChessMove::from_indices(indices);
        // assert_eq!(mv.from, reconstructed.from);
        // assert_eq!(mv.to, reconstructed.to);
    }

    #[test]
    fn test_promotion_encoding() {
        // Pawn promotion: e7 to e8 with knight promotion
        let mv = ChessMove::new_with_promotion(Square(52), Square(60), PromotionPiece::Knight);
        // let indices = mv.to_indices();
        // assert_eq!(indices[0], 6); // rank 6
        // assert_eq!(indices[1], 4); // file 4
        // assert_eq!(indices[2], 65); // Knight underpromotion straight (64 + 0*3 + 1)

        // Reconstruct
        // let reconstructed = ChessMove::from_indices(indices);
        // assert_eq!(mv.from, reconstructed.from);
        // assert_eq!(mv.promotion, reconstructed.promotion);
    }

    // #[test]
    // fn test_action_space() {
    //     let space = ChessMove.shape();
    //     assert_eq!(space, [8, 8, 73]);
    // }

    // #[test]
    // fn test_is_valid() {
    //     let valid_move = ChessMove::new(Square(12), Square(28));
    //     assert!(valid_move.is_valid());

    //     let invalid_same_square = ChessMove::new(Square(12), Square(12));
    //     assert!(!invalid_same_square.is_valid());
    // }
}
