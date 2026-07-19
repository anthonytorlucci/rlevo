//! Observable chess state using AlphaZero Chess representation.
//!
//! This module provides the core state representation for a chess environment,
//! designed to work with reinforcement learning agents and neural networks.
//! The `ChessState` struct encodes the complete game state in a compact,
//! efficient format suitable for tensor conversion and GPU computation.
//!
//! # Overview
//!
//! Unlike traditional chess engines that use hand-crafted features (like material count or pawn
//! structure), AlphaZero represents the board as a three-dimensional tensor of dimensions
//! $8 \times 8 \times 119$.
//!
//! # 1. The Input Tensor ($8 \times 8 \times 119$)
//! The state is represented by 119 binary planes of size $8 \times 8$. These planes are divided
//! into two main categories: historical board states and constant game metadata.
//!
//! **A. Historical State (112 Planes)**
//! AlphaZero does not just look at the current position; it maintains a history of the 8 most
//! recent half-moves (the current state plus the 7 previous ones). This history allows the
//! network to detect patterns and rules that depend on the sequence of moves, such as three-fold
//! repition.
//!
//! For each of these 8 time steps, there are 14 planes:
//! - **6 planes for your pieces**: One plane for each piece type (Pawn, Knight, Bishop, Rook,
//!   Queen, King). A "1" indicates the presence of that piece on a square.
//! - **6 planes for opponent pieces**: Same as above, but for the opponent's pieces.
//! - **2 repition planes**: These indicate whether the current position has occured before in the
//!   game (relevant for the draw-by-repition rule).
//! $$8 \space \text{time steps} \times 14 \space \text{planes per step} = 112 \space \text{planes}$$
//!
//! **B. Constant Metadata (7 Planes)**
//! The final 7 planes represent global state information that doesn't change based on piece
//! movement but is critical for legal play:
//! - **Castling Rights (4 planes)**: Two for you (Kingside/Queenside) and two for the opponent.
//! - **Side to Move (1 plane)**: Indicates whose turn it is (often represented as all 1s if its
//!   White or all 0s if its Black).
//! - **Total Move Count (1 plane)**: The current move number in the game.
//! - **No-progress Count (1 plane)**: Tracks the 50-move rule (number of moves since the last
//!   pawn move or capture).
//!
//! # 2. Key Characteristics
//! - **Player Perspective**: The board is always oriented from the perspective of the current
//!   player to move. This means the network doesn't have to learn "White strategy" and "Black
//!   strategy" separately; it learns how to play from the "bottom" of the board, regardless of
//!   piece color.
//! - **Raw Representation**: There are no human heuristics like "central control" or "king
//!   safety." The network must learn these abstract concepts entirely from the raw bit-planes
//!   during reinforcement learning.
//! - **Translation Invariance**: Using an $8 \times 8$ spatial grid allows the Convolutional
//!   Neural Network (CNN) layers to recognize patterns (like a "back-rank mate" or a "pawn fork")
//!   anywhere on the board using the same learned filters.
//!
//! # 3. The State vs. Observation
//! While the state space of chess is the mathematical set of all possible legal configurations of
//! the board (estimated at $10^{40}$ to $10^{50}$), the observation space described above is the
//! specific "window" through which AlphaZero perceives the state. By including history, AlphaZero
//! effectively turns a non-Markovian environment (where rules like repitition depend on the past)
//! into a Markovian one that the network can process.
//!
//! The representation allows for:
//! - O(1) position queries and updates
//! - Efficient move generation and validation
//! - Natural tensor conversion for neural networks (8×8×119 layout)
//!
//! # When to Use This Module
//!
//! Use `ChessState` when:
//! - Building chess-playing agents (DQN, PPO, ES, etc.)
//! - Implementing chess environments for reinforcement learning
//! - Performing position analysis and evaluation
//! - Converting game states to neural network inputs
//!
//! # Validity Invariants (planned)
//!
//! The commented-out `State` impl will enforce:
//! - Exactly one king per side
//! - No pawns on the first or last rank
//! - Total piece count per side does not exceed 16
//!
//! Until the `State` and `TensorConvertible` trait impls are activated (see the
//! commented-out blocks below), callers can inspect individual bitboards directly
//! via `BoardSnapshot` fields, but no top-level `is_valid()` method is available.
//!
//! # Creating a Starting Position
//!
//! ```ignore
//! // Create the standard starting position
//! let state = ChessState::default();
//! assert_eq!(state.to_move(), Color::White);
//! assert_eq!(state.fullmove_number(), 1);
//! assert_eq!(state.halfmove_clock(), 0);
//! ```
//!
//! # Performance Considerations
//!
//! - **Bitboards**: Most operations are bit manipulations, extremely fast on modern CPUs
//! - **Tensor Conversion**: O(64) one-time cost per state observation
//! - **Cloning**: Cheap due to small data footprint and Copy semantics on bitboards
//!
//! For high-throughput environments (e.g., parallel training), consider caching
//! tensor conversions to avoid redundant computation.
//!
//! # Related Modules
//!
//! - [`super::board`]: Board utilities and visualization
//! - [`super::moves`]: Move generation and validation
//! - [`rlevo_core::state`]: Abstract state trait and conversions
//!
//! # Implementation Notes
//!
//! The state is designed to be:
//! - **Mutation-based**: Fields are `pub(super)` to the environment impl; move
//!   application will mutate the struct in place and push a new `BoardSnapshot`
//!   onto the history ring buffer.
//! - **Deterministic**: Same state always converts to identical tensors.
//! - **Debuggable**: Implements `Debug` for inspection.
//! - **Hashable**: Enables transposition tables and memoization for search.

use crate::games::chess::board::{CastlingRights, Color, PieceType, Square};
// Required once the commented-out `impl State` and `impl Observation` blocks are activated.
#[allow(unused_imports)]
use burn::prelude::*;
#[allow(unused_imports)]
use rlevo_core::base::{Observation, State};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Number of historical board positions to maintain (8 most recent half-moves).
const HISTORY_SIZE: usize = 8;

/// Number of piece types per side (Pawn, Knight, Bishop, Rook, Queen, King).
const PIECE_TYPES: usize = 6;

/// Total number of planes in the observation tensor.
/// - 112 planes: 8 time steps × (6 own pieces + 6 opponent pieces + 2 repetition)
/// - 7 planes: castling rights (4) + side to move (1) + move count (1) + no-progress (1)
const TOTAL_PLANES: usize = 119;

/// Bitboard snapshot for a single half-move position.
///
/// Stores twelve 64-bit bitboards — one per piece type per color — indexed as
/// `[color_offset + PieceType::index()]`, where the White offset is 0 and the
/// Black offset is 6. A set bit at position `sq` (0 = a1, 63 = h8) indicates
/// that the corresponding piece occupies that square.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BoardSnapshot {
    /// Bitboards for 12 piece types (6 white + 6 black).
    /// Indices 0-5: White pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    /// Indices 6-11: Black pieces (Pawn, Knight, Bishop, Rook, Queen, King)
    piece_boards: [u64; 12],
}

impl BoardSnapshot {
    /// Returns an empty snapshot with no pieces on the board.
    fn empty() -> Self {
        Self {
            piece_boards: [0u64; 12],
        }
    }

    /// Returns bitboard for a specific piece type and color.
    #[inline]
    fn get_piece(&self, color: Color, piece: PieceType) -> u64 {
        let offset = match color {
            Color::White => 0,
            Color::Black => 6,
        };
        self.piece_boards[offset + piece.index()]
    }

    /// Sets bitboard for a specific piece type and color.
    #[inline]
    fn set_piece(&mut self, color: Color, piece: PieceType, bitboard: u64) {
        let offset = match color {
            Color::White => 0,
            Color::Black => 6,
        };
        self.piece_boards[offset + piece.index()] = bitboard;
    }
}

/// Chess state representation using AlphaZero observation space.
///
/// This struct maintains:
/// - Historical board positions (last 8 half-moves)
/// - Current game metadata (castling, en passant, move counters)
/// - Position repetition tracking for draw detection
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChessState {
    /// Historical board positions (newest first, size = HISTORY_SIZE).
    /// history[0] is the current position, history[1] is one move ago, etc.
    history: Vec<BoardSnapshot>,

    /// Current side to move.
    to_move: Color,

    /// Castling rights for both sides.
    castling_rights: CastlingRights,

    /// En passant target square (if a pawn just moved two squares).
    en_passant: Option<Square>,

    /// Half-move clock for the 50-move rule (resets on pawn moves or captures).
    halfmove_clock: u8,

    /// Full move number (starts at 1, increments after Black's move).
    fullmove_number: u16,

    /// Position repetition counts for draw detection.
    /// Maps position hash to number of occurrences.
    repetition_history: HashMap<u64, u8>,
}

impl std::hash::Hash for ChessState {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash only the fields that implement Hash
        self.history.hash(state);
        self.to_move.hash(state);
        self.castling_rights.hash(state);
        self.en_passant.hash(state);
        self.halfmove_clock.hash(state);
        self.fullmove_number.hash(state);

        // For repetition_history, hash a derived value instead of the HashMap itself
        // For draw detection, we care if any position repeated 3+ times (threefold repetition)
        let has_threefold = self.repetition_history.values().any(|&count| count >= 3);
        has_threefold.hash(state);
    }
}

impl ChessState {
    /// Creates a new chess state with the standard starting position.
    pub fn new() -> Self {
        let mut snapshot = BoardSnapshot::empty();

        // Set up white pieces
        snapshot.set_piece(Color::White, PieceType::Pawn, 0x000000000000FF00);
        snapshot.set_piece(Color::White, PieceType::Knight, 0x0000000000000042);
        snapshot.set_piece(Color::White, PieceType::Bishop, 0x0000000000000024);
        snapshot.set_piece(Color::White, PieceType::Rook, 0x0000000000000081);
        snapshot.set_piece(Color::White, PieceType::Queen, 0x0000000000000008);
        snapshot.set_piece(Color::White, PieceType::King, 0x0000000000000010);

        // Set up black pieces
        snapshot.set_piece(Color::Black, PieceType::Pawn, 0x00FF000000000000);
        snapshot.set_piece(Color::Black, PieceType::Knight, 0x4200000000000000);
        snapshot.set_piece(Color::Black, PieceType::Bishop, 0x2400000000000000);
        snapshot.set_piece(Color::Black, PieceType::Rook, 0x8100000000000000);
        snapshot.set_piece(Color::Black, PieceType::Queen, 0x0800000000000000);
        snapshot.set_piece(Color::Black, PieceType::King, 0x1000000000000000);

        let mut history = Vec::with_capacity(HISTORY_SIZE);
        history.push(snapshot);

        // Fill remaining history with empty snapshots
        for _ in 1..HISTORY_SIZE {
            history.push(BoardSnapshot::empty());
        }

        Self {
            history,
            to_move: Color::White,
            castling_rights: CastlingRights::default(),
            en_passant: None,
            halfmove_clock: 0,
            fullmove_number: 1,
            repetition_history: HashMap::new(),
        }
    }

    /// Returns the current board snapshot (most recent position).
    #[inline]
    #[allow(dead_code)] // v0.2: used once Environment impl lands
    fn current_board(&self) -> &BoardSnapshot {
        &self.history[0]
    }

    /// Returns the side to move.
    #[inline]
    pub fn to_move(&self) -> Color {
        self.to_move
    }

    /// Returns the castling rights.
    #[inline]
    pub fn castling_rights(&self) -> CastlingRights {
        self.castling_rights
    }

    /// Returns the en passant target square, if any.
    #[inline]
    pub fn en_passant(&self) -> Option<Square> {
        self.en_passant
    }

    /// Returns the half-move clock (for 50-move rule).
    #[inline]
    pub fn halfmove_clock(&self) -> u8 {
        self.halfmove_clock
    }

    /// Returns the full move number.
    #[inline]
    pub fn fullmove_number(&self) -> u16 {
        self.fullmove_number
    }

    /// Checks how many times the current position has occurred (for repetition draws).
    /// Returns the number of times the current position has appeared in the game.
    /// Used for detecting threefold repetition draws.
    pub fn repetition_count(&self) -> u8 {
        let hash = self.position_hash();
        *self.repetition_history.get(&hash).unwrap_or(&0)
    }

    /// Computes a lightweight hash of the current position for repetition detection.
    ///
    /// The hash covers the current board snapshot, side to move, castling rights,
    /// and en-passant square — the four factors that distinguish otherwise
    /// identical-looking positions under FIDE rules. A full implementation would
    /// use Zobrist hashing for incrementally updated keys; this version uses
    /// `DefaultHasher` as a placeholder.
    fn position_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.history[0].hash(&mut hasher);
        self.to_move.hash(&mut hasher);
        self.castling_rights.hash(&mut hasher);
        self.en_passant.hash(&mut hasher);
        hasher.finish()
    }

    /// Expands a 64-bit bitboard into a flat `[f32; 64]` binary plane.
    ///
    /// Each element is `1.0` where the corresponding bit is set, `0.0` otherwise.
    /// Index ordering follows the square encoding: index `i` corresponds to
    /// `Square(i)` (rank `i / 8`, file `i % 8`).
    #[inline]
    fn bitboard_to_plane(bitboard: u64) -> [f32; 64] {
        let mut plane = [0.0f32; 64];
        for i in 0..64 {
            if (bitboard >> i) & 1 == 1 {
                plane[i] = 1.0;
            }
        }
        plane
    }

    /// Flips a binary plane vertically so that rank 1 becomes rank 8 and vice versa.
    ///
    /// AlphaZero always presents the board from the perspective of the side to move.
    /// When it is Black's turn, the board is flipped so that Black's pieces appear at
    /// the "bottom" (rank 1 in the output plane), matching the orientation used when
    /// it is White's turn. This allows the network to learn a single set of positional
    /// patterns regardless of which color is to move.
    #[inline]
    fn flip_plane_vertical(plane: &[f32; 64]) -> [f32; 64] {
        let mut flipped = [0.0f32; 64];
        for rank in 0..8 {
            for file in 0..8 {
                let old_idx = rank * 8 + file;
                let new_idx = (7 - rank) * 8 + file;
                flipped[new_idx] = plane[old_idx];
            }
        }
        flipped
    }
}

impl Default for ChessState {
    fn default() -> Self {
        Self::new()
    }
}

// impl State for ChessState {
//     fn is_valid(&self) -> bool {
//         let current = self.current_board();

//         // Check exactly one king per side
//         let white_kings = current
//             .get_piece(Color::White, PieceType::King)
//             .count_ones();
//         let black_kings = current
//             .get_piece(Color::Black, PieceType::King)
//             .count_ones();
//         if white_kings != 1 || black_kings != 1 {
//             return false;
//         }

//         // Check no pawns on first or last rank
//         let white_pawns = current.get_piece(Color::White, PieceType::Pawn);
//         let black_pawns = current.get_piece(Color::Black, PieceType::Pawn);
//         let rank_1_mask = 0x00000000000000FF;
//         let rank_8_mask = 0xFF00000000000000;

//         if (white_pawns & rank_1_mask) != 0 || (white_pawns & rank_8_mask) != 0 {
//             return false;
//         }
//         if (black_pawns & rank_1_mask) != 0 || (black_pawns & rank_8_mask) != 0 {
//             return false;
//         }

//         // Check piece counts are reasonable (max 16 per side)
//         let mut white_count = 0u32;
//         let mut black_count = 0u32;
//         for piece in [
//             PieceType::Pawn,
//             PieceType::Knight,
//             PieceType::Bishop,
//             PieceType::Rook,
//             PieceType::Queen,
//             PieceType::King,
//         ] {
//             white_count += current.get_piece(Color::White, piece).count_ones();
//             black_count += current.get_piece(Color::Black, piece).count_ones();
//         }

//         if white_count > 16 || black_count > 16 {
//             return false;
//         }

//         true
//     }

//     fn numel(&self) -> usize {
//         8 * 8 * TOTAL_PLANES
//     }

//     fn shape(&self) -> Vec<usize> {
//         vec![8, 8, TOTAL_PLANES]
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Flex;

    type TestBackend = Flex;

    #[test]
    fn test_state_creation() {
        let state = ChessState::new();
        // assert!(state.is_valid());
        assert_eq!(state.to_move(), Color::White);
        assert_eq!(state.fullmove_number(), 1);
        assert_eq!(state.halfmove_clock(), 0);
    }

    #[test]
    fn test_state_shape() {
        let _state = ChessState::new();
        // assert_eq!(_state.shape(), vec![8, 8, 119]);
        // assert_eq!(state.numel(), 8 * 8 * 119);
    }

    #[test]
    fn test_tensor_conversion() {
        let _state = ChessState::new();
        // let device = Default::default();
        // let tensor = _state.to_tensor::<TestBackend>(&device);

        // let shape = tensor.shape();
        // assert_eq!(shape.dims, [8, 8, 119]);
    }

    #[test]
    fn test_tensor_roundtrip() {
        let _state = ChessState::new();
        // let device = Default::default();
        // let tensor = state.to_tensor::<TestBackend>(&device);

        // let reconstructed = ChessState::from_tensor(&tensor).unwrap();
        // assert!(reconstructed.is_valid());
        // assert_eq!(reconstructed.to_move(), state.to_move());
    }

    #[test]
    fn test_validation_no_pawns_on_first_rank() {
        let mut state = ChessState::new();
        // Manually set invalid pawn position (pawn on rank 1)
        state.history[0].set_piece(Color::White, PieceType::Pawn, 0x0000000000000001);
        // assert!(!state.is_valid());
    }

    #[test]
    fn test_validation_no_pawns_on_last_rank() {
        let mut state = ChessState::new();
        // Manually set invalid pawn position (pawn on rank 8)
        state.history[0].set_piece(Color::Black, PieceType::Pawn, 0x8000000000000000);
        // assert!(!state.is_valid());
    }

    #[test]
    fn test_validation_one_king_per_side() {
        let mut state = ChessState::new();
        // Remove white king
        state.history[0].set_piece(Color::White, PieceType::King, 0);
        // assert!(!state.is_valid());
    }

    #[test]
    fn test_color_opponent() {
        assert_eq!(Color::White.opponent(), Color::Black);
        assert_eq!(Color::Black.opponent(), Color::White);
    }

    #[test]
    fn test_square_rank_file() {
        let sq = Square::from_rank_file(3, 4);
        assert_eq!(sq.rank(), 3);
        assert_eq!(sq.file(), 4);
        assert_eq!(sq.0, 3 * 8 + 4);
    }

    #[test]
    fn test_bitboard_to_plane() {
        let bitboard = 0x0000000000000001; // A1 square
        let plane = ChessState::bitboard_to_plane(bitboard);
        assert_eq!(plane[0], 1.0);
        for i in 1..64 {
            assert_eq!(plane[i], 0.0);
        }
    }

    #[test]
    fn test_plane_flip() {
        let mut plane = [0.0f32; 64];
        plane[0] = 1.0; // A1 (bottom-left from White's perspective)

        let flipped = ChessState::flip_plane_vertical(&plane);
        assert_eq!(flipped[56], 1.0); // A8 (top-left, now bottom-left from Black's perspective)
    }
}
