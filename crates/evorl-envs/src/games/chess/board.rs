/// Represents a square on the chess board (0-63).
///
/// Square 0 corresponds to a1, and square 63 corresponds to h8.
/// Squares are indexed as: `rank * 8 + file`, where both rank and file are 0-indexed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Square(pub u8);

impl Square {
    /// Creates a square from rank and file (0-7).
    #[inline]
    pub const fn from_rank_file(rank: u8, file: u8) -> Self {
        Self(rank * 8 + file)
    }

    /// Returns the rank (0-7, where 0 is rank 1).
    #[inline]
    pub const fn rank(self) -> u8 {
        self.0 / 8
    }

    /// Returns the file (0-7, where 0 is file A).
    #[inline]
    pub const fn file(self) -> u8 {
        self.0 % 8
    }

    /// Returns the bitboard mask for this square.
    #[inline]
    pub const fn bitboard(self) -> u64 {
        1u64 << self.0
    }
}

/// Side/color in chess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Color {
    White,
    Black,
}

impl Color {
    /// Returns the opposite color.
    #[inline]
    pub const fn opponent(self) -> Self {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

/// Chess piece type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    /// Returns the index for bitboard access.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// Castling rights for both sides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CastlingRights {
    pub white_kingside: bool,
    pub white_queenside: bool,
    pub black_kingside: bool,
    pub black_queenside: bool,
}

impl Default for CastlingRights {
    fn default() -> Self {
        Self {
            white_kingside: true,
            white_queenside: true,
            black_kingside: true,
            black_queenside: true,
        }
    }
}
