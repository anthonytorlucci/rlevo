/// Represents a square on the chess board (0-63).
///
/// Square 0 corresponds to a1, and square 63 corresponds to h8.
/// Squares are indexed as: `rank * 8 + file`, where both rank and file are 0-indexed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Square(pub u8);

/// Static lookup table for algebraic notation of all 64 squares.
///
/// Provides O(1) conversion from square index (0-63) to algebraic notation (e.g., "a1", "e4", "h8").
/// Squares are indexed as: `rank * 8 + file`, where rank and file are both 0-indexed.
/// - Rank 0 = chess rank 1 (rows 0-7: a1-h1)
/// - Rank 1 = chess rank 2 (rows 8-15: a2-h2)
/// - ...
/// - Rank 7 = chess rank 8 (rows 56-63: a8-h8)
const SQUARE_NOTATIONS: &[&str] = &[
    // Rank 1
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", // Rank 2
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2", // Rank 3
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3", // Rank 4
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4", // Rank 5
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5", // Rank 6
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6", // Rank 7
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7", // Rank 8
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
];

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

    /// Returns the algebraic notation for this square.
    ///
    /// Algebraic notation is the standard system used in chess to denote board squares.
    /// Combines the file (column) letter (a-h) with the rank (row) number (1-8).
    ///
    /// # Examples
    /// ```ignore
    /// let sq = Square::from_rank_file(0, 4);  // rank 1, file e
    /// assert_eq!(sq.algebraic_notation(), "e1");
    ///
    /// let sq = Square::from_rank_file(7, 7);  // rank 8, file h
    /// assert_eq!(sq.algebraic_notation(), "h8");
    ///
    /// let sq = Square(0);  // a1
    /// assert_eq!(sq.algebraic_notation(), "a1");
    /// ```
    #[inline]
    pub const fn algebraic_notation(self) -> &'static str {
        SQUARE_NOTATIONS[self.0 as usize]
    }

    /// Creates a square from algebraic notation (e.g., "e4", "d5", "h8").
    ///
    /// Algebraic notation consists of a file letter (a-h) followed by a rank number (1-8).
    /// This constructor is particularly useful when parsing chess notation or user input.
    ///
    /// # Arguments
    /// * `notation` - A two-character string representing the square in algebraic notation.
    ///
    /// # Returns
    /// Returns `Some(Square)` if the input is valid, or `None` if the input is malformed.
    ///
    /// # Examples
    /// ```ignore
    /// let sq = Square::from_algebraic_notation("e4").unwrap();
    /// assert_eq!(sq.rank(), 3);  // rank 4 (0-indexed as 3)
    /// assert_eq!(sq.file(), 4);  // file e (0-indexed as 4)
    ///
    /// let sq = Square::from_algebraic_notation("a1").unwrap();
    /// assert_eq!(sq, Square(0));
    ///
    /// let sq = Square::from_algebraic_notation("h8").unwrap();
    /// assert_eq!(sq, Square(63));
    ///
    /// // Invalid inputs return None
    /// assert!(Square::from_algebraic_notation("i9").is_none());
    /// assert!(Square::from_algebraic_notation("e").is_none());
    /// ```
    pub fn from_algebraic_notation(notation: &str) -> Option<Self> {
        // Algebraic notation must be exactly 2 characters
        if notation.len() != 2 {
            return None;
        }

        let bytes = notation.as_bytes();
        let file_char = bytes[0];
        let rank_char = bytes[1];

        // Parse file letter (a-h maps to 0-7)
        let file = match file_char {
            b'a'..=b'h' => file_char - b'a',
            _ => return None,
        };

        // Parse rank digit (1-8 maps to 0-7)
        let rank = match rank_char {
            b'1'..=b'8' => rank_char - b'1',
            _ => return None,
        };

        Some(Self::from_rank_file(rank, file))
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

// Run the tests with:
// ```ignore
// cargo test --package rlevo-envs games::chess::board::tests
// ```
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_algebraic_notation_valid_squares() {
        // Test corner squares
        let a1 = Square::from_algebraic_notation("a1").expect("a1 should parse");
        assert_eq!(a1, Square(0));
        assert_eq!(a1.rank(), 0);
        assert_eq!(a1.file(), 0);
        assert_eq!(a1.algebraic_notation(), "a1");

        let h8 = Square::from_algebraic_notation("h8").expect("h8 should parse");
        assert_eq!(h8, Square(63));
        assert_eq!(h8.rank(), 7);
        assert_eq!(h8.file(), 7);
        assert_eq!(h8.algebraic_notation(), "h8");

        let a8 = Square::from_algebraic_notation("a8").expect("a8 should parse");
        assert_eq!(a8, Square(56));
        assert_eq!(a8.rank(), 7);
        assert_eq!(a8.file(), 0);

        let h1 = Square::from_algebraic_notation("h1").expect("h1 should parse");
        assert_eq!(h1, Square(7));
        assert_eq!(h1.rank(), 0);
        assert_eq!(h1.file(), 7);
    }

    #[test]
    fn test_from_algebraic_notation_center_squares() {
        // Test common opening squares
        let e4 = Square::from_algebraic_notation("e4").expect("e4 should parse");
        assert_eq!(e4.rank(), 3); // rank 4 (0-indexed)
        assert_eq!(e4.file(), 4); // file e (0-indexed)
        assert_eq!(e4.algebraic_notation(), "e4");

        let d5 = Square::from_algebraic_notation("d5").expect("d5 should parse");
        assert_eq!(d5.rank(), 4); // rank 5 (0-indexed)
        assert_eq!(d5.file(), 3); // file d (0-indexed)
        assert_eq!(d5.algebraic_notation(), "d5");

        let c3 = Square::from_algebraic_notation("c3").expect("c3 should parse");
        assert_eq!(c3.rank(), 2); // rank 3 (0-indexed)
        assert_eq!(c3.file(), 2); // file c (0-indexed)
    }

    #[test]
    fn test_from_algebraic_notation_all_files() {
        // Test that all file letters parse correctly
        for (file_index, file_letter) in "abcdefgh".chars().enumerate() {
            let notation = format!("{}1", file_letter);
            let sq = Square::from_algebraic_notation(&notation)
                .unwrap_or_else(|| panic!("{}1 should parse", file_letter));
            assert_eq!(sq.file(), file_index as u8);
            assert_eq!(sq.rank(), 0);
        }
    }

    #[test]
    fn test_from_algebraic_notation_all_ranks() {
        // Test that all rank numbers parse correctly
        for (rank_index, rank_digit) in "12345678".chars().enumerate() {
            let notation = format!("a{}", rank_digit);
            let sq = Square::from_algebraic_notation(&notation)
                .unwrap_or_else(|| panic!("a{} should parse", rank_digit));
            assert_eq!(sq.rank(), rank_index as u8);
            assert_eq!(sq.file(), 0);
        }
    }

    #[test]
    fn test_from_algebraic_notation_invalid_length() {
        // Test strings that are too short or too long
        assert!(
            Square::from_algebraic_notation("").is_none(),
            "empty string should fail"
        );
        assert!(
            Square::from_algebraic_notation("e").is_none(),
            "single char should fail"
        );
        assert!(
            Square::from_algebraic_notation("e45").is_none(),
            "three chars should fail"
        );
        assert!(
            Square::from_algebraic_notation("e4x").is_none(),
            "invalid third char should fail"
        );
    }

    #[test]
    fn test_from_algebraic_notation_invalid_file() {
        // Test invalid file letters
        assert!(
            Square::from_algebraic_notation("i4").is_none(),
            "file i should fail"
        );
        assert!(
            Square::from_algebraic_notation("z5").is_none(),
            "file z should fail"
        );
        assert!(
            Square::from_algebraic_notation("@4").is_none(),
            "special char @ should fail"
        );
        assert!(
            Square::from_algebraic_notation("A4").is_none(),
            "uppercase A should fail"
        );
        assert!(
            Square::from_algebraic_notation(" 4").is_none(),
            "space should fail"
        );
    }

    #[test]
    fn test_from_algebraic_notation_invalid_rank() {
        // Test invalid rank numbers
        assert!(
            Square::from_algebraic_notation("e9").is_none(),
            "rank 9 should fail"
        );
        assert!(
            Square::from_algebraic_notation("e0").is_none(),
            "rank 0 should fail"
        );
        assert!(
            Square::from_algebraic_notation("ex").is_none(),
            "non-digit should fail"
        );
        assert!(
            Square::from_algebraic_notation("eA").is_none(),
            "uppercase letter should fail"
        );
        assert!(
            Square::from_algebraic_notation("e ").is_none(),
            "space should fail"
        );
    }

    #[test]
    fn test_from_algebraic_notation_roundtrip() {
        // Test that parsing a notation and converting back yields the same notation
        let notations = vec![
            "a1", "b2", "c3", "d4", "e5", "f6", "g7", "h8", "e4", "d5", "c6", "b7", "a8", "h1",
            "g2", "f3",
        ];

        for notation in notations {
            let sq = Square::from_algebraic_notation(notation)
                .unwrap_or_else(|| panic!("{} should parse", notation));
            assert_eq!(
                sq.algebraic_notation(),
                notation,
                "roundtrip failed for {}",
                notation
            );
        }
    }

    #[test]
    fn test_from_algebraic_notation_bitboard() {
        // Test that parsed squares have correct bitboard masks
        let a1 = Square::from_algebraic_notation("a1").expect("a1 should parse");
        assert_eq!(a1.bitboard(), 1u64); // 2^0

        let b1 = Square::from_algebraic_notation("b1").expect("b1 should parse");
        assert_eq!(b1.bitboard(), 2u64); // 2^1

        let a2 = Square::from_algebraic_notation("a2").expect("a2 should parse");
        assert_eq!(a2.bitboard(), 1u64 << 8); // 2^8

        let h8 = Square::from_algebraic_notation("h8").expect("h8 should parse");
        assert_eq!(h8.bitboard(), 1u64 << 63); // 2^63
    }

    #[test]
    fn test_from_algebraic_notation_equivalence() {
        // Test that from_algebraic_notation is equivalent to from_rank_file
        let sq_from_notation = Square::from_algebraic_notation("e4").expect("e4 should parse");
        let sq_from_rank_file = Square::from_rank_file(3, 4); // rank 4 (0-indexed as 3), file e (0-indexed as 4)

        assert_eq!(sq_from_notation, sq_from_rank_file);
        assert_eq!(sq_from_notation.rank(), sq_from_rank_file.rank());
        assert_eq!(sq_from_notation.file(), sq_from_rank_file.file());
    }
}
