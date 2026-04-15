//! Color palette shared by all grid entities.

use serde::{Deserialize, Serialize};

/// The six colors Minigrid uses for doors, keys, balls, and boxes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Color {
    Red,
    Green,
    Blue,
    Purple,
    Yellow,
    Grey,
}

impl Color {
    /// Every color in a stable order, suitable for enumeration.
    pub const ALL: [Self; 6] = [
        Self::Red,
        Self::Green,
        Self::Blue,
        Self::Purple,
        Self::Yellow,
        Self::Grey,
    ];

    /// Encode this color as a 1-based byte for observation channels.
    ///
    /// The byte range `1..=6` is used so that `0` can represent "no color"
    /// in entities that don't have one.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        match self {
            Self::Red => 1,
            Self::Green => 2,
            Self::Blue => 3,
            Self::Purple => 4,
            Self::Yellow => 5,
            Self::Grey => 6,
        }
    }

    /// Single-letter label for ASCII rendering.
    #[must_use]
    pub const fn ascii(self) -> char {
        match self {
            Self::Red => 'R',
            Self::Green => 'G',
            Self::Blue => 'B',
            Self::Purple => 'P',
            Self::Yellow => 'Y',
            Self::Grey => 'W',
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_u8_is_unique_and_nonzero() {
        let mut seen = Vec::with_capacity(Color::ALL.len());
        for c in Color::ALL {
            let v = c.to_u8();
            assert!(v >= 1, "color byte must be non-zero");
            assert!(!seen.contains(&v), "duplicate color byte for {c:?}");
            seen.push(v);
        }
    }

    #[test]
    fn all_has_six_colors() {
        assert_eq!(Color::ALL.len(), 6);
    }
}
