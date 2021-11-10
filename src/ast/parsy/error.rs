use std::{
    error::Error,
    fmt,
    num::{ParseFloatError, ParseIntError},
};

#[derive(Clone, Debug)]
pub enum ParseError {
    IncorrectToken,
    InvalidIntLiteral,
    InvalidFloatLiteral,
    Expected(&'static str, String),
    Error(&'static str),
    Other,
}

impl From<ParseIntError> for ParseError {
    fn from(_: ParseIntError) -> Self {
        Self::InvalidIntLiteral
    }
}

impl From<ParseFloatError> for ParseError {
    fn from(_: ParseFloatError) -> Self {
        Self::InvalidIntLiteral
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::IncorrectToken => f.write_str("Parser encountered incorrect token"),
            ParseError::InvalidIntLiteral => {
                f.write_str("Parser encountered invalid integer literal")
            }
            ParseError::InvalidFloatLiteral => {
                f.write_str("Parser encountered invalid float literal")
            }
            ParseError::Expected(exp, found) => {
                write!(f, "Parser encountered error, expected {} found {}", exp, found)
            }
            ParseError::Error(exp) => {
                write!(f, "Parser encountered error, expected {}", exp)
            }
            ParseError::Other => f.write_str("ICE"),
        }
    }
}
