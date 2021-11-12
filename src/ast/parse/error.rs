use std::{
    error::Error,
    fmt,
    num::{ParseFloatError, ParseIntError},
};

use crate::ast::types::Range;

#[derive(Clone, Debug)]
pub enum ParseError {
    IncorrectToken(Range),
    InvalidIntLiteral(Range),
    InvalidFloatLiteral(Range),
    Expected(&'static str, String, Range),
    Error(&'static str, Range),
    Other(Range),
}

impl ParseError {
    fn span(&self) -> Range {
        match self {
            ParseError::IncorrectToken(span) => *span,
            ParseError::InvalidIntLiteral(span) => *span,
            ParseError::InvalidFloatLiteral(span) => *span,
            ParseError::Expected(_, _, span) => *span,
            ParseError::Error(_, span) => *span,
            ParseError::Other(span) => *span,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::IncorrectToken(..) => f.write_str("Parser encountered incorrect token"),
            ParseError::InvalidIntLiteral(..) => {
                f.write_str("Parser encountered invalid integer literal")
            }
            ParseError::InvalidFloatLiteral(..) => {
                f.write_str("Parser encountered invalid float literal")
            }
            ParseError::Expected(exp, found, ..) => {
                write!(f, "Parser encountered error, expected {} found {}", exp, found)
            }
            ParseError::Error(exp, ..) => {
                write!(f, "Parser encountered error, expected {}", exp)
            }
            ParseError::Other(..) => f.write_str("ICE"),
        }
    }
}

#[derive(Debug)]
pub struct PrettyError<'a> {
    name: &'a str,
    input: &'a str,
    span: Range,
    msg: ParseError,
}

impl<'a> PrettyError<'a> {
    crate fn from_parse(name: &'a str, input: &'a str, err: ParseError) -> String {
        Self { name, input, span: err.span(), msg: err }.to_string()
    }
}

impl<'a> Error for PrettyError<'a> {}

impl<'a> fmt::Display for PrettyError<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        crate::error::Error::error_from_parts(
            self.name,
            self.input,
            self.span,
            &self.msg.to_string(),
        )
        .fmt(f)
    }
}
