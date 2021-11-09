use std::fmt;

use crate::ast::types::Range;

use super::ast::DUMMY;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Ident {
    span: Range,
    name: String,
}

impl PartialEq<str> for Ident {
    fn eq(&self, other: &str) -> bool {
        self.name == other
    }
}
impl PartialEq<Ident> for str {
    fn eq(&self, other: &Ident) -> bool {
        self == other.name
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name.fmt(f)
    }
}

impl Ident {
    pub fn new(span: Range, name: String) -> Self {
        Self { span, name }
    }

    pub fn dummy() -> Self {
        Self { span: DUMMY, name: String::new() }
    }

    pub fn span(&self) -> &Range {
        &self.span
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}
