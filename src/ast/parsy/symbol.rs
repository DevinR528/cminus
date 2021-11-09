use crate::ast::types::Range;

use super::ast::DUMMY;

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct Ident {
    span: Range,
    name: String,
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
