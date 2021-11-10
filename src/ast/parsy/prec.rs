use crate::ast::{
    parsy::{Token, TokenKind as token},
    types::{BinOp, UnOp},
};

/// Associative operator with precedence.
///
/// This is the enum which specifies operator precedence and fixity to the parser.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum AssocOp {
    /// `+`
    Add,
    /// `-`
    Subtract,
    /// `*`
    Multiply,
    /// `/`
    Divide,
    /// `%`
    Modulus,
    /// `&&`
    LAnd,
    /// `||`
    LOr,
    /// `^`
    BitXor,
    /// `&`
    BitAnd,
    /// `|`
    BitOr,
    /// `<<`
    ShiftLeft,
    /// `>>`
    ShiftRight,
    /// `==`
    Equal,
    /// `<`
    Less,
    /// `<=`
    LessEqual,
    /// `!=`
    NotEqual,
    /// `>`
    Greater,
    /// `>=`
    GreaterEqual,
    /// `=`
    Assign,
    /// `?=` where ? is one of the BinOpToken
    AssignOp(BinOp),
    /// `as`
    As,
    /// `..` range
    DotDot,
    /// `..=` range
    DotDotEq,
    /// `:`
    Colon,
}

#[derive(PartialEq, Debug)]
pub enum Fixit {
    /// The operator is left-associative
    Left,
    /// The operator is right-associative
    Right,
    /// The operator is not associative
    None,
}

impl AssocOp {
    /// Gets the precedence of this operator
    pub fn precedence(&self) -> usize {
        use AssocOp::*;
        match *self {
            As | Colon => 14,
            Multiply | Divide | Modulus => 13,
            Add | Subtract => 12,
            ShiftLeft | ShiftRight => 11,
            BitAnd => 10,
            BitXor => 9,
            BitOr => 8,
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => 7,
            LAnd => 6,
            LOr => 5,
            DotDot | DotDotEq => 4,
            Assign | AssignOp(_) => 2,
        }
    }

    /// Gets the fixity of this operator
    pub fn fixity(&self) -> Fixit {
        use AssocOp::*;
        // NOTE: it is a bug to have an operators that has same precedence but different fixities!
        match *self {
            Assign | AssignOp(_) => Fixit::Right,
            As | Multiply | Divide | Modulus | Add | Subtract | ShiftLeft | ShiftRight | BitAnd
            | BitXor | BitOr | Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual
            | LAnd | LOr | Colon => Fixit::Left,
            DotDot | DotDotEq => Fixit::None,
        }
    }

    pub fn is_comparison(&self) -> bool {
        use AssocOp::*;
        match *self {
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => true,
            Assign | AssignOp(_) | As | Multiply | Divide | Modulus | Add | Subtract
            | ShiftLeft | ShiftRight | BitAnd | BitXor | BitOr | LAnd | LOr | DotDot | DotDotEq
            | Colon => false,
        }
    }

    pub fn is_assign_like(&self) -> bool {
        use AssocOp::*;
        match *self {
            Assign | AssignOp(_) => true,
            Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual | As | Multiply
            | Divide | Modulus | Add | Subtract | ShiftLeft | ShiftRight | BitAnd | BitXor
            | BitOr | LAnd | LOr | DotDot | DotDotEq | Colon => false,
        }
    }

    pub fn to_ast_binop(&self) -> Option<BinOp> {
        use AssocOp::*;
        match *self {
            Less => Some(BinOp::Lt),
            Greater => Some(BinOp::Gt),
            LessEqual => Some(BinOp::Le),
            GreaterEqual => Some(BinOp::Ge),
            Equal => Some(BinOp::Eq),
            NotEqual => Some(BinOp::Ne),
            Multiply => Some(BinOp::Mul),
            Divide => Some(BinOp::Div),
            Modulus => Some(BinOp::Rem),
            Add => Some(BinOp::Add),
            Subtract => Some(BinOp::Sub),
            ShiftLeft => Some(BinOp::LeftShift),
            ShiftRight => Some(BinOp::RightShift),
            BitAnd => Some(BinOp::BitAnd),
            BitXor => Some(BinOp::BitXor),
            BitOr => Some(BinOp::BitOr),
            LAnd => Some(BinOp::And),
            LOr => Some(BinOp::Or),
            Assign | AssignOp(_) | As | DotDot | DotDotEq | Colon => None,
        }
    }
}
