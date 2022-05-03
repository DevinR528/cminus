use std::fmt;

use crate::ast::{
	parse::{Token, TokenKind as token},
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
}

impl fmt::Display for AssocOp {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			AssocOp::Add => '+'.fmt(f),
			AssocOp::Subtract => '-'.fmt(f),
			AssocOp::Multiply => '*'.fmt(f),
			AssocOp::Divide => '/'.fmt(f),
			AssocOp::Modulus => '%'.fmt(f),
			AssocOp::LAnd => "&&".fmt(f),
			AssocOp::LOr => "||".fmt(f),
			AssocOp::BitXor => '^'.fmt(f),
			AssocOp::BitAnd => '&'.fmt(f),
			AssocOp::BitOr => '|'.fmt(f),
			AssocOp::ShiftLeft => "<<".fmt(f),
			AssocOp::ShiftRight => ">>".fmt(f),
			AssocOp::Equal => "==".fmt(f),
			AssocOp::Less => '<'.fmt(f),
			AssocOp::LessEqual => "<=".fmt(f),
			AssocOp::NotEqual => "!=".fmt(f),
			AssocOp::Greater => '>'.fmt(f),
			AssocOp::GreaterEqual => ">=".fmt(f),
			AssocOp::Assign => '='.fmt(f),
			AssocOp::AssignOp(op) => format!(
				"{}=",
				match op {
					BinOp::Mul => "*",
					BinOp::Div => "/",
					BinOp::Rem => "%",
					BinOp::Add => "+",
					BinOp::Sub => "-",
					BinOp::LeftShift => "<<",
					BinOp::RightShift => ">>",
					BinOp::BitAnd => "&",
					BinOp::BitXor => "^",
					BinOp::BitOr => "|",
					_ => unreachable!("illegal assign op"),
				}
			)
			.fmt(f),
		}
	}
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
			Multiply | Divide | Modulus => 13,
			Add | Subtract => 12,
			ShiftLeft | ShiftRight => 11,
			BitAnd => 10,
			BitXor => 9,
			BitOr => 8,
			Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => 7,
			LAnd => 6,
			LOr => 5,
			Assign | AssignOp(_) => 2,
		}
	}

	/// Gets the fixity of this operator
	pub fn fixity(&self) -> Fixit {
		use AssocOp::*;
		// NOTE: it is a bug to have an operators that has same precedence but different fixities!
		match *self {
			Assign | AssignOp(_) => Fixit::Right,
			Multiply | Divide | Modulus | Add | Subtract | ShiftLeft | ShiftRight | BitAnd
			| BitXor | BitOr | Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual
			| LAnd | LOr => Fixit::Left,
		}
	}

	pub fn is_comparison(&self) -> bool {
		use AssocOp::*;
		match *self {
			Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual => true,
			Assign | AssignOp(_) | Multiply | Divide | Modulus | Add | Subtract | ShiftLeft
			| ShiftRight | BitAnd | BitXor | BitOr | LAnd | LOr => false,
		}
	}

	pub fn is_assign_like(&self) -> bool {
		use AssocOp::*;
		match *self {
			Assign | AssignOp(_) => true,
			Less | Greater | LessEqual | GreaterEqual | Equal | NotEqual | Multiply | Divide
			| Modulus | Add | Subtract | ShiftLeft | ShiftRight | BitAnd | BitXor | BitOr
			| LAnd | LOr => false,
		}
	}

	pub fn to_ast_binop(self) -> Option<BinOp> {
		use AssocOp::*;
		match self {
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
			Assign | AssignOp(_) => None,
		}
	}
}
