use std::{
	convert::TryInto,
	future::Future,
	pin::Pin,
	sync::mpsc::{channel, Receiver, Sender},
};

use crate::{
	ast::{
		lex::{self, Base, LiteralKind, Token, TokenKind, TokenMatch},
		parse::{
			error::ParseError,
			ops::{AssocOp, Fixit},
			symbol::Ident,
		},
		types::{
			self as ast, AsmBlock, Decl, Expr, FuncKind, Instruction, Location, Path, Spanned,
			Spany, Stmt, Type, Val,
		},
	},
	data_struc::{rawvec::RawVec, str_help::StripEscape},
	gen::asm::inst::{FloatRegister, Register},
	typeck::scope::hash_file,
};

use super::types::Else;

crate mod error;
crate mod kw;
mod ops;
crate mod symbol;

pub type ParseResult<T> = Result<T, ParseError>;

pub type AstSender = Sender<ParseResult<ParsedBlob>>;

pub struct ParsedBlob {
	pub file: &'static str,
	pub input: &'static str,
	pub count: usize,
	pub decl: ast::Declaration,
}
// TODO: this is basically one file = one mod/crate/program unit add mod linking or
// whatever.
/// Create an AST from input `str`.
#[derive(Debug, Default)]
pub struct AstBuilder<'a> {
	tokens: Vec<lex::Token>,
	curr: lex::Token,
	input: &'a str,
	file: &'a str,
	file_id: u64,
	input_idx: usize,

	// HACK: FIXME
	in_match_stmt: bool,

	items: Vec<ast::Declaration>,

	call_stack: Vec<&'static str>,
	stack_idx: usize,

	snd: Option<AstSender>,
}

// FIXME: audit the whitespace eating, pretty sure I call it unnecessarily
impl<'a> AstBuilder<'a> {
	pub fn new(input: &'a str, file: &'a str, snd: AstSender) -> Self {
		let mut tokens =
			lex::tokenize(input).chain(Some(Token::new(TokenKind::Eof, 0))).collect::<Vec<_>>();
		let curr = tokens.remove(0);
		Self {
			tokens,
			curr,
			input,
			file,
			file_id: hash_file(file),
			snd: Some(snd),
			..Default::default()
		}
	}

	pub fn items(&self) -> &[ast::Declaration] {
		&self.items
	}

	pub fn into_items(self) -> Vec<ast::Declaration> {
		self.items
	}

	fn push_call_stack(&mut self, s: &'static str) {
		self.call_stack.push(s)
	}

	pub fn parse(&mut self) -> ParseResult<()> {
		self.eat_whitespace();
		loop {
			if self.curr.kind == TokenKind::Eof {
				break;
			}

			match self.curr.kind {
				// Ignore
				TokenKind::LineComment { .. }
				| TokenKind::BlockComment { .. }
				| TokenKind::Whitespace { .. } => {
					self.eat_tkn();
					continue;
				}
				TokenKind::Ident => {
					let keyword: kw::Keywords = self.input_curr().try_into()?;
					match keyword {
						kw::Const => {
							let item = self.parse_const()?;
							self.items.push(item);
						}
						kw::Fn => {
							let item = self.parse_fn()?;
							self.items.push(item);
						}
						kw::Linked => {
							let item = self.parse_linked_fn()?;
							self.items.push(item);
						}
						kw::Impl => {
							let item = self.parse_impl()?;
							self.items.push(item);
						}
						kw::Struct => {
							let item = self.parse_struct()?;
							self.items.push(item);
						}
						kw::Enum => {
							let item = self.parse_enum()?;
							self.items.push(item);
						}
						kw::Trait => {
							let item = self.parse_trait()?;
							self.items.push(item);
						}
						kw::Import => {
							let start = self.curr_span();
							let item = self.parse_import()?;

							if let ast::Decl::Import(path) = &item.val {
								// TODO: handle ::foo::bar::item; not just ::foo::item;
								let path = path.segs[0];
								let mut p = std::path::PathBuf::from(self.file);

								// This is always valid it's just so AstBuilder can impl Default
								let snd = self.snd.as_ref().unwrap().clone();
								std::thread::spawn(move || {
									p.pop();
									p.push(path.name());
									p.set_extension("cm");

									let s = p.to_string_lossy().to_string();
									let input = std::fs::read_to_string(&s).map_err(|e| {
										ParseError::Error("invalid file name", start)
									})?;

									let mut parser = AstBuilder::new(&input, &s, snd.clone());
									if let Err(err) = parser.parse() {
										snd.send(Err(err));
										return Ok::<_, ParseError>(());
									}

									let mut cnt = parser.items().len();
									let items = parser.into_items();
									let file = Box::leak(box s);
									let input = Box::leak(box input);
									// TODO: the receiver has to wait on all items so we really
									// don't get much benefit from threading now
									for item in items {
										cnt -= 1;
										snd.send(Ok(ParsedBlob {
											file,
											input,
											count: cnt,
											decl: item,
										}));
									}
									Ok(())
								});
							} else {
								return Err(ParseError::Error("malformed import", start));
							}

							self.items.push(item);
						}
						tkn => unreachable!("Token is unaccounted for `{}`", tkn.text()),
					}
				}
				TokenKind::Pound => {
					self.eat_attr();
					continue;
				}
				TokenKind::CloseBrace => {
					self.eat_if(&TokenMatch::CloseBrace);
					if self.curr.kind == TokenKind::Eof {
						break;
					}
				}
				TokenKind::Unknown => {
					return Err(ParseError::Error("encountered unknown token", self.curr_span()))
				}
				_ => {
					return Err(ParseError::Error("encountered incorrect token", self.curr_span()))
				}
			}
			self.eat_whitespace();
		}
		Ok(())
	}

	// Parse `const [mut] name: type = expr;`
	fn parse_const(&mut self) -> ParseResult<ast::Declaration> {
		self.push_call_stack("parse_const");
		let start = self.input_idx;

		self.eat_if_kw(kw::Const);
		self.eat_whitespace();

		let mutable = self.eat_if_kw(kw::Mut);
		self.eat_whitespace();

		let id = self.make_ident()?;
		self.eat_whitespace();

		self.eat_if(&TokenMatch::Colon);
		self.eat_whitespace();

		let ty = self.make_ty()?;

		self.eat_whitespace();
		self.eat_if(&TokenMatch::Eq);
		self.eat_whitespace();

		let init = self.make_expr()?;

		self.eat_whitespace();
		self.eat_if(&TokenMatch::Semi);

		let span = ast::to_rng(start..self.input_idx, self.file_id);
		Ok(ast::Decl::Const(ast::Const { ident: id, ty, init, mutable, span }).into_spanned(span))
	}

	// Parse `fn name<T>(it: T) -> int { .. }` with or without generics.
	fn parse_fn(&mut self) -> ParseResult<ast::Declaration> {
		self.push_call_stack("parse_fn");
		let start = self.input_idx;

		self.eat_if_kw(kw::Fn);
		self.eat_whitespace();

		let ident = self.make_ident()?;

		let generics = self.make_generics()?;

		self.eat_if(&TokenMatch::OpenParen);
		self.eat_whitespace();

		let params =
			if !self.eat_if(&TokenMatch::CloseParen) { self.make_params()? } else { vec![] };

		self.eat_if(&TokenMatch::CloseParen);
		self.eat_whitespace();

		let ret = if self.eat_if(&TokenMatch::Colon) {
			self.eat_whitespace();
			self.make_ty()?
		} else {
			self.eat_whitespace();
			ast::Ty::Void.into_spanned(self.curr_span())
		};
		self.eat_whitespace();

		let stmts = self.make_block()?;

		let span = ast::to_rng(start..self.input_idx(), self.file_id);

		Ok(ast::Decl::Func(ast::Func {
			ident,
			ret: crate::rawptr!(ret),
			generics,
			params,
			stmts,
			kind: FuncKind::Normal,
			span,
		})
		.into_spanned(span))
	}

	// Parse `fn name<T>(it: T) -> int { .. }` with or without generics.
	fn parse_linked_fn(&mut self) -> ParseResult<ast::Declaration> {
		self.push_call_stack("parse_linked_fn");
		let start = self.input_idx;

		self.eat_if_kw(kw::Linked);
		self.eat_whitespace();

		self.eat_if_kw(kw::Fn);
		self.eat_whitespace();

		let ident = self.make_ident()?;

		let generics = self.make_generics()?;

		self.eat_if(&TokenMatch::OpenParen);
		self.eat_whitespace();

		let params =
			if !self.eat_if(&TokenMatch::CloseParen) { self.make_params()? } else { vec![] };

		self.eat_if(&TokenMatch::CloseParen);
		self.eat_whitespace();

		let ret = if self.eat_if(&TokenMatch::Colon) {
			self.eat_whitespace();
			self.make_ty()?
		} else {
			self.eat_whitespace();
			ast::Ty::Void.into_spanned(self.curr_span())
		};
		self.eat_whitespace();

		let span = self.curr_span();
		let stmts = if self.eat_if(&TokenMatch::Semi) {
			ast::Block { stmts: crate::raw_vec![], span }
		} else {
			self.make_block()?
		};

		let span = ast::to_rng(start..self.input_idx, self.file_id);

		Ok(ast::Decl::Func(ast::Func {
			ident,
			ret: crate::rawptr!(ret),
			generics,
			params,
			stmts,
			kind: FuncKind::Linked,
			span,
		})
		.into_spanned(span))
	}

	fn parse_impl(&mut self) -> ParseResult<ast::Declaration> {
		self.push_call_stack("parse_fn");
		let start = self.input_idx;

		self.eat_if_kw(kw::Impl);
		self.eat_whitespace();

		let path = self.make_path()?;
		let type_arguments = self.make_types(&TokenMatch::Lt, &TokenMatch::Gt)?;

		self.eat_if(&TokenMatch::OpenBrace);
		self.eat_whitespace();

		let method = if let ast::Decl::Func(func) = self.parse_fn()?.val {
			func
		} else {
			unreachable!("we should error before this [parse func in impl]")
		};

		self.eat_if(&TokenMatch::CloseBrace);
		let span = ast::to_rng(start..self.input_idx, self.file_id);
		Ok(ast::Decl::Impl(ast::Impl { path, type_arguments, method, span }).into_spanned(span))
	}

	fn parse_struct(&mut self) -> ParseResult<ast::Declaration> {
		self.push_call_stack("parse_struct");
		let start = self.input_idx;

		self.eat_if_kw(kw::Struct);
		self.eat_whitespace();

		let ident = self.make_ident()?;
		let generics = self.make_generics()?;

		self.eat_if(&TokenMatch::OpenBrace);
		self.eat_whitespace();

		let fields = self.make_fields()?;

		self.eat_if(&TokenMatch::CloseBrace);
		let span = ast::to_rng(start..self.input_idx, self.file_id);
		Ok(ast::Decl::Adt(ast::Adt::Struct(ast::Struct { ident, fields, generics, span }))
			.into_spanned(span))
	}

	fn parse_enum(&mut self) -> ParseResult<ast::Declaration> {
		self.push_call_stack("parse_enum");
		let start = self.input_idx;

		self.eat_if_kw(kw::Enum);
		self.eat_whitespace();

		let ident = self.make_ident()?;
		let generics = self.make_generics()?;

		self.eat_if(&TokenMatch::OpenBrace);
		self.eat_whitespace();

		let variants = self.make_variants()?;

		self.eat_if(&TokenMatch::CloseBrace);
		let span = ast::to_rng(start..self.input_idx, self.file_id);
		Ok(ast::Decl::Adt(ast::Adt::Enum(ast::Enum { ident, variants, generics, span }))
			.into_spanned(span))
	}

	fn parse_trait(&mut self) -> ParseResult<ast::Declaration> {
		self.push_call_stack("parse_trait");
		let start = self.input_idx;

		self.eat_if_kw(kw::Trait);
		self.eat_whitespace();

		// This will never be a path since this is always a local decl but
		// we need this to be a path when we trait solve and for impls
		let path = self.make_path()?;
		let generics = self.make_generics()?;

		self.eat_if(&TokenMatch::OpenBrace);
		self.eat_whitespace();

		let method = self.make_trait_fn(&generics)?;

		self.eat_if(&TokenMatch::CloseBrace);
		let span = ast::to_rng(start..self.input_idx, self.file_id);
		Ok(ast::Decl::Trait(ast::Trait { path, generics, method, span }).into_spanned(span))
	}

	fn parse_import(&mut self) -> ParseResult<ast::Declaration> {
		self.push_call_stack("parse_import");
		let start = self.input_idx;

		self.eat_if_kw(kw::Import);
		self.eat_whitespace();

		let path = self.make_path()?;

		self.eat_if(&TokenMatch::Semi);

		let span = ast::to_rng(start..self.input_idx, self.file_id);
		Ok(ast::Decl::Import(path).into_spanned(span))
	}

	/// Parse `fn name(it: T) -> int;` with or without generics.
	///
	/// The `gen` list allows the function to use the generics declared at the trait scope.
	fn make_trait_fn(&mut self, gens: &[ast::Generic]) -> ParseResult<ast::TraitMethod> {
		self.push_call_stack("make_trait_fn");
		let start = self.input_idx;

		self.eat_if_kw(kw::Fn);
		self.eat_whitespace();

		let ident = self.make_ident()?;

		self.eat_if(&TokenMatch::OpenParen);
		self.eat_whitespace();

		let mut params =
			if !self.eat_if(&TokenMatch::CloseParen) { self.make_params()? } else { vec![] };

		self.eat_if(&TokenMatch::CloseParen);
		self.eat_whitespace();

		let ret = if self.eat_if(&TokenMatch::Colon) {
			self.eat_whitespace();
			self.make_ty()?
		} else {
			self.eat_whitespace();
			ast::Ty::Void.into_spanned(self.curr_span())
		};
		self.eat_whitespace();

		let default = self.eat_if(&TokenMatch::Semi);
		Ok(if default {
			let stmts = self.make_block()?;
			let span = ast::to_rng(start..self.input_idx, self.file_id);
			ast::TraitMethod::Default(ast::Func {
				ident,
				ret: crate::rawptr!(ret),
				generics: gens.to_vec(),
				params,
				stmts,
				kind: FuncKind::Normal,
				span,
			})
		} else {
			let stmts = ast::Block { stmts: crate::raw_vec![], span: self.curr_span() };
			let span = ast::to_rng(start..self.input_idx, self.file_id);
			ast::TraitMethod::NoBody(ast::Func {
				ident,
				ret: crate::rawptr!(ret),
				generics: gens.to_vec(),
				params,
				stmts,
				kind: FuncKind::EmptyTrait,
				span,
			})
		})
	}

	/// Parse `name[ws]([ws]type[ws],...)[ws], name(a, b), ..`.
	fn make_variants(&mut self) -> ParseResult<Vec<ast::Variant>> {
		self.push_call_stack("make_variants");

		let mut variants = vec![];
		loop {
			let start = self.input_idx;

			// We have reached the end of the enum def (this allows trailing commas)
			if self.curr.kind == TokenMatch::CloseBrace {
				break;
			}

			let ident = self.make_ident()?;
			let types = self.make_types(&TokenMatch::OpenParen, &TokenMatch::CloseParen)?;

			let span = ast::to_rng(start..self.input_idx, self.file_id);
			variants.push(ast::Variant { ident, types: RawVec::from_vec(types), span });

			// TODO: report errors when missing commas if possible
			self.eat_whitespace();
			if self.eat_if(&TokenMatch::Comma) {
				self.eat_whitespace();
				continue;
			} else {
				break;
			}
		}

		self.eat_whitespace();
		Ok(variants)
	}

	/// Parse `name[ws]:[ws]type[ws],[ws]...`
	fn make_fields(&mut self) -> ParseResult<Vec<ast::Field>> {
		self.push_call_stack("make_fields");
		let mut params = vec![];
		loop {
			// We have reached the end of the struct def (this allows trailing commas)
			if self.curr.kind == TokenMatch::CloseBrace {
				break;
			}
			let start = self.input_idx;
			let ident = self.make_ident()?;

			self.eat_if(&TokenMatch::Colon);
			self.eat_whitespace();

			let ty = self.make_ty()?;

			let span = ast::to_rng(start..self.input_idx(), self.file_id);
			params.push(ast::Field { ident, ty: crate::rawptr!(ty), span });

			self.eat_whitespace();
			if self.eat_if(&TokenMatch::Comma) {
				self.eat_whitespace();
				continue;
			} else {
				break;
			}
		}

		self.eat_whitespace();
		Ok(params)
	}

	/// parse a list of expressions.
	fn make_field_list(&mut self) -> ParseResult<Vec<ast::FieldInit>> {
		self.push_call_stack("make_field_list");

		Ok(if self.eat_if(&TokenMatch::OpenBrace) {
			let mut exprs = vec![];
			loop {
				self.eat_whitespace();
				// We have reached the end of the list (trailing comma or empty)
				if self.eat_if(&TokenMatch::CloseBrace) {
					break;
				}
				let start = self.input_idx;
				let ident = self.make_ident()?;
				self.eat_whitespace();
				self.eat_if(&TokenMatch::Colon);
				self.eat_whitespace();
				let init = self.make_expr()?;
				let span = ast::to_rng(start..self.input_idx(), self.file_id);
				exprs.push(ast::FieldInit { ident, init, span });
				if self.eat_if(&TokenMatch::Comma) {
					self.eat_whitespace();
					continue;
				} else {
					break;
				}
			}
			self.eat_if(&TokenMatch::CloseBrace);
			self.eat_whitespace();
			exprs
		} else {
			return Err(ParseError::Error(
				"expected `{` for struct init expression",
				self.curr_span(),
			));
		})
	}

	/// parse a list of expressions, eats trailing whitespace.
	fn make_expr_list(
		&mut self,
		open: &TokenMatch,
		close: &TokenMatch,
	) -> ParseResult<Vec<ast::Expression>> {
		self.push_call_stack("make_expr_list");

		Ok(if self.eat_if(open) {
			let mut exprs = vec![];
			loop {
				self.eat_whitespace();
				// We have reached the end of the list (trailing comma or empty)
				if self.eat_if(close) {
					break;
				}
				exprs.push(self.make_expr()?);
				if self.eat_if(&TokenMatch::Comma) {
					self.eat_whitespace();
					continue;
				} else {
					break;
				}
			}
			self.eat_if(close);
			self.eat_whitespace();
			exprs
		} else {
			vec![]
		})
	}

	/// This handles top-level expressions.
	///
	/// - array init
	/// - enum/struct init
	/// - tuples (eventually)
	/// - expression tress
	fn make_expr(&mut self) -> ParseResult<ast::Expression> {
		self.push_call_stack("make_expr");
		self.eat_whitespace();
		let start = self.input_idx;

		// array init
		if self.curr.kind == TokenMatch::OpenBracket {
			if self.cmp_seq_ignore_ws(&[
				TokenMatch::OpenBracket,
				TokenMatch::Literal,
				TokenMatch::Semi,
			]) {
				self.eat_if(&TokenMatch::OpenBracket);
				self.eat_whitespace();

				let lit = self.make_literal()?;
				self.eat_whitespace();

				self.eat_if(&TokenMatch::Semi);
				self.eat_whitespace();

				let count = if let ast::Val::Int(int) = self.make_literal()?.val {
					int as usize
				} else {
					return Err(ParseError::Error(
						"array literal must must be of the forum`[expr; int]`",
						ast::to_rng(start..self.input_idx, self.file_id),
					));
				};
				self.eat_whitespace();
				self.eat_if(&TokenMatch::CloseBracket);

				let span = ast::to_rng(start..self.input_idx(), self.file_id);
				let lit_span = lit.span;
				Ok(ast::Expr::ArrayInit {
					items: std::iter::repeat(ast::Expr::Value(lit).into_spanned(lit_span))
						.take(count)
						.collect(),
				}
				.into_spanned(span))
			} else {
				let items =
					self.make_expr_list(&TokenMatch::OpenBracket, &TokenMatch::CloseBracket)?;
				let span = ast::to_rng(start..self.input_idx(), self.file_id);
				Ok(ast::Expr::ArrayInit { items }.into_spanned(span))
			}
		} else if self.curr.kind == TokenMatch::Lt {
			// trait method calls
			let start = self.input_idx;

			// Outer `<` token
			self.eat_if(&TokenMatch::Lt);
			let type_args = self.make_type_args()?;

			let trait_ = self.make_path()?;

			self.eat_whitespace();
			self.eat_if(&TokenMatch::Gt);

			self.eat_if(&TokenMatch::OpenParen);
			let mut args = self.make_arg_list()?;
			// This is duplicated iff we have a no arg call
			self.eat_whitespace();
			self.eat_if(&TokenMatch::CloseParen);

			let span = ast::to_rng(start..self.input_idx, self.file_id);
			Ok(ast::Expr::TraitMeth { trait_, type_args, args }.into_spanned(span))
		} else if self.curr.kind == TokenMatch::At {
			let start = self.input_idx;

			self.eat_if(&TokenMatch::At);
			Ok(match self.input_curr() {
				"bottom" => {
					self.eat_if(&TokenMatch::Ident);
					ast::Expr::Builtin(ast::Builtin::Bottom)
				}
				"size_of" => {
					self.eat_if(&TokenMatch::Ident);
					let mut ty = self.make_type_args()?;
					if ty.len() != 1 {
						return Err(ParseError::Error(
							"@size_of takes one type argument",
							self.curr_span(),
						));
					}
					ast::Expr::Builtin(ast::Builtin::SizeOf(crate::rawptr!(ty.remove(0))))
				}
				_ => {
					return Err(ParseError::Error("builtin", self.curr_span()));
				}
			}
			.into_spanned(ast::to_rng(start..self.input_idx, self.file_id)))
		} else if matches!(
			self.curr.kind,
			TokenKind::Ident
				| TokenKind::Literal { .. } // cstr, numbers, etc.
				| TokenKind::Star           // deref
				| TokenKind::Minus          // negative numbers
				| TokenKind::And            // addrof
				| TokenKind::Bang           // not
				| TokenKind::Tilde          // urnary negate
				| TokenKind::OpenParen // a parenthesized expression
		) {
			// FIXME: we don't want to have to say `let x = enum foo::bar;` just `let x = foo::bar;`
			// TODO: don't parse for keywords if its a lit DUH
			let x: Result<kw::Keywords, _> = self.input_curr().try_into();
			if let Ok(key) = x {
				match key {
					// kw::Enum => self.make_enum_init(),
					// kw::Struct => self.make_struct_init(),
					kw::True => {
						return Ok(
							ast::Expr::Value(self.make_literal()?).into_spanned(self.curr_span())
						);
					}
					kw::False => {
						return Ok(
							ast::Expr::Value(self.make_literal()?).into_spanned(self.curr_span())
						);
					}
					t => {
						return Err(ParseError::Error("unexpected keyword", self.curr_span()));
					}
				}
			}
			// TODO: factor out
			// Shunting Yard algo http://en.wikipedia.org/wiki/Shunting_yard_algorithm
			let mut output: Vec<ast::Expression> = vec![];
			let mut opstack: Vec<AssocOp> = vec![];
			while self.curr.kind != TokenMatch::Semi {
				self.eat_whitespace();
				let (ex, op) = self.advance_to_op()?;
				self.eat_whitespace();
				if let Some(next) = op {
					// if the previous operator is of a higher precedence than the incoming
					match opstack.pop() {
						Some(prev)
							if next.precedence() < prev.precedence()
								|| (next.precedence() == prev.precedence()
									&& matches!(prev.fixity(), Fixit::Left)) =>
						{
							let lhs = output.pop().unwrap();

							let span = ast::to_rng(lhs.span.start..ex.span.end, self.file_id);
							let mut finish = ast::Expr::Binary {
								op: prev.to_ast_binop().unwrap(),
								lhs: box lhs,
								rhs: box ex,
							}
							.into_spanned(span);

							while let Some(lfix_op) = opstack.pop() {
								if matches!(lfix_op.fixity(), Fixit::Left) {
									let lhs = output.pop().unwrap();
									let span =
										ast::to_rng(lhs.span.start..finish.span.end, self.file_id);
									finish = ast::Expr::Binary {
										op: lfix_op.to_ast_binop().unwrap(),
										lhs: box lhs,
										rhs: box finish,
									}
									.into_spanned(span);
								} else {
									opstack.push(lfix_op);
									break;
								}
							}

							output.push(finish);
							opstack.push(next);
						}
						Some(prev) => {
							output.push(ex);
							opstack.push(prev);
							opstack.push(next);
						}
						None => {
							output.push(ex);
							opstack.push(next);
						}
					}
				// TODO: cleanup
				// There is probably a better way to do this
				} else {
					output.push(ex);

					// We have to process the stack/output and the last (expr, binop) pair
					loop {
						match (output.pop(), opstack.pop()) {
							(Some(rhs), Some(bin)) => {
								if let Some(first) = output.pop() {
									let span =
										ast::to_rng(first.span.start..rhs.span.end, self.file_id);
									let finish = ast::Expr::Binary {
										op: bin.to_ast_binop().unwrap(),
										lhs: box first,
										rhs: box rhs,
									}
									.into_spanned(span);

									output.push(finish);
								} else {
									break;
								}
							}
							(Some(expr), None) => {
								return Ok(expr);
							}
							(None, Some(op)) => {
								return Err(ParseError::Expected(
									"a term",
									format!("only the {} operand", op),
									self.curr_span(),
								));
							}
							(None, None) => {
								unreachable!("dont think this is possible")
							}
						}
					}
					break;
				}
			}

			output
				.pop()
				.ok_or_else(|| ParseError::Error("failed to generate expression", self.curr_span()))
		} else {
			// panic!("{:?}", self.curr);
			// See the above 4 todos/fixes
			Err(ParseError::Error("no top level expression", self.curr_span()))
		}
	}

	/// Helper to build a "left" hand expression and an optional `AssocOp`.
	///
	/// - anything with that starts with an ident
	///     - field access, calls, etc.
	/// - literals
	/// - check for negation and not
	/// - pointers
	/// - addrof maybe
	fn advance_to_op(&mut self) -> ParseResult<(ast::Expression, Option<AssocOp>)> {
		self.push_call_stack("advance_to_op");
		Ok(if self.curr.kind == TokenMatch::Ident {
			let id = self.make_lh_expr()?;
			self.eat_whitespace();

			let op = self.make_op()?;
			(id, op)
		} else if matches!(self.curr.kind, TokenKind::Minus | TokenKind::Literal { .. }) {
			let start = self.input_idx;
			let ex = ast::Expr::Value(self.make_literal()?)
				.into_spanned(ast::to_rng(start..self.input_idx, self.file_id));
			self.eat_whitespace();

			let op = self.make_op()?;
			(ex, op)
		} else if self.curr.kind == TokenMatch::Bang {
			// Not `!expr`
			let start = self.input_idx;

			self.eat_if(&TokenMatch::Bang);
			let ex = self.make_expr()?;
			let expr = ast::Expr::Urnary { op: ast::UnOp::Not, expr: box ex }
				.into_spanned(ast::to_rng(start..self.input_idx, self.file_id));

			self.eat_whitespace();

			let op = self.make_op()?;
			(expr, op)
		} else if self.curr.kind == TokenMatch::Minus {
			// negative number `-10;`
			let start = self.input_idx;

			self.eat_if(&TokenMatch::Minus);
			let mut lit = self.make_literal()?;
			match lit.val {
				ast::Val::Int(ref mut v) => {
					*v = -(*v);
				}
				ast::Val::Float(ref mut v) => {
					*v = -(*v);
				}
				_ => {
					return Err(ParseError::Error(
						"float or int can be negative",
						ast::to_rng(start..self.input_idx, self.file_id),
					));
				}
			}
			let ex = ast::Expr::Value(lit)
				.into_spanned(ast::to_rng(start..self.input_idx, self.file_id));
			self.eat_whitespace();

			let op = self.make_op()?;
			(ex, op)
		} else if self.curr.kind == TokenMatch::Tilde {
			// Ones comp `~expr`
			let start = self.input_idx;

			self.eat_if(&TokenMatch::Tilde);
			let ex = self.make_expr()?;
			let expr = ast::Expr::Urnary { op: ast::UnOp::OnesComp, expr: box ex }
				.into_spanned(ast::to_rng(start..self.input_idx, self.file_id));

			self.eat_whitespace();

			let op = self.make_op()?;
			(expr, op)
		} else if self.curr.kind == TokenMatch::Star {
			let start = self.input_idx;

			let mut indir = 0;
			loop {
				if self.eat_if(&TokenMatch::Star) {
					indir += 1;
				} else {
					break;
				}
			}
			let id = self.make_lh_expr()?;
			let expr = ast::Expr::Deref { indir, expr: box id }
				.into_spanned(ast::to_rng(start..self.input_idx, self.file_id));

			self.eat_whitespace();

			let op = self.make_op()?;
			(expr, op)
		} else if self.curr.kind == TokenMatch::And {
			let start = self.input_idx;

			self.eat_if(&TokenMatch::And);
			let ex = self.make_expr()?;
			let expr = ast::Expr::AddrOf(box ex)
				.into_spanned(ast::to_rng(start..self.input_idx, self.file_id));

			self.eat_whitespace();

			let op = self.make_op()?;
			(expr, op)
		} else if self.curr.kind == TokenMatch::OpenParen {
			let start = self.input_idx;
			// N.B.
			// We know we are in the middle of some kind of binop
			self.eat_if(&TokenMatch::OpenParen);

			let ex = self.make_expr()?;

			let expr = ast::Expr::Parens(box ex)
				.into_spanned(ast::to_rng(start..self.input_idx, self.file_id));
			self.eat_whitespace();

			self.eat_if(&TokenMatch::CloseParen);
			self.eat_whitespace();

			let op = self.make_op()?;
			(expr, op)
		} else {
			return Err(ParseError::Error("lit, ptr, parens, ones comp, nots", self.curr_span()));
		})
	}

	/// Builds left hand expressions.
	///
	/// - idents
	/// - field access
	/// - array index
	/// - fn call
	fn make_lh_expr(&mut self) -> ParseResult<ast::Expression> {
		self.push_call_stack("make_lh_expr");
		Ok(if matches!(self.curr.kind, TokenKind::Ident | TokenKind::Star) {
			let start = self.input_idx;

			if self.curr.kind == TokenMatch::Star {
				let indir = self.count_eaten_seq(&TokenMatch::Star);

				ast::Expr::Deref { indir, expr: box self.make_lh_expr()? }
					.into_spanned(ast::to_rng(start..self.input_idx, self.file_id))
			} else if self.check_next(&TokenMatch::Dot) {
				// We are in a field access
				let lhs = ast::Expr::Ident(self.make_ident()?)
					.into_spanned(ast::to_rng(start..self.input_idx, self.file_id));
				self.eat_if(&TokenMatch::Dot);

				ast::Expr::FieldAccess { lhs: box lhs, rhs: box self.make_lh_expr()? }
					.into_spanned(ast::to_rng(start..self.input_idx, self.file_id))
			} else if self.check_next(&TokenMatch::OpenBracket) {
				// We are in an array index expr
				let start = self.input_idx;

				let ident = self.make_ident()?;

				self.eat_if(&TokenMatch::OpenBracket);
				self.eat_whitespace();

				let mut exprs = vec![];
				loop {
					exprs.push(self.make_expr()?);
					self.eat_whitespace();
					if self.eat_seq_ignore_ws(&[TokenMatch::CloseBracket, TokenMatch::OpenBracket])
					{
						self.eat_whitespace();
						continue;
					} else {
						break;
					}
				}
				self.eat_if(&TokenMatch::CloseBracket);

				ast::Expr::Array { ident, exprs }
					.into_spanned(ast::to_rng(start..self.input_idx, self.file_id))
			} else {
				self.eat_whitespace();
				let start = self.input_idx;

				let mut path = self.make_path()?;
				let is_func_call = (self.cmp_seq_ignore_ws(&[
					TokenMatch::Colon,
					TokenMatch::Colon,
					TokenMatch::Lt,
				]) || self.curr.kind == TokenMatch::OpenParen);

				// TODO: there has GOT to be a clearer way of writing this...
				let is_struct_in_match = if self.in_match_stmt {
					// FIXME: this would NOT catch `match struct_foo {} {}` where `struct_foo` is
					// empty
					self.cmp_seq_ignore_ws(&[
						TokenMatch::Ident,
						TokenMatch::Colon,
						TokenMatch::Ident,
						TokenMatch::Comma,
					])
				} else {
					false
				};
				let is_struct = !path.segs.is_empty()
					&& ((self.curr.kind == TokenMatch::OpenBrace && !self.in_match_stmt)
						|| is_struct_in_match);

				let is_itemless_enum = path.segs.len() > 1 && self.curr.kind == TokenMatch::Semi;

				if path.segs.len() == 1 && !is_func_call && !is_struct && !is_itemless_enum {
					ast::Expr::Ident(path.segs.remove(0))
						.into_spanned(ast::to_rng(start..self.input_idx, self.file_id))
				} else if is_func_call {
					// We are most likely in a function call
					let type_args = self.make_type_args()?;

					self.eat_whitespace();
					self.eat_if(&TokenMatch::OpenParen);

					let mut args = self.make_arg_list()?;
					// This is duplicated iff we have a no arg call
					self.eat_whitespace();
					self.eat_if(&TokenMatch::CloseParen);

					// TODO an enum that is just a path
					// This can be an enum also
					ast::Expr::Call { path, type_args: RawVec::from_vec(type_args), args }
						.into_spanned(ast::to_rng(start..self.input_idx, self.file_id))
				} else if is_struct {
					let fields = self.make_field_list()?;
					let span = ast::to_rng(start..self.input_idx, self.file_id);
					ast::Expr::StructInit { path, fields }.into_spanned(span)
				} else if is_itemless_enum {
					let variant = path.segs.pop().unwrap();
					let span = ast::to_rng(start..self.input_idx, self.file_id);
					ast::Expr::EnumInit { path, variant, items: vec![] }.into_spanned(span)
				} else {
					return Err(ParseError::Error(
						"ident, field access, array index or fn call",
						self.curr_span(),
					));
				}
			}
		} else {
			return Err(ParseError::Error(
				"ident, field access, array index or fn call",
				self.curr_span(),
			));
		})
	}

	/// Build an optional `AssocOp`.
	fn make_op(&mut self) -> ParseResult<Option<AssocOp>> {
		self.push_call_stack("make_op");
		Ok(match self.curr.kind {
			TokenKind::Eq => {
				self.eat_if(&TokenMatch::Eq);
				if self.eat_if(&TokenMatch::Eq) {
					Some(AssocOp::Equal)
				} else {
					None
				}
			}
			TokenKind::Bang => {
				self.eat_if(&TokenMatch::Bang);
				if self.eat_if(&TokenMatch::Eq) {
					Some(AssocOp::NotEqual)
				} else {
					unreachable!("urnary expr parsed before make_op")
				}
			}
			TokenKind::Lt => {
				self.eat_if(&TokenMatch::Lt);
				// @copypast match bellow
				match self.curr.kind {
					TokenKind::Lt => {
						self.eat_if(&TokenMatch::Lt);
						Some(AssocOp::ShiftLeft)
					}
					TokenKind::Eq => {
						self.eat_if(&TokenMatch::Eq);
						Some(AssocOp::LessEqual)
					}
					// valid ends of `<` token
					TokenKind::Ident | TokenKind::Literal { .. } | TokenKind::Whitespace { .. } => {
						Some(AssocOp::Less)
					}
					_ => todo!(),
				}
			}
			TokenKind::Gt => {
				self.eat_if(&TokenMatch::Gt);
				match self.curr.kind {
					TokenKind::Gt => {
						self.eat_if(&TokenMatch::Gt);
						Some(AssocOp::ShiftRight)
					}
					TokenKind::Eq => {
						self.eat_if(&TokenMatch::Eq);
						Some(AssocOp::GreaterEqual)
					}
					// valid ends of `>` token
					TokenKind::Ident | TokenKind::Literal { .. } | TokenKind::Whitespace { .. } => {
						Some(AssocOp::Greater)
					}
					_ => todo!(),
				}
			}
			TokenKind::Plus => {
				self.eat_if(&TokenMatch::Plus);
				Some(AssocOp::Add)
			}
			TokenKind::Minus => {
				self.eat_if(&TokenMatch::Minus);
				Some(AssocOp::Subtract)
			}
			TokenKind::Star => {
				self.eat_if(&TokenMatch::Star);
				Some(AssocOp::Multiply)
			}
			TokenKind::Slash => {
				self.eat_if(&TokenMatch::Slash);
				Some(AssocOp::Divide)
			}
			TokenKind::Percent => {
				self.eat_if(&TokenMatch::Percent);
				Some(AssocOp::Modulus)
			}
			TokenKind::And => {
				self.eat_if(&TokenMatch::And);
				if self.eat_if(&TokenMatch::And) {
					Some(AssocOp::LAnd)
				} else {
					Some(AssocOp::BitAnd)
				}
			}
			TokenKind::Or => {
				self.eat_if(&TokenMatch::Or);
				if self.eat_if(&TokenMatch::Or) {
					Some(AssocOp::LOr)
				} else {
					Some(AssocOp::BitOr)
				}
			}
			TokenKind::Caret => {
				self.eat_if(&TokenMatch::Caret);
				Some(AssocOp::BitXor)
			}
			TokenKind::CloseParen => {
				// self.eat_if(&TokenMatch::CloseParen);
				None
			}
			TokenKind::CloseBracket => {
				// self.eat_if(&TokenMatch::CloseBracket);
				None
			}
			TokenKind::CloseBrace => {
				// self.eat_if(&TokenMatch::CloseBrace);
				None
			}
			// The stops expressions at `match (expr)` sites
			TokenKind::OpenBrace => None,
			// comma: argument lists, array elements, any initializer stuff (structs, enums)
			// semi: we need to recurse out of our expression tree before we eat this token
			TokenKind::Comma | TokenKind::Semi => None,
			t => return Err(ParseError::Error("unexpected token found", self.curr_span())),
		})
	}

	fn make_block(&mut self) -> ParseResult<ast::Block> {
		self.push_call_stack("make_block");
		let start = self.input_idx;
		let mut stmts = crate::raw_vec![];

		// If the function body is empty
		if self.cmp_seq_ignore_ws(&[TokenMatch::OpenBrace, TokenMatch::CloseBrace]) {
			self.eat_seq_ignore_ws(&[TokenMatch::OpenBrace, TokenMatch::CloseBrace]);
			let span = ast::to_rng(start..self.input_idx, self.file_id);
			return Ok(ast::Block {
				stmts: crate::raw_vec![ast::Stmt::Exit.into_spanned(span)],
				span,
			});
		}

		self.eat_whitespace();
		if self.eat_if(&TokenMatch::OpenBrace) {
			loop {
				self.eat_whitespace();
				// Empty loop body
				if self.curr.kind == TokenMatch::CloseBrace {
					self.eat_if(&TokenMatch::CloseBrace);
					break;
				}
				stmts.push(self.make_stmt()?);
				self.eat_whitespace();

				if self.eat_if(&TokenMatch::CloseBrace) {
					break;
				}
			}
		}
		let span = ast::to_rng(start..self.input_idx(), self.file_id);
		Ok(ast::Block { stmts, span })
	}

	fn make_stmt(&mut self) -> ParseResult<ast::Statement> {
		self.push_call_stack("make_stmt");
		let start = self.input_idx;
		let stmt = if self.eat_if_kw(kw::Let) {
			self.make_assignment()?
		} else if self.eat_if_kw(kw::If) {
			self.make_if_stmt()?
		} else if self.eat_if_kw(kw::While) {
			self.make_while_stmt()?
		} else if self.eat_if_kw(kw::Return) {
			self.make_return_stmt()?
		} else if self.eat_if_kw(kw::Match) {
			self.in_match_stmt = true;
			let stmt = self.make_match_stmt()?;
			self.in_match_stmt = false;
			stmt
		} else if self.eat_if_kw(kw::Asm) {
			self.make_asm_stmt()?
		} else if self.eat_if_kw(kw::Exit) {
			self.eat_whitespace();
			ast::Stmt::Exit
		} else if self.eat_if(&TokenMatch::At) {
			match self.input_curr() {
				"bottom" => {
					self.eat_if(&TokenMatch::Ident);
					ast::Stmt::Builtin(ast::Builtin::Bottom)
				}
				"size_of" => {
					self.eat_if(&TokenMatch::Ident);
					let mut ty = self.make_type_args()?;
					if ty.len() != 1 {
						return Err(ParseError::Error(
							"@size_of takes one type argument",
							self.curr_span(),
						));
					}
					ast::Stmt::Builtin(ast::Builtin::SizeOf(crate::rawptr!(ty.remove(0))))
				}
				_ => {
					return Err(ParseError::Error("builtin", self.curr_span()));
				}
			}
		} else {
			self.make_expr_stmt()?
		};

		self.eat_whitespace();
		self.eat_if(&TokenMatch::Semi);
		let span = ast::to_rng(start..self.input_idx, self.file_id);
		Ok(stmt.into_spanned(span))
	}

	fn make_assignment(&mut self) -> ParseResult<ast::Stmt> {
		self.push_call_stack("make_assignment");
		self.eat_whitespace();

		let lval = self.make_lh_expr()?;
		self.eat_whitespace();

		let ty = if self.eat_if(&TokenMatch::Colon) {
			self.eat_whitespace();
			let ty = Some(self.make_ty()?);
			self.eat_whitespace();
			ty
		} else {
			None
		};

		self.eat_if(&TokenMatch::Eq);
		self.eat_whitespace();

		let rval = self.make_expr()?;

		Ok(ast::Stmt::Assign { lval, rval, ty, is_let: true })
	}

	fn make_if_stmt(&mut self) -> ParseResult<ast::Stmt> {
		self.push_call_stack("make_if_stmt");
		self.eat_whitespace();

		let cond = self.make_expr()?;
		self.eat_whitespace();

		let blk = self.make_block()?;
		self.eat_whitespace();

		let mut els = vec![];
		while self.eat_if_kw(kw::Else) {
			self.eat_whitespace();

			if self.eat_if_kw(kw::If) {
				self.eat_whitespace();
				let cond = Some(self.make_expr()?);
				self.eat_whitespace();

				let block = self.make_block()?;
				self.eat_whitespace();

				els.push(Else { cond, block });
			} else {
				els.push(Else { cond: None, block: self.make_block()? });
				break;
			}
		}

		Ok(ast::Stmt::If { cond, blk, els })
	}

	fn make_while_stmt(&mut self) -> ParseResult<ast::Stmt> {
		self.push_call_stack("make_while_stmt");
		self.eat_whitespace();

		let cond = self.make_expr()?;
		self.eat_whitespace();

		let stmts = self.make_block()?;
		self.eat_whitespace();

		Ok(ast::Stmt::While { cond, blk: stmts })
	}

	fn make_match_stmt(&mut self) -> ParseResult<ast::Stmt> {
		self.push_call_stack("make_match_stmt");
		self.eat_whitespace();

		let expr = self.make_expr()?;
		self.eat_whitespace();

		self.eat_if(&TokenMatch::OpenBrace);
		let arms = self.make_arms()?;

		self.eat_whitespace();
		self.eat_if(&TokenMatch::CloseBrace);
		// self.eat_whitespace();

		Ok(ast::Stmt::Match { expr, arms })
	}

	fn make_return_stmt(&mut self) -> ParseResult<ast::Stmt> {
		self.push_call_stack("make_return_stmt");
		self.eat_whitespace();

		if self.curr.kind == TokenMatch::Semi {
			return Ok(ast::Stmt::Exit);
		}
		let expr = self.make_expr()?;

		Ok(ast::Stmt::Ret(expr))
	}

	fn make_asm_stmt(&mut self) -> ParseResult<ast::Stmt> {
		self.push_call_stack("make_asm_stmt");
		self.eat_whitespace();

		let start = self.input_idx;
		let assembly = if self.eat_if(&TokenMatch::OpenBrace) {
			self.eat_whitespace();

			let mut assembly = vec![];
			while self.curr.kind != TokenMatch::CloseBrace {
				assembly.push(self.make_assembly_inst()?);
				self.eat_whitespace();
			}

			self.eat_if(&TokenMatch::CloseBrace);
			self.eat_whitespace();

			self.eat_if(&TokenMatch::Semi);

			assembly
		} else {
			return Err(ParseError::Error(
				"assembly must be in a block",
				ast::to_rng(start..self.input_idx, self.file_id),
			));
		};

		Ok(ast::Stmt::InlineAsm(AsmBlock {
			assembly,
			span: ast::to_rng(start..self.input_idx, self.file_id),
		}))
	}

	fn make_assembly_inst(&mut self) -> ParseResult<Instruction> {
		let mut inst = if self.curr.kind == TokenMatch::Ident {
			Instruction { inst: self.make_ident()?, src: None, dst: None }
		} else {
			return Err(ParseError::Error("invalid assembly instruction", self.curr_span()));
		};

		if !self.eat_if(&TokenMatch::Semi) {
			inst.src = self.make_location()?;
			self.eat_whitespace();
			if self.eat_if(&TokenMatch::Comma) {
				self.eat_whitespace();
				inst.dst = self.make_location()?;
			}

			if !self.eat_if(&TokenMatch::Semi) {
				return Err(ParseError::Error("invalid number of operands", self.curr_span()));
			}
		}

		Ok(inst)
	}

	fn make_location(&mut self) -> ParseResult<Option<Location>> {
		Ok(if matches!(self.curr.kind, TokenKind::OpenParen) {
			self.eat_if(&TokenMatch::OpenParen);

			let loc = Location::InlineVar(self.make_ident()?);

			self.eat_if(&TokenMatch::CloseParen);

			Some(loc)
		} else if matches!(self.curr.kind, TokenKind::Minus | TokenKind::Literal { .. }) {
			Some(Location::Const(self.make_literal()?.val))
		} else if matches!(self.curr.kind, TokenKind::Percent) {
			self.eat_if(&TokenMatch::Percent);

			let reg = match self.input_curr() {
				"rax" => Location::Register(Register::RAX),
				"rcx" => Location::Register(Register::RCX),
				"rdx" => Location::Register(Register::RDX),
				"rbx" => Location::Register(Register::RBX),
				"rsp" => Location::Register(Register::RSP),
				"rbp" => Location::Register(Register::RBP),
				"rsi" => Location::Register(Register::RSI),
				"rdi" => Location::Register(Register::RDI),
				"r8" => Location::Register(Register::R8),
				"r9" => Location::Register(Register::R9),
				"r10" => Location::Register(Register::R10),
				"r11" => Location::Register(Register::R11),
				"r12" => Location::Register(Register::R12),
				"r13" => Location::Register(Register::R13),
				"r14" => Location::Register(Register::R14),
				"r15" => Location::Register(Register::R15),
				"xmm0" => Location::FloatReg(FloatRegister::XMM0),
				"xmm1" => Location::FloatReg(FloatRegister::XMM1),
				"xmm2" => Location::FloatReg(FloatRegister::XMM2),
				"xmm3" => Location::FloatReg(FloatRegister::XMM3),
				"xmm4" => Location::FloatReg(FloatRegister::XMM4),
				"xmm5" => Location::FloatReg(FloatRegister::XMM5),
				"xmm6" => Location::FloatReg(FloatRegister::XMM6),
				"xmm7" => Location::FloatReg(FloatRegister::XMM7),
				_ => {
					return Err(ParseError::Error("no register by that name", self.curr_span()));
				}
			};
			self.eat_if(&TokenMatch::Ident);
			Some(reg)
		} else {
			panic!("{:?}", self.curr);
			return Err(ParseError::Error("invalid operand", self.curr_span()));
		})
	}

	fn make_expr_stmt(&mut self) -> ParseResult<ast::Stmt> {
		self.push_call_stack("make_expr_stmt");
		// @copypaste We are sort of taking this from `advance_to_op` but limiting the choices to
		// just calls, trait method calls and blocks
		Ok(if matches!(self.curr.kind, TokenKind::Ident | TokenKind::Star) {
			let expr = self.make_lh_expr()?;
			self.eat_whitespace();

			match expr.val {
				// TODO: assignment is also valid here
				//
				// `x[0] = 6; x = call; v.v.f = yo;
				ast::Expr::Ident(_) => self.make_assign_stmt_expr(expr)?,
				ast::Expr::Call { .. } => ast::Stmt::Call(expr),
				ast::Expr::Deref { .. } => self.make_assign_stmt_expr(expr)?,
				ast::Expr::Array { .. } => self.make_assign_stmt_expr(expr)?,
				ast::Expr::FieldAccess { .. } => self.make_assign_stmt_expr(expr)?,
				// Maybe this is ok??
				ast::Expr::Parens(_) => todo!(),

				ast::Expr::AddrOf(_)
				| ast::Expr::Urnary { .. }
				| ast::Expr::Binary { .. }
				| ast::Expr::TraitMeth { .. }
				| ast::Expr::StructInit { .. }
				| ast::Expr::EnumInit { .. }
				| ast::Expr::ArrayInit { .. }
				| ast::Expr::Builtin(..)
				| ast::Expr::Value(_) => {
					return Err(ParseError::Error(
						"invalid left hand side of statement",
						self.curr_span(),
					));
				}
			}
		} else if self.curr.kind == TokenMatch::Lt {
			todo!("Trait method calls {}", self.call_stack.join("\n"))
		} else if self.curr.kind == TokenMatch::OpenBrace {
			let blk = self.make_block()?;
			ast::Stmt::Block(blk)
		} else {
			return Err(ParseError::Error("make statement bottom out", self.curr_span()));
		})
	}

	fn make_assign_stmt_expr(&mut self, expr: ast::Expression) -> ParseResult<ast::Stmt> {
		// +=
		Ok(if self.cmp_seq(&[TokenMatch::Plus, TokenMatch::Eq]) {
			self.eat_seq(&[TokenMatch::Plus, TokenMatch::Eq]);
			self.eat_whitespace();
			let rval = self.make_expr()?;
			ast::Stmt::AssignOp { lval: expr, rval, op: ast::BinOp::Add }
		// -=
		} else if self.cmp_seq(&[TokenMatch::Minus, TokenMatch::Eq]) {
			self.eat_seq(&[TokenMatch::Minus, TokenMatch::Eq]);
			self.eat_whitespace();
			let rval = self.make_expr()?;
			ast::Stmt::AssignOp { lval: expr, rval, op: ast::BinOp::Sub }
		// *=
		} else if self.cmp_seq(&[TokenMatch::Star, TokenMatch::Eq]) {
			self.eat_seq(&[TokenMatch::Star, TokenMatch::Eq]);
			self.eat_whitespace();
			let rval = self.make_expr()?;
			ast::Stmt::AssignOp { lval: expr, rval, op: ast::BinOp::Mul }
		// /=
		} else if self.cmp_seq(&[TokenMatch::Slash, TokenMatch::Eq]) {
			self.eat_seq(&[TokenMatch::Slash, TokenMatch::Eq]);
			self.eat_whitespace();
			let rval = self.make_expr()?;
			ast::Stmt::AssignOp { lval: expr, rval, op: ast::BinOp::Div }
		// |=
		} else if self.cmp_seq(&[TokenMatch::Or, TokenMatch::Eq]) {
			self.eat_seq(&[TokenMatch::Or, TokenMatch::Eq]);
			self.eat_whitespace();
			let rval = self.make_expr()?;
			ast::Stmt::AssignOp { lval: expr, rval, op: ast::BinOp::BitOr }
		// &=
		} else if self.cmp_seq(&[TokenMatch::And, TokenMatch::Eq]) {
			self.eat_seq(&[TokenMatch::And, TokenMatch::Eq]);
			self.eat_whitespace();
			let rval = self.make_expr()?;
			ast::Stmt::AssignOp { lval: expr, rval, op: ast::BinOp::BitAnd }
		} else if self.curr.kind == TokenMatch::Eq {
			self.eat_if(&TokenMatch::Eq);
			self.eat_whitespace();
			let rval = self.make_expr()?;
			ast::Stmt::Assign { lval: expr, rval, ty: None, is_let: false }
		} else {
			todo!("{}", &self.input[self.input_idx..])
		})
	}

	fn make_arms(&mut self) -> ParseResult<Vec<ast::MatchArm>> {
		self.push_call_stack("make_arms");
		self.eat_whitespace();
		let mut arms = vec![];
		loop {
			let start = self.input_idx;

			self.eat_whitespace();
			if self.curr.kind == TokenMatch::CloseBrace {
				// The calling method cleans up the close brace of the `match {}` <--
				break;
			}
			let pat = self.make_pat()?;

			self.eat_whitespace();
			self.eat_if(&TokenMatch::Minus);
			self.eat_if(&TokenMatch::Gt);
			self.eat_whitespace();

			let blk = self.make_block()?;

			let span = ast::to_rng(start..self.input_idx(), self.file_id);
			self.eat_if(&TokenMatch::Comma);
			arms.push(ast::MatchArm { pat, blk, span })
		}
		Ok(arms)
	}

	fn make_pat(&mut self) -> ParseResult<ast::Pattern> {
		self.push_call_stack("make_pat");
		let start = self.input_idx;

		// TODO: make this more robust
		// could be `::mod::Name::Variant`
		Ok(if self.curr.kind == TokenKind::Ident {
			let mut path = self.make_path()?;
			// TODO: make this more robust
			// eventually calling an enum by variant needs to work which is the same as an ident
			if path.segs.len() > 1 {
				let variant = path.segs.pop().ok_or_else(|| {
					ParseError::Expected("pattern", "nothing".to_string(), self.curr_span())
				})?;

				// @PARSE_ENUMS
				let items = if self.eat_if(&TokenMatch::OpenParen) {
					self.eat_whitespace();
					let mut pats = self.make_pat_list()?;

					self.eat_whitespace();
					self.eat_if(&TokenMatch::CloseParen);

					pats
				} else {
					vec![]
				};

				let span = ast::to_rng(start..self.input_idx(), self.file_id);
				ast::Pat::Enum { path, variant, items }.into_spanned(span)
			} else {
				// TODO: binding needs span
				let span = ast::to_rng(start..self.input_idx(), self.file_id);
				ast::Pat::Bind(ast::Binding::Wild(path.segs.remove(0))).into_spanned(span)
			}
		} else if self.eat_if(&TokenMatch::OpenBracket) {
			self.eat_whitespace();
			let mut pats = self.make_pat_list()?;
			self.eat_whitespace();
			self.eat_if(&TokenMatch::CloseBracket);

			let span = ast::to_rng(start..self.input_idx(), self.file_id);
			ast::Pat::Array { size: pats.len(), items: pats }.into_spanned(span)
		} else if matches!(self.curr.kind, TokenKind::Minus | TokenKind::Literal { .. }) {
			// Literal
			let span = ast::to_rng(start..self.input_idx(), self.file_id);
			ast::Pat::Bind(ast::Binding::Value(self.make_literal()?)).into_spanned(span)
		} else {
			todo!("{:?}", self.curr)
		})
	}

	fn make_pat_list(&mut self) -> ParseResult<Vec<ast::Pattern>> {
		let mut pats = vec![];
		loop {
			// We have reached the end of the patterns (this allows trailing commas)
			if self.curr.kind == TokenMatch::CloseBracket {
				break;
			}

			pats.push(self.make_pat()?);
			if self.eat_if(&TokenMatch::Comma) {
				self.eat_whitespace();
				continue;
			} else {
				break;
			}
		}
		Ok(pats)
	}

	/// Parse `ident[ws]:[ws]type[ws],[ws]ident: type[ws]` everything inside the parens is optional.
	fn make_params(&mut self) -> ParseResult<Vec<ast::Param>> {
		self.push_call_stack("make_params");
		let mut params = vec![];
		loop {
			let start = self.input_idx;

			// TODO: if tuples or fn pointer are types this will break
			// If we have a trailing comma this will prevent us from looping and extra time
			if self.eat_if(&TokenMatch::CloseParen) {
				break;
			}

			let ident = self.make_ident()?;

			self.eat_if(&TokenMatch::Colon);
			self.eat_whitespace();

			let mut ty = self.make_ty()?;

			let span = ast::to_rng(start..self.input_idx(), self.file_id);

			params.push(ast::Param { ident, ty: crate::rawptr!(ty), span });

			self.eat_whitespace();
			if self.eat_if(&TokenMatch::Comma) {
				self.eat_whitespace();
				continue;
			} else {
				break;
			}
		}

		self.eat_whitespace();
		Ok(params)
	}

	/// Parse `<ident: ident, ident: ident>[ws]` all optional.
	fn make_generics(&mut self) -> ParseResult<Vec<ast::Generic>> {
		self.push_call_stack("make_generics");
		let mut gens = vec![];
		if self.eat_if(&TokenMatch::Lt) {
			loop {
				let start = self.input_idx;
				let ident = self.make_ident()?;
				let bound = if self.eat_if(&TokenMatch::Colon) {
					self.eat_whitespace();
					Some(self.make_path()?)
				} else {
					None
				};
				let span = ast::to_rng(start..self.input_idx(), self.file_id);
				gens.push(ast::Generic { ident, bound, span });

				self.eat_whitespace();
				if self.eat_if(&TokenMatch::Comma) {
					self.eat_whitespace();

					continue;
				} else {
					break;
				}
			}
			self.eat_if(&TokenMatch::Gt);
			self.eat_whitespace();
		}
		self.eat_whitespace();
		Ok(gens)
	}

	/// Parse a literal.
	fn make_literal(&mut self) -> ParseResult<ast::Value> {
		self.push_call_stack("make_literal");

		let neg_ident = if self.eat_if(&TokenMatch::Minus) { -1 } else { 1 };

		// @copypaste
		#[allow(clippy::wildcard_in_or_patterns)]
		Ok(match self.curr.kind {
			TokenKind::Ident => {
				let keyword: kw::Keywords = self.input_curr().try_into()?;
				match keyword {
					kw::True => {
						let expr = Val::Bool(true).into_spanned(self.curr_span());
						self.eat_if_kw(kw::True);
						expr
					}
					kw::False => {
						let expr = Val::Bool(false).into_spanned(self.curr_span());
						self.eat_if_kw(kw::False);
						expr
					}
					_ => todo!(),
				}
			}
			TokenKind::Literal { kind, suffix_start } => {
				let span = self.curr_span();
				let text = self.input_curr();

				let val = match kind {
					LiteralKind::Int { base, empty_int } => Val::Int(match base {
						Base::Binary => isize::from_str_radix(
							text.strip_prefix("0b")
								.ok_or_else(|| ParseError::InvalidIntLiteral(self.curr_span()))?,
							2,
						)
						.map_err(|_| ParseError::InvalidIntLiteral(self.curr_span()))?,
						Base::Hexadecimal => isize::from_str_radix(
							text.strip_prefix("0x")
								.ok_or_else(|| ParseError::InvalidIntLiteral(self.curr_span()))?,
							16,
						)
						.map_err(|_| ParseError::InvalidIntLiteral(self.curr_span()))?,
						Base::Octal => isize::from_str_radix(
							text.strip_prefix("0o")
								.ok_or_else(|| ParseError::InvalidIntLiteral(self.curr_span()))?,
							8,
						)
						.map_err(|_| ParseError::InvalidIntLiteral(self.curr_span()))?,
						Base::Decimal => {
							neg_ident
								* text.parse::<isize>()
									.map_err(|_| ParseError::InvalidIntLiteral(self.curr_span()))?
						}
					}),
					LiteralKind::Float { base, empty_exponent } => Val::Float(
						(neg_ident as f64)
							* text
								.parse::<f64>()
								.map_err(|_| ParseError::InvalidFloatLiteral(self.curr_span()))?,
					),
					LiteralKind::Char { terminated } if neg_ident > 0 => {
						let end = text.len() - 1;
						let text = &text[1..end];
						if text.replace('\\', "").len() == 1 {
							Val::Char(StripEscape::new(text).next().unwrap())
						} else {
							return Err(ParseError::Error(
								"multi character `char`",
								self.curr_span(),
							));
						}
					}
					LiteralKind::Str { terminated } if neg_ident > 0 => {
						let mut text = text.to_string();
						// Remove the first " and pop the last "
						text.remove(0);
						text.pop();

						Val::Str(text.len(), Ident::new(self.curr_span(), &text))
					}
					LiteralKind::Byte { terminated } => {
						let end = text.len() - 1;
						// We skip the `b'` (so the b and the "'"")
						let text = &text[2..end];
						if text.replace('\\', "").len() == 1 {
							Val::Int(text.chars().next().unwrap() as isize)
						} else {
							return Err(ParseError::Error(
								"multi character `char`",
								self.curr_span(),
							));
						}
					},
					LiteralKind::ByteStr { .. }
					| LiteralKind::RawStr { .. }
					| LiteralKind::RawByteStr { .. }
					| _ => todo!(),
				};
				self.eat_if(&TokenMatch::Literal);
				val.into_spanned(span)
			}
			tkn => {
				// todo!("{:?}", tkn)
				return Err(ParseError::Error("literal", self.curr_span()));
			}
		})
	}

	fn make_arg_list(&mut self) -> ParseResult<Vec<ast::Expression>> {
		let mut args = vec![];
		loop {
			self.eat_whitespace();
			// A no argument function call or trailing comma
			if self.eat_if(&TokenMatch::CloseParen) {
				break;
			}

			args.push(self.make_expr()?);
			self.eat_whitespace();
			if self.eat_if(&TokenMatch::Comma) {
				self.eat_whitespace();
				continue;
			} else {
				break;
			}
		}
		Ok(args)
	}

	/// Parses `::<type, type...>` where the `type` is concrete or generic.
	fn make_type_args(&mut self) -> ParseResult<Vec<Type>> {
		self.eat_seq_ignore_ws(&[TokenMatch::Colon, TokenMatch::Colon]);
		Ok(if self.curr.kind == TokenMatch::Lt {
			self.eat_if(&TokenMatch::Lt);
			let mut gen_args = vec![];
			loop {
				self.eat_whitespace();
				gen_args.push(self.make_ty()?);
				self.eat_whitespace();
				if self.eat_if(&TokenMatch::Comma) {
					self.eat_whitespace();
					continue;
				} else {
					break;
				}
			}
			self.eat_if(&TokenMatch::Gt);

			gen_args
		} else {
			vec![]
		})
	}

	/// parse a list of types .
	fn make_types(&mut self, open: &TokenMatch, close: &TokenMatch) -> ParseResult<Vec<ast::Type>> {
		self.push_call_stack("make_types");
		Ok(if self.eat_if(open) {
			let mut tys = vec![];
			loop {
				self.eat_whitespace();
				// We have reached the end of the list (trailing comma or empty)
				if self.eat_if(close) {
					break;
				}
				tys.push(self.make_ty()?);
				if self.eat_if(&TokenMatch::Comma) {
					self.eat_whitespace();
					continue;
				} else {
					break;
				}
			}
			self.eat_if(close);
			self.eat_whitespace();
			tys
		} else {
			vec![]
		})
	}

	/// Parse `type[ws]`.
	///
	/// This also handles `*type, [lit_int, type], foo<T>` and user defined items.
	fn make_ty(&mut self) -> ParseResult<ast::Type> {
		self.push_call_stack("make_ty");

		let start = self.input_idx;
		Ok(match self.curr.kind {
			TokenKind::Ident => {
				let text = self.input_curr();

				let key: Result<kw::Keywords, _> = text.try_into();
				if let Ok(key) = key {
					if let kw::Fn = key {
						let start = self.input_idx;
						self.eat_keyword(kw::Fn);
						self.eat_if(&TokenMatch::OpenParen);

						let mut params = vec![];
						loop {
							self.eat_whitespace();
							// A no argument function call or trailing comma
							if self.eat_if(&TokenMatch::CloseParen) {
								break;
							}

							params.push(self.make_ty()?.val);
							self.eat_whitespace();
							if self.eat_if(&TokenMatch::Comma) {
								self.eat_whitespace();
								continue;
							} else {
								self.eat_if(&TokenMatch::CloseParen);
								break;
							}
						}
						let ret = if self.eat_if(&TokenMatch::Colon) {
							self.eat_whitespace();
							self.make_ty()?.val
						} else {
							self.eat_whitespace();
							ast::Ty::Void
						};
						ast::Ty::Func { ident: Ident::dummy(), params, ret: box ret }
							.into_spanned(ast::to_rng(start..self.input_idx, self.file_id))
					} else {
						return Err(ParseError::Expected(
							"invalid keyword in type",
							self.input_curr().to_string(),
							self.curr_span(),
						));
					}
				} else {
					let span = self.curr_span();
					let ty = match text {
						"void" => ast::Ty::Void.into_spanned(span),
						"bool" => ast::Ty::Bool.into_spanned(span),
						"char" => ast::Ty::Char.into_spanned(span),
						"int" => ast::Ty::Int.into_spanned(span),
						"uint" => ast::Ty::UInt.into_spanned(span),
						"float" => ast::Ty::Float.into_spanned(span),
						"cstr" => ast::Ty::ConstStr(0).into_spanned(span),
						_ => {
							let start = self.input_idx;
							let path = self.make_path()?;

							if self.curr.kind == TokenMatch::Lt {
								let tys = self.make_types(&TokenMatch::Lt, &TokenMatch::Gt)?;
								ast::Ty::Path(path).into_spanned(span)
							} else {
								let span = ast::to_rng(start..self.input_idx(), self.file_id);
								ast::Ty::Path(path).into_spanned(span)
							}
						}
					};
					self.eat_if(&TokenMatch::Ident);
					ty
				}
			}
			TokenKind::OpenParen => {
				panic!("{}", self.call_stack.join("\n"))
			}
			TokenKind::OpenBracket => {
				let start = self.input_idx;
				self.eat_if(&TokenMatch::OpenBracket);
				self.eat_whitespace();
				self.make_array_type(start)?
			}
			TokenKind::Star => {
				// Eat `*`
				self.eat_tkn();
				ast::Ty::Ptr(box self.make_ty()?).into_spanned(self.curr_span())
			}
			// TokenKind::Lt => {}
			// TokenKind::Gt => {}
			tkn => todo!("Unknown token {:?} {}", tkn, &self.call_stack.join("\n")),
		})
	}

	/// Any type that follows `lit_int; type][ws]`.
	fn make_array_type(&mut self, start: usize) -> ParseResult<ast::Type> {
		let size = if let TokenKind::Literal {
			kind: LiteralKind::Int { base: Base::Decimal, .. },
			..
		} = self.curr.kind
		{
			self.input_curr()
				.parse()
				.map_err(|_| ParseError::InvalidIntLiteral(self.curr_span()))?
		} else {
			return Err(ParseError::Expected(
				"lit",
				self.input_curr().to_string(),
				self.curr_span(),
			));
		};
		// [ -->lit; -->type]
		self.eat_if(&TokenMatch::Literal);
		self.eat_whitespace();
		self.eat_if(&TokenMatch::Semi);
		self.eat_whitespace();

		let ty = self.make_ty()?;
		let x = Ok(ast::Ty::Array { size, ty: box ty }
			.into_spanned(ast::to_rng(start..self.input_idx(), self.file_id)));
		self.eat_if(&TokenMatch::CloseBracket);
		x
	}

	/// Parse `::ident[w]::ident[ws]...`.
	fn make_path(&mut self) -> ParseResult<Path> {
		let start = self.input_idx;
		let segs = self.make_seg()?;
		Ok(Path { segs, span: ast::to_rng(start..self.input_idx(), self.file_id) })
	}

	/// Parse `ident[ws]::ident[ws]...`.
	fn make_seg(&mut self) -> ParseResult<Vec<Ident>> {
		let mut ids = vec![];
		loop {
			let full_type_args = self.cmp_seq_ignore_ws(&[
				TokenMatch::Ident,
				TokenMatch::Colon,
				TokenMatch::Colon,
				TokenMatch::Lt,
			]);
			let just_type_args =
				self.cmp_seq_ignore_ws(&[TokenMatch::Colon, TokenMatch::Colon, TokenMatch::Lt]);

			let is_explicit_type =
				self.cmp_seq_ignore_ws(&[TokenMatch::Ident, TokenMatch::Colon, TokenMatch::Ident]);

			// Stops eating `call::<T, U>();` colons and stops eating int explicit types
			// `let x: type = ...`
			if !(full_type_args || just_type_args || is_explicit_type) {
				self.eat_seq(&[TokenMatch::Colon, TokenMatch::Colon]);
			} else if just_type_args {
				break;
			}

			ids.push(self.make_ident()?);
			if self.cmp_seq(&[TokenMatch::Colon, TokenMatch::Colon])
				|| self.cmp_seq(&[TokenMatch::Ident])
			{
				self.eat_whitespace();
				continue;
			} else {
				break;
			}
		}
		self.eat_whitespace();
		ids.retain(|id| {
			let text = id.name();
			!(text.is_empty() || text.chars().all(lex::is_whitespace))
		});

		Ok(ids)
	}

	/// Parse `ident[ws]`
	fn make_ident(&mut self) -> ParseResult<Ident> {
		let span = self.curr_span();
		let id = Ident::new(span, &self.input[span.start..span.end]);
		self.eat_if(&TokenMatch::Ident);
		self.eat_whitespace();
		Ok(id)
	}

	fn eat_whitespace(&mut self) {
		while self.eat_if(&TokenMatch::Whitespace)
			|| self.eat_if(&TokenMatch::LineComment)
			|| self.eat_if(&TokenMatch::BlockComment)
		{}
	}

	/// Checks the next token in the `tokens` vector.
	fn check_next(&self, tkn: &TokenMatch) -> bool {
		self.tokens.first().map_or(false, |t| t.kind == *tkn)
	}

	/// FIXME: for now we ignore attributes.
	fn eat_attr(&mut self) {
		if matches!(self.peek().unwrap_or(&TokenKind::Unknown), TokenKind::OpenBracket) {
			self.eat_until(&TokenMatch::CloseBracket);
			// eat the `]`
			self.eat_tkn();
		}
	}

	/// Eat the key word iff it matches `kw`.
	fn eat_keyword(&mut self, kw: kw::Keywords) {
		if self.input_curr() == kw.text() {
			self.eat_tkn();
		}
	}

	/// Eat the key word iff it matches `kw` and return true if eaten.
	fn eat_if_kw(&mut self, kw: kw::Keywords) -> bool {
		if kw.text() == self.input_curr() {
			self.eat_tkn();
			return true;
		}
		false
	}

	/// Check if a sequence matches `iter`, non destructively.
	fn cmp_seq<'i>(&self, mut iter: impl IntoIterator<Item = &'i TokenMatch>) -> bool {
		let mut iter = iter.into_iter();
		let first = iter.next().unwrap_or(&TokenMatch::Unknown);
		if first != &self.curr.kind {
			return false;
		}

		let tkns = self.tokens.iter();
		tkns.zip(iter).all(|(ours, cmp)| cmp == &ours.kind)
	}

	/// Throw away a sequence of tokens.
	///
	/// Returns true if all the given tokens were matched.
	fn eat_seq<'i>(&mut self, iter: impl IntoIterator<Item = &'i TokenMatch>) -> bool {
		for kind in iter {
			if kind == &self.curr.kind {
				self.eat_tkn();
			} else {
				return false;
			}
		}
		true
	}

	/// Check if a sequence matches `iter` ignoring whitespace, non destructively.
	fn cmp_seq_ignore_ws<'i>(&self, mut iter: impl IntoIterator<Item = &'i TokenMatch>) -> bool {
		let mut iter = iter.into_iter();
		if !matches!(
			self.curr.kind,
			TokenKind::Whitespace | TokenKind::LineComment { .. } | TokenKind::BlockComment { .. }
		) {
			let first = iter.next().unwrap_or(&TokenMatch::Unknown);
			if first != &self.curr.kind {
				return false;
			}
		}

		let tkns = self.tokens.iter().filter(|t| {
			!matches!(
				t.kind,
				TokenKind::Whitespace
					| TokenKind::LineComment { .. }
					| TokenKind::BlockComment { .. }
			)
		});
		for (cmp, ours) in tkns.zip(iter) {
			if cmp.kind != *ours {
				return false;
			}
		}
		true
	}

	/// Throw away a sequence of tokens.
	///
	/// Returns true if all the given tokens were matched.
	fn eat_seq_ignore_ws<'i>(&mut self, iter: impl IntoIterator<Item = &'i TokenMatch>) -> bool {
		for kind in iter {
			if matches!(
				self.curr.kind,
				TokenKind::Whitespace
					| TokenKind::LineComment { .. }
					| TokenKind::BlockComment { .. }
			) {
				self.eat_tkn();
			}
			if kind == &self.curr.kind {
				self.eat_tkn();
			} else {
				return false;
			}
		}
		true
	}

	/// Eat tokens until `pat` matches current.
	fn eat_until(&mut self, pat: &TokenMatch) {
		while pat != &self.curr.kind {
			self.eat_tkn();
		}
	}

	/// Eat tokens until `pat` matches current.
	fn eat_while(&mut self, pat: &TokenMatch) {
		while pat == &self.curr.kind {
			self.eat_tkn();
		}
	}

	/// Count the tokens eaten that matched `pat`.
	fn count_eaten_seq(&mut self, pat: &TokenMatch) -> usize {
		let mut count = 0;
		while pat == &self.curr.kind {
			count += 1;
			self.eat_tkn();
		}
		count
	}

	/// Bump the current token if it matches `pat`.
	fn eat_if(&mut self, pat: &TokenMatch) -> bool {
		if pat == &self.curr.kind {
			self.eat_tkn();
			return true;
		}
		false
	}

	/// Bump the next token into the current spot.
	fn eat_tkn(&mut self) {
		self.input_idx += self.curr.len;
		self.curr = self.tokens.remove(0);
	}

	/// Peek the next token.
	fn peek(&self) -> Option<&TokenKind> {
		self.tokens.first().map(|t| &t.kind)
	}

	/// Peek the next `n` tokens.
	fn peek_n(&self, n: usize) -> impl Iterator<Item = &TokenKind> {
		self.tokens.iter().take(n).map(|t| &t.kind)
	}

	/// Peek until the closure returns `false`.
	fn peek_until<P: FnMut(&&lex::Token) -> bool>(
		&self,
		p: P,
	) -> impl Iterator<Item = &lex::Token> {
		self.tokens.iter().take_while(p)
	}

	/// The input `str` from current index to `stop`.
	fn input_to(&self, stop: usize) -> &str {
		&self.input[self.input_idx..stop]
	}

	/// The input `str` from current index to `Token` length.
	fn input_curr(&self) -> &str {
		let stop = self.input_idx + self.curr.len;
		&self.input[self.input_idx..stop]
	}

	/// The input `str` from current index to `stop`.
	fn curr_span(&self) -> ast::Range {
		let current_len = if matches!(
			self.curr.kind,
			TokenKind::Whitespace | TokenKind::BlockComment { .. } | TokenKind::LineComment { .. }
		) {
			0
		} else {
			self.curr.len
		};
		let stop = self.input_idx + current_len;
		ast::to_rng(self.input_idx..stop, self.file_id)
	}

	/// The end count of the current cursor index.
	fn input_idx(&self) -> usize {
		self.input_idx + self.curr.len
	}
}

#[test]
fn parse_char_lit() {
	let input = r#"
const foo: char = '\n';
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();

	if let Decl::Const(k) = &parser.items()[0].val {
		if let Expr::Value(Spanned { val: Val::Char(ch), .. }) = k.init.val {
			assert_eq!(ch, '\n');
		}
	}
}

#[test]
fn parse_lit_const() {
	let input = r#"
const foo: [3; int] = 1;
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_func_header() {
	let input = r#"
fn add(x: int, y: int): int {  }
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn struct_decl() {
	let input = r#"
struct foo {
	x: int,
	y: string,
	z: [3; char],
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn struct_decl_generics() {
	let input = r#"
struct foo<T, U> {
	x: T,
	y: U,
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn enum_decl() {
	let input = r#"
enum foo {
	a, b, c
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn enum_decl_generics() {
	let input = r#"
enum foo<T, X, U> {
	a(X, string), b, c(T, U), d([2; int])
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn impl_decl() {
	let input = r#"
impl add<string> {
	fn add(a: string, b: string): string {
		return concat(a, b);
	}
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn import_decl() {
	let input = r#"
import foo::bar::baz;
import ::bar::baz;
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 2);
}

#[test]
fn parse_multi_binop_ident() {
	let input = r#"
fn add(x: int, y: int): int {
	let z = a + b * (c + d);
	return z;
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_multi_binop_lit() {
	let input = r#"
fn add(x: int, y: int): int {
	let z = 1 + 2 * 5 + 6;
	return z;
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_func_call_no_args() {
	let input = r#"
fn add() {
	let x = foo();
	foo();
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_func_call_args_generics() {
	let input = r#"
fn add() {
	let x = foo::<T, U>(a, 1 + 2);
	foo::<T>(a);
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_assign_op() {
	let input = r#"
fn add() {
	let x = 10;
	x += 3+5;
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_negative_lit() {
	let input = r#"
fn add() {
	let z = -1;
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_assign_array() {
	let input = r#"
fn add() {
	let x = [10, call(), 1+1+2];
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_assign_struct() {
	let input = r#"
fn add() {
	let x = foo { x: 1, y: "string" };
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_assign_enum() {
	let input = r#"
fn add() {
	let x = foo::bar(a, 1+0, 1.6);
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_if_stmt() {
	let input = r#"
fn add() {
	if (x > 10) {
		call(1, 2, 3);
	} else {
		exit;
	}
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_match_stmt() {
	let input = r#"
fn add() {
	match x {
		bar::foo(a, b) -> {},
		bar::foo(a, b) -> {},
		bar::bar -> {
			call();
			let x = y + 5 * z;
			if (x > y) {
				return 1;
			}
		}
	}
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_while_stmt() {
	let input = r#"
fn add() {
	while (x > 10) {
		call(1, 2, 3);
		x += 1;
	}
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_trait_method() {
	let input = r#"
fn add() {
	let x = <<T>::add>(1, 2);
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_exprs() {
	let input = r#"
fn add() {
	let x = **x;
	let y = &call();
	let z = !x && false || !y;
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_exprs_fail() {
	let input = r#"
fn add() {
	let z = 4 >> 2 & 0 << 3;
	let x = 4 + 2 & 0 + 3;
	z += 1;
	let y = z | (1 & ~5);
	z = y + add(1, 1);
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_exprs_fail2() {
	let input = r#"
fn add() {
	let y = z | (1 & ~5);
	z += 1;
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_ambig_1() {
	let input = r#"
fn add() {
	let x = y < x;
	let y = call::<T>();
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser
		.parse()
		.map_err(|e| crate::ast::parse::error::PrettyError::from_parse("test", input, e))
		.unwrap_or_else(|e| panic!("{}", e));
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_import_ambig_func() {
	let input = r#"
fn add() {
	let y = foo::bar::call::<T>();
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser
		.parse()
		.map_err(|e| crate::ast::parse::error::PrettyError::from_parse("test", input, e))
		.unwrap_or_else(|e| panic!("{}", e));
	if let Decl::Func(func) = &parser.items()[0].val {
		if let Stmt::Assign { rval, .. } = &func.stmts.stmts[0].val {
			if let Expr::Call { path, .. } = &rval.val {
				assert_eq!(path.segs[0], "foo");
				assert_eq!(path.segs[1], "bar");
				assert_eq!(path.segs[2], "call");
			} else {
				panic!("assert failed for function call with import path")
			}
		} else {
			panic!("assert failed for function call with import path")
		}
	} else {
		panic!("assert failed for function call with import path")
	}
}

#[test]
fn parse_trait_span() {
	let input = r#"
fn add() {
	let y = <<int>::add>(a, b);
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser
		.parse()
		.map_err(|e| crate::ast::parse::error::PrettyError::from_parse("test", input, e))
		.unwrap_or_else(|e| panic!("{}", e));
	if let Decl::Func(func) = &parser.items()[0].val {
		let mut x = input.split("let");
		let start = x.next().unwrap().chars().count();
		let end = x.next().unwrap().chars().count() + start;
		assert_eq!(func.stmts.stmts[0].span, ast::to_rng(start..end, hash_file("test.file")))
	} else {
		panic!("assert failed for function call with import path")
	}
}

#[test]
fn parse_type_assign() {
	let input = r#"
fn add() {
	let y: string = call();
	let z: bool = !x && false || !y;
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn func_parse() {
	let input = r#"
fn add(cb: fn(int, char, cstr): float) {
	cb();
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn inline_assembly_parse() {
	let input = r#"
fn assembly() {
	asm {
		cmov 1, %rax;
		ret;
		leave;
		pop %rax;
		pushq %xmm0;
		mov %rax, %rdi;
	}
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
#[ignore = "todo -> conditional expr with no `(cond)`"]
fn non_paren_cond_parse() {
	// TODO: make this work
	let input = r#"
fn while_loops() {
	while true {
		j -= 1;

		while b[j] > x {
			j -= 1;
		}

		i += 1;
		while b[i] < x {
			i += 1;
		}

		printf("j: %d\n", j);

		printf("i: %d\n", i);

		if i < j {
			t = b[i];
			b[i] = b[j];
			b[j] = t;
		} else {
			return j;
		}
	}
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}

#[test]
fn parse_index_field_access() {
	let input = r#"
fn add() {
	let x = a.b[0];
}
"#;
	let mut parser = AstBuilder::new(input, "test.file", std::sync::mpsc::channel().0);
	parser.parse().unwrap();
	assert_eq!(parser.items().len(), 1);
}
