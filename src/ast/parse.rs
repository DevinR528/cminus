use std::convert::TryInto;

crate use crate::ast::{
    lex::{self, TokenKind, TokenMatch},
    parse::symbol::Ident,
    types as ast,
};
use crate::ast::{
    lex::{LiteralKind, Token},
    types::{Path, Spany, Val},
};

mod error;
crate mod kw;
mod prec;
mod symbol;

use error::ParseError;
use prec::{AssocOp, Fixit};

use self::lex::Base;

pub type ParseResult<T> = Result<T, ParseError>;

// TODO: this is basically one file = one mod/crate/program unit add mod linking or
// whatever.
/// Create an AST from input `str`.
#[derive(Debug, Default)]
pub struct AstBuilder<'a> {
    tokens: Vec<lex::Token>,
    curr: lex::Token,
    input: &'a str,
    input_idx: usize,
    items: Vec<ast::Declaration>,
}

// FIXME: audit the whitespace eating, pretty sure I call it unnecessarily
impl<'a> AstBuilder<'a> {
    pub fn new(input: &'a str) -> Self {
        let mut tokens =
            lex::tokenize(input).chain(Some(Token::new(TokenKind::Eof, 0))).collect::<Vec<_>>();
        // println!("{:#?}", tokens);
        let curr = tokens.remove(0);
        Self { tokens, curr, input, ..Default::default() }
    }

    pub fn items(&self) -> &[ast::Declaration] {
        &self.items
    }

    pub fn into_items(self) -> Vec<ast::Declaration> {
        self.items
    }

    pub fn parse(&mut self) -> ParseResult<()> {
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
                    // println!("{}", self.input_curr());
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
                            let item = self.parse_import()?;
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
                TokenKind::Unknown => return Err(ParseError::Error("encountered unknown token")),
                tkn => unreachable!("Unknown token {:?}", tkn),
            }
            self.eat_whitespace();
        }
        Ok(())
    }

    // Parse `const name: type = expr;`
    fn parse_const(&mut self) -> ParseResult<ast::Declaration> {
        let start = self.input_idx;

        self.eat_if_kw(kw::Const);
        self.eat_whitespace();

        let id = self.make_ident()?;
        self.eat_whitespace();

        self.eat_if(&TokenMatch::Colon);
        self.eat_whitespace();

        let ty = self.make_ty()?;

        self.eat_whitespace();
        self.eat_if(&TokenMatch::Eq);
        self.eat_whitespace();

        let expr = self.make_expr()?;

        self.eat_whitespace();
        self.eat_if(&TokenMatch::Semi);

        let span = ast::to_rng(start..self.input_idx);
        Ok(ast::Decl::Const(ast::Const { ident: id, ty, init: expr, span }).into_spanned(span))
    }

    // Parse `fn name<T>(it: T) -> int { .. }` with or without generics.
    fn parse_fn(&mut self) -> ParseResult<ast::Declaration> {
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

        let ret = if self.eat_seq(&[TokenMatch::Minus, TokenMatch::Gt]) {
            self.eat_whitespace();
            self.make_ty()?
        } else {
            self.eat_whitespace();
            ast::Ty::Void.into_spanned(self.curr_span())
        };
        self.eat_whitespace();

        let stmts = self.make_block()?;

        let span = ast::to_rng(start..self.input_idx);

        Ok(ast::Decl::Func(ast::Func { ident, ret, generics, params, stmts, span })
            .into_spanned(span))
    }

    fn parse_impl(&mut self) -> ParseResult<ast::Declaration> {
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
        let span = ast::to_rng(start..self.input_idx);
        Ok(ast::Decl::Impl(ast::Impl { path, type_arguments, method, span }).into_spanned(span))
    }

    fn parse_struct(&mut self) -> ParseResult<ast::Declaration> {
        let start = self.input_idx;

        self.eat_if_kw(kw::Struct);
        self.eat_whitespace();

        let ident = self.make_ident()?;
        let generics = self.make_generics()?;

        self.eat_if(&TokenMatch::OpenBrace);
        self.eat_whitespace();

        let fields = self.make_fields()?;

        self.eat_if(&TokenMatch::CloseBrace);
        let span = ast::to_rng(start..self.input_idx);
        Ok(ast::Decl::Adt(ast::Adt::Struct(ast::Struct { ident, fields, generics, span }))
            .into_spanned(span))
    }

    fn parse_enum(&mut self) -> ParseResult<ast::Declaration> {
        let start = self.input_idx;

        self.eat_if_kw(kw::Enum);
        self.eat_whitespace();

        let ident = self.make_ident()?;
        let generics = self.make_generics()?;

        self.eat_if(&TokenMatch::OpenBrace);
        self.eat_whitespace();

        let variants = self.make_variants()?;

        self.eat_if(&TokenMatch::CloseBrace);
        let span = ast::to_rng(start..self.input_idx);
        Ok(ast::Decl::Adt(ast::Adt::Enum(ast::Enum { ident, variants, generics, span }))
            .into_spanned(span))
    }

    fn parse_trait(&mut self) -> ParseResult<ast::Declaration> {
        let start = self.input_idx;

        self.eat_if_kw(kw::Impl);
        self.eat_whitespace();

        let path = self.make_path()?;
        let generics = self.make_generics()?;

        println!("{:?}", self.curr);

        self.eat_if(&TokenMatch::OpenBrace);
        self.eat_whitespace();

        let method = self.make_trait_fn()?;

        self.eat_if(&TokenMatch::CloseBrace);
        let span = ast::to_rng(start..self.input_idx);
        Ok(ast::Decl::Trait(ast::Trait { path, generics, method, span }).into_spanned(span))
    }

    fn parse_import(&mut self) -> ParseResult<ast::Declaration> {
        let start = self.input_idx;

        self.eat_if_kw(kw::Import);
        self.eat_whitespace();

        let path = self.make_path()?;

        self.eat_if(&TokenMatch::Semi);

        let span = ast::to_rng(start..self.input_idx);
        Ok(ast::Decl::Import(path).into_spanned(span))
    }

    // Parse `fn name<T>(it: T) -> int;` with or without generics.
    fn make_trait_fn(&mut self) -> ParseResult<ast::TraitMethod> {
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

        let ret = if self.eat_seq(&[TokenMatch::Minus, TokenMatch::Gt]) {
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

            let span = ast::to_rng(start..self.input_idx);
            ast::TraitMethod::Default(ast::Func { ident, ret, generics, params, stmts, span })
        } else {
            let stmts = ast::Block { stmts: vec![], span: self.curr_span() };
            let span = ast::to_rng(start..self.input_idx);
            ast::TraitMethod::NoBody(ast::Func { ident, ret, generics, params, stmts, span })
        })
    }

    /// Parse `name[ws]([ws]type[ws],...)[ws], name(a, b), ..`.
    fn make_variants(&mut self) -> ParseResult<Vec<ast::Variant>> {
        let mut variants = vec![];
        loop {
            let start = self.input_idx;

            // We have reached the end of the enum def (this allows trailing commas)
            if self.curr.kind == TokenMatch::CloseBrace {
                break;
            }

            let ident = self.make_ident()?;
            let types = self.make_types(&TokenMatch::OpenParen, &TokenMatch::CloseParen)?;

            let span = ast::to_rng(start..self.curr_span().end);
            variants.push(ast::Variant { ident, types, span });

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

            let span = ast::to_rng(start..self.curr_span().end);
            params.push(ast::Field { ident, ty, span });

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
                let span = ast::to_rng(start..self.curr_span().end);
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
            vec![]
        })
    }

    /// parse a list of expressions.
    fn make_expr_list(
        &mut self,
        open: &TokenMatch,
        close: &TokenMatch,
    ) -> ParseResult<Vec<ast::Expression>> {
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
        let start = self.input_idx;
        self.eat_whitespace();

        if self.curr.kind == TokenMatch::OpenBracket {
            // array init
            let items = self.make_expr_list(&TokenMatch::OpenBracket, &TokenMatch::CloseBracket)?;

            let span = ast::to_rng(start..self.curr_span().end);
            Ok(ast::Expr::ArrayInit { items }.into_spanned(span))
        } else if self.curr.kind == TokenMatch::Lt {
            // trait method calls
            let start = self.curr_span().start;

            // Outer `<` token
            self.eat_if(&TokenMatch::Lt);
            let type_args = if self.curr.kind == TokenMatch::Lt {
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
            };

            let trait_ = self.make_path()?;

            self.eat_whitespace();
            self.eat_if(&TokenMatch::Gt);

            self.eat_if(&TokenMatch::OpenParen);
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
            // This is duplicated iff we have a no arg call
            self.eat_whitespace();
            self.eat_if(&TokenMatch::CloseParen);

            let span = ast::to_rng(start..self.curr_span().end);
            Ok(ast::Expr::TraitMeth { trait_, type_args, args }.into_spanned(span))
        } else if self.curr.kind == TokenMatch::OpenParen {
            self.eat_if(&TokenMatch::OpenParen);

            let ex = self.make_expr()?;

            self.eat_if(&TokenMatch::OpenParen);

            // tuple
            // TODO: check there are only commas maybe??
            Ok(ex)
        } else if matches!(
            self.curr.kind,
            TokenKind::Ident
                | TokenKind::Literal { .. }
                | TokenKind::Star
                | TokenKind::And
                | TokenKind::Bang
                | TokenKind::Tilde
        ) {
            // FIXME: we don't want to have to say `let x = enum foo::bar;` just `let x = foo::bar;`
            // TODO: don't parse for keywords if its a lit DUH
            let x: Result<kw::Keywords, _> = self.input_curr().try_into();
            if let Ok(key) = x {
                match key {
                    kw::Enum => self.make_enum_init(),
                    kw::Struct => self.make_struct_init(),
                    t => todo!("error {:?}", self.curr),
                }
            } else {
                // TODO: Refactor out
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

                                let span = ast::to_rng(lhs.span.start..ex.span.end);
                                let mut finish = ast::Expr::Binary {
                                    op: prev.to_ast_binop().unwrap(),
                                    lhs: box lhs,
                                    rhs: box ex,
                                }
                                .into_spanned(span);

                                while let Some(lfix_op) = opstack.pop() {
                                    if matches!(lfix_op.fixity(), Fixit::Left) {
                                        let lhs = output.pop().unwrap();
                                        let span = ast::to_rng(lhs.span.start..finish.span.end);
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
                                        let span = ast::to_rng(first.span.start..rhs.span.end);
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

                output.pop().ok_or(ParseError::Error("failed to generate expression"))
            }
        } else {
            // See the above 4 todos/fixes
            todo!("{:?}", self.curr);
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
        Ok(if self.curr.kind == TokenMatch::Ident {
            let id = self.make_lh_expr()?;
            self.eat_whitespace();

            let op = self.make_op()?;
            (id, op)
        } else if self.curr.kind == TokenMatch::Literal {
            let start = self.curr_span().start;
            let ex = ast::Expr::Value(self.make_literal()?)
                .into_spanned(ast::to_rng(start..self.curr_span().end));
            self.eat_whitespace();

            let op = self.make_op()?;
            (ex, op)
        } else if self.curr.kind == TokenMatch::Bang {
            // Not `!expr`
            let start = self.input_idx;

            self.eat_if(&TokenMatch::Bang);
            let id = self.make_lh_expr()?;
            let expr = ast::Expr::Urnary { op: ast::UnOp::Not, expr: box id }
                .into_spanned(ast::to_rng(start..self.curr_span().end));

            self.eat_whitespace();

            let op = self.make_op()?;
            (expr, op)
        } else if self.curr.kind == TokenMatch::Minus {
            todo!("negative")
        } else if self.curr.kind == TokenMatch::Tilde {
            // Negation `~expr`
            let start = self.input_idx;

            self.eat_if(&TokenMatch::Tilde);
            let id = self.make_lh_expr()?;
            let expr = ast::Expr::Urnary { op: ast::UnOp::OnesComp, expr: box id }
                .into_spanned(ast::to_rng(start..self.curr_span().end));

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
                .into_spanned(ast::to_rng(start..self.curr_span().end));

            self.eat_whitespace();

            let op = self.make_op()?;
            (expr, op)
        } else if self.curr.kind == TokenMatch::And {
            let start = self.input_idx;

            self.eat_if(&TokenMatch::And);
            let ex = self.make_lh_expr()?;
            let expr =
                ast::Expr::AddrOf(box ex).into_spanned(ast::to_rng(start..self.curr_span().end));

            self.eat_whitespace();

            let op = self.make_op()?;
            (expr, op)
        } else if self.curr.kind == TokenMatch::OpenParen {
            // N.B.
            // We know we are in the middle of some kind of binop
            self.eat_if(&TokenMatch::OpenParen);

            let id = self.make_expr()?;
            self.eat_whitespace();

            let op = self.make_op()?;
            (id, op)
        } else {
            todo!("{:?}", self.curr)
        })
    }

    /// Builds left hand expressions.
    ///
    /// - idents
    /// - field access
    /// - array index
    /// - fn call
    fn make_lh_expr(&mut self) -> ParseResult<ast::Expression> {
        Ok(if self.curr.kind == TokenMatch::Ident {
            let start = self.curr_span().start;

            if self.check_next(&TokenMatch::Dot) {
                // We are in a field access
                let lhs = ast::Expr::Ident(self.make_ident()?)
                    .into_spanned(ast::to_rng(start..self.curr_span().end));
                self.eat_if(&TokenMatch::Dot);

                ast::Expr::FieldAccess { lhs: box lhs, rhs: box self.make_lh_expr()? }
                    .into_spanned(ast::to_rng(start..self.curr_span().end))
            } else if self.check_next(&TokenMatch::OpenBracket) {
                // We are in an array index expr
                let start = self.curr_span().start;

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
                    .into_spanned(ast::to_rng(start..self.curr_span().end))
            } else {
                let start = self.curr_span().start;

                let mut path = self.make_path()?;
                self.eat_whitespace();
                let is_func_call =
                    (self.curr.kind == TokenMatch::Lt || self.curr.kind == TokenMatch::OpenParen);

                if path.segs.len() == 1 && !is_func_call {
                    ast::Expr::Ident(path.segs.remove(0))
                        .into_spanned(ast::to_rng(start..self.curr_span().end))
                } else {
                    // We are most likely in a function call
                    if is_func_call {
                        let start = self.curr_span().start;

                        let type_args = if self.curr.kind == TokenMatch::Lt {
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
                        };

                        self.eat_whitespace();
                        self.eat_if(&TokenMatch::OpenParen);

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
                        // This is duplicated iff we have a no arg call
                        self.eat_whitespace();
                        self.eat_if(&TokenMatch::CloseParen);

                        ast::Expr::Call { path, type_args, args }
                            .into_spanned(ast::to_rng(start..self.curr_span().end))
                    } else {
                        todo!("{:?}", self.curr)
                    }
                }
            }
        } else {
            todo!()
        })
    }

    /// Build an optional `AssocOp`.
    fn make_op(&mut self) -> ParseResult<Option<AssocOp>> {
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
            TokenKind::Semi => {
                self.eat_if(&TokenMatch::Semi);
                None
            }
            TokenKind::CloseParen => {
                self.eat_if(&TokenMatch::CloseParen);
                None
            }
            TokenKind::CloseBracket => {
                self.eat_if(&TokenMatch::CloseBracket);
                None
            }
            TokenKind::CloseBrace => {
                self.eat_if(&TokenMatch::CloseBrace);
                None
            }
            // The stops expressions at `match (expr)` sites
            TokenKind::OpenBrace => {
                self.eat_if(&TokenMatch::OpenBrace);
                None
            }
            // Argument lists, array elements, any initializer stuff (structs, enums)
            TokenKind::Comma => None,
            t => todo!("Error found {:?} {:?}", t, &self.items()),
        })
    }

    fn make_enum_init(&mut self) -> ParseResult<ast::Expression> {
        let start = self.input_idx;

        self.eat_if_kw(kw::Enum);
        let mut path = self.make_path()?;
        let variant = path.segs.pop().unwrap();
        let items = self.make_expr_list(&TokenMatch::OpenParen, &TokenMatch::CloseParen)?;

        let span = ast::to_rng(start..self.curr_span().end);
        Ok(ast::Expr::EnumInit { path, variant, items }.into_spanned(span))
    }

    fn make_struct_init(&mut self) -> ParseResult<ast::Expression> {
        let start = self.input_idx;

        self.eat_if_kw(kw::Struct);
        let path = self.make_path()?;

        let fields = self.make_field_list()?;

        let span = ast::to_rng(start..self.curr_span().end);
        Ok(ast::Expr::StructInit { path, fields }.into_spanned(span))
    }

    fn make_block(&mut self) -> ParseResult<ast::Block> {
        let start = self.input_idx;
        let mut stmts = vec![];
        // println!("{:?}", self.curr);
        // println!("{:?}", self.tokens);
        if self.cmp_seq_ignore_ws(&[TokenMatch::OpenBrace, TokenMatch::CloseBrace]) {
            self.eat_seq_ignore_ws(&[TokenMatch::OpenBrace, TokenMatch::CloseBrace]);
            let span = ast::to_rng(start..self.curr_span().end);
            return Ok(ast::Block { stmts: vec![ast::Stmt::Exit.into_spanned(span)], span });
        }

        self.eat_whitespace();
        if self.eat_if(&TokenMatch::OpenBrace) {
            loop {
                self.eat_whitespace();
                stmts.push(self.make_stmt()?);
                self.eat_whitespace();

                if self.eat_if(&TokenMatch::CloseBrace) {
                    break;
                }
            }
        }
        let span = ast::to_rng(start..self.curr_span().end);
        Ok(ast::Block { stmts, span })
    }

    fn make_stmt(&mut self) -> ParseResult<ast::Statement> {
        let start = self.input_idx;
        let stmt = if self.eat_if_kw(kw::Let) {
            self.make_assignment()?
        } else if self.eat_if_kw(kw::If) {
            self.make_if_stmt()?
        } else if self.eat_if_kw(kw::While) {
            self.make_while_stmt()?
        } else if self.eat_if_kw(kw::Match) {
            self.make_match_stmt()?
        } else if self.eat_if_kw(kw::Return) {
            self.make_return_stmt()?
        } else if self.eat_if_kw(kw::Exit) {
            self.eat_whitespace();
            ast::Stmt::Exit
        } else {
            self.make_expr_stmt()?
        };

        self.eat_whitespace();
        self.eat_if(&TokenMatch::Semi);
        let span = ast::to_rng(start..self.curr_span().end);
        Ok(stmt.into_spanned(span))
    }

    fn make_assignment(&mut self) -> ParseResult<ast::Stmt> {
        self.eat_whitespace();

        let lval = self.make_lh_expr()?;
        self.eat_whitespace();

        self.eat_if(&TokenMatch::Eq);
        self.eat_whitespace();

        let rval = self.make_expr()?;

        self.eat_whitespace();
        self.eat_if(&TokenMatch::Semi);
        Ok(ast::Stmt::Assign { lval, rval })
    }

    fn make_if_stmt(&mut self) -> ParseResult<ast::Stmt> {
        self.eat_whitespace();

        let cond = self.make_expr()?;
        self.eat_whitespace();

        let blk = self.make_block()?;
        self.eat_whitespace();

        let els = if self.eat_if_kw(kw::Else) {
            self.eat_whitespace();
            Some(self.make_block()?)
        } else {
            None
        };
        self.eat_whitespace();
        self.eat_if(&TokenMatch::Semi);
        Ok(ast::Stmt::If { cond, blk, els })
    }

    fn make_while_stmt(&mut self) -> ParseResult<ast::Stmt> {
        self.eat_whitespace();

        let cond = self.make_expr()?;
        self.eat_whitespace();

        let stmts = self.make_block()?;
        self.eat_whitespace();

        self.eat_if(&TokenMatch::Semi);
        Ok(ast::Stmt::While { cond, stmts })
    }

    fn make_match_stmt(&mut self) -> ParseResult<ast::Stmt> {
        self.eat_whitespace();

        let expr = self.make_expr()?;
        self.eat_whitespace();

        self.eat_if(&TokenMatch::OpenBrace);
        let arms = self.make_arms()?;

        self.eat_whitespace();
        self.eat_if(&TokenMatch::OpenBrace);
        self.eat_whitespace();

        self.eat_if(&TokenMatch::Semi);
        Ok(ast::Stmt::Match { expr, arms })
    }

    fn make_return_stmt(&mut self) -> ParseResult<ast::Stmt> {
        self.eat_whitespace();

        if self.curr.kind == TokenMatch::Semi {
            return Ok(ast::Stmt::Exit);
        }
        let expr = self.make_expr()?;
        self.eat_whitespace();
        self.eat_if(&TokenMatch::Semi);

        Ok(ast::Stmt::Ret(expr))
    }

    fn make_expr_stmt(&mut self) -> ParseResult<ast::Stmt> {
        // @copypaste We are sort of taking this from `advance_to_op` but limiting the choices to
        // just calls and trait method calls
        Ok(if self.curr.kind == TokenMatch::Ident {
            let expr = self.make_lh_expr()?;
            self.eat_whitespace();

            match expr.val {
                ast::Expr::Ident(_) => {
                    // +=
                    if self.cmp_seq(&[TokenMatch::Plus, TokenMatch::Eq]) {
                        self.eat_seq(&[TokenMatch::Plus, TokenMatch::Eq]);
                        self.eat_whitespace();
                        println!("{}", &self.input[self.input_idx..]);
                        let rval = self.make_expr()?;
                        ast::Stmt::AssignOp { lval: expr, rval, op: ast::BinOp::Add }
                    // -=
                    } else if self.cmp_seq(&[TokenMatch::Minus, TokenMatch::Eq]) {
                        self.eat_seq(&[TokenMatch::Minus, TokenMatch::Eq]);
                        self.eat_whitespace();
                        let rval = self.make_expr()?;
                        ast::Stmt::AssignOp { lval: expr, rval, op: ast::BinOp::Add }
                    } else if self.curr.kind == TokenMatch::Eq {
                        self.eat_if(&TokenMatch::Eq);
                        self.eat_whitespace();
                        let rval = self.make_expr()?;
                        ast::Stmt::Assign { lval: expr, rval }
                    } else {
                        todo!("{}", &self.input[self.input_idx..])
                    }
                }
                ast::Expr::Deref { indir, expr } => todo!(),
                ast::Expr::AddrOf(_) => todo!(),
                ast::Expr::Array { ident, exprs } => todo!(),
                ast::Expr::Urnary { op, expr } => todo!(),
                ast::Expr::Binary { op, lhs, rhs } => todo!(),
                ast::Expr::Parens(_) => todo!(),
                ast::Expr::Call { .. } => ast::Stmt::Call(expr),
                ast::Expr::TraitMeth { trait_, args, type_args } => todo!(),
                ast::Expr::FieldAccess { lhs, rhs } => todo!(),
                ast::Expr::StructInit { path, fields } => todo!(),
                ast::Expr::EnumInit { path, variant, items } => todo!(),
                ast::Expr::ArrayInit { items } => todo!(),
                ast::Expr::Value(_) => todo!(),
            }
        } else if self.curr.kind == TokenMatch::Lt {
            todo!("Trait method calls")
        } else {
            // TODO: handle blocks `{}` as stmt and expr
            todo!("{:?}", self.curr)
        })
    }

    fn make_arms(&mut self) -> ParseResult<Vec<ast::MatchArm>> {
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
            self.eat_if(&TokenMatch::Comma);

            let span = ast::to_rng(start..self.curr_span().end);
            arms.push(ast::MatchArm { pat, blk, span })
        }
        Ok(arms)
    }

    fn make_pat(&mut self) -> ParseResult<ast::Pattern> {
        let start = self.input_idx;

        // TODO: make this more robust
        // could be `::mod::Name::Variant`
        Ok(if self.curr.kind == TokenKind::Ident {
            let mut path = self.make_path()?;
            // TODO: make this more robust
            // eventually calling an enum by variant needs to work which is the same as an ident
            if path.segs.len() > 1 {
                let variant = path
                    .segs
                    .pop()
                    .ok_or_else(|| ParseError::Expected("pattern", "nothing".to_string()))?;

                // @PARSE_ENUMS
                let items = if self.eat_if(&TokenMatch::OpenParen) {
                    self.eat_whitespace();
                    let mut pats = vec![];
                    loop {
                        // We have reached the end of the patterns (this allows trailing commas)
                        if self.curr.kind == TokenMatch::CloseParen {
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

                    self.eat_whitespace();
                    self.eat_if(&TokenMatch::CloseParen);

                    pats
                } else {
                    vec![]
                };

                let span = ast::to_rng(start..self.curr_span().end);
                ast::Pat::Enum { path, variant, items }.into_spanned(span)
            } else {
                // TODO: binding needs span
                let span = ast::to_rng(start..self.curr_span().end);
                ast::Pat::Bind(ast::Binding::Wild(path.segs.remove(0))).into_spanned(span)
            }
        } else if self.eat_if(&TokenMatch::OpenBracket) {
            self.eat_whitespace();
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
            self.eat_whitespace();
            self.eat_if(&TokenMatch::CloseBracket);

            let span = ast::to_rng(start..self.curr_span().end);
            ast::Pat::Array { size: pats.len(), items: pats }.into_spanned(span)
        } else if self.curr.kind == TokenMatch::Literal {
            // Literal
            let span = ast::to_rng(start..self.curr_span().end);
            ast::Pat::Bind(ast::Binding::Value(self.make_literal()?)).into_spanned(span)
        } else {
            todo!("{:?}", self.curr)
        })
    }

    /// Parse `ident[ws]:[ws]type[ws],[ws]ident: type[ws]` everything inside the parens is optional.
    fn make_params(&mut self) -> ParseResult<Vec<ast::Param>> {
        let mut params = vec![];
        loop {
            let start = self.input_idx;
            let ident = self.make_ident()?;

            self.eat_if(&TokenMatch::Colon);
            self.eat_whitespace();

            let ty = self.make_ty()?;

            let span = ast::to_rng(start..self.curr_span().end);
            params.push(ast::Param { ident, ty, span });

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
                let span = ast::to_rng(start..self.curr_span().end);
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
        // @copypaste
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
                    LiteralKind::Int { base, empty_int } => Val::Int(text.parse()?),
                    LiteralKind::Float { base, empty_exponent } => Val::Float(text.parse()?),
                    LiteralKind::Char { terminated } => {
                        if text.len() == 1 {
                            Val::Char(self.input_curr().chars().next().unwrap())
                        } else {
                            return Err(ParseError::Error("multi character `char`"));
                        }
                    }
                    LiteralKind::Str { terminated } => {
                        Val::Str(Ident::new(self.curr_span(), &self.input_curr().replace("\"", "")))
                    }
                    LiteralKind::ByteStr { .. }
                    | LiteralKind::RawStr { .. }
                    | LiteralKind::RawByteStr { .. }
                    | LiteralKind::Byte { .. } => todo!(),
                };
                self.eat_if(&TokenMatch::Literal);
                val.into_spanned(span)
            }
            tkn => {
                todo!("{:?}", tkn)
                // return Err(ParseError::IncorrectToken);
            }
        })
    }

    /// parse a list of types .
    fn make_types(&mut self, open: &TokenMatch, close: &TokenMatch) -> ParseResult<Vec<ast::Type>> {
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
    /// This also handles `*type, [lit_int, type]` and user defined items.
    fn make_ty(&mut self) -> ParseResult<ast::Type> {
        let start = self.input_idx;
        Ok(match self.curr.kind {
            TokenKind::Ident => {
                let key: Result<kw::Keywords, _> = self.input_curr().try_into();
                if let Ok(key) = key {
                    match key {
                        _ => todo!(),
                    }
                } else {
                    let segs = self.make_seg()?;
                    let span = ast::to_rng(start..self.curr_span().end);
                    self.eat_whitespace();
                    ast::Ty::Path(Path { segs, span }).into_spanned(span)
                }
            }
            TokenKind::OpenParen => {
                todo!()
            }
            TokenKind::OpenBracket => {
                let start = self.curr_span().start;
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
            tkn => todo!("Unknown token {:?} {}", tkn, &self.input[self.input_idx..]),
        })
    }

    /// Any type that follows `lit_int; type][ws]`.
    fn make_array_type(&mut self, start: usize) -> ParseResult<ast::Type> {
        let size = if let TokenKind::Literal {
            kind: LiteralKind::Int { base: Base::Decimal, .. },
            ..
        } = self.curr.kind
        {
            self.input_curr().parse()?
        } else {
            return Err(ParseError::Expected("lit", self.input_curr().to_string()));
        };
        // [ -->lit; -->type]
        self.eat_if(&TokenMatch::Literal);
        self.eat_whitespace();
        self.eat_if(&TokenMatch::Semi);
        self.eat_whitespace();

        let ty = self.make_ty()?;
        let x = Ok(ast::Ty::Array { size, ty: box ty }.into_spanned(start..self.curr_span().end));
        self.eat_if(&TokenMatch::CloseBracket);
        x
    }

    /// Parse `::ident[w]::ident[ws]...`.
    fn make_path(&mut self) -> ParseResult<Path> {
        let start = self.input_idx;
        let segs = self.make_seg()?;
        Ok(Path { segs, span: ast::to_rng(start..self.curr_span().end) })
    }

    /// Parse `ident[ws]::ident[ws]...`.
    fn make_seg(&mut self) -> ParseResult<Vec<Ident>> {
        let mut ids = vec![];
        loop {
            self.eat_seq(&[TokenMatch::Colon, TokenMatch::Colon]);
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
        let first = iter.next().unwrap_or(&TokenMatch::Unknown);
        if first != &self.curr.kind && self.curr.kind != TokenMatch::Whitespace {
            return false;
        }

        let tkns = self.tokens.iter().filter(|t| t.kind != TokenMatch::Whitespace);
        tkns.zip(iter).all(|(ours, cmp)| cmp == &ours.kind)
    }

    /// Throw away a sequence of tokens.
    ///
    /// Returns true if all the given tokens were matched.
    fn eat_seq_ignore_ws<'i>(&mut self, iter: impl IntoIterator<Item = &'i TokenMatch>) -> bool {
        for kind in iter {
            if kind == &self.curr.kind || self.curr.kind == TokenMatch::Whitespace {
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
        let stop = self.input_idx + self.curr.len;
        (self.input_idx..stop).into()
    }
}

#[test]
fn parse_lit_const() {
    let input = r#"
const foo: [3; int] = 1;
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn parse_func_header() {
    let input = r#"
fn add(x: int, y: int) -> int {  }
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
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
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn struct_decl_generics() {
    let input = r#"
struct foo<T, U> {
    x: T,
    y: U,
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn enum_decl() {
    let input = r#"
enum foo {
    a, b, c
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn enum_decl_generics() {
    let input = r#"
enum foo<T, X, U> {
    a(X, string), b, c(T, U), d([2; int])
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn impl_decl() {
    let input = r#"
impl add<string> {
    fn add(a: string, b: string) -> string {
        return concat(a, b);
    }
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn import_decl() {
    let input = r#"
import foo::bar::baz;
import ::bar::baz;
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn parse_multi_binop_ident() {
    let input = r#"
fn add(x: int, y: int) -> int {
    let z = a + b * (c + d);
    return z;
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn parse_multi_binop_lit() {
    let input = r#"
fn add(x: int, y: int) -> int {
    let z = 1 + 2 * 5 + 6;
    return z;
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn parse_func_call_no_args() {
    let input = r#"
fn add() {
    let x = foo();
    foo();
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn parse_func_call_args_generics() {
    let input = r#"
fn add() {
    let x = foo<T, U>(a, 1 + 2);
    foo<T>(a);
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn parse_assign_op() {
    let input = r#"
fn add() {
    let x = 10;
    x += 3+5;
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn parse_assign_array() {
    let input = r#"
fn add() {
    let x = [10, call(), 1+1+2];
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn parse_assign_struct() {
    let input = r#"
fn add() {
    let x = struct foo { x: 1, y: "string" };
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn parse_assign_enum() {
    let input = r#"
fn add() {
    let x = enum foo::bar(a, 1+0, 1.6);
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
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
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
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
        }
    }
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn parse_while_stmt() {
    let input = r#"
fn add() {
    while (x > 10) {
        call(1, 2, 3);
        x += 1;
    }
    return;
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}

#[test]
fn parse_trait_method() {
    let input = r#"
fn add() {
    let x = <<T>::add>(1, 2);
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
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
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
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
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}
