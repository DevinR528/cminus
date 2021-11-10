use std::convert::TryInto;

crate use crate::ast::{
    lex::{self, TokenKind, TokenMatch},
    parsy::symbol::Ident,
    types as ast,
};
use crate::ast::{
    lex::{LiteralKind, Token},
    types::{Path, Spany, Val},
};

mod error;
mod kw;
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
            lex::tokenize(input).chain(Some(Token::new(TokenKind::EOF, 0))).collect::<Vec<_>>();
        println!("{:#?}", tokens);
        let curr = tokens.remove(0);
        Self { tokens, curr, input, ..Default::default() }
    }

    pub fn items(&self) -> &[ast::Declaration] {
        &self.items
    }

    pub fn parse(&mut self) -> ParseResult<()> {
        loop {
            if self.curr.kind == TokenKind::EOF {
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
                    println!("{}", self.input_curr());
                    let keyword: kw::Keywords = self.input_curr().try_into()?;
                    match keyword {
                        kw::Keywords::Const => self.parse_const()?,
                        kw::Keywords::Fn => self.parse_fn()?,
                        kw::Keywords::Impl => {}
                        kw::Keywords::Struct => {}
                        kw::Keywords::Enum => {}
                        kw::Keywords::Trait => {}
                        kw::Keywords::Use => {}
                        _ => todo!("can we reach here?"),
                    }
                }
                TokenKind::Pound => {
                    self.eat_attr();
                    continue;
                }
                TokenKind::CloseBrace => {
                    self.eat_if(&TokenMatch::CloseBrace);
                    if self.curr.kind == TokenKind::EOF {
                        break;
                    }
                }
                TokenKind::Unknown => todo!("Unknown token"),
                tkn => todo!("Unknown token {:?}", tkn),
            }
            self.eat_whitespace();
        }
        Ok(())
    }

    // Parse `const name: type = expr;`
    fn parse_const(&mut self) -> ParseResult<()> {
        let start = self.input_idx;

        self.eat_if_kw(kw::Keywords::Const);
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
        self.items.push(
            ast::Decl::Const(ast::Const { ident: id, ty, init: expr, span }).into_spanned(span),
        );

        Ok(())
    }

    // Parse `fn name<T>(it: T) -> int { .. }` with or without generics.
    fn parse_fn(&mut self) -> ParseResult<()> {
        let start = self.input_idx;

        self.eat_if_kw(kw::Keywords::Fn);
        self.eat_whitespace();

        let ident = self.make_ident()?;

        let generics = self.make_generics()?;

        self.eat_if(&TokenMatch::OpenParen);
        self.eat_whitespace();

        let params = self.make_params()?;

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
        self.items.push(
            ast::Decl::Func(ast::Func { ident, ret, generics, params, stmts, span })
                .into_spanned(span),
        );
        println!("{:?}", self.items());
        Ok(())
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
            todo!()
        } else if self.curr.kind == TokenMatch::OpenParen {
            self.eat_if(&TokenMatch::OpenParen);

            let ex = self.make_expr()?;

            self.eat_if(&TokenMatch::OpenParen);

            // tuple
            // TODO: check there are only commas maybe??
            Ok(ex)
        } else if self.curr.kind == TokenMatch::Ident {
            let x: Result<kw::Keywords, _> = self.input_curr().try_into();
            if let Ok(key) = x {
                match key {
                    kw::Enum => self.make_enum_init(),
                    kw::Struct => self.make_struct_init(),
                    t => todo!("error {:?}", self.curr),
                }
            } else {
                // Shunting Yard algo http://en.wikipedia.org/wiki/Shunting_yard_algorithm
                let mut exstack: Vec<ast::Expression> = vec![];
                let mut opstack: Vec<AssocOp> = vec![];
                while self.curr.kind != TokenMatch::Semi {
                    self.eat_whitespace();
                    let (ex, op) = self.advance_to_op()?;
                    if let Some(next) = op {
                        let prev = opstack.pop();
                        // if the previous operator is of a higher precedence than the incoming
                        match prev {
                            Some(prev) if next.precedence() < prev.precedence() => {
                                let lhs = exstack.pop().unwrap();
                                let bin = prev;

                                println!(
                                    "prev: {:?} next: {:?} lhs {:?} rhs {:?}",
                                    prev, next, lhs, ex
                                );
                                let span = ast::to_rng(lhs.span.start..ex.span.end);
                                let finish = ast::Expr::Binary {
                                    op: bin.to_ast_binop().unwrap(),
                                    lhs: box lhs,
                                    rhs: box ex,
                                }
                                .into_spanned(span);

                                if matches!(bin.fixity(), Fixit::Left) {
                                    if let Some(lhs_fix) = exstack.pop() {
                                        let bin = opstack.pop().unwrap();
                                        let lfix = ast::Expr::Binary {
                                            op: bin.to_ast_binop().unwrap(),
                                            lhs: box lhs_fix,
                                            rhs: box finish,
                                        }
                                        .into_spanned(span);
                                        exstack.push(lfix);
                                    } else {
                                        exstack.push(finish);
                                    }
                                } else {
                                    // I don't think we should get here because of
                                    // advance_to_op/make_lh_expr
                                    todo!()
                                }
                            }
                            _ => {
                                exstack.push(ex);
                                if let Some(p) = prev {
                                    opstack.push(p);
                                }
                                opstack.push(next);
                            }
                        }
                    } else if self.eat_if(&TokenMatch::Semi) {
                        loop {
                            println!("{:?}", exstack);
                            println!("{:?}", opstack);
                            match (exstack.pop(), opstack.pop()) {
                                (Some(lhs), Some(bin)) => {
                                    println!(
                                        "prev: {:?} next: {:?} lhs {:?} rhs {:?}",
                                        bin, op, lhs, ex
                                    );
                                    if let Some(first) = exstack.pop() {
                                        let span = ast::to_rng(lhs.span.start..first.span.end);
                                        let finish = ast::Expr::Binary {
                                            op: bin.to_ast_binop().unwrap(),
                                            lhs: box first,
                                            rhs: box lhs,
                                        }
                                        .into_spanned(span);

                                        exstack.push(finish);
                                    } else {
                                        let span = ast::to_rng(lhs.span.start..ex.span.end);
                                        let finish = ast::Expr::Binary {
                                            op: bin.to_ast_binop().unwrap(),
                                            lhs: box lhs,
                                            rhs: box ex,
                                        }
                                        .into_spanned(span);

                                        exstack.push(finish);
                                        break;
                                    }
                                }
                                (Some(expr), None) => {
                                    return Ok(expr);
                                }
                                (None, Some(_)) => todo!("no expression but an op"),
                                (None, None) => {
                                    return Ok(ex);
                                }
                            }
                        }
                        break;
                    } else {
                        todo!("{:?}", self.tokens)
                    }
                }

                exstack.pop().ok_or(ParseError::Error("failed to generate expression"))
            }
        } else {
            todo!()
        }
    }

    /// Helper to build a left hand expression and an optional `AssocOp`.
    ///
    /// - anything with that starts with an ident
    ///     - field access, calls, etc.
    /// - literals
    /// - trait method calls
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
        } else if self.curr.kind == TokenMatch::Lt {
            todo!("Trait method calls")
        } else if self.curr.kind == TokenMatch::Bang {
            todo!("not")
        } else if self.curr.kind == TokenMatch::Minus {
            todo!("negative")
        } else if self.curr.kind == TokenMatch::Tilde {
            todo!("negation")
        } else if self.curr.kind == TokenMatch::Star {
            todo!("pointer")
        } else if self.curr.kind == TokenMatch::And {
            todo!("addrof")
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

            if self.curr.kind == TokenMatch::Dot {
                let lhs = ast::Expr::Ident(self.make_ident()?)
                    .into_spanned(ast::to_rng(start..self.curr_span().end));

                self.eat_if(&TokenMatch::Dot);

                ast::Expr::FieldAccess { lhs: box lhs, rhs: box self.make_lh_expr()? }
                    .into_spanned(ast::to_rng(start..self.curr_span().end))
            } else if self.curr.kind == TokenMatch::OpenParen {
                let start = self.curr_span().start;

                let segs = self.make_seg()?;
                let path = ast::Path { segs, span: ast::to_rng(start..self.curr_span().end) };

                let type_args = if self.curr.kind == TokenMatch::Lt {
                    self.eat_if(&TokenMatch::Lt);
                    let mut gen_args = vec![];
                    loop {
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

                self.eat_if(&TokenMatch::OpenParen);

                let mut args = vec![];
                loop {
                    args.push(self.make_expr()?);
                    self.eat_whitespace();
                    if self.eat_if(&TokenMatch::Comma) {
                        self.eat_whitespace();
                        continue;
                    } else {
                        break;
                    }
                }

                self.eat_if(&TokenMatch::CloseParen);

                ast::Expr::Call { path, type_args, args }
                    .into_spanned(ast::to_rng(start..self.curr_span().end))
            } else if self.curr.kind == TokenMatch::OpenBracket {
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
                ast::Expr::Ident(self.make_ident()?)
                    .into_spanned(ast::to_rng(start..self.curr_span().end))
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
            TokenKind::Semi => None,
            t => todo!("Error found {:?}", t),
        })
    }

    fn make_enum_init(&mut self) -> ParseResult<ast::Expression> {
        todo!()
    }

    fn make_struct_init(&mut self) -> ParseResult<ast::Expression> {
        todo!()
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

        if self.eat_if(&TokenMatch::OpenBrace) {
            loop {
                self.eat_whitespace();
                stmts.push(self.make_stmt()?);
                if self.eat_if(&TokenMatch::CloseBrace) {
                    break;
                }
            }
            self.eat_if(&TokenMatch::CloseBrace);
        }
        let span = ast::to_rng(start..self.curr_span().end);
        Ok(ast::Block { stmts, span })
    }

    fn make_stmt(&mut self) -> ParseResult<ast::Statement> {
        let start = self.input_idx;
        let stmt = if self.eat_if_kw(kw::Keywords::Let) {
            self.make_assignment()?
        } else if self.eat_if_kw(kw::Keywords::If) {
            self.make_if_stmt()?
        } else if self.eat_if_kw(kw::Keywords::While) {
            self.make_while_stmt()?
        } else if self.eat_if_kw(kw::Keywords::Match) {
            self.make_match_stmt()?
        } else if self.eat_if_kw(kw::Keywords::Return) {
            self.make_return_stmt()?
        } else if self.eat_if_kw(kw::Keywords::Exit) {
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

        let els = if self.eat_if_kw(kw::Keywords::Else) {
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

        let arms = self.make_arms()?;
        self.eat_whitespace();

        self.eat_if(&TokenMatch::Semi);
        Ok(ast::Stmt::Match { expr, arms })
    }

    fn make_return_stmt(&mut self) -> ParseResult<ast::Stmt> {
        self.eat_whitespace();

        let expr = self.make_expr()?;
        self.eat_whitespace();
        self.eat_if(&TokenMatch::Semi);

        Ok(ast::Stmt::Ret(expr))
    }

    fn make_expr_stmt(&mut self) -> ParseResult<ast::Stmt> {
        todo!("{:?}", self.curr)
    }

    fn make_arms(&mut self) -> ParseResult<Vec<ast::MatchArm>> {
        self.eat_whitespace();
        let mut arms = vec![];
        loop {
            let start = self.input_idx;

            let pat = self.make_pat()?;
            let blk = self.make_block()?;
            let span = ast::to_rng(start..self.curr_span().end);
            arms.push(ast::MatchArm { pat, blk, span })
        }

        self.eat_whitespace();
        self.eat_if(&TokenMatch::Semi);
        Ok(arms)
    }

    fn make_pat(&mut self) -> ParseResult<ast::Pattern> {
        self.eat_whitespace();
        let start = self.input_idx;

        // TODO: make this more robust
        // could be `::mod::Name::Variant`
        Ok(if self.curr.kind == TokenKind::Ident {
            // TODO: make this more robust
            // eventually calling an enum by variant needs to work which is the same as an ident
            if self.cmp_seq(&[TokenMatch::Colon, TokenMatch::Colon, TokenMatch::Ident]) {
                let mut ident = self.make_path()?;
                let variant = ident
                    .segs
                    .pop()
                    .ok_or(ParseError::Expected("pattern", "nothing".to_string()))?;

                // @PARSE_ENUMS
                let items = if self.eat_if(&TokenMatch::OpenParen) {
                    self.eat_whitespace();
                    let mut pats = vec![];
                    loop {
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
                ast::Pat::Enum { ident, variant, items }.into_spanned(span)
            } else {
                let ident = self.make_ident()?;
                // TODO: binding needs span
                let span = ast::to_rng(start..self.curr_span().end);
                ast::Pat::Bind(ast::Binding::Wild(ident)).into_spanned(span)
            }
        } else if self.eat_if(&TokenMatch::OpenBracket) {
            self.eat_whitespace();
            let mut pats = vec![];
            loop {
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
        } else if matches!(self.curr.kind, TokenKind::Literal { .. }) {
            let span = ast::to_rng(start..self.curr_span().end);
            ast::Pat::Bind(ast::Binding::Value(self.make_literal()?)).into_spanned(span)
        } else {
            todo!("{:?}", self.curr)
            // return Err(ParseError::IncorrectToken);
        })
    }

    /// Parse a literal.
    fn make_literal(&mut self) -> ParseResult<ast::Value> {
        // @copypaste
        Ok(match self.curr.kind {
            TokenKind::Ident => {
                let keyword: kw::Keywords = self.input_curr().try_into()?;
                match keyword {
                    kw::Keywords::True => {
                        let expr = Val::Bool(true).into_spanned(self.curr_span());
                        self.eat_if_kw(kw::Keywords::True);
                        expr
                    }
                    kw::Keywords::False => {
                        let expr = Val::Bool(false).into_spanned(self.curr_span());
                        self.eat_if_kw(kw::Keywords::False);
                        expr
                    }
                    _ => todo!(),
                }
            }
            TokenKind::Literal { kind, suffix_start } => match kind {
                LiteralKind::Int { base, empty_int } => {
                    let span = self.curr_span();
                    let text = self.input_curr();
                    let expr = parse_integer(text, base, span)?;
                    self.eat_if(&TokenMatch::Literal);
                    expr
                }
                LiteralKind::Float { base, empty_exponent } => todo!(),
                LiteralKind::Char { terminated } => todo!(),
                LiteralKind::Byte { terminated } => todo!(),
                LiteralKind::Str { terminated } => todo!(),
                LiteralKind::ByteStr { terminated } => todo!(),
                LiteralKind::RawStr { n_hashes, err } => todo!(),
                LiteralKind::RawByteStr { n_hashes, err } => todo!(),
            },
            tkn => {
                todo!("{:?}", tkn)
                // return Err(ParseError::IncorrectToken);
            }
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
            tkn => todo!("Unknown token {:?}", tkn),
        })
    }

    /// Any type that follows `[lit_int; type]`
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

    /// Parse `(ident: type, ident: type)[ws]` everything inside the parens is optional.
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
        Ok(gens)
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
                continue;
            } else {
                break;
            }
        }
        Ok(ids)
    }

    /// Parse `ident[ws]`
    fn make_ident(&mut self) -> ParseResult<Ident> {
        let span = self.curr_span();
        let id = Ident::new(span, self.input[span.start..span.end].to_string());
        self.eat_if(&TokenMatch::Ident);
        Ok(id)
    }

    fn eat_whitespace(&mut self) {
        while self.eat_if(&TokenMatch::Whitespace) {}
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

fn parse_integer(num: &str, base: Base, span: ast::Range) -> ParseResult<ast::Value> {
    Ok(Val::Int(num.parse()?).into_spanned(span))
}

// #[test]
// fn do_parse_stuff_const() {
//     let input = r#"
// const foo: [3; int] = 1;
// "#;
//     let mut parser = AstBuilder::new(input);
//     parser.parse().unwrap();
//     println!("{:#?}", parser.items());
// }

// #[test]
// fn do_parse_stuff() {
//     let input = r#"
// fn add(x: int, y: int) -> int {  }
// "#;
//     let mut parser = AstBuilder::new(input);
//     parser.parse().unwrap();
//     println!("{:#?}", parser.items());
// }

#[test]
fn do_parse_stuff() {
    let input = r#"
fn add(x: int, y: int) -> int {
    let z = x + y + a + 1;
    return z;
}
"#;
    let mut parser = AstBuilder::new(input);
    parser.parse().unwrap();
    println!("{:#?}", parser.items());
}
