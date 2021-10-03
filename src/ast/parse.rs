use pest::iterators::{Pair, Pairs};

use crate::{
    ast::types::{
        BinOp, Block, Decl, Declaration, Expr, Expression, Field, FieldInit, Func, Param, Range,
        Statement, Stmt, Struct, Ty, Type, UnOp, Val, Var,
    },
    precedence::{Assoc, Operator, PrecClimber},
    Rule,
};

crate fn parse_decl(pair: Pair<'_, Rule>) -> Vec<Declaration> {
    pair.into_inner()
        .flat_map(|decl| {
            let span = to_span(&decl);
            match decl.as_rule() {
                Rule::func_decl => {
                    vec![Decl::Func(parse_func(decl.into_inner())).into_spanned(span)]
                }
                Rule::var_decl => parse_var_decl(decl.into_inner())
                    .into_iter()
                    .map(|var| Decl::Var(var).into_spanned(span))
                    .collect(),
                Rule::adt_decl => {
                    vec![Decl::Adt(parse_struct(decl.into_inner(), span)).into_spanned(span)]
                }
                _ => unreachable!("malformed declaration"),
            }
        })
        .collect()
}

fn parse_struct(struct_: Pairs<Rule>, span: Range) -> Struct {
    match struct_.map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        [(Rule::STRUCT, _), (Rule::ident, ident), (Rule::LBR, _), fields @ .., (Rule::RBR, _), (Rule::SC, _)] => {
            Struct {
                ident: ident.as_str().to_string(),
                fields: fields
                    .iter()
                    .map(|(_, p)| {
                        match p
                            .clone()
                            .into_inner()
                            .map(|p| (p.as_rule(), p))
                            .collect::<Vec<_>>()
                            .as_slice()
                        {
                            [(Rule::param, param), (Rule::SC, _)] => {
                                parse_struct_field_decl(param.clone())
                            }
                            _ => unreachable!("malformed struct fields"),
                        }
                    })
                    .collect(),
                span,
            }
        }
        _ => unreachable!("malformed function parameter"),
    }
}

fn parse_struct_field_decl(param: Pair<Rule>) -> Field {
    match param.into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        [(Rule::type_, ty), (Rule::var_name, var)] => {
            let ty = parse_ty(ty.clone());
            let span = to_span(var);
            match var.clone().into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice()
            {
                [(Rule::addrof, addr), (Rule::ident, id), array_parts @ ..] => {
                    let inner_span = (addr.as_span().start()..id.as_span().end()).into();
                    let ty = {
                        let indirection = addr.as_str().matches('*').count();
                        build_recursive_pointer_ty(indirection, ty, inner_span)
                    };
                    if !array_parts.is_empty() {
                        Field {
                            ty: build_recursive_ty(
                                array_parts.iter().filter_map(|(r, p)| match r {
                                    Rule::integer => {
                                        Some(p.as_str().parse().expect("invalid integer"))
                                    }
                                    Rule::LBK | Rule::RBK => None,
                                    _ => unreachable!("malformed array access"),
                                }),
                                ty,
                            ),
                            ident: id.as_str().to_string(),
                            span,
                        }
                    } else {
                        Field { ty, ident: id.as_str().to_string(), span: inner_span }
                    }
                }
                _ => unreachable!("malformed variable"),
            }
        }
        _ => unreachable!("malformed function parameter"),
    }
}

fn parse_param(param: Pair<Rule>) -> Param {
    match param.into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        [(Rule::type_, ty), (Rule::var_name, var)] => {
            let span = to_span(var);
            Param { ty: parse_ty(ty.clone()), ident: var.as_str().to_string(), span }
        }
        _ => unreachable!("malformed function parameter"),
    }
}

#[rustfmt::skip]
fn parse_func(func: Pairs<Rule>) -> Func {
    // println!("func = {}", func.to_json());
    match func.into_iter().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // int foo(int a, int b) { stmts }
        [(Rule::type_, ty), (Rule::ident, ident),
         (Rule::LP, _), (Rule::param_list, params), (Rule::RP, _), (Rule::LBR, _),
         action @ ..,
         (Rule::RBR, rbr)
        ] => {
            Func {
                ret: parse_ty(ty.clone()),
                ident: ident.as_str().to_string(),
                params: params.clone().into_inner().filter_map(|param| match param.as_rule() {
                    Rule::param => Some(parse_param(param)),
                    Rule::CM => None,
                    _ => unreachable!("malformed call statement"),
                }).collect(),
                stmts: action.iter().map(|(r, s)| match r {
                    Rule::stmt => parse_stmt(s.clone()),
                    Rule::var_decl => Stmt::VarDecl(
                        parse_var_decl(s.clone().into_inner())).into_spanned(to_span(s)
                    ),
                    _ => unreachable!("malformed statement"),
                }).collect(),
                span: (ty.as_span().start()..rbr.as_span().end()).into(),
            }
        }
        // int foo() { stmts }
        [(Rule::type_, ty), (Rule::ident, ident),
         (Rule::LP, _), (Rule::RP, _), (Rule::LBR, _),
         action @ ..,
         (Rule::RBR, rbr)
        ] => {
            Func {
                ret: parse_ty(ty.clone()),
                ident: ident.as_str().to_string(),
                params: vec![],
                stmts: action.iter().map(|(r, s)| match r {
                    Rule::stmt => parse_stmt(s.clone()),
                    Rule::var_decl => Stmt::VarDecl(
                        parse_var_decl(s.clone().into_inner())
                    ).into_spanned(to_span(s)),
                    _ => unreachable!("malformed statement"),
                }).collect(),
                span: (ty.as_span().start()..rbr.as_span().end()).into(),
            }
        }
        _ => unreachable!("malformed function"),
    }
}

fn parse_stmt(stmt: Pair<Rule>) -> Statement {
    let stmt = stmt.into_inner().next().unwrap();
    let span = to_span(&stmt);
    #[rustfmt::skip]
    match stmt.as_rule() {
        Rule::assing => {
            match stmt.clone()
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // [*]var = [*]expr;
                [(Rule::expr, var), (Rule::ASSIGN, _),
                    (Rule::expr, expr), (Rule::SC, _)
                ] => {
                    parse_lvalue(var.clone(), expr.clone(), span)
                }
                // var = { field: expr, optional: expr };
                [(Rule::expr, var), (Rule::ASSIGN, _),
                    (Rule::struct_assign, expr), (Rule::SC, _)
                ] => {
                    parse_lvalue(var.clone(), expr.clone(), span)
                }
                // var = { 1, 2, };
                [(Rule::expr, var), (Rule::ASSIGN, _),
                    (Rule::arr_init, expr), (Rule::SC, _)
                ] => {
                    parse_lvalue(var.clone(), expr.clone(), span)
                }
                _ => unreachable!("malformed assingment {}", stmt.to_json()),
            }
        }
        Rule::call_stmt => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // foo(x,y);
                [(Rule::ident, name), (Rule::LP, _),
                    (Rule::arg_list, args),
                (Rule::RP, _), (Rule::SC, _)] => {
                    Stmt::Call {
                        ident: name.as_str().to_string(),
                        args: args
                            .clone()
                            .into_inner()
                            .filter_map(|arg| match arg.as_rule() {
                                Rule::expr => Some(parse_expr(arg)),
                                Rule::CM => None,
                                _ => unreachable!("malformed call statement"),
                            })
                            .collect(),
                    }.into_spanned(span)
                }
                // foo();
                [(Rule::ident, name), (Rule::LP, _), (Rule::RP, _), (Rule::SC, _)] => {
                    Stmt::Call { ident: name.as_str().to_string(), args: vec![] }.into_spanned(span)
                }
                _ => unreachable!("malformed assingment"),
            }
        }
        Rule::if_stmt => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // if expr { stmts } [ else { stmts }]
                [(Rule::IF, _), (Rule::LP, _), (Rule::expr, expr), (Rule::RP, _),
                    (Rule::block_stmt, block), else_blk @ ..
                ] => {
                    Stmt::If {
                        cond: parse_expr(expr.clone()),
                        blk: parse_block(block.clone()),
                        els: match else_blk {
                            [(Rule::ELSE, _), (Rule::block_stmt, blk)] => {
                                Some(parse_block(blk.clone()))
                            }
                            [] => None,
                            _ => unreachable!("malformed if statement"),
                        },
                    }.into_spanned(span)
                }
                _ => unreachable!("malformed assingment"),
            }
        }
        Rule::while_stmt => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // while (expr) { stmts }
                [(Rule::WHILE, _), (Rule::LP, _), (Rule::expr, expr), (Rule::RP, _),
                    (Rule::stmt, stmt),
                ] => {
                    Stmt::While {
                        cond: parse_expr(expr.clone()),
                        // This will mostly be `Stmt::Block(..)` but sometimes just
                        // assignment i = i + 1;
                        stmt: box parse_stmt(stmt.clone()),
                    }.into_spanned(span)
                }
                _ => unreachable!("malformed assingment"),
            }
        }
        Rule::io_stmt => {
            let span = to_span(&stmt);
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // read(var);
                [(Rule::READ, _), (Rule::LP, _),
                    (Rule::variable, var),
                (Rule::RP, _), (Rule::SC, _)] => {
                    Stmt::Read (var.as_str().to_string())
                }
                // write(expr);
                [(Rule::WRITE, _), (Rule::LP, _),
                    (arg_rule, arg),
                (Rule::RP, _), (Rule::SC, _)] => {
                    Stmt::Write {
                        expr: match arg_rule {
                            Rule::expr => parse_expr(arg.clone()),
                            Rule::string => Expr::Value(
                                Val::Str(arg.as_str().replace("\"", "")).into_spanned(to_span(arg))
                            ).into_spanned(to_span(arg)),
                            _ => unreachable!("malformed write statement")
                        },
                    }
                }
                _ => unreachable!("malformed IO statement"),
            }.into_spanned(span)
        }
        Rule::ret_stmt => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // return expr;
                [(Rule::RETURN, _), (Rule::expr, expr), (Rule::SC, _)] => {
                    Stmt::Ret(parse_expr(expr.clone())).into_spanned(span)
                }
                _ => unreachable!("malformed return statement"),
            }
        }
        // exit;
        Rule::exit_stmt => Stmt::Exit.into_spanned(span),
        // { stmts }
        Rule::block_stmt => Stmt::Block(parse_block(stmt)).into_spanned(span),
        _ => unreachable!("malformed statement"),
    }
}

fn parse_ty(ty: Pair<Rule>) -> Type {
    let span = to_span(&ty);
    match ty.clone().into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // int x,y,z;
        [(rule, ty), (Rule::addrof, addr)] => {
            let indirection = addr.as_str().matches('*').count();
            let t = match rule {
                Rule::INT => Ty::Int,
                Rule::FLOAT => Ty::Float,
                Rule::CHAR => Ty::Char,
                Rule::VOID if addr.as_str().is_empty() => Ty::Void,
                _ => unreachable!("malformed addrof type {}", ty.to_json()),
            }
            .into_spanned(to_span(ty));
            build_recursive_pointer_ty(indirection, t, span).val
        }
        [(Rule::STRUCT, s), (Rule::ident, ident), (Rule::addrof, addr)] => {
            let indirection = addr.as_str().matches('*').count();
            build_recursive_pointer_ty(
                indirection,
                Ty::Adt(ident.as_str().to_string())
                    .into_spanned(s.as_span().start()..addr.as_span().end()),
                (s.as_span().start()..addr.as_span().end()).into(),
            )
            .val
        }
        _ => unreachable!("malformed type {}", ty.to_json()),
    }
    .into_spanned(span)
}

fn parse_block(blk: Pair<Rule>) -> Block {
    let span = to_span(&blk);
    match blk.into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // { stmt* }
        [(Rule::LBR, _), stmts @ .., (Rule::RBR, _)] => {
            Block { stmts: stmts.iter().map(|(_, s)| parse_stmt(s.clone())).collect(), span }
        }
        _ => unreachable!("malformed function"),
    }
}

fn parse_lvalue(var: Pair<Rule>, expr: Pair<'_, Rule>, span: Range) -> Statement {
    let lval = parse_expr(var);
    valid_lval(&lval.val).unwrap();
    Stmt::Assign { lval, rval: parse_expr(expr) }.into_spanned(span)
}

fn valid_lval(ex: &Expr) -> Result<(), String> {
    let mut stack = vec![ex];

    // validate lValue
    while let Some(val) = stack.pop() {
        match val {
            Expr::Parens(expr) | Expr::Deref { expr, .. } | Expr::AddrOf(expr) => {
                stack.push(&expr.val)
            }
            Expr::FieldAccess { lhs, rhs } => {
                stack.push(&lhs.val);
                stack.push(&rhs.val);
            }
            // Valid expressions that are not recursive
            Expr::Ident(..) | Expr::Array { .. } => {}
            Expr::Call { .. } => {
                return Err("no call expression".to_owned());
            }
            Expr::Urnary { .. } => {
                return Err("no urnary expression".to_owned());
            }
            Expr::Binary { .. } => {
                return Err("no binary expression".to_owned());
            }
            Expr::StructInit { .. } => {
                return Err("no struct init".to_owned());
            }
            Expr::ArrayInit { .. } => {
                return Err("no struct init".to_owned());
            }
            Expr::Value(_) => {
                return Err("no values".to_owned());
            }
        }
    }
    Ok(())
}

fn build_recursive_ty<I: Iterator<Item = usize>>(mut dims: I, base_ty: Type) -> Type {
    if let Some(size) = dims.next() {
        let span = base_ty.span;
        Ty::Array { size, ty: box build_recursive_ty(dims, base_ty) }.into_spanned(span)
    } else {
        base_ty
    }
}

fn build_recursive_pointer_ty(mut indir: usize, base_ty: Type, outer_span: Range) -> Type {
    if indir > 0 {
        let span = base_ty.span;
        Ty::Ptr(box build_recursive_pointer_ty(indir - 1, base_ty, outer_span))
            .into_spanned(outer_span)
    } else {
        base_ty
    }
}

fn parse_var_decl(var: Pairs<Rule>) -> Vec<Var> {
    match var.clone().into_iter().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // int x,y,z;
        [(Rule::type_, ty), (Rule::var_name, var_name), names @ .., (Rule::SC, _)] => {
            let ty = parse_ty(ty.clone());
            let parse_var_array = |var: &Pair<Rule>| {
                let span = to_span(var);
                match var
                    .clone()
                    .into_inner()
                    .map(|p| (p.as_rule(), p))
                    .collect::<Vec<_>>()
                    .as_slice()
                {
                    [(Rule::addrof, addr), (Rule::ident, id), array_parts @ ..] => {
                        let inner_span = (addr.as_span().start()..id.as_span().end()).into();
                        let ty = {
                            let indirection = addr.as_str().matches('*').count();
                            build_recursive_pointer_ty(indirection, ty.clone(), inner_span)
                        };
                        if !array_parts.is_empty() {
                            Var {
                                ty: build_recursive_ty(
                                    array_parts.iter().filter_map(|(r, p)| match r {
                                        Rule::integer => {
                                            Some(p.as_str().parse().expect("invalid integer"))
                                        }
                                        Rule::LBK | Rule::RBK => None,
                                        _ => unreachable!("malformed array access"),
                                    }),
                                    ty,
                                ),
                                ident: id.as_str().to_string(),
                                span,
                            }
                        } else {
                            Var { ty, ident: id.as_str().to_string(), span: inner_span }
                        }
                    }
                    _ => unreachable!("malformed variable"),
                }
            };

            vec![parse_var_array(var_name)]
                .into_iter()
                .chain(names.iter().filter_map(|(r, n)| match r {
                    Rule::var_name => Some(parse_var_array(n)),
                    Rule::CM => None,
                    _ => unreachable!("malformed variable declaration"),
                }))
                .collect()
        }
        _ => unreachable!("malformed function {:?}", var.map(|p| p.as_rule()).collect::<Vec<_>>()),
    }
}

fn parse_expr(expr: Pair<Rule>) -> Expression {
    // println!("{}", expr.to_json());
    let climber = PrecClimber::new(vec![
        // &&
        Operator::new(Rule::AND, Assoc::Left),
        // ||
        Operator::new(Rule::OR, Assoc::Left),
        // ==, !=
        Operator::new(Rule::EQ, Assoc::Left) | Operator::new(Rule::NE, Assoc::Left),
        //  >, >=, <, <=
        Operator::new(Rule::GT, Assoc::Left)
            | Operator::new(Rule::GE, Assoc::Left)
            | Operator::new(Rule::LT, Assoc::Left)
            | Operator::new(Rule::LE, Assoc::Left),
        // +, -
        Operator::new(Rule::PLUS, Assoc::Left) | Operator::new(Rule::SUB, Assoc::Left),
        // *, /
        Operator::new(Rule::MUL, Assoc::Left) | Operator::new(Rule::DIV, Assoc::Left),
        // ., ->
        Operator::new(Rule::DOT, Assoc::Left) | Operator::new(Rule::ARROW, Assoc::Left),
    ]);

    consume(expr, &climber, true)
}

fn consume(expr: Pair<'_, Rule>, climber: &PrecClimber<Rule>, first: bool) -> Expression {
    let primary = |p: Pair<'_, _>| consume(p, climber, false);
    let infix = |lhs: Expression, op: Pair<Rule>, rhs: Expression| {
        let span = lhs.span.start..rhs.span.end;
        let expr = match op.as_rule() {
            Rule::DOT => match &lhs.val {
                Expr::Deref { indir, expr: inner } => Expr::Deref {
                    indir: *indir,
                    expr: box Expr::FieldAccess { lhs: inner.clone(), rhs: box rhs }
                        .into_spanned(inner.span.start..span.end),
                },
                _ => Expr::FieldAccess { lhs: box lhs, rhs: box rhs },
            },
            Rule::ARROW => Expr::FieldAccess {
                lhs: box Expr::Deref { indir: 1, expr: box lhs }
                    .into_spanned(span.start..rhs.span.start),
                rhs: box rhs,
            },
            Rule::PLUS => Expr::Binary { op: BinOp::Add, lhs: box lhs, rhs: box rhs },
            Rule::SUB => Expr::Binary { op: BinOp::Sub, lhs: box lhs, rhs: box rhs },
            Rule::MUL => Expr::Binary { op: BinOp::Mul, lhs: box lhs, rhs: box rhs },
            Rule::DIV => Expr::Binary { op: BinOp::Div, lhs: box lhs, rhs: box rhs },
            Rule::GT => Expr::Binary { op: BinOp::Gt, lhs: box lhs, rhs: box rhs },
            Rule::GE => Expr::Binary { op: BinOp::Ge, lhs: box lhs, rhs: box rhs },
            Rule::LT => Expr::Binary { op: BinOp::Lt, lhs: box lhs, rhs: box rhs },
            Rule::LE => Expr::Binary { op: BinOp::Le, lhs: box lhs, rhs: box rhs },
            Rule::EQ => Expr::Binary { op: BinOp::Eq, lhs: box lhs, rhs: box rhs },
            Rule::NE => Expr::Binary { op: BinOp::Ne, lhs: box lhs, rhs: box rhs },
            Rule::AND => Expr::Binary { op: BinOp::And, lhs: box lhs, rhs: box rhs },
            Rule::OR => Expr::Binary { op: BinOp::Or, lhs: box lhs, rhs: box rhs },
            _ => unreachable!(),
        };
        expr.into_spanned(span)
    };

    let span = to_span(&expr);
    enum ExprKind {
        Deref(usize),
        AddrOf,
        None,
    }
    let mut wrapper = ExprKind::None;
    let ex = match expr.as_rule() {
        Rule::expr => climber.climb(expr.into_inner(), primary, infix),
        Rule::op => climber.climb(expr.into_inner(), primary, infix),
        Rule::term => {
            match expr.into_inner().map(|r| (r.as_rule(), r)).collect::<Vec<_>>().as_slice() {
                // x,y[],z.z
                [(Rule::variable, var)] => {
                    let span = to_span(var);
                    match var
                        .clone()
                        .into_inner()
                        .map(|p| (p.as_rule(), p))
                        .collect::<Vec<_>>()
                        .as_slice()
                    {
                        [(Rule::deref, deref), (Rule::ident, ident)] => {
                            let indir = deref.as_str();
                            if indir == "&" {
                                wrapper = ExprKind::AddrOf;
                            } else {
                                let indirection = indir.matches('*').count();
                                if indirection > 0 {
                                    wrapper = ExprKind::Deref(indirection);
                                }
                            }
                            Expr::Ident(ident.as_str().to_string()).into_spanned(to_span(ident))
                        }
                        [(Rule::deref, deref), (Rule::ident, ident), (Rule::LBK, _), (Rule::expr, expr), (Rule::RBK, _), rest @ ..] =>
                        {
                            let indir = deref.as_str();

                            let arr = Expr::Array {
                                ident: ident.as_str().to_string(),
                                exprs: vec![parse_expr(expr.clone())]
                                    .into_iter()
                                    .chain(rest.iter().filter_map(|(r, p)| match r {
                                        Rule::LBK | Rule::RBK => None,
                                        Rule::expr => Some(parse_expr(p.clone())),
                                        _ => unreachable!("malformed multi-dim array"),
                                    }))
                                    .collect(),
                            };
                            if indir == "&" {
                                wrapper = ExprKind::AddrOf;
                            } else {
                                let indirection = indir.matches('*').count();
                                if indirection > 0 {
                                    wrapper = ExprKind::Deref(indirection);
                                }
                            }

                            arr.into_spanned(ident.as_span().start()..span.end)
                        }
                        _ => unreachable!("malformed variable name {}", var.to_json()),
                    }
                }
                // 1,true,'a', "string"
                [(Rule::const_, konst)] => {
                    let inner_span = to_span(konst);
                    match konst.clone().into_inner().next().unwrap().as_rule() {
                        Rule::integer => Expr::Value(
                            Val::Int(konst.as_str().parse().unwrap()).into_spanned(inner_span),
                        ),
                        Rule::decimal => Expr::Value(
                            Val::Float(konst.as_str().parse().unwrap()).into_spanned(inner_span),
                        ),
                        Rule::charstr => {
                            let ch = konst.as_str().replace('\'', "").chars().collect::<Vec<_>>();
                            if let [c] = ch.as_slice() {
                                Expr::Value(Val::Char(*c).into_spanned(inner_span))
                            } else {
                                unreachable!("multiple char char is not allowed")
                            }
                        }
                        Rule::string => Expr::Value(
                            Val::Str(konst.as_str().to_string()).into_spanned(inner_span),
                        ),
                        r => unreachable!("malformed const expression {:?}", r),
                    }
                    .into_spanned(span)
                }
                // call()
                [(Rule::ident, ident), (Rule::LP, _), arg_list @ .., (Rule::RP, _)] => {
                    Expr::Call {
                        ident: ident.as_str().to_string(),
                        args: if let [(Rule::arg_list, args)] = arg_list {
                            args.clone()
                                .into_inner()
                                .filter_map(|p| match p.as_rule() {
                                    Rule::expr => Some(
                                        // if there are expressions as arguments they are
                                        // ordered on their own `1 + call(2*3)`  the 2*3
                                        // has nothing to do with 1 + whatever
                                        climber.climb(p.clone().into_inner(), primary, infix),
                                    ),
                                    Rule::CM => None,
                                    _ => unreachable!("malformed arguments in call {:?}", arg_list),
                                })
                                .collect::<Vec<_>>()
                        } else {
                            vec![]
                        },
                    }
                    .into_spanned(span)
                }
                // !true
                [(Rule::NOT, _), (Rule::expr, expr)] => {
                    let inner = climber.climb(expr.clone().into_inner(), primary, infix);
                    Expr::Urnary { op: UnOp::Not, expr: box inner }.into_spanned(span)
                }
                // (1 + 1)
                [(Rule::LP, _), (Rule::expr, expr), (Rule::RP, _)] => {
                    let inner = climber.climb(expr.clone().into_inner(), primary, infix);
                    Expr::Parens(box inner).into_spanned(span)
                }
                _ => unreachable!("malformed expression"),
            }
        }
        Rule::struct_assign => {
            let span = to_span(&expr);
            match expr.into_inner().map(|r| (r.as_rule(), r)).collect::<Vec<_>>().as_slice() {
                [(Rule::ident, name), (Rule::LBR, _), fields @ .., (Rule::RBR, _)] => {
                    Expr::StructInit {
                        name: name.as_str().to_string(),
                        fields: fields
                            .iter()
                            .filter_map(|(r, p)| match r {
                                Rule::field_expr => Some(parse_field_init(p.clone())),
                                Rule::CM => None,
                                _ => unreachable!("malformed struct initializer"),
                            })
                            .collect(),
                    }
                    .into_spanned(span)
                }
                _ => unreachable!("malformed struct assignment"),
            }
        }
        Rule::arr_init => {
            let span = to_span(&expr);
            match expr.into_inner().map(|r| (r.as_rule(), r)).collect::<Vec<_>>().as_slice() {
                [(Rule::LBR, _), fields @ .., (Rule::RBR, _)] => Expr::ArrayInit {
                    items: fields
                        .iter()
                        .filter_map(|(r, p)| match r {
                            Rule::arr_init | Rule::expr => Some(parse_expr(p.clone())),
                            Rule::CM => None,
                            _ => unreachable!("malformed array initializer"),
                        })
                        .collect(),
                }
                .into_spanned(span),
                _ => unreachable!("malformed struct assignment"),
            }
        }
        err => unreachable!("{:?}", err),
    };
    match wrapper {
        ExprKind::Deref(indir) => Expr::Deref { indir, expr: box ex }.into_spanned(span),
        ExprKind::AddrOf => Expr::AddrOf(box ex).into_spanned(span),
        ExprKind::None => ex,
    }
}

fn parse_field_init(field: Pair<Rule>) -> FieldInit {
    match field.clone().into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        [(Rule::ident, ident), (Rule::COLON, _), (Rule::expr, expr)] => FieldInit {
            ident: ident.as_str().to_string(),
            init: parse_expr(expr.clone()),
            span: (ident.as_span().start()..expr.as_span().end()).into(),
        },
        [(Rule::ident, ident), (Rule::COLON, _), (Rule::arr_init, expr)] => FieldInit {
            ident: ident.as_str().to_string(),
            init: parse_expr(expr.clone()),
            span: (ident.as_span().start()..expr.as_span().end()).into(),
        },
        [(Rule::ident, ident), (Rule::COLON, _), (Rule::struct_assign, expr)] => FieldInit {
            ident: ident.as_str().to_string(),
            init: parse_expr(expr.clone()),
            span: (ident.as_span().start()..expr.as_span().end()).into(),
        },
        _ => unreachable!("malformed struct fields {}", field.to_json()),
    }
}

fn to_span(p: &Pair<Rule>) -> Range {
    let sp = p.as_span();
    (sp.start()..sp.end()).into()
}
