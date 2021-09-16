use std::ops::Range;

use pest::iterators::{Pair, Pairs};

use crate::{
    ast::types::{
        BinOp, Block, Decl, Declaration, Expr, Expression, Field, FieldInit, Func, Param,
        Statement, Stmt, Struct, Ty, Type, UnOp, Val, Value, Var,
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
                Rule::var_decl => parse_var(decl.into_inner())
                    .into_iter()
                    .map(|var| Decl::Var(var).into_spanned(span.clone()))
                    .collect(),
                Rule::adt_decl => {
                    vec![Decl::Adt(parse_struct(decl.into_inner(), span.clone())).into_spanned(span)]
                }
                _ => unreachable!("malformed declaration"),
            }
        })
        .collect()
}

fn parse_struct(struct_: Pairs<Rule>, span: Range<usize>) -> Struct {
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
            let span = to_span(var);
            Field { ident: var.as_str().to_string(), ty: parse_ty(ty.clone()), span }
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
                        parse_var(s.clone().into_inner())).into_spanned(to_span(s)
                    ),
                    _ => unreachable!("malformed statement"),
                }).collect(),
                span: ty.as_span().start()..rbr.as_span().end(),
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
                        parse_var(s.clone().into_inner())
                    ).into_spanned(to_span(s)),
                    _ => unreachable!("malformed statement"),
                }).collect(),
                span: ty.as_span().start()..rbr.as_span().end(),
            }
        }
        _ => unreachable!("malformed function"),
    }
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

fn parse_stmt(mut stmt: Pair<Rule>) -> Statement {
    let stmt = stmt.into_inner().next().unwrap();
    let span = to_span(&stmt);
    #[rustfmt::skip]
    match stmt.as_rule() {
        Rule::assing => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // var = expr;
                [(Rule::variable, name), (Rule::ASSIGN, _),
                    (Rule::expr, expr), (Rule::SC, _)
                ] => {
                    match name.clone().into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
                        [(Rule::ident, ident)] => {
                            Stmt::Assign {
                                ident: name.as_str().to_string(),
                                expr: parse_expr(expr.clone()),
                            }.into_spanned(span)
                        }
                        [(Rule::ident, ident), (Rule::LBK, _), (Rule::expr, expr), (Rule::RBK, _)] => {
                            Stmt::ArrayAssign {
                                ident: ident.as_str().to_string(),
                                expr: parse_expr(expr.clone()),
                            }.into_spanned(span)
                        }
                        _ => unreachable!("malformed variable name")
                    }
                }
                // var = { field: expr, optional: expr };
                [(Rule::variable, name), (Rule::ASSIGN, _),
                    (Rule::struct_assign, expr), (Rule::SC, _)
                ] => {
                    match name.clone().into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
                        [(Rule::ident, ident)] => {
                            Stmt::Assign {
                                ident: name.as_str().to_string(),
                                expr: parse_expr(expr.clone()),
                            }.into_spanned(span)
                        }
                        _ => unreachable!("malformed variable name")
                    }
                }
                _ => unreachable!("malformed assingment"),
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
    match ty.into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // int x,y,z;
        [(Rule::CHAR, _)] => Ty::Char,
        [(Rule::INT, _)] => Ty::Int,
        [(Rule::FLOAT, _)] => Ty::Float,
        [(Rule::VOID, _)] => Ty::Void,
        [(Rule::STRUCT, _), (Rule::ident, ident)] => Ty::Adt(ident.as_str().to_string()),
        _ => unreachable!("malformed function"),
    }
    .into_spanned(span)
}

fn build_recursive_ty<I: Iterator<Item = usize>>(mut dims: I, base_ty: Type) -> Type {
    if let Some(size) = dims.next() {
        let span = base_ty.span.clone();
        Ty::Array { size, ty: box build_recursive_ty(dims, base_ty) }.into_spanned(span)
    } else {
        base_ty
    }
}

fn parse_var(var: Pairs<Rule>) -> Vec<Var> {
    match var.into_iter().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // int x,y,z;
        [(Rule::type_, ty), (Rule::var_name, var_name), names @ .., (Rule::SC, _)] => {
            let ty = parse_ty(ty.clone());
            let parse_var_array = |var: &Pair<Rule>| {
                let span = to_span(var);
                if var.clone().into_inner().any(|p| matches!(p.as_rule(), Rule::LBK)) {
                    let mut id = None;
                    let mut dim: Vec<usize> = vec![];
                    for p in var.clone().into_inner() {
                        match p.as_rule() {
                            Rule::ident => id = Some(p.as_str().to_string()),
                            Rule::integer => dim.push(p.as_str().parse().unwrap()),
                            Rule::LBK | Rule::RBK => {}
                            _ => unreachable!("malformed array variable declaration {:?}", p),
                        }
                    }

                    Var {
                        ty: build_recursive_ty(dim.into_iter(), ty.clone()),
                        ident: id.unwrap(),
                        span,
                    }
                } else {
                    Var { ty: ty.clone(), ident: var.as_str().to_string(), span }
                }
            };

            vec![parse_var_array(var_name)]
                .into_iter()
                .chain(names.iter().filter_map(|(r, n)| {
                    if matches!(r, Rule::var_name) {
                        Some(parse_var_array(n))
                    } else {
                        None
                    }
                }))
                .collect()
        }
        _ => unreachable!("malformed function"),
    }
}

fn parse_expr(mut expr: Pair<Rule>) -> Expression {
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
    ]);

    consume(expr, &climber)
}

fn consume(expr: Pair<'_, Rule>, climber: &PrecClimber<Rule>) -> Expression {
    let primary = |p: Pair<'_, _>| consume(p, climber);
    let infix = |lhs: Expression, op: Pair<Rule>, rhs: Expression| {
        let span = lhs.span.start..rhs.span.end;
        let expr = match op.as_rule() {
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
            Rule::AND => Expr::Binary { op: BinOp::Add, lhs: box lhs, rhs: box rhs },
            Rule::OR => Expr::Binary { op: BinOp::Or, lhs: box lhs, rhs: box rhs },
            _ => unreachable!(),
        };
        expr.into_spanned(span)
    };

    match expr.as_rule() {
        Rule::expr => climber.climb(expr.into_inner(), primary, infix),
        Rule::op => climber.climb(expr.into_inner(), primary, infix),
        Rule::term => {
            let span = to_span(&expr);
            match expr.into_inner().map(|r| (r.as_rule(), r)).collect::<Vec<_>>().as_slice() {
                // x,y,z
                [(Rule::variable, var)] => {
                    let span = to_span(var);
                    match var
                        .clone()
                        .into_inner()
                        .map(|p| (p.as_rule(), p))
                        .collect::<Vec<_>>()
                        .as_slice()
                    {
                        [(Rule::ident, ident)] => Expr::Ident(ident.as_str().to_string()),
                        [(Rule::ident, ident), (Rule::LBK, _), (Rule::expr, expr), (Rule::RBK, _)] => {
                            Expr::Array {
                                ident: ident.as_str().to_string(),
                                expr: box parse_expr(expr.clone()),
                            }
                        }
                        _ => unreachable!("malformed variable name"),
                    }.into_spanned(span)
                }
                // 1,true,"a"
                [(Rule::const_, konst)] => {
                    let inner_span = to_span(konst);
                    match konst.clone().into_inner().next().unwrap().as_rule() {
                        Rule::integer => Expr::Value(
                            Val::Int(konst.as_str().parse().unwrap()).into_spanned(inner_span),
                        ),
                        Rule::decimal => Expr::Value(
                            Val::Float(konst.as_str().parse().unwrap()).into_spanned(inner_span),
                        ),
                        Rule::charstr => Expr::Value(
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
                [(Rule::LBR, _), fields @ .., (Rule::RBR, _)] => Expr::StructInit(
                    fields
                        .iter()
                        .filter_map(|(r, p)| {
                            if matches!(r, Rule::field_expr) {
                                Some(parse_field_init(p.clone()))
                            } else {
                                None
                            }
                        })
                        .collect(),
                )
                .into_spanned(span),
                _ => unreachable!("malformed struct assignment"),
            }
        }
        err => unreachable!("{:?}", err),
    }
}

fn parse_field_init(field: Pair<Rule>) -> FieldInit {
    match field.into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        [(Rule::ident, ident), (Rule::COLON, _), (Rule::expr, expr)] => FieldInit {
            ident: ident.as_str().to_string(),
            init: parse_expr(expr.clone()),
            span: ident.as_span().start()..expr.as_span().end(),
        },
        _ => unreachable!("malformed struct fields"),
    }
}

fn to_span(p: &Pair<Rule>) -> Range<usize> {
    let sp = p.as_span();
    sp.start()..sp.end()
}
