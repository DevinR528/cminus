use pest::iterators::{Pair, Pairs};

use crate::{
    ast::types::{BinOp, Block, Decl, Expr, Func, Param, Stmt, Ty, UnOp, Val, Var},
    precedence::{Assoc, Operator, PrecClimber},
    Rule,
};

crate fn parse_decl(pair: Pair<'_, Rule>) -> Vec<Decl> {
    pair.into_inner()
        .flat_map(|decl| match decl.as_rule() {
            Rule::func_decl => vec![Decl::Func(parse_func(decl.into_inner()))],
            Rule::var_decl => parse_var(decl.into_inner()).into_iter().map(Decl::Var).collect(),
            _ => unreachable!("malformed declaration"),
        })
        .collect()
}

fn parse_param(param: Pair<Rule>) -> Param {
    match param.into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        [(Rule::type_, ty), (Rule::var_name, var)] => {
            Param { ty: parse_ty(ty.clone()), ident: var.as_str().to_string() }
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
         (Rule::RBR, _)
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
                    Rule::var_decl => Stmt::VarDecl(parse_var(s.clone().into_inner())),
                    _ => unreachable!("malformed statement"),
                }).collect()

            }
        }
        // int foo() { stmts }
        [(Rule::type_, ty), (Rule::ident, ident),
         (Rule::LP, _), (Rule::RP, _), (Rule::LBR, _),
         action @ ..,
         (Rule::RBR, _)
        ] => {
            Func {
                ret: parse_ty(ty.clone()),
                ident: ident.as_str().to_string(),
                params: vec![],
                stmts: action.iter().map(|(r, s)| match r {
                    Rule::stmt => parse_stmt(s.clone()),
                    Rule::var_decl => Stmt::VarDecl(parse_var(s.clone().into_inner())),
                    _ => unreachable!("malformed statement"),
                }).collect()

            }
        }
        _ => unreachable!("malformed function"),
    }
}

fn parse_block(blk: Pair<Rule>) -> Block {
    match blk.into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // { stmt* }
        [(Rule::LBR, _), stmts @ .., (Rule::RBR, _)] => {
            Block { stmts: stmts.iter().map(|(_, s)| parse_stmt(s.clone())).collect() }
        }
        _ => unreachable!("malformed function"),
    }
}

fn parse_stmt(mut stmt: Pair<Rule>) -> Stmt {
    let stmt = stmt.into_inner().next().unwrap();
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
                    Stmt::Assign {
                        ident: name.as_str().to_string(),
                        expr: parse_expr(expr.clone()),
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
                    }
                }
                // foo();
                [(Rule::ident, name), (Rule::LP, _), (Rule::RP, _), (Rule::SC, _)] => {
                    Stmt::Call { ident: name.as_str().to_string(), args: vec![] }
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
                    }
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
                    }
                }
                _ => unreachable!("malformed assingment"),
            }
        }
        Rule::io_stmt => {
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
                    Stmt::Write { expr: match arg_rule {
                        Rule::expr => parse_expr(arg.clone()),
                        Rule::string => Expr::Value(Val::Str(arg.as_str().replace("\"", ""))),
                        _ => unreachable!("malformed write statement")
                    } }
                }
                _ => unreachable!("malformed IO statement"),
            }
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
                    Stmt::Ret(parse_expr(expr.clone()))
                }
                _ => unreachable!("malformed return statement"),
            }
        }
        // exit;
        Rule::exit_stmt => Stmt::Exit,
        // { stmts }
        Rule::block_stmt => Stmt::Block(parse_block(stmt)),
        _ => unreachable!("malformed statement"),
    }
}

fn parse_ty(ty: Pair<Rule>) -> Ty {
    match ty.into_inner().next().unwrap().as_rule() {
        // int x,y,z;
        Rule::CHAR => Ty::Char,
        Rule::INT => Ty::Int,
        Rule::FLOAT => Ty::Float,
        Rule::VOID => Ty::Void,
        _ => unreachable!("malformed function"),
    }
}

fn parse_var(var: Pairs<Rule>) -> Vec<Var> {
    match var.into_iter().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // int x,y,z;
        [(Rule::type_, ty), (Rule::var_name, var_name), names @ .., (Rule::SC, _)] => {
            let ty = parse_ty(ty.clone());
            vec![Var { ty: ty.clone(), ident: var_name.as_str().to_string() }]
                .into_iter()
                .chain(names.iter().filter_map(|(r, n)| {
                    if matches!(r, Rule::var_name) {
                        Some(Var { ty: ty.clone(), ident: n.as_str().to_string() })
                    } else {
                        None
                    }
                }))
                .collect()
        }
        _ => unreachable!("malformed function"),
    }
}

fn parse_expr(mut expr: Pair<Rule>) -> Expr {
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

fn consume(expr: Pair<'_, Rule>, climber: &PrecClimber<Rule>) -> Expr {
    let primary = |p: Pair<'_, _>| consume(p, climber);
    let infix = |lhs: Expr, op: Pair<Rule>, rhs: Expr| match op.as_rule() {
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

    match expr.as_rule() {
        Rule::expr => climber.climb(expr.into_inner(), primary, infix),
        Rule::op => climber.climb(expr.into_inner(), primary, infix),
        Rule::term => {
            match expr.into_inner().map(|r| (r.as_rule(), r)).collect::<Vec<_>>().as_slice() {
                // x,y,z
                [(Rule::variable, var)] => Expr::Ident(var.as_str().to_string()),
                // 1,true,"a"
                [(Rule::const_, konst)] => {
                    match konst.clone().into_inner().next().unwrap().as_rule() {
                        Rule::integer => Expr::Value(Val::Int(konst.as_str().parse().unwrap())),
                        Rule::decimal => Expr::Value(Val::Float(konst.as_str().parse().unwrap())),
                        Rule::charstr => Expr::Value(Val::Str(konst.as_str().to_string())),
                        r => unreachable!("malformed const expression {:?}", r),
                    }
                }
                // call()
                [(Rule::ident, ident), (Rule::LP, _), arg_list @ .., (Rule::RP, _)] => Expr::Call {
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
                },
                // !true
                [(Rule::NOT, _), (Rule::expr, expr)] => {
                    let inner = climber.climb(expr.clone().into_inner(), primary, infix);
                    Expr::Urnary { op: UnOp::Not, expr: box inner }
                }
                // (1 + 1)
                [(Rule::LP, _), (Rule::expr, expr), (Rule::RP, _)] => {
                    let inner = climber.climb(expr.clone().into_inner(), primary, infix);
                    Expr::Parens(box inner)
                }
                _ => unreachable!("malformed expression"),
            }
        }
        _ => unreachable!(),
    }
}
