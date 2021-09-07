use pest::iterators::{Pair, Pairs};

use crate::{Rule, precedence::{Assoc, Operator, PrecClimber}, ast::types::{BinOp, Expr, UnOp, Val, Decl, Func, Ty, Var, Stmt, Param, Block}};

crate fn parse_decl(pair: Pair<'_, Rule>) -> Vec<Decl> {
    pair.into_inner().flat_map(|decl| {
        match decl.as_rule() {
            Rule::func_decl => vec![Decl::Func(parse_func(decl.into_inner()))],
            Rule::var_decl => parse_var(decl.into_inner()).into_iter().map(Decl::Var).collect(),
            _ => unreachable!("malformed declaration"),
        }
    }).collect()
}

fn parse_param(param: Pair<Rule>) -> Param {
    todo!("{}", param.to_json())
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
                params: params.clone().into_inner().map(parse_param).collect(),
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

fn parse_block(var: Pair<Rule>) -> Block {
    match var.into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // { stmt* }
        [(Rule::LBR, _), stmts @ .., (Rule::RBR, _)] => {
            Block { stmts: stmts.iter().map(|(_, s)| parse_stmt(s.clone())).collect() }
        }
        _ => unreachable!("malformed function"),
    }
}

fn parse_stmt(mut stmt: Pair<Rule>) -> Stmt {
    let stmt = stmt.into_inner().next().unwrap();
    match stmt.as_rule() {
        #[rustfmt::skip]
        Rule::assing => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {

                // var = expr;
                [(Rule::variable, name), (Rule::ASSIGN, _), (Rule::expr, expr), (Rule::SC, _)] =>
                {
                    Stmt::Assign { ident: name.as_str().to_string(), expr: parse_expr(expr.clone()) }
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
                [(Rule::ident, name), (Rule::LP, _), (Rule::arg_list, args), (Rule::RP, _), (Rule::SC, _)] =>
                {
                    Stmt::Call { ident: name.as_str().to_string(), args: args.clone().into_inner().filter_map(|arg| {
                        match arg.as_rule() {
                            Rule::expr => Some(parse_expr(arg)),
                            Rule::CM => None,
                            _ => unreachable!("malformed call statement")
                        }
                    }).collect() }
                }
                // foo();
                [(Rule::ident, name), (Rule::LP, _), (Rule::RP, _), (Rule::SC, _)] =>
                {
                    Stmt::Call { ident: name.as_str().to_string(), args: vec![] }
                }
                _ => unreachable!("malformed assingment"),
            }
        },
        Rule::if_stmt => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {

                // if expr { stmts } [ else { stmts }]
                [(Rule::IF, _), (Rule::LP, _), (Rule::expr, expr), (Rule::RP, _), (Rule::block_stmt, block), else_blk @ ..] =>
                {
                    Stmt::If { cond: parse_expr(expr.clone()), blk: parse_block(block.clone()), els: match else_blk {
                        [(Rule::ELSE, _), (Rule::block_stmt, blk)] => Some(parse_block(blk.clone())),
                        [] => None,
                        _ => unreachable!("malformed if statement")
                    }}
                }
                _ => unreachable!("malformed assingment"),
            }
        },
        Rule::while_stmt => todo!("{}", stmt.to_json()),
        Rule::io_stmt => todo!("{}", stmt.to_json()),
        Rule::ret_stmt => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {

                // foo(x,y);
                [(Rule::RETURN, _), (Rule::expr, expr), (Rule::SC, _)] =>
                {
                    Stmt::Ret(parse_expr(expr.clone()))
                }
                _ => unreachable!("malformed return statement"),
            }
        },
        Rule::exit_stmt => Stmt::Exit,
        Rule::block_stmt => todo!("{}", stmt.to_json()),
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
            vec![Var {ty: ty.clone(), ident: var_name.as_str().to_string() }].into_iter().chain(
                names.iter().filter_map(|(r, n)| if matches!(r, Rule::var_name) {
                Some(Var { ty: ty.clone(), ident: n.as_str().to_string() })
            } else {
                None
            })).collect()
        }
        _ => unreachable!("malformed function"),
    }
}

fn parse_expr(mut expr: Pair<Rule>) -> Expr {
    // println!("expr = {}", expr.to_json());

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
        Operator::new(Rule::MUL, Assoc::Left)
            | Operator::new(Rule::DIV, Assoc::Left),
    ]);

    let ans = consume(expr, &climber);
    ans.1
}

fn consume(expr: Pair<'_, Rule>, climber: &PrecClimber<Rule>) -> (String, Expr) {
    let primary = |p: Pair<'_, _>| consume(p, climber);
    let infix = |lhs: (String, Expr), op: Pair<Rule>, rhs: (String, Expr)| match op
        .as_rule()
    {
        Rule::PLUS => (
            format!("({} + {})", lhs.0, rhs.0),
            Expr::Binary { op: BinOp::Add, lhs: Box::new(lhs.1), rhs: Box::new(rhs.1) },
        ),
        Rule::SUB => (
            format!("({} - {})", lhs.0, rhs.0),
            Expr::Binary { op: BinOp::Sub, lhs: Box::new(lhs.1), rhs: Box::new(rhs.1) },
        ),
        Rule::MUL => (
            format!("({} * {})", lhs.0, rhs.0),
            Expr::Binary { op: BinOp::Mul, lhs: Box::new(lhs.1), rhs: Box::new(rhs.1) },
        ),
        Rule::DIV => (
            format!("({} / {})", lhs.0, rhs.0),
            Expr::Binary { op: BinOp::Div, lhs: Box::new(lhs.1), rhs: Box::new(rhs.1) },
        ),
        Rule::GT => (
            format!("({} > {})", lhs.0, rhs.0),
            Expr::Binary { op: BinOp::Gt, lhs: Box::new(lhs.1), rhs: Box::new(rhs.1) },
        ),
        Rule::GE => (
            format!("({} >= {})", lhs.0, rhs.0),
            Expr::Binary { op: BinOp::Ge, lhs: Box::new(lhs.1), rhs: Box::new(rhs.1) },
        ),
        Rule::LT => (
            format!("({} < {})", lhs.0, rhs.0),
            Expr::Binary { op: BinOp::Lt, lhs: Box::new(lhs.1), rhs: Box::new(rhs.1) },
        ),
        Rule::LE => (
            format!("({} <= {})", lhs.0, rhs.0),
            Expr::Binary { op: BinOp::Le, lhs: Box::new(lhs.1), rhs: Box::new(rhs.1) },
        ),
        Rule::EQ => (
            format!("({} == {})", lhs.0, rhs.0),
            Expr::Binary { op: BinOp::Eq, lhs: Box::new(lhs.1), rhs: Box::new(rhs.1) },
        ),
        Rule::NE => (
            format!("({} != {})", lhs.0, rhs.0),
            Expr::Binary { op: BinOp::Ne, lhs: Box::new(lhs.1), rhs: Box::new(rhs.1) },
        ),
        Rule::AND => (
            format!("({} && {})", lhs.0, rhs.0),
            Expr::Binary { op: BinOp::Add, lhs: Box::new(lhs.1), rhs: Box::new(rhs.1) },
        ),
        Rule::OR => (
            format!("({} || {})", lhs.0, rhs.0),
            Expr::Binary { op: BinOp::Or, lhs: Box::new(lhs.1), rhs: Box::new(rhs.1) },
        ),
        _ => unreachable!(),
    };

    match expr.as_rule() {
        Rule::expr => climber.climb(expr.into_inner(), primary, infix),
        Rule::op => climber.climb(expr.into_inner(), primary, infix),
        Rule::term => {
            match expr
                .into_inner()
                .map(|r| (r.as_rule(), r))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // x,y,z
                [(Rule::variable, var)] => {
                    (var.as_str().to_owned(), Expr::Ident(var.as_str().to_string()))
                }
                // 1,true,"a"
                [(Rule::const_, konst)] => match konst.clone().into_inner().next().unwrap().as_rule() {
                    Rule::integer => (
                        konst.as_str().to_owned(),
                        Expr::Value(Val::Int(konst.as_str().parse().unwrap())),
                    ),
                    Rule::decimal => (
                        konst.as_str().to_owned(),
                        Expr::Value(Val::Float(konst.as_str().parse().unwrap())),
                    ),
                    Rule::charstr => (
                        konst.as_str().to_owned(),
                        Expr::Value(Val::Str(konst.as_str().to_string())),
                    ),
                    r => unreachable!("malformed const expression {:?}", r),
                },
                // call()
                [(Rule::ident, ident), (Rule::LP, _), arg_list @ .., (Rule::RP, _)] => (
                    format!(
                        "{}({})",
                        ident.as_str(),
                        arg_list.iter().map(|(_, arg)| arg.as_str()).collect::<String>()
                    ),
                    Expr::Call {
                        ident: ident.as_str().to_string(),
                        args: arg_list
                            .iter()
                            .map(|(r, p)| climber.climb(p.clone().into_inner(), primary, infix).1)
                            .collect(),
                    },
                ),
                // !true
                [(Rule::NOT, _), (Rule::expr, expr)] => {
                    let inner = climber.climb(expr.clone().into_inner(), primary, infix);
                    (format!("!{}", inner.0), Expr::Urnary { op: UnOp::Not, expr: box inner.1 })
                }
                // (1 + 1)
                [(Rule::LP, _), (Rule::expr, expr), (Rule::RP, _)] => {
                    let inner = climber.climb(expr.clone().into_inner(), primary, infix);
                    (format!("({})", inner.0), Expr::Parens(box inner.1))
                }
                _ => unreachable!("malformed expression"),
            }
        }
        _ => unreachable!(),
    }
}
