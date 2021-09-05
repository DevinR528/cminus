use pest::iterators::{Pair, Pairs};

use crate::{Rule, precedence::{Assoc, Operator, PrecClimber}, ast::types::{BinOp, Expr, UnOp, Val}};

crate fn parse_decl(pair: Pair<'_, Rule>) {
    for decl in pair.into_inner() {
        match decl.as_rule() {
            Rule::func_decl => parse_func(decl.into_inner()),
            Rule::var_decl => parse_var(decl.into_inner()),
            _ => unreachable!("malformed declaration"),
        }
    }
}

#[rustfmt::skip]
fn parse_func(func: Pairs<Rule>) {
    // println!("func = {}", func.to_json());
    match func.into_iter().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // int foo(int a, int b) { stmts }
        [(Rule::type_, ty), (Rule::ident, ident),
         (Rule::LP, _), (Rule::param_list, params), (Rule::RP, _), (Rule::LBR, _),
         action @ ..,
         (Rule::RBR, _)
        ] => {}
        // int foo() { stmts }
        [(Rule::type_, ty), (Rule::ident, ident),
         (Rule::LP, _), (Rule::RP, _), (Rule::LBR, _),
         action @ ..,
         (Rule::RBR, _)
        ] => {
            for (r, pair) in action {
                match r {
                    Rule::stmt => parse_stmt(pair.clone().into_inner()),
                    Rule::var_decl => parse_var(pair.clone().into_inner()),
                    _ => unreachable!("malformed code between braces")
                }
            }
        }
        _ => unreachable!("malformed function"),
    }
}

fn parse_stmt(mut stmts: Pairs<Rule>) {
    // println!("stmt = {}", stmts.to_json());
    let stmt = stmts.next().expect("statement has one child");
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
                    // println!("{} = {}", name.as_str(), expr.as_str());
                    let expr = parse_expr(expr.clone());
                }
                _ => unreachable!("malformed assingment"),
            }
        }
        Rule::call_stmt => {}
        Rule::if_stmt => {}
        Rule::while_stmt => {}
        Rule::io_stmt => {}
        Rule::ret_stmt => {}
        Rule::exit_stmt => {}
        Rule::cpd_stmt => {}
        _ => unreachable!("malformed statement"),
    }
    assert!(stmts.next().is_none(), "statement had 2 children");
}

fn parse_var(var: Pairs<Rule>) {
    match var.into_iter().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // int x,y,z;
        [(Rule::type_, ty), (Rule::var_name, var_name), names @ .., (Rule::SC, _)] => {
            println!(
                "{} named {}{}",
                ty.as_str(),
                var_name.as_str(),
                names
                    .iter()
                    .filter_map(|(r, n)| if matches!(r, Rule::var_name) {
                        Some(format!(", {}", n.as_str()))
                    } else {
                        None
                    })
                    .collect::<String>()
            )
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
    panic!("{} {:?}", ans.0, ans.1);

    Expr::Ident("foolio".to_string())
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
