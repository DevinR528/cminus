#![allow(unused)]

use std::{env, fs};

use pest::{Parser as _, iterators::{Pair, Pairs}};
use pest_derive::Parser;

mod precedence;

use precedence::{PrecClimber, Assoc, Operator};

#[derive(Parser)]
#[grammar = "../grammar.pest"]
struct CMinusParser;

fn parse_decl(pair: Pair<'_, Rule>) {
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
        Rule::assing => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
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
    println!("var = {}", var.to_json());
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

pub enum Val {
    Float(f64),
    Int(isize),
    Char(char),
    Str(String),
}

pub enum UnOp {
    Not,
    Inc,
}

pub enum BinOp {
    Not,
    Inc,
}

pub enum Expr {
    Urnary { op: UnOp, expr: Box<Expr> },
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Ident(String),
    Value(Val),
    Call { ident: String, args: Vec<Expr> },
}

fn parse_expr(mut expr: Pair<Rule>) -> Expr {
    println!("expr = {}", expr.to_json());

    let climber = PrecClimber::new(vec![
        Operator::new(Rule::PLUS, Assoc::Left) | Operator::new(Rule::MINUS, Assoc::Left),
        Operator::new(Rule::TIMES, Assoc::Left) | Operator::new(Rule::DIVIDE, Assoc::Left),
    ]);

    let ans = consume(expr, &climber);
    panic!("{} {}", ans.0, ans.1);

    Expr::Ident("foolio".to_string())
}

fn consume(pair: Pair<'_, Rule>, climber: &PrecClimber<Rule>) -> (String, i32) {
    let primary = |p: Pair<'_, _>| {
        consume(p, climber)
    };
    let infix = |lhs: (String, i32), op: Pair<Rule>, rhs: (String, i32)| match op.as_rule() {
        Rule::PLUS => (format!("({} + {})", lhs.0, rhs.0), lhs.1 + rhs.1),
        Rule::MINUS => (format!("({} - {})", lhs.0, rhs.0), lhs.1 - rhs.1),
        Rule::TIMES => (format!("({} * {})", lhs.0, rhs.0), lhs.1 * rhs.1),
        Rule::DIVIDE => (format!("({} / {})", lhs.0, rhs.0), lhs.1 / rhs.1),
        _ => unreachable!(),
    };


    match pair.as_rule() {
        Rule::expr => climber.climb(pair.into_inner(), primary, infix),
        Rule::op => climber.climb(pair.into_inner(), primary, infix),
        Rule::term => (pair.as_str().to_string(), pair.as_str().parse().unwrap()),
        _ => unreachable!(),
    }
}


fn main() {
    const VARS: &str = "
void foo() {
    y = 1 + 1 * 2;
}
";

    let args = env::args().collect::<Vec<_>>();

    match args.iter().map(|s| s.as_str()).collect::<Vec<_>>().as_slice() {
        [] => panic!("need to specify file to compile"),
        [a, rest @ ..] => {}
    };

    let prog = fs::read_to_string("./nb/input/arith.cm").unwrap();

    let file = match CMinusParser::parse(Rule::program, VARS) {
        Ok(mut parsed) => parsed.next().unwrap(), // doesn't fail
        Err(err) => panic!("{}\n{:?}", err, err),
    };

    for item in file.into_inner() {
        match item.as_rule() {
            Rule::decl => parse_decl(item),
            Rule::EOI => break,
            _ => unreachable!(),
        }
    }
}

#[test]
fn fn_with_float() {
    const VARS: &str = "
void foo() {
    x = 1.1;
}
";
    let file = match CMinusParser::parse(Rule::program, VARS) {
        Ok(mut parsed) => parsed.next().unwrap(), // doesn't fail
        Err(err) => panic!("{}", err),
    };

    println!("{:?}", file);
}

#[test]
fn var_decls() {
    const DECLS: &str = "
int x,y;
int a[15];
float vector[100];
";
    let file = match CMinusParser::parse(Rule::program, DECLS) {
        Ok(mut parsed) => parsed.next().unwrap(), // doesn't fail
        Err(err) => panic!("{}", err),
    };

    println!("{:?}", file);
}

#[test]
fn fn_decls() {
    const FNS: &str = "
int decls() {
  return 7;
}

float foo() {
  return 7.3;
}

void main() {
  write(decls());
  write(foo());
    exit;
}
";
    let file = match CMinusParser::parse(Rule::program, FNS) {
        Ok(mut parsed) => parsed.next().unwrap(), // doesn't fail
        Err(err) => panic!("{}\n{:?}", err, err),
    };

    println!("{:?}", file);
}

#[test]
fn precedence() {
    const ORD: &str = "
void foo() {
    x = 1 + 1 + 1;
    y = x == 3;
    z = (x > 1) || (y < 1);
}
";
    let file = match CMinusParser::parse(Rule::program, ORD) {
        Ok(mut parsed) => parsed.next().unwrap(), // doesn't fail
        Err(err) => panic!("{}", err),
    };

    println!("{}", file.to_json());
}

#[test]
fn a_buncho_precedence() {
    const ORDER: &str = "
void main () {
 int i,j,k,l,m;

    i=1; j=2; k=3; l=4;

    m=i<j;
    write(m);
    write(i == j);
    write(i == i);
    write(l>k);
    write(j>=j);
    write(k<=i);
    write(i!=j);
    write(!(l>k));
    write((i > j) || (l > k));
    write((j > i) && (k > l));
    write((i == j) || ((i<j)&&(k!=l)));

    exit;
}
";
    let file = match CMinusParser::parse(Rule::program, ORDER) {
        Ok(mut parsed) => parsed.next().unwrap(), // doesn't fail
        Err(err) => panic!("{}", err),
    };

    println!("{}", file.to_json());
}
