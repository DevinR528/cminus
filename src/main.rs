#![feature(box_syntax, box_patterns, try_blocks, crate_visibility_modifier)]
#![allow(unused)]

use std::{
    env,
    fs::{self, DirEntry},
    path::Path,
};

use pest::{
    iterators::{Pair, Pairs},
    Parser as _,
};
use pest_derive::Parser;

mod ast;
mod precedence;

use ast::{parse::parse_decl, types::Expr};
use precedence::{Assoc, Operator, PrecClimber};

/// This is a procedural macro (fancy Rust macro) that expands the `grammar.pest` file
/// into a struct with a `CMinusParser::parse` method.
#[derive(Parser)]
#[grammar = "../grammar.pest"]
struct CMinusParser;

/// Driver function responsible for lexing and parsing input.
fn process_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let prog = fs::read_to_string(path)?;

    let file = match CMinusParser::parse(Rule::program, &prog) {
        Ok(mut parsed) => parsed.next().unwrap(), // doesn't fail
        Err(err) => return Err(err.to_string().into()),
    };

    // This is AST construction which is where operator precedence happens.
    // It does work correctly (see src/precedence.rs and src/ast/parse.rs:99:1 for more
    // details)
    //
    // for item in file.into_inner() {
    //     match item.as_rule() {
    //         Rule::decl => parse_decl(item),
    //         Rule::EOI => break,
    //         _ => unreachable!(),
    //     }
    // }

    Ok(())
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    println!("{:?}", args);
    match args.iter().map(|s| s.as_str()).collect::<Vec<_>>().as_slice() {
        [] => panic!("need to specify file to compile"),
        [a, rest @ ..] => {
            for f in rest {
                process_file(f).unwrap();
            }
        }
    };
}

#[test]
fn parse_all() {
    let mut dirs =
        fs::read_dir("./nb/input").unwrap().filter_map(|f| f.ok()).collect::<Vec<_>>();
    dirs.sort_by_key(|a| a.path());

    for f in dirs.into_iter() {
        let path = f.path();
        if path.is_file() && path.extension() == Some(Path::new("cm").as_os_str()) {
            let prog = fs::read_to_string(&path).unwrap();
            match CMinusParser::parse(Rule::program, &prog) {
                Ok(_) => {} // doesn't fail
                Err(err) => panic!("{}\n{:?}\nin file: {}", err, err, path.display()),
            };
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
        Ok(mut parsed) => parsed.next().unwrap(),
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
        Ok(mut parsed) => parsed.next().unwrap(),
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
        Ok(mut parsed) => parsed.next().unwrap(),
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
        Ok(mut parsed) => parsed.next().unwrap(),
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
        Ok(mut parsed) => parsed.next().unwrap(),
        Err(err) => panic!("{}", err),
    };

    println!("{}", file.to_json());
}
