// fun rust features I'm turning on
#![feature(
    box_syntax,
    box_patterns,
    try_blocks,
    crate_visibility_modifier,
    stmt_expr_attributes,
    btree_drain_filter,
    panic_info_message,
    path_file_prefix
)]
// TODO: remove
// tell rust not to complain about unused anything
#![allow(unused)]

use std::{
    cell::Cell,
    env,
    fs::{self, DirEntry},
    path::Path,
    sync::{
        atomic::{AtomicU8, Ordering},
        Arc, Mutex,
    },
};

use pest::Parser as _;
use pest_derive::Parser;

mod ast;
mod error;
mod lir;
mod typeck;
mod visit;

use ast::parse::parse_decl;

use crate::{
    lir::visit::Visit as IrVisit,
    visit::{Visit, VisitMut},
};

/// This is a procedural macro (fancy Rust macro) that expands the `grammar.pest` file
/// into a struct with a `CMinusParser::parse` method.
#[derive(Parser)]
#[grammar = "../grammar.pest"]
struct CMinusParser;

/// Driver function responsible for lexing and parsing input.
fn process_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Read the file to string
    let prog = fs::read_to_string(path)?;

    // Using the generated parser from `grammar.pest` lex/parse the input
    let file = match CMinusParser::parse(Rule::program, &prog) {
        // Parsing passed
        Ok(mut parsed) => {
            parsed.next().expect("CMinusParser will always have a parse tree if parsing succeeded")
        }
        // Parsing has failed prints error like
        Err(err) => {
            println!("{:?}", err);
            return Err(err.to_string().into());
        }
    };

    // println!("{}", file.to_json());

    // This is AST construction which is where operator precedence happens.
    // It does work correctly (see src/precedence.rs and src/ast/parse.rs (parse_expr) for more
    // details)
    //
    let mut items = vec![];
    for item in file.into_inner() {
        match item.as_rule() {
            Rule::decl => items.extend(parse_decl(item)),
            Rule::EOI => break,
            _ => unreachable!(),
        }
    }

    // println!("{:?}", items);

    let mut tyck = typeck::TyCheckRes::new(&prog, path);

    tyck.visit_prog(&items);
    let res = tyck.report_errors()?;
    // res.unwrap();

    println!("{:#?}", tyck);

    let lowered = lir::lower::lower_items(&items, tyck);

    // println!("\n\n{:?}", lowered);

    // let ctxt = inkwell::context::Context::create();
    // let mut gen = lir::llvmgen::LLVMGen::new(&ctxt, Path::new(path));

    // let mut gen = lir::asmgen::CodeGen::new(Path::new(path));
    // gen.visit_prog(&lowered);
    // gen.dump_asm();

    Ok(())
}

/// Run it!
fn main() {
    // Get arguments this is c's argv
    let args = env::args().collect::<Vec<_>>();

    std::panic::set_hook(Box::new(|panic_info| {
        let _: Option<()> = try {
            let msg = format!("{}", panic_info.message()?);

            if msg.contains("ICE") {
                eprintln!("{}", msg);
                let loc = panic_info.location()?;
                eprintln!("ICE location: {}", loc);
            } else if msg.contains("not yet implemented") {
                eprintln!("ICE needs implementation (undone TODO item)");

                eprintln!("{}", msg);

                let loc = panic_info.location()?;
                eprintln!("ICE location: {}", loc);
            } else {
                eprintln!("{}", msg);
            }
        };
        let _: Option<()> = try {
            let payload = panic_info.payload().downcast_ref::<&str>()?;
            eprintln!("`{}`", payload);
        };
    }));
    std::panic::take_hook();

    match args.iter().map(|s| s.as_str()).collect::<Vec<_>>().as_slice() {
        [] => panic!("need to specify file to compile"),
        // ignore binary name and process all file names passed
        [_bin_name, file_names @ ..] => {
            let mut errors = 0;
            for f in file_names {
                // match std::panic::catch_unwind(|| process_file(f)) {
                //     Ok(Ok(_)) => {}
                //     Ok(Err(e)) => {
                //         errors += 1;
                //         // eprintln!("{}", e)
                //     }
                //     Err(e) => {
                //         errors += 1;
                //         if let Some(e) = e.downcast_ref::<&str>() {
                //             eprintln!("{}", e);
                //         }
                //     }
                // }
                match process_file(f) {
                    Ok(_) => {}
                    Err(e) => {
                        errors += 1;
                        eprintln!("{}", e);
                    }
                }
            }

            if errors != 0 {
                eprintln!(
                    "compilation stopped found {} error{}",
                    errors,
                    if errors > 1 { "s" } else { "" }
                );
                std::process::exit(1)
            }
        }
    }
}

#[test]
#[ignore = "file_system"]
fn parse_all() {
    let mut dirs = fs::read_dir("./input").unwrap().filter_map(|f| f.ok()).collect::<Vec<_>>();
    dirs.sort_by_key(|a| a.path());

    for f in dirs.into_iter() {
        let path = f.path();
        if path.is_file() && path.extension() == Some(Path::new("cm").as_os_str()) {
            println!("{}", path.display());
            process_file(&path.as_os_str().to_string_lossy()).unwrap();
        }
    }
}

#[test]
#[ignore = "file_system"]
fn open_all() {
    let mut dirs = fs::read_dir(".").unwrap().filter_map(|f| f.ok()).collect::<Vec<_>>();
    dirs.sort_by_key(|a| a.path());

    for f in dirs.into_iter() {
        let path = f.path();
        if path.is_file() && path.extension() == Some(Path::new("pdf").as_os_str()) {
            println!("{}", path.display());
            std::process::Command::new("firefox")
                .arg(path.as_os_str())
                .status()
                .expect("failed to execute `dot -Tpdf ...`");
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
