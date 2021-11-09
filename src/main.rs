// fun rust features I'm turning on
#![feature(
    box_syntax,
    box_patterns,
    try_blocks,
    crate_visibility_modifier,
    stmt_expr_attributes,
    btree_drain_filter,
    panic_info_message,
    path_file_prefix,
    io_error_more,
    const_fn_trait_bound
)]
// TODO: remove
// tell rust not to complain about unused anything
#![allow(clippy::if_then_panic, unused)]

use std::{
    alloc::System,
    env,
    fs::{self},
    path::Path,
    time::Instant,
};

use ::pest::Parser as _;

mod alloc;
mod ast;
mod error;
mod lir;
mod typeck;
mod visit;

use crate::{
    alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM},
    ast::parse::{parse_decl, CMinusParser, Rule},
    lir::visit::Visit as IrVisit,
    visit::Visit,
};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// Driver function responsible for lexing and parsing input.
fn process_file(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Read the file to string
    let prog = fs::read_to_string(path)?;

    let mut parse_mem = Region::new(GLOBAL);
    let parse_time = Instant::now();
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
    println!("    lexing & parsing:  {}s", parse_time.elapsed().as_secs_f64());
    println!("    lexing & parsing:  {}", parse_mem.change_and_reset());

    // println!("{:?}", items);

    let tyck_time = Instant::now();
    let mut tyck = typeck::TyCheckRes::new(&prog, path);
    tyck.visit_prog(&items);
    let _res = tyck.report_errors()?;
    println!("    type checking:     {}s", tyck_time.elapsed().as_secs_f64());
    // res.unwrap();

    // println!("{:#?}", tyck);

    let lower_time = Instant::now();
    let lowered = lir::lower::lower_items(&items, tyck);
    println!("    lowering:          {}s", lower_time.elapsed().as_secs_f64());

    // println!("{:?}", lowered);

    // let ctxt = inkwell::context::Context::create();
    // let mut gen = lir::llvmgen::LLVMGen::new(&ctxt, Path::new(path));

    let gen_time = Instant::now();
    let mut gen = lir::asmgen::CodeGen::new(Path::new(path));
    gen.visit_prog(&lowered);
    gen.dump_asm()?;
    println!("    code generation:   {}s", gen_time.elapsed().as_secs_f64());

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
    let _ = std::panic::take_hook();

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

#[allow(dead_code)]
fn dump_lowered_items(lowered: Vec<lir::lower::Item>) {
    for item in lowered {
        match item {
            lir::lower::Item::Adt(lir::lower::Adt::Struct(adt)) => {
                println!(
                    "struct {}<{}>",
                    adt.ident,
                    adt.generics.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "),
                );
            }
            lir::lower::Item::Adt(lir::lower::Adt::Enum(adt)) => {
                println!(
                    "enum {}<{}>",
                    adt.ident,
                    adt.generics.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "),
                );
            }
            lir::lower::Item::Func(f) => {
                println!(
                    "fn {}<{}>({}) -> {}",
                    f.ident,
                    f.generics.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "),
                    f.params.iter().map(|p| p.ty.to_string()).collect::<Vec<_>>().join(", "),
                    f.ret,
                );
            }
            lir::lower::Item::Trait(t) => {
                println!(
                    "trait {}<{}>({}) -> {}",
                    t.ident,
                    t.generics.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "),
                    t.method
                        .function()
                        .params
                        .iter()
                        .map(|p| p.ty.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    t.method.function().ret,
                );
            }
            lir::lower::Item::Impl(i) => {
                println!(
                    "impl {}<{}>({}) -> {}",
                    i.ident,
                    i.type_arguments.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "),
                    i.method.params.iter().map(|p| p.ty.to_string()).collect::<Vec<_>>().join(", "),
                    i.method.ret,
                );
            }
            lir::lower::Item::Var(_v) => todo!(),
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
