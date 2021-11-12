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
    const_fn_trait_bound,
    hash_drain_filter
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
    ast::parse::AstBuilder,
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

    let mut parser = AstBuilder::new(&prog);
    parser.parse()?;
    let mut items = parser.into_items();

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
                    adt.generics
                        .iter()
                        .map(|t| format!(
                            "{}: {}",
                            t.ident,
                            t.bound.as_ref().map(|p| p.to_string()).unwrap_or("".into())
                        ))
                        .collect::<Vec<_>>()
                        .join(", "),
                );
            }
            lir::lower::Item::Adt(lir::lower::Adt::Enum(adt)) => {
                println!(
                    "enum {}<{}>",
                    adt.ident,
                    adt.generics
                        .iter()
                        .map(|t| format!(
                            "{}: {}",
                            t.ident,
                            t.bound.as_ref().map(|p| p.to_string()).unwrap_or("".into())
                        ))
                        .collect::<Vec<_>>()
                        .join(", "),
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
                    i.type_arguments.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(
                        ",
"
                    ),
                    i.method.params.iter().map(|p| p.ty.to_string()).collect::<Vec<_>>().join(", "),
                    i.method.ret,
                );
            }
            lir::lower::Item::Var(_v) => todo!(),
        }
    }
}
