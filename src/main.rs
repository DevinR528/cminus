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

use clap::{App, Arg, ArgMatches};

mod alloc;
mod ast;
mod error;
mod lir;
mod typeck;
mod visit;

use crate::{
    alloc::{Region, StatsAlloc, INSTRUMENTED_SYSTEM},
    ast::parse::{error::PrettyError, AstBuilder},
    lir::visit::Visit as IrVisit,
    visit::Visit,
};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

/// Driver function responsible for lexing and parsing input.
fn process_file<'a>(
    path: &str,
    args: &ArgMatches<'a>,
) -> Result<(), Box<dyn std::error::Error + 'a>> {
    let need_stats = args.is_present("stats");
    let verbose = args.is_present("verbose");
    let backend = args.value_of("backend");
    let assemble = args.is_present("assemble");
    let output = args.value_of("output");

    let input = std::fs::read_to_string(path)?;
    // Read the file to string

    let mut parse_mem = Region::new(GLOBAL);
    let parse_time = Instant::now();

    let (snd, rcv) = std::sync::mpsc::channel();
    let mut parser = AstBuilder::new(&input, path, snd);
    parser.parse().map_err(|e| PrettyError::from_parse(path, &input, e))?;
    let mut items = parser.into_items();

    if need_stats {
        println!("    lexing & parsing:  {}s", parse_time.elapsed().as_secs_f64());
        println!("    lexing & parsing:  {}", parse_mem.change_and_reset());
        if verbose {
            println!("{:?}", items);
        }
    }

    let mut tyck_mem = Region::new(GLOBAL);
    let tyck_time = Instant::now();
    let mut tyck = typeck::TyCheckRes::new(&input, path, rcv);
    tyck.visit_prog(&items);
    let _res = tyck.report_errors()?;

    if need_stats {
        println!("    type checking:     {}s", tyck_time.elapsed().as_secs_f64());
        println!("    type checking:     {}", tyck_mem.change_and_reset());

        if verbose {
            println!("{:#?}", tyck);
        }
    }

    let mut lower_mem = Region::new(GLOBAL);
    let lower_time = Instant::now();
    let lowered = lir::lower::lower_items(&items, tyck);

    if need_stats {
        println!("    lowering:          {}s", lower_time.elapsed().as_secs_f64());
        println!("    lowering:          {}", lower_mem.change_and_reset());

        if verbose {
            println!("{:?}", lowered);
        }
    }

    if !assemble {
        return Ok(());
    }

    if backend == Some("llvm") {
        // let ctxt = inkwell::context::Context::create();
        // let mut gen = lir::llvmgen::LLVMGen::new(&ctxt, Path::new(path));
    }

    let out = if let Some(out) = output { Path::new(out) } else { Path::new(path) };

    let mut gen_mem = Region::new(GLOBAL);
    let gen_time = Instant::now();
    let mut gen = lir::asmgen::CodeGen::new(out);
    gen.visit_prog(&lowered);
    gen.dump_asm()?;

    if need_stats {
        println!("    code generation:   {}s", gen_time.elapsed().as_secs_f64());
        println!("    code generation:   {}", gen_mem.change_and_reset());
    }
    Ok(())
}

/// Run it!
#[tokio::main]
fn main() {
    // std::panic::set_hook(Box::new(|panic_info| {
    //     let _: Option<()> = try {
    //         let msg = format!("{}", panic_info.message()?);

    //         if msg.contains("ICE") {
    //             eprintln!("{}", msg);
    //             let loc = panic_info.location()?;
    //             eprintln!("ICE location: {}", loc);
    //         } else if msg.contains("not yet implemented") {
    //             eprintln!("ICE needs implementation (undone TODO item)");

    //             eprintln!("{}", msg);

    //             let loc = panic_info.location()?;
    //             eprintln!("ICE location: {}", loc);
    //         } else {
    //             eprintln!("{}", msg);
    //         }
    //     };
    //     let _: Option<()> = try {
    //         let payload = panic_info.payload().downcast_ref::<&str>()?;
    //         eprintln!("`{}`", payload);
    //     };
    // }));
    // let _ = std::panic::take_hook();

    let app = App::new("My Program")
        .author(clap::crate_authors!("\n"))
        .version(clap::crate_version!())
        .arg(
            Arg::with_name("input")
                .value_name("INPUT")
                .long("input")
                .short("i")
                .multiple(true)
                .help("sets the files that should be compiled"),
        )
        .arg(Arg::with_name("stats").long("stats").short("s").help("compile with stats"))
        .arg(
            Arg::with_name("verbose")
                .long("verbose")
                .short("v")
                .help("compile with verbose printing of IR"),
        )
        .arg(
            Arg::with_name("backend")
                .long("backend")
                .short("b")
                .possible_values(&["llvm", "hasm"])
                .default_value("hasm")
                .help("specify the backend (llvm or hand-rolled asm)"),
        )
        .arg(
            Arg::with_name("assemble")
                .long("assemble")
                .short("a")
                .help("enumc will produce assembly output"),
        )
        .arg(
            Arg::with_name("output")
                .long("output")
                .short("o")
                .help("specify the assembly file name"),
        );

    let matches = app.get_matches();

    let file_names: Vec<_> = matches.values_of("input").unwrap().into_iter().collect();
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
        match process_file(f, &matches).await {
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
