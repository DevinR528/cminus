#![feature(stmt_expr_attributes)]

use std::{
    env,
    fs::OpenOptions,
    io::{self, Read, Write},
    os::unix::prelude::FileExt,
    path::PathBuf,
};

use termcolor::{BufferWriter, Color, ColorChoice, ColorSpec, WriteColor};

use xshell::cmd;

const ENUMC_DEBUG: &str = "./target/debug/enumc";
// const ENUMC_RELEASE: &str = "./target/release/enumc";

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();

    match args.iter().map(|s| s.as_str()).collect::<Vec<_>>().as_slice() {
        [] => println!("version 0.0.0 of enumc"),
        ["run" | "r", more @ ..] => {
            cmd!("cargo b").run().unwrap();
            if let Err(e) = build_files(more, "-as") {
                writeln_red("Error: ", &e.to_string()).unwrap();
                std::process::exit(1);
            }
        }
        ["debug" | "d", more @ ..] => {
            cmd!("cargo b").run().unwrap();
            if let Err(e) = build_files(more, "-asv") {
                writeln_red("Error: ", &e.to_string()).unwrap();
                std::process::exit(1);
            }
        }
        ["asm" | "a", more @ ..] => {
            if let Err(e) = build_files(more, "-a") {
                writeln_red("Error: ", &e.to_string()).unwrap();
                std::process::exit(1);
            }
        }
        ["fuzz" | "f", more @ ..] => {
            let count = more.iter().find_map(|n| n.parse::<usize>().ok()).unwrap();
            let more: Vec<_> =
                more.iter().filter(|n| n.parse::<usize>().is_err()).copied().collect();
            for _ in 0..count {
                if let Err(e) = build_files(&more, "-a") {
                    writeln_red("Error: ", &e.to_string()).unwrap();
                    std::process::exit(1);
                }
            }
        }
        ["ctest" | "ct", more @ ..] => {
            cmd!("cargo b").run().unwrap();
            if let Err(e) = ui_test_run(more, "") {
                writeln_red("Error: ", &e.to_string()).unwrap();
                std::process::exit(1);
            }
        }
        ["build" | "b", more @ ..] => {
            if let Err(e) = build_files(more, "") {
                writeln_red("Error: ", &e.to_string()).unwrap();
                std::process::exit(1);
            }
        }
        ["test" | "t"] => {
            cmd!("cargo b").run().unwrap();

            #[rustfmt::skip]
            if let Err(e) = build_files(
                &[
                    "./stuff/asmgen/add/add.cm",
                    "./stuff/asmgen/add/div.cm",
                    "./stuff/asmgen/add/sub.cm",
                    "./stuff/asmgen/array/arraycall.cm",
                    "./stuff/asmgen/array/arrayinit.cm",
                    "./stuff/asmgen/bool/print.cm",
                    "./stuff/asmgen/call/call_obj.cm",
                    "./stuff/asmgen/call/call.cm",
                    "./stuff/asmgen/enum/two.cm",
                    "./stuff/asmgen/gen/gen.cm", // fairly complex generic arguments
                    "./stuff/types/string/string.cm", // test all kinds of printf/scanf types
                    "./stuff/asmgen/ifs/simp.cm", // test if/else blocks
                    "./stuff/asmgen/while/bubble.cm", /* test while loops and creating variable
                                                  * in blocks */
                    "./stuff/asmgen/while/sort.cm", // same as above with ints
                    "./stuff/asmgen/args/args.cm",  // test getting argc and argv in main
                    "./stuff/types/dynarr/field.cm", // test passing struct pointer and mutating
                    "./stuff/types/func/fnptr.cm",
                    "./stuff/asmgen/asm/assert.cm",
                    "./stuff/asmgen/asm/asm.cm",
                    "./stuff/asmgen/float/floats.cm",
                    "./stuff/assert/assert.cm",
                    "./stuff/types/size_of/size.cm",
                    // "./stuff/types/dynarr/field_ptr.cm", // field that is pointer
                ],
                "-as",
            ) {
                writeln_red("Error: ", &e.to_string()).unwrap();
                std::process::exit(1);
            }
        }
        [..] => {}
    }
}

fn build_files(files: &[&str], args: &str) -> Result<(), Box<dyn std::error::Error>> {
    for p in files {
        let path = PathBuf::from(p);
        if path.is_dir() {
            continue;
        }

        if args.is_empty() {
            cmd!("{ENUMC_DEBUG} -i {path}").run()?;
        } else {
            cmd!("{ENUMC_DEBUG} {args} -i {path}").run()?;
        }

        if args.contains('a') {
            let mut build_dir = path.clone();
            let file = build_dir
                .file_name()
                .ok_or_else::<Box<dyn std::error::Error>, _>(|| "file not found".into())?
                .to_os_string();
            build_dir.pop();
            build_dir.push("build");
            build_dir.push(file);

            let mut asm = build_dir.clone();
            asm.set_extension("s");

            let mut out = build_dir.clone();
            out.set_extension("");

            cmd!("gcc -no-pie {asm} -o {out}").run()?;
            if cmd!("{out}").read_stderr()?.contains("SEG") {
                return Err("SIGSEGV assembly crashed with segmentation fault".into());
            };
        }
    }
    Ok(())
}

fn ui_test_run(files: &[&str], args: &str) -> Result<(), Box<dyn std::error::Error>> {
    for p in files {
        let mut path = PathBuf::from(p);
        if path.is_dir() {
            continue;
        }
        let source_code = std::fs::read_to_string(&path)?;
        let should_fail = source_code.starts_with("// Fail");

        if args.is_empty() {
            let out = cmd!("{ENUMC_DEBUG} -i {path}").ignore_status().output()?;
            if should_fail {
                if out.stderr.is_empty() {
                    writeln_red(
                        "Error: ",
                        &format!("`{}` should fail but successfully compiled", path.display()),
                    )?;
                }
                path.set_extension("stderr");
                let mut fd = OpenOptions::new().read(true).write(true).create(true).open(&path)?;
                let mut buf = vec![];
                let cleaned_err = strip_ansi_escapes::strip(&out.stderr)?;
                if fd.read_to_end(&mut buf)? > 0 {
                    if buf == cleaned_err {
                        writeln_green("Pass: ", &format!("{}", path.display()))?;
                    } else {
                        writeln_red(
                            "Error: ",
                            &format!(
                                "`{}` compile errors did not match\n\n{}",
                                path.display(),
                                String::from_utf8_lossy(&out.stderr)
                            ),
                        )?;
                        std::process::exit(1)
                    }
                } else {
                    println!("{}", path.display());
                    // We truncate the file just incase
                    fd.write_all_at(&cleaned_err, 0)?;
                    writeln_green(
                        "Written: ",
                        &format!("{}\nre-run to test if output is consistent", path.display()),
                    )?;
                }
            } else if !out.stderr.is_empty() {
                writeln_red(
                    "Error: ",
                    &format!(
                        "`{}` should pass but failed to compile\n\n{}",
                        path.display(),
                        String::from_utf8_lossy(&out.stderr)
                    ),
                )?;
                std::process::exit(1)
            }
        } else {
            cmd!("{ENUMC_DEBUG} {args} -i {path}").run()?;
        }

        if args.contains('a') {
            let mut build_dir = path.clone();
            let file = build_dir
                .file_name()
                .ok_or_else::<Box<dyn std::error::Error>, _>(|| "file not found".into())?
                .to_os_string();
            build_dir.pop();
            build_dir.push("build");
            build_dir.push(file);

            let mut asm = build_dir.clone();
            asm.set_extension("s");

            let mut out = build_dir.clone();
            out.set_extension("");

            cmd!("gcc -no-pie {asm} -o {out}").run()?;
            if cmd!("{out}").read_stderr()?.contains("SEG") {
                return Err("SIGSEGV assembly crashed with segmentation fault".into());
            };
        }
    }
    Ok(())
}

fn writeln_green(text: &str, msg: &str) -> io::Result<()> {
    let bufwtr = BufferWriter::stderr(ColorChoice::Always);
    let mut buffer = bufwtr.buffer();
    buffer.set_color(ColorSpec::new().set_fg(Some(Color::Green)))?;
    write!(&mut buffer, "{}", text)?;
    buffer.reset()?;
    writeln!(&mut buffer, "{}", msg)?;
    bufwtr.print(&buffer)
}

fn writeln_red(text: &str, msg: &str) -> io::Result<()> {
    let bufwtr = BufferWriter::stderr(ColorChoice::Always);
    let mut buffer = bufwtr.buffer();
    buffer.set_color(ColorSpec::new().set_fg(Some(Color::Red)))?;
    write!(&mut buffer, "{}", text)?;
    buffer.reset()?;
    writeln!(&mut buffer, "{}", msg)?;
    bufwtr.print(&buffer)
}
