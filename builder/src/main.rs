use std::{env, path::PathBuf};

use xshell::cmd;

const ENUMC_DEBUG: &str = "./target/debug/enumc";
// const ENUMC_RELEASE: &str = "./target/release/enumc";

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.iter().map(|s| s.as_str()).collect::<Vec<_>>().as_slice() {
        [] => println!("version 0.0.0 of enumc"),
        ["test" | "t", _more @ ..] => {
            println!("{:?}", cmd!("cargo b").read());
        }
        ["run" | "r", more @ ..] => {
            cmd!("cargo b").run().unwrap();
            if let Err(e) = build_files(more) {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        }
        ["asm" | "a", more @ ..] => {
            if let Err(e) = build_files(more) {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        }
        ["fuzz" | "f", more @ ..] => {
            let count = more.iter().find_map(|n| n.parse::<usize>().ok()).unwrap();
            let more: Vec<_> =
                more.iter().filter(|n| n.parse::<usize>().is_err()).copied().collect();
            for _ in 0..count {
                if let Err(e) = build_files(&more) {
                    eprintln!("{}", e);
                    std::process::exit(1);
                }
            }
        }
        ["build" | "b", _more @ ..] => {
            println!("{:?}", cmd!("cargo b").read());
        }
        [..] => {}
    }
}

fn build_files(files: &[&str]) -> Result<(), Box<dyn std::error::Error>> {
    for p in files {
        let path = PathBuf::from(p);
        if path.is_dir() {
            continue;
        }
        cmd!("{ENUMC_DEBUG} {path}").run()?;

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
    Ok(())
}
