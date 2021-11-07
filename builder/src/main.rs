use std::{env, path::PathBuf};

use xshell::cmd;

const ENUMC_DEBUG: &str = "./target/debug/enumc";
const ENUMC_RELEASE: &str = "./target/release/enumc";

fn main() {
    let args: Vec<_> = env::args().skip(1).collect();
    match args.iter().map(|s| s.as_str()).collect::<Vec<_>>().as_slice() {
        [] => println!("version 0.0.0 of enumc"),
        ["test" | "t", more @ ..] => {
            println!("{:?}", cmd!("cargo b").read());
        }
        ["run" | "r", more @ ..] => {
            cmd!("cargo b").run().unwrap();
            for p in more {
                let path = PathBuf::from(p);
                if path.is_dir() {
                    continue;
                }
                cmd!("{ENUMC_DEBUG} {path}").run().unwrap();

                let mut build_dir = path.clone();
                let file = build_dir.file_name().unwrap().to_os_string();
                build_dir.pop();
                build_dir.push("build");
                build_dir.push(file);

                let mut asm = build_dir.clone();
                asm.set_extension("s");

                let mut out = build_dir.clone();
                out.set_extension("");
                cmd!("gcc -no-pie {asm} -o {out}").run().expect("cmd failed");
                if cmd!("{out}").read_stderr().expect("cmd failed").contains("SIG") {
                    panic!("seg fault")
                };
            }
        }
        ["build" | "b", more @ ..] => {
            println!("{:?}", cmd!("cargo b").read());
        }
        [..] => {}
    }
}
