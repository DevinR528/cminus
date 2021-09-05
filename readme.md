# Hello Dr. Carr ;)

There is no need to run `./compile`; the PEG parser uses a fancy Rust macro to generate it when I compile it.
So just `./run input/foo.cm` will work and `./run input/foo.cm input/bar.cm input/baz.cm` will also work.

The `grammar.pest` file is the `CminusSkeleton.g4` equivalent (PEG grammar is slightly different). The
precedence of expressions is defined partially in the grammar file but more so using precedence climbing
[here](./src/ast/parse.rs) in the function `parse_expr` and the precedence climbing algorithm [here](./src/precedence.rs).


