# Hello Dr. Carr ;)

There is no need to run `./compile`; the PEG parser uses a fancy Rust macro to generate it when I compile it.
So just `./run input/foo.cm` will work and `./run input/foo.cm input/bar.cm input/baz.cm` will also work.

The `grammar.pest` file is the `CminusSkeleton.g4` equivalent (PEG grammar is slightly different). The
precedence of expressions is defined partially in the grammar file but more so using precedence climbing
[here](./src/ast/parse.rs) in the function `parse_expr` and the precedence climbing algorithm [here](./src/precedence.rs).

## I have added

- `multidim.cm` for multi-dimensional arrays, array initialization, array indexing as lvalue
- `struct.cm` for struct declarations, field access as lvalue and rvalue
- `pointer.cm` for the reference operators (`*` and `->`) and address of (`&`)


### I plan to possibly add (because most of this is only interesting with type checking)

- generic types `gen<T> T foo(T x) { return x; }` (exact syntax tbd)
- interfaces/trait/type classes, to continue with the above syntax that would look like
`gen<T: impl add> T foo(T x) { return x + x; }`
and implementing would look like
`impl add for struct point { add(self, struct point other) { self.x + other.x } }`
ok so a few problems with something like this: we need to introduce impl declarations, methods (sugar for `add(struct point self, struct point other)`), the `add` trait should be `gen<T> trait add<T> { Self add(Self, Self) {...} }` where `T` constrains the thing being added `int + int` etc, so traits will have to be added also if we want the to be interesting then in the type checking trait solving would be needed (ouch that's a bunch of work)
