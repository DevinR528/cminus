# TODO

## Code/Refactor
  
  - CLEANUP
  - use parking_lot for `Mutex` and crossbeam for `Sender/Receiver/channel` stuff?
  - Refactor `asmgen.rs/iloc.rs` into something readable!!
  - Improve name resolution pass
    - Add scopes/namespaces
    - All rvalues (i.e. structs/enums/calls, not builtin types) are now paths for name resolution to resolve
    - Eventually do this "async", each name-res candidate has a dependency graph (these are fulfilled with declarations)
  - Remove `string` type for `struct str/string` and all const strings are now `[char; size]`

## Compiler work
  - Get imports working robustly
    - get multiple level paths working (`import ::foo::bar::baz::item;`)
    - std library
  - Make useful parsing errors (check the eat_if's and fail if not there when we can)
  - extern/foreign/link/clang/dynamic some sort of keyword to signify dy linked function/type
    - we currently use linked
  - Add `Stmt::Exit` to anything that has no return stmt and is a void func so codegen can correctly
    add ret instructions, and we don't rely on hardcoded crap...
  - var args for native printf
    - string needs to be convertable to a slice/array thing or impled as a struct with len and bufff
  - `size_of` or something so that adding to pointer types isn't hardcoded crap...
  - MAKE STRING WORK AS AN ARRAY!!!!! and vise versa in some way
  - if `fn call<T>(a: T): T {...}` is generic make `call::<int>(x)` and `call(x)` more robust
    - fix this `write::<int>(**ptr_ptr);` inference needs some help in this case, it should be ok
      but it gives this error "found `**&&int` expected `*&&int`"

# The Whole Point

  - Add enum inheritance and variants as types to simulate subtyping


## things I need to remember

  - get & addrof working (works in iloc kinda)
  - make sure there are no name_resolve or patch_path calls anywhere except pre type checking
