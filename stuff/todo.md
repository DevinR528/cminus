# TODO

## Code/Refactor
  
  - split typeck into infer, stmt check, expr collect files and cleanup
  - use parking_lot for `Mutex` and crossbeam for `Sender/Receiver/channel` stuff
  - Refactor `asmgen.rs` into something readable
  - Add name resolution pass
    - Add scopes/namespaces
    - All rvalues (i.e. structs/enums/calls, not builtin types) are now paths for name resolution to resolve
    - Eventually do this "async", each name-res candidate has a dependency graph (these are fulfilled with declarations)

## Compiler work
  - Get imports working robustly
    - get multiple level paths working (`import ::foo::bar::baz::item;`)
    - std library
  - Make useful parsing errors (check the eat_if's and fail if not there when we can)
  - extern/foreign/link/clang/dynamic some sort of keyword to signify dy linked function/type
    - we currently use linked
  - if `fn call<T>(a: T): T {...}` is generic make `call::<int>(x)` and `call(x)` work
  - add support for uninitialized values MAYBE???
  - var args for native printf
    - string needs to be convertable to a slice/array thing or impled as a struct with len and bufff
  - and...

# The Whole Point

  - Add enum inheritance and variants as types to simulate subtyping
