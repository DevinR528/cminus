# TODO

## Refactor
  
  - split typeck into infer, stmt check, expr collect files

  - Get imports working robustly
    - get multiple level paths working (`import ::foo::bar::baz::item;`)
    - std library
  - Make useful parsing errors (check the eat_if's and fail if not there when we can)
  - extern/foreign/link/clang/dynamic some sort of keyword to signify dy linked function/type
    - we currently use linked
  - if `fn call<T>(a: T): T {...}` is generic make `call::<int>(x)` and `call(x)` work
  - and...

# The Whole Point

  - Add enum inheritance and variants as types to simulate subtyping
