## The Original Compiler

The original compiler requirements were a compiler capable of

  - declaring variables
    - constants (variables in the read only data section of assmbly)
    - locals
  - expressions
    - math
    - function calls
    - boolean logic
    - array indexing
  - statements
    - function calls
    - i/o
    - return
    - if
    - while
    - assignments
  - functions
    - named parameters
    - a single scalar return value
  - i/o
    - hardcoded write/read compiler builtin "functions"

The original language had five types:

  - float
  - int
  - char
  - bool
  - array of a previously defined type (not multi-dimension)
  - and void, used only for return type

Lexing and parsing were achieved by specifying a grammar and generating an AST from the generated tree.
Type checking consists of checking that the declaration (all const and local variable declarations
are type first like C) and the expression match. In addition, there were a specified set of type promotions
that were allowed (implicit casts), and as long as the type and the operation were legal, it type-checked.

## The Enum Compiler

The compiler adds several item/declaration kinds

  - import
  - algebraic data type (struct, enum)
  - trait block
  - impl block

and expression kinds

  - dereference
  - address of
  - trait method call
  - field access
  - struct init
  - array init
  - enum init

and statement kinds

  - assingment operator (`+=, -=, *=, etc`)
  - trait method call
  - match statement (similar to switch)
  - inline assembly
  - builtin calls

a few types were added

  - The generic type
  - struct type
  - enum type
  - a path
  - pointer
  - function pointer
  - bottom (or never)

I broke down and added a few compiler builtins, as I could see no way around implementing
these in the compiler.

  - @bottom
  - @size_of<T>
  - @line and @file

I removed the hardcoded read/write functions in favor of generic printf/scanf. These functions are annotated
with the `linked` keyword. The `linked` keyword tells the compiler to ignore them during the code generation phase;
otherwise, they are treated as any other function with no body. I also removed implicit type conversion/promotion
in expressions, parameters, and return position. There are now specific conversion functions that provide
an explicit type-safe way to convert between types.

The Enum compiler (enumc) has a multi-threaded lex and parse step. All files that the root file imports
are kicked off in another thread, and type checking waits on them before beginning. The lexer is a handwritten
tokenizer that emits a stream of tokens to the parser. The parser is also a handwritten recursive descent parser.
Finally, the parser threads communicate via channels (message channels) back to the main thread, where type
checking is waiting (this will be improved).

Type checking is the most complicated phase of the compiler. All local variable declaration uses a simple
Hindley-Milner type inference algorithm. Simple because it can not leave the expression tree it started in
to infer the types of declarations.

```rust
let x = y + 8;
// x is infered as the result of unification of inference on `y` and an `int`
```

The compiler implements parametric polymorphism using a similar approach to the type inference.
Each declaration (think function, struct, enum, trait) has all generic parameters recorded, and
each use keeps track of the type that each generic parameter was. Sets of concrete types are kept,
so for every use with a unique set, there is a unique, compiler-created version of that function or
trait implementation. This is known as monomorphisation, where each invocation of a polymorphic function
is converted to it's concrete form. It turns out structs and enums don't have to be copied/pasted (like functions)
for each concrete use, as long as the function tracks that information, everything can be type-checked,
and code generated correctly.

```rust
fn printf<T>(fmt_str: cstr, val: T) { ... }

printf("%d\n", 10); // now there is a resolved `printf<int>`
```

In the new compiler, one can import items from another file and use them as if they are defined in the
current file. This was done using a multi-threaded parsing step. Items are added to the main type checking
process via channels. The dependency graph is built implicitly by waiting on each thread to finish parsing
each imported file.

```java
import ::foo::bar;

// bar is now available
```

Traits are similar in idea to Haskell's type class. In practice, this was much simpler, only supporting direct
trait bounds with no super classing (when a trait must implement another trait) and no higher kinded-ness.
Like C#, rust, and Haskell, an unbounded generic type can have no behavior in the enum language. This is
opposite to a language like C++ where an unbounded template parameter can do anything.

\pagebreak

```rust
// This is like an interface or Haskell's type class
trait add<T> {
    fn add(a: T, b: T): T;
}

// implementing that "interface"
impl add<int> {
    fn add(a: int, b: int): int {
        return a + b;
    }
}

// This function has a bound on the generic `T` that says
// `T` must implement the `add` "interface"
fn foo<T: add>(a: T, b: T,): T {
    return <<T>::add>(a, b);
}
```

After generics and traits, the most important feature is the enum and match statement. An enum is implemented
as a tagged union. Each variant is a sequential tag and all the items contained within that variant are like the
fields of an anonymous union. A match statement branches on the tag and then exposes the fields of that variant.
It is impossible to access fields of the variant since the only way to get to them is through a match.

```rust
enum option<T> {
  // tag of 0
  some(T),
  // tag of 1
  none
}

match option::some(10) {
  option::some(num) -> {
        // num is now bound in this scope with the value 10
    },
    option::none -> {
        // no variant fields so nothing is bound in this scope
    }
}
```

The compiler uses assembly blocks for unsafe C like operations, so they can be encapsulated in a type-safe
wrapper (function). Since there is no implicit conversion, this is done inside a function using assembly
and the `@bottom` compiler builtin.

\pagebreak

```rust
fn cvti2f(_from: int): float {
    asm {
        cvtsi2sd (_from), %xmm0;
        leave;
        ret;
    }
    // This satisfies the float type since it is sort of an `any` type
    //
    // The bottom intrinsic instructs the compiler to emit an illegal instruction (ud2) since
    // it should never reach here
    return @bottom;
}

let a = 1.23 + cvti2f(1); // this type checks a-ok!
```

With the addition of the function pointer type, it is possible to pass them around as values. A function can be
passed as a parameter to another function and then called inside the function.

```rust
fn call_fn(handler: fn(int)) { ... }
fn do_thing(num: int) { ... }

call_fn(do_thing);
```

Like in most languages, structs are supported and can contain any arbitrary type except a naked "self" field
(as long as it is indirect, it is ok). Structs are similar to arrays in implementation. Structs can be passed to a
function by value (copied) or by reference. Any function returning a struct must have the caller set up the stack
so the callee (the function returning the struct) can fill it. I would like to make the copy/by value behavior
more explicit by adding a keyword or something like that. Structs added a bit of work to type checking. They can be
valid lvalues or rvalues, which means a different expression type to check. The type checker also keeps track of each
field in every struct enabling checking each expression in field access.

```rust
struct foo {
  x: int,
  y: cstr,
  z: float,
  self: *foo,
}
```

Last but not least, pointers and address-of. Adding these added a lot of complexity to the language's semantics. The
type checking now has to be aware of Lvalues and Rvalues for taking the address-of. The compiler also tracks the levels
of indirection for every expression so dereferencing can be a type-checked operation. The pointer type allows passing
references into functions so variables can be mutated anywhere as long as we know the address.

```rust
fn mutate_ref(ptr: *int) {
    *ptr += 1;
}

let x = 2;
mutate_ref(&x);
// now x = 3
```
