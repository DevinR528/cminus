Devin Ragotzy
CS-1210 Introduction to C
Midterm Paper

## Type Systems

First some definitions:

### Weak Typing

A programming language with a weak type system has variables that can change type during execution, can be declared without type declarations, and type checking is absent. One thing most weakly typed languages have in common is that variables can become any type. It is perfectly valid in Javascript to declare x as a number `var x = 10;` then bind x to a string `x = "hello"`. Notice there is no type declaration in Javascript's variable declaration. This ties into the next point that type checking is impossible before runtime and so is turned into runtime exceptions or some other behavior (at best errors and crashing at worst undefined behavior).

### Strong Typing

A programming language is said to be strongly typed if a variable can only be assigned/reassigned to values of the same type/size. A compiler or interpreter has a type-checking phase. Variables in a strongly typed language need to be declared with a type or can be inferred according to some inference system (Hindley-Milner). In languages considered to have a strong type system, when a variable is declared to be a specific type, it cannot be reassigned to a different type. The somewhat (in)famous counter-example of this is C's pointer-cast "any" type. To interpret the bits of one type as the bits of another type, one can declare a variable `int x = 68;` get a pointer to that variable `int* y = &x` then cast and dereference `char z = *(char*)y;` z now has a value of 'D'. I like to think of C as bit-ly typed like Javascript (and python) are string-ly typed, functions "type-check" based on a string argument.

```js
func eventHandler(event, callback) {
    switch event {
        case "on_click":
            //
        case "etc":
    }
}
```
In general, a strongly typed language has "compile" time errors. These errors are mistakes that are caught because of the type-system. In C if I `int x = 'c';` the program will not compile because of a strong(ish) type system.

### Static Typing

A static-ly typed language, like strong typing, tries to move errors to compile time. By using a type system, the compiler is able to enforce that all values have a type and follow the rules laid out by the system. A static type system allows the compiler (or more specifically, the type-checker) to know a lot more about a program. This allows the compiler can catch typos, unused/undeclared variables, and all kinds of other bugs that in a weak or dynamic type system would be caught at runtime. A way to think about a static type system is that all types are static; once declared, a type cannot change. For example, in Rust `let x = foo + 12;` even though `x`'s type is inferred it cannot change after declaration, the compiler uses the type of `foo` and `12` (numbers default to signed long) to check/infer `x`.

### Dynamic Typing

A dynamic type system allows types to be changed dynamically. These languages have to be managed languages. They have a garbage collector and allocate everything on the heap because no size or type information is availible. A language like this can change the type of a variable to any other type at any time.

```python
def foo():
    return 10

a = 10
a = foo
a() + a # this will throw a runtime exception
```

A dynamically typed language has no type checking phase since no assumptions can be made about the type of each variable.

### Duck Typing

Python is one of the few duck typed language, the idea is, if it quacks like a duck, it is a duck. This relates to a type system through everything in python being an object so `x + y` works because x quacks like a number that can be added to another number. This is checked by accessing properties or methods of the object in question. If you want to print a type you have created, you must make it quack like a string.

```python
class Foo:
    def init(self):
    def __str__(self):
        return "class Foo"
print(Foo()) # prints 'class Foo'
```



## C is Strong and Weak

C is considered strongly typed because everything need to have a type. All declarations are typed and once declared they cannot be changed. C is weakly typed because you can, in practice, subvert this entirely by casting one value to another value. This doesn't break C's strong typing because another variable has to be declared, you still cannot alter the type of a declared variable. Boom strong typing. The compiler does do type checking but, because of casts and pointer arithmetic, they are a best effort. Boom weak typing.

## IMHO

I dislike weak and dynamic typing. I really don't like using some python library and vsCode gives me `def foo(x: Any, y: Any) -> Any` as a hoover definition. A weak type system (which I will use to refer to weak/dynamic/duck) does not do a good job of communicating intent or meaning to the user of that library/program. A weak type system does allow fast prototyping since there is almost ZERO boilerplate, but this seems to be lost when it comes time to fix bugs, upkeep, refactor, and general maintenance.

I think Rust is an amazing example of what can be accomplished with a type system. It is a strong static typed language, that offers solutions to problems with C/C++. Rust fixes a majority of memory bugs and, I think, with it's type system enables writing libraries where it is impossible to misuse it.

```rust
// like a c enum or union this can only be one or the other (sum type)
enum Result<T, E> { // T and E are generic
    Ok(T),
    Err(E),
}
// An exported function
pub fn add(x: u8, y: u8) -> Result<u8, String> {
    if x.checked_add(y).is_err() {
        return Result::Err("we failed".to_string());
    } else {
        return Result::Ok(x + y)
    }
}
// No one can use this without handling both cases
match add(1, 255) {
    Result::Ok(num) => println!("{}", num), // num is the x + y
    Result::Err(err) => println!("{}", err), // err is the string
}
```

As for the memory safety

```rust
pub fn good<'a>(x: &'a str) -> &'a str {
    // x's lifetime goes on after good returns so this is ok
    return x;
}
pub fn bad<'a>() -> &'a str {
    // since this isn't c++ even heap allocation lifetimes are tracked
    let x = String::new(); // x is owned by this function
    return x.as_str(); // return a reference to the string
    // x is dropped (freed here)
}
pub fn good() -> &'static str {
    return "this is ok since I'm in static memory";
}
```
