Devin Ragotzy
CS-1210 Introduction to C
Midterm Paper

## Type Systems

First some definitions:

### Weak Typing

A programming language with a weak type system has variables that can change type during execution, can be declared without type declarations, and type checking is absent. One thing most weakly typed languages have in common is that variables can become any type. It is perfectly valid in Javascript to declare x as a number `var x = 10;` then bind x to a string `x = "hello"`. Notice there is no type declaration in Javascript's variable declaration. This ties into the next point that type checking is impossible before runtime and so is turned into runtime exceptions or some other behavior (at best errors and crashing at worst undefined behavior).

### Strong Typing

A programming language is said to be strongly typed if a variable can only be assigned/reassigned to values of the same type/size. A compiler or interpreter has a type-checking phase. Variables in a strongly typed language need to be declared with a type or can be inferred according to some inference system (Hindley-Milner). In languages considered to have a strong type system, when a variable is declared to be a specific type it cannot be reassigned to a different type. The somewhat (in)famous counter-example of this is C's pointer-cast "any" type. To interpret the bits of one type as the bits of another type, one can declare a variable `int x = 68;` get a pointer to that variable `int* y = &x` then cast and dereference `char z = *(char*)y;` z now has a value of 'D'. I like to think of C as bit-ly typed like Javascript (and python) are string-ly typed, functions "type-check" based on a string argument.

```js
func add_evenhandler(event, callback) {
    switch event {
        case "on_click":
            //
        case "etc":
    }
}
```
In general a strongly typed language has "compile" time errors. These errors are mistakes that are caught because of the type-system. In C if I `int x = 'c';` the program will not compile.
