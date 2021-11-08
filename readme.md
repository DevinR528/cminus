# Hello Dr. Carr ;)

There is no need to run `./compile`; the PEG parser uses a fancy Rust macro to generate it when I compile it.
So just `./run input/foo.cm` will work and `./run input/foo.cm input/bar.cm input/baz.cm` will also work. The assembly files will be in
`./input/build/foo.s`

I tested that I didn't disturb my existing output (your test files) by doing a sort of diff of my compiler error messages. The rust compiler has +10,000 of these kinds of tests. They are really helpful to make sure you don't lose or add failure modes to the compiler. (see the added `./input.stderr`)file

## I have added

- a `./stuff` sub folder in input that has a bunch of passing and failing test files
`asmgen` is test for codegen and `parse_ui` is parsing/type checking/polymorphization/monomorphization tests
the `./stuff/idea.cm` is just ideas for where this will go.


### I added

- generic types `T foo<T>(T x) { return x; }`
- interfaces/trait/type classes, to continue with the above syntax that would look like
`T add_one<T: add>(T x) { return <<T>::add>(x, 1); }`
and implementing would look like
`impl add<struct point> { add(struct point this, struct point other) { this.x += other.x } }`

I would still like to remove coercion and replace it with conversion traits like the [into trait](./input/stuff/trait/into.cm)

I am going to remove variable declaration then instantiation next time so, all my input files will be adjusted but not this time.
