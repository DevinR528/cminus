#[derive(derive_help::Debug)]
struct Foo {
    a: bool,
    #[dbg_ignore]
    c: usize,
}

#[derive(derive_help::Debug)]
struct Fooa<T> {
    a: bool,
    c: T,
}

#[derive(derive_help::Debug)]
struct Foob<T> {
    a: bool,
    #[dbg_ignore]
    c: T,
}

fn main() {}
