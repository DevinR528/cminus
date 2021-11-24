fn debug_fail(foo: &usize, bar: &usize) -> &'static str {
    "hello"
}

#[derive(derive_help::Debug)]
enum Enum2<T> {
    A(T),
    #[dbg_with(debug_fail)]
    B(usize, usize),
    #[dbg_ignore]
    Car {
        foo: usize,
        bar: bool,
    },
}

fn debug_pass<T: std::fmt::Debug>(foo: T) -> &'static str {
    "hello"
}

#[derive(derive_help::Debug)]
struct Foob<T> {
    a: bool,
    #[dbg_with(debug_pass)]
    b: T,
    #[dbg_ignore]
    c: T,
}

#[derive(derive_help::Debug)]
struct Foo<T>(#[dbg_with(debug_pass)] T);

#[derive(derive_help::Debug)]
struct FooDouble<T>(#[dbg_with(debug_pass)] T, usize);

fn debug_no_lt<T: std::fmt::Debug>(foo: T) -> String {
    "hello".into()
}

struct Bar;
impl Bar {
    fn print(&self) -> &'static str {
        "bar"
    }
}

#[derive(derive_help::Debug)]
enum EnumNoLt {
    #[dbg_with(Bar::print)]
    A(Bar),
    #[dbg_with(debug_no_lt)]
    B(usize),
}

fn main() {}
