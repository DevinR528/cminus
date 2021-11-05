#[test]
fn compile_test() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/enum.rs");
    t.pass("tests/ui/struct.rs");
}

fn main() {}
