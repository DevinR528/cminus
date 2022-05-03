#[derive(derive_help::Debug)]
enum Enum {
	A,
	B(usize),
	Car {
		foo: usize,
		#[dbg_ignore]
		bar: bool,
	},
}

#[derive(derive_help::Debug)]
enum Enum2<T> {
	A,
	B(T),
	#[dbg_ignore]
	Car {
		foo: usize,
		bar: bool,
	},
}

fn main() {}
