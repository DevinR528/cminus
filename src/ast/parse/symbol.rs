use std::fmt;

use crate::ast::types::Range;

use self::intern::with_intern;

use super::ast::DUMMY;

#[derive(Clone, Copy, Hash, Eq, PartialEq)]
pub struct Ident {
    span: Range,
    tkn: u32,
}

impl fmt::Debug for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name().fmt(f)
    }
}

impl PartialEq<str> for Ident {
    fn eq(&self, other: &str) -> bool {
        with_intern(|intern| Some(self.tkn) == intern.lookup_tkn(other))
    }
}
impl PartialEq<Ident> for str {
    fn eq(&self, other: &Ident) -> bool {
        with_intern(|intern| Some(other.tkn) == intern.lookup_tkn(self))
    }
}

impl fmt::Display for Ident {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name().fmt(f)
    }
}

impl Ident {
    pub fn new(span: Range, name: &str) -> Self {
        Self { span, tkn: with_intern(|intern| intern.intern(name)) }
    }

    pub fn dummy() -> Self {
        // This is the "" empty interned token `0`
        Self { span: DUMMY, tkn: 0 }
    }

    pub fn span(&self) -> Range {
        self.span
    }

    pub fn name(&self) -> &str {
        intern::with_intern(|intern| intern.lookup_str(self.tkn))
    }
}

pub mod intern {
    use rustc_hash::FxHashMap;
    use std::mem;

    crate fn with_intern<T, F: FnMut(&mut Interner) -> T>(mut f: F) -> T {
        f(&mut super::super::kw::INTERN.lock())
    }

    crate struct Interner {
        map: FxHashMap<&'static str, u32>,
        vec: Vec<&'static str>,
        buf: String,
        full: Vec<String>,
    }

    impl Interner {
        pub fn pre_load(stuff: &[&'static str]) -> Interner {
            Interner {
                map: stuff.iter().copied().enumerate().map(|(a, b)| (b, a as u32)).collect(),
                vec: stuff.into(),
                buf: String::with_capacity(stuff.len() / 2),
                full: Vec::new(),
            }
        }

        pub fn intern(&mut self, name: &str) -> u32 {
            if let Some(&id) = self.map.get(name) {
                return id;
            }
            let name = unsafe { self.alloc(name) };
            let id = self.map.len() as u32;
            self.map.insert(name, id);
            self.vec.push(name);

            debug_assert!(self.lookup_str(id) == name);
            debug_assert!(self.intern(name) == id);

            id
        }

        pub fn lookup_tkn(&self, id: &str) -> Option<u32> {
            self.map.get(id).copied()
        }

        pub fn lookup_str(&self, id: u32) -> &'static str {
            self.vec[id as usize]
        }

        unsafe fn alloc(&mut self, name: &str) -> &'static str {
            let cap = self.buf.capacity();
            if cap < self.buf.len() + name.len() {
                let new_cap = (cap.max(name.len()) + 1).next_power_of_two();
                let new_buf = String::with_capacity(new_cap);
                let old_buf = mem::replace(&mut self.buf, new_buf);
                self.full.push(old_buf);
            }

            let interned = {
                let start = self.buf.len();
                self.buf.push_str(name);
                &self.buf[start..]
            };

            &*(interned as *const str)
        }
    }
}
