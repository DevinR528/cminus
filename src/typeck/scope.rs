use std::{
    collections::VecDeque,
    fmt,
    hash::{Hash, Hasher},
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::{
    ast::{
        parse::symbol::Ident,
        types::{Const, Expr, Path, Statement, Ty},
    },
    typeck::Visit,
};

crate fn hash_file(file: &str) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write(file.as_bytes());
    hasher.finish()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
crate enum ScopedName {
    File(u64),
    Func { file: u64, func: Ident, id: Ident },
    Adt { file: u64, adt: Ident, id: Ident },
    Global { file: u64, id: Ident },
}

impl ScopedName {
    crate fn adt_scope(adt: Ident, id: Ident, file: u64) -> Self {
        ScopedName::Adt { file, adt, id }
    }

    crate fn func_scope(func: Ident, id: Ident, file: u64) -> Self {
        ScopedName::Func { file, func, id }
    }

    crate fn global(file: u64, name: Ident) -> Self {
        ScopedName::Global { file, id: name }
    }

    crate fn ident(&self) -> Option<Ident> {
        match self {
            ScopedName::File(_) => None,
            ScopedName::Func { id, .. }
            | ScopedName::Adt { id, .. }
            | ScopedName::Global { id, .. } => Some(*id),
        }
    }
}

#[derive(Clone, Debug)]
crate struct ScopeContents {
    scope: HashMap<ScopedName, ScopeContents>,
    contents: Vec<Ident>,
}

crate struct ScopeWalker {
    global_scope: HashMap<ScopedName, ScopeContents>,
}

impl<'ast> Visit<'ast> for ScopeWalker {
    fn visit_stmt(&mut self, stmt: &'ast Statement) {}
}
