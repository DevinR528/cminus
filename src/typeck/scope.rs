use std::{
    collections::{hash_map::Entry, VecDeque},
    fmt,
    hash::{Hash, Hasher},
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::{
    ast::{
        parse::symbol::Ident,
        types::{Const, Expr, Path, Range, Statement, Ty},
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
crate enum Scope {
    Trait { file: u64, trait_: Ident },
    Impl { file: u64, imp: Ident },
    Func { file: u64, func: Ident },
    Adt { file: u64, adt: Ident },
    Block(Range),
    File(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
crate enum ScopeItem {
    Var(Ident),
    Field(Ident),
    Variant(Ident),
}

#[derive(Clone, Debug, Default)]
crate struct ScopeContents {
    contents: HashMap<Scope, Vec<ScopeItem>>,
}

#[derive(Clone, Debug, Default)]
crate struct ScopeWalker {
    global_scope: HashMap<Scope, ScopeContents>,
}

impl ScopeWalker {
    crate fn add_file_scopes(&mut self, files: &HashMap<u64, &str>) {
        for k in files.keys() {
            self.global_scope.insert(Scope::File(*k), ScopeContents::default());
        }
    }

    crate fn add_file_scope(&mut self, file: u64) {
        self.global_scope.insert(Scope::File(file), ScopeContents::default());
    }

    crate fn add_decl(&mut self, file: u64, decl_scope: Scope) {
        if let Some(items) = self.global_scope.get_mut(&Scope::File(file)) {
            items.contents.insert(decl_scope, vec![]);
        }
    }

    crate fn add_item(&mut self, file: u64, decl_scope: Scope, item: ScopeItem) {
        if let Some(items) = self.global_scope.get_mut(&Scope::File(file)) {
            match items.contents.entry(decl_scope) {
                Entry::Occupied(mut inner) => {
                    inner.get_mut().push(item);
                }
                Entry::Vacant(v) => {
                    v.insert(vec![item]);
                }
            };
        }
    }
}

impl<'ast> Visit<'ast> for ScopeWalker {
    fn visit_stmt(&mut self, stmt: &'ast Statement) {}
}
