use std::{
    collections::{hash_map::Entry, VecDeque},
    fmt,
    hash::{Hash, Hasher},
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::{
    ast::{
        parse::symbol::Ident,
        types::{Const, Expr, Path, Range, Spany, Statement, Stmt, Ty, DUMMY},
    },
    typeck::{TyCheckRes, Visit},
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

pub type FileScope = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
crate enum Scope {
    Trait { file: u64, trait_: Ident },
    Impl { file: u64, imp: Ident },
    Func { file: u64, func: Ident },
    Struct { file: u64, adt: Ident },
    Enum { file: u64, adt: Ident },
    Global { file: u64, name: Ident },
    Block(Range),
}

impl Scope {
    crate fn ident(&self) -> Ident {
        match self {
            Scope::Trait { file, trait_ } => *trait_,
            Scope::Impl { file, imp } => *imp,
            Scope::Func { file, func } => *func,
            Scope::Struct { file, adt } => *adt,
            Scope::Enum { file, adt } => *adt,
            Scope::Global { file, name } => *name,
            Scope::Block(_) => todo!(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
crate enum ItemIn {
    Var(Ident),
    Field(Ident),
    Variant(Ident),
}

#[derive(Clone, Debug)]
crate struct Items {
    parent: Scope,
    items: Vec<ItemIn>,
}
#[derive(Clone, Debug, Default)]
crate struct ScopeContents {
    contents: HashMap<Ident, Items>,
}

#[derive(Clone, Debug, Default)]
crate struct ScopeWalker {
    crate global_scope: HashMap<FileScope, ScopeContents>,
    // FIXME: this could support multiple levels of scope, it now is only:
    // `file -> decl -> items` arbitrary nesting could work or module scope??
    scope_stack: Vec<Scope>,
}

impl ScopeWalker {
    crate fn add_file_scopes(&mut self, files: &HashMap<u64, &str>) {
        for k in files.keys() {
            self.global_scope.insert(*k, ScopeContents::default());
        }
    }

    crate fn add_to_scope_stack(&mut self, scope: Scope) {
        self.scope_stack.push(scope)
    }

    crate fn pop_scope_stack(&mut self) {
        self.scope_stack.pop();
    }

    crate fn add_file_scope(&mut self, file: u64) {
        self.global_scope.insert(file, ScopeContents::default());
    }

    crate fn add_decl(&mut self, file: u64, decl_ty: Scope) {
        if let Some(items) = self.global_scope.get_mut(&file) {
            items.contents.insert(decl_ty.ident(), Items { parent: decl_ty, items: vec![] });
        }
    }

    crate fn add_item(&mut self, file: u64, in_scope: Scope, item: ItemIn) {
        if let Some(items) = self.global_scope.get_mut(&file) {
            match items.contents.entry(in_scope.ident()) {
                Entry::Occupied(mut inner) => {
                    inner.get_mut().items.push(item);
                }
                Entry::Vacant(v) => {
                    v.insert(Items { parent: in_scope, items: vec![item] });
                }
            };
        }
    }

    /// Resolve a `Ty::Path` to it's canonical type.
    ///
    /// Name resolution happens when a name is used as a type i.e. `fn call(a: foo, b: int): bar`
    /// both `foo` and `bar` are `Path`s until `resolve_name` can convert them to useful types (Path
    /// is a type just not particularly helpful).
    crate fn resolve_name(&self, ty: &Ty, tctx: &TyCheckRes<'_, '_>) -> Option<Ty> {
        Some(match ty {
            Ty::Array { size, ty: t } => Ty::Array {
                size: *size,
                ty: box self.resolve_name(&t.val, tctx)?.into_spanned(DUMMY),
            },
            Ty::Struct { ident, gen } => Ty::Struct {
                ident: *ident,
                gen: gen
                    .iter()
                    .map(|t| Some(self.resolve_name(&t.val, tctx)?.into_spanned(DUMMY)))
                    .collect::<Option<Vec<_>>>()?,
            },
            Ty::Enum { ident, gen } => Ty::Enum {
                ident: *ident,
                gen: gen
                    .iter()
                    .map(|t| Some(self.resolve_name(&t.val, tctx)?.into_spanned(DUMMY)))
                    .collect::<Option<Vec<_>>>()?,
            },
            Ty::Path(path) => self.type_from_path(path, tctx)?,
            Ty::Ptr(t) => Ty::Ptr(box self.resolve_name(&t.val, tctx)?.into_spanned(DUMMY)),
            Ty::Ref(t) => Ty::Ref(box self.resolve_name(&t.val, tctx)?.into_spanned(DUMMY)),
            Ty::Func { ident, ret, params } => Ty::Func {
                ident: *ident,
                params: params
                    .iter()
                    .map(|t| self.resolve_name(t, tctx))
                    .collect::<Option<Vec<_>>>()?,
                ret: box self.resolve_name(&**ret, tctx)?,
            },
            _ => ty.clone(),
        })
    }

    crate fn type_from_path(&self, path: &Path, tctx: &TyCheckRes<'_, '_>) -> Option<Ty> {
        let mut p = path.clone();
        let item = p.segs.first()?;
        self.global_scope.get(&path.span.file_id)?.contents.get(item).and_then(|scope| {
            match scope.parent {
                Scope::Trait { file, trait_ } => todo!(),
                Scope::Impl { file, imp } => todo!(),
                // TODO: just make fn a type already
                Scope::Func { file, func } => None,
                Scope::Struct { file, adt } => tctx.name_struct.get(&adt).map(|it| Ty::Struct {
                    ident: it.ident,
                    gen: it
                        .generics
                        .iter()
                        .map(|g| {
                            Ty::Generic { ident: g.ident, bound: g.bound.clone() }
                                .into_spanned(DUMMY)
                        })
                        .collect(),
                }),
                Scope::Enum { file, adt } => tctx.name_enum.get(&adt).map(|it| Ty::Enum {
                    ident: it.ident,
                    gen: it
                        .generics
                        .iter()
                        .map(|g| {
                            Ty::Generic { ident: g.ident, bound: g.bound.clone() }
                                .into_spanned(DUMMY)
                        })
                        .collect(),
                }),
                Scope::Global { file, name } => tctx.global.get(&name).cloned(),
                Scope::Block(_) => todo!(),
            }
        })
    }

    // crate fn resolve_name<'a>(&self, ty: &'a Ty) -> Cow<'a, Ty> {
    //     match ty {
    //         Ty::Array { size, ty: t } => Cow::Owned(Ty::Array {
    //             size: *size,
    //             ty: box self.resolve_name(&t.val).into_spanned(DUMMY),
    //         }),
    //         Ty::Struct { ident, gen } => todo!(),
    //         Ty::Enum { ident, gen } => todo!(),
    //         Ty::Path(_) => todo!(),
    //         Ty::Ptr(_) => todo!(),
    //         Ty::Ref(_) => todo!(),
    //         Ty::Func { ident, ret, params } => todo!(),
    //         _ => Cow::Borrowed(ty),
    //     }
    // }
}

impl<'ast> Visit<'ast> for ScopeWalker {
    fn visit_stmt(&mut self, stmt: &'ast Statement) {
        if let Stmt::Assign { lval, is_let: true, .. } = &stmt.val {
            let ident = lval.val.as_ident();
            self.add_item(
                stmt.span.file_id,
                self.scope_stack.last().copied().expect("ICE: statement found outside of scope"),
                ItemIn::Var(ident),
            )
        }
    }
}

// #[test]
// #[allow(non_camel_case_types)]
// fn scope() {
//     trait foo {}
//     struct foo;

//     enum foo { bar, a, }

// }
