use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    iter,
};

use crate::{
    ast::types::{Expr, Func, Impl, Range, Spany, Trait, TraitMethod, Ty, Type, Var, DUMMY},
    error::Error,
    typeck::TyCheckRes,
};

#[derive(Debug, Default)]
crate struct TraitSolve<'ast> {
    traits: BTreeMap<String, &'ast Trait>,
    impls: BTreeMap<String, HashMap<Vec<&'ast Ty>, &'ast Impl>>,
}

impl<'ast> TraitSolve<'ast> {
    crate fn add_trait(&mut self, t: &'ast Trait) -> Option<&'ast Trait> {
        self.traits.insert(t.ident.to_string(), t)
    }

    crate fn add_impl(&mut self, t: &'ast Impl) -> Result<(), String> {
        if !self.traits.contains_key(&t.ident) {
            return Err("no trait".to_owned());
        }
        let set = t.type_arguments.iter().map(|t| &t.val).collect();
        let mut map = iter::once((set, t)).collect();
        if self.impls.insert(t.ident.to_string(), map).is_some() {
            return Err("found duplicate impl".to_owned());
        }
        Ok(())
    }

    crate fn solve(
        &self,
        tcxt: &TyCheckRes<'_, '_>,
        trait_: &str,
        types: &[&Ty],
        span: Range,
    ) -> Result<&[Func], String> {
        self.impls
            .get(trait_)
            .and_then(|map| map.get(types).map(|meth| meth.methods.as_slice()))
            .ok_or_else(|| {
                format!(
                    "{}",
                    Error::error_with_span(
                        tcxt,
                        span,
                        &format!(
                            "no implementation `{}` found for {}",
                            trait_,
                            types.iter().map(|t| format!("`{}`", t)).collect::<Vec<_>>().join(", ")
                        )
                    )
                )
            })
    }
}
