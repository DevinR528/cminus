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
    crate traits: BTreeMap<String, &'ast Trait>,
    impls: BTreeMap<String, HashMap<Vec<&'ast Ty>, &'ast Impl>>,
    proof_stack: BTreeMap<String, Vec<Vec<&'ast Ty>>>,
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

    crate fn to_solve(&mut self, trait_: &str, types: Vec<&'ast Ty>) {
        self.proof_stack.entry(trait_.to_owned()).or_default().push(types);
        // self.impls
        //     .get(trait_)
        //     .and_then(|map| map.get(types).map(|meth| meth.methods.as_slice()))
        //     .ok_or_else(|| {
        //         format!(
        //             "{}",
        //             Error::error_with_span(
        //                 tcxt,
        //                 span,
        //                 &format!(
        //                     "no implementation `{}` found for {}",
        //                     trait_,
        //                     types.iter().map(|t| format!("`{}`", t)).collect::<Vec<_>>().join(",
        // ")                 )
        //             )
        //         )
        //     })
    }
}
