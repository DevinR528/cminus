use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    iter,
};

use crate::{
    ast::types::{Expr, Func, Impl, Range, Spany, Trait, TraitMethod, Ty, Type, Var, DUMMY},
    error::Error,
    typeck::{generic::Node, TyCheckRes},
};

#[derive(Debug, Default)]
crate struct ToUnify<'ast> {
    /// All the types that have to be unified and proven.
    solution_stack: Vec<&'ast Ty>,
    /// The dependence chain and location of each trait use.
    chain: Vec<Node>,
}

#[derive(Debug, Default)]
crate struct TraitSolve<'ast> {
    /// The name of the trait to the declaration.
    crate traits: BTreeMap<String, &'ast Trait>,
    /// The name of the trait to each type implementation.
    ///
    /// This would consider `trait foo<int, bool>` distinct from `trait foo<bool, int>`.
    crate impls: BTreeMap<String, HashMap<Vec<&'ast Ty>, &'ast Impl>>,
    proof_stack: BTreeMap<String, Vec<ToUnify<'ast>>>,
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
        if self.impls.entry(t.ident.to_string()).or_default().insert(set, t).is_some() {
            return Err("found duplicate impl".to_owned());
        }
        Ok(())
    }

    #[allow(clippy::wrong_self_convention)]
    crate fn to_solve(
        &mut self,
        trait_: &str,
        solution_stack: Vec<&'ast Ty>,
        chain: Option<Vec<Node>>,
    ) {
        self.proof_stack
            .entry(trait_.to_owned())
            .or_default()
            .push(ToUnify { solution_stack, chain: chain.into_iter().flatten().collect() });
    }

    crate fn unify(&self, tcxt: &TyCheckRes<'_, '_>, bound_generic: &Ty, to_unify: &Ty) -> bool {
        true
    }
}
