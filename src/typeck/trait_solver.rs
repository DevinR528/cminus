use std::fmt;

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{
    ast::{
        parse::Ident,
        types::{Impl, Path, Trait, Ty},
    },
    typeck::generic::Node,
};

#[allow(dead_code)]
#[derive(Debug, Default)]
crate struct ToUnify<'ast> {
    /// All the types that have to be unified and proven.
    solution_stack: Vec<&'ast Ty>,
    /// The dependence chain and location of each trait use.
    chain: Vec<Node>,
}

#[derive(Default)]
crate struct TraitSolve<'ast> {
    /// The name of the trait to the declaration.
    crate traits: HashMap<Path, &'ast Trait>,
    /// The name of the trait to each type implementation.
    ///
    /// This would consider `trait foo<int, bool>` distinct from `trait foo<bool, int>`.
    crate impls: HashMap<Path, HashMap<Vec<&'ast Ty>, &'ast Impl>>,
    proof_stack: HashMap<Ident, Vec<ToUnify<'ast>>>,
}

impl fmt::Debug for TraitSolve<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TraitSolver")
            .field(
                "impls",
                &self
                    .impls
                    .iter()
                    .map(|(k, map)| (k, map.keys().collect::<Vec<_>>()))
                    .collect::<Vec<_>>(),
            )
            .field("proof_stack", &self.proof_stack)
            .finish()
    }
}

impl<'ast> TraitSolve<'ast> {
    crate fn add_trait(&mut self, t: &'ast Trait) -> Option<&'ast Trait> {
        self.traits.insert(t.path.clone(), t)
    }

    crate fn add_impl(&mut self, t: &'ast Impl) -> Result<(), String> {
        if !self.traits.contains_key(&t.path) {
            return Err("no trait".to_owned());
        }
        let set = t.type_arguments.iter().map(|t| &t.val).collect();
        if self.impls.entry(t.path.clone()).or_default().insert(set, t).is_some() {
            return Err("found duplicate impl".to_owned());
        }
        Ok(())
    }

    #[allow(clippy::wrong_self_convention)]
    crate fn to_solve(
        &mut self,
        trait_: Ident,
        solution_stack: Vec<&'ast Ty>,
        chain: Option<Vec<Node>>,
    ) {
        self.proof_stack
            .entry(trait_)
            .or_default()
            .push(ToUnify { solution_stack, chain: chain.into_iter().flatten().collect() });
    }

    // crate fn has_impl(&self, imp: &str, _to_unify: &Ty) -> bool {
    //     if let Some(_imp) = self.impls.get(imp) {
    //         // imp.
    //     }
    //     false
    // }

    // crate fn unify(&self, _tcxt: &TyCheckRes<'_, '_>, _bound_generic: &Ty, _to_unify: &Ty) ->
    // bool {     true
    // }
}
