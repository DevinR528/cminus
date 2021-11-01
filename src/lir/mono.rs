use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt,
};

use crate::{
    ast::types::{self as ty, Spany, DUMMY},
    error::Error,
    lir::lower::Func,
    typeck::{
        generic::{GenericArgument, Node},
        TyCheckRes,
    },
    visit::Visit,
};

impl TyCheckRes<'_, '_> {
    crate fn mono_func(&self, func: &ty::Func) -> Vec<ty::Func> {
        let node = Node::Func(func.ident.clone());
        let mut mono_items = vec![];
        // Resolved type mono's so `T` -> `int` for function `foo`
        if let Some(res_list) = self.generic_res.resolved(&node) {
            // Mono the original function
            let mono_funcs = sub_mono_generic(&func, res_list);
            mono_items.extend_from_slice(&mono_funcs);
            // If `foo` was generic itself then any calls to generic functions `foo` makes
            // are dependent on the mono of `foo`
            let relations = self.generic_res.generic_dag().get(&node).unwrap();
            for node in relations.child_iter().filter(|n| matches!(n, Node::Func(_))) {
                let dep_func = self.var_func.name_func.get(node.name()).unwrap();
                let mono_dep_funcs = sub_mono_generic(dep_func, res_list);
                mono_items.extend_from_slice(&mono_dep_funcs)
            }
        }
        println!("{:#?}", mono_items);
        mono_items
    }
}

/// Monomorphize `foo` and dependent functions with the known types `res_list`.
fn sub_mono_generic(
    func: &ty::Func,
    res_list: &HashMap<usize, HashSet<GenericArgument<'_>>>,
) -> Vec<ty::Func> {
    // let mut gen_idx = HashSet::new();
    // for mono_ty in res_list {
    //     gen_idx.insert(mono_ty.gen_idx);
    //     let gen = func.generics.get(mono_ty.gen_idx).unwrap();
    //     for param in func.params.iter_mut() {
    //         param.ty.val.subst_generic(gen.val.generic(), &mono_ty.ty);
    //     }
    //     if gen.val.generics() == func.ret.val.generics() {
    //         func.ret.val.subst_generic(gen.val.generic(), &mono_ty.ty);
    //     }
    // }

    // for idx in gen_idx {
    //     func.generics.remove(idx);
    // }
    vec![]
}
