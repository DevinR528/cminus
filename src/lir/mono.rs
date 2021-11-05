use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fmt,
};

use crate::{
    ast::types::{self as ty, Spany, Ty, DUMMY},
    error::Error,
    lir::lower::Func,
    typeck::{
        generic::{GenericArgument, Node},
        TyCheckRes,
    },
    visit::{Visit, VisitMut},
};

struct GenSubstitution<'a> {
    generic: &'a Ty,
    ty: &'a Ty,
    tcxt: &'a TyCheckRes<'a, 'a>,
}

impl<'ast> VisitMut<'ast> for GenSubstitution<'ast> {
    fn visit_func(&mut self, func: &'ast mut ty::Func) {
        func.ident.push_str(&self.ty.to_string());
        crate::visit::walk_mut_func(self, func);
    }

    fn visit_expr(&mut self, expr: &'ast mut ty::Expression) {
        if let ty::Expr::TraitMeth { trait_, type_args, args } = &expr.val {
            println!(
                "{}: {} -> {}",
                type_args.iter().map(|t| t.val.to_string()).collect::<Vec<_>>().join(", "),
                trait_,
                self.ty
            );
        }
    }

    fn visit_ty(&mut self, ty: &mut ty::Type) {
        ty.val.subst_generic(self.generic.generic(), self.ty)
    }
}

crate struct TraitRes<'a> {
    type_args: Vec<&'a Ty>,
    tcxt: &'a TyCheckRes<'a, 'a>,
}

impl<'a> TraitRes<'a> {
    crate fn new(tcxt: &'a TyCheckRes<'_, '_>, type_args: Vec<&'a Ty>) -> Self {
        Self { tcxt, type_args }
    }
}

impl<'ast, 'a> VisitMut<'ast> for TraitRes<'a> {
    fn visit_expr(&mut self, expr: &'ast mut ty::Expression) {
        let mut x = None;
        if let ty::Expr::TraitMeth { trait_, type_args, args } = &mut expr.val {
            if let Some(i) =
                self.tcxt.trait_solve.impls.get(trait_).and_then(|imp| imp.get(&self.type_args))
            {
                let mut args = args.clone();
                for arg in &mut args {
                    self.visit_expr(arg);
                }
                let ident = format!(
                    "{}<{}>",
                    trait_,
                    self.type_args.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
                );
                x = Some(ty::Expr::Call { ident, args, type_args: vec![i.method.ret.clone()] });
            } else {
                panic!(
                    "{}",
                    Error::error_with_span(
                        self.tcxt,
                        expr.span,
                        &format!(
                            "`{}` is not implemented for `<{}>`",
                            trait_,
                            self.type_args
                                .iter()
                                .map(|t| t.to_string())
                                .collect::<Vec<_>>()
                                .join(", "),
                        )
                    )
                )
            }
        }

        if let Some(replace) = x {
            expr.val = replace;
        }
    }

    fn visit_stmt(&mut self, stmt: &'ast mut ty::Statement) {
        let mut x = None;
        if let ty::Stmt::TraitMeth(ty::Spanned {
            val: ty::Expr::TraitMeth { trait_, type_args, args },
            ..
        }) = &mut stmt.val
        {
            if let Some(i) =
                self.tcxt.trait_solve.impls.get(trait_).and_then(|imp| imp.get(&self.type_args))
            {
                let mut args = args.clone();
                for arg in &mut args {
                    self.visit_expr(arg);
                }
                let ident = format!(
                    "{}<{}>",
                    trait_,
                    self.type_args.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","),
                );
                x = Some(ty::Stmt::Call(
                    ty::Expr::Call { ident, args, type_args: vec![i.method.ret.clone()] }
                        .into_spanned(DUMMY),
                ));
            } else {
                panic!(
                    "{}",
                    Error::error_with_span(
                        self.tcxt,
                        stmt.span,
                        &format!(
                            "`{}` is not implemented for `<{}>`",
                            trait_,
                            self.type_args
                                .iter()
                                .map(|t| t.to_string())
                                .collect::<Vec<_>>()
                                .join(", "),
                        )
                    )
                )
            }
        }
        if let Some(replace) = x {
            stmt.val = replace;
        }
        // `walk_stmt` here to recurse into `Expr`, `visit_expr` stops (no walk call)
        crate::visit::walk_mut_stmt(self, stmt);
    }
}

impl TyCheckRes<'_, '_> {
    crate fn mono_func(&self, func: &ty::Func) -> Vec<ty::Func> {
        let node = Node::Func(func.ident.clone());
        let mut mono_items = vec![];
        // Resolved type mono's so `T` -> `int` for function `foo`
        if let Some(res_list) = self.generic_res.resolved(&node) {
            // Mono the original function
            let mono_funcs = sub_mono_generic(func, res_list, self);
            mono_items.extend_from_slice(&mono_funcs);
            // If `foo` was generic itself then any calls to generic functions `foo` makes
            // are dependent on the mono of `foo`
            let relations = self.generic_res.generic_dag().get(&node).unwrap();
            for node in relations.child_iter().filter(|n| matches!(n, Node::Func(_))) {
                let dep_func = self.var_func.name_func.get(node.name()).unwrap();
                let mono_dep_funcs = sub_mono_generic(dep_func, res_list, self);
                mono_items.extend_from_slice(&mono_dep_funcs)
            }
        }
        // println!("{:#?}", mono_items);
        mono_items
    }
}

/// Monomorphize `foo` and dependent functions with the known types `res_list`.
fn sub_mono_generic(
    func: &ty::Func,
    res_list: &HashMap<usize, HashSet<GenericArgument<'_>>>,
    tcxt: &TyCheckRes<'_, '_>,
) -> Vec<ty::Func> {
    // We know that there is at least one generic so getting the zeroth value should be fine
    let number_of_specializations = res_list.get(&0).map_or(0, |m| m.len());
    let mut map: HashMap<_, Vec<_>> = HashMap::new();
    for arg in res_list.iter().flat_map(|(_, a)| a) {
        map.entry(arg.instance_id).or_default().push(arg);
    }

    let mut functions = vec![func.clone(); number_of_specializations];
    for (idx, mut generics) in map.into_values().enumerate() {
        functions[idx].ident.push('<');
        generics.sort_by(|a, b| a.gen_idx.cmp(&b.gen_idx));

        for (i, gen) in generics.iter().enumerate() {
            let gen_param = functions[idx].generics.get(gen.gen_idx).unwrap().clone();
            // Replace ALL uses of this generic and remove the generic parameters
            let mut subs = GenSubstitution { generic: &gen_param.val, ty: &gen.ty, tcxt };
            subs.visit_func(&mut functions[idx]);
            if i != generics.len() - 1 {
                functions[idx].ident.push(',');
            }
        }
        functions[idx].ident.push('>');

        let mut trait_res =
            TraitRes { type_args: generics.iter().map(|g| &g.ty).collect::<Vec<_>>(), tcxt };
        trait_res.visit_func(&mut functions[idx]);
    }

    for f in &mut functions {
        if f.generics.len() == res_list.len() {
            f.generics.clear();
        }
    }

    functions
}
