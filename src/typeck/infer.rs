use std::{
    cell::{Cell, RefCell},
    fmt,
    sync::mpsc::Receiver,
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{
    ast::{
        parse::{symbol::Ident, ParseResult},
        types::{
            to_rng, Adt, BinOp, Binding, Block, Const, Decl, Declaration, Enum, Expr, Expression,
            Field, FieldInit, Func, Generic, Impl, MatchArm, Param, Pat, Path, Range, Spany,
            Statement, Stmt, Struct, Trait, Ty, Type, TypeEquality, UnOp, Val, Variant, DUMMY,
        },
    },
    error::Error,
    typeck::{
        check::{fold_ty, resolve_ty},
        generic::{Node, TyRegion},
        scope::ScopedName,
        TyCheckRes,
    },
    visit::Visit,
};

//
//
//
// This handles type inference for us.
#[derive(Debug)]
crate struct TypeInfer<'v, 'ast, 'input> {
    crate tcxt: &'v mut TyCheckRes<'ast, 'input>,
}

impl<'ast> TypeInfer<'_, 'ast, '_> {
    fn unify(&self, ty: Option<&Ty>, with: Option<&Ty>) -> Option<Ty> {
        match (ty, with) {
            (Some(t1), Some(t2)) => match (t1, t2) {
                (Ty::Generic { ident: i1, bound: b1 }, Ty::Generic { ident: i2, bound: b2 }) => {
                    todo!()
                }
                (Ty::Array { size: s1, ty: ty1 }, Ty::Array { size: s2, ty: ty2 }) => {
                    if s1 == s2 {
                        Some(Ty::Array {
                            size: *s1,
                            ty: box self.unify(Some(t1), Some(t2))?.into_spanned(DUMMY),
                        })
                    } else {
                        None
                    }
                }
                (Ty::Struct { ident: i1, gen: g1 }, Ty::Struct { ident: i2, gen: g2 }) => {
                    if i1 == i2 {
                        Some(Ty::Struct {
                            ident: *i1,
                            gen: g1
                                .iter()
                                .zip(g2)
                                .map(|(t1, t2)| {
                                    self.unify(Some(&t1.val), Some(&t2.val))
                                        .map(|t| t.into_spanned(DUMMY))
                                })
                                .collect::<Option<Vec<_>>>()?,
                        })
                    } else {
                        None
                    }
                }
                (Ty::Enum { ident: i1, gen: g1 }, Ty::Enum { ident: i2, gen: g2 }) => {
                    if i1 == i2 {
                        Some(Ty::Struct {
                            ident: *i1,
                            gen: g1
                                .iter()
                                .zip(g2)
                                .map(|(t1, t2)| {
                                    self.unify(Some(&t1.val), Some(&t2.val))
                                        .map(|t| t.into_spanned(DUMMY))
                                })
                                .collect::<Option<Vec<_>>>()?,
                        })
                    } else {
                        None
                    }
                }
                (Ty::Path(p1), Ty::Path(p2)) => {
                    if p1 == p2 {
                        Some(Ty::Path(p1.clone()))
                    } else {
                        None
                    }
                }
                (Ty::Ptr(t1), Ty::Ptr(t2)) => {
                    Some(Ty::Ptr(box self.unify(Some(&t1.val), Some(&t2.val))?.into_spanned(DUMMY)))
                }
                (Ty::Ref(t1), Ty::Ref(t2)) => {
                    Some(Ty::Ref(box self.unify(Some(&t1.val), Some(&t2.val))?.into_spanned(DUMMY)))
                }
                (Ty::String, Ty::String) => Some(Ty::String),
                (Ty::Int, Ty::Int) => Some(Ty::Int),
                (Ty::Char, Ty::Char) => Some(Ty::Char),
                (Ty::Float, Ty::Float) => Some(Ty::Float),
                (Ty::Bool, Ty::Bool) => Some(Ty::Bool),
                (Ty::Void, Ty::Void) => Some(Ty::Void),
                (
                    Ty::Func { ident: i1, ret: r1, params: p1 },
                    Ty::Func { ident: i2, ret: r2, params: p2 },
                ) => todo!(),
                _ => {
                    println!("mismatched inference types");
                    None
                }
            },
            (Some(t), None) => {
                println!("THIS SHOULD NOT HAPPEN");
                Some(t.clone())
            }
            (None, Some(t)) => Some(t.clone()),
            (None, None) => None,
        }
    }

    fn infer_rhs_field(&mut self, lhs_ty: &Ty, rhs: &'ast Expression, parent: &'ast Expression) {
        fn fetch_fields<'a>(
            lhs_ty: &Ty,
            span: Range,
            tcxt: &mut TyCheckRes<'a, '_>,
        ) -> Option<&'a Struct> {
            match lhs_ty {
                Ty::Struct { ident, gen } => tcxt.name_struct.get(ident).copied(),
                Ty::Path(path) => tcxt.name_struct.get(&path.local_ident()).copied(),
                Ty::Ptr(_) => todo!(),
                Ty::Ref(_) => todo!(),
                Ty::Generic { ident, bound } => None,
                Ty::Array { size, ty } => todo!(),
                _ => {
                    tcxt.errors.write().push(Error::error_with_span(
                        tcxt,
                        span,
                        &format!("[E0tc] invalid field accesor target"),
                    ));
                    tcxt.error_in_current_expr_tree.set(true);
                    None
                }
            }
        }
        let fields = if let Some(s) = fetch_fields(lhs_ty, parent.span, self.tcxt) {
            &s.fields
        } else {
            return;
        };
        match &rhs.val {
            Expr::Ident(id) => {
                if let Some(field_ty) = fields.iter().find_map(|f| if f.ident == *id { Some(f.ty.val.clone()) } else { None }) {
                    self.tcxt.expr_ty.insert(rhs, field_ty.clone());
                    // We are at the end of the field access expression so, the whole expr resolves to this
                    self.tcxt.expr_ty.insert(parent, field_ty);
                } else {
                    self.tcxt.errors.write().push(Error::error_with_span(
                        self.tcxt,
                        parent.span,
                        &format!("[E0i] could not infer type of deref expression"),
                    ));
                    self.tcxt.error_in_current_expr_tree .set(true);
                }
            },
            Expr::Deref { indir, expr } => {
                self.infer_rhs_field(lhs_ty, &**expr, parent);

                if let Some(refed_ty) = self.tcxt.expr_ty.get(&**expr) {
                    let mut res_ty = refed_ty.clone();
                    for _ in 0..*indir {
                        res_ty = Ty::Ref(box res_ty.into_spanned(DUMMY));
                    }

                    self.tcxt.expr_ty.insert(expr, res_ty.clone());
                    // We are at the end of the field access expression so, the whole expr resolves to this
                    self.tcxt.expr_ty.insert(parent, res_ty);
                } else {
                    self.tcxt.errors.write().push(Error::error_with_span(
                        self.tcxt,
                        parent.span,
                        &format!("[E0i] could not infer type of deref expression"),
                    ));
                    self.tcxt.error_in_current_expr_tree .set(true);
                }
            }
            Expr::Array { ident, exprs } => {
                if let arr @ Some(ty @ Ty::Array { .. }) = fields
                    .iter()
                    .find_map(|f| if f.ident == *ident { Some(&f.ty.val) } else { None })
                {
                    let dim = ty.array_dim();
                    if exprs.len() != dim {
                        self.tcxt.errors.write().push(Error::error_with_span(
                            self.tcxt,
                            parent.span,
                            &format!("[E0i] mismatched array dimension\nfound `{}` expected `{}`", exprs.len(), dim),
                        ));
                        self.tcxt.error_in_current_expr_tree .set(true);

                    } else {
                        self.tcxt.expr_ty.insert(rhs, ty.clone());
                    // We are at the end of the field access expression so, the whole expr resolves to this
                        self.tcxt.expr_ty.insert(parent, ty.clone());
                    }
                } else {
                    self.tcxt.errors.write().push(Error::error_with_span(
                        self.tcxt,
                        parent.span,
                        &format!("[E0i] ident `{}` not array", ident),
                    ));
                    self.tcxt.error_in_current_expr_tree .set(true);
                }
            },
            Expr::FieldAccess { lhs, rhs: inner } => {
                let id = lhs.val.as_ident();
                if let Some(t @ Ty::Struct { ident, .. }) = &self.tcxt.type_of_ident(id, inner.span).and_then(|t| t.resolve()) {
                    self.infer_rhs_field(&t, &**inner, parent);

                    if let Some(accty) = self.tcxt.expr_ty.get(&**inner).cloned() {
                        self.tcxt.expr_ty.insert(&**inner, accty);
                    } else {
                        self.tcxt.errors.write().push(Error::error_with_span(
                            self.tcxt,
                            parent.span,
                            &format!("[E0tc] ident `{}` not struct", ident),
                        ));
                        self.tcxt.error_in_current_expr_tree .set(true);
                    }
                } else {
                    self.tcxt.errors.write().push(Error::error_with_span(
                        self.tcxt,
                        inner.span,
                        &format!("[E0i] tried to access field of non struct"),
                    ));
                    self.tcxt.error_in_current_expr_tree .set(true);
                }
            },
            Expr::AddrOf(_)
            // invalid lval
            | Expr::Urnary { .. }
            | Expr::Binary { .. }
            | Expr::Parens(_)
            | Expr::Call { .. }
            | Expr::TraitMeth { .. }
            | Expr::StructInit { .. }
            | Expr::EnumInit { .. }
            | Expr::ArrayInit { .. }
            | Expr::Value(_) => {
                self.tcxt.errors.write().push(
                    Error::error_with_span(self.tcxt, parent.span, "[E0i] invalid lValue")
                );
                self.tcxt.error_in_current_expr_tree .set(true);
            }
        }
    }
}

impl<'ast> Visit<'ast> for TypeInfer<'_, 'ast, '_> {
    fn visit_stmt(&mut self, stmt: &'ast Statement) {
        match &stmt.val {
            Stmt::Const(_) => todo!(),
            // TODO: deal with user explicitly provided types
            Stmt::Assign { lval, rval, ty: given_ty, is_let } => {
                self.visit_expr(rval);
                let ty = if let Some(t) =
                    self.tcxt.expr_ty.get(rval).or_else(|| given_ty.as_ref().map(|t| &t.val))
                {
                    t.clone()
                } else {
                    self.tcxt.errors.write().push(Error::error_with_span(
                        self.tcxt,
                        lval.span,
                        &format!("[E0i] variable not found `{}`", lval.val.as_ident()),
                    ));
                    self.tcxt.error_in_current_expr_tree.set(true);
                    return;
                };

                // Set after walking the right side trees
                self.tcxt.set_record_used_vars(!is_let);

                // @cleanup: this is duplicated in `TypeCheck::visit_var`
                if let Some(fn_id) = self.tcxt.curr_fn {
                    if *is_let {
                        // let node = Node::Func(fn_id);
                        // let mut stack = if self.tcxt.generic_res.has_generics(&node) {
                        //     vec![node]
                        // } else {
                        //     vec![]
                        // };
                        // self.tcxt.generic_res.collect_generic_usage(
                        //     &ty,
                        //     self.tcxt.unique_id(),
                        //     0,
                        //     &[TyRegion::Expr(&lval.val)],
                        //     &mut stack,
                        // );

                        // TODO: match this out so we know its an lval or just wait for later
                        // when that's checked by `StmtCheck`
                        let ident = lval.val.as_ident();
                        self.tcxt.var_func.unsed_vars.insert(
                            ScopedName::func_scope(fn_id, ident, lval.span.file_id),
                            (lval.span, Cell::new(false)),
                        );

                        if self
                            .tcxt
                            .var_func
                            .func_refs
                            .entry(fn_id)
                            .or_default()
                            .insert(ident, ty)
                            .is_some()
                        {
                            self.tcxt.errors.write().push(Error::error_with_span(
                                self.tcxt,
                                ident.span(),
                                &format!("[E0i] duplicate variable name `{}`", ident),
                            ));
                            self.tcxt.error_in_current_expr_tree.set(true);
                        }
                    } else if let Some(lhs) = self
                        .tcxt
                        .type_of_ident(lval.val.as_ident(), lval.span)
                        .and_then(|t| resolve_ty(self.tcxt, lval, Some(&t)))
                    {
                        if !ty.is_ty_eq(&lhs) {
                            self.tcxt.errors.write().push(Error::error_with_span(
                                self.tcxt,
                                rval.span,
                                &format!(
                                    "[E0i] assigned to wrong type\nfound `{}` expected `{}`",
                                    ty, lhs,
                                ),
                            ));
                            self.tcxt.error_in_current_expr_tree.set(true);
                        }
                    }
                    // For any assignment we need to know the type of the lvalue, this is because
                    // each node is unique in the expr -> type map
                    self.visit_expr(lval);
                }
            }
            Stmt::AssignOp { lval, rval, op } => {
                self.visit_expr(rval);
                let rty = self.tcxt.expr_ty.get(rval);

                // We must know the type of `lvar` now
                let lty = self.tcxt.type_of_ident(lval.val.as_ident(), lval.span);
                if lty.is_none() {
                    self.tcxt.errors.write().push(Error::error_with_span(
                        self.tcxt,
                        lval.span,
                        &format!("[E0i] undeclared variable name `{}`", lval.val.as_ident()),
                    ));
                    self.tcxt.error_in_current_expr_tree.set(true);
                }

                if let Some(unified) = fold_ty(
                    self.tcxt,
                    lty.as_ref(),
                    rty,
                    op,
                    to_rng(lval.span.start..rval.span.end, {
                        debug_assert!(lval.span.file_id == rval.span.file_id);
                        lval.span.file_id
                    }),
                ) {
                    self.tcxt.expr_ty.insert(rval, unified);
                }
                // For any assignment we need to know the type of the lvalue, this is because each
                // node is unique in the expr -> type map
                self.visit_expr(lval);
            }
            Stmt::Call(expr) => self.visit_expr(expr),
            Stmt::TraitMeth(expr) => self.visit_expr(expr),
            Stmt::If { cond, blk, els } => {
                self.visit_expr(cond);
                // DO NOT WALK DEEPER the calling method is doing the walking
            }
            Stmt::While { cond, blk } => {
                self.visit_expr(cond);
                // DO NOT WALK DEEPER the calling method is doing the walking
            }
            Stmt::Match { expr: ex, arms } => {
                self.visit_expr(ex);
                // DO NOT WALK DEEPER the calling method is doing the walking
            }
            Stmt::Ret(expr) => {
                self.visit_expr(expr);
            }
            Stmt::Exit => {}
            Stmt::Block(blk) => {
                for stmt in &blk.stmts {
                    self.visit_stmt(stmt);
                }
            }
        }
    }

    fn visit_expr(&mut self, expr: &'ast Expression) {
        match &expr.val {
            Expr::Ident(ident) => {
                if let Some(ty) = self.tcxt.type_of_ident(*ident, expr.span) {
                    self.tcxt.expr_ty.insert(expr, ty);
                } else {
                    self.tcxt.errors.write().push(Error::error_with_span(
                        self.tcxt,
                        expr.span,
                        &format!("[E0i] no type infered for `{}`", ident),
                    ));
                    self.tcxt.error_in_current_expr_tree.set(true);
                }
            }
            Expr::Deref { indir, expr } => todo!(),
            Expr::AddrOf(ex) => {
                self.visit_expr(ex);
                let exprty = self.tcxt.expr_ty.get(&**ex);

                if let Some(ty) = exprty.cloned() {
                    self.tcxt.expr_ty.insert(expr, Ty::Ptr(box ty.into_spanned(DUMMY)));
                }
            }
            Expr::Array { ident, exprs } => {
                if let Some(ty) = self.tcxt.type_of_ident(*ident, expr.span) {
                    for ex in exprs {
                        self.visit_expr(ex);
                    }
                    if let Some(t) = ty.index_dim(self.tcxt, exprs, expr.span) {
                        self.tcxt.expr_ty.insert(expr, t);
                    }
                } else {
                    self.tcxt.errors.write().push(Error::error_with_span(
                        self.tcxt,
                        expr.span,
                        &format!("[E0i] no type infered for `{}`", ident),
                    ));
                    self.tcxt.error_in_current_expr_tree.set(true);
                }
            }
            Expr::Urnary { op, expr: ex } => {
                self.visit_expr(ex);
                let exprty = self.tcxt.expr_ty.get(&**ex);

                if let Some(ty) = exprty.cloned() {
                    self.tcxt.expr_ty.insert(expr, ty);
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);

                let rhsty = self.tcxt.expr_ty.get(&**rhs);
                let lhsty = self.tcxt.expr_ty.get(&**lhs);

                // println!("FOLD {:?} == {:?}", lhsty, rhsty);

                if let Some(unified) = fold_ty(self.tcxt, lhsty, rhsty, op, expr.span) {
                    self.tcxt.expr_ty.insert(expr, unified);
                }
            }
            Expr::Parens(ex) => {
                self.visit_expr(ex);
                let exprty = self.tcxt.expr_ty.get(&**ex);

                if let Some(ty) = exprty.cloned() {
                    self.tcxt.expr_ty.insert(expr, ty);
                }
            }
            Expr::Call { path, args, type_args } => {
                let mut infered_ty_args = vec![];
                for (idx, arg) in args.iter().enumerate() {
                    self.visit_expr(arg);
                }

                let func = self.tcxt.var_func.name_func.get(&path.segs[0]);
                if let Some(func) = func {
                    if type_args.is_empty() && !func.generics.is_empty() {
                        for (arg, param) in args.iter().zip(&func.params) {
                            if let Some(expr_ty) = self.tcxt.expr_ty.get(&arg) {
                                if param.ty.val.has_generics() {
                                    if let Some(ty_gen_pair) =
                                        peel_out_ty(Some(expr_ty), &param.ty.val)
                                    {
                                        infered_ty_args.push(ty_gen_pair);
                                    }
                                }
                            } else {
                                self.tcxt.errors.write().push(Error::error_with_span(
                                    self.tcxt,
                                    expr.span,
                                    &format!(
                                        "[E0i] no type infered for argument `{}`",
                                        param.ident
                                    ),
                                ));
                                self.tcxt.error_in_current_expr_tree.set(true);
                            }
                        }

                        let last = infered_ty_args.len() - 1;
                        for gen in &func.generics {
                            let idx =
                                infered_ty_args.iter().position(|(_, g)| gen.ident == *g).unwrap();
                            // move idx to the head of the list
                            infered_ty_args.swap(last, idx);
                        }

                        for (ty, _) in infered_ty_args {
                            // SAFETY maybe:
                            //
                            // This will only ever be done on one thread at a time and nothing else
                            // can mutate it, we are the only ones handling `type_args`.
                            unsafe { type_args.push_shared(ty.into_spanned(DUMMY)) };
                        }
                    }

                    let ret_val = self.tcxt.patch_generic_from_path(&func.ret, expr.span).val;
                    // If the function is generic do complicated stuff
                    if ret_val.has_generics() {
                        let mut subed_ty = ret_val;

                        // If there are no explicit type args rely on inference of the arguments
                        if type_args.is_empty() {
                            // Do any of the param generics match the return type
                            let params = func.params.iter().enumerate().filter(|(_, p)| {
                                func.ret
                                    .val
                                    .generics()
                                    .iter()
                                    .any(|g| p.ty.val.generics().contains(g))
                            });

                            for (idx, param) in params {
                                let expr_ty = self.tcxt.expr_ty.get(&args[idx]);
                                if let Some((ty, gen)) = peel_out_ty(expr_ty, &param.ty.val) {
                                    subed_ty.subst_generic(gen, &ty);
                                }
                            }
                        // There are type args to use yay!
                        } else {
                            // Find all the matching generic types so a `fn call<T, U>` is
                            // `call::<int, bool>()` we need to know
                            // which type/generic goes with which
                            let idx_gen = func.generics.iter().enumerate().filter_map(|(i, g)| {
                                if func.ret.val.generics().contains(&&g.ident) {
                                    Some((i, g.ident))
                                } else {
                                    None
                                }
                            });
                            for (idx, gen) in idx_gen {
                                let ty_arg = self
                                    .tcxt
                                    .patch_generic_from_path(&type_args.slice()[idx], expr.span);
                                subed_ty.subst_generic(gen, &ty_arg.val);
                            }
                        }
                        self.tcxt.expr_ty.insert(expr, subed_ty);
                    } else {
                        self.tcxt.expr_ty.insert(expr, func.ret.val.clone());
                    }
                }
            }
            Expr::TraitMeth { trait_, args, type_args } => {
                for (idx, arg) in args.iter().enumerate() {
                    self.visit_expr(arg);
                }

                let opt_imp = self
                    .tcxt
                    .trait_solve
                    .impls
                    .get(trait_)
                    .unwrap()
                    .get(&type_args.iter().map(|t| &t.val).collect::<Vec<_>>());

                if let Some(imp) = opt_imp {
                    self.tcxt.expr_ty.insert(expr, imp.method.ret.val.clone());
                }
            }
            Expr::FieldAccess { lhs, rhs } => {
                self.visit_expr(lhs);
                if let Some(lhs_ty) = self.tcxt.expr_ty.get(&**lhs).cloned() {
                    // `infer_rhs_field` adds the expressions type to type context
                    self.infer_rhs_field(&lhs_ty, &rhs, expr);
                } else {
                    self.tcxt.errors.write().push(Error::error_with_span(
                        self.tcxt,
                        expr.span,
                        &format!("[E0i] no type infered for argument `{}`", lhs.val.as_ident()),
                    ));
                    self.tcxt.error_in_current_expr_tree.set(true);
                }
            }
            Expr::StructInit { path, fields } => {
                let struc = self.tcxt.name_struct.get(&path.segs[0]);
                if let Some(struc) = struc {
                    let gen =
                        struc.generics.iter().map(|g| g.to_type().into_spanned(g.span)).collect();
                    let ident = struc.ident;

                    // TODO remove??
                    for field in fields.iter() {
                        self.visit_expr(&field.init);
                    }

                    self.tcxt.expr_ty.insert(expr, Ty::Struct { ident, gen });
                }
            }
            Expr::EnumInit { path, variant, items } => {
                let enm = self.tcxt.name_enum.get(&path.segs[0]);

                if let Some(enm) = enm {
                    let gen =
                        enm.generics.iter().map(|g| g.to_type().into_spanned(g.span)).collect();
                    let ident = enm.ident;

                    // TODO remove??
                    for arg in items.iter() {
                        self.visit_expr(arg);
                    }

                    self.tcxt.expr_ty.insert(expr, Ty::Enum { ident, gen });
                }
            }
            Expr::ArrayInit { items } => {
                let size = items.len();
                let mut ty = None;
                for ex in items {
                    self.visit_expr(ex);
                    ty = self.unify(ty.as_ref(), self.tcxt.expr_ty.get(ex));
                }
                self.tcxt.expr_ty.insert(
                    expr,
                    Ty::Array { size, ty: box ty.unwrap_or(Ty::Void).into_spanned(DUMMY) },
                );
            }
            Expr::Value(val) => {
                self.tcxt.expr_ty.insert(expr, val.val.to_type());
            }
        }
    }
}

/// Return the concrete type and the matching generic.
crate fn peel_out_ty(exty: Option<&Ty>, has_gen: &Ty) -> Option<(Ty, Ident)> {
    match (exty?, has_gen) {
        (t, Ty::Generic { ident, .. }) => Some((t.clone(), *ident)),
        (Ty::Array { ty: t1, .. }, Ty::Array { ty: t2, .. }) => peel_out_ty(Some(&t1.val), &t2.val),
        (Ty::Struct { ident: i1, gen: g1 }, Ty::Struct { ident: i2, gen: g2 }) if i1 == i2 => {
            g1.iter().zip(g2).find_map(|(a, b)| peel_out_ty(Some(&a.val), &b.val))
        }
        (Ty::Enum { ident: i1, gen: g1 }, Ty::Enum { ident: i2, gen: g2 }) if i1 == i2 => {
            g1.iter().zip(g2).find_map(|(a, b)| peel_out_ty(Some(&a.val), &b.val))
        }
        (Ty::Ptr(t1), Ty::Ptr(t2)) => peel_out_ty(Some(&t1.val), &t2.val),
        (Ty::Ref(t1), Ty::Ref(t2)) => peel_out_ty(Some(&t1.val), &t2.val),
        (Ty::Func { .. }, _) => todo!(),
        _ => None,
    }
}
