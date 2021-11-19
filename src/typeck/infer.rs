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
        ScopedIdent, TyCheckRes,
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
}

impl<'ast> Visit<'ast> for TypeInfer<'_, 'ast, '_> {
    fn visit_stmt(&mut self, stmt: &'ast Statement) {
        match &stmt.val {
            Stmt::Const(_) => todo!(),
            // TODO: deal with user explicitly provided types
            Stmt::Assign { lval, rval, ty: given_ty, is_let } => {
                self.visit_expr(rval);
                let ty = self
                    .tcxt
                    .expr_ty
                    .get(rval)
                    .or_else(|| given_ty.as_ref().map(|t| &t.val))
                    .unwrap_or_else(|| panic!("{:?}", rval))
                    .clone();

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
                            ScopedIdent::func_scope(fn_id, ident),
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
                            self.tcxt.errors.push(Error::error_with_span(
                                self.tcxt,
                                ident.span(),
                                &format!("[E0i] duplicate variable name `{}`", ident),
                            ));
                            self.tcxt.error_in_current_expr_tree = true;
                        }
                    } else if let Some(lhs) = self
                        .tcxt
                        .type_of_ident(lval.val.as_ident(), lval.span)
                        .and_then(|t| resolve_ty(self.tcxt, lval, Some(&t)))
                    {
                        if !ty.is_ty_eq(&lhs) {
                            self.tcxt.errors.push(Error::error_with_span(
                                self.tcxt,
                                rval.span,
                                &format!(
                                    "[E0i] assigned to wrong type\nfound `{}` expected `{}`",
                                    ty, lhs,
                                ),
                            ));
                            self.tcxt.error_in_current_expr_tree = true;
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
                    self.tcxt.errors.push(Error::error_with_span(
                        self.tcxt,
                        lval.span,
                        &format!("[E0i] undeclared variable name `{}`", lval.val.as_ident()),
                    ));
                    self.tcxt.error_in_current_expr_tree = true;
                }

                if let Some(unified) = fold_ty(
                    self.tcxt,
                    lty.as_ref(),
                    rty,
                    op,
                    to_rng(lval.span.start..rval.span.end),
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
                    self.tcxt.errors.push(Error::error_with_span(
                        self.tcxt,
                        expr.span,
                        &format!("[E0i] no type infered for `{}`", ident),
                    ));
                    self.tcxt.error_in_current_expr_tree = true;
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
                    self.tcxt.errors.push(Error::error_with_span(
                        self.tcxt,
                        expr.span,
                        &format!("[E0i] no type infered for `{}`", ident),
                    ));
                    self.tcxt.error_in_current_expr_tree = true;
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
                for (idx, arg) in args.iter().enumerate() {
                    self.visit_expr(arg);
                }
                let func = self.tcxt.var_func.name_func.get(&path.segs[0]);
                if let Some(func) = func {
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
                                let ty_arg =
                                    self.tcxt.patch_generic_from_path(&type_args[idx], expr.span);
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
            Expr::FieldAccess { lhs, rhs } => todo!(),
            Expr::StructInit { path, fields } => todo!(),
            Expr::EnumInit { path, variant, items } => {
                let enm = self.tcxt.name_enum.get(&path.segs[0]);

                if let Some(enm) = enm {
                    let gen =
                        enm.generics.iter().map(|g| g.to_type().into_spanned(g.span)).collect();
                    let ident = enm.ident;
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
