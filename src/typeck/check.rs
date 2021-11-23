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
        generic::{Node, TyRegion},
        TyCheckRes,
    },
    visit::Visit,
};

//
//
//
// All the following is used for actual type checking after the collection phase.

#[derive(Debug)]
crate struct StmtCheck<'v, 'ast, 'input> {
    crate tcxt: &'v mut TyCheckRes<'ast, 'input>,
}

impl<'ast> StmtCheck<'_, 'ast, '_> {
    fn check_assignment(
        &mut self,
        lval: &'ast Expression,
        rval: &'ast Expression,
        span: Range,
        is_let: bool,
    ) {
        let orig_lty = lvalue_type(self.tcxt, lval, span);
        let lval_ty = resolve_ty(self.tcxt, lval, orig_lty.as_ref());

        let mut stack = if let Some((def, ident)) = self
            .tcxt
            .curr_fn
            .as_ref()
            .and_then(|f| Some((self.tcxt.var_func.name_func.get(f)?, *f)))
        {
            if def.generics.is_empty() {
                vec![]
            } else {
                vec![Node::Func(ident)]
            }
        } else {
            vec![]
        };
        collect_enum_generics(self.tcxt, lval_ty.as_ref(), &rval.val, &mut stack);

        let orig_rty = self.tcxt.expr_ty.get(rval);
        let mut rval_ty = resolve_ty(self.tcxt, rval, orig_rty);

        check_used_enum_generics(
            self.tcxt,
            lval_ty.as_ref(),
            rval_ty.as_mut(),
            rval.span,
            &rval.val,
        );

        coercion(lval_ty.as_ref(), rval_ty.as_mut());

        if self.tcxt.errors.is_poisoned() {
            return;
        }

        if !lval_ty.as_ref().is_ty_eq(&rval_ty.as_ref()) {
            self.tcxt.errors.push_error(Error::error_with_span(
                self.tcxt,
                span,
                &format!(
                    "[E0tc] assign to expression of wrong type\nfound `{}` expected `{}`",
                    orig_rty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                    orig_lty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                ),
            ));
        } else if let Expr::Ident(id) = &lval.val {
            if let Expr::Value(val) = &rval.val {
                // TODO: I don't remember what consts does in const folding???
                self.tcxt.consts.insert(*id, &val.val);
            }
        }
    }
}

impl<'ast> Visit<'ast> for StmtCheck<'_, 'ast, '_> {
    fn visit_stmt(&mut self, stmt: &'ast Statement) {
        match &stmt.val {
            Stmt::Const(_) => {}
            Stmt::Assign { lval, rval, is_let, ty: _ty } => {
                self.check_assignment(lval, rval, stmt.span, *is_let)
            }
            Stmt::AssignOp { lval, rval, .. } => {
                self.check_assignment(lval, rval, stmt.span, false)
            }
            Stmt::Call(_expr) => {
                // Hmm we need something here?
            }
            Stmt::TraitMeth(_e) => {
                // TODO:
            }
            Stmt::If { cond, blk: Block { stmts, .. }, els } => {
                let cond_ty =
                    self.tcxt.expr_ty.get(cond).and_then(|t| resolve_ty(self.tcxt, cond, Some(t)));

                // TODO: type coercions :( REMOVE
                if !is_truthy(cond_ty.as_ref()) {
                    self.tcxt.errors.push_error(Error::error_with_span(
                        self.tcxt,
                        stmt.span,
                        "[E0tc] condition of if must be of type bool",
                    ));
                    self.tcxt.errors.poisoned(true);
                }

                for stmt in stmts {
                    self.visit_stmt(stmt);
                }

                if let Some(Block { stmts, .. }) = els {
                    for stmt in stmts {
                        self.visit_stmt(stmt);
                    }
                }
            }
            Stmt::While { cond, blk } => {
                let cond_ty =
                    self.tcxt.expr_ty.get(cond).and_then(|t| resolve_ty(self.tcxt, cond, Some(t)));

                // TODO: type coercions :( REMOVE
                if !is_truthy(cond_ty.as_ref()) {
                    self.tcxt.errors.push_error(Error::error_with_span(
                        self.tcxt,
                        stmt.span,
                        &format!(
                            "condition of while must be of truthy, got `{}`",
                            cond_ty.map_or("<unknown>".to_owned(), |t| t.to_string())
                        ),
                    ));
                    self.tcxt.errors.poisoned(true);
                }
                for stmt in &blk.stmts {
                    self.visit_stmt(stmt);
                }
            }
            Stmt::Match { expr, arms } => {
                let match_ty = resolve_ty(self.tcxt, expr, self.tcxt.expr_ty.get(expr));

                // TODO: handle array
                match match_ty.as_ref().unwrap() {
                    Ty::Array { size: _, ty: _ } => todo!(),
                    Ty::Enum { ident, gen: _ } => {
                        let mut bound_vars = HashMap::default();
                        for arm in arms {
                            check_pattern_type(
                                self.tcxt,
                                &arm.pat.val,
                                match_ty.as_ref(),
                                arm.span,
                                &mut bound_vars,
                            );

                            let fn_name = self
                                .tcxt
                                .var_func
                                .get_fn_by_span(stmt.span)
                                .expect("in a function");

                            // Add the bound locals if any
                            for (variable, ty) in &bound_vars {
                                self.tcxt
                                    .var_func
                                    .func_refs
                                    .entry(fn_name)
                                    .or_default()
                                    .insert(*variable, ty.clone());
                            }

                            for stmt in &arm.blk.stmts {
                                self.tcxt.visit_stmt(stmt);
                                // self.visit_stmt(stmt);
                            }

                            // TODO: I need to deal with this some way or I will have ghost vars
                            //

                            // // Remove the bound locals after the arm leaves scope
                            // for (id, _) in bound_vars.drain_filter(|_, _| true) {
                            //     self.tcxt
                            //         .var_func
                            //         .func_refs
                            //         .get_mut(&fn_name)
                            //         .map(|map| map.remove(&id));
                            // }
                        }
                    }
                    Ty::Int => {
                        let mut bound_vars = HashMap::default();
                        for arm in arms {
                            check_pattern_type(
                                self.tcxt,
                                &arm.pat.val,
                                match_ty.as_ref(),
                                arm.span,
                                &mut bound_vars,
                            );
                            let fn_name = self
                                .tcxt
                                .var_func
                                .get_fn_by_span(stmt.span)
                                .expect("in a function");

                            // Add the bound locals if any
                            for (variable, ty) in &bound_vars {
                                self.tcxt
                                    .var_func
                                    .func_refs
                                    .entry(fn_name)
                                    .or_default()
                                    .insert(*variable, ty.clone());
                            }

                            // println!("{} {:?} {}", fn_name, bound_vars, arm);

                            for stmt in &arm.blk.stmts {
                                self.tcxt.visit_stmt(stmt);
                                // self.visit_stmt(stmt);
                            }

                            // Remove the bound locals after the arm leaves scope
                            for (id, _) in bound_vars.drain_filter(|_, _| true) {
                                self.tcxt
                                    .var_func
                                    .func_refs
                                    .get_mut(&fn_name)
                                    .map(|map| map.remove(&id));
                            }
                        }
                    }
                    Ty::Char => todo!(),
                    Ty::Float => todo!(),
                    Ty::Bool => todo!(),
                    _ => panic!(
                        "{}",
                        Error::error_with_span(
                            self.tcxt,
                            stmt.span,
                            &format!(
                                "[E0tc] not a valid match type found: `{}`",
                                match_ty.map_or("<unknown>".to_owned(), |t| t.to_string())
                            ),
                        )
                    ),
                }
            }
            Stmt::Ret(expr) => {
                let mut ret_ty =
                    resolve_ty(self.tcxt, expr, self.tcxt.expr_ty.get(expr)).map(|t| {
                        self.tcxt.patch_generic_from_path(&t.into_spanned(DUMMY), stmt.span).val
                    });

                let func_ret_ty = self.tcxt.var_func.get_fn_by_span(expr.span).and_then(|fname| {
                    self.tcxt.var_func.func_return.insert(fname);
                    self.tcxt.var_func.name_func.get(&fname).map(|f| f.ret.val.clone())
                });

                let mut stack = if let Some((def, ident)) = self
                    .tcxt
                    .curr_fn
                    .as_ref()
                    .and_then(|f| Some((self.tcxt.var_func.name_func.get(f)?, f)))
                {
                    if def.generics.is_empty() {
                        vec![]
                    } else {
                        vec![Node::Func(*ident)]
                    }
                } else {
                    vec![]
                };
                collect_enum_generics(self.tcxt, ret_ty.as_ref(), &expr.val, &mut stack);
                check_used_enum_generics(
                    self.tcxt,
                    func_ret_ty.as_ref(),
                    ret_ty.as_mut(),
                    expr.span,
                    &expr.val,
                );

                if !ret_ty.as_ref().is_ty_eq(&func_ret_ty.as_ref()) {
                    self.tcxt.errors.push_error(Error::error_with_span(
                        self.tcxt,
                        stmt.span,
                        &format!(
                            "[E0tc] wrong return type\nfound `{}` expected `{}`",
                            ret_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                            func_ret_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                        ),
                    ));
                }
            }
            Stmt::Exit => {
                let func_ret_ty =
                    self.tcxt.var_func.get_fn_by_span(stmt.span).and_then(|fname| {
                        self.tcxt.var_func.name_func.get(&fname).map(|f| &f.ret.val)
                    });
                if !func_ret_ty.is_ty_eq(&Some(&Ty::Void)) {
                    self.tcxt.errors.push_error(Error::error_with_span(
                        self.tcxt,
                        stmt.span,
                        &format!(
                            "[E0tc] return type must be void `{}`",
                            func_ret_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                        ),
                    ));
                }
            }
            Stmt::Block(Block { stmts, .. }) => {
                for stmt in stmts {
                    self.visit_stmt(stmt);
                }
            }
            Stmt::AssignOp { lval, rval, op } => todo!(),
        }
    }
}

/// TODO: remove coercion
crate fn is_truthy(ty: Option<&Ty>) -> bool {
    if let Some(t) = ty {
        match t {
            Ty::Ptr(_) | Ty::Ref(_) | Ty::String | Ty::Int | Ty::Char | Ty::Float | Ty::Bool => {
                true
            }
            _ => false,
        }
    } else {
        false
    }
}

crate fn coercion(lhs: Option<&Ty>, rhs: Option<&mut Ty>) -> Option<()> {
    match lhs? {
        Ty::Int => match rhs? {
            r @ Ty::Float => {
                *r = Ty::Int;
            }
            r @ Ty::Bool => {
                *r = Ty::Int;
            }
            _ => return None,
        },
        Ty::Float => match rhs? {
            r @ Ty::Int => {
                *r = Ty::Float;
            }
            r @ Ty::Bool => {
                *r = Ty::Float;
            }
            _ => return None,
        },
        Ty::Bool => match rhs? {
            _r @ Ty::Int => {
                // anything but 0 is true ..
            }
            _r @ Ty::Float => {
                // anything but 0 is true ..
            }
            _ => return None,
        },
        Ty::Ptr(_) => match rhs? {
            _r @ Ty::Int => {
                // pointer maths
            }
            _ => return None,
        },
        Ty::Ref(_) => todo!(),
        Ty::Generic { ident: _, bound: _ } => return None,
        // TODO: char has no coercion as of now
        // array has no coercion
        _ => return None,
    }
    Some(())
}

/// Fill the unused generic types if a variant is missing some.
///
/// `enum result<int, string> foo = result::error("blah");` is an example of generic args that
/// need to be filled, the expression would be typed as `result<string>`.
fn check_used_enum_generics(
    tcxt: &TyCheckRes<'_, '_>,
    lty: Option<&Ty>,
    rty: Option<&mut Ty>,
    span: Range,
    rexpr: &Expr,
) {
    let dumb = rty.as_ref().map(|x| (*x).clone());
    let _: Option<()> = try {
        if let (Ty::Enum { ident, gen }, Ty::Enum { ident: _rid, gen: rgen }) = (lty?, rty?) {
            // Oops we don't collect anything, if we continue after here the inner `else` panics
            if gen.is_empty() && rgen.is_empty() {
                return;
            }
            let def = tcxt.name_enum.get(ident)?;
            if let Expr::EnumInit { variant, .. } = rexpr {
                let var = def.variants.iter().find(|v| v.ident == *variant)?;
                let pos = def
                    .generics
                    .iter()
                    .enumerate()
                    // Find the generic type position that this variant needs to satisfy
                    // `result<int, char> foo = result::error('c');` works because
                    // the generic types of result are `generics = [int, char]` where
                    // `error` variant generic type index is 1.
                    .filter(|(_, g)| var.types.iter().any(|t| g.is_ty_eq(&t.val)))
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>();

                if !pos.is_empty()
                    && pos.iter().all(|idx| rgen.iter().any(|t| gen.get(*idx).is_ty_eq(&Some(t))))
                {
                    *rgen = gen.clone();
                } else {
                    panic!(
                        "{}",
                        Error::error_with_span(
                            tcxt,
                            span,
                            &format!(
                                "[E0tc] enum `{}::{}` found with wrong items \nfound `{}` expected `{}`",
                                ident,
                                variant,
                                dumb.map_or("<unknown>".to_owned(), |t| t.to_string()),
                                lty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                            ),
                        )
                    );
                }
            } else {
                return;
            }
        } else {
            return;
        };
    };
}

/// If variants with generics are not constructed the generic parameters are never resolved,
/// this will collect them based on the generic arguments in the type def.
fn collect_enum_generics<'ast>(
    tcxt: &mut TyCheckRes<'ast, '_>,
    lty: Option<&Ty>,
    expr: &'ast Expr,
    stack: &mut Vec<Node>,
) {
    let _: Option<()> = try {
        if let Ty::Enum { ident, gen } = lty? {
            stack.push(Node::Enum(ident.to_owned()));
            let gen_param_id = tcxt.unique_id();
            for (idx, gen_arg) in gen.iter().enumerate() {
                tcxt.generic_res.collect_generic_usage(
                    &gen_arg.val,
                    gen_param_id,
                    idx,
                    &[TyRegion::Expr(expr)],
                    stack,
                );
            }
        } else {
            return;
        };
    };
}

/// Type check the patterns of a match arm.
///
/// Panic's with a good compiler error if types do not match.
fn check_pattern_type(
    tcxt: &mut TyCheckRes<'_, '_>,
    pat: &Pat,
    ty: Option<&Ty>,
    span: Range,
    bound_vars: &mut HashMap<Ident, Ty>,
) {
    let matcher_ty = if let Some(t) = ty {
        t
    } else {
        tcxt.errors.push_error(Error::error_with_span(
            tcxt,
            span,
            &format!("[E0tc] unknown pattern ident found `{}`", pat),
        ));
        tcxt.errors.poisoned(true);
        return;
    };
    match matcher_ty {
        Ty::Array { size, ty: t } => match pat {
            Pat::Enum { path, variant, .. } => {
                tcxt.errors.push_error(Error::error_with_span(
                    tcxt,
                    span,
                    &format!("[E0tc] expected array found `{}::{}`", path, variant),
                ));
                tcxt.errors.poisoned(true);
            }
            Pat::Array { size: p_size, items } => {
                if size != p_size {
                    tcxt.errors.push_error(Error::error_with_span(
                        tcxt,
                        span,
                        &format!(
                            "[E0tc] found array of different sizes\nexpected `{}` found `{}`",
                            size, p_size
                        ),
                    ));
                    tcxt.errors.poisoned(true);
                }
                for item in items {
                    check_pattern_type(tcxt, &item.val, Some(&t.val), span, bound_vars);
                }
            }
            Pat::Bind(bind) => match bind {
                Binding::Wild(id) => {
                    bound_vars.insert(*id, ty.cloned().unwrap());
                }
                Binding::Value(val) => {
                    tcxt.errors.push_error(Error::error_with_span(
                        tcxt,
                        span,
                        &format!("[E0tc] expected array found `{}`", val),
                    ));
                    tcxt.errors.poisoned(true);
                }
            },
        },
        Ty::Struct { ident: _, gen: _ } => todo!(),
        Ty::Enum { ident, gen } => {
            let enm = tcxt.name_enum.get(ident).expect("matched undefined enum");
            match pat {
                Pat::Enum { path, variant, items, .. } => {
                    if !(path.segs.len() == 1 && (*ident) == path.segs[0]) {
                        tcxt.errors.push_error(Error::error_with_span(
                            tcxt,
                            span,
                            &format!(
                                "[E0tc] no enum variant `{}::{}` found for `{}`",
                                path, variant, ident
                            ),
                        ));
                        tcxt.errors.poisoned(true);
                        return;
                    }

                    let var_ty =
                        if let Some(var) = enm.variants.iter().find(|v| v.ident == *variant) {
                            var
                        } else {
                            tcxt.errors.push_error(Error::error_with_span(
                                tcxt,
                                span,
                                &format!(
                                    "[E0tc] no enum variant `{}::{}` found for `{}`",
                                    path, variant, ident
                                ),
                            ));
                            tcxt.errors.poisoned(true);
                            return;
                        };

                    for (idx, it) in items.iter().enumerate() {
                        let var_ty = var_ty.types.get(idx).map(|t| {
                            if let Ty::Generic { .. } = &t.val {
                                &gen[idx].val
                            } else {
                                &t.val
                            }
                        });

                        check_pattern_type(tcxt, &it.val, var_ty, span, bound_vars);
                    }
                }
                Pat::Array { size: _, items: _ } => todo!(),
                Pat::Bind(bind) => match bind {
                    Binding::Wild(id) => {
                        bound_vars.insert(*id, ty.cloned().unwrap());
                    }
                    Binding::Value(val) => {
                        tcxt.errors.push_error(Error::error_with_span(
                            tcxt,
                            span,
                            &format!("[E0tc] expected enum found `{}`", val),
                        ));
                        tcxt.errors.poisoned(true);
                    }
                },
            }
        }
        Ty::String => check_val_pat(tcxt, pat, ty, "string", span, bound_vars),
        Ty::Float => check_val_pat(tcxt, pat, ty, "float", span, bound_vars),
        Ty::Int => check_val_pat(tcxt, pat, ty, "int", span, bound_vars),
        Ty::Char => check_val_pat(tcxt, pat, ty, "char", span, bound_vars),
        Ty::Bool => check_val_pat(tcxt, pat, ty, "bool", span, bound_vars),
        _ => {
            tcxt.errors.push_error(Error::error_with_span(
                tcxt,
                span,
                &format!(
                    "[E0tc] must match a valid enum found: `{}`",
                    ty.map_or("<unknown>".to_owned(), |t| t.to_string())
                ),
            ));
            tcxt.errors.poisoned(true);
        }
    }
}

/// Panic with a good compiler error if the type of `Pat` is not the correct `Binding::Value`.
fn check_val_pat(
    tcxt: &TyCheckRes<'_, '_>,
    pat: &Pat,
    ty: Option<&Ty>,
    expected: &str,
    span: Range,
    bound_vars: &mut HashMap<Ident, Ty>,
) {
    match pat {
        Pat::Enum { path, variant, .. } => {
            tcxt.errors.push_error(Error::error_with_span(
                tcxt,
                span,
                &format!("[E0tc] expected `{}` found `{}::{}`", expected, path, variant),
            ));
            tcxt.errors.poisoned(true);
        }
        Pat::Array { .. } => {
            tcxt.errors.push_error(Error::error_with_span(
                tcxt,
                span,
                &format!("expected `{}` found `{}`", expected, pat),
            ));
            tcxt.errors.poisoned(true);
        }
        Pat::Bind(bind) => match bind {
            Binding::Wild(id) => {
                bound_vars.insert(*id, ty.cloned().unwrap());
            }
            Binding::Value(val) => {
                if Some(&lit_to_type(&val.val)) != ty {
                    tcxt.errors.push_error(Error::error_with_span(
                        tcxt,
                        span,
                        &format!("[E0tc] expected `{}` found `{}`", expected, val),
                    ));
                    tcxt.errors.poisoned(true);
                };
            }
        },
    }
}

crate fn resolve_ty(tcxt: &TyCheckRes<'_, '_>, expr: &Expression, ty: Option<&Ty>) -> Option<Ty> {
    match &expr.val {
        Expr::Deref { indir: _, expr: _ } => ty.and_then(|t| t.resolve()),
        Expr::Array { ident: _, exprs } => ty.and_then(|t| t.index_dim(tcxt, exprs, expr.span)),
        Expr::AddrOf(_) => ty.cloned(),
        Expr::FieldAccess { lhs: _, rhs } => resolve_ty(tcxt, rhs, ty),
        Expr::Ident(_)
        | Expr::Urnary { .. }
        | Expr::Binary { .. }
        | Expr::Parens(_)
        | Expr::Call { .. }
        | Expr::TraitMeth { .. }
        | Expr::StructInit { .. }
        | Expr::EnumInit { .. }
        | Expr::ArrayInit { .. }
        | Expr::Value(_) => ty.cloned(),
    }
}

fn lvalue_type(tcxt: &mut TyCheckRes<'_, '_>, lval: &Expression, stmt_span: Range) -> Option<Ty> {
    let lval_ty = match &lval.val {
        Expr::Ident(_id) => tcxt.expr_ty.get(lval).cloned(),
        Expr::Deref { indir, expr } => {
            lvalue_type(tcxt, expr, stmt_span)
                .map(|t| t.dereference(*indir))
        }
        Expr::Array { ident, exprs } => {
            if let Some(ty @ Ty::Array { .. }) = &tcxt.type_of_ident(*ident, stmt_span) {
                let dim = ty.array_dim();
                if exprs.len() != dim {
                    tcxt.errors.push_error(Error::error_with_span(
                        tcxt,
                        stmt_span,
                        &format!("[E0tc] mismatched array dimension\nfound `{}` expected `{}`", exprs.len(), dim),
                    ));
                    tcxt.errors.poisoned(true);
                    None
                } else {
                    ty.index_dim(tcxt, exprs, stmt_span)
                }
            } else {
                panic!("ICE: todo `{:?}`", lval);
                // TODO: specific error here?
                // None
            }
        },
        Expr::FieldAccess { lhs, rhs } => {
            if let Some(Ty::Struct { ident, .. }) = tcxt.expr_ty.get(&**lhs).and_then(|t| t.resolve()) {
                let fields = tcxt.name_struct.get(&ident).map(|s| s.fields.clone()).unwrap_or_default();

                walk_field_access(tcxt, &fields, rhs)
            } else {
                tcxt.errors.push_error(Error::error_with_span(
                    tcxt,
                    stmt_span,
                    &format!(
                        "[E0tc] no struct `{}` found",
                        tcxt.type_of_ident(lhs.val.as_ident(), lhs.span)
                            .map_or("<unknown>".to_owned(), |t| t.to_string()),
                    ),
                ));
                tcxt.errors.poisoned(true);
                None
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
            panic!(
                "{}",
                Error::error_with_span(tcxt, stmt_span, "[E0tc] invalid lValue")
            )
        }
    };
    lval_ty
}

fn walk_field_access(
    tcxt: &mut TyCheckRes<'_, '_>,
    fields: &[Field],
    expr: &Expression,
) -> Option<Ty> {
    match &expr.val {
        Expr::Ident(id) => fields.iter().find_map(|f| if f.ident == *id { Some(f.ty.val.clone()) } else { None }),
        Expr::Deref { indir, expr } => {
            if let Some(ty) = walk_field_access(tcxt, fields, expr) {
                Some(ty.dereference(*indir))
            } else {
                unreachable!("no type for dereference {:?}", expr)
            }
        }
        Expr::Array { ident, exprs } => {
            if let arr @ Some(ty @ Ty::Array { .. }) = fields
                .iter()
                .find_map(|f| if f.ident == *ident { Some(&f.ty.val) } else { None })
            {
                let dim = ty.array_dim();
                if exprs.len() != dim {
                    tcxt.errors.push_error(Error::error_with_span(
                        tcxt,
                        expr.span,
                        &format!("[E0tc] mismatched array dimension\nfound `{}` expected `{}`", exprs.len(), dim),
                    ));
                    tcxt.errors.poisoned(true);
                    None
                } else {
                    arr.cloned()
                }
            } else {
                tcxt.errors.push_error(Error::error_with_span(
                    tcxt,
                    expr.span,
                    &format!("[E0tc] ident `{}` not array", ident),
                ));
                tcxt.errors.poisoned(true);
                // TODO: specific error here?
                None
            }
        },
        Expr::FieldAccess { lhs, rhs } => {
            let id = lhs.val.as_ident();
            if let Some(Ty::Struct { ident: name, .. }) = tcxt.type_of_ident(id, expr.span).and_then(|t| t.resolve()) {
                // TODO: this is kinda ugly because of the clone but it complains about tcxt otherwise
                // or default not being impl'ed \o/
                let fields = tcxt.name_struct.get(&name).map(|s| s.fields.clone()).unwrap_or_default();
                walk_field_access(tcxt, &fields, rhs)
            } else {
                tcxt.errors.push_error(Error::error_with_span(
                    tcxt,
                    expr.span,
                    &format!("[E0tc] no struct `{}` found", id),
                ));
                tcxt.errors.poisoned(true);
                None
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
            tcxt.errors.push_error(
                Error::error_with_span(tcxt, expr.span, "[E0tc] invalid lValue")
            );
            tcxt.errors.poisoned(true);
            None
        }
    }
}

// TODO: finish the type folding
crate fn fold_ty(
    tcxt: &TyCheckRes<'_, '_>,
    lhs: Option<&Ty>,
    rhs: Option<&Ty>,
    op: &BinOp,
    span: Range,
) -> Option<Ty> {
    let res = match (lhs?, rhs?) {
        (Ty::Int, Ty::Int) => math_ops(tcxt, op, Ty::Int, span),
        (Ty::Float, Ty::Float) => math_ops(tcxt, op, Ty::Float, span),
        // TODO: remove Carr's rules
        (Ty::Int, Ty::Float) => math_ops(tcxt, op, Ty::Float, span),
        (Ty::Float, Ty::Int) => math_ops(tcxt, op, Ty::Float, span),
        (Ty::Char, Ty::Char) => match op {
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => Some(Ty::Bool),
            BinOp::AddAssign | BinOp::SubAssign => {
                panic!(
                    "{}",
                    Error::error_with_span(tcxt, span, "[E0tc] cannot assign operation to a char")
                )
            }
            // HACK: TODO: this is special case for array checking, make this work correctly
            BinOp::Add => Some(Ty::Char),
            _ => {
                panic!(
                    "{}",
                    Error::error_with_span(tcxt, span, "[E0tc] not a legal operation for `char`")
                )
            }
        },
        (Ty::String, Ty::String) => todo!(),
        (Ty::Ptr(t), Ty::Int) => match op {
            BinOp::Add
            | BinOp::Sub
            | BinOp::Mul
            | BinOp::Div
            | BinOp::Rem
            | BinOp::LeftShift
            | BinOp::RightShift
            | BinOp::BitAnd
            | BinOp::BitXor
            | BinOp::BitOr => Some(Ty::Ptr(t.clone())),
            _ => panic!("illegal operation"),
        },
        // swap left and write so the above arm catches
        (l @ Ty::Int, r @ Ty::Ptr(_)) => fold_ty(tcxt, Some(r), Some(l), op, span),
        (Ty::Array { size, ty: t1 }, Ty::Array { size: s, ty: t2 }) if size == s => {
            Some(Ty::Array {
                size: *size,
                ty: box fold_ty(tcxt, Some(&t1.val), Some(&t2.val), op, span)?.into_spanned(DUMMY),
            })
        }
        (Ty::Void, Ty::Void) => Some(Ty::Void),
        (Ty::Bool, Ty::Bool) => match op {
            BinOp::And | BinOp::Or => Some(Ty::Bool),
            _ => panic!("illegal boolean operation"),
        },
        // TODO: deal with structs/enums
        (Ty::Struct { .. }, _) => todo!(""),
        (Ty::Enum { .. }, _) => todo!(""),
        (Ty::Ptr(_), _) => todo!("{:?} {:?}", lhs?, rhs?),
        (r @ Ty::Ref(_), t @ Ty::Ref(_)) => {
            fold_ty(tcxt, r.resolve().as_ref(), t.resolve().as_ref(), op, span)
        }
        (r @ Ty::Ref(_), t) => fold_ty(tcxt, r.resolve().as_ref(), Some(t), op, span),
        (r, t @ Ty::Ref(_)) => fold_ty(tcxt, Some(r), t.resolve().as_ref(), op, span),

        (Ty::Generic { .. }, _) => {
            unreachable!("since no unresolved generic item will ever be in maths")
        }
        (Ty::Func { .. }, _) => unreachable!("Func should never be folded"),
        _ => None,
    };
    // println!("in fold {:?} {:?} == {:?}", lhs, rhs, res);
    res
}

fn math_ops(tcxt: &TyCheckRes<'_, '_>, op: &BinOp, ret_ty: Ty, span: Range) -> Option<Ty> {
    match op {
        BinOp::Add
        | BinOp::Sub
        | BinOp::Mul
        | BinOp::Div
        | BinOp::Rem
        | BinOp::LeftShift
        | BinOp::RightShift
        | BinOp::BitAnd
        | BinOp::BitXor
        | BinOp::BitOr => Some(ret_ty),
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => Some(Ty::Bool),
        // TODO: Carr's rules remove
        BinOp::And | BinOp::Or => Some(Ty::Bool),
        BinOp::AddAssign | BinOp::SubAssign => {
            panic!(
                "{}",
                Error::error_with_span(
                    tcxt,
                    span,
                    "[E0tc] cannot assign to a statement, this isn't Rust ;)"
                )
            )
        }
    }
}

fn lit_to_type(lit: &Val) -> Ty {
    match lit {
        Val::Float(_) => Ty::Float,
        Val::Int(_) => Ty::Int,
        Val::Char(_) => Ty::Char,
        Val::Bool(_) => Ty::Bool,
        Val::Str(_) => Ty::String,
    }
}
