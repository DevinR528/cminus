use crate::lir::lower::{Adt, Const, Expr, Func, Impl, Item, MatchArm, Param, Stmt, Trait};

use super::lower::{Block, FieldInit, LValue};

pub trait Visit<'ast>: Sized {
    fn visit_prog(&mut self, items: &'ast [Item]) {
        walk_items(self, items)
    }

    fn visit_decl(&mut self, item: &'ast Item) {
        walk_decl(self, item)
    }

    fn visit_func(&mut self, func: &'ast Func) {
        walk_func(self, func)
    }

    fn visit_trait(&mut self, item: &'ast Trait) {
        walk_trait(self, item)
    }

    fn visit_impl(&mut self, item: &'ast Impl) {
        walk_impl(self, item)
    }

    fn visit_adt(&mut self, _adt: &'ast Adt) {}

    fn visit_var(&mut self, _var: &'ast Const) {}

    fn visit_params(&mut self, _params: &[Param]) {}

    fn visit_lval(&mut self, lval: &LValue) {
        walk_lval(self, lval)
    }

    fn visit_stmt(&mut self, stmt: &'ast Stmt) {
        walk_stmt(self, stmt)
    }

    fn visit_match_arm(&mut self, arms: &'ast [MatchArm]) {
        walk_match_arm(self, arms)
    }

    fn visit_expr(&mut self, expr: &'ast Expr) {
        walk_expr(self, expr)
    }
}

crate fn walk_items<'ast, V: Visit<'ast>>(visit: &mut V, items: &'ast [Item]) {
    for item in items {
        visit.visit_decl(item);
    }
}

crate fn walk_decl<'ast, V: Visit<'ast>>(visit: &mut V, item: &'ast Item) {
    match item {
        Item::Func(func) => {
            visit.visit_func(func);
        }
        Item::Const(var) => {
            visit.visit_var(var);
        }
        Item::Trait(trait_) => visit.visit_trait(trait_),
        Item::Impl(imp) => visit.visit_impl(imp),
        Item::Adt(struc) => visit.visit_adt(struc),
    }
}

crate fn walk_func<'ast, V: Visit<'ast>>(visit: &mut V, func: &'ast Func) {
    let Func { ident: _, params, stmts, ret: _, generics: _ } = func;
    // visit.visit_ident(ident);
    // visit.visit_generics(generics);
    visit.visit_params(params);
    // visit.visit_ty(ret);
    for stmt in stmts {
        visit.visit_stmt(stmt);
    }
}

crate fn walk_trait<'ast, V: Visit<'ast>>(_visit: &mut V, tr: &'ast Trait) {
    let Trait { ident: _, method: _, generics: _ } = tr;
    // visit.visit_ident(ident);
    // visit.visit_ty(ret);
    // match method {
    //     TraitMethod::Default(f) => visit.visit_func(f),
    //     TraitMethod::NoBody(f) => visit.visit_func(f),
    // }
}

crate fn walk_impl<'ast, V: Visit<'ast>>(visit: &mut V, tr: &'ast Impl) {
    let Impl { ident: _, method, type_arguments: _ } = tr;
    // visit.visit_ident(ident);
    // for ty in type_arguments {
    //     visit.visit_ty(ty);
    // }
    // visit.visit_ty(ret);
    visit.visit_func(method)
}

crate fn walk_lval<'ast, V: Visit<'ast>>(_visit: &mut V, lval: &LValue) {
    match lval {
        LValue::Ident { ident: _, ty: _ } => todo!(),
        LValue::Deref { indir: _, expr: _, ty: _ } => todo!(),
        LValue::Array { ident: _, exprs: _, ty: _ } => todo!(),
        LValue::FieldAccess { lhs: _, def: _, rhs: _, field_idx: _ } => todo!(),
    }
}

crate fn walk_match_arm<'ast, V: Visit<'ast>>(visit: &mut V, arms: &'ast [MatchArm]) {
    for MatchArm { pat: _, blk: Block { stmts, .. }, .. } in arms {
        for stmt in stmts {
            visit.visit_stmt(stmt);
        }
    }
}

crate fn walk_stmt<'ast, V: Visit<'ast>>(visit: &mut V, stmt: &'ast Stmt) {
    match stmt {
        Stmt::Const(var) => visit.visit_var(var),
        Stmt::Assign { lval, rval, .. } => {
            // visit.visit_ident(ident);
            visit.visit_lval(lval);
            visit.visit_expr(rval);
        }
        Stmt::Call { expr, def: _ } => {
            for arg in &expr.args {
                visit.visit_expr(arg);
            }
        }
        Stmt::TraitMeth { expr, def: _ } => {
            for arg in &expr.args {
                visit.visit_expr(arg);
            }
        }
        Stmt::If { cond, blk: Block { stmts, .. }, els } => {
            visit.visit_expr(cond);
            for stmt in stmts {
                visit.visit_stmt(stmt);
            }
            if let Some(Block { stmts, .. }) = els {
                for stmt in stmts {
                    visit.visit_stmt(stmt);
                }
            }
        }
        Stmt::While { cond, stmts } => {
            visit.visit_expr(cond);
            for stmt in &stmts.stmts {
                visit.visit_stmt(stmt);
            }
        }
        Stmt::Match { expr, arms, .. } => {
            visit.visit_expr(expr);
            visit.visit_match_arm(arms);
        }
        Stmt::Ret(expr, _ty) => visit.visit_expr(expr),
        Stmt::Exit => {}
        Stmt::Block(Block { stmts, .. }) => {
            for stmt in stmts {
                visit.visit_stmt(stmt);
            }
        }
    }
}

crate fn walk_expr<'ast, V: Visit<'ast>>(visit: &mut V, expr: &'ast Expr) {
    match expr {
        Expr::Ident { .. } => {
            // visit.visit_ident(id)
        }
        Expr::Array { ident: _, exprs, .. } => {
            // visit.visit_ident(ident);
            for expr in exprs {
                visit.visit_expr(expr)
            }
        }
        Expr::Urnary { op: _, expr, .. } => {
            visit.visit_expr(expr);
        }
        Expr::Deref { indir: _, expr, .. } => {
            visit.visit_expr(expr);
        }
        Expr::AddrOf(expr) => {
            visit.visit_expr(expr);
        }
        Expr::Binary { op: _, lhs, rhs, .. } => {
            visit.visit_expr(lhs);
            visit.visit_expr(rhs)
        }
        Expr::Parens(expr) => visit.visit_expr(expr),
        Expr::StructInit { path: _, fields, .. } => {
            for FieldInit { ident: _, init, ty: _ } in fields {
                visit.visit_expr(init);
            }
        }
        Expr::EnumInit { items, .. } => {
            for expr in items {
                visit.visit_expr(expr);
            }
        }
        Expr::ArrayInit { items, .. } => {
            for expr in items {
                visit.visit_expr(expr);
            }
        }
        Expr::FieldAccess { lhs, rhs, .. } => {
            visit.visit_expr(lhs);
            visit.visit_expr(rhs)
        }
        Expr::Call { path: _, args, type_args: _, .. } => {
            for expr in args {
                visit.visit_expr(expr);
            }
        }
        Expr::TraitMeth { trait_: _, args, type_args: _, .. } => {
            for expr in args {
                visit.visit_expr(expr);
            }
        }
        Expr::Value(_) => {
            // visit.visit_value(val);
        }
    }
}
