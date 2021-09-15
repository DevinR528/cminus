use std::collections::{HashMap, HashSet};

use pest::prec_climber::Operator;

use crate::{
    ast::types::{
        BinOp, Block, Decl, Expr, Expression, Func, Param, Statement, Stmt, Ty, Type, UnOp, Val,
        Value, Var, DUMMY,
    },
    visit::Visit,
};

#[derive(Debug, Default)]
crate struct TyCheckRes<'ast> {
    global: HashMap<String, Ty>,
    curr_fn: Option<String>,
    func_refs: HashMap<String, HashMap<String, Ty>>,
    var_func: HashMap<String, String>,
    func_ret: HashMap<String, Ty>,
    func_params: HashMap<String, Vec<(String, Ty)>>,
    expr_ty: HashMap<&'ast Expression, Ty>,
    stmt_func: HashMap<&'ast Stmt, String>,
}

impl TyCheckRes<'_> {
    fn type_of_ident(&self, id: &str) -> Option<Ty> {
        self.var_func
            .get(id)
            .and_then(|f| self.func_refs.get(f).and_then(|s| s.get(id)))
            .or_else(|| self.global.get(id))
            .cloned()
    }
}

impl<'ast> Visit<'ast> for TyCheckRes<'ast> {
    fn visit_func(&mut self, func: &'ast Func) {
        if self.curr_fn.is_none() {
            self.curr_fn = Some(func.ident.clone());
            if self.func_ret.insert(func.ident.clone(), func.ret.val.clone()).is_some() {
                panic!("multiple function return types")
            }
        } else {
            panic!("fn in fn error")
        }

        crate::visit::walk_func(self, func);

        // We have left this functions scope
        self.curr_fn.take();
    }

    fn visit_var(&mut self, var: &Var) {
        if let Some(fn_id) = self.curr_fn.clone() {
            if self
                .func_refs
                .entry(fn_id.clone())
                .or_default()
                .insert(var.ident.clone(), var.ty.val.clone())
                .is_some()
            {
                panic!("function with variable name error")
            }
            if self.var_func.insert(var.ident.clone(), fn_id).is_some() {
                unreachable!("this should be check param names")
            }
        } else if self.global.insert(var.ident.clone(), var.ty.val.clone()).is_some() {
            panic!("global variable name error")
        }

        crate::visit::walk_var(self, var)
    }

    fn visit_params(&mut self, params: &[Param]) {
        for Param { ident, ty, .. } in params {
            if let Some(fn_id) = self.curr_fn.clone() {
                if self
                    .func_refs
                    .entry(fn_id.clone())
                    .or_default()
                    .insert(ident.clone(), ty.val.clone())
                    .is_some()
                {
                    panic!("function with variable name error")
                }

                // Add and check function parameters
                self.func_params
                    .entry(fn_id.clone())
                    .or_default()
                    .push((ident.clone(), ty.val.clone()));
                if self.func_params.len() > params.len() {
                    panic!("function with param error")
                }

                // Insert param names to reverse look-up table
                if self.var_func.insert(ident.clone(), fn_id).is_some() {
                    unreachable!("this should be check param names")
                }
            }
        }
    }

    fn visit_stmt(&mut self, stmt: &'ast Statement) {
        crate::visit::walk_stmt(self, stmt);

        // check the statement after walking incase there were var declarations
        let mut check = StmtCheck { tyck: self };
        check.visit_stmt(stmt);
    }

    fn visit_expr(&mut self, expr: &'ast Expression) {
        match &expr.val {
            Expr::Ident(var_name) => {
                if let Some(ty) = self.type_of_ident(var_name) {
                    if self.expr_ty.insert(expr, ty).is_some() {
                        unimplemented!("NOT SURE TODO")
                    }
                } else {
                    panic!("no type found for ident expr")
                }
            }
            Expr::Array { ident, expr } => {
                if let Some(ty) = self.type_of_ident(ident) {
                    if self.expr_ty.insert(expr, ty).is_some() {
                        unimplemented!("NOT SURE TODO")
                    }
                } else {
                    panic!("no type found for array expr")
                }
            }
            Expr::Urnary { op, expr } => {
                self.visit_expr(expr);
                let ty = self.expr_ty.get(&**expr);
                match op {
                    UnOp::Not => todo!(),
                    UnOp::Inc => todo!(),
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
                let lhs_ty = self.expr_ty.get(&**lhs);
                let rhs_ty = self.expr_ty.get(&**rhs);
                if let Some(ty) = fold_ty(lhs_ty, rhs_ty, op) {
                    if self.expr_ty.insert(expr, ty).is_some() {
                        unimplemented!("NOT SURE TODO")
                    }
                } else {
                    panic!("no type found for array expr {:?} != {:?}", lhs_ty, rhs_ty)
                }
            }
            Expr::Parens(_) => {}
            Expr::Call { ident, args } => {
                if let Some(ret) = self.func_ret.get(ident) {
                    if self.expr_ty.insert(expr, ret.clone()).is_some() {
                        unimplemented!("NOT SURE TODO")
                    }
                } else {
                    panic!("unknown function name")
                }
            }
            Expr::Value(val) => {
                if self
                    .expr_ty
                    .insert(
                        expr,
                        match val.val {
                            Val::Float(_) => Ty::Float,
                            Val::Int(_) => Ty::Int,
                            Val::Char(_) => Ty::Char,
                            Val::Str(_) => Ty::String,
                        },
                    )
                    .is_some()
                {
                    panic!("duplicate value expr {:?}\n{:?}", self.expr_ty, expr)
                }
            }
        }
        // We do NOT call walk_expr here since we recursively walk the exprs
        // when ever found so we have folded the expr types depth first
    }
}

crate struct StmtCheck<'v, 'ast> {
    tyck: &'v TyCheckRes<'ast>,
}

impl<'ast> Visit<'ast> for StmtCheck<'_, 'ast> {
    fn visit_stmt(&mut self, stmt: &Statement) {
        match &stmt.val {
            Stmt::VarDecl(_) => {}
            Stmt::Assign { ident, expr } => {
                if let Some(global_ty) = self.tyck.global.get(ident) {
                    if self.tyck.expr_ty.get(expr) != Some(global_ty) {
                        panic!("global type mismatch")
                    }
                } else if let Some(var_ty) =
                    self.tyck.var_func.get(ident).and_then(|name| {
                        self.tyck.func_refs.get(name).and_then(|vars| vars.get(ident))
                    })
                {
                    if self.tyck.expr_ty.get(expr) != Some(var_ty) {
                        panic!("variable type mismatch")
                    }
                } else {
                    panic!("assign to undeclared variable")
                }
            }
            Stmt::ArrayAssign { ident, expr } => {}
            Stmt::Call { ident, args } => {}
            Stmt::If { cond, blk, els } => {}
            Stmt::While { cond, stmt } => {}
            Stmt::Read(_) => {}
            Stmt::Write { expr } => {}
            Stmt::Ret(_) => {}
            Stmt::Exit => {}
            Stmt::Block(_) => {}
        }
    }
}

fn fold_ty(lhs: Option<&Ty>, rhs: Option<&Ty>, op: &BinOp) -> Option<Ty> {
    match (lhs?, rhs?) {
        (Ty::Int, Ty::Int) => match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => Some(Ty::Int),
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => Some(Ty::Bool),
            _ => panic!("illegal operation"),
        },
        (Ty::Int, _) => None,
        (Ty::Char, Ty::Char) => todo!(),
        (Ty::Char, _) => None,
        (Ty::String, Ty::String) => todo!(),
        (Ty::String, _) => None,
        (Ty::Float, Ty::Float) => todo!(),
        (Ty::Float, _) => None,
        (Ty::Array { size, ty }, Ty::Array { size: s, ty: t }) => Some(Ty::Array {
            size: 0,
            ty: box fold_ty(Some(&ty.val), Some(&ty.val), op)?.into_spanned(DUMMY),
        }),
        (Ty::Array { .. }, _) => None,
        (Ty::Void, Ty::Void) => Some(Ty::Void),
        (Ty::Void, _) => None,
        (Ty::Bool, Ty::Bool) => match op {
            BinOp::And | BinOp::Or => Some(Ty::Bool),
            _ => panic!("illegal boolean operation"),
        },
        (Ty::Bool, _) => None,
    }
}

fn after_op(ty: &Ty, op: BinOp) -> Option<Ty> {
    match op {
        BinOp::Add => todo!(),
        BinOp::Sub => todo!(),
        BinOp::Mul => todo!(),
        BinOp::Div => todo!(),
        BinOp::Rem => todo!(),
        BinOp::And => todo!(),
        BinOp::Or => todo!(),
        BinOp::Eq => todo!(),
        BinOp::Lt => todo!(),
        BinOp::Le => todo!(),
        BinOp::Ne => todo!(),
        BinOp::Ge => todo!(),
        BinOp::Gt => todo!(),
    }
}
