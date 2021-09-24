use std::collections::{HashMap, HashSet};

use pest::prec_climber::Operator;

use crate::{
    ast::types::{
        BinOp, Block, Decl, Expr, Expression, Field, Func, Param, Statement, Stmt, Struct, Ty,
        Type, UnOp, Val, Value, Var, DUMMY,
    },
    visit::Visit,
};

#[derive(Debug, Default)]
crate struct TyCheckRes<'ast> {
    /// Global variables declared outside of functions.
    global: HashMap<String, Ty>,
    /// The name of the function currently in or `None` if global.
    curr_fn: Option<String>,
    /// The variables in functions, mapped fn name -> variables.
    func_refs: HashMap<String, HashMap<String, Ty>>,
    /// A backwards mapping of variable name -> function name.
    var_func: HashMap<String, String>,
    /// Function name to return type.
    func_ret: HashMap<String, Ty>,
    /// The parameters of a function, fn name -> parameters.
    func_params: HashMap<String, Vec<(String, Ty)>>,
    /// A mapping of expression -> type, this is the main inference table.
    expr_ty: HashMap<&'ast Expression, Ty>,
    /// A mapping of statement to the function it is contained in.
    /// This will check the return value.
    stmt_func: HashMap<&'ast Stmt, String>,
    /// A mapping of struct name to the fields of that struct.
    struct_fields: HashMap<String, Vec<Field>>,
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

    fn visit_adt(&mut self, struc: &'ast Struct) {
        if self.struct_fields.insert(struc.ident.clone(), struc.fields.clone()).is_some() {
            unreachable!("duplicate struct names")
        }
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
            Expr::Array { ident, exprs } => {
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
            Expr::Deref { indir, expr: inner_expr } => {
                self.visit_expr(inner_expr);

                let ty = self.expr_ty.get(&**inner_expr).expect("type for address of");
                let ty = ty.dereference(*indir).expect("a dereferencable type");
                self.expr_ty.insert(expr, ty);
            }
            Expr::AddrOf(inner_expr) => {
                self.visit_expr(inner_expr);

                let ty = self.expr_ty.get(&**inner_expr).expect("type for address of").clone();
                self.expr_ty.insert(expr, Ty::AddrOf(box ty.into_spanned(DUMMY)));
            }
            Expr::Binary { op, lhs, rhs } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);

                let lhs_ty = self.expr_ty.get(&**lhs);
                let rhs_ty = self.expr_ty.get(&**rhs);

                if let Some(ty) = fold_ty(lhs_ty, rhs_ty, op) {
                    // TODO: duplicate expression, should be impossible??
                    if self.expr_ty.insert(expr, ty).is_some() {
                        unimplemented!("NOT SURE TODO")
                    }
                } else {
                    panic!("no type found for bin expr {:?} != {:?}", lhs_ty, rhs_ty)
                }
            }
            Expr::Parens(inner_expr) => {
                self.visit_expr(inner_expr);
                if let Some(ty) = self.expr_ty.get(&**inner_expr).cloned() {
                    if self.expr_ty.insert(expr, ty).is_some() {
                        unimplemented!("NOT SURE TODO")
                    }
                } else {
                    panic!("no type found for paren expr")
                }
            }
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
            Expr::StructInit { name, fields } => {
                // TODO: check fields and
                if self.expr_ty.insert(expr, Ty::Adt(name.clone())).is_some() {
                    unimplemented!("No duplicates")
                }
            }
            Expr::ArrayInit { items } => {
                // TODO: fold array type sort of then add whole type to expr_ty
            }
            Expr::FieldAccess { lhs, rhs } => {
                self.visit_expr(lhs);
                let lhs_ty = self.expr_ty.get(&**lhs);

                let (name, fields) = if let Some(Ty::Adt(name)) = lhs_ty {
                    (name, self.struct_fields.get(name).expect("no struct definition found"))
                } else {
                    unreachable!("left hand side must be struct")
                };

                let field_ty = match &rhs.val {
                    Expr::Ident(ident) | Expr::Array { ident, .. } => fields
                        .iter()
                        .find(|f| &f.ident == ident)
                        .unwrap_or_else(|| panic!("no field {} found for struct {}", ident, name)),
                    // TODO: see below
                    Expr::FieldAccess { lhs, rhs } => todo!("this is special"),
                    _ => unreachable!("access struct with non ident"),
                };

                // TODO: duplicate expression, should be impossible??
                if self.expr_ty.insert(expr, field_ty.ty.val.clone()).is_some() {
                    unimplemented!("NOT SURE TODO")
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
            Stmt::Assign { deref, ident, expr } => {
                if let Some(global_ty) = self.tyck.global.get(ident) {
                    if self.tyck.expr_ty.get(expr) != Some(global_ty) {
                        panic!("global type mismatch")
                    }
                } else if let Some(var_ty) =
                    self.tyck.var_func.get(ident).and_then(|name| {
                        self.tyck.func_refs.get(name).and_then(|vars| vars.get(ident))
                    })
                {
                    if self.tyck.expr_ty.get(expr) != var_ty.dereference(*deref).as_ref() {
                        println!("{:?}", self.tyck.expr_ty.get(expr));
                        panic!("variable type mismatch {:?}", var_ty);
                    }
                } else {
                    panic!("assign to undeclared variable")
                }
            }
            Stmt::ArrayAssign { deref, ident, exprs } => {}
            Stmt::FieldAssign { deref, access, expr } => {}
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
        // TODO: deal with structs expr will need field access
        (Ty::Adt(_), _) => todo!(""),
        // TODO: we should NOT get here (I think...)??
        (Ty::AddrOf(_), _) => todo!("{:?} {:?}", lhs?, rhs?),
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
