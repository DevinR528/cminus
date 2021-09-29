use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt,
};

use pest::prec_climber::Operator;

use crate::{
    ast::types::{
        BinOp, Block, Decl, Expr, Expression, Field, FieldInit, Func, Param, Range, Statement,
        Stmt, Struct, Ty, Type, TypeEquality, UnOp, Val, Value, Var, DUMMY,
    },
    error::Error,
    visit::Visit,
};

#[derive(Debug, Default)]
crate struct VarInFunction {
    func_spans: BTreeMap<Range, String>,
}

impl VarInFunction {
    fn get(&self, span: Range) -> Option<&str> {
        self.func_spans.iter().find_map(|(k, v)| {
            if k.start <= span.start && k.end >= span.end {
                Some(&**v)
            } else {
                None
            }
        })
    }

    fn insert(&mut self, rng: Range, name: String) -> Option<String> {
        self.func_spans.insert(rng, name)
    }
}

#[derive(Default)]
crate struct TyCheckRes<'ast, 'input> {
    /// The name of the file being checked.
    crate name: &'input str,
    /// The content of the file as a string.
    crate input: &'input str,
    /// Global variables declared outside of functions.
    global: HashMap<String, Ty>,
    /// The name of the function currently in or `None` if global.
    curr_fn: Option<String>,
    /// The variables in functions, mapped fn name -> variables.
    func_refs: HashMap<String, HashMap<String, Ty>>,
    /// A backwards mapping of variable span -> function name.
    var_func: VarInFunction,
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
    /// Errors collected during parsing and type checking.
    errors: Vec<Error<'input>>,
    // TODO:
    /// Unrecoverable error.
    bail: Option<Error<'input>>,
}

impl fmt::Debug for TyCheckRes<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TyCheckResult")
            .field("global", &self.global)
            .field("curr_fn", &self.curr_fn)
            .field("func_refs", &self.func_refs)
            .field("func_params", &self.func_params)
            .field("expr_ty", &self.expr_ty)
            .field("stmt_func", &self.stmt_func)
            .field("struct_fields", &self.struct_fields)
            .finish()
    }
}

impl<'input> TyCheckRes<'_, 'input> {
    crate fn new(input: &'input str, name: &'input str) -> Self {
        Self { name, input, ..Self::default() }
    }

    crate fn report_errors(&self) -> Result<(), ()> {
        if !self.errors.is_empty() {
            for e in &self.errors {
                eprintln!("{}", e)
            }
            return Err(());
        }
        Ok(())
    }

    // TODO: struct declarations
    fn type_of_ident(&self, id: &str, span: Range) -> Option<Ty> {
        self.var_func
            .get(span)
            .and_then(|f| self.func_refs.get(f).and_then(|s| s.get(id)))
            .or_else(|| self.global.get(id))
            .cloned()
    }
}

impl<'ast, 'input> Visit<'ast> for TyCheckRes<'ast, 'input> {
    fn visit_func(&mut self, func: &'ast Func) {
        if self.curr_fn.is_none() {
            self.curr_fn = Some(func.ident.clone());
            if self.var_func.insert(func.span, func.ident.clone()).is_some() {
                self.errors.push(Error::error_with_span(
                    self,
                    func.span,
                    "function takes up same span as other function",
                ));
            }
            if self.func_ret.insert(func.ident.clone(), func.ret.val.clone()).is_some() {
                self.errors.push(Error::error_with_span(
                    self,
                    func.span,
                    "multiple function return types",
                ));
            }
        } else {
            panic!(
                "{}",
                Error::error_with_span(self, func.span, "function defined within function")
            )
        }

        crate::visit::walk_func(self, func);

        // We have left this functions scope
        self.curr_fn.take();
    }

    fn visit_adt(&mut self, struc: &'ast Struct) {
        if self.struct_fields.insert(struc.ident.clone(), struc.fields.clone()).is_some() {
            self.errors.push(Error::error_with_span(self, struc.span, "duplicate struct names"));
        }
    }

    fn visit_var(&mut self, var: &Var) {
        if let Some(fn_id) = self.curr_fn.clone() {
            if self
                .func_refs
                .entry(fn_id)
                .or_default()
                .insert(var.ident.clone(), var.ty.val.clone())
                .is_some()
            {
                self.errors.push(Error::error_with_span(
                    self,
                    var.span,
                    "function with variable name error",
                ));
            }
        } else if self.global.insert(var.ident.clone(), var.ty.val.clone()).is_some() {
            self.errors.push(Error::error_with_span(self, var.span, "global variable name error"));
        }

        crate::visit::walk_var(self, var)
    }

    fn visit_params(&mut self, params: &[Param]) {
        for Param { ident, ty, span } in params {
            if let Some(fn_id) = self.curr_fn.clone() {
                if self
                    .func_refs
                    .entry(fn_id.clone())
                    .or_default()
                    .insert(ident.clone(), ty.val.clone())
                    .is_some()
                {
                    self.errors.push(Error::error_with_span(
                        self,
                        *span,
                        "function with variable name error",
                    ));
                }

                // Add and check function parameters
                self.func_params
                    .entry(fn_id.clone())
                    .or_default()
                    .push((ident.clone(), ty.val.clone()));

                if self.func_params.get(&fn_id).map_or(0, |p| p.len()) > params.len() {
                    unreachable!("to many parameters parsed ICE");
                }
            }
        }
    }

    fn visit_stmt(&mut self, stmt: &'ast Statement) {
        crate::visit::walk_stmt(self, stmt);

        // check the statement after walking incase there were var declarations
        let mut check = StmtCheck { tcxt: self };
        check.visit_stmt(stmt);
    }

    fn visit_expr(&mut self, expr: &'ast Expression) {
        match &expr.val {
            Expr::Ident(var_name) => {
                if let Some(ty) = self.type_of_ident(var_name, expr.span) {
                    if self.expr_ty.insert(expr, ty).is_some() {
                        unimplemented!("NOT SURE TODO")
                    }
                } else {
                    self.errors.push(Error::error_with_span(
                        self,
                        expr.span,
                        "no type found for ident expr",
                    ));
                }
            }
            Expr::Array { ident, exprs } => {
                if let Some(ty) = self.type_of_ident(ident, expr.span) {
                    if self.expr_ty.insert(expr, ty).is_some() {
                        unimplemented!("NOT SURE TODO")
                    }
                } else {
                    self.errors.push(Error::error_with_span(
                        self,
                        expr.span,
                        "no type found for array expr",
                    ));
                }
            }
            Expr::Urnary { op, expr: inner_expr } => {
                self.visit_expr(inner_expr);
                let ty = self.expr_ty.get(&**inner_expr);
                match op {
                    UnOp::Not => {
                        if let Some(Ty::Bool) = ty {
                            self.expr_ty.insert(expr, Ty::Bool);
                        } else {
                            self.errors.push(Error::error_with_span(
                                self,
                                expr.span,
                                "cannot negate non bool type",
                            ));
                        }
                    } // UnOp::Inc => todo!(),
                }
            }
            Expr::Deref { indir, expr: inner_expr } => {
                self.visit_expr(inner_expr);

                let ty = self.expr_ty.get(&**inner_expr).expect("type for dereference");
                let ty = ty.dereference(*indir);

                check_dereference(self, inner_expr, *indir);

                self.expr_ty.insert(expr, ty);
            }
            Expr::AddrOf(inner_expr) => {
                self.visit_expr(inner_expr);

                // TODO: if inner_expr isn't found likely that var isn't declared if ident
                let ty = self.expr_ty.get(&**inner_expr).expect("type for address of").clone();
                self.expr_ty.insert(expr, Ty::Ptr(box ty.into_spanned(DUMMY)));
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
                    self.errors.push(Error::error_with_span(
                        self,
                        expr.span,
                        &format!("no type found for bin expr {:?} != {:?}", lhs_ty, rhs_ty),
                    ));
                }
            }
            Expr::Parens(inner_expr) => {
                self.visit_expr(inner_expr);
                if let Some(ty) = self.expr_ty.get(&**inner_expr).cloned() {
                    if self.expr_ty.insert(expr, ty).is_some() {
                        unimplemented!("NOT SURE TODO")
                    }
                } else {
                    self.errors.push(Error::error_with_span(
                        self,
                        expr.span,
                        "no type found for paren expr",
                    ));
                }
            }
            Expr::Call { ident, args } => {
                if let Some(ret) = self.func_ret.get(ident) {
                    if self.expr_ty.insert(expr, ret.clone()).is_some() {
                        unimplemented!("NOT SURE TODO")
                    }
                } else {
                    self.errors.push(Error::error_with_span(
                        self,
                        expr.span,
                        "unknown function name",
                    ));
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
                    self.errors.push(Error::error_with_span(
                        self,
                        expr.span,
                        &format!("duplicate value expr {:?}\n{:?}", self.expr_ty, expr),
                    ));
                }
            }
            Expr::StructInit { name, fields } => {
                for FieldInit { ident, init, .. } in fields {
                    self.visit_expr(init);
                    check fields??
                }
                // TODO: check fields and
                if self.expr_ty.insert(expr, Ty::Adt(name.clone())).is_some() {
                    unimplemented!("No duplicates")
                }
            }
            Expr::ArrayInit { items } => {
                for item in items {
                    self.visit_expr(item);
                }

                let arr_ty = items.chunks(2).fold(
                    Option::<Ty>::None,
                    // TODO: this might be overkill?
                    |mut ty, arr| match arr {
                        [] => None,
                        [a] if ty.is_none() => self.expr_ty.get(a).cloned(),
                        [a] => fold_ty(ty.as_ref(), self.expr_ty.get(a), &BinOp::Add),
                        [a, b] if ty.is_none() => {
                            fold_ty(self.expr_ty.get(a), self.expr_ty.get(b), &BinOp::Add)
                        }
                        [a, b] => fold_ty(
                            fold_ty(ty.as_ref(), self.expr_ty.get(a), &BinOp::Add).as_ref(),
                            self.expr_ty.get(b),
                            &BinOp::Add,
                        ),
                        [..] => unreachable!("{:?}", arr),
                    },
                );

                if self
                    .expr_ty
                    .insert(
                        expr,
                        Ty::Array {
                            size: items.len(),
                            ty: box arr_ty.unwrap().into_spanned(DUMMY),
                        },
                    )
                    .is_some()
                {
                    unimplemented!("No duplicates")
                }
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

fn check_dereference(tcxt: &mut TyCheckRes<'_, '_>, expr: &Expression, indirection: usize) {
    match &expr.val {
        Expr::Ident(id) => {
            if let Some(ty) = tcxt.type_of_ident(id, expr.span) {
                println!("{:?} == {:?}", ty, tcxt.expr_ty.get(expr))
            } else {
                tcxt.errors.push(Error::error_with_span(tcxt, expr.span, "undeclared var"));
            }
        }
        Expr::Deref { indir, expr } => check_dereference(tcxt, expr, *indir),
        Expr::AddrOf(expr) => todo!(),
        Expr::FieldAccess { lhs, rhs } => todo!(),
        Expr::Array { ident, exprs } => todo!(),

        Expr::Urnary { .. }
        | Expr::Binary { .. }
        | Expr::Parens(_)
        | Expr::Call { .. }
        | Expr::StructInit { .. }
        | Expr::ArrayInit { .. }
        | Expr::Value(_) => todo!(),
    }
}

#[derive(Debug)]
crate struct StmtCheck<'v, 'ast, 'input> {
    tcxt: &'v mut TyCheckRes<'ast, 'input>,
}

impl<'ast> Visit<'ast> for StmtCheck<'_, 'ast, '_> {
    fn visit_stmt(&mut self, stmt: &Statement) {
        match &stmt.val {
            // Nothing to do here, TODO: could maybe record for dead code?
            Stmt::VarDecl(_) => {}
            Stmt::Assign { lval, rval } => {
                let lval_ty = lvalue_type(self.tcxt, lval, stmt.span);

                let rval_ty = self.tcxt.expr_ty.get(rval);

                println!("{:?} == {:?}", lval_ty, rval_ty);

                if !lval_ty.as_ref().is_ty_eq(&rval_ty) {
                    self.tcxt.errors.push(Error::error_with_span(
                        self.tcxt,
                        stmt.span,
                        &format!(
                            "assign to expression of wrong type\nfound {} expected {}",
                            rval_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                            lval_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                        ),
                    ));
                }
            }
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

fn lvalue_type(tcxt: &mut TyCheckRes<'_, '_>, lval: &Expression, stmt_span: Range) -> Option<Ty> {
    let lval_ty = match &lval.val {
        Expr::Ident(id) => tcxt.type_of_ident(id, stmt_span),
        Expr::Deref { indir, expr } => {
            // TODO: may need to resolve the deref'ed type
            tcxt
                // TODO: this need some tweaking (the ident is not always valid)
                // (as_ident_string)
                .type_of_ident(&expr.val.as_ident_string(), stmt_span)
                .map(|t| t.dereference(*indir))
        }
        Expr::Array { ident, exprs } => {
            if let Some(ty @ Ty::Array { .. }) = &tcxt.type_of_ident(ident, stmt_span) {
                let dim = ty.array_dim();
                if exprs.len() != dim {
                    tcxt.errors.push(Error::error_with_span(
                        tcxt,
                        stmt_span,
                        &format!("mismatched array dimension found {} expected {}", exprs.len(), dim),
                    ));
                    None
                } else {
                    Some(ty.index_dim(dim))
                }
            } else {
                // TODO: specific error here?
                None
            }
        },
        Expr::FieldAccess { lhs, rhs } => {
            let id = lhs.val.as_ident_string();
            if let Some(Ty::Adt(name)) = tcxt.type_of_ident(&id, stmt_span) {
                let fields = tcxt.struct_fields.get(&name).cloned().unwrap_or_default();
                walk_field_access(tcxt, &fields, rhs)
            } else {
                tcxt.errors.push(Error::error_with_span(
                    tcxt,
                    stmt_span,
                    &format!("no struct {} found", id),
                ));
                None
            }
        },
        Expr::AddrOf(_)
        // invalid lval
        | Expr::Urnary { .. }
        | Expr::Binary { .. }
        | Expr::Parens(_)
        | Expr::Call { .. }
        | Expr::StructInit { .. }
        | Expr::ArrayInit { .. }
        | Expr::Value(_) => {
            panic!(
                "{}",
                Error::error_with_span(tcxt, stmt_span, "invalid lValue")
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
            walk_field_access(tcxt, fields, expr)
        }
        Expr::Array { ident, exprs } => {
            if let arr @ Some(ty @ Ty::Array { .. }) = &tcxt.type_of_ident(ident, expr.span) {
                let dim = ty.array_dim();
                if exprs.len() != dim {
                    tcxt.errors.push(Error::error_with_span(
                        tcxt,
                        expr.span,
                        &format!("mismatched array dimension found {} expected {}", exprs.len(), dim),
                    ));
                    None
                } else {
                    arr.clone()
                }
            } else {
                // TODO: specific error here?
                None
            }
        },
        Expr::FieldAccess { lhs, rhs } => {
            let id = lhs.val.as_ident_string();
            if let Some(Ty::Adt(name)) = tcxt.type_of_ident(&id, expr.span) {
                // TODO: this is kinda ugly because of the clone but it complains about tcxt otherwise
                // or default not being impl'ed /o\
                let fields = tcxt.struct_fields.get(&name).cloned().unwrap_or_default();
                walk_field_access(tcxt, &fields, rhs)
            } else {
                tcxt.errors.push(Error::error_with_span(
                    tcxt,
                    expr.span,
                    &format!("no struct {} found", id),
                ));
                None
            }
        },
        Expr::AddrOf(_)
        // invalid lval
        | Expr::Urnary { .. }
        | Expr::Binary { .. }
        | Expr::Parens(_)
        | Expr::Call { .. }
        | Expr::StructInit { .. }
        | Expr::ArrayInit { .. }
        | Expr::Value(_) => {
            panic!(
                "{}",
                Error::error_with_span(tcxt, expr.span, "invalid lValue")
            )
        }
    }
}

fn fold_ty(lhs: Option<&Ty>, rhs: Option<&Ty>, op: &BinOp) -> Option<Ty> {
    // println!("fold: {:?} {:?}", lhs, rhs);
    let res = match (lhs?, rhs?) {
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
        (Ty::Array { size, ty: t1 }, Ty::Array { size: s, ty: t2 }) if size == s => {
            Some(Ty::Array {
                size: *size,
                ty: box fold_ty(Some(&t1.val), Some(&t2.val), op)?.into_spanned(DUMMY),
            })
        }
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
        (Ty::Ptr(_), _) => todo!("{:?} {:?}", lhs?, rhs?),
        (r @ Ty::Ref(_), t @ Ty::Ref(_)) => fold_ty(r.resolve().as_ref(), t.resolve().as_ref(), op),
        (r @ Ty::Ref(_), t) => fold_ty(r.resolve().as_ref(), Some(t), op),
        (r, t @ Ty::Ref(_)) => fold_ty(Some(r), t.resolve().as_ref(), op),
    };
    // println!("fold result: {:?}", res);
    res
}
