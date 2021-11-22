use std::{
    cell::{Cell, RefCell},
    fmt,
    hash::{Hash, Hasher},
    iter::FromIterator,
    sync::mpsc::Receiver,
};

use parking_lot::RwLock;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet, FxHasher};

use crate::{
    ast::{
        parse::{symbol::Ident, ParseResult, ParsedBlob},
        types::{
            to_rng, Adt, BinOp, Binding, Block, Const, Decl, Declaration, Enum, Expr, Expression,
            Field, FieldInit, Func, Generic, Impl, MatchArm, Param, Pat, Path, Range, Spany,
            Statement, Stmt, Struct, Trait, Ty, Type, TypeEquality, UnOp, Val, Variant, DUMMY,
        },
    },
    error::Error,
    typeck::{
        check::{coercion, is_truthy, resolve_ty, StmtCheck},
        generic::TyRegion,
        infer::TypeInfer,
    },
    visit::Visit,
};

crate mod check;
crate mod generic;
crate mod infer;
crate mod rawvec;
crate mod scope;
crate mod trait_solver;

use check::fold_ty;
use generic::{GenericResolver, Node};
use scope::{hash_file, ScopeContents, ScopeWalker, ScopedName};
use trait_solver::TraitSolve;

use self::scope::{Scope, ScopeItem};

#[derive(Debug, Default)]
crate struct VarInFunction<'ast> {
    /// A backwards mapping of variable span -> function name.
    func_spans: HashMap<Range, Ident>,
    /// The variables in functions, mapped fn name -> variables.
    func_refs: HashMap<Ident, HashMap<Ident, Ty>>,
    /// Name to the function it represents.
    crate name_func: HashMap<Ident, &'ast Func>,
    /// Does this function have any return statements.
    func_return: HashSet<Ident>,
    /// All of the variables in a scope that are declared. We track them to determine if they are
    /// used.
    unsed_vars: HashMap<ScopedName, (Range, Cell<bool>)>,
}

impl VarInFunction<'_> {
    crate fn get_fn_by_span(&self, span: Range) -> Option<Ident> {
        self.func_spans.iter().find_map(|(k, v)| {
            if k.start <= span.start && k.end >= span.end {
                Some(*v)
            } else {
                None
            }
        })
    }

    fn insert(&mut self, rng: Range, name: Ident) -> Option<Ident> {
        self.func_spans.insert(rng, name)
    }
}

pub type AstReceiver = Receiver<ParseResult<ParsedBlob>>;

#[derive(Default)]
crate struct TyCheckRes<'ast, 'input> {
    /// The name of the file being checked.
    crate file_names: HashMap<u64, &'input str>,
    /// The content of the file as a string.
    crate inputs: HashMap<u64, &'input str>,

    /// The name of the function currently in or `None` if global.
    curr_fn: Option<Ident>,
    /// Global variables declared outside of functions.
    global: HashMap<Ident, Ty>,

    /// All the info about variables local to a specific function.
    ///
    /// Parameters are included in the locals.
    crate var_func: VarInFunction<'ast>,

    /// A mapping of expression -> type, this is the main inference table.
    crate expr_ty: HashMap<&'ast Expression, Ty>,

    /// An `Expression` -> `Ty` mapping made after monomorphization.
    ///
    /// Types reflect specializations that happens to the expressions. This
    /// only effects expressions where parameters are used (as far as I can tell) since
    /// `GenSubstitution` removes all the typed statements and expressions.
    crate mono_expr_ty: RefCell<HashMap<Expression, Ty>>,

    /// A mapping of identities -> val, this is how const folding keeps track of `Expr::Ident`s.
    crate consts: HashMap<Ident, &'ast Val>,

    // /// A mapping of struct name to the fields of that struct.
    // struct_fields: HashMap<Path, (Vec<Type>, Vec<Field>)>,
    // /// A mapping of enum name to the variants of that enum.
    // enum_fields: HashMap<Path, (Vec<Type>, Vec<Variant>)>,
    /// A mapping of struct name to struct def.
    crate name_struct: HashMap<Ident, &'ast Struct>,
    /// A mapping of enum name to enum def.
    crate name_enum: HashMap<Ident, &'ast Enum>,

    /// Resolve generic types at the end of type checking.
    crate generic_res: GenericResolver<'ast>,
    /// Trait resolver for checking the bounds on generic types.
    crate trait_solve: TraitSolve<'ast>,
    /// Name resolution and scope tracking.
    crate name_res: ScopeWalker,

    // TODO: this isn't ideal since you can forget to set/unset...
    /// Do we record uses of this variable during the following expr tree walk.
    record_used: bool,

    uniq_generic_instance_id: Cell<usize>,

    /// Errors collected during parsing and type checking.
    crate errors: RwLock<Vec<Error<'input>>>,

    rcv: Option<AstReceiver>,
    crate imported_items: Vec<&'static Declaration>,
}

impl fmt::Debug for TyCheckRes<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TyCheckResult")
            // .field("global", &self.global)
            // .field("curr_fn", &self.curr_fn)
            // .field("func_refs", &self.var_func.func_refs)
            // .field("func_params", &self.func_params)
            // .field("expr_ty", &self.expr_ty)
            // .field("struct_fields", &self.struct_fields)
            // .field("enum_fields", &self.enum_fields)
            .field("generic_res", &self.generic_res)
            .field("trait_solve", &self.trait_solve)
            .finish()
    }
}

impl<'input> TyCheckRes<'_, 'input> {
    crate fn new(input: &'input str, name: &'input str, rcv: AstReceiver) -> Self {
        let file_id = hash_file(name);
        Self {
            file_names: HashMap::from_iter([(file_id, name)]),
            inputs: HashMap::from_iter([(file_id, input)]),
            rcv: Some(rcv),
            record_used: true,
            ..Self::default()
        }
    }

    crate fn report_errors(&self) -> Result<(), &'static str> {
        if !self.errors.read().is_empty() {
            for e in self.errors.read().iter() {
                // if span_deduper.iter().any(|)
                eprintln!("{}", e)
            }
            // println!("{:?}", self);
            return Err("errors");
        }
        Ok(())
    }

    crate fn unique_id(&self) -> usize {
        let x = self.uniq_generic_instance_id.get();
        self.uniq_generic_instance_id.set(x + 1);
        x
    }

    crate fn set_record_used_vars(&mut self, used: bool) -> bool {
        let old = self.record_used;
        self.record_used = used;
        old
    }

    /// Find the `Type` of this identifier AND mark it as used.
    crate fn type_of_ident(&self, id: Ident, span: Range) -> Option<Ty> {
        self.var_func
            .get_fn_by_span(span)
            .and_then(|f| {
                if self.record_used {
                    // TODO: unused leaks into other scope
                    if let Some((_, b)) =
                        self.var_func.unsed_vars.get(&ScopedName::func_scope(f, id, span.file_id))
                    {
                        b.set(true);
                    }
                }
                self.var_func.func_refs.get(&f).and_then(|s| s.get(&id))
            })
            .or_else(|| {
                self.global.get(&id).map(|ty| {
                    if let Some((_, b)) =
                        self.var_func.unsed_vars.get(&ScopedName::global(span.file_id, id))
                    {
                        b.set(true);
                    }
                    ty
                })
            })
            .cloned()
    }

    crate fn patch_generic_from_path(&self, ty: &Type, span: Range) -> Type {
        // Hack: until import/name resolution is a thing
        //
        // Then have another struct that walks all types and can check if there
        // is a matching generic in the scope and convert Path(T) -> Generic {T}
        if let Ty::Path(p) = &ty.val {
            if let Some(gens_in_scope) = self
                .var_func
                .get_fn_by_span(span)
                .and_then(|name| self.var_func.name_func.get(&name))
                .map(|f| &f.generics)
            {
                if let Some(gen) = gens_in_scope.iter().find(|gty| gty.ident == p.segs[0]) {
                    return Ty::Generic { ident: gen.ident, bound: gen.bound.clone() }
                        .into_spanned(ty.span);
                }
            }
        }
        ty.clone()
    }
}

// @cleanup: my guess is this will somewhat go away, stmt and smaller will be handled by TypeInferer
impl<'ast, 'input> Visit<'ast> for TyCheckRes<'ast, 'input> {
    /// We first walk declarations and save function headers then once all the declarations have
    /// been collected we start type checking expressions.
    fn visit_prog(&mut self, items: &'ast [crate::ast::types::Declaration]) {
        // We just add all files we know about at this point
        // we will do the same thing when we see an import too
        self.name_res.add_file_scopes(&self.file_names);

        let mut funcs = vec![];
        let mut impls = vec![];
        for item in items {
            match &item.val {
                Decl::Func(func) => {
                    self.visit_func(func);
                    funcs.push(func);
                }
                Decl::Const(var) => {
                    self.visit_var(var);
                }
                Decl::Trait(trait_) => self.visit_trait(trait_),
                Decl::Impl(imp) => {
                    self.visit_impl(imp);
                    self.visit_func(&imp.method);
                    impls.push(imp);
                }
                Decl::Adt(adt) => self.visit_adt(adt),
                Decl::Const(co) => {}
                Decl::Import(_) => {
                    let mut items = vec![];
                    let mut added_file = false;
                    // TODO: handle errors
                    while let Ok(Ok(blob)) = self.rcv.as_ref().unwrap().recv() {
                        if !added_file {
                            let file_id = hash_file(blob.file);

                            self.name_res.add_file_scope(file_id);
                            self.file_names.insert(file_id, blob.file);
                            self.inputs.insert(file_id, blob.input);
                        }

                        items.push(blob.decl);

                        if blob.count == 0 {
                            break;
                        }
                    }
                    let imports: &'static [Declaration] = items.leak();

                    self.visit_prog(imports);
                    self.imported_items.extend(imports);
                }
            }
        }
        // Stabilize order which I'm not sure how it gets unordered
        funcs.sort_by(|a, b| a.span.start.cmp(&b.span.start));
        for func in funcs {
            self.curr_fn = Some(func.ident);

            crate::visit::walk_func(self, func);

            if !matches!(func.ret.val, Ty::Void) && !self.var_func.func_return.contains(&func.ident)
            {
                    self.errors.write().push(Error::error_with_span(
                        self,
                        func.span,
                        &format!(
                            "function `{}` has return type `{}` but no return statement",
                            func.ident, func.ret.val
                        ),
                    ));
            }
            self.curr_fn.take();
        }

        // stabilize order
        impls.sort_by(|a, b| a.span.start.cmp(&b.span.start));
        for trait_ in impls {
            self.curr_fn = Some(trait_.method.ident);
            crate::visit::walk_func(self, &trait_.method);
            // TODO: check return just like above
            self.curr_fn.take();
        }

        let mut unused = self
            .var_func
            .unsed_vars
            .iter()
            .filter(|(id, (_, used))| {
                !used.get() && !id.ident().map_or(false, |n| n.name().starts_with('_'))
            })
            .map(|(id, (sp, _))| (id.ident().unwrap(), *sp))
            .collect::<Vec<_>>();

        unused.sort_by(|a, b| a.1.cmp(&b.1));

        // TODO: see about unused declarations
        // After all checking then we can check for unused vars
        for (unused, span) in unused {
                self.errors.write().push(Error::error_with_span(
                    self,
                    span,
                    &format!("unused variable `{}`, remove or reference", unused),
                ));
        }
    }

    fn visit_trait(&mut self, t: &'ast Trait) {
        if self.trait_solve.add_trait(t).is_some() {
                self.errors.write().push(Error::error_with_span(
                    self,
                    t.span,
                    &format!("duplicate trait `{}` found", t.path),
                ));
        } else {
            self.name_res.add_decl(
                t.span.file_id,
                Scope::Trait { file: t.span.file_id, trait_: *t.path.segs.last().unwrap() },
            );
        }
    }

    fn visit_impl(&mut self, imp: &'ast Impl) {
        if let Err(e) = self.trait_solve.add_impl(imp) {
            self.errors.write().push(Error::error_with_span(
                self,
                imp.span,
                &format!("no trait `{}` found for this implementation", imp.path),
            ));
        } else {
            self.name_res.add_decl(
                imp.span.file_id,
                Scope::Impl { file: imp.span.file_id, imp: *imp.path.segs.last().unwrap() },
            );
        }
    }

    fn visit_func(&mut self, func: &'ast Func) {
        if self.curr_fn.is_none() {
            self.name_res.add_decl(
                func.span.file_id,
                Scope::Func { file: func.span.file_id, func: func.ident },
            );
            // Current function scope (also the name)
            self.curr_fn = Some(func.ident);

            if self.var_func.insert(func.span, func.ident).is_some() {
                self.errors.write().push(Error::error_with_span(
                    self,
                    func.span,
                    "function takes up same span as other function",
                ));
            }

            if func.generics.is_empty() && func.ret.val.has_generics() {
                self.errors.write().push(Error::error_with_span(
                    self,
                    func.span,
                    "generic type used without being declared",
                ));
            }

            if !func.generics.is_empty() {
                self.generic_res.collect_generic_params(
                    &Node::Func(func.ident),
                    &Ty::Func {
                        ident: func.ident,
                        ret: box func.ret.val.clone(),
                        params: func
                            .generics
                            .iter()
                            .map(|t| Ty::Generic { ident: t.ident, bound: t.bound.clone() })
                            .collect(),
                    },
                );
            }

            // Now we can check the return value incase it was generic we did that ^^
            //
            // We take from the `generics` to get bound info
            let ty = &func.ret;
            if func.ret.val.has_generics() {
                self.generic_res.collect_generic_usage(
                    &ty.val,
                    self.unique_id(),
                    0,
                    &[],
                    &mut vec![Node::Func(func.ident)],
                );

                let matching_gen = func.generics.iter().any(|g| g.ident == *ty.val.generics()[0]);
                if !matching_gen {
                    self.errors.write().push(Error::error_with_span(
                        self,
                        func.span,
                        &format!(
                            "found return `{}` which is not a declared generic type",
                            func.ret.val
                        ),
                    ));
                }
            };

            if self.var_func.name_func.insert(func.ident.to_owned(), func).is_some() {
                self.errors.write().push(Error::error_with_span(
                    self,
                    func.span,
                    &format!("multiple function declaration `{}`", func.ident),
                ));
            }
        } else {
            panic!(
                "{}",
                Error::error_with_span(self, func.span, "function defined within function")
            )
        }
        // We have left this functions scope
        self.curr_fn.take();
    }

    fn visit_adt(&mut self, adt: &'ast Adt) {
        let span;
        let ident;
        match adt {
            Adt::Struct(struc) => {
                span = struc.span;
                ident = struc.ident;

                if self.name_struct.insert(struc.ident, struc).is_some() {
                    self.errors.write().push(Error::error_with_span(
                        self,
                        struc.span,
                        "duplicate struct names",
                    ));
                }

                if !struc.generics.is_empty() {
                    self.generic_res.collect_generic_params(
                        &Node::Struct(struc.ident),
                        &Ty::Struct {
                            ident: struc.ident,
                            gen: struc
                                .generics
                                .iter()
                                .map(|g| g.to_type().into_spanned(g.span))
                                .collect(),
                        },
                    );
                }
            }
            Adt::Enum(en) => {
                span = en.span;
                ident = en.ident;

                if self.name_enum.insert(en.ident, en).is_some() {
                    self.errors.write().push(Error::error_with_span(
                        self,
                        en.span,
                        "duplicate struct names",
                    ));
                }

                if !en.generics.is_empty() {
                    self.generic_res.collect_generic_params(
                        &Node::Enum(en.ident),
                        &Ty::Enum {
                            ident: en.ident,
                            gen: en
                                .generics
                                .iter()
                                .map(|g| g.to_type().into_spanned(g.span))
                                .collect(),
                        },
                    );
                }
            }
        }
        self.name_res.add_decl(span.file_id, Scope::Adt { file: span.file_id, adt: ident })
    }

    // TODO: this is not what it used to be
    fn visit_var(&mut self, var: &'ast Const) {
        #[allow(clippy::if-then-panic)]
        if let Some(fn_id) = self.curr_fn {
            let node = Node::Func(fn_id);
            let mut stack = if self.generic_res.has_generics(&node) { vec![node] } else { vec![] };
            self.generic_res.collect_generic_usage(
                &var.ty.val,
                self.unique_id(),
                0,
                &[TyRegion::Const(var)],
                &mut stack,
            );

            if self
                .var_func
                .func_refs
                .entry(fn_id)
                .or_default()
                .insert(var.ident, var.ty.val.clone())
                .is_some()
            {
                self.errors.write().push(Error::error_with_span(
                    self,
                    var.span,
                    &format!("[E0w] duplicate variable name `{}`", var.ident),
                ));
            }
            self.var_func.unsed_vars.insert(
                ScopedName::func_scope(fn_id, var.ident, var.span.file_id),
                (var.span, Cell::new(false)),
            );

            // This const is declared inside of a function
            self.name_res.add_item(
                var.span.file_id,
                Scope::Func { file: var.span.file_id, func: fn_id },
                ScopeItem::Var(var.ident),
            );

            // bail out before we set the scope as global
            return;
        } else {
            if self.global.insert(var.ident, var.ty.val.clone()).is_some() {
                self.errors.write().push(Error::error_with_span(
                    self,
                    var.span,
                    &format!("global variable `{}` is already declared", var.ident),
                ));
            }
            self.expr_ty.insert(&var.init, var.init.val.type_of().unwrap());
        }

        // This const is declared in the file scope
        self.name_res.add_item(
            var.span.file_id,
            Scope::File(var.span.file_id),
            ScopeItem::Var(var.ident),
        );
        self.var_func
            .unsed_vars
            .insert(ScopedName::global(var.span.file_id, var.ident), (var.span, Cell::new(false)));
    }

    fn visit_params(&mut self, params: &[Param]) {
        if let Some(fn_id) = self.curr_fn {
            for Param { ident, ty, span } in params {
                // This param is declared inside of a function
                self.name_res.add_item(
                    span.file_id,
                    Scope::Func { file: span.file_id, func: fn_id },
                    ScopeItem::Var(*ident),
                );

                // TODO: Do this for returns and any place we match for Ty::Generic {..}
                if ty.val.has_generics() {
                    self.generic_res.collect_generic_usage(
                        &ty.val,
                        self.unique_id(),
                        0,
                        &[],
                        &mut vec![Node::Func(fn_id)],
                    );

                    let matching_gen = self
                        .var_func
                        .name_func
                        .get(&fn_id)
                        .and_then(|f| {
                            // TODO: this doesn't work for something like `enum result<T, E>`
                            // only checks `T` now
                            f.generics.iter().find(|g| g.ident == *ty.val.generics()[0])
                        })
                        .is_some();

                    if !matching_gen {
                        self.errors.write().push(Error::error_with_span(
                            self,
                            *span,
                            &format!(
                                "found parameter `{}` which is not a declared generic type",
                                ty.val
                            ),
                        ));
                    }
                };
                if self
                    .var_func
                    .func_refs
                    .entry(fn_id)
                    .or_default()
                    .insert(*ident, ty.val.clone())
                    .is_some()
                {
                    self.errors.write().push(Error::error_with_span(
                        self,
                        *span,
                        &format!("duplicate param name `{}`", ident),
                    ));
                }
                self.var_func.unsed_vars.insert(
                    ScopedName::func_scope(fn_id, *ident, span.file_id),
                    (*span, Cell::new(false)),
                );
            }
        } else {
            panic!("{}", Error::error_with_span(self, DUMMY, &format!("{:?}", params)))
        }
    }

    /// We overwrite this so that no type checking of the arm statements happens until we
    /// gather the nested scope from binding in match arms.
    ///
    /// See `StmtCheck::visit_stmt` for what happens.
    fn visit_match_arm(&mut self, _arms: &'ast [MatchArm]) {}

    fn visit_stmt(&mut self, stmt: &'ast Statement) {
        self.error_in_current_expr_tree.set(false);

        self.name_res.visit_stmt(stmt);

        let mut infer = TypeInfer { tcxt: self };
        infer.visit_stmt(stmt);

        crate::visit::walk_stmt(self, stmt);

        if self.error_in_current_expr_tree.get() {
            return;
        }
        self.set_record_used_vars(true);

        // check the statement after walking incase there were var declarations
        let mut check = StmtCheck { tcxt: self };
        check.visit_stmt(stmt);
    }

    fn visit_expr(&mut self, expr: &'ast Expression) {
        if self.error_in_current_expr_tree.get() {
            return;
        }

        match &expr.val {
            Expr::Ident(var_name) => {
                if let Some(ty) = self.type_of_ident(*var_name, expr.span) {
                    // self.expr_ty.insert(expr, ty);
                    // Ok because of `x += 1;` turns into `x = x + 1;`
                } else {
                    self.errors.write().push(Error::error_with_span(
                        self,
                        expr.span,
                        "no type found for ident expr",
                    ));
                    self.error_in_current_expr_tree.set(true);
                }
            }
            Expr::Array { ident, exprs } => {
                for ex in exprs {
                    self.visit_expr(ex);
                }

                if self.error_in_current_expr_tree.get() {
                    return;
                }

                for e in exprs {
                    let ty = self.expr_ty.get(e);
                    if !matches!(ty, Some(Ty::Int)) {
                        self.errors.write().push(Error::error_with_span(
                            self,
                            expr.span,
                            &format!(
                                "cannot index array with {}",
                                ty.map_or("<unknown>".to_owned(), |t| t.to_string())
                            ),
                        ));
                        self.error_in_current_expr_tree.set(true);
                    }
                }
                if let Some(ty) = self.type_of_ident(*ident, expr.span) {
                    // if self.expr_ty.insert(expr, ty).is_some() {
                    // Ok because of `x[0] += 1;` turns into `x[0] = x[0] + 1;`
                    // }
                } else {
                    self.errors.write().push(Error::error_with_span(
                        self,
                        expr.span,
                        "no type found for array expr",
                    ));
                    self.error_in_current_expr_tree.set(true);
                }
            }
            Expr::Urnary { op, expr: inner_expr } => {
                self.visit_expr(inner_expr);
                let ty = self.expr_ty.get(&**inner_expr);
                match op {
                    UnOp::Not => {
                        if is_truthy(ty) {
                            self.expr_ty.insert(expr, Ty::Bool);
                        } else {
                            self.errors.write().push(Error::error_with_span(
                                self,
                                expr.span,
                                "cannot negate non bool type",
                            ));
                            self.error_in_current_expr_tree.set(true);
                        }
                    }
                    UnOp::OnesComp => {
                        // TODO: think about pointer maths
                        if let Some(Ty::Int | Ty::Ptr(_)) = ty {
                            self.expr_ty.insert(expr, Ty::Int);
                        } else {
                            self.errors.write().push(Error::error_with_span(
                                self,
                                expr.span,
                                "cannot negate non bool type",
                            ));
                            self.error_in_current_expr_tree.set(true);
                        }
                    }
                }
            }
            Expr::Deref { indir, expr: inner_expr } => {
                self.visit_expr(inner_expr);

                let ty = self.expr_ty.get(&**inner_expr).expect("expression to be walked already");
                let ty = ty.dereference(*indir);

                check_dereference(self, inner_expr);

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
                if self.error_in_current_expr_tree.get() {
                    return;
                }
                self.visit_expr(rhs);
                if self.error_in_current_expr_tree.get() {
                    return;
                }

                let lhs_ty = self.expr_ty.get(&**lhs);
                let rhs_ty = self.expr_ty.get(&**rhs);

                if let Some(ty) = fold_ty(
                    self,
                    resolve_ty(self, lhs, lhs_ty).as_ref(),
                    resolve_ty(self, rhs, rhs_ty).as_ref(),
                    op,
                    expr.span,
                ) {
                    if let Some(t2) = self.expr_ty.insert(expr, ty.clone()) {
                        if !ty.is_ty_eq(&t2) {
                            self.errors.write().push(Error::error_with_span(
                                self,
                                expr.span,
                                "ICE: something went wrong in the compiler",
                            ));
                            self.error_in_current_expr_tree.set(true);
                        }
                    }
                } else {
                    self.errors.write().push(Error::error_with_span(
                        self,
                        expr.span,
                        &format!("no type found for bin expr {:?} != {:?}", lhs_ty, rhs_ty),
                    ));
                    self.error_in_current_expr_tree.set(true);
                }
            }
            Expr::Parens(inner_expr) => {
                self.visit_expr(inner_expr);
                if let Some(ty) = self.expr_ty.get(&**inner_expr).cloned() {
                    if let Some(t2) = self.expr_ty.insert(expr, ty.clone()) {
                        if !ty.is_ty_eq(&t2) {
                            self.errors.write().push(Error::error_with_span(
                                self,
                                expr.span,
                                "ICE: something went wrong in the compiler",
                            ));
                            self.error_in_current_expr_tree.set(true);
                        }
                    }
                } else {
                    self.errors.write().push(Error::error_with_span(
                        self,
                        expr.span,
                        "no type found for paren expr",
                    ));
                    self.error_in_current_expr_tree.set(true);
                }
            }
            Expr::Call { path, args, type_args } => {
                let ident = path.segs.last().unwrap();
                if self.var_func.name_func.get(ident).is_none() {
                    self.errors.write().push(Error::error_with_span(
                        self,
                        expr.span,
                        &format!("no function named `{}`", path),
                    ));
                    self.error_in_current_expr_tree.set(true);
                }

                for arg in args {
                    self.visit_expr(arg);
                }

                // Check type_args agrees
                let stack = build_stack(self, Node::Func(*ident));

                let gen_arg_set_id = self.unique_id();
                let mut gen_arg_map = HashMap::default();
                let func = self.var_func.name_func.get(ident).expect("all functions are collected");
                // Iter the type arguments at the call site
                for (gen_arg_idx, ty_arg) in type_args.iter().enumerate() {
                    // TODO: name resolution
                    let ty_arg = self.patch_generic_from_path(ty_arg, expr.span).val;
                    // Don't use the same stack for each iteration
                    let mut stack = stack.clone();

                    let gen = &func.generics[gen_arg_idx];
                    // Find the param that is the "generic" and check against type argument
                    let mut arguments = vec![];
                    for (i, p) in func.params.iter().enumerate() {
                        if gen.is_ty_eq(&p.ty.val) {
                            arguments.push(TyRegion::Expr(&args[i].val));
                        }
                    }

                    self.generic_res.collect_generic_usage(
                        &ty_arg,
                        gen_arg_set_id,
                        gen_arg_idx,
                        &arguments,
                        &mut stack,
                    );

                    // println!("CALL IN CALL {:?} == {:?} {:?}", ty, gen, stack);

                    gen_arg_map.insert(gen.ident, ty_arg.clone());
                }

                let func_params = self
                    .var_func
                    .name_func
                    .get(ident)
                    .map(|f| &f.params)
                    .expect("function is known with params");

                if args.len() != func_params.len() {
                    self.errors.write().push(Error::error_with_span(
                        self,
                        expr.span,
                        "wrong number of arguments",
                    ));
                    self.error_in_current_expr_tree.set(true);
                }

                for (idx, arg) in args.iter().enumerate() {
                    let mut param_ty = func_params.get(idx).map(|p| p.ty.val.clone());
                    let mut arg_ty = self.expr_ty.get(arg).cloned();

                    if let Some(Ty::Generic { ident, .. }) = &param_ty {
                        if let Some(ty_arg) = gen_arg_map.get(ident).cloned() {
                            param_ty = Some(ty_arg);
                        }
                    }

                    // TODO: remove
                    coercion(param_ty.as_ref(), arg_ty.as_mut());

                    if !param_ty.as_ref().is_ty_eq(&arg_ty.as_ref()) {
                        self.errors.write().push(Error::error_with_span(
                            self,
                            arg.span,
                            &format!(
                                "[E0tc] call with wrong argument type\nfound `{}` expected `{}`",
                                arg_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                                param_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                            ),
                        ));
                        self.error_in_current_expr_tree.set(true);
                    }
                }

                // This is commented out because of inference
                // TODO: do more of this

                // let t = &f.ret.val;
                // let ret_ty = if t.has_generics() {
                //     subs_type_args(t, type_args, &f.generics)
                // } else {
                //     t.clone()
                // };
                // self.expr_ty.insert(expr, ret_ty);
                // because of x += 1;
            }
            Expr::TraitMeth { trait_, args, type_args } => {
                let ident = *trait_.segs.last().unwrap();
                if self.trait_solve.traits.get(trait_).is_none() {
                    self.errors.write().push(Error::error_with_span(
                        self,
                        expr.span,
                        &format!("no trait named `{}`", trait_),
                    ));
                    self.error_in_current_expr_tree.set(true);
                }
                for expr in args {
                    self.visit_expr(expr);
                }

                let trait_def =
                    self.trait_solve.traits.get(trait_).cloned().expect("trait is defined");

                let mut stack = build_stack(self, Node::Trait(ident));

                let gen_arg_set_id = self.unique_id();
                let mut gen_arg_map = HashMap::default();
                for (gen_arg_idx, ty_arg) in type_args.iter().enumerate() {
                    let ty_arg = self.patch_generic_from_path(ty_arg, expr.span);

                    let gen = &trait_def.generics[gen_arg_idx];

                    let arguments = trait_def
                        .method
                        .function()
                        .params
                        .iter()
                        .enumerate()
                        .filter(|(_i, p)| gen.is_ty_eq(&p.ty.val))
                        .map(|(i, _)| TyRegion::Expr(&args[i].val))
                        .collect::<Vec<_>>();

                    self.generic_res.collect_generic_usage(
                        &ty_arg.val,
                        gen_arg_set_id,
                        gen_arg_idx,
                        &arguments,
                        &mut stack,
                    );

                    gen_arg_map.insert(gen.ident, ty_arg.val.clone());
                }

                let mut has_generic = false;
                let func_params = &trait_def.method.function().params;
                for (idx, arg) in args.iter().enumerate() {
                    let mut param_ty = func_params.get(idx).map(|p| p.ty.val.clone());
                    let arg_ty = self.expr_ty.get(arg).cloned();

                    if let Some(Ty::Generic { ident, .. }) = &param_ty {
                        has_generic = true;
                        if let Some(ty_arg) = gen_arg_map.get(ident).cloned() {
                            param_ty = Some(ty_arg);
                        }
                    }

                    if !param_ty.as_ref().is_ty_eq(&arg_ty.as_ref()) {
                        self.errors.write().push(Error::error_with_span(
                            self,
                            arg.span,
                            &format!(
                                "trait call with wrong argument type\nfound `{}` expected `{}`",
                                arg_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                                param_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                            ),
                        ));
                        self.error_in_current_expr_tree.set(true);
                    }
                }

                let generic_dependence = if has_generic { Some(stack) } else { None };
                self.trait_solve.to_solve(
                    ident,
                    type_args.iter().map(|t| &t.val).collect::<Vec<_>>(),
                    generic_dependence,
                );

                let def_fn = self.trait_solve.traits.get(trait_).expect("trait is defined");
                let t = &def_fn.method.return_ty().val;
                let ret_ty = if t.has_generics() {
                    subs_type_args(t, type_args, &trait_def.generics)
                } else {
                    t.clone()
                };
                self.expr_ty.insert(expr, ret_ty);
            }
            Expr::Value(val) => {
                // inference collects these
            }
            Expr::StructInit { path, fields } => {
                let name = *path.segs.last().unwrap();
                let struc =
                    (*self.name_struct.get(&name).expect("initialized undefined struct")).clone();

                let gen_arg_set_id = self.unique_id();
                let mut gen_args = HashMap::default();
                for FieldInit { ident, init, .. } in fields {
                    self.visit_expr(init);

                    let field_ty = struc
                        .fields
                        .iter()
                        .find_map(|f| if f.ident == *ident { Some(&f.ty.val) } else { None })
                        .expect("no field with that name found");

                    let exprty = self.expr_ty.get(&*init).cloned();

                    let mut stack = build_stack(self, Node::Struct(name));

                    // Collect the generic parameter `struct list<T> vec;` (this has to be a
                    // dependent parameter) or a type argument `struct list<int> vec;`
                    for gen in field_ty.generics().into_iter() {
                        if let Some(idx) = struc.generics.iter().enumerate().find_map(|(i, t)| {
                            if t.ident == *gen {
                                Some(i)
                            } else {
                                None
                            }
                        }) {
                            self.generic_res.collect_generic_usage(
                                exprty.as_ref().unwrap(),
                                gen_arg_set_id,
                                idx,
                                &[TyRegion::Expr(&init.val)],
                                &mut stack,
                            );

                            gen_args.insert(gen, exprty.clone().unwrap().into_spanned(DUMMY));
                        } else {
                            panic!("undefined generic type used")
                        }
                    }

                    // Skip checking type equivalence
                    if field_ty.has_generics() {
                        continue;
                    }

                    if !exprty.as_ref().is_ty_eq(&Some(&field_ty)) {
                        self.errors.write().push(Error::error_with_span(
                            self,
                            init.span,
                            &format!(
                                "field initialized with mismatched type\nfound `{}` expected `{}`",
                                exprty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                                field_ty,
                            ),
                        ));
                        self.error_in_current_expr_tree.set(true);
                    }
                }

                if self
                    .expr_ty
                    .insert(
                        expr,
                        Ty::Struct {
                            ident: name,
                            gen: struc
                                .generics
                                .iter()
                                .filter_map(|g| gen_args.remove(&g.ident))
                                .collect(),
                        },
                    )
                    .is_some()
                {
                    // unimplemented!("No duplicates")
                }
            }
            Expr::EnumInit { path, variant, items } => {
                let ident = *path.segs.last().unwrap();
                let enm =
                    (*self.name_enum.get(&ident).expect("initialized undefined enum")).clone();

                let found_variant =
                    if let Some(vars) = enm.variants.iter().find(|v| v.ident == *variant) {
                        vars
                    } else {
                        self.errors.write().push(Error::error_with_span(
                            self,
                            expr.span,
                            &format!("enum `{}` has no variant `{}`", path, variant),
                        ));
                        self.error_in_current_expr_tree.set(true);
                        return;
                    };

                let mut gen_args = HashMap::default();
                for (_idx, (item, variant_ty)) in items.iter().zip(&found_variant.types).enumerate()
                {
                    // Visit inner expressions
                    self.visit_expr(item);

                    // Gather expression and expected (declared) type
                    let exprty = self.expr_ty.get(&*item).cloned();

                    let _stack = build_stack(self, Node::Enum(ident));

                    // Collect the generic parameter `enum option<T> opt;` (this has to be a
                    // dependent parameter) or a type argument `enum option<int>
                    // opt;`
                    for gen in variant_ty.val.generics().into_iter() {
                        if let Some(_idx) = enm.generics.iter().enumerate().find_map(|(i, t)| {
                            if t.ident == *gen {
                                Some(i)
                            } else {
                                None
                            }
                        }) {
                            // We do NOT call `generic_res.collect_generic_usage(..)` because we
                            // will only collect a variants worth of generic type info.
                            // We instead wait for the assignment to collect all generic arguments.

                            gen_args.insert(gen, exprty.clone().unwrap().into_spanned(DUMMY));
                        } else {
                            panic!("undefined generic type used")
                        }
                    }

                    // Skip checking type equivalence
                    if variant_ty.val.has_generics() {
                        continue;
                    }

                    if !exprty.as_ref().is_ty_eq(&Some(&variant_ty.val)) {
                        self.errors.write().push(Error::error_with_span(
                            self,
                            item.span,
                            &format!(
                                "enum tuple initialized with mismatched type\nfound `{}` expected `{}`",
                                exprty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                                variant_ty.val,
                            ),
                        ));
                        self.error_in_current_expr_tree.set(true);
                    }
                }

                if self
                    .expr_ty
                    .insert(
                        expr,
                        Ty::Enum {
                            ident,
                            gen: enm
                                .generics
                                .iter()
                                .filter_map(|g| gen_args.remove(&g.ident))
                                .collect(),
                        },
                    )
                    .is_some()
                {
                    // TODO: investigate
                    // The enum value from inference is overwritten here so this happens
                }
            }
            Expr::ArrayInit { items } => {
                for item in items {
                    self.visit_expr(item);
                }

                let arr_ty = items.chunks(2).fold(
                    Option::<Ty>::None,
                    // this might be overkill, but `{1 + 1, 2, call()}` all need to be checked
                    |ty, arr| match arr {
                        [] => None,
                        [a] if ty.is_none() => self.expr_ty.get(a).cloned(),
                        [a] => fold_ty(self, ty.as_ref(), self.expr_ty.get(a), &BinOp::Add, a.span),
                        [a, b] if ty.is_none() => fold_ty(
                            self,
                            self.expr_ty.get(a),
                            self.expr_ty.get(b),
                            &BinOp::Add,
                            to_rng(a.span.start..b.span.end, {
                                debug_assert!(a.span.file_id == b.span.file_id);
                                a.span.file_id
                            }),
                        ),
                        [a, b] => fold_ty(
                            self,
                            fold_ty(self, ty.as_ref(), self.expr_ty.get(a), &BinOp::Add, a.span)
                                .as_ref(),
                            self.expr_ty.get(b),
                            &BinOp::Add,
                            b.span,
                        ),
                        [..] => unreachable!("{:?}", arr),
                    },
                );

                self.expr_ty.insert(
                    expr,
                    Ty::Array { size: items.len(), ty: box arr_ty.unwrap().into_spanned(DUMMY) },
                );
                // no is_some check: because of `x[0] += 1;` being lowered to `x[0] = w[0] + 1;`
            }
            Expr::FieldAccess { lhs, rhs } => {
                self.visit_expr(lhs);

                // rhs is saved in `check_field_access`
                let field_ty = check_field_access(self, lhs, rhs);
                if let Some(ty) = field_ty {
                    self.expr_ty.insert(expr, ty);
                    // no is_some check: because of `x.y += 1;` being lowered to `x.y = w.y + 1;`
                } else {
                    // TODO: this error is crappy
                    self.errors.write().push(Error::error_with_span(
                        self,
                        expr.span,
                        "no type found for field access",
                    ));
                    self.error_in_current_expr_tree.set(true);
                }
            }
        }
        // We do NOT call walk_expr here since we recursively walk the exprs
        // when ever found so we have folded the expr types depth first
    }
}

/// Return a type that has been substituted as much as possible at this stage.
///
/// If any generics are left it is because the variable/call they come from has a
/// un-substituted/unresolved generic parameter.
fn subs_type_args(ty: &Ty, ty_args: &[Type], generics: &[Generic]) -> Ty {
    let mut typ = ty.clone();
    for gen in ty.generics() {
        let pos = generics.iter().position(|g| g.ident == *gen).expect("no matching generic");
        typ.subst_generic(*gen, &ty_args[pos].val);
    }
    typ
}

/// Create a stack for the current generic location.
///
/// Filters out function calls with no generic arguments (remove main).
fn build_stack(tcxt: &TyCheckRes<'_, '_>, kind: Node) -> Vec<Node> {
    if let Some((def, ident)) =
        tcxt.curr_fn.as_ref().and_then(|f| Some((tcxt.var_func.name_func.get(f)?, *f)))
    {
        if def.generics.is_empty() {
            vec![kind]
        } else {
            vec![Node::Func(ident), kind]
        }
    } else {
        vec![kind]
    }
}

/// The left hand side of field access has been collected calling this collects the right side.
///
/// The is used in the collection of expressions.
fn check_field_access<'ast>(
    tcxt: &mut TyCheckRes<'ast, '_>,
    lhs: &'ast Expression,
    rhs: &'ast Expression,
) -> Option<Ty> {
    let lhs_ty = tcxt.expr_ty.get(lhs);

    let (name, struc) = if let Some(Ty::Struct { ident, .. }) = lhs_ty.and_then(|t| t.resolve()) {
        // FIXME: come on clone here that's cray
        (ident, (*tcxt.name_struct.get(&ident).expect("no struct definition found")).clone())
    } else {
        panic!("{}", Error::error_with_span(tcxt, lhs.span, "not valid field access"));
    };

    match &rhs.val {
        Expr::Ident(ident) => {
            let rty = struc
                .fields
                .iter()
                .find_map(|f| if f.ident == *ident { Some(f.ty.val.clone()) } else { None })
                .unwrap_or_else(|| panic!("no field `{}` found for struct `{}`", ident, name));
            tcxt.expr_ty.insert(rhs, rty.clone());
            Some(rty)
        }
        Expr::Array { ident, exprs } => {
            for expr in exprs {
                tcxt.visit_expr(expr);
            }

            let rty = struc
                .fields
                .iter()
                .find_map(|f| if f.ident == *ident { Some(f.ty.val.clone()) } else { None })
                .unwrap_or_else(|| panic!("no field `{}` found for struct `{}`", ident, name));

            tcxt.expr_ty.insert(rhs, rty.clone());

            rty.index_dim(tcxt, exprs, rhs.span)
        }
        Expr::FieldAccess { lhs, rhs } => {
            tcxt.visit_expr(lhs);

            let accty = check_field_access(tcxt, lhs, rhs);
            if let Some(ty) = &accty {
                tcxt.expr_ty.insert(rhs, ty.clone());
            }
            accty
        }
        _ => unreachable!("access struct with non ident"),
    }
}

/// The is used in the collection of expressions.
/// TODO: return an error
fn check_dereference(tcxt: &mut TyCheckRes<'_, '_>, expr: &Expression) {
    match &expr.val {
        Expr::Ident(id) => {
            let ty = tcxt.type_of_ident(*id, expr.span).or_else(|| tcxt.expr_ty.get(expr).cloned());
            if let Some(_ty) = ty {
                // println!("{:?} == {:?}", ty, tcxt.expr_ty.get(expr))
            } else {
                // panic!("{:?}", expr);
                tcxt.errors.write().push(Error::error_with_span(
                    tcxt,
                    expr.span,
                    &format!(
                        "cannot dereference `{}`",
                        ty.map_or("<unknown>".to_owned(), |t| t.to_string())
                    ),
                ));
            }
        }
        Expr::Deref { indir: _, expr } => check_dereference(tcxt, expr),
        Expr::AddrOf(expr) => check_dereference(tcxt, expr),
        Expr::FieldAccess { lhs, rhs } => {
            check_dereference(tcxt, lhs);
            check_dereference(tcxt, rhs);
        }
        Expr::Array { ident, exprs } => {
            let ty = tcxt
                .type_of_ident(*ident, expr.span)
                .and_then(|ty| ty.index_dim(tcxt, exprs, expr.span));
            if let Some(_ty) = ty {
                // println!("{:?} == {:?}", ty, tcxt.expr_ty.get(expr))
            } else {
                tcxt.errors.write().push(Error::error_with_span(
                    tcxt,
                    expr.span,
                    &format!(
                        "cannot dereference array `{}`",
                        ty.map_or("<unknown>".to_owned(), |t| t.to_string())
                    ),
                ));
                tcxt.error_in_current_expr_tree.set(true);
            }
        }

        Expr::Urnary { .. }
        | Expr::Binary { .. }
        | Expr::Parens(_)
        | Expr::Call { .. }
        | Expr::TraitMeth { .. }
        | Expr::StructInit { .. }
        | Expr::EnumInit { .. }
        | Expr::ArrayInit { .. }
        | Expr::Value(_) => todo!(),
    }
}
