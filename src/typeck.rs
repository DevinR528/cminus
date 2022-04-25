//! The type checking pass of the compiler.
//!
//! ## The Steps
//!
//! - we visit a slice of declarations [Declaration]s
//!   - walking each [Decl] collecting names but not walking statements/expressions
//!   - if we encounter an import we apply this process from start to finish before moving on from
//!     here
//!   - we collect generics in functions/enums/structs
//!   - we collect paths for name resolution
//! - At this point all names are known we start resolving
//!   - we resolve any parameter types and return types (turn [Ty::Path] into the appropriate type,
//!     enums or structs)
//!   - resolve enum init expressions (convert [Expr::Call] into [Expr::EnumInit] if appropriate)
//!     - eventually any usage of types `let x: path::to::type = ...` will also be resolved
//!       otherwise parsing can tell the difference??
//! - Walking statements inside functions and impls (see [TyCheckRes::visit_stmt])
//!   - reset the poisoned flag (each statement can have it's own error)
//!   - infer the types if possible
//!     - for each expression (lvalues and rvalues) we try to collect the type of the expression
//!     - trait and function calls are a bit more involved (they can error during inference, not
//!       enough args for example)
//!       - calls are mutated so all arguments are concrete types and the results of name res are
//!         applied also
//!     - inference needs to collect expression -> type mappings for further up the tree inference,
//!       a leaf needs info so the branch can infer, DON'T DELETE the `tcxt.expr_ty.insert(expr,
//!       ty)` calls
//!     - structs and enums are similar to calls but not as picky
//!     - done with inference
//!   - walk the statement, we recursively repeat the above for any statement within a statement
//!     (while, if, blocks, etc)
//!   - actual type checking [StmtCheck]
//!     - checks declarations and assignments are valid
//!     - checks the conditions of if, while and match statements
//!       - checks matches have arm patterns that match the expression
//!       - collects the bound variables for new identifiers live in each arm's block
//!
//! Without further ado type checking 🖖

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
            to_rng, Adt, BinOp, Binding, Block, Builtin, Const, Decl, Declaration, Enum, Expr,
            Expression, Field, FieldInit, Func, FuncKind, Generic, Impl, MatchArm, Param, Pat,
            Path, Range, Spany, Statement, Stmt, Struct, Trait, Ty, Type, TypeEquality, UnOp, Val,
            Variant, DUMMY,
        },
    },
    error::{Error, ErrorReport},
    typeck::{
        check::{is_truthy, resolve_ty, StmtCheck},
        generic::TyRegion,
        infer::TypeInfer,
    },
    visit::{Visit, VisitMut}, rawptr,
};

crate mod check;
crate mod generic;
crate mod infer;
crate mod scope;
crate mod trait_solver;

use check::fold_ty;
use generic::{GenericResolver, Node};
use scope::{hash_file, ScopeContents, ScopeWalker, ScopedName};
use trait_solver::TraitSolve;

use self::scope::{ItemIn, Scope};

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

#[derive(Default, derive_help::Debug)]
crate struct TyCheckRes<'ast, 'input> {
    /// The name of the file being checked.
    crate file_names: HashMap<u64, &'input str>,
    /// The content of the file as a string.
    crate inputs: HashMap<u64, &'input str>,

    /// The name of the function currently in or `None` if global.
    #[dbg_ignore]
    curr_fn: Option<Ident>,
    /// Global variables declared outside of functions.
    #[dbg_ignore]
    global: HashMap<Ident, Ty>,

    /// All the info about variables local to a specific function.
    ///
    /// Parameters are included in the locals.
    #[dbg_ignore]
    crate var_func: VarInFunction<'ast>,

    /// A mapping of expression -> type, this is the main inference table.
    // #[dbg_ignore]
    crate expr_ty: HashMap<&'ast Expression, Ty>,

    /// An `Expression` -> `Ty` mapping made after monomorphization.
    ///
    /// Types reflect specializations that happens to the expressions. This
    /// only effects expressions where parameters are used (as far as I can tell) since
    /// `GenSubstitution` removes all the typed statements and expressions.
    #[dbg_ignore]
    crate mono_expr_ty: RefCell<HashMap<Expression, Ty>>,

    // TODO: const folding could fold "const" idents but it would have to track between stmts which
    // we do not
    /// A mapping of identities -> val, this is how const folding keeps track of `Expr::Ident`s.
    // #[dbg_ignore]
    // crate consts: HashMap<Ident, &'ast Val>,

    // /// A mapping of struct name to the fields of that struct.
    // struct_fields: HashMap<Path, (Vec<Type>, Vec<Field>)>,
    // /// A mapping of enum name to the variants of that enum.
    // enum_fields: HashMap<Path, (Vec<Type>, Vec<Variant>)>,
    /// A mapping of struct name to struct def.
    #[dbg_ignore]
    crate name_struct: HashMap<Ident, &'ast Struct>,
    /// A mapping of enum name to enum def.
    #[dbg_ignore]
    crate name_enum: HashMap<Ident, &'ast Enum>,

    /// Resolve generic types at the end of type checking.
    #[dbg_ignore]
    crate generic_res: GenericResolver<'ast>,
    /// Trait resolver for checking the bounds on generic types.
    #[dbg_ignore]
    crate trait_solve: TraitSolve<'ast>,
    /// Name resolution and scope tracking.
    crate name_res: ScopeWalker,

    // TODO: this isn't ideal since you can forget to set/unset...
    /// Do we record uses of this variable during the following expr tree walk.
    #[dbg_ignore]
    record_used: bool,

    #[dbg_ignore]
    uniq_generic_instance_id: Cell<usize>,

    #[dbg_ignore]
    crate errors: ErrorReport<'input>,

    #[dbg_ignore]
    rcv: Option<AstReceiver>,
    #[dbg_ignore]
    crate imported_items: Vec<&'static Declaration>,
}

lazy_static::lazy_static! {
    static ref CONST_STR: Struct = {
        let string = Ident::new(DUMMY, "__const_str");
        Struct {
            ident: string,
            fields: vec![Field { ident: Ident::new(DUMMY, "len"), ty: rawptr!(Ty::Int.into_spanned(DUMMY)), span: DUMMY }],
            generics: vec![],
            span: DUMMY
        }
    };
}

impl<'input> TyCheckRes<'_, 'input> {
    crate fn new(input: &'input str, name: &'input str, rcv: AstReceiver) -> Self {
        let mut builtin_structs = HashMap::default();
        let string = Ident::new(DUMMY, "__const_str");

        builtin_structs.insert(string, &*CONST_STR);

        let file_id = hash_file(name);
        Self {
            file_names: HashMap::from_iter([(file_id, name)]),
            inputs: HashMap::from_iter([(file_id, input)]),
            rcv: Some(rcv),
            record_used: true,
            name_struct: builtin_structs,
            ..Self::default()
        }
    }

    crate fn report_errors(&self) -> Result<(), usize> {
        if !self.errors.is_empty() {
            for e in self.errors.errors().iter() {
                eprintln!("{}", e)
            }
            // TODO: see ./src/main.rs for comment
            return Err(self.errors.errors().len());
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

    // TODO: this should use the stuff from scope not a mix of `var_func`, `globals` etc.
    /// Find the `Type` of this identifier AND mark it as used.
    crate fn type_of_ident(&self, id: Ident, span: Range) -> Option<Ty> {
        self.var_func
            .get_fn_by_span(span)
            .and_then(|f| {
                if self.record_used {
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

    crate fn patch_generic_from_path(&self, ty: &Type, gens_in_scope: &[Generic]) -> Option<Type> {
        // Then have another struct that walks all types and can check if there
        // is a matching generic in the scope and convert Path(T) -> Generic {T}
        Some(match &ty.val {
            Ty::Array { size, ty: arrty } => Ty::Array {
                size: *size,
                ty: box self.patch_generic_from_path(arrty, gens_in_scope)?,
            }
            .into_spanned(ty.span),
            Ty::Struct { ident, gen } => Ty::Struct {
                ident: *ident,
                gen: gen
                    .iter()
                    .map(|g| self.patch_generic_from_path(g, gens_in_scope))
                    .collect::<Option<Vec<_>>>()?,
            }
            .into_spanned(ty.span),
            Ty::Enum { ident, gen } => Ty::Enum {
                ident: *ident,
                gen: gen
                    .iter()
                    .map(|g| self.patch_generic_from_path(g, gens_in_scope))
                    .collect::<Option<Vec<_>>>()?,
            }
            .into_spanned(ty.span),
            Ty::Path(p) => {
                if let Some(gen) = gens_in_scope.iter().find(|gty| gty.ident == p.segs[0]) {
                    Ty::Generic { ident: gen.ident, bound: gen.bound.clone() }.into_spanned(ty.span)
                } else {
                    return None;
                }
            }
            Ty::Ptr(inner) => Ty::Ptr(box self.patch_generic_from_path(&**inner, gens_in_scope)?)
                .into_spanned(ty.span),
            Ty::Ref(inner) => Ty::Ref(box self.patch_generic_from_path(&**inner, gens_in_scope)?)
                .into_spanned(ty.span),
            Ty::Func { ident, ret, params } => todo!(),
            _ => {
                return None;
            }
        })
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
                    self.visit_const(var);
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
            // Storage space for function pointers passed as params
            //
            let mut fn_ptr_storage = vec![];

            self.curr_fn = Some(func.ident);
            self.name_res
                .add_to_scope_stack(Scope::Func { file: func.span.file_id, func: func.ident });

            // If the parameters are struct or enum names resolve them (eventually type aliases??)
            for param in &func.params {
                let resolved = self.name_res.resolve_name(&param.ty.get().val, self);
                if let Some(res) = resolved {
                    param.ty.set(res.into_spanned(param.ty.get().span));
                }

                // Add this to the global function scope for the duration of this functions scope
                // (the one we are in now)
                if let Ty::Func { ident, params, ret } = &param.ty.get().val {
                    let idx = fn_ptr_storage.len();
                    fn_ptr_storage.push(Func {
                        ident: Ident::new(param.ident.span(), &format!("{}fnptr", param.ident)),
                        params: params
                            .iter()
                            .enumerate()
                            .map(|(idx, t)| Param {
                                ty: crate::rawptr!(t.clone().into_spanned(DUMMY)),
                                ident: Ident::new(
                                    param.span,
                                    &format!("{}arg{}", param.ident, idx),
                                ),
                                span: DUMMY,
                            })
                            .collect(),
                        ret: crate::rawptr!(ret.clone().into_spanned(DUMMY)),
                        generics: vec![],
                        stmts: Block { stmts: crate::raw_vec![], span: DUMMY },
                        kind: FuncKind::Normal,
                        span: DUMMY,
                    });

                    self.var_func
                        .name_func
                        .insert(param.ident, unsafe { std::mem::transmute(&fn_ptr_storage[idx]) });
                }
            }
            // Resolve the return value (if struct or enum or type)
            let resolved = self.name_res.resolve_name(&func.ret.get().val, self);
            if let Some(res) = resolved {
                func.ret.set(res.into_spanned(func.ret.get().span));
            }

            struct NameResUserTypes<'ast, 'b> {
                res: &'ast ScopeWalker,
                tcxt: &'ast TyCheckRes<'ast, 'b>,
                func: &'ast Func,
            }
            impl<'ast, 'b> VisitMut<'ast> for NameResUserTypes<'ast, 'b> {
                fn visit_expr(&mut self, expr: &'ast mut Expression) {
                    if let Expr::Call { path, args, type_args } = &expr.val {
                        let fixed_ty = self.res.type_from_path(path, self.tcxt);
                        if let Some(Ty::Enum { ident, .. }) = fixed_ty {
                            let mut path = path.clone();
                            let variant = path.segs.pop().unwrap();
                            expr.val = Expr::EnumInit { path, variant, items: args.clone() };
                        }
                    }
                    // if it's still a call resolve any dependent generic type args from `Ty::Path
                    // -> Ty::Generic`
                    if let Expr::Call { path, type_args, .. } = &expr.val {
                        for ty_arg in unsafe { type_args.iter_mut_shared() } {
                            if !ty_arg.val.has_path() {
                                continue;
                            }

                            if let Some(Ty::Path(path)) = ty_arg.val.resolve() {
                                if let Some(fixed_ty) = self.res.type_from_path(&path, self.tcxt) {
                                    *ty_arg = fixed_ty.into_spanned(ty_arg.span);
                                    // It can't be a generic if we determined it was a struct
                                    continue;
                                }
                            }

                            let resolved =
                                self.tcxt.patch_generic_from_path(ty_arg, &self.func.generics);
                            if let Some(res) = resolved {
                                *ty_arg = res;
                            }
                        }
                    } else if let Expr::Builtin(Builtin::SizeOf(ty)) = &expr.val {
                        if !matches!(ty.get().val, Ty::Path(..)) {
                            return;
                        }
                        let resolved =
                            self.tcxt.patch_generic_from_path(ty.get(), &self.func.generics);
                        if let Some(res) = resolved {
                            ty.set(res);
                        }
                    }
                }
            }

            // Fix enum inits parsed as call expressions
            for stmt in unsafe { func.stmts.stmts.iter_mut_shared() } {
                NameResUserTypes { res: &self.name_res, tcxt: self, func }.visit_stmt(stmt);
            }

            crate::visit::walk_func(self, func);

            if matches!(func.kind, FuncKind::Normal)
                && !matches!(func.ret.get().val, Ty::Void)
                && !self.var_func.func_return.contains(&func.ident)
            {
                self.errors.push_error(Error::error_with_span(
                    self,
                    func.span,
                    &format!(
                        "function `{}` has return type `{}` but no return statement",
                        func.ident,
                        func.ret.get().val
                    ),
                ));
            }

            // Remove any functions we added from the params
            for param in &func.params {
                if let Ty::Func { ident, params, ret } = &param.ty.get().val {
                    self.var_func
                        .name_func
                        .remove(&param.ident)
                        .expect("if we added it it must still be there");
                }
            }

            self.name_res.pop_scope_stack();
            self.curr_fn.take();
        }

        // stabilize order
        impls.sort_by(|a, b| a.span.start.cmp(&b.span.start));
        for imp in impls {
            self.name_res
                .add_to_scope_stack(Scope::Impl { file: imp.span.file_id, imp: imp.method.ident });

            self.curr_fn = Some(imp.method.ident);
            crate::visit::walk_func(self, &imp.method);

            self.name_res.pop_scope_stack();
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
            self.errors.push_error(Error::error_with_span(
                self,
                span,
                &format!("unused variable `{}`, remove or reference", unused),
            ));
        }
    }

    fn visit_trait(&mut self, tr: &'ast Trait) {
        if self.trait_solve.add_trait(tr).is_some() {
            self.errors.push_error(Error::error_with_span(
                self,
                tr.span,
                &format!("duplicate trait `{}` found", tr.path),
            ));
        } else {
            self.name_res.add_decl(
                tr.span.file_id,
                Scope::Trait { file: tr.span.file_id, trait_: *tr.path.segs.last().unwrap() },
            );

            // If there are generics anytime they are referenced it's a `Ty::Path` so we
            // need to convert them to `Ty::Generic {..}` if they match i.e.
            // `Ty::Path(T) == Ty::Genereic {T, bound}`
            if !tr.generics.is_empty() {
                for ty in tr.method.function().params.iter().map(|p| &p.ty) {
                    let patched = self.patch_generic_from_path(ty.get(), &tr.generics);
                    if let Some(t) = patched {
                        ty.set(t);
                    }
                }
                let patched =
                    self.patch_generic_from_path(tr.method.function().ret.get(), &tr.generics);
                if let Some(t) = patched {
                    tr.method.function().ret.set(t);
                }
            }
        }
    }

    fn visit_impl(&mut self, imp: &'ast Impl) {
        if let Err(e) = self.trait_solve.add_impl(imp) {
            self.errors.push_error(Error::error_with_span(
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
            // If there are generics anytime they are referenced it's a `Ty::Path` so we
            // need to convert them to `Ty::Generic {..}` if they match i.e.
            // `Ty::Path(T) == Ty::Genereic {T, bound}`
            if !func.generics.is_empty() {
                for ty in func.params.iter().map(|p| &p.ty) {
                    let patched = self.patch_generic_from_path(ty.get(), &func.generics);
                    if let Some(t) = patched {
                        ty.set(t);
                    }
                }
                let patched = self.patch_generic_from_path(func.ret.get(), &func.generics);
                if let Some(t) = patched {
                    func.ret.set(t);
                }
            }

            self.name_res.add_decl(
                func.span.file_id,
                Scope::Func { file: func.span.file_id, func: func.ident },
            );
            // Current function scope (also the name)
            self.curr_fn = Some(func.ident);

            if self.var_func.insert(func.span, func.ident).is_some() {
                self.errors.push_error(Error::error_with_span(
                    self,
                    func.span,
                    "function takes up same span as other function",
                ));
            }

            if func.generics.is_empty() && func.ret.get().val.has_generics() {
                self.errors.push_error(Error::error_with_span(
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
                        ret: box func.ret.get().val.clone(),
                        // TODO: this is a HACK: we should NOT do this weird special case thing with
                        // the `Ty::Func` where we fill the params with generic parameters.
                        //
                        // If we pass the actual parameter types the method panics, we could just
                        // not panic but it will need more investigation
                        // 12/5/21
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
            if func.ret.get().val.has_generics() {
                self.generic_res.collect_generic_usage(
                    &func.ret.get().val,
                    self.unique_id(),
                    0,
                    &[],
                    &mut vec![Node::Func(func.ident)],
                );

                let matching_gen =
                    func.generics.iter().any(|g| g.ident == *func.ret.get().val.generics()[0]);
                if !matching_gen {
                    self.errors.push_error(Error::error_with_span(
                        self,
                        func.span,
                        &format!(
                            "found return `{}` which is not a declared generic type",
                            func.ret.get().val
                        ),
                    ));
                }
            }

            if self.var_func.name_func.insert(func.ident.to_owned(), func).is_some() {
                self.errors.push_error(Error::error_with_span(
                    self,
                    func.span,
                    &format!("multiple function declaration `{}`", func.ident),
                ));
            } else {
                // Add explicit return to void functions with no last return stmt
                if matches!(func.ret.get().val, Ty::Void)
                    && !matches!(func.stmts.stmts.slice().last().map(|s| &s.val), Some(Stmt::Exit))
                    // Don't add a dummy Exit to functions with no block
                    && !matches!(func.kind, FuncKind::Linked | FuncKind::Extern)
                {
                    unsafe {
                        // TODO: HACK: to give the Exit a span...
                        func.stmts.stmts.push_shared(
                            Stmt::Exit.into_spanned(func.stmts.stmts.slice().last().unwrap().span),
                        );
                    }
                }
                // Finally add this to the global namespace since it is a usable identifier now
                self.global.insert(
                    func.ident,
                    // TODO: this is the correct usage of this `Ty` (see above TODO about
                    // `GenericResolver::collect_generic_params`)
                    Ty::Func {
                        ident: func.ident,
                        ret: box func.ret.get().val.clone(),
                        params: func.params.iter().map(|p| p.ty.get().val.clone()).collect(),
                    },
                );
            }
        } else {
            self.errors.push_error(Error::error_with_span(
                self,
                func.span,
                "function defined within function",
            ));
        }
        // We have left this functions scope
        self.curr_fn.take();
    }

    fn visit_params(&mut self, params: &[Param]) {
        if let Some(fn_id) = self.curr_fn {
            for Param { ident, ty, span } in params {
                if self.global.contains_key(ident) {
                    self.errors.push_error(Error::error_with_span(
                        self,
                        *span,
                        &format!("found parameter `{}` that conflicts with global name", ident),
                    ));
                }
                // This param is declared inside of a function
                self.name_res.add_item(
                    span.file_id,
                    Scope::Func { file: span.file_id, func: fn_id },
                    ItemIn::Var(*ident),
                );

                let ty = self
                    .name_res
                    .resolve_name(&ty.get().val, self)
                    .unwrap_or_else(|| ty.get().val.clone());

                // TODO: Do this for returns and any place we match for Ty::Generic {..}
                if ty.has_generics() {
                    self.generic_res.collect_generic_usage(
                        &ty,
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
                            f.generics.iter().find(|g| g.ident == *ty.generics()[0])
                        })
                        .is_some();

                    if !matching_gen {
                        self.errors.push_error(Error::error_with_span(
                            self,
                            *span,
                            &format!(
                                "found parameter `{}` which is not a declared generic type",
                                ty
                            ),
                        ));
                    }
                };
                if self
                    .var_func
                    .func_refs
                    .entry(fn_id)
                    .or_default()
                    .insert(*ident, ty.clone())
                    .is_some()
                {
                    self.errors.push_error(Error::error_with_span(
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
            self.errors.push_error(Error::error_with_span(self, DUMMY, &format!("{:?}", params)));
        }
    }

    fn visit_adt(&mut self, adt: &'ast Adt) {
        match adt {
            Adt::Struct(struc) => {
                if !struc.generics.is_empty() {
                    for ty in struc.fields.iter().map(|f| &f.ty) {
                        let patched = self.patch_generic_from_path(ty.get(), &struc.generics);
                        if let Some(t) = patched {
                            ty.set(t);
                        }
                    }
                }

                if self.name_struct.insert(struc.ident, struc).is_some() {
                    self.errors.push_error(Error::error_with_span(
                        self,
                        struc.span,
                        "duplicate struct names",
                    ));
                }

                // Add to namespace/scope
                for field in &struc.fields {
                    self.name_res.add_item(
                        struc.span.file_id,
                        Scope::Struct { file: struc.span.file_id, adt: struc.ident },
                        ItemIn::Field(field.ident),
                    );
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
                if !en.generics.is_empty() {
                    for ty in en.variants.iter().flat_map(|v| unsafe { v.types.iter_mut_shared() })
                    {
                        let patched = self.patch_generic_from_path(ty, &en.generics);
                        if let Some(t) = patched {
                            *ty = t;
                        }
                    }
                }

                if self.name_enum.insert(en.ident, en).is_some() {
                    self.errors.push_error(Error::error_with_span(
                        self,
                        en.span,
                        "duplicate struct names",
                    ));
                }

                // Add to namespace/scope
                for variant in &en.variants {
                    self.name_res.add_item(
                        en.span.file_id,
                        Scope::Enum { file: en.span.file_id, adt: en.ident },
                        ItemIn::Variant(variant.ident),
                    );
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
    }

    fn visit_const(&mut self, var: &'ast Const) {
        if let Some(fn_id) = self.curr_fn {
            if self.global.contains_key(&var.ident) {
                self.errors.push_error(Error::error_with_span(
                    self,
                    var.span,
                    &format!("found scoped const `{}` that conflicts with global name", var.ident),
                ));
            }

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
                self.errors.push_error(Error::error_with_span(
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
                ItemIn::Var(var.ident),
            );

            // bail out before we set the scope as global
            return;
        } else {
            if self.global.insert(var.ident, var.ty.val.clone()).is_some() {
                self.errors.push_error(Error::error_with_span(
                    self,
                    var.span,
                    &format!("global variable `{}` is already declared", var.ident),
                ));
            }
            self.expr_ty.insert(&var.init, var.init.val.type_of().unwrap());
        }

        // TODO: if we don't record global exprs (so far only for arrays) we panic in lowering
        // because we don't have a type in the expr_ty map
        match &var.init.val {
            Expr::Ident(_) => todo!(),
            Expr::Array { ident, exprs } => todo!(),
            Expr::Urnary { op, expr } => todo!(),
            Expr::Binary { op, lhs, rhs } => todo!(),
            Expr::Parens(_) => todo!(),
            Expr::StructInit { path, fields } => todo!(),
            Expr::EnumInit { path, variant, items } => todo!(),
            Expr::ArrayInit { items } => {
                for ex in items {
                    self.expr_ty.insert(
                        ex,
                        if let Ty::Array { ty, .. } = &var.ty.val {
                            ty.val.clone()
                        } else {
                            self.errors.push_error(Error::error_with_span(
                                self,
                                var.span,
                                "type and expression do not agree",
                            ));
                            return;
                        },
                    );
                }
            }
            Expr::Value(..) => {}
            _ => {
                self.errors.push_error(Error::error_with_span(
                    self,
                    var.init.span,
                    "invalid const expression",
                ));
            }
        }

        // This const is declared in the file scope
        self.name_res.add_item(
            var.span.file_id,
            Scope::Global { file: var.span.file_id, name: var.ident },
            ItemIn::Var(var.ident),
        );
        self.var_func
            .unsed_vars
            .insert(ScopedName::global(var.span.file_id, var.ident), (var.span, Cell::new(false)));
    }

    /// We overwrite this so that no type checking of the arm statements happens until we
    /// gather the nested scope from binding in match arms.
    ///
    /// See `StmtCheck::visit_stmt` for what happens.
    fn visit_match_arm(&mut self, _arms: &'ast [MatchArm]) {}

    fn visit_stmt(&mut self, stmt: &'ast Statement) {
        self.errors.poisoned(false);

        // Collect all the `let x = ..` assignments and add them to our current scope (whatever
        // function scope we are in)
        self.name_res.visit_stmt(stmt);

        let mut infer = TypeInfer { tcxt: self };
        infer.visit_stmt(stmt);

        crate::visit::walk_stmt(self, stmt);

        self.set_record_used_vars(true);

        // check the statement after walking incase there were var declarations
        let mut check = StmtCheck { tcxt: self };
        check.visit_stmt(stmt);
    }

    fn visit_expr(&mut self, expr: &'ast Expression) {
        if self.errors.is_poisoned() {
            return;
        }

        match &expr.val {
            Expr::Ident(var_name) => {
                if let Some(ty) = self.type_of_ident(*var_name, expr.span) {
                    // TODO: at this point all types should be known but it there may be a few
                    // stragglers especially parts of field access ie.
                    // `thing.x.y.z` (check)
                } else {
                    self.errors.push_error(Error::error_with_span(
                        self,
                        expr.span,
                        "no type found for ident expr",
                    ));
                    self.errors.poisoned(true);
                }
            }
            Expr::Array { ident, exprs } => {
                for ex in exprs {
                    self.visit_expr(ex);
                }

                for e in exprs {
                    let ty = self.expr_ty.get(e);
                    if !matches!(ty, Some(Ty::Int)) {
                        self.errors.push_error(Error::error_with_span(
                            self,
                            expr.span,
                            &format!(
                                "[E0ty] cannot index array with {}",
                                ty.map_or("<unknown>".to_owned(), |t| t.to_string())
                            ),
                        ));
                        self.errors.poisoned(true);
                    }
                }
                if let Some(ty) = self.type_of_ident(*ident, expr.span) {
                    // if self.expr_ty.insert(expr, ty).is_some() {
                    // Ok because of `x[0] += 1;` turns into `x[0] = x[0] + 1;`
                    // }
                } else {
                    self.errors.push_error(Error::error_with_span(
                        self,
                        expr.span,
                        "[E0ty] no type found for array expr",
                    ));
                    self.errors.poisoned(true);
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
                            self.errors.push_error(Error::error_with_span(
                                self,
                                expr.span,
                                "cannot negate non bool type",
                            ));
                            self.errors.poisoned(true);
                        }
                    }
                    UnOp::OnesComp => {
                        // TODO: think about pointer maths
                        if let Some(Ty::Int | Ty::Ptr(_)) = ty {
                            self.expr_ty.insert(expr, Ty::Int);
                        } else {
                            self.errors.push_error(Error::error_with_span(
                                self,
                                expr.span,
                                "[E0ty] cannot negate non bool type",
                            ));
                            self.errors.poisoned(true);
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

                let t = if let Some(ty) = self.expr_ty.get(&**inner_expr) {
                    if matches!(
                        inner_expr.val,
                        Expr::Ident(..)
                            | Expr::Array { .. }
                            | Expr::AddrOf(..)
                            | Expr::FieldAccess { .. }
                    ) {
                        Some(ty.clone())
                    } else {
                        self.errors.push_error(Error::error_with_span(
                            self,
                            inner_expr.span,
                            "[E0ty] cannot take the address of an rvalue",
                        ));
                        self.errors.poisoned(true);
                        None
                    }
                } else {
                    self.errors.push_error(Error::error_with_span(
                        self,
                        expr.span,
                        &format!("[E0ty] identifier `{}` not found", inner_expr.val.debug_ident()),
                    ));
                    self.errors.poisoned(true);
                    None
                };
                if let Some(ty) = t {
                    self.expr_ty.insert(expr, Ty::Ptr(box ty.into_spanned(DUMMY)));
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);

                let lhs_ty = self.expr_ty.get(&**lhs);
                let rhs_ty = self.expr_ty.get(&**rhs);

                if let Some(ty) = fold_ty(
                    self,
                    resolve_ty(self, lhs, lhs_ty).as_ref(),
                    resolve_ty(self, rhs, rhs_ty).as_ref(),
                    op,
                    expr.span,
                ) {
                    // TODO: is this needed?? we are just overwriting the result of inference and
                    // checking that it was equal
                    if let Some(t2) = self.expr_ty.insert(expr, ty.clone()) {
                        if !ty.is_ty_eq(&t2) {
                            self.errors.push_error(Error::error_with_span(
                                self,
                                expr.span,
                                "ICE: something went wrong in the compiler",
                            ));
                            self.errors.poisoned(true);
                        }
                    }
                } else {
                    self.errors.push_error(Error::error_with_span(
                        self,
                        expr.span,
                        &format!("[E0ty] no type found for bin expr {:?} != {:?}", lhs_ty, rhs_ty),
                    ));
                    self.errors.poisoned(true);
                }
            }
            Expr::Parens(inner_expr) => {
                self.visit_expr(inner_expr);
                if let Some(ty) = self.expr_ty.get(&**inner_expr).cloned() {
                    if let Some(t2) = self.expr_ty.insert(expr, ty.clone()) {
                        if !ty.is_ty_eq(&t2) {
                            self.errors.push_error(Error::error_with_span(
                                self,
                                expr.span,
                                "ICE: something went wrong in the compiler",
                            ));
                            self.errors.poisoned(true);
                        }
                    }
                } else {
                    self.errors.push_error(Error::error_with_span(
                        self,
                        expr.span,
                        "[E0ty] no type found for paren expr",
                    ));
                    self.errors.poisoned(true);
                }
            }
            Expr::Call { path, args, type_args } => {
                let ident = path.segs.last().unwrap();

                // TODO: this is also kinda a HACK (similar to above from same commit)
                // If this is a function pointer record it's use
                if let Some(f) = self.var_func.get_fn_by_span(expr.span) {
                    if let Some((_, b)) = self.var_func.unsed_vars.get(&ScopedName::func_scope(
                        f,
                        *ident,
                        expr.span.file_id,
                    )) {
                        b.set(true);
                    }
                }

                for arg in args {
                    self.visit_expr(arg);
                }

                // Check type_args agrees
                let stack = build_stack(self, Node::Func(*ident));

                let gen_arg_set_id = self.unique_id();
                let mut gen_arg_map = HashMap::default();
                let func = if let Some(f) = self.var_func.name_func.get(ident) {
                    f
                } else {
                    self.errors.push_error(Error::error_with_span(
                        self,
                        expr.span,
                        &format!("[E0ty] no function named `{}` defined", path),
                    ));
                    return;
                };

                // Iter the type arguments at the call site
                for (gen_arg_idx, ty_arg) in type_args.iter().enumerate() {
                    // Don't use the same stack for each iteration
                    let mut stack = stack.clone();

                    let gen = &func.generics[gen_arg_idx];
                    // Find the param that is the "generic" and check against type argument
                    let mut arguments = vec![];
                    for (i, p) in func.params.iter().enumerate() {
                        if gen.is_ty_eq(&p.ty.get().val) {
                            arguments.push(TyRegion::Expr(&args[i].val));
                        }
                    }

                    self.generic_res.collect_generic_usage(
                        &ty_arg.val,
                        gen_arg_set_id,
                        gen_arg_idx,
                        &arguments,
                        &mut stack,
                    );

                    gen_arg_map.insert(gen.ident, ty_arg.val.clone());
                }

                let func_params = self
                    .var_func
                    .name_func
                    .get(ident)
                    .map(|f| &f.params)
                    .expect("function is known with params");

                if args.len() != func_params.len() {
                    self.errors.push_error(Error::error_with_span(
                        self,
                        expr.span,
                        "[E0ty] wrong number of arguments",
                    ));
                    self.errors.poisoned(true);
                }

                for (idx, arg) in args.iter().enumerate() {
                    let mut param_ty = func_params.get(idx).map(|p| p.ty.get().val.clone());
                    let mut arg_ty = self.expr_ty.get(arg).cloned();

                    // The call to `replace_with_concrete_ty` is needed to fill nested types in ie.
                    // &T or foo<T>
                    if let Some(ty_arg) =
                        replace_with_concrete_ty(self, param_ty.as_ref(), &gen_arg_map)
                    {
                        param_ty = Some(ty_arg);
                    }

                    if !param_ty.as_ref().is_ty_eq(&arg_ty.as_ref()) {
                        self.errors.push_error(Error::error_with_span(
                            self,
                            arg.span,
                            &format!(
                                "[E0ty] call with wrong argument type\nfound `{}` expected `{}`",
                                arg_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                                param_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                            ),
                        ));
                        self.errors.poisoned(true);
                    }
                }
            }
            Expr::TraitMeth { trait_, args, type_args } => {
                let ident = *trait_.segs.last().unwrap();
                if self.trait_solve.traits.get(trait_).is_none() {
                    self.errors.push_error(Error::error_with_span(
                        self,
                        expr.span,
                        &format!("[E0ty] no trait named `{}`", trait_),
                    ));
                    self.errors.poisoned(true);
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
                    let gen = &trait_def.generics[gen_arg_idx];

                    let arguments = trait_def
                        .method
                        .function()
                        .params
                        .iter()
                        .enumerate()
                        .filter(|(_i, p)| gen.is_ty_eq(&p.ty.get().val))
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
                    let mut param_ty = func_params.get(idx).map(|p| p.ty.get().val.clone());
                    let arg_ty = self.expr_ty.get(arg).cloned();

                    if let Some(Ty::Generic { ident, .. }) = &param_ty {
                        has_generic = true;
                        if let Some(ty_arg) = gen_arg_map.get(ident).cloned() {
                            param_ty = Some(ty_arg);
                        }
                    }

                    if !param_ty.as_ref().is_ty_eq(&arg_ty.as_ref()) {
                        self.errors.push_error(Error::error_with_span(
                            self,
                            arg.span,
                            &format!(
                                "[E0ty] trait call with wrong argument type\nfound `{}` expected `{}`",
                                arg_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                                param_ty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                            ),
                        ));
                        self.errors.poisoned(true);
                    }
                }

                let generic_dependence = if has_generic { Some(stack) } else { None };
                self.trait_solve.to_solve(
                    ident,
                    type_args.iter().map(|t| &t.val).collect::<Vec<_>>(),
                    generic_dependence,
                );

                // TODO: remove once we make trait calls like function calls in inference 11/26/21
                let def_fn = self.trait_solve.traits.get(trait_).expect("trait is defined");
                let t = &def_fn.method.return_ty().val;
                let ret_ty = if t.has_generics() {
                    subs_type_args(t, type_args, &trait_def.generics)
                } else {
                    t.clone()
                };
                self.expr_ty.insert(expr, ret_ty);
            }
            Expr::FieldAccess { lhs, rhs } => {
                self.visit_expr(lhs);

                // rhs is saved in `check_field_access`
                let field_ty = check_field_access(self, lhs, rhs);
                if let Some(ty) = field_ty {
                    self.expr_ty.insert(expr, ty);
                    // no is_some check: because of `x.y += 1;` being lowered to `x.y = x.y + 1;`
                } else {
                    // TODO: this error is crappy
                    self.errors.push_error(Error::error_with_span(
                        self,
                        expr.span,
                        "[E0ty] no type found for field access",
                    ));
                    self.errors.poisoned(true);
                }
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
                        .find_map(|f| {
                            if f.ident == *ident {
                                self.name_res.resolve_name(&f.ty.get().val, self)
                            } else {
                                None
                            }
                        })
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

                            gen_args.insert(*gen, exprty.clone().unwrap().into_spanned(DUMMY));
                        } else {
                            panic!("undefined generic type used")
                        }
                    }

                    // TODO: make sure this happens in StmtCheck?
                    // Skip checking type equivalence
                    if field_ty.has_generics() {
                        continue;
                    }

                    if !exprty.as_ref().is_ty_eq(&Some(&field_ty)) {
                        self.errors.push_error(Error::error_with_span(
                            self,
                            init.span,
                            &format!(
                                "[E0ty] field initialized with mismatched type\nfound `{}` expected `{}`",
                                exprty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                                field_ty,
                            ),
                        ));
                        self.errors.poisoned(true);
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
                        self.errors.push_error(Error::error_with_span(
                            self,
                            expr.span,
                            &format!("[E0ty] enum `{}` has no variant `{}`", path, variant),
                        ));
                        self.errors.poisoned(true);
                        return;
                    };

                let mut gen_args = HashMap::default();
                for (_idx, (item, variant_ty)) in
                    items.iter().zip(found_variant.types.slice()).enumerate()
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
                        self.errors.push_error(Error::error_with_span(
                            self,
                            item.span,
                            &format!(
                                "[E0ty] enum tuple initialized with mismatched type\nfound `{}` expected `{}`",
                                exprty.map_or("<unknown>".to_owned(), |t| t.to_string()),
                                variant_ty.val,
                            ),
                        ));
                        self.errors.poisoned(true);
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
            Expr::Value(val) => {
                // inference collects these
            }
            Expr::Builtin(bin) => {
                if let Builtin::SizeOf(t) = bin {
                    if t.get().val.has_generics() {
                        let mut stack =
                            build_stack(self, Node::Builtin(Ident::new(expr.span, "size_of")));
                        self.generic_res.collect_generic_usage(
                            &t.get().val,
                            self.unique_id(),
                            0,
                            &[TyRegion::Expr(&expr.val)],
                            &mut stack,
                        );
                    }
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
crate fn check_field_access<'ast>(
    tcxt: &mut TyCheckRes<'ast, '_>,
    lhs: &'ast Expression,
    rhs: &'ast Expression,
) -> Option<Ty> {
    fn field_access(ty: &Ty) -> Option<Ty> {
        Some(match ty {
            Ty::Struct { ident, gen } => ty.clone(),
            Ty::Enum { ident, gen } => ty.clone(),
            Ty::Ptr(inner) => field_access(&inner.val)?,
            Ty::Ref(inner) => field_access(&inner.val)?,
            Ty::ConstStr(_) => Ty::Struct { ident: Ident::new(DUMMY, "__const_str"), gen: vec![] },
            _ => return None,
        })
    }

    // Because we use `check_field_access` in the infer phase we can't rely on
    // `tcxt.expr_ty.get()`, we do collect the lhs expr
    let lhs_ty = tcxt.type_of_ident(lhs.val.as_ident(), lhs.span);

    let (name, struc) =
        if let Some(Ty::Struct { ident, .. }) = lhs_ty.as_ref().and_then(field_access) {
            // FIXME: come on clone here that's cray
            (ident, (*tcxt.name_struct.get(&ident).expect("no struct definition found")).clone())
        } else {
            tcxt.errors.push_error(Error::error_with_span(
                tcxt,
                lhs.span,
                &format!(
                    "[E0ty] not valid field access `{}`",
                    lhs_ty.map_or("<unknown>".into(), |t| t.to_string())
                ),
            ));
            tcxt.errors.poisoned(true);
            return None;
        };

    let opt_ident_type = |ident: Ident, tcxt: &TyCheckRes<'_, '_>| -> Option<Ty> {
        if let Some(rty) = struc.fields.iter().find_map(|f| {
            if f.ident == ident {
                Some(f.ty.get().val.clone())
            } else {
                None
            }
        }) {
            Some(rty)
        } else {
            tcxt.errors.push_error(Error::error_with_span(
                tcxt,
                struc.span,
                &format!("[E0ty] no field `{}` found for struct `{}`", ident, name,),
            ));
            tcxt.errors.poisoned(true);
            None
        }
    };

    match &rhs.val {
        Expr::Ident(ident) => {
            let rty = opt_ident_type(*ident, tcxt)?;
            tcxt.expr_ty.insert(rhs, rty.clone());
            Some(rty)
        }
        Expr::Array { ident, exprs } => {
            for expr in exprs {
                tcxt.visit_expr(expr);
            }

            let rty = opt_ident_type(*ident, tcxt)?;

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

/// This is used in the collection of expressions and ONLY checks that it is an expressing kind that
/// can be dereferenced, there is no type checking (can this pointer type be deref'ed this many
/// times).
fn check_dereference(tcxt: &mut TyCheckRes<'_, '_>, expr: &Expression) {
    match &expr.val {
        Expr::Ident(id) => {
            let ty = tcxt.type_of_ident(*id, expr.span).or_else(|| tcxt.expr_ty.get(expr).cloned());
            if let Some(_ty) = ty {
                // MAYBE: check and add expr to expr_ty map
            } else {
                tcxt.errors.push_error(Error::error_with_span(
                    tcxt,
                    expr.span,
                    &format!(
                        "[E0ty] cannot dereference `{}`",
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
                // MAYBE: check and add expr to expr_ty map
            } else {
                tcxt.errors.push_error(Error::error_with_span(
                    tcxt,
                    expr.span,
                    &format!(
                        "[E0ty] cannot dereference array `{}`",
                        ty.map_or("<unknown>".to_owned(), |t| t.to_string())
                    ),
                ));
                tcxt.errors.poisoned(true);
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
        | Expr::Builtin(..)
        | Expr::Value(_) => todo!(),
    }
}

/// This should not do any name resolution or path -> generic patching but instead only pull out
/// generic types and replace them with a concrete one.
///
/// If we have a generic type nested in a type this replaces it `foo<&T>` will become
/// `foo<&concrete_type>`.
fn replace_with_concrete_ty(
    tcxt: &TyCheckRes<'_, '_>,
    param_ty: Option<&Ty>,
    gen_arg_map: &HashMap<Ident, Ty>,
) -> Option<Ty> {
    Some(match param_ty? {
        Ty::Generic { ident, .. } => gen_arg_map.get(ident)?.clone(),
        t @ Ty::Path(..) => {
            println!("should be resolved");
            tcxt.name_res.resolve_name(t, tcxt)?
        }
        Ty::Array { size, ty: arrty } => Ty::Array {
            size: *size,
            ty: box replace_with_concrete_ty(tcxt, Some(&arrty.val), gen_arg_map)?
                .into_spanned(DUMMY),
        },
        Ty::Struct { ident, gen } => Ty::Struct {
            ident: *ident,
            gen: gen
                .iter()
                .map(|g| {
                    replace_with_concrete_ty(tcxt, Some(&g.val), gen_arg_map)
                        .map(|t| t.into_spanned(DUMMY))
                })
                .collect::<Option<Vec<_>>>()?,
        },
        Ty::Enum { ident, gen } => Ty::Enum {
            ident: *ident,
            gen: gen
                .iter()
                .map(|g| {
                    replace_with_concrete_ty(tcxt, Some(&g.val), gen_arg_map)
                        .map(|t| t.into_spanned(DUMMY))
                })
                .collect::<Option<Vec<_>>>()?,
        },
        Ty::Ptr(inner) => Ty::Ptr(
            box replace_with_concrete_ty(tcxt, Some(&inner.val), gen_arg_map)?.into_spanned(DUMMY),
        ),
        Ty::Ref(inner) => Ty::Ref(
            box replace_with_concrete_ty(tcxt, Some(&inner.val), gen_arg_map)?.into_spanned(DUMMY),
        ),
        _ => {
            return None;
        }
    })
}
