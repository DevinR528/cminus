use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction},
    module::Module,
    passes::PassManager,
    targets::{InitializationConfig, Target},
    values::FunctionValue,
    OptimizationLevel,
};

use crate::{
    ast::types::{
        Adt, BinOp, Binding, Block, Decl, Expr, Expression, Field, FieldInit, Func, Generic, Impl,
        MatchArm, Param, Pat, Range, Spanned, Spany, Statement, Stmt, Struct, Trait, Ty, Type,
        TypeEquality, UnOp, Val, Value, Var, Variant, DUMMY,
    },
    error::Error,
    visit::Visit,
};

#[derive(Debug)]
crate struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    pass: PassManager<FunctionValue<'ctx>>,
}

impl<'ctx> CodeGen<'ctx> {
    crate fn new(ctxt: &'ctx Context) -> CodeGen<'ctx> {
        Target::initialize_native(&InitializationConfig::default())
            .expect("Failed to initialize native target");

        let builder = ctxt.create_builder();
        let module = ctxt.create_module("libfile");
        let execution_engine = module.create_execution_engine().unwrap();

        let pass = PassManager::create(&module);
        pass.add_instruction_combining_pass();
        pass.add_instruction_simplify_pass();
        pass.add_cfg_simplification_pass();
        pass.add_basic_alias_analysis_pass();
        pass.add_promote_memory_to_register_pass();
        pass.add_reassociate_pass();
        pass.add_gvn_pass();
        pass.initialize();

        Self { context: ctxt, module, builder, execution_engine, pass }
    }
}

impl<'ast> Visit<'ast> for CodeGen<'_> {
    fn visit_prog(&mut self, items: &'ast [crate::ast::types::Declaration]) {
        for item in items {
            crate::visit::walk_decl(self, item)
        }
    }
}
