use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction},
    module::Module,
    passes::PassManager,
    targets::{InitializationConfig, Target},
    types::{AnyTypeEnum, BasicType, BasicTypeEnum},
    values::FunctionValue,
    AddressSpace, OptimizationLevel,
};

use crate::{
    ast::types::{
        Adt, BinOp, Binding, Block, Decl, Expr, Expression, Field, FieldInit, Func, Generic, Impl,
        MatchArm, Param, Pat, Range, Spanned, Spany, Statement, Stmt, Struct, Trait, Ty, Type,
        TypeEquality, UnOp, Val, Value, Var, Variant, DUMMY,
    },
    error::Error,
    typeck::TyCheckRes,
    visit::Visit,
};

#[derive(Debug)]
crate struct CodeGen<'ctx, 'input> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    pass: PassManager<FunctionValue<'ctx>>,
    tyctx: &'ctx TyCheckRes<'ctx, 'input>,
}

impl<'ctx, 'input> CodeGen<'ctx, 'input> {
    crate fn new(
        ctxt: &'ctx Context,
        tyctx: &'ctx TyCheckRes<'ctx, 'input>,
    ) -> CodeGen<'ctx, 'input> {
        Target::initialize_native(&InitializationConfig::default())
            .expect("Failed to initialize native target");

        let builder = ctxt.create_builder();
        let module = ctxt.create_module("libfile");
        let execution_engine = module.create_execution_engine().unwrap();

        let pass = PassManager::create(&module);
        pass.add_instruction_combining_pass();
        pass.add_cfg_simplification_pass();
        pass.add_basic_alias_analysis_pass();
        pass.add_promote_memory_to_register_pass();
        pass.add_reassociate_pass();
        pass.add_gvn_pass();
        pass.initialize();

        Self { context: ctxt, module, builder, execution_engine, pass, tyctx }
    }
}

impl<'ast, 'input> Visit<'ast> for CodeGen<'ast, 'input> {
    fn visit_var(&mut self, var: &'ast Var) {
        let global = self.module.add_global(
            var.ty.val.const_llvm_lower(self.context),
            Some(AddressSpace::Global),
            &var.ident,
        );
    }
    fn visit_func(&mut self, func: &'ast Func) {
        // self.context.struct_type(field_types, packed);
    }
}

impl Ty {
    fn const_llvm_lower<'ctx>(&self, context: &'ctx Context) -> BasicTypeEnum<'ctx> {
        match self {
            Ty::Array { size, ty } => {
                BasicTypeEnum::ArrayType(ty.val.const_llvm_lower(context).array_type(*size as u32))
            }
            Ty::Struct { ident, gen } => {
                BasicTypeEnum::StructType(context.opaque_struct_type(ident))
            }
            Ty::Enum { ident, gen } => BasicTypeEnum::StructType(context.opaque_struct_type(ident)),
            Ty::String => BasicTypeEnum::ArrayType(context.i16_type().array_type(0)),
            Ty::Int => context.i64_type().into(),
            Ty::Char => todo!(),
            Ty::Float => todo!(),
            Ty::Bool => context.bool_type().into(),
            t => todo!("{:?}", t),
        }
    }
}
