use std::{collections::HashMap, path::Path, vec};

use either::Either;
use inkwell::{
    builder::Builder,
    context::Context,
    execution_engine::{ExecutionEngine, JitFunction},
    module::{Linkage, Module},
    passes::PassManager,
    targets::{CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine},
    types::{AnyTypeEnum, BasicType, BasicTypeEnum, FunctionType, PointerType},
    values::{
        AggregateValue, BasicValue, BasicValueEnum, FunctionValue, InstructionOpcode, PointerValue,
    },
    AddressSpace, OptimizationLevel,
};

use crate::{
    error::Error,
    lir::{
        lower::{
            Adt, BinOp, CallExpr, Enum, Expr, Func, Impl, Item, LValue, MatchArm, Stmt, Struct, Ty,
            Val, Var,
        },
        visit::Visit,
    },
    typeck::TyCheckRes,
};

impl Ty {
    fn as_llvm_type<'ctx>(&self, context: &'ctx Context) -> BasicTypeEnum<'ctx> {
        match self {
            Ty::Array { size, ty } => {
                BasicTypeEnum::ArrayType(ty.as_llvm_type(context).array_type(*size as u32))
            }
            Ty::Struct { ident, gen, def } => BasicTypeEnum::StructType(context.struct_type(
                &def.fields.iter().map(|f| f.ty.as_llvm_type(context)).collect::<Vec<_>>(),
                false,
            )),
            Ty::Enum { ident, gen, def } => {
                BasicTypeEnum::StructType(context.opaque_struct_type(ident))
            }
            Ty::String => BasicTypeEnum::ArrayType(context.i16_type().array_type(0)),
            Ty::Int => context.i64_type().into(),
            Ty::Char => context.i8_type().into(),
            Ty::Float => context.f64_type().into(),
            Ty::Bool => context.bool_type().into(),
            Ty::Ptr(t) => t.as_llvm_type(context).ptr_type(AddressSpace::Generic).into(),
            hmm => todo!("{:?}", hmm),
        }
    }
}

#[derive(Debug)]
crate struct LLVMGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
    pass: PassManager<FunctionValue<'ctx>>,
    machine: TargetMachine,
    vars: HashMap<&'ctx str, BasicValueEnum<'ctx>>,
    path: &'ctx Path,
}

impl<'ctx> LLVMGen<'ctx> {
    crate fn new(ctxt: &'ctx Context, path: &'ctx Path) -> LLVMGen<'ctx> {
        Target::initialize_native(&InitializationConfig::default())
            .expect("Failed to initialize native target");

        let triple = TargetMachine::get_default_triple();
        let cpu = TargetMachine::get_host_cpu_name().to_string();
        let features = TargetMachine::get_host_cpu_features().to_string();
        let target = Target::from_triple(&triple).unwrap();

        let machine = target
            .create_target_machine(
                &triple,
                &cpu,
                &features,
                OptimizationLevel::Aggressive,
                RelocMode::Default,
                CodeModel::Default,
            )
            .unwrap();

        let builder = ctxt.create_builder();
        let module = ctxt.create_module(&path.file_prefix().unwrap().to_string_lossy());
        let execution_engine = module.create_execution_engine().unwrap();

        let pass = PassManager::create(&module);
        pass.add_instruction_combining_pass();
        pass.add_cfg_simplification_pass();
        pass.add_basic_alias_analysis_pass();
        pass.add_promote_memory_to_register_pass();
        pass.add_reassociate_pass();
        pass.add_gvn_pass();
        pass.initialize();

        let mut this = Self {
            context: ctxt,
            module,
            builder,
            execution_engine,
            pass,
            machine,
            vars: HashMap::new(),
            path,
        };

        this.linked_func(
            "printf",
            vec![this.context.i8_type().ptr_type(AddressSpace::Generic).into()],
            this.context.i8_type().into(),
        )
        .unwrap();

        this
    }

    crate fn dump_asm(&self) -> Result<(), String> {
        let mut p = self.path.to_path_buf();
        p.set_extension("s");
        self.machine.write_to_file(&self.module, FileType::Assembly, &p).map_err(|e| e.to_string())
    }

    fn coerce_store_ptr_val(&self, ptr: BasicValueEnum<'_>, val: BasicValueEnum<'_>) {
        match ptr {
            BasicValueEnum::ArrayValue(arr) => unreachable!("arrays are never left value types"),
            BasicValueEnum::IntValue(_) | BasicValueEnum::FloatValue(_) => {
                unreachable!("const fold removed all constant expressions")
            }
            BasicValueEnum::PointerValue(mut inner) => {
                if inner.get_type().get_element_type().is_pointer_type() {
                    inner = self.builder.build_load(inner, "coerce_deref").into_pointer_value();
                }
                self.builder.build_store(
                    inner,
                    match val {
                        BasicValueEnum::PointerValue(p)
                            if !inner.get_type().get_element_type().is_pointer_type() =>
                        {
                            self.builder.build_load(p, "deref")
                        }
                        _ => val,
                    },
                );
            }
            BasicValueEnum::StructValue(_) => todo!(),
            BasicValueEnum::VectorValue(_) => todo!(),
        }
    }

    fn deref_to_value(&self, ptr: BasicValueEnum<'ctx>, ty: &Ty) -> BasicValueEnum<'ctx> {
        match ty {
            Ty::Ptr(t) => {
                if let BasicValueEnum::PointerValue(_) = ptr {
                    ptr
                } else {
                    unreachable!("ptr type not ptr value")
                }
            }
            Ty::Ref(_) => todo!("{:?} == {:?}", ptr, ty),
            Ty::Array { .. }
            | Ty::Struct { .. }
            | Ty::Enum { .. }
            | Ty::String
            | Ty::Int
            | Ty::Char
            | Ty::Float
            | Ty::Bool => {
                if let BasicValueEnum::PointerValue(ptr) = ptr {
                    self.deref_to_value(self.builder.build_load(ptr, "deref_to_val"), ty)
                } else {
                    ptr
                }
            }
            _ => todo!(),
        }
    }

    fn index_arr(
        &self,
        arr_ptr: PointerValue<'ctx>,
        idx_exprs: &'ctx [Expr],
    ) -> Option<PointerValue<'ctx>> {
        let mut indexes = idx_exprs
            .iter()
            .map(|e| Some(self.build_value(e, None)?.into_int_value()))
            .collect::<Option<Vec<_>>>()?;

        // Always index with i64 types
        //
        // This is to access the value as a pointer (we do `address + 0` first, llvm oddity
        // kinda)
        indexes.insert(0, self.context.i64_type().const_zero());

        unsafe { self.builder.build_in_bounds_gep(arr_ptr, &indexes, "lhs_arr_index").into() }
    }

    fn create_entry_block_alloca(
        &self,
        name: &str,
        ty: &Ty,
        fnval: FunctionValue<'ctx>,
    ) -> PointerValue<'ctx> {
        let builder = self.context.create_builder();
        let entry = fnval.get_first_basic_block().unwrap();

        match entry.get_first_instruction() {
            Some(first_instr) => builder.position_before(&first_instr),
            None => builder.position_at_end(entry),
        }
        builder.build_alloca(ty.as_llvm_type(self.context), name)
    }

    fn get_pointer(&mut self, expr: &'ctx LValue) -> Option<BasicValueEnum<'ctx>> {
        Some(match expr {
            LValue::Ident { ident, ty } => self.vars.get(ident.as_str()).copied()?,
            LValue::Deref { indir, expr } => self.get_pointer(expr)?,
            LValue::Array { ident, exprs } => {
                let arr_ptr = self.vars.get(ident.as_str()).copied()?;
                self.index_arr(arr_ptr.into_pointer_value(), exprs)?.into()
            }
            LValue::FieldAccess { lhs, rhs } => todo!(),
        })
    }

    fn build_value(
        &self,
        expr: &'ctx Expr,
        assigned: Option<&str>,
    ) -> Option<BasicValueEnum<'ctx>> {
        Some(match expr {
            Expr::Ident { ident, ty } => self.vars.get(ident.as_str()).copied()?,
            Expr::Deref { indir, expr, ty } => {
                let mut ptr = self.build_value(expr, None)?;
                let mut t = ty;
                while let Ty::Ref(next) = t {
                    ptr = self.builder.build_load(ptr.into_pointer_value(), "ptr_deref");
                    t = next;
                }
                ptr
            }
            Expr::AddrOf(expr) => {
                let val = self.build_value(expr, None)?;
                val
            }
            Expr::Array { ident, exprs, ty } => {
                let arr_ptr = self.vars.get(ident.as_str()).copied()?;
                let idx_ptr = self.index_arr(arr_ptr.into_pointer_value(), exprs)?;
                self.builder.build_load(idx_ptr, "arr_ele")
            }
            Expr::Urnary { op, expr, ty } => todo!(),
            Expr::Binary { op, lhs, rhs, ty } => {
                let lval = self.deref_to_value(self.build_value(lhs, None)?, ty);
                let rval = self.deref_to_value(self.build_value(rhs, None)?, ty);
                match op {
                    BinOp::Mul => match ty {
                        Ty::Int => BasicValueEnum::IntValue(self.builder.build_int_mul(
                            lval.into_int_value(),
                            rval.into_int_value(),
                            "intmul",
                        )),
                        Ty::Float => BasicValueEnum::FloatValue(self.builder.build_float_mul(
                            lval.into_float_value(),
                            rval.into_float_value(),
                            "floatmul",
                        )),
                        _ => unreachable!(),
                    },
                    BinOp::Div => todo!(),
                    BinOp::Rem => todo!(),
                    BinOp::Add => match ty {
                        Ty::Int => BasicValueEnum::IntValue(self.builder.build_int_add(
                            lval.into_int_value(),
                            rval.into_int_value(),
                            "intadd",
                        )),
                        Ty::Float => BasicValueEnum::FloatValue(self.builder.build_float_add(
                            lval.into_float_value(),
                            rval.into_float_value(),
                            "floatadd",
                        )),
                        _ => unreachable!(),
                    },
                    BinOp::Sub => todo!(),
                    BinOp::LeftShift => todo!(),
                    BinOp::RightShift => todo!(),
                    BinOp::Lt => todo!(),
                    BinOp::Le => todo!(),
                    BinOp::Ge => todo!(),
                    BinOp::Gt => todo!(),
                    BinOp::Eq => todo!(),
                    BinOp::Ne => todo!(),
                    BinOp::BitAnd => todo!(),
                    BinOp::BitXor => todo!(),
                    BinOp::BitOr => todo!(),
                    BinOp::And => todo!(),
                    BinOp::Or => todo!(),
                    BinOp::AddAssign => todo!(),
                    BinOp::SubAssign => todo!(),
                }
            }
            Expr::Parens(_) => todo!(),
            Expr::Call { ident, args, type_args, def } => {
                let func = self.module.get_function(ident).unwrap();
                let args =
                    args.iter().map(|e| self.build_value(e, None).unwrap()).collect::<Vec<_>>();
                match self.builder.build_call(func, &args, "calltmp").try_as_basic_value() {
                    Either::Left(val) => val,
                    Either::Right(inst) => todo!(),
                }
            }
            Expr::TraitMeth { trait_, args, type_args, def } => todo!(),
            Expr::FieldAccess { lhs, def, rhs, field_idx } => {
                let struct_ptr = self.build_value(lhs, None).unwrap().into_pointer_value();

                let field =
                    self.builder.build_struct_gep(struct_ptr, *field_idx, &def.ident).unwrap();
                match &**rhs {
                    Expr::Ident { ident, .. } => {
                        self.builder.build_load(field, &format!("{}.{}", def.ident, ident))
                    }
                    Expr::Deref { indir, expr, ty } => todo!(),
                    Expr::AddrOf(_) => todo!(),
                    Expr::Array { ident, exprs, ty } => {
                        let elptr = self.index_arr(field, exprs)?;
                        self.builder.build_load(elptr, &format!("{}.{}[]", def.ident, ident))
                    }
                    Expr::FieldAccess { lhs, def, rhs, field_idx } => todo!(),
                    _ => unreachable!("not a possible right side field access"),
                }
            }
            Expr::StructInit { name, fields, def } => {
                let struct_ptr = self.vars.get(assigned?).copied()?.into_pointer_value();
                for (idx, field) in fields.iter().enumerate() {
                    let val = self.build_value(&field.init, None).unwrap();
                    let ptr = self
                        .builder
                        .build_struct_gep(
                            struct_ptr,
                            idx as u32,
                            &format!("{}.{}.init", name, field.ident),
                        )
                        .unwrap();

                    self.builder.build_store(
                        ptr,
                        match val {
                            BasicValueEnum::PointerValue(p) => self.builder.build_load(p, "deref"),
                            _ => val,
                        },
                    );
                }
                struct_ptr.into()
            }
            Expr::EnumInit { ident, variant, items, def } => todo!(),
            Expr::ArrayInit { items, ty } => {
                let memptr = self.builder.build_alloca(ty.as_llvm_type(self.context), "arrinit");
                for (idx, expr) in items.iter().enumerate() {
                    let val = self.build_value(expr, None).unwrap();

                    let ptr = unsafe {
                        self.builder.build_in_bounds_gep(
                            memptr,
                            &[
                                self.context.i64_type().const_zero(),
                                self.context.i64_type().const_int(idx as u64, false),
                            ],
                            &format!("array.{}.init", idx),
                        )
                    };

                    self.builder.build_store(ptr, val);
                }
                memptr.into()
            }
            Expr::Value(v) => match v {
                Val::Float(f) => {
                    BasicValueEnum::FloatValue(self.context.f64_type().const_float(*f))
                }
                Val::Int(i) => {
                    BasicValueEnum::IntValue(self.context.i64_type().const_int(*i as u64, true))
                }
                Val::Char(c) => {
                    BasicValueEnum::IntValue(self.context.i8_type().const_int(*c as u64, false))
                }
                Val::Bool(b) => {
                    BasicValueEnum::IntValue(self.context.bool_type().const_int((*b) as u64, false))
                }
                Val::Str(s) => BasicValueEnum::ArrayValue(
                    self.context.i8_type().const_array(
                        &s.as_bytes()
                            .iter()
                            .map(|b| self.context.i8_type().const_int(*b as u64, false))
                            .collect::<Vec<_>>(),
                    ),
                ),
            },
        })
    }

    fn gen_statement(&mut self, fnval: FunctionValue<'ctx>, stmt: &'ctx Stmt) {
        match stmt {
            Stmt::VarDecl(vars) => {
                for var in vars {
                    let alloca = self.create_entry_block_alloca(&var.ident, &var.ty, fnval);
                    self.builder
                        .build_store(alloca, var.ty.as_llvm_type(self.context).const_zero());
                    self.vars.insert(&var.ident, alloca.as_basic_value_enum());
                }
            }
            Stmt::Assign { lval, rval } => {
                let mut lptr = self.get_pointer(lval).unwrap();
                let mut rvalue = self.build_value(rval, lval.as_ident()).unwrap();
                self.coerce_store_ptr_val(lptr, rvalue);
            }
            Stmt::Call { expr: CallExpr { ident, args, type_args }, def } => {
                let func = self.module.get_function(ident).unwrap();
                let args =
                    args.iter().map(|e| self.build_value(e, None).unwrap()).collect::<Vec<_>>();
                match self.builder.build_call(func, &args, "calltmp").try_as_basic_value() {
                    Either::Left(val) => {}
                    Either::Right(inst) => {
                        // eprintln!("{:?}", inst);
                    }
                }
            }
            Stmt::TraitMeth { expr, def } => todo!(),
            Stmt::If { cond, blk, els } => todo!(),
            Stmt::While { cond, stmt } => todo!(),
            Stmt::Match { expr, arms } => todo!(),
            Stmt::Read(_) => todo!(),
            Stmt::Write { expr } => {
                let function = self.module.get_function("printf").unwrap();
                let val =
                    self.deref_to_value(self.build_value(expr, None).unwrap(), &expr.type_of());

                let fmtstr =
                    self.builder.build_global_string_ptr("%d\n", "fmtstr").as_basic_value_enum();

                let args = vec![fmtstr, val];
                self.builder.build_call(function, &args, "printret");
            }
            Stmt::Ret(expr, ty) => {
                let value = self.deref_to_value(self.build_value(expr, None).unwrap(), ty);
                self.builder.build_return(Some(&value));
            }
            Stmt::Exit => {
                self.builder.build_return(None);
            }
            Stmt::Block(_) => todo!(),
        }
    }

    /// Add a function header (signature) to LLVM
    fn linked_func(
        &self,
        name: &str,
        args: Vec<BasicTypeEnum<'ctx>>,
        ret: BasicTypeEnum<'ctx>,
    ) -> Result<FunctionValue<'ctx>, &'static str> {
        let fn_type = ret.fn_type(&args, true);
        let fn_val = self.module.add_function(name, fn_type, Some(Linkage::External));

        // Zero is C calling convention
        // fn_val.set_call_conventions(0);

        for (i, arg) in fn_val.get_param_iter().enumerate() {
            if i == 0 {
                arg.set_name("fmt_str");
            }
            arg.set_name(&i.to_string());
        }
        Ok(fn_val)
    }

    /// Add a function header (signature) to LLVM
    fn compile_prototype(&self, func: &Func) -> Result<FunctionValue<'ctx>, &'static str> {
        let args_types =
            func.params.iter().map(|p| p.ty.as_llvm_type(self.context)).collect::<Vec<_>>();

        let fn_type = if matches!(&func.ret, Ty::Void) {
            self.context.void_type().fn_type(&args_types, false)
        } else {
            func.ret.as_llvm_type(self.context).fn_type(&args_types, false)
        };
        let fn_val = self.module.add_function(&func.ident, fn_type, None);

        for (i, arg) in fn_val.get_param_iter().enumerate() {
            arg.set_name(&func.params[i].ident);
        }

        Ok(fn_val)
    }
}

impl<'ast> Visit<'ast> for LLVMGen<'ast> {
    fn visit_var(&mut self, var: &'ast Var) {
        let global = self.module.add_global(
            var.ty.as_llvm_type(self.context),
            Some(AddressSpace::Global),
            &var.ident,
        );
        self.vars.insert(&var.ident, global.as_basic_value_enum());
    }

    fn visit_func(&mut self, func: &'ast Func) {
        let function = self.compile_prototype(func).unwrap();
        if func.stmts.is_empty() {
            function.print_to_stderr();
            return;
        }

        let entry = self.context.append_basic_block(function, &func.ident);

        self.builder.position_at_end(entry);

        // update fn field
        // build variables map

        for (i, arg) in function.get_param_iter().enumerate() {
            let alloca =
                self.create_entry_block_alloca(&func.params[i].ident, &func.params[i].ty, function);
            self.builder.build_store(alloca, arg);
            self.vars.insert(&func.params[i].ident, alloca.as_basic_value_enum());
        }

        for stmt in &func.stmts {
            self.gen_statement(function, stmt);
        }

        if matches!(func.ret, Ty::Void) && entry.get_terminator().is_none() {
            self.builder.build_return(None);
        }

        if function.verify(true) {
            self.pass.run_on(&function);
            function.print_to_stderr();
        } else {
            std::thread::sleep(std::time::Duration::from_secs(1));
            function.print_to_stderr();

            panic!("invalid generated function")
        }
    }
}
