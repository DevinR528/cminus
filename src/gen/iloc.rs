use std::{
    fs::{create_dir_all, OpenOptions},
    hash,
    io::{ErrorKind, Write},
    path::Path,
    vec,
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{
    ast::{
        parse::symbol::Ident,
        types::{self as ty, FuncKind, DUMMY},
    },
    data_struc::str_help::StripEscape,
    gen::iloc::inst::{Global, Instruction, Loc, Reg},
    lir::{
        lower::{
            BinOp, Binding, Block, Builtin, CallExpr, Const, Else, Expr, FieldInit, Func, LValue,
            MatchArm, Pat, Stmt, Struct, Ty, UnOp, Val,
        },
        visit::Visit,
    },
    typeck::scope::hash_any,
};

mod inst;

use inst::Operation;

#[derive(Debug)]
crate struct IlocGen<'ctx> {
    iloc_buf: Vec<Instruction>,

    stack_size: isize,
    ident_address: HashMap<Ident, isize>,
    /// If present this is the register that hold that memory address for this variable
    load_from_addr: HashMap<Ident, Reg>,

    vars: HashMap<Ident, Loc>,
    globals: HashMap<Ident, Global>,

    registers: HashMap<Ident, Reg>,
    expr_regs: HashMap<Operation, Reg>,
    values: HashMap<Val, Reg>,
    /// This is a map of `Val` -> (name of global, register global is loaded to, optional register
    /// for the conversion to float)
    global_regs: HashMap<Val, (Ident, Reg, Reg)>,

    /// This skips `%vr0, %vr1, %vr2, %vr3, %vr4` since they are special reserved registers.
    curr_register: usize,
    curr_label: usize,

    current_fn_params: HashSet<Ident>,
    path: &'ctx Path,
}

impl<'ctx> IlocGen<'ctx> {
    crate fn new(path: &'ctx Path) -> IlocGen<'ctx> {
        Self {
            iloc_buf: vec![],

            stack_size: 0,
            ident_address: HashMap::default(),
            load_from_addr: HashMap::default(),

            vars: HashMap::default(),
            globals: HashMap::default(),

            registers: HashMap::default(),
            expr_regs: HashMap::default(),
            values: HashMap::default(),
            global_regs: HashMap::default(),

            // `0` is a protected register (it's the stack/frame pointer)
            curr_register: 5,
            curr_label: 0,

            current_fn_params: HashSet::default(),
            path,
        }
    }

    crate fn dump_asm(&self) -> Result<(), String> {
        let mut build_dir = self.path.to_path_buf();
        let file = build_dir.file_name().unwrap().to_os_string();
        build_dir.pop();
        build_dir.push("build");

        if let Err(e) = create_dir_all(&build_dir) {
            if !matches!(e.kind(), ErrorKind::IsADirectory) {
                panic!("{}", e)
            }
        };

        build_dir.push(file);
        build_dir.set_extension("il");

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(build_dir)
            .map_err(|e| e.to_string())?;

        let globals = self
            .globals
            .values()
            .map(|decl| self.to_global(decl))
            .collect::<Vec<String>>()
            .join("\n");
        let assembly =
            self.iloc_buf.iter().map(|inst| inst.to_string()).collect::<Vec<String>>().join("\n");

        file.write_all(format!(".data\n{}\n.text\n{}\n", globals, assembly).as_bytes())
            .map_err(|e| e.to_string())
    }

    crate fn to_global(&self, glob: &Global) -> String {
        use std::fmt::Write;

        let mut buf = String::new();
        match glob {
            Global::Text { name, content } => {
                write!(buf, "    .string {}, {}", name, content);
            }
            Global::Array { name, content } => {
                write!(buf, "    .array {}, {}, [", name, content.len() * 4);
                write!(
                    buf,
                    "{}",
                    content.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ")
                );
                write!(buf, "]");
            }
            Global::Int { name, content } => {
                // TODO: confirm this works for global integers
                write!(buf, "    .global {}, 4, 4", name);
            }
            Global::Float { name, content } => {
                write!(buf, "    .float {}, {:.8}", name, content);
            }
            Global::Char { name, content } => {
                // TODO: there are no char's but try it
                write!(buf, "    .global {}, 4, 4", name);
            }
        };
        buf
    }

    fn next_label(&mut self) -> usize {
        let curr = self.curr_label;
        self.curr_label += 1;
        curr
    }

    fn ident_to_reg(&mut self, ident: Ident) -> Reg {
        let num = self.curr_register;
        if let Some(num) = self.registers.get(&ident) {
            *num
        } else {
            self.curr_register += 1;
            self.registers.insert(ident, Reg::Var(num));
            Reg::Var(num)
        }
    }
    fn value_to_reg(&mut self, val: Val) -> Reg {
        let num = self.curr_register;
        if let Some(num) = self.values.get(&val) {
            *num
        } else {
            self.curr_register += 1;
            self.values.insert(val, Reg::Var(num));
            Reg::Var(num)
        }
    }
    fn expr_to_reg(&mut self, expr: Operation) -> Reg {
        let num = self.curr_register;
        if let Some(num) = self.expr_regs.get(&expr) {
            *num
        } else {
            self.curr_register += 1;
            self.expr_regs.insert(expr, Reg::Var(num));
            Reg::Var(num)
        }
    }
    /// The returns the
    fn global_to_reg_float(&mut self, val: Val, name: Ident) -> (Ident, Reg, Reg) {
        let num = self.curr_register;
        if let Some(name_num) = self.global_regs.get(&val) {
            *name_num
        } else {
            self.curr_register += 2;
            self.global_regs.insert(val, (name, Reg::Var(num), Reg::Var(num + 1)));
            (name, Reg::Var(num), Reg::Var(num + 1))
        }
    }
    fn global_to_reg(&mut self, val: Val, name: Ident) -> (Ident, Reg) {
        let num = self.curr_register;
        if let Some(name_num) = self.global_regs.get(&val) {
            assert_eq!(name_num.1, name_num.2);
            (name_num.0, name_num.1)
        } else {
            self.curr_register += 1;
            self.global_regs.insert(val, (name, Reg::Var(num), Reg::Var(num)));
            (name, Reg::Var(num))
        }
    }

    fn int_binop(&mut self, op: BinOp, lhs_reg: Reg, rhs_reg: Reg) -> Reg {
        match op {
            BinOp::Mul => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::Mult { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            BinOp::Div => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::Div { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            BinOp::Rem => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::Mod { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            BinOp::Add => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::Add { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            BinOp::Sub => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::Sub { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            BinOp::LeftShift => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::LShift { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            BinOp::RightShift => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::RShift { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            // This, and the next few, are bitwise operations, in Iloc there is no
            // difference between logical and bitwise operations
            BinOp::BitAnd => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::And { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            BinOp::BitXor => {
                todo!("No exclusive OR in ILOC...")
            }
            BinOp::BitOr => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::Or { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            // Logical and/or operations, treated the same as bitwise in Iloc
            BinOp::And => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::And { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            BinOp::Or => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::Or { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            // Comparison operations
            BinOp::Lt => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                let test_dst = self.expr_to_reg(Operation::Test(op, dst));
                self.iloc_buf.extend([
                    Instruction::Comp { a: lhs_reg, b: rhs_reg, dst },
                    Instruction::TestGE { test: dst, dst: test_dst },
                ]);
                test_dst
            }
            BinOp::Le => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                let test_dst = self.expr_to_reg(Operation::Test(op, dst));
                self.iloc_buf.extend([
                    Instruction::Comp { a: lhs_reg, b: rhs_reg, dst },
                    Instruction::TestGT { test: dst, dst: test_dst },
                ]);
                test_dst
            }
            BinOp::Ge => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                let test_dst = self.expr_to_reg(Operation::Test(op, dst));
                self.iloc_buf.extend([
                    Instruction::Comp { a: lhs_reg, b: rhs_reg, dst },
                    Instruction::TestLT { test: dst, dst: test_dst },
                ]);
                test_dst
            }
            BinOp::Gt => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                let test_dst = self.expr_to_reg(Operation::Test(op, dst));
                self.iloc_buf.extend([
                    Instruction::Comp { a: lhs_reg, b: rhs_reg, dst },
                    Instruction::TestLE { test: dst, dst: test_dst },
                ]);
                test_dst
            }
            BinOp::Eq => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                let test_dst = self.expr_to_reg(Operation::Test(op, dst));
                self.iloc_buf.extend([
                    Instruction::Comp { a: lhs_reg, b: rhs_reg, dst },
                    Instruction::TestNE { test: dst, dst: test_dst },
                ]);
                test_dst
            }
            BinOp::Ne => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                let test_dst = self.expr_to_reg(Operation::Test(op, dst));
                self.iloc_buf.extend([
                    Instruction::Comp { a: lhs_reg, b: rhs_reg, dst },
                    Instruction::TestEQ { test: dst, dst: test_dst },
                ]);
                test_dst
            }
            // `lir::lower` converts all `op=` into the full expression
            BinOp::AddAssign | BinOp::SubAssign => {
                unreachable!("this is converted to a full add/sub expression")
            }
        }
    }
    fn float_binop(&mut self, op: BinOp, lhs_reg: Reg, rhs_reg: Reg) -> Reg {
        match op {
            BinOp::Add => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::FAdd { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            BinOp::Sub => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::FSub { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            BinOp::Mul => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::FMult { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            BinOp::Div => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                self.iloc_buf.push(Instruction::FDiv { src_a: lhs_reg, src_b: rhs_reg, dst });
                dst
            }
            // Comparison operations
            BinOp::Le => todo!(),
            BinOp::Ge => todo!(),
            BinOp::Gt => todo!(),
            BinOp::Eq => todo!(),
            BinOp::Ne => todo!(),
            BinOp::Lt => {
                let dst = self.expr_to_reg(Operation::BinOp(op, lhs_reg, rhs_reg));
                let test_dst = self.expr_to_reg(Operation::Test(op, dst));
                self.iloc_buf.extend([
                    Instruction::FComp { src_a: lhs_reg, src_b: rhs_reg, dst },
                    Instruction::TestGE { test: dst, dst: test_dst },
                ]);
                test_dst
            }

            BinOp::Rem
            | BinOp::LeftShift
            | BinOp::RightShift
            | BinOp::BitAnd
            | BinOp::BitXor
            | BinOp::BitOr
            | BinOp::And
            | BinOp::Or => {
                unreachable!("no float `{:?}`", op)
            }
            BinOp::AddAssign | BinOp::SubAssign => {
                unreachable!("this is converted to a full add/sub expression")
            }
        }
    }

    fn gen_index(&mut self, arr_reg: Reg, exprs: &[Expr]) -> Reg {
        let stack_pad = self.value_to_reg(Val::Int(-4));
        let size_of = self.value_to_reg(Val::Int(4));
        let arr_start = self.expr_to_reg(Operation::FramePointer);

        if let [expr] = exprs {
            let idx = self.gen_expression(expr);
            let calc_idx = self.expr_to_reg(Operation::BinOp(BinOp::Mul, idx, size_of));
            let calc_arr_start =
                self.expr_to_reg(Operation::BinOp(BinOp::Add, arr_start, stack_pad));
            let arr_slot = self.expr_to_reg(Operation::BinOp(BinOp::Sub, calc_arr_start, calc_idx));

            self.iloc_buf.extend([
                // Move array start address to register
                Instruction::I2I { src: arr_reg, dst: arr_start },
                // Load the number of bytes on the stack for no reason
                Instruction::ImmLoad { src: inst::Val::Integer(-4), dst: stack_pad },
                // Add -4 to make up for stack padding
                Instruction::Add { src_a: arr_start, src_b: stack_pad, dst: calc_arr_start },
                // Size of type
                Instruction::ImmLoad { src: inst::Val::Integer(4), dst: size_of },
                // index * size of type
                Instruction::Mult { src_a: idx, src_b: size_of, dst: calc_idx },
                // array start - (index * size_of)
                Instruction::Sub { src_a: calc_arr_start, src_b: calc_idx, dst: arr_slot },
            ]);
            arr_slot
        } else {
            todo!("No multi dim arrays yet...")
        }
    }

    fn gen_expression(&mut self, expr: &Expr) -> Reg {
        match expr {
            Expr::Ident { ident, ty } => {
                if let Some(mem_addr) = self.load_from_addr.get(ident).copied() {
                    let dst = self.expr_to_reg(Operation::Load(mem_addr));
                    self.iloc_buf.extend([Instruction::Load { src: mem_addr, dst }]);
                    return dst;
                }
                match ty {
                    Ty::Array { .. } => self.ident_to_reg(*ident),
                    Ty::Struct { .. } => self.ident_to_reg(*ident),
                    Ty::Ptr(_) => self.ident_to_reg(*ident),
                    Ty::Ref(_) => todo!(),
                    Ty::ConstStr(_) => {
                        if let Some(Global::Text { name, .. }) = self.globals.get(ident).cloned() {
                            let tmp = self.expr_to_reg(Operation::ImmLoad(*ident));
                            self.iloc_buf.push(Instruction::ImmLoad {
                                src: inst::Val::Location(name.to_string()),
                                dst: tmp,
                            });
                            tmp
                        } else {
                            self.ident_to_reg(*ident)
                        }
                    }
                    Ty::Int => self.ident_to_reg(*ident),
                    Ty::Char => self.ident_to_reg(*ident),
                    Ty::Float => self.ident_to_reg(*ident),
                    Ty::Bool => self.ident_to_reg(*ident),
                    _ => todo!(),
                }
            }
            Expr::Deref { indir, expr, ty } => {
                let val = self.gen_expression(expr);
                let dst = self.expr_to_reg(Operation::Load(val));
                self.iloc_buf.extend([Instruction::Load { src: val, dst }]);
                dst
            }
            Expr::AddrOf(expr) => {
                let start = *self
                    .ident_address
                    .get(&expr.as_ident())
                    .unwrap_or_else(|| panic!("{:?}", expr));

                let src = self.gen_expression(&*expr);

                if let Expr::Array { .. } = &**expr {
                    return src;
                }

                let curr_stack = self.value_to_reg(Val::Int(start));
                let fp = self.expr_to_reg(Operation::FramePointer);
                let addr = self.expr_to_reg(Operation::BinOp(BinOp::Sub, fp, curr_stack));

                self.iloc_buf.extend([
                    Instruction::I2I { src: Reg::Var(0), dst: fp },
                    Instruction::ImmLoad { src: inst::Val::Integer(start), dst: curr_stack },
                    Instruction::Sub { src_a: fp, src_b: curr_stack, dst: addr },
                    Instruction::Store { src, dst: addr },
                ]);

                self.load_from_addr.insert(expr.as_ident(), addr);

                addr
            }
            Expr::Array { ident, exprs, ty } => {
                let reg = if let Some(Global::Array { name, .. }) = self.globals.get(ident).cloned()
                {
                    let tmp = self.expr_to_reg(Operation::ImmLoad(*ident));
                    self.iloc_buf.push(Instruction::ImmLoad {
                        src: inst::Val::Location(name.to_string()),
                        dst: tmp,
                    });
                    tmp
                } else {
                    self.ident_to_reg(*ident)
                };
                self.gen_index(reg, exprs)
            }
            Expr::Urnary { op, expr, ty } => {
                let ex = self.gen_expression(expr);
                match op {
                    UnOp::Not => {
                        let dst = self.expr_to_reg(Operation::UnOp(*op, ex));
                        self.iloc_buf.push(Instruction::Not { src: ex, dst });
                        dst
                    }
                    UnOp::OnesComp => {
                        todo!("what is this...");
                        let dst = self.expr_to_reg(Operation::UnOp(*op, ex));
                        self.iloc_buf.push(Instruction::Not { src: ex, dst });
                        dst
                    }
                }
            }
            Expr::Binary { op, lhs, rhs, ty } => {
                let mut lhs_reg = self.gen_expression(lhs);
                if matches!(lhs.type_of(), Ty::Array { .. }) {
                    let dst = self.expr_to_reg(Operation::Load(lhs_reg));
                    self.iloc_buf.push(Instruction::Load { src: lhs_reg, dst });
                    lhs_reg = dst;
                }
                let mut rhs_reg = self.gen_expression(rhs);
                if matches!(rhs.type_of(), Ty::Array { .. }) {
                    let dst = self.expr_to_reg(Operation::Load(rhs_reg));
                    self.iloc_buf.push(Instruction::Load { src: rhs_reg, dst });
                    rhs_reg = dst;
                }
                if let Ty::Float = ty {
                    self.float_binop(*op, lhs_reg, rhs_reg)
                } else {
                    self.int_binop(*op, lhs_reg, rhs_reg)
                }
            }
            Expr::Parens(expr) => self.gen_expression(expr),
            Expr::Call { path, args, type_args, def } => {
                let name = path.to_string();
                let arg_regs: Vec<_> = args.iter().map(|a| self.gen_expression(a)).collect();

                if name == "flt_int" && !matches!(def.kind, FuncKind::Normal | FuncKind::Pointer) {
                    assert!(args.len() == 1);
                    assert!(args[0].type_of() == Ty::Float);
                    let arg = arg_regs[0];
                    let dst = self.expr_to_reg(Operation::CvtInt(arg));
                    self.iloc_buf.push(Instruction::F2I { src: arg, dst });
                    dst
                } else if name == "int_flt"
                    && !matches!(def.kind, FuncKind::Normal | FuncKind::Pointer)
                {
                    assert!(args.len() == 1);
                    assert!(args[0].type_of() == Ty::Int);
                    let arg = arg_regs[0];
                    let dst = self.expr_to_reg(Operation::CvtFloat(arg));
                    self.iloc_buf.push(Instruction::I2F { src: arg, dst });
                    dst
                } else if name == "malloc"
                    && !matches!(def.kind, FuncKind::Normal | FuncKind::Pointer)
                {
                    assert!(args.len() == 1);
                    assert!(args[0].type_of() == Ty::Int);
                    let arg = arg_regs[0];
                    let dst = self.expr_to_reg(Operation::Malloc(arg));
                    self.iloc_buf.push(Instruction::Malloc { size: arg, dst });
                    dst
                } else if name == "realloc"
                    && !matches!(def.kind, FuncKind::Normal | FuncKind::Pointer)
                {
                    assert!(args.len() == 2);
                    assert!(args[1].type_of() == Ty::Int);
                    let old = arg_regs[0];
                    let size = arg_regs[1];
                    let dst = self.expr_to_reg(Operation::Realloc(old, size));
                    self.iloc_buf.push(Instruction::Realloc { src: old, size, dst });
                    dst
                } else if matches!(def.ret, Ty::Void) {
                    self.iloc_buf.push(Instruction::Call { name, args: arg_regs });
                    // A fake register, this should NEVER be used by the caller since this
                    // returns nothing
                    Reg::Var(0)
                } else {
                    let ret = self.expr_to_reg(Operation::ImmCall(path.local_ident()));
                    self.iloc_buf.push(Instruction::ImmCall { name, args: arg_regs, ret });
                    ret
                }
            }
            Expr::TraitMeth { trait_, args, type_args, def } => todo!(),
            Expr::FieldAccess { lhs, def, rhs } => {
                let start = self.gen_expression(lhs);
                let access = construct_field_offset(self, rhs, 0, start, def);
                match rhs.type_of() {
                    Ty::Struct { .. }
                    | Ty::Enum { .. }
                    | Ty::Func { .. }
                    | Ty::Ref(_) => todo!("{:?}", rhs),
                    Ty::Ptr(inner) => {
                        // Just return the pointer to the thing
                        access
                    }
                    Ty::Array { size, ty } => {
                        let load_reg = self.expr_to_reg(Operation::Load(access));
                        self.iloc_buf.push(Instruction::Load { src: access, dst: load_reg });
                        load_reg
                    }
                    Ty::ConstStr(_) | Ty::Int | Ty::Char | Ty::Float | Ty::Bool | Ty::Void => {
                        let load_reg = self.expr_to_reg(Operation::Load(access));
                        self.iloc_buf.push(Instruction::Load { src: access, dst: load_reg });
                        load_reg
                    }
                    Ty::Bottom | Ty::Generic { .. } => todo!("{:?}", rhs),
                }
            }
            Expr::StructInit { path, fields, def } => {
                fn flatten_struct_init(f: &FieldInit) -> Vec<&Expr> {
                    match &f.init {
                        Expr::StructInit { path, fields, def } => {
                            fields.iter().flat_map(flatten_struct_init).collect::<Vec<_>>()
                        }
                        Expr::EnumInit { path, variant, items, def } => {
                            items.iter().collect::<Vec<_>>()
                        }
                        Expr::ArrayInit { items, ty } => items.iter().collect::<Vec<_>>(),
                        _ => vec![&f.init],
                    }
                }

                let start = self.stack_size;
                let stack_pad = self.value_to_reg(Val::Int(-4));
                let mut struct_reg = self.expr_to_reg(Operation::StructInit(start as u64));
                let mut struct_start = self.expr_to_reg(Operation::FramePointer);

                let mut running_offset = 0;
                for expr in fields.iter().flat_map(flatten_struct_init) {
                    let reg = self.gen_expression(expr);

                    let offset = self.value_to_reg(Val::Int(running_offset));
                    let calc_struct_start =
                        self.expr_to_reg(Operation::BinOp(BinOp::Add, struct_start, stack_pad));
                    let arr_slot =
                        self.expr_to_reg(Operation::BinOp(BinOp::Sub, calc_struct_start, offset));

                    self.iloc_buf.push(
                        // Load struct offset
                        Instruction::ImmLoad {
                            src: inst::Val::Integer(running_offset),
                            dst: offset,
                        },
                    );
                    // Move array start address to register
                    if start > 0 {
                        let dst = self.value_to_reg(Val::Int(start));
                        let tmp = struct_start;
                        struct_start = self.expr_to_reg(Operation::BinOp(BinOp::Sub, tmp, dst));
                        self.iloc_buf.extend([
                            Instruction::I2I { src: Reg::Var(0), dst: tmp },
                            Instruction::ImmLoad { src: inst::Val::Integer(start), dst },
                            Instruction::Sub { src_a: tmp, src_b: dst, dst: struct_start },
                        ])
                    } else {
                        self.iloc_buf
                            .extend([Instruction::I2I { src: Reg::Var(0), dst: struct_start }])
                    }
                    self.iloc_buf.extend([
                        // Load the number of bytes on the stack for no reason
                        Instruction::ImmLoad { src: inst::Val::Integer(-4), dst: stack_pad },
                        // Add -4 to make up for stack padding
                        Instruction::Add {
                            src_a: struct_start,
                            src_b: stack_pad,
                            dst: calc_struct_start,
                        },
                        // struct start - (running offset)
                        Instruction::Sub { src_a: calc_struct_start, src_b: offset, dst: arr_slot },
                        //
                        Instruction::Store { src: reg, dst: arr_slot },
                    ]);

                    // TODO: the size of every type is 4...
                    running_offset += 4;
                }

                self.iloc_buf.push(Instruction::I2I { src: struct_start, dst: struct_reg });
                struct_reg
            }
            Expr::EnumInit { path, variant, items, def } => todo!(),
            Expr::ArrayInit { items, ty } => {
                let start = self.stack_size;

                let arr_reg = self.expr_to_reg(Operation::ArrayInit(start as u64));
                let stack_pad = self.value_to_reg(Val::Int(-4));
                let size_of = self.value_to_reg(Val::Int(4));
                let mut arr_start = self.expr_to_reg(Operation::FramePointer);

                for (idx, item) in items.iter().enumerate() {
                    // load const => vrtmp
                    let reg = self.gen_expression(item);

                    let idx_dst = self.value_to_reg(Val::Int(idx as isize));
                    let calc_idx = self.expr_to_reg(Operation::BinOp(BinOp::Mul, idx_dst, size_of));
                    let calc_arr_start =
                        self.expr_to_reg(Operation::BinOp(BinOp::Add, arr_start, stack_pad));
                    let arr_slot =
                        self.expr_to_reg(Operation::BinOp(BinOp::Sub, calc_arr_start, calc_idx));

                    self.iloc_buf.push(
                        // Load array index
                        Instruction::ImmLoad {
                            src: inst::Val::Integer(idx as isize),
                            dst: idx_dst,
                        },
                    );
                    // Move array start address to register
                    if start > 0 {
                        let dst = self.value_to_reg(Val::Int(start));
                        let tmp = arr_start;
                        arr_start = self.expr_to_reg(Operation::BinOp(BinOp::Sub, tmp, dst));
                        self.iloc_buf.extend([
                            Instruction::I2I { src: Reg::Var(0), dst: tmp },
                            Instruction::ImmLoad { src: inst::Val::Integer(start), dst },
                            Instruction::Sub { src_a: tmp, src_b: dst, dst: arr_start },
                        ])
                    } else {
                        self.iloc_buf
                            .extend([Instruction::I2I { src: Reg::Var(0), dst: arr_start }])
                    }
                    self.iloc_buf.extend([
                        // Load the number of bytes on the stack for no reason
                        Instruction::ImmLoad { src: inst::Val::Integer(-4), dst: stack_pad },
                        // Add -4 to make up for stack padding
                        Instruction::Add {
                            src_a: arr_start,
                            src_b: stack_pad,
                            dst: calc_arr_start,
                        },
                        // Size of type
                        Instruction::ImmLoad { src: inst::Val::Integer(4), dst: size_of },
                        // index * size of type
                        Instruction::Mult { src_a: idx_dst, src_b: size_of, dst: calc_idx },
                        // array start - (index * size_of)
                        Instruction::Sub { src_a: calc_arr_start, src_b: calc_idx, dst: arr_slot },
                        //
                        Instruction::Store { src: reg, dst: arr_slot },
                    ]);
                }
                // Save the register that will be tied to the name that identifies this array init
                self.iloc_buf.push(Instruction::I2I { src: arr_start, dst: arr_reg });
                arr_reg
            }
            Expr::Value(val) => match val {
                Val::Float(f) => {
                    let (name, tmp1, tmp2) = self.global_to_reg_float(
                        val.clone(),
                        Ident::new(DUMMY, &format!(".float_const_{}", self.globals.len())),
                    );
                    // Only insert if we haven't seen this before
                    self.globals
                        .entry(name)
                        .or_insert(Global::Float { name: name.to_string(), content: *f });
                    // Load the constant float value
                    self.iloc_buf.extend([
                        Instruction::ImmLoad {
                            src: inst::Val::Location(name.to_string()),
                            dst: tmp1,
                        },
                        Instruction::FLoad { src: tmp1, dst: tmp2 },
                    ]);
                    tmp2
                }
                Val::Int(i) => {
                    let tmp = self.value_to_reg(val.clone());
                    self.iloc_buf
                        .push(Instruction::ImmLoad { src: inst::Val::Integer(*i), dst: tmp });
                    tmp
                }
                Val::Char(ch) => {
                    let tmp = self.value_to_reg(val.clone());
                    self.iloc_buf.push(Instruction::ImmLoad {
                        src: inst::Val::Integer(*ch as isize),
                        dst: tmp,
                    });
                    tmp
                }
                Val::Bool(b) => {
                    let tmp = self.value_to_reg(val.clone());
                    self.iloc_buf.push(Instruction::ImmLoad {
                        src: inst::Val::Integer(if *b { 1 } else { 0 }),
                        dst: tmp,
                    });
                    tmp
                }
                Val::Str(s) => {
                    let (name, tmp) = self.global_to_reg(
                        val.clone(),
                        Ident::new(DUMMY, &format!(".str_const_{}", self.globals.len())),
                    );
                    // Only insert if we haven't seen this before
                    self.globals
                        .entry(name)
                        .or_insert(Global::Text { name: name.to_string(), content: s.to_string() });
                    // Load the constant float value
                    self.iloc_buf.push(Instruction::ImmLoad {
                        src: inst::Val::Location(name.to_string()),
                        dst: tmp,
                    });
                    tmp
                }
            },
            Expr::Builtin(_) => todo!(),
        }
    }

    fn get_pointer(&mut self, lval: &LValue) -> Reg {
        match lval {
            LValue::Ident { ident, ty } => self.ident_to_reg(*ident),
            LValue::Deref { indir, expr, ty } => {
                let mut loc = self.get_pointer(expr);
                let mut l_dst = self.expr_to_reg(Operation::Load(loc));

                for _ in 0..*indir {
                    self.iloc_buf.push(Instruction::Load { src: loc, dst: l_dst });

                    // We followed the address, if the memory is another address we load from there
                    loc = l_dst;
                    l_dst = self.expr_to_reg(Operation::Load(loc));
                }
                l_dst
            }
            LValue::Array { ident, exprs, ty } => self.gen_expression(&Expr::Array {
                ident: *ident,
                exprs: exprs.clone(),
                ty: ty.clone(),
            }),
            LValue::FieldAccess { lhs, def, rhs, field_idx } => {
                let lhs_reg = self.ident_to_reg(lhs.as_ident().unwrap());
                construct_field_offset_lvalue(self, &*rhs, lhs_reg, def)
            }
        }
    }

    fn gen_statement(&mut self, stmt: &'ctx Stmt) {
        match stmt {
            Stmt::Const(_) => todo!(),
            // For let x = 5;
            //  - loadI 5 => %vr5; i2i %vr5 => %vrx; (where x is the final register for ident)
            // For x = 10;
            //  - loadI 10 => %vr10; i2i %vr10 => %vrx;
            // For *x = 2;
            //  - loadI 2 => %vr2; store %vr2 => %vrx; (because at some point `let x = &y;`)
            Stmt::Assign { lval, rval, is_let } => {
                match rval.type_of() {
                    Ty::Array { size, ty } if *is_let && matches!(&*ty, Ty::Int) => {
                        self.ident_address.insert(lval.as_ident().unwrap(), self.stack_size);
                        self.stack_size += (4 * size as isize);
                    }
                    Ty::Struct { def, .. } if *is_let => {
                        self.ident_address.insert(lval.as_ident().unwrap(), self.stack_size);
                        // TODO: since each type is the same size...
                        self.stack_size += (4 * def.fields.len() as isize);
                    }
                    Ty::Int | Ty::Float if *is_let => {
                        self.ident_address.insert(lval.as_ident().unwrap(), self.stack_size);
                        self.stack_size += 4;
                    }
                    _ => (),
                }
                let mut val = self.gen_expression(rval);

                // If we are looking at an element of an array or dereferencing memory we get the
                // value
                if matches!(rval, Expr::Array { .. } | Expr::Deref { .. }) {
                    let dst = self.expr_to_reg(Operation::Load(val));
                    self.iloc_buf.push(Instruction::Load { src: val, dst });
                    val = dst;
                }
                let dst = self.get_pointer(lval);
                match lval {
                    LValue::Ident { ident, .. } => match lval.type_of() {
                        Ty::Int | Ty::Array { .. } | Ty::Ptr(..) => {
                            self.iloc_buf.push(Instruction::I2I { src: val, dst })
                        }
                        Ty::Float => {
                            self.iloc_buf.push(Instruction::F2F { src: val, dst });
                        }
                        Ty::ConstStr(..) => {
                            if let Some(Global::Text { name, content }) = self.globals.get(ident) {
                                self.iloc_buf.push(Instruction::I2I { src: val, dst });
                            } else {
                                self.iloc_buf.push(Instruction::I2I { src: val, dst });
                            }
                        }
                        Ty::Struct { .. } => {
                            self.iloc_buf.push(Instruction::I2I { src: val, dst });
                        }
                        t => todo!("{:?}", lval),
                    },
                    LValue::Deref { indir, expr, ty } => {
                        self.iloc_buf.push(Instruction::Store { src: val, dst });
                    }
                    LValue::Array { exprs, .. } => {
                        if let [expr] = exprs.as_slice() {
                            self.iloc_buf.push(
                                // Store the value calculated before the LValue match `val`
                                Instruction::Store { src: val, dst },
                            );
                        } else {
                            unreachable!("No multi dim arrays yet...")
                        }
                    }
                    LValue::FieldAccess { .. } => {
                        self.iloc_buf.push(Instruction::Store { src: val, dst });
                    }
                    _ => (),
                }
            }
            Stmt::Call { expr, def } => match expr.path.to_string().as_str() {
                "write" if !matches!(def.kind, FuncKind::Normal | FuncKind::Pointer) => {
                    assert!(expr.args.len() == 1);
                    let arg = self.gen_expression(&expr.args[0]);
                    match expr.args[0].type_of() {
                        Ty::Int | Ty::Ptr(..) | Ty::Ref(..) => {
                            self.iloc_buf.push(Instruction::IWrite(arg))
                        }
                        Ty::Float => self.iloc_buf.push(Instruction::FWrite(arg)),
                        Ty::ConstStr(..) => self.iloc_buf.push(Instruction::SWrite(arg)),
                        Ty::Array { ty, .. } if matches!(&*ty, Ty::Int) => {
                            let dst = self.expr_to_reg(Operation::Load(arg));
                            self.iloc_buf.extend([
                                Instruction::Load { src: arg, dst },
                                Instruction::IWrite(dst),
                            ]);
                        }
                        t => unreachable!("not writeable {:?}", t),
                    }
                }
                "putchar" if !matches!(def.kind, FuncKind::Normal | FuncKind::Pointer) => {
                    assert!(expr.args.len() == 1);
                    let arg = self.gen_expression(&expr.args[0]);
                    match expr.args[0].type_of() {
                        Ty::Int => self.iloc_buf.push(Instruction::PutChar(arg)),
                        t => unreachable!("not writeable {:?} {:?}", t, expr.args[0]),
                    }
                }
                "scan" if !matches!(def.kind, FuncKind::Normal | FuncKind::Pointer) => {
                    assert!(expr.args.len() == 1);
                    let arg = self.gen_expression(&expr.args[0]);

                    match expr.args[0].type_of() {
                        Ty::Ptr(ty) if matches!(&*ty, Ty::Int) => {
                            self.iloc_buf.push(Instruction::IRead(arg))
                        }
                        Ty::Ptr(ty) if matches!(&*ty, Ty::Float) => {
                            self.iloc_buf.push(Instruction::FRead(arg))
                        }
                        Ty::Array { ty, .. } if matches!(&*ty, Ty::Int) => {
                            self.iloc_buf.push(Instruction::IRead(arg));
                        }
                        t => unreachable!("not readable {:?}", t),
                    }
                }
                "free" if !matches!(def.kind, FuncKind::Normal | FuncKind::Pointer) => {
                    assert!(expr.args.len() == 1);
                    let arg = self.gen_expression(&expr.args[0]);
                    self.iloc_buf.push(Instruction::Free(arg));
                }
                // `malloc` and `realloc` will be handled here, it would be odd to use malloc
                // as a stmt `malloc(size);` doesn't do anything...
                _ => {
                    self.gen_expression(&Expr::Call {
                        path: expr.path.clone(),
                        args: expr.args.clone(),
                        type_args: expr.type_args.clone(),
                        def: def.clone(),
                    });
                }
            },
            Stmt::TraitMeth { expr, def } => todo!(),
            Stmt::If { cond, blk, els } => {
                let cond = self.gen_expression(cond);
                let else_label = format!(".L{}:", self.next_label());
                let after_blk = format!(".L{}:", self.next_label());

                if !els.is_empty() {
                    // If the `if` is not true jump to else
                    self.iloc_buf
                        .push(Instruction::CbrT { cond, loc: Loc(else_label.replace(':', "")) });
                    // Otherwise fall through to the true case
                    for stmt in &blk.stmts {
                        self.gen_statement(stmt)
                    }
                    // Don't fallthrough to the else case, jump to the merge after all else/elseif
                    self.iloc_buf.push(Instruction::ImmJump(Loc(after_blk.replace(":", ""))));

                    // Jump to the else block which doubles as the start of our elseif's
                    self.iloc_buf.push(Instruction::Label(else_label));
                    for Else { block: Block { stmts }, cond } in els {
                        let elseif_label = format!(".L{}:", self.next_label());

                        if let Some(c) = cond {
                            let cond = self.gen_expression(c);
                            self.iloc_buf.push(Instruction::CbrT {
                                cond,
                                loc: Loc(elseif_label.replace(':', "")),
                            });
                        }
                        for stmt in stmts {
                            self.gen_statement(stmt)
                        }
                        self.iloc_buf.push(Instruction::ImmJump(Loc(after_blk.replace(":", ""))));
                        self.iloc_buf.push(Instruction::Label(elseif_label));
                    }

                    self.iloc_buf.push(Instruction::Label(after_blk));
                } else {
                    self.iloc_buf
                        .push(Instruction::CbrT { cond, loc: Loc(after_blk.replace(':', "")) });

                    for stmt in &blk.stmts {
                        self.gen_statement(stmt)
                    }
                    self.iloc_buf.push(Instruction::Label(after_blk));
                }
            }
            Stmt::While { cond: cond_expr, stmts } => {
                let cond = self.gen_expression(cond_expr);

                let while_case = format!(".L{}:", self.next_label());
                let after_blk = format!(".L{}:", self.next_label());

                self.iloc_buf.extend([
                    Instruction::CbrT { cond, loc: Loc(after_blk.replace(':', "")) },
                    Instruction::Label(while_case.clone()),
                ]);

                for stmt in &stmts.stmts {
                    self.gen_statement(stmt)
                }

                // Regenerate the condition but, since we generate the opposite logic
                // we continue the loop only for false cases (since we jump the loop for true cases)
                let cond = self.gen_expression(cond_expr);
                self.iloc_buf.extend([
                    Instruction::CbrF { cond, loc: Loc(while_case.replace(':', "")) },
                    Instruction::Label(after_blk),
                ]);
            }
            Stmt::Match { expr, arms, ty } => todo!(),
            Stmt::Ret(ex, _) => {
                let ret_reg = self.gen_expression(ex);
                self.iloc_buf.push(Instruction::ImmRet(ret_reg));
            }
            Stmt::Exit => {
                self.iloc_buf.push(Instruction::Ret);
            }
            Stmt::Block(_) => todo!(),
            Stmt::InlineAsm(_) => todo!(),
            Stmt::Builtin(_) => todo!(),
        }
    }
}

impl<'ast> Visit<'ast> for IlocGen<'ast> {
    fn visit_const(&mut self, var: &'ast Const) {
        let name = format!(".Lglobal_{}", var.ident);
        match var.ty {
            Ty::Generic { .. } | Ty::Struct { .. } | Ty::Enum { .. } | Ty::Ptr(_) | Ty::Ref(_) => {
                todo!()
            }
            Ty::Array { .. } => {
                self.globals.insert(
                    var.ident,
                    Global::Array {
                        name: name.clone(),
                        content: if let Expr::ArrayInit { items, .. } = &var.init {
                            items
                                .iter()
                                .map(|ex| {
                                    if let Expr::Value(val) = ex {
                                        Ok(match val {
                                            Val::Bool(b) => {
                                                inst::Val::Integer(if *b { 1 } else { 0 })
                                            }
                                            Val::Char(c) => inst::Val::Integer(*c as isize),
                                            Val::Int(int) => inst::Val::Integer(*int),
                                            Val::Float(f) => inst::Val::Float(*f),
                                            _ => return Err("type not supported in const arrays"),
                                        })
                                    } else {
                                        Err("found non const value in const array")
                                    }
                                })
                                .collect::<Result<Vec<_>, _>>()
                                .unwrap()
                        } else {
                            unreachable!("non const string value used in constant")
                        },
                    },
                );
            }
            Ty::ConstStr(..) => {
                self.globals.insert(
                    var.ident,
                    Global::Text {
                        name: name.clone(),
                        content: if let Expr::Value(Val::Str(s)) = var.init {
                            s.name().to_owned()
                        } else {
                            unreachable!("non const string value used in constant")
                        },
                    },
                );
            }
            Ty::Char => {
                self.globals.insert(
                    var.ident,
                    Global::Char {
                        name: name.clone(),
                        content: if let Expr::Value(Val::Char(s)) = var.init {
                            s as u8
                        } else {
                            unreachable!("non char value used in constant")
                        },
                    },
                );
            }
            Ty::Float => {
                self.globals.insert(
                    var.ident,
                    Global::Int {
                        name: name.clone(),
                        content: if let Expr::Value(Val::Float(f)) = var.init {
                            f.to_bits() as i64
                        } else {
                            unreachable!("non char value used in constant")
                        },
                    },
                );
            }
            Ty::Int => {
                self.globals.insert(
                    var.ident,
                    Global::Int {
                        name: name.clone(),
                        content: if let Expr::Value(Val::Int(i)) = var.init {
                            i as i64
                        } else {
                            unreachable!("non char value used in constant")
                        },
                    },
                );
            }
            Ty::Bool => {
                self.globals.insert(
                    var.ident,
                    Global::Int {
                        name: name.clone(),
                        content: if let Expr::Value(Val::Bool(b)) = var.init {
                            if b {
                                1
                            } else {
                                0
                            }
                        } else {
                            unreachable!("non char value used in constant")
                        },
                    },
                );
            }
            Ty::Func { .. } | Ty::Void | Ty::Bottom => unreachable!(),
        };
        self.vars.insert(var.ident, Loc(name));
    }

    fn visit_func(&mut self, func: &'ast Func) {
        let function_name = func.ident.name().to_string();
        self.vars.insert(func.ident, Loc(function_name.clone()));
        if func.stmts.is_empty()
            || matches!(func.kind, FuncKind::Linked | FuncKind::EmptyTrait | FuncKind::Extern)
        {
            return;
        }

        self.current_fn_params = func.params.iter().map(|p| p.ident).collect();
        let params = func.params.iter().map(|p| self.ident_to_reg(p.ident)).collect();
        let frame_inst = Instruction::Frame { name: function_name, params, size: 58008 };

        let frame_idx = self.iloc_buf.len();
        self.iloc_buf.push(frame_inst);

        self.stack_size = 0;
        for stmt in &func.stmts {
            self.gen_statement(stmt);
        }

        if let Instruction::Frame { size, .. } = &mut self.iloc_buf[frame_idx] {
            *size = self.stack_size as usize;
        }
    }
}

fn construct_field_offset<'a>(
    gen: &mut IlocGen<'a>,
    rhs: &Expr,
    offset: isize,
    start: Reg,
    def: &Struct,
) -> Reg {
    match rhs {
        Expr::Ident { ident, ty } => {
            // FIXME: this is crap
            // Start with the -4 of the stack offset so we don't need to do the instruction
            let mut count = 4;
            for f in &def.fields {
                // println!("{} = {} - {}", ident, offset, count);
                if f.ident == *ident {
                    let val = gen.value_to_reg(Val::Int(offset + count));
                    let reg = gen.expr_to_reg(Operation::BinOp(BinOp::Sub, start, val));
                    gen.iloc_buf.extend([
                        Instruction::ImmLoad { src: inst::Val::Integer(offset + count), dst: val },
                        Instruction::Sub { src_a: start, src_b: val, dst: reg },
                    ]);
                    return reg;
                }

                // Do this after so first field is the 0th offset
                count += 4;
            }

            // type checking missed this field somehow
            unreachable!("type checking missed an invalid field access")
        }
        Expr::Deref { indir, expr, ty } => {
            todo!("follow the pointer")
        }
        Expr::Array { ident, exprs, ty } => {
            // FIXME: this is crap
            // Start with the -4 of the stack offset so we don't need to do the instruction
            let mut count = 4;
            for f in &def.fields {
                if f.ident == *ident {
                    let tmp = gen.value_to_reg(Val::Int(count));
                    let loc = gen.expr_to_reg(Operation::BinOp(BinOp::Sub, start, tmp));
                    let arr_loc = gen.expr_to_reg(Operation::Load(loc));
                    gen.iloc_buf.extend([
                        Instruction::ImmLoad { src: inst::Val::Integer(count), dst: tmp },
                        Instruction::Sub { src_a: start, src_b: tmp, dst: loc },
                    ]);
                    return loc;
                }
                count += 4;
            }

            // type checking missed this field somehow
            unreachable!("type checking missed an invalid field access")
        }
        Expr::FieldAccess { lhs, rhs: inner, def: inner_def } => {
            // FIXME: this is crap
            // Start with the -4 of the stack offset so we don't need to do the instruction
            let mut count = 4;
            for f in &def.fields {
                if f.ident == lhs.as_ident() {
                    return construct_field_offset(gen, inner, offset + count, start, inner_def);
                }
                count += 4;
            }

            // type checking missed this field somehow
            unreachable!("type checking missed an invalid field access")
        }
        Expr::AddrOf(_) => todo!(),
        Expr::Call { path, args, type_args, def } => todo!(),
        _ => unreachable!("not a valid struct field accessor"),
    }
}

// TODO: @copypaste this whole thing could be removed if `LValue -> Expr` worked but the
// lifetimes can't match when creating an `Expr` from a `LValue`
fn construct_field_offset_lvalue<'a>(
    gen: &mut IlocGen<'a>,
    rhs: &LValue,
    lhs_reg: Reg,
    def: &Struct,
) -> Reg {
    match rhs {
        LValue::Ident { ident, ty } => {
            // FIXME: this is crap
            // Start with the -4 of the stack offset so we don't need to do the instruction
            let mut count = 4;
            for f in &def.fields {
                if f.ident == *ident {
                    let tmp = gen.value_to_reg(Val::Int(count));
                    let loc = gen.expr_to_reg(Operation::BinOp(BinOp::Sub, lhs_reg, tmp));
                    gen.iloc_buf.extend([
                        Instruction::ImmLoad { src: inst::Val::Integer(count), dst: tmp },
                        Instruction::Sub { src_a: lhs_reg, src_b: tmp, dst: loc },
                    ]);
                    return loc;
                }
                // TODO: make this not just 4...
                // Do this after so first field is the 0th offset
                count += 4;
            }
            unreachable!("type checking failed for this field access {:?}", rhs)
        }
        LValue::Deref { indir, expr, ty } => {
            todo!("follow the pointer")
        }
        LValue::Array { ident, exprs, ty } => {
            // FIXME: this is crap
            // Start with the -4 of the stack offset so we don't need to do the instruction
            let mut count = 4;
            for f in &def.fields {
                if f.ident == *ident {
                    let tmp = gen.value_to_reg(Val::Int(count));
                    let loc = gen.expr_to_reg(Operation::BinOp(BinOp::Sub, lhs_reg, tmp));
                    let arr_loc = gen.expr_to_reg(Operation::Load(loc));
                    gen.iloc_buf.extend([
                        Instruction::ImmLoad { src: inst::Val::Integer(count), dst: tmp },
                        Instruction::Sub { src_a: lhs_reg, src_b: tmp, dst: loc },
                        Instruction::Load { src: loc, dst: arr_loc },
                    ]);
                    // let ele_size =
                    // if let Ty::Array { ty, .. } = ty { ty.size() } else { ty.size() };
                    return gen.gen_index(arr_loc, exprs);
                }
                count += 4;
            }
            unreachable!("type checking failed for this field access {:?}", rhs)
        }
        LValue::FieldAccess { lhs, rhs: inner, def: inner_def, field_idx } => {
            // FIXME: this is crap
            // Start with the -4 of the stack offset so we don't need to do the instruction
            let mut count = 4;
            for f in &def.fields {
                if Some(f.ident) == lhs.as_ident() {
                    return construct_field_offset_lvalue(
                        gen, inner, // offset - count,
                        lhs_reg, inner_def,
                    );
                }
                count += 4;
            }
            unreachable!("type checking failed for this field access {:?}", rhs)
        }
        _ => unreachable!("not a valid struct field accessor"),
    }
}
