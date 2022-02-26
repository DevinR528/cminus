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
            BinOp, Binding, Builtin, CallExpr, Const, Expr, FieldInit, Func, LValue, MatchArm, Pat,
            Stmt, Struct, Ty, UnOp, Val,
        },
        visit::Visit,
    },
    typeck::scope::hash_any,
};

use self::inst::Operation;

mod inst;

#[derive(Debug)]
crate struct IlocGen<'ctx> {
    iloc_buf: Vec<Instruction>,

    stack_size: usize,

    vars: HashMap<Ident, Loc>,
    globals: HashMap<Ident, Global>,

    registers: HashMap<Ident, Reg>,
    expr_regs: HashMap<Operation, Reg>,
    values: HashMap<Val, Reg>,
    global_regs: HashMap<Val, (Ident, Reg, Reg)>,

    curr_register: usize,
    curr_stack: isize,
    curr_label: usize,

    current_fn_params: HashSet<Ident>,
    path: &'ctx Path,
}

impl<'ctx> IlocGen<'ctx> {
    crate fn new(path: &'ctx Path) -> IlocGen<'ctx> {
        Self {
            iloc_buf: vec![],
            stack_size: 0,
            globals: HashMap::default(),
            current_fn_params: HashSet::default(),
            registers: HashMap::default(),
            expr_regs: HashMap::default(),
            values: HashMap::default(),
            global_regs: HashMap::default(),
            // `0` is a protected register (it's the stack/frame pointer)
            curr_register: 1,
            curr_stack: 0,
            curr_label: 0,
            vars: HashMap::default(),
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
                write!(buf, "    .string {}, {:?}", name, content);
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

    fn gen_expression(&mut self, expr: &Expr) -> Reg {
        match expr {
            Expr::Ident { ident, ty } => match ty {
                Ty::Array { size, ty } => todo!(),
                Ty::Ptr(_) => todo!(),
                Ty::Ref(_) => todo!(),
                Ty::ConstStr(_) => todo!(),
                Ty::Int => self.ident_to_reg(*ident),
                Ty::Char => self.ident_to_reg(*ident),
                Ty::Float => self.ident_to_reg(*ident),
                Ty::Bool => self.ident_to_reg(*ident),
                _ => todo!(),
            },
            Expr::Deref { indir, expr, ty } => todo!(),
            Expr::AddrOf(_) => todo!(),
            Expr::Array { ident, exprs, ty } => {
                let arr_reg = self.ident_to_reg(*ident);
                let stack_pad = self.value_to_reg(Val::Int(-4));
                let size_of = self.value_to_reg(Val::Int(4));
                let arr_start = self.expr_to_reg(Operation::FramePointer);

                if let [expr] = exprs.as_slice() {
                    let idx = self.gen_expression(expr);
                    let calc_idx = self.expr_to_reg(Operation::BinOp(BinOp::Mul, idx, size_of));
                    let calc_arr_start =
                        self.expr_to_reg(Operation::BinOp(BinOp::Add, arr_start, stack_pad));
                    let arr_slot =
                        self.expr_to_reg(Operation::BinOp(BinOp::Sub, calc_arr_start, calc_idx));

                    self.iloc_buf.extend([
                        // Move array start address to register
                        Instruction::I2I { src: arr_reg, dst: arr_start },
                        // Load the number of bytes on the stack for no reason
                        Instruction::ImmLoad { src: inst::Val::Integer(-4), dst: stack_pad },
                        // // Add -4 to make up for stack padding
                        Instruction::Add {
                            src_a: arr_start,
                            src_b: stack_pad,
                            dst: calc_arr_start,
                        },
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
                let args = args.iter().map(|a| self.gen_expression(a)).collect();

                if matches!(def.ret, Ty::Void) {
                    self.iloc_buf.push(Instruction::Call { name, args });
                    // A fake register, this should NEVER be used by the caller since this
                    // returns nothing
                    Reg::Var(0)
                } else {
                    let ret = self.expr_to_reg(Operation::ImmCall(path.local_ident()));
                    self.iloc_buf.push(Instruction::ImmCall { name, args, ret });
                    ret
                }
            }
            Expr::TraitMeth { trait_, args, type_args, def } => todo!(),
            Expr::FieldAccess { lhs, def, rhs } => todo!(),
            Expr::StructInit { path, fields, def } => todo!(),
            Expr::EnumInit { path, variant, items, def } => todo!(),
            Expr::ArrayInit { items, ty } => {
                let start = self.curr_stack;

                self.curr_stack += (items.len() as isize * 4);

                let arr_reg = self.expr_to_reg(Operation::ArrayInit(hash_any(
                    &items.iter().map(|e| format!("{:?}", e)).collect::<Vec<_>>(),
                )));
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
                Val::Char(_) => todo!(),
                Val::Bool(_) => todo!(),
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
                    Ty::Array { size, ty } if matches!(&*ty, Ty::Int) => {
                        self.stack_size += (4 * size);
                    }
                    _ => (),
                }
                let val = self.gen_expression(rval);
                match lval {
                    LValue::Ident { ident, ty } => {
                        let dst = self.ident_to_reg(*ident);
                        match ty {
                            Ty::Int => {
                                //
                                self.iloc_buf.push(Instruction::I2I { src: val, dst })
                            }
                            Ty::Float => {
                                //
                                self.iloc_buf.push(Instruction::F2F { src: val, dst })
                            }
                            Ty::Array { .. } => {
                                //
                                self.iloc_buf.push(Instruction::I2I { src: val, dst })
                            }
                            t => todo!("{:?}", t),
                        }
                    }
                    LValue::Deref { indir, expr, ty } => todo!(),
                    LValue::Array { ident, exprs, ty } => {
                        let arr_slot = self.gen_expression(&Expr::Array {
                            ident: *ident,
                            exprs: exprs.clone(),
                            ty: ty.clone(),
                        });
                        if let [expr] = exprs.as_slice() {
                            self.iloc_buf.push(
                                // Store the value calculated before the LValue match `val`
                                Instruction::Store { src: val, dst: arr_slot },
                            );
                        } else {
                            unreachable!("No multi dim arrays yet...")
                        }
                    }
                    LValue::FieldAccess { lhs, def, rhs, field_idx } => todo!(),
                }
            }
            Stmt::Call { expr, def } => match expr.path.to_string().as_str() {
                "print" if !matches!(def.kind, FuncKind::Normal | FuncKind::Pointer) => {
                    assert!(expr.args.len() == 1);
                    let arg = self.gen_expression(&expr.args[0]);
                    match expr.args[0].type_of() {
                        Ty::Int => self.iloc_buf.push(Instruction::IWrite(arg)),
                        Ty::Float => self.iloc_buf.push(Instruction::FWrite(arg)),
                        Ty::ConstStr(..) => self.iloc_buf.push(Instruction::SWrite(arg)),
                        Ty::Array { ty, .. } if matches!(&*ty, Ty::Int) => {
                            let dst = self.expr_to_reg(Operation::Load(arg));
                            self.iloc_buf.extend([
                                Instruction::Load { src: arg, dst },
                                Instruction::IWrite(dst),
                            ]);
                        }
                        _ => unreachable!("not writeable"),
                    }
                }
                "scan" if !matches!(def.kind, FuncKind::Normal | FuncKind::Pointer) => {
                    assert!(expr.args.len() == 1);
                    let arg = self.gen_expression(&expr.args[0]);

                    match expr.args[0].type_of() {
                        Ty::Int => self.iloc_buf.push(Instruction::IRead(arg)),
                        Ty::Float => self.iloc_buf.push(Instruction::FRead(arg)),
                        Ty::Array { ty, .. } if matches!(&*ty, Ty::Int) => {
                            let dst = self.expr_to_reg(Operation::Load(arg));
                            self.iloc_buf.extend([
                                Instruction::Load { src: arg, dst },
                                Instruction::IRead(dst),
                            ]);
                        }
                        _ => unreachable!("not readable"),
                    }
                }
                _ => todo!(),
            },
            Stmt::TraitMeth { expr, def } => todo!(),
            Stmt::If { cond, blk, els } => {
                let cond = self.gen_expression(cond);
                let true_case = format!(".L{}:", self.next_label());
                let after_blk = format!(".L{}:", self.next_label());
                if let Some(els) = els {
                    self.iloc_buf.extend([
                        Instruction::CbrT { cond, loc: Loc(after_blk.replace(':', "")) },
                        Instruction::Label(true_case),
                    ]);
                    for stmt in &blk.stmts {
                        self.gen_statement(stmt)
                    }
                    self.iloc_buf.push(Instruction::Label(after_blk));
                    for stmt in &els.stmts {
                        self.gen_statement(stmt)
                    }
                } else {
                    self.iloc_buf.extend([
                        Instruction::CbrT { cond, loc: Loc(after_blk.replace(':', "")) },
                        Instruction::Label(true_case),
                    ]);
                    for stmt in &blk.stmts {
                        self.gen_statement(stmt)
                    }
                    self.iloc_buf.push(Instruction::Label(after_blk));
                }
            }
            Stmt::While { cond, stmts } => {
                let cond = self.gen_expression(cond);

                // The last 2 instructions will always be `comp ...` and
                // `test* ...`, copy them and insert them after the loop
                let start = self.iloc_buf.len() - 3;
                let post_loop_cond = self.iloc_buf[start..].to_vec();

                let while_case = format!(".L{}:", self.next_label());
                let after_blk = format!(".L{}:", self.next_label());

                self.iloc_buf.extend([
                    Instruction::CbrT { cond, loc: Loc(after_blk.replace(':', "")) },
                    Instruction::Label(while_case.clone()),
                ]);

                for stmt in &stmts.stmts {
                    self.gen_statement(stmt)
                }

                // the comp/test
                self.iloc_buf.extend(post_loop_cond);
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
                todo!()
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

    // TODO: we double the `leave; ret;` instructions for functions that actually return
    // TODO: we double the `leave; ret;` instructions for functions that actually return
    // TODO: we double the `leave; ret;` instructions for functions that actually return
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
            *size = self.stack_size;
        }
    }
}
