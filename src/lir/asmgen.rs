use std::{
    collections::{HashMap, HashSet},
    fs::OpenOptions,
    io::Write,
    path::Path,
    vec,
};

use crate::lir::{
    asmgen::inst::{CondFlag, FloatRegister, USABLE_FLOAT_REGS},
    lower::{BinOp, Expr, Func, LValue, Stmt, Ty, Val, Var},
    visit::Visit,
};

mod inst;
use inst::{Global, Instruction, Location, Register, ARG_REGS};

use self::inst::USABLE_REGS;

const STATIC_PREAMBLE: &str = r#"
.text

.char_wformat: .string "%c\n"
.int_wformat: .string "%d\n"
.float_wformat: .string "%f\n"
.str_wformat: .string "%s\n"
.char_rformat: .string "%c"
.int_rformat: .string "%d"
.float_rformat: .string "%f"
.bool_true: .string "true"
.bool_false: .string "false""#;

#[derive(Debug)]
crate struct CodeGen<'ctx> {
    asm_buf: Vec<Instruction>,
    globals: HashMap<&'ctx str, Global>,
    used_regs: HashSet<Register>,
    used_float_regs: HashSet<FloatRegister>,
    current_stack: usize,
    vars: HashMap<&'ctx str, Location>,
    path: &'ctx Path,
}

impl<'ctx> CodeGen<'ctx> {
    crate fn new(path: &'ctx Path) -> CodeGen<'ctx> {
        Self {
            asm_buf: vec![],
            globals: HashMap::new(),
            used_regs: HashSet::new(),
            used_float_regs: HashSet::new(),
            current_stack: 0,
            vars: HashMap::new(),
            path,
        }
    }

    crate fn to_asm(&self, inst: &Instruction) -> String {
        let mnemonic_from = |size: usize| match size {
            1 => "b",
            4 => "q", // TODO: everything is 64 bits
            8 => "q",
            _ => "big",
            // _ => unreachable!("larger than 8 bytes isn't valid to move in one go"),
        };

        match inst {
            Instruction::Meta(meta) => meta.clone(),
            Instruction::Label(label) => {
                format!("{}:", label)
            }
            Instruction::Push { loc, size } => {
                let mnemonic = mnemonic_from(*size);
                format!("    push{} {}", mnemonic, loc)
            }
            Instruction::Call(call) => format!("    call {}", call),
            Instruction::Jmp(_) => todo!(),
            Instruction::Leave => "    leave".to_owned(),
            Instruction::Ret => "    ret".to_owned(),
            Instruction::Cmp { src, dst } => format!("    cmp {}, {}", src, dst),
            Instruction::Mov { src, dst } => format!("    mov {}, {}", src, dst),
            Instruction::FloatMov { src, dst } => format!("    movsd {}, {}", src, dst),
            Instruction::SizedMov { src, dst, size } => {
                let mnemonic = mnemonic_from(*size);
                format!("    mov{} {}, {}", mnemonic, src, dst)
            }
            Instruction::CondMov { src, dst, cond } => {
                format!("    cmov{} {}, {}", cond.to_string(), src, dst)
            }
            Instruction::Load { src, dst, size: _ } => {
                // TODO: re-enable
                // let mnemonic = mnemonic_from(*size);
                format!("    leaq {}, {}", src, dst)
            }
            Instruction::Alloca { amount, reg } => format!("    sub ${}, %{}", amount, reg),
            Instruction::Math { src, dst, op } => {
                format!("    {} {}, {}", op.as_instruction(), src, dst)
            }
            Instruction::FloatMath { src, dst, op } => {
                format!("    {}ss {}, {}", op.as_instruction(), src, dst)
            }
            Instruction::Cvt { src, dst } => format!("    cvtss2sd {}, {}", src, dst),
        }
    }

    crate fn to_glob(&self, glob: &Global) -> String {
        match glob {
            Global::Text { name, content } => {
                format!("{}\n    .string {:?}\n    .text", name, content)
            }
            Global::Int { name, content } => {
                format!("{}\n    .long {}\n    .section .rodata", name, content)
            }
        }
    }

    crate fn dump_asm(&self) -> Result<(), String> {
        let mut p = self.path.to_path_buf();
        p.set_extension("s");

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(p)
            .map_err(|e| e.to_string())?;

        let globals = self
            .globals
            .values()
            .map(|decl| self.to_glob(decl))
            .collect::<Vec<String>>()
            .join("\n");
        let assembly =
            self.asm_buf.iter().map(|inst| self.to_asm(inst)).collect::<Vec<String>>().join("\n");

        file.write_all(format!("{}\n{}\n{}\n", STATIC_PREAMBLE, globals, assembly).as_bytes())
            .map_err(|e| e.to_string())
    }

    fn free_reg(&mut self) -> Register {
        let reg = *USABLE_REGS.difference(&self.used_regs).next().expect("ran out of registers");
        self.use_reg(reg);
        reg
    }

    fn use_reg(&mut self, reg: Register) -> bool {
        self.used_regs.insert(reg)
    }

    fn clear_regs_except(&mut self, loc: Option<&Location>) {
        self.used_regs.clear();
        let loc = if let Some(l) = loc {
            l
        } else {
            return;
        };
        match loc {
            Location::RegAddr { reg, .. } | Location::Register(reg) => {
                self.use_reg(*reg);
            }
            Location::Const { .. }
            | Location::Label(_)
            | Location::NamedOffset(_)
            | Location::FloatReg(_)
            | Location::Indexable { .. }
            | Location::NumberedOffset { .. } => {}
        };
    }

    fn free_float_reg(&mut self) -> FloatRegister {
        let reg = *USABLE_FLOAT_REGS
            .difference(&self.used_float_regs)
            .next()
            .expect("ran out of FloatRegister");
        self.use_float_reg(reg);
        reg
    }

    fn use_float_reg(&mut self, reg: FloatRegister) -> bool {
        self.used_float_regs.insert(reg)
    }

    fn clear_float_regs_except(&mut self, loc: Option<&Location>) {
        self.used_float_regs.clear();
        let loc = if let Some(l) = loc {
            l
        } else {
            return;
        };
        match loc {
            Location::FloatReg(reg) => {
                self.use_float_reg(*reg);
            }
            Location::Const { .. }
            | Location::Register(_)
            | Location::RegAddr { .. }
            | Location::Label(_)
            | Location::NamedOffset(_)
            | Location::Indexable { .. }
            | Location::NumberedOffset { .. } => {}
        };
    }

    #[allow(clippy::single_match)]
    fn order_operands(&mut self, lval: &mut Location, rval: &mut Location) {
        match (lval, rval) {
            (lval, rval @ Location::Const { .. }) => {
                std::mem::swap(lval, rval);
            }
            _ => {}
        }
    }

    #[allow(dead_code)]
    fn deref_to_value(&self, _ptr: Location, _ty: &Ty) {}

    fn index_arr(
        &mut self,
        arr: Location,
        exprs: &'ctx [Expr],
        ele_size: usize,
    ) -> Option<Location> {
        Some(if let Location::NumberedOffset { offset, reg } = arr {
            // Const indexing
            if let Expr::Value(Val::Int(idx)) = exprs[0] {
                if exprs.len() == 1 {
                    Location::Indexable {
                        reg,
                        end: offset,
                        ele_pos: offset - ((idx as usize) * ele_size),
                    }
                } else {
                    todo!("multidim arrays")
                }
            // TODO: add bounds checking maybe
            // Dynamic indexing
            } else if exprs.len() == 1 {
                let rval = self.build_value(&exprs[0], None)?;

                let tmpidx = self.free_reg();
                let array_reg = self.free_reg();
                // movq -16(%rbp), %rdx // array
                // movq -8(%rbp), %rbx // index
                // addq %rdx, %rbx
                // movq (%rbx), %rax
                self.asm_buf.extend_from_slice(&[
                    Instruction::SizedMov {
                        src: rval,
                        dst: Location::Register(tmpidx),
                        size: exprs[0].type_of().size(),
                    },
                    Instruction::SizedMov {
                        src: arr,
                        dst: Location::Register(array_reg),
                        size: ele_size,
                    },
                    Instruction::Math {
                        src: Location::Const { val: Val::Int(ele_size as isize) },
                        dst: Location::Register(tmpidx),
                        op: BinOp::Mul,
                    },
                    Instruction::Math {
                        src: Location::Register(tmpidx),
                        dst: Location::Register(array_reg),
                        op: BinOp::Add,
                    },
                    Instruction::SizedMov {
                        src: Location::NumberedOffset { offset: 0, reg: array_reg },
                        dst: Location::Register(tmpidx),
                        size: ele_size,
                    },
                ]);

                Location::Register(tmpidx)
            } else {
                todo!("multidim arrays")
            }
        } else {
            unreachable!("array must be numbered offset location")
        })
    }

    fn push_stack(&mut self, ty: &Ty) {
        match ty {
            Ty::Array { size, ty } => {
                for _el in 0..*size {
                    self.asm_buf.push(Instruction::Push {
                        loc: Location::Const { val: ty.null_val() },
                        size: ty.size(),
                    });
                }
            }
            Ty::Struct { ident: _, gen: _, def } => {
                for field in &def.fields {
                    self.push_stack(&field.ty)
                }
            }
            Ty::Enum { ident: _, gen: _, def: _ } => todo!(),
            Ty::String | Ty::Ptr(_) | Ty::Int | Ty::Float => {
                self.asm_buf.push(Instruction::Push {
                    loc: Location::Const { val: ty.null_val() },
                    size: 8,
                });
            }
            Ty::Char | Ty::Bool => {
                self.asm_buf.push(Instruction::Push {
                    loc: Location::Const { val: ty.null_val() },
                    size: 4,
                });
            }
            _ => unreachable!(),
        }
    }

    fn alloc_stack(&mut self, name: &'ctx str, ty: &Ty) -> Location {
        self.push_stack(ty);

        self.current_stack += ty.size();

        let ref_loc = Location::NumberedOffset { offset: self.current_stack, reg: Register::RBP };

        self.vars.insert(name, ref_loc.clone());
        ref_loc
    }

    fn alloc_arg(&mut self, count: usize, name: &'ctx str, ty: &Ty) -> Location {
        let size = match ty {
            // An array is converted to a pointer like thing
            Ty::Array { .. } => 8,
            t => t.size(),
        };

        self.current_stack += size;

        let ref_loc = Location::NumberedOffset { offset: self.current_stack, reg: Register::RBP };

        self.asm_buf.extend_from_slice(&[Instruction::Push {
            loc: Location::Register(ARG_REGS[count]),
            size,
        }]);

        self.vars.insert(name, ref_loc.clone());
        ref_loc
    }

    fn get_pointer(&mut self, expr: &'ctx LValue) -> Option<Location> {
        Some(match expr {
            LValue::Ident { ident, ty: _ } => self.vars.get(ident.as_str())?.clone(),
            LValue::Deref { indir: _, expr: _, ty: _ } => todo!(),
            LValue::Array { ident, exprs, ty } => {
                let arr = self.vars.get(ident.as_str())?.clone();
                let ele_size = if let Ty::Array { ty, .. } = ty {
                    ty.size()
                } else {
                    unreachable!("array type must be array")
                };
                self.index_arr(arr, exprs, ele_size)?
            }
            LValue::FieldAccess { lhs: _, def: _, rhs: _, field_idx: _ } => todo!(),
        })
    }

    fn build_value(&mut self, expr: &'ctx Expr, assigned: Option<&str>) -> Option<Location> {
        Some(match expr {
            Expr::Ident { ident, ty: _ } => self.vars.get(ident.as_str())?.clone(),
            Expr::Deref { indir: _, expr: _, ty: _ } => todo!(),
            Expr::AddrOf(_) => todo!(),
            Expr::Array { ident, exprs, ty } => {
                let arr = self.vars.get(ident.as_str())?.clone();
                let ele_size = if let Ty::Array { ty, .. } = ty {
                    ty.size()
                } else {
                    unreachable!("array type must be array {:?}", ty)
                };
                self.index_arr(arr, exprs, ele_size)?
            }
            Expr::Urnary { op: _, expr: _, ty: _ } => todo!(),
            Expr::Binary { op, lhs, rhs, ty } => {
                let mut lloc = self.build_value(lhs, None)?;
                let mut rloc = self.build_value(rhs, None)?;
                self.order_operands(&mut lloc, &mut rloc);

                if matches!(ty, Ty::Float) {
                    let register = self.free_float_reg();
                    let lfloatloc = if let Location::Const { .. } = lloc {
                        self.asm_buf.extend_from_slice(&[
                            // This transfers the constant to the stack then we can push it to a
                            // xmm[x] reg
                            Instruction::Push { loc: lloc, size: 8 },
                            Instruction::FloatMov {
                                src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                                dst: Location::FloatReg(register),
                            },
                            Instruction::FloatMov {
                                src: Location::FloatReg(register),
                                dst: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                            },
                        ]);
                        Location::NumberedOffset { offset: 0, reg: Register::RSP }
                    } else {
                        lloc
                    };

                    let rfloatloc = if rloc.is_stack_offset() {
                        self.asm_buf.push(Instruction::FloatMov {
                            src: rloc.clone(),
                            dst: Location::FloatReg(register),
                        });
                        Location::FloatReg(register)
                    } else {
                        rloc
                    };

                    self.asm_buf.push(Instruction::from_binop_float(
                        lfloatloc,
                        rfloatloc.clone(),
                        op,
                    ));

                    rfloatloc
                } else {
                    let inst = if let Location::NumberedOffset { .. } = &rloc {
                        let new_reg = self.free_reg();
                        let x = vec![
                            Instruction::Mov {
                                src: rloc.clone(),
                                dst: Location::Register(new_reg),
                            },
                            Instruction::from_binop(lloc, Location::Register(new_reg), op),
                        ];
                        rloc = Location::Register(new_reg);
                        x
                    } else {
                        vec![Instruction::from_binop(lloc, rloc.clone(), op)]
                    };

                    let store_reg = self.free_reg();
                    self.asm_buf.extend_from_slice(&inst);
                    self.asm_buf.extend_from_slice(&[Instruction::Mov {
                        src: rloc,
                        dst: Location::Register(store_reg),
                    }]);

                    Location::Register(store_reg)
                }
            }
            Expr::Parens(ex) => self.build_value(ex, assigned)?,
            Expr::Call { ident, args, type_args, def: _ } => {
                for (idx, arg) in args.iter().enumerate() {
                    let val = self.build_value(arg, None).unwrap();
                    let ty = arg.type_of();
                    if let Ty::Array { size: _, ty } = ty {
                        self.asm_buf.push(Instruction::Load {
                            src: val,
                            dst: Location::Register(ARG_REGS[idx]),
                            size: ty.size(),
                        });
                    } else {
                        self.asm_buf.push(Instruction::SizedMov {
                            src: val,
                            dst: Location::Register(ARG_REGS[idx]),
                            size: arg.type_of().size(),
                        });
                    }
                }

                let ident = if type_args.is_empty() {
                    ident.to_owned()
                } else {
                    format!(
                        "{}{}",
                        ident,
                        type_args.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("0"),
                    )
                };
                self.asm_buf.push(Instruction::Call(Location::Label(ident)));
                Location::Register(Register::RAX)
            }
            Expr::TraitMeth { trait_, args, type_args, def: _ } => {
                for (idx, arg) in args.iter().enumerate() {
                    let val = self.build_value(arg, None).unwrap();
                    let ty = arg.type_of();
                    if let Ty::Array { size: _, ty } = ty {
                        self.asm_buf.push(Instruction::Load {
                            src: val,
                            dst: Location::Register(ARG_REGS[idx]),
                            size: ty.size(),
                        });
                    } else {
                        self.asm_buf.push(Instruction::SizedMov {
                            src: val,
                            dst: Location::Register(ARG_REGS[idx]),
                            size: arg.type_of().size(),
                        });
                    }
                }
                let ident = format!(
                    "{}{}",
                    trait_,
                    type_args.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("0"),
                );
                self.asm_buf.push(Instruction::Call(Location::Label(ident)));
                Location::Register(Register::RAX)
            }
            Expr::FieldAccess { lhs: _, def: _, rhs: _, field_idx: _ } => todo!(),
            Expr::StructInit { name: _, fields: _, def: _ } => todo!(),
            Expr::EnumInit { ident: _, variant: _, items: _, def: _ } => todo!(),
            Expr::ArrayInit { items, ty } => {
                let _arr_size = ty.size();
                let ele_size = match ty {
                    Ty::Array { ty, .. } => ty.size(),
                    t => unreachable!("not an array for array init {:?}", t),
                };

                let lval: Option<Location> = try { self.vars.get(assigned?)?.clone() };

                let _start_of_arr_stack = self.current_stack;
                for (idx, item) in items.iter().enumerate() {
                    let rval = self.build_value(item, None).unwrap();

                    if let Some(Location::NumberedOffset { offset, reg }) = lval {
                        self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                            // Move the value on the right hand side of the `= here`
                            src: rval,
                            // to the left hand side of `here =`
                            dst: Location::Indexable {
                                end: offset,
                                ele_pos: offset - (idx * ele_size),
                                reg,
                            },
                            size: ele_size,
                        }]);
                    } else {
                        self.asm_buf
                            .extend_from_slice(&[Instruction::Push { loc: rval, size: ele_size }]);
                    }
                }
                if let Some(lval @ Location::NumberedOffset { .. }) = lval {
                    lval
                } else {
                    Location::NumberedOffset { offset: self.current_stack, reg: Register::RBP }
                }
            }
            Expr::Value(val) => Location::Const { val: val.clone() },
        })
    }

    fn gen_statement(&mut self, stmt: &'ctx Stmt) {
        match stmt {
            Stmt::VarDecl(vars) => {
                for var in vars {
                    self.alloc_stack(&var.ident, &var.ty);
                }
            }
            Stmt::Assign { lval, rval } => {
                if let Some(global) = self.globals.get_mut(lval.as_ident().unwrap()) {
                    match rval {
                        Expr::Ident { ident: _, ty: _ } => todo!(),
                        Expr::StructInit { name: _, fields: _, def: _ } => todo!(),
                        Expr::EnumInit { ident: _, variant: _, items: _, def: _ } => todo!(),
                        Expr::ArrayInit { items: _, ty: _ } => todo!(),
                        Expr::Value(val) => match val {
                            Val::Float(_) => todo!(),
                            Val::Int(num) => match global {
                                Global::Int { name: _, content } => {
                                    *content = *num as i64;
                                }
                                _ => todo!(),
                            },
                            Val::Bool(boo) => match global {
                                Global::Int { name: _, content } => {
                                    *content = *boo as i64;
                                }
                                _ => todo!(),
                            },
                            Val::Char(c) => match global {
                                Global::Text { name: _, content } => {
                                    *content = c.to_string();
                                }
                                _ => todo!(),
                            },
                            Val::Str(s) => match global {
                                Global::Text { name: _, content } => {
                                    *content = s.clone();
                                }
                                _ => todo!(),
                            },
                        },
                        _ => {}
                    }
                } else {
                    let lloc = self.get_pointer(lval).unwrap();
                    let rloc = self.build_value(rval, lval.as_ident()).unwrap();

                    if let Location::Const { .. } = lloc {
                        unreachable!("{:?}", lloc);
                    }

                    if lloc == rloc {
                        return;
                    }

                    let ty = lval.type_of();
                    let size = match ty {
                        Ty::Array { ty, .. } => ty.size(),
                        t => t.size(),
                    };
                    // TODO: only add 3 instructions for const -> float
                    if matches!(ty, Ty::Float) {
                        if rloc.is_float_reg() {
                            self.clear_float_regs_except(Some(&lloc));
                            self.asm_buf.extend_from_slice(&[Instruction::FloatMov {
                                src: rloc,
                                dst: lloc,
                            }]);
                        } else if matches!(rloc, Location::Const { .. }) {
                            let register = self.free_float_reg();

                            self.clear_float_regs_except(Some(&lloc));

                            self.asm_buf.extend_from_slice(&[
                                // From stack pointer
                                Instruction::Push { loc: rloc, size: 8 },
                                // To xmm? to store as float
                                Instruction::FloatMov {
                                    src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                                    dst: Location::FloatReg(register),
                                },
                                // Move xmm? value to where it is supposed to be
                                Instruction::FloatMov {
                                    src: Location::FloatReg(register),
                                    dst: lloc,
                                },
                            ]);
                        } else {
                            let register = self.free_float_reg();
                            self.clear_float_regs_except(Some(&lloc));
                            self.asm_buf.extend_from_slice(&[
                                Instruction::FloatMov {
                                    src: rloc,
                                    dst: Location::FloatReg(register),
                                },
                                Instruction::FloatMov {
                                    src: Location::FloatReg(register),
                                    dst: lloc,
                                },
                            ]);
                        }
                    } else if lloc.is_stack_offset() && rloc.is_stack_offset() {
                        let register = self.free_reg();
                        self.clear_regs_except(None);

                        self.asm_buf.extend_from_slice(&[
                            Instruction::SizedMov {
                                // Move the value on the right hand side of the `= here`
                                src: rloc,
                                // to the left hand side of `here =`
                                dst: Location::Register(register),
                                size,
                            },
                            Instruction::SizedMov {
                                // Move the value on the right hand side of the `= here`
                                src: Location::Register(register),
                                // to the left hand side of `here =`
                                dst: lloc,
                                size,
                            },
                        ]);
                    } else {
                        self.clear_regs_except(Some(&lloc));

                        self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                            // Move the value on the right hand side of the `= here`
                            src: rloc,
                            // to the left hand side of `here =`
                            dst: lloc,
                            size,
                        }]);
                    }
                }
            }
            Stmt::Call { expr: _, def: _ } => todo!(),
            Stmt::TraitMeth { expr: _, def: _ } => todo!(),
            Stmt::If { cond: _, blk: _, els: _ } => todo!(),
            Stmt::While { cond: _, stmt: _ } => todo!(),
            Stmt::Match { expr: _, arms: _ } => todo!(),
            Stmt::Read(_) => todo!(),
            Stmt::Write { expr } => {
                fn format_str(ty: &Ty) -> &str {
                    match ty {
                        Ty::Ptr(_) | Ty::Ref(_) | Ty::Int | Ty::Bool => ".int_wformat",
                        Ty::String => ".str_wformat",
                        Ty::Char => ".char_wformat",
                        Ty::Float => ".float_wformat",
                        Ty::Array { ty, .. } => format_str(ty),
                        _ => unreachable!("not valid print strings"),
                    }
                }

                let expr_type = expr.type_of();
                let fmt_str = format_str(&expr_type).to_string();
                let size = expr_type.size();

                let val = self.build_value(expr, None).unwrap();
                if matches!(expr_type, Ty::Float) {
                    if matches!(val, Location::Const { .. }) {
                        self.used_float_regs.insert(FloatRegister::XMM0);
                        let register = self.free_float_reg();
                        self.clear_float_regs_except(None);

                        self.asm_buf.extend_from_slice(&[
                            // From stack pointer
                            Instruction::Push { loc: val, size: 8 },
                            // To xmm? to store as float
                            Instruction::FloatMov {
                                src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                                dst: Location::FloatReg(register),
                            },
                            // Move back see if this helps
                            Instruction::FloatMov {
                                src: Location::FloatReg(register),
                                dst: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                            },
                            Instruction::Cvt {
                                src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                                dst: Location::FloatReg(FloatRegister::XMM0),
                            },
                            Instruction::Mov {
                                src: Location::Const { val: Val::Int(1) },
                                dst: Location::Register(Register::RAX),
                            },
                            Instruction::Load {
                                src: Location::NamedOffset(fmt_str),
                                dst: Location::Register(Register::RDI),
                                size,
                            },
                            Instruction::Call(Location::Label("printf".to_owned())),
                        ]);
                    } else {
                        self.asm_buf.extend_from_slice(&[
                            Instruction::Cvt {
                                src: val,
                                dst: Location::FloatReg(FloatRegister::XMM0),
                            },
                            Instruction::Mov {
                                src: Location::Const { val: Val::Int(1) },
                                dst: Location::Register(Register::RAX),
                            },
                            Instruction::Load {
                                src: Location::NamedOffset(fmt_str),
                                dst: Location::Register(Register::RDI),
                                size,
                            },
                            Instruction::Call(Location::Label("printf".to_owned())),
                        ]);
                    }
                } else if matches!(expr_type, Ty::Bool) {
                    let free_reg = self.free_reg();
                    // TODO: print string "true" or "false"
                    // have globals and

                    // leaq .bool_false(%rip), %rsi
                    // cmp $1, %bool_loc
                    // leaq .bool_true(%rip), %freereg
                    // cmovz %freereg, %rsi
                    if let Location::Const { val: Val::Bool(b) } = val {
                        self.asm_buf.extend_from_slice(&[
                            Instruction::Load {
                                src: if b {
                                    Location::NamedOffset(".bool_true".into())
                                } else {
                                    Location::NamedOffset(".bool_false".into())
                                },
                                dst: Location::Register(Register::RSI),
                                size: 8,
                            },
                            Instruction::Mov {
                                src: Location::Const { val: Val::Int(0) },
                                dst: Location::Register(Register::RAX),
                            },
                            Instruction::Load {
                                src: Location::NamedOffset(".str_wformat".into()),
                                dst: Location::Register(Register::RDI),
                                size,
                            },
                            Instruction::Call(Location::Label("printf".to_owned())),
                        ]);
                    } else {
                        self.asm_buf.extend_from_slice(&[
                            Instruction::Load {
                                src: Location::NamedOffset(".bool_false".into()),
                                dst: Location::Register(Register::RSI),
                                size,
                            },
                            Instruction::Cmp {
                                src: Location::Const { val: Val::Int(1) },
                                dst: val,
                            },
                            Instruction::Load {
                                src: Location::NamedOffset(".bool_true".into()),
                                dst: Location::Register(free_reg),
                                size: 8,
                            },
                            Instruction::CondMov {
                                src: Location::Register(free_reg),
                                dst: Location::Register(Register::RSI),
                                cond: CondFlag::Eq,
                            },
                            Instruction::Mov {
                                src: Location::Const { val: Val::Int(0) },
                                dst: Location::Register(Register::RAX),
                            },
                            Instruction::Load {
                                src: Location::NamedOffset(".str_wformat".into()),
                                dst: Location::Register(Register::RDI),
                                size,
                            },
                            Instruction::Call(Location::Label("printf".to_owned())),
                        ]);
                    }
                } else {
                    self.asm_buf.extend_from_slice(&[
                        Instruction::Mov { src: val, dst: Location::Register(Register::RSI) },
                        Instruction::Mov {
                            src: Location::Const { val: Val::Int(0) },
                            dst: Location::Register(Register::RAX),
                        },
                        Instruction::Load {
                            src: Location::NamedOffset(fmt_str),
                            dst: Location::Register(Register::RDI),
                            size,
                        },
                        Instruction::Call(Location::Label("printf".to_owned())),
                    ]);
                }
            }
            Stmt::Ret(expr, _ty) => {
                let val = self.build_value(expr, None).unwrap();
                self.asm_buf.extend_from_slice(&[
                    // return value is stored in %rax
                    Instruction::Mov { src: val, dst: Location::Register(Register::RAX) },
                ]);
            }
            Stmt::Exit => {}
            Stmt::Block(_) => todo!(),
        }
    }
}

impl<'ast> Visit<'ast> for CodeGen<'ast> {
    fn visit_var(&mut self, var: &'ast Var) {
        let name = format!(".Lglobal_{}", var.ident);
        match var.ty {
            Ty::Generic { .. }
            | Ty::Array { .. }
            | Ty::Struct { .. }
            | Ty::Enum { .. }
            | Ty::Ptr(_)
            | Ty::Ref(_) => todo!(),
            Ty::String | Ty::Char => {
                self.globals.insert(
                    &var.ident,
                    Global::Text { name: name.clone(), content: String::new() },
                );
            }
            Ty::Float => todo!(),
            Ty::Int => {
                self.globals.insert(&var.ident, Global::Int { name: name.clone(), content: 0 });
            }

            Ty::Bool => {
                self.globals.insert(&var.ident, Global::Int { name: name.clone(), content: 0 });
            }
            Ty::Void => unreachable!(),
        };
        self.vars.insert(&var.ident, Location::NamedOffset(name));
    }

    fn visit_func(&mut self, func: &'ast Func) {
        if func.stmts.is_empty() {
            return;
        }

        self.asm_buf.extend_from_slice(&[
            Instruction::Meta(format!(
                ".global {name}\n.type {name},@function\n",
                name = func.ident
            )),
            Instruction::Label(func.ident.clone()),
            Instruction::Push { loc: Location::Register(Register::RBP), size: 8 },
            Instruction::Mov {
                src: Location::Register(Register::RSP),
                dst: Location::Register(Register::RBP),
            },
        ]);

        for (i, arg) in func.params.iter().enumerate() {
            let alloca = self.alloc_arg(i, &arg.ident, &arg.ty);
            self.vars.insert(&arg.ident, alloca);
        }

        for stmt in &func.stmts {
            self.gen_statement(stmt);
        }

        self.current_stack = 0;
        self.clear_regs_except(None);
        self.clear_float_regs_except(None);

        self.asm_buf.extend_from_slice(&[Instruction::Leave, Instruction::Ret]);
    }
}
