use std::{
    collections::{HashMap, HashSet},
    fs::OpenOptions,
    io::Write,
    path::Path,
    vec,
};

use crate::{
    error::Error,
    lir::{
        asmgen::inst::{FloatRegister, USABLE_FLOAT_REGS},
        lower::{
            Adt, BinOp, CallExpr, Enum, Expr, Func, Impl, Item, LValue, MatchArm, Stmt, Struct, Ty,
            Val, Var,
        },
        visit::Visit,
    },
    typeck::TyCheckRes,
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
.float_rformat: .string "%f""#;

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
            4 => "l",
            8 => "q",
            _ => "big",
            _ => unreachable!("larger than 8 bytes isn't valid to move in one go"),
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
            Instruction::Mov { src, dst } => format!("    mov {}, {}", src, dst),
            Instruction::FloatMov { src, dst } => format!("    movsd {}, {}", src, dst),
            Instruction::SizedMov { src, dst, size } => {
                let mnemonic = mnemonic_from(*size);
                format!("    mov{} {}, {}", mnemonic, src, dst)
            }
            Instruction::Load { src, dst } => format!("    lea {}, {}", src, dst),
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

    fn clear_regs_except(&mut self, loc: &Location) {
        self.used_regs.clear();
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
            | Location::FloatReg(_)
            | Location::Indexable { .. }
            | Location::NumberedOffset { .. } => {}
        };
    }

    fn order_operands(&mut self, lval: &mut Location, rval: &mut Location) {
        match (lval, rval) {
            (lval, rval @ Location::Const { .. }) => {
                std::mem::swap(lval, rval);
            }
            _ => {}
        }
    }

    fn deref_to_value(&self, ptr: Location, ty: &Ty) {}

    fn index_arr(&self, arr: Location, exprs: &'ctx [Expr], ele_size: i64) -> Option<Location> {
        Some(if let Location::NumberedOffset { offset, reg } = arr {
            if let Expr::Value(Val::Int(idx)) = exprs[0] {
                if exprs.len() == 1 {
                    Location::Indexable { reg, end: offset, ele_idx: idx as i64, ele_size }
                } else {
                    todo!("multidim arrays")
                }
            } else {
                todo!("deal with dynamic array access")
            }
        } else {
            unreachable!("array must be numbered offset location")
        })
    }

    fn alloc_stack(&mut self, name: &'ctx str, ty: &Ty) -> Location {
        let size = ty.size();

        self.current_stack += size;

        let ref_loc =
            Location::NumberedOffset { offset: self.current_stack as i64, reg: Register::RBP };

        self.vars.insert(name, ref_loc.clone());
        ref_loc
    }

    fn alloc_arg(&mut self, count: usize, name: &'ctx str, ty: &Ty) -> Location {
        let size = ty.size();

        self.current_stack += size;

        let ref_loc =
            Location::NumberedOffset { offset: self.current_stack as i64, reg: Register::RBP };

        self.asm_buf.extend_from_slice(&[Instruction::Push {
            loc: Location::Register(ARG_REGS[count]),
            size,
        }]);

        self.vars.insert(name, ref_loc.clone());
        ref_loc
    }

    fn get_pointer(&mut self, expr: &'ctx LValue) -> Option<Location> {
        Some(match expr {
            LValue::Ident { ident, ty } => self.vars.get(ident.as_str())?.clone(),
            LValue::Deref { indir, expr } => todo!(),
            LValue::Array { ident, exprs, ty } => {
                let arr = self.vars.get(ident.as_str())?.clone();
                let ele_size = if let Ty::Array { ty, .. } = ty {
                    ty.size()
                } else {
                    unreachable!("array type must be array")
                };
                self.index_arr(arr, exprs, ele_size as i64)?
            }
            LValue::FieldAccess { lhs, rhs } => todo!(),
        })
    }

    fn build_value(&mut self, expr: &'ctx Expr, assigned: Option<&str>) -> Option<Location> {
        Some(match expr {
            Expr::Ident { ident, ty } => self.vars.get(ident.as_str())?.clone(),
            Expr::Deref { indir, expr, ty } => todo!(),
            Expr::AddrOf(_) => todo!(),
            Expr::Array { ident, exprs, ty } => {
                let arr = self.vars.get(ident.as_str())?.clone();
                let ele_size = if let Ty::Array { ty, .. } = ty {
                    ty.size()
                } else {
                    unreachable!("array type must be array")
                };
                self.index_arr(arr, exprs, ele_size as i64)?
            }
            Expr::Urnary { op, expr, ty } => todo!(),
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

                    let rfloatloc = if rloc.is_memory_ref() {
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
            Expr::Parens(_) => todo!(),
            Expr::Call { ident, args, type_args, def } => {
                for (idx, arg) in args.iter().enumerate() {
                    let val = self.build_value(arg, None).unwrap();
                    self.asm_buf.push(Instruction::SizedMov {
                        src: val,
                        dst: Location::Register(ARG_REGS[idx]),
                        size: arg.type_of().size(),
                    });
                }
                self.asm_buf.push(Instruction::Call(Location::Label(ident.to_owned())));
                Location::Register(Register::RAX)
            }
            Expr::TraitMeth { trait_, args, type_args, def } => todo!(),
            Expr::FieldAccess { lhs, def, rhs, field_idx } => todo!(),
            Expr::StructInit { name, fields, def } => todo!(),
            Expr::EnumInit { ident, variant, items, def } => todo!(),
            Expr::ArrayInit { items, ty } => todo!(),
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
                        Expr::Ident { ident, ty } => todo!(),
                        Expr::StructInit { name, fields, def } => todo!(),
                        Expr::EnumInit { ident, variant, items, def } => todo!(),
                        Expr::ArrayInit { items, ty } => todo!(),
                        Expr::Value(val) => match val {
                            Val::Float(_) => todo!(),
                            Val::Int(num) => match global {
                                Global::Int { name, content } => {
                                    *content = *num as i64;
                                }
                                _ => todo!(),
                            },
                            Val::Bool(boo) => match global {
                                Global::Int { name, content } => {
                                    *content = *boo as i64;
                                }
                                _ => todo!(),
                            },
                            Val::Char(c) => match global {
                                Global::Text { name, content } => {
                                    *content = c.to_string();
                                }
                                _ => todo!(),
                            },
                            Val::Str(s) => match global {
                                Global::Text { name, content } => {
                                    *content = s.clone();
                                }
                                _ => todo!(),
                            },
                        },
                        _ => {}
                    }
                } else {
                    let mut lloc = self.get_pointer(lval).unwrap();
                    let mut rloc = self.build_value(rval, None).unwrap();

                    if let Location::Const { .. } = lloc {
                        unreachable!("{:?}", lloc);
                    }

                    let ty = lval.type_of();

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
                                // To xmm3 to store as float
                                Instruction::FloatMov {
                                    src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                                    dst: Location::FloatReg(register),
                                },
                                // Move xmm3 value to where it is supposed to be
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
                    } else {
                        self.clear_regs_except(&lloc);

                        self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                            // Move the value on the right hand side of the `= here`
                            src: rloc,
                            // to the left hand side of `here =`
                            dst: lloc,
                            size: match ty {
                                Ty::Array { ty, .. } => ty.size(),
                                t => t.size(),
                            },
                        }]);
                    }
                }
            }
            Stmt::Call { expr, def } => todo!(),
            Stmt::TraitMeth { expr, def } => todo!(),
            Stmt::If { cond, blk, els } => todo!(),
            Stmt::While { cond, stmt } => todo!(),
            Stmt::Match { expr, arms } => todo!(),
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

                let val = self.build_value(expr, None).unwrap();
                if matches!(expr_type, Ty::Float) {
                    self.asm_buf.extend_from_slice(&[
                        Instruction::Cvt { src: val, dst: Location::FloatReg(FloatRegister::XMM0) },
                        Instruction::Mov {
                            src: Location::Const { val: Val::Int(1) },
                            dst: Location::Register(Register::RAX),
                        },
                        Instruction::Load {
                            src: Location::NamedOffset(fmt_str),
                            dst: Location::Register(Register::RDI),
                        },
                        Instruction::Call(Location::Label("printf".to_owned())),
                    ]);
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
                        },
                        Instruction::Call(Location::Label("printf".to_owned())),
                    ]);
                }
            }
            Stmt::Ret(expr, ty) => {
                let val = self.build_value(expr, None).unwrap();
                self.asm_buf.extend_from_slice(&[
                    // return value is stored in %rax
                    Instruction::Mov { src: val, dst: Location::Register(Register::RAX) },
                ]);
            }
            Stmt::Exit => todo!(),
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

        self.asm_buf.extend_from_slice(&[Instruction::Leave, Instruction::Ret]);
    }
}
