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
            current_stack: 0,
            vars: HashMap::new(),
            path,
        }
    }

    crate fn to_asm(&self, inst: &Instruction) -> String {
        match inst {
            Instruction::Meta(meta) => meta.clone(),
            Instruction::Label(label) => {
                format!("{}:", label)
            }
            Instruction::Push(val) => format!("    push {}", val),
            Instruction::Call(call) => format!("    call {}", call),
            Instruction::Jmp(_) => todo!(),
            Instruction::Leave => "    leave".to_owned(),
            Instruction::Ret => "    ret".to_owned(),
            Instruction::Mov { src, dst } => format!("    mov {}, {}", src, dst),
            Instruction::Load { src, dst } => format!("    lea {}, {}", src, dst),
            Instruction::Alloca { amount, reg } => format!("    sub ${}, %{}", amount, reg),
            Instruction::Math { src, dst, op } => {
                format!("    {} {}, {}", op.as_instruction(), src, dst)
            }
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

    fn order_operands(&mut self, lval: &mut Location, rval: &mut Location) {
        match (lval, rval) {
            (lval @ Location::Const(_), rval) => {
                self.asm_buf.push(Instruction::Mov {
                    src: lval.clone(),
                    dst: Location::Register(Register::R12),
                });
                *lval = Location::Register(Register::R12);
            }
            (lval, rval @ Location::Const(_))
                if !matches!(lval, Location::NumberedOffset(_) | Location::NamedOffset(_)) =>
            {
                std::mem::swap(lval, rval)
            }
            (lval, rval @ Location::Const(_)) => {
                self.asm_buf.push(Instruction::Mov {
                    src: rval.clone(),
                    dst: Location::Register(Register::R14),
                });
                *rval = Location::Register(Register::R14);
                std::mem::swap(lval, rval);
            }
            _ => {}
        }
    }

    fn deref_to_value(&self, ptr: Location, ty: &Ty) {}

    fn index_arr(&self, arr_ptr: Location, idx_exprs: &'ctx [Expr]) -> Option<Location> {
        todo!()
    }

    fn alloc_stack(&mut self, name: &'ctx str, ty: &Ty) -> Location {
        let size = ty.size();

        let ref_loc = Location::NumberedOffset(self.current_stack as i64);
        self.current_stack += size;

        self.asm_buf
            .extend_from_slice(&[Instruction::Alloca { amount: size as i64, reg: Register::RSP }]);

        self.vars.insert(name, ref_loc.clone());
        ref_loc
    }

    fn alloc_arg(&mut self, count: usize, name: &'ctx str, ty: &Ty) -> Location {
        let size = ty.size();
        let ref_loc = Location::NumberedOffset(self.current_stack as i64);
        self.current_stack += size;

        self.asm_buf.extend_from_slice(&[Instruction::Push(Location::Register(ARG_REGS[count]))]);
        self.vars.insert(name, ref_loc.clone());
        ref_loc
    }

    fn get_pointer(&mut self, expr: &'ctx LValue) -> Option<Location> {
        Some(match expr {
            LValue::Ident { ident, ty } => self.vars.get(ident.as_str())?.clone(),
            LValue::Deref { indir, expr } => todo!(),
            LValue::Array { ident, exprs } => todo!(),
            LValue::FieldAccess { lhs, rhs } => todo!(),
        })
    }

    fn build_value(&mut self, expr: &'ctx Expr, assigned: Option<&str>) -> Option<Location> {
        Some(match expr {
            Expr::Ident { ident, ty } => self.vars.get(ident.as_str())?.clone(),
            Expr::Deref { indir, expr, ty } => todo!(),
            Expr::AddrOf(_) => todo!(),
            Expr::Array { ident, exprs, ty } => todo!(),
            Expr::Urnary { op, expr, ty } => todo!(),
            Expr::Binary { op, lhs, rhs, ty } => {
                let mut lloc = self.build_value(lhs, None)?;
                let mut rloc = self.build_value(rhs, None)?;

                self.order_operands(&mut lloc, &mut rloc);

                let inst = if let Location::NumberedOffset(num) = &rloc {
                    let x = vec![
                        Instruction::Mov {
                            src: rloc.clone(),
                            dst: Location::Register(Register::R11),
                        },
                        Instruction::from_binop(lloc, Location::Register(Register::R11), op),
                    ];
                    rloc = Location::Register(Register::R11);
                    x
                } else {
                    vec![Instruction::from_binop(lloc, rloc.clone(), op)]
                };
                self.asm_buf.extend_from_slice(&inst);
                self.asm_buf.extend_from_slice(&[Instruction::Mov {
                    src: rloc,
                    dst: Location::Register(Register::R10),
                }]);

                Location::Register(Register::R10)
            }
            Expr::Parens(_) => todo!(),
            Expr::Call { ident, args, type_args, def } => todo!(),
            Expr::TraitMeth { trait_, args, type_args, def } => todo!(),
            Expr::FieldAccess { lhs, def, rhs, field_idx } => todo!(),
            Expr::StructInit { name, fields, def } => todo!(),
            Expr::EnumInit { ident, variant, items, def } => todo!(),
            Expr::ArrayInit { items, ty } => todo!(),
            Expr::Value(val) => Location::Const(val.to_string()),
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

                    self.order_operands(&mut lloc, &mut rloc);

                    self.asm_buf.push(Instruction::Mov { src: lloc, dst: rloc });
                }
            }
            Stmt::Call { expr, def } => todo!(),
            Stmt::TraitMeth { expr, def } => todo!(),
            Stmt::If { cond, blk, els } => todo!(),
            Stmt::While { cond, stmt } => todo!(),
            Stmt::Match { expr, arms } => todo!(),
            Stmt::Read(_) => todo!(),
            Stmt::Write { expr } => {
                let fmt_str = match expr.type_of() {
                    Ty::Ptr(_) | Ty::Ref(_) | Ty::Int | Ty::Bool => ".int_wformat",
                    Ty::String => ".str_wformat",
                    Ty::Char => ".char_wformat",
                    Ty::Float => ".float_wformat",
                    _ => unreachable!("not valid print strings"),
                }
                .to_string();

                let val = self.build_value(expr, None).unwrap();
                self.asm_buf.extend_from_slice(&[
                    Instruction::Mov { src: val, dst: Location::Register(Register::RSI) },
                    Instruction::Mov {
                        src: Location::Const("0".to_string()),
                        dst: Location::Register(Register::RAX),
                    },
                    Instruction::Load {
                        src: Location::NamedOffset(fmt_str),
                        dst: Location::Register(Register::RDI),
                    },
                    Instruction::Call(Location::Label("printf".to_owned())),
                ]);
            }
            Stmt::Ret(_, _) => todo!(),
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
            Instruction::Push(Location::Register(Register::RBP)),
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
