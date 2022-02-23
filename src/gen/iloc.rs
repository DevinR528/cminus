use std::{
    fs::{create_dir_all, OpenOptions},
    io::{ErrorKind, Write},
    path::Path,
    vec,
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{
    ast::{
        parse::symbol::Ident,
        types::{self as ty, FuncKind},
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
};

mod inst;

#[derive(Debug)]
crate struct IlocGen<'ctx> {
    iloc_buf: Vec<Instruction>,

    stack_size: usize,

    vars: HashMap<Ident, Loc>,
    globals: HashMap<Ident, Global>,

    registers: HashMap<Ident, Reg>,
    expr_regs: HashMap<(BinOp, Reg, Reg), Reg>,
    values: HashMap<Val, Reg>,
    curr_register: usize,

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
            curr_register: 0,
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

        file.write_all(format!("{}\n{}\n", globals, assembly).as_bytes()).map_err(|e| e.to_string())
    }

    crate fn to_global(&self, glob: &Global) -> String {
        use std::fmt::Write;

        let mut buf = String::new();
        match glob {
            Global::Text { name, content, mutable } => {
                if *mutable {
                    writeln!(buf, ".global {}\n.data", name);
                }
                writeln!(buf, "{}:    .string {:?}\n.text", name, content);
            }
            Global::Int { name, content, mutable } => {
                if *mutable {
                    writeln!(buf, ".global {}\n.data", name);
                }
                writeln!(buf, "{}:   .quad {}\n.text", name, content);
            }
            Global::Float { name, content, mutable } => {
                if *mutable {
                    writeln!(buf, ".global {}\n.data", name);
                }
                writeln!(buf, "{}:   .quad {}\n.text", name, content);
            }
            Global::Char { name, content, mutable } => {
                if *mutable {
                    writeln!(buf, ".global {}\n.data", name);
                }
                writeln!(buf, "{}:   .quad {}\n.text", name, content);
            }
        };
        buf
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
    fn expr_to_reg(&mut self, expr: (BinOp, Reg, Reg)) -> Reg {
        let num = self.curr_register;
        if let Some(num) = self.expr_regs.get(&expr) {
            *num
        } else {
            self.curr_register += 1;
            self.expr_regs.insert(expr, Reg::Var(num));
            Reg::Var(num)
        }
    }

    fn gen_expression(&mut self, expr: &'ctx Expr) -> Reg {
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
            Expr::Array { ident, exprs, ty } => todo!(),
            Expr::Urnary { op, expr, ty } => todo!(),
            Expr::Binary { op, lhs, rhs, ty } => {
                let lhs_reg = self.gen_expression(lhs);
                let rhs_reg = self.gen_expression(rhs);
                match op {
                    BinOp::Mul => {
                        let dst = self.expr_to_reg((*op, lhs_reg, rhs_reg));
                        self.iloc_buf.push(Instruction::Mult {
                            src_a: lhs_reg,
                            src_b: rhs_reg,
                            dst,
                        });
                        dst
                    }
                    BinOp::Div => {
                        todo!("No division in ILOC... weak")
                    }
                    BinOp::Rem => {
                        let dst = self.expr_to_reg((*op, lhs_reg, rhs_reg));
                        self.iloc_buf.push(Instruction::Mod {
                            src_a: lhs_reg,
                            src_b: rhs_reg,
                            dst,
                        });
                        dst
                    }
                    BinOp::Add => {
                        let dst = self.expr_to_reg((*op, lhs_reg, rhs_reg));
                        self.iloc_buf.push(Instruction::Add {
                            src_a: lhs_reg,
                            src_b: rhs_reg,
                            dst,
                        });
                        dst
                    }
                    BinOp::Sub => {
                        let dst = self.expr_to_reg((*op, lhs_reg, rhs_reg));
                        self.iloc_buf.push(Instruction::Sub {
                            src_a: lhs_reg,
                            src_b: rhs_reg,
                            dst,
                        });
                        dst
                    }
                    BinOp::LeftShift => {
                        let dst = self.expr_to_reg((*op, lhs_reg, rhs_reg));
                        self.iloc_buf.push(Instruction::LShift {
                            src_a: lhs_reg,
                            src_b: rhs_reg,
                            dst,
                        });
                        dst
                    }
                    BinOp::RightShift => {
                        let dst = self.expr_to_reg((*op, lhs_reg, rhs_reg));
                        self.iloc_buf.push(Instruction::RShift {
                            src_a: lhs_reg,
                            src_b: rhs_reg,
                            dst,
                        });
                        dst
                    }
                    BinOp::Lt => todo!(),
                    BinOp::Le => todo!(),
                    BinOp::Ge => todo!(),
                    BinOp::Gt => todo!(),
                    BinOp::Eq => todo!(),
                    BinOp::Ne => todo!(),
                    BinOp::BitAnd => {
                        let dst = self.expr_to_reg((*op, lhs_reg, rhs_reg));
                        self.iloc_buf.push(Instruction::And {
                            src_a: lhs_reg,
                            src_b: rhs_reg,
                            dst,
                        });
                        dst
                    }
                    BinOp::BitXor => {
                        todo!("No exclusive OR in ILOC...")
                    }
                    BinOp::BitOr => {
                        let dst = self.expr_to_reg((*op, lhs_reg, rhs_reg));
                        self.iloc_buf.push(Instruction::Or { src_a: lhs_reg, src_b: rhs_reg, dst });
                        dst
                    }
                    BinOp::And => {
                        let dst = self.expr_to_reg((*op, lhs_reg, rhs_reg));
                        self.iloc_buf.push(Instruction::And {
                            src_a: lhs_reg,
                            src_b: rhs_reg,
                            dst,
                        });
                        dst
                    }
                    BinOp::Or => {
                        let dst = self.expr_to_reg((*op, lhs_reg, rhs_reg));
                        self.iloc_buf.push(Instruction::Or { src_a: lhs_reg, src_b: rhs_reg, dst });
                        dst
                    }
                    BinOp::AddAssign => {
                        unreachable!("this is converted to a full add expression")
                    }
                    BinOp::SubAssign => {
                        unreachable!("this is converted to a full sub expression")
                    }
                }
            }
            Expr::Parens(expr) => self.gen_expression(expr),
            Expr::Call { path, args, type_args, def } => todo!(),
            Expr::TraitMeth { trait_, args, type_args, def } => todo!(),
            Expr::FieldAccess { lhs, def, rhs } => todo!(),
            Expr::StructInit { path, fields, def } => todo!(),
            Expr::EnumInit { path, variant, items, def } => todo!(),
            Expr::ArrayInit { items, ty } => todo!(),
            Expr::Value(val) => {
                let tmp = self.value_to_reg(val.clone());
                match val {
                    Val::Float(_) => todo!(),
                    Val::Int(i) => self
                        .iloc_buf
                        .push(Instruction::ImmLoad { src: inst::Val::Integer(*i), dst: tmp }),
                    Val::Char(_) => todo!(),
                    Val::Bool(_) => todo!(),
                    Val::Str(_) => todo!(),
                }
                tmp
            }
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
                                self.iloc_buf.push(Instruction::F2I { src: val, dst })
                            }
                            t => todo!("{:?}", t),
                        }
                    }
                    LValue::Deref { indir, expr, ty } => todo!(),
                    LValue::Array { ident, exprs, ty } => todo!(),
                    LValue::FieldAccess { lhs, def, rhs, field_idx } => todo!(),
                }
            }
            Stmt::Call { expr, def } => match expr.path.to_string().as_str() {
                "print" if !matches!(def.kind, FuncKind::Normal | FuncKind::Pointer) => {
                    assert!(expr.args.len() == 1);
                    let arg = self.gen_expression(&expr.args[0]);
                    self.iloc_buf.push(Instruction::IWrite(arg));
                }
                "scan" if !matches!(def.kind, FuncKind::Normal | FuncKind::Pointer) => {
                    assert!(expr.args.len() == 1);
                    let arg = self.gen_expression(&expr.args[0]);
                    self.iloc_buf.push(Instruction::IRead(arg));
                }
                _ => todo!(),
            },
            Stmt::TraitMeth { expr, def } => todo!(),
            Stmt::If { cond, blk, els } => {}
            Stmt::While { cond, stmts } => todo!(),
            Stmt::Match { expr, arms, ty } => todo!(),
            Stmt::Ret(ex, _) => {
                let ret_reg = self.gen_expression(ex);
                self.iloc_buf.push(Instruction::ImmRet(ret_reg));
            }
            Stmt::Exit => todo!(),
            Stmt::Block(_) => todo!(),
            Stmt::InlineAsm(_) => todo!(),
            Stmt::Builtin(_) => todo!(),
        }
    }
}

impl<'ast> Visit<'ast> for IlocGen<'ast> {
    fn visit_const(&mut self, var: &'ast Const) {
        fn convert_to_const(e: &Expr) -> Val {
            match e {
                Expr::Urnary { op, expr, ty } => todo!(),
                Expr::Binary { op, lhs, rhs, ty } => todo!(),
                Expr::Parens(_) => todo!(),
                Expr::StructInit { path, fields, def } => todo!(),
                Expr::EnumInit { path, variant, items, def } => todo!(),
                Expr::ArrayInit { items, ty } => todo!(),
                Expr::Value(val) => val.clone(),
                _ => unreachable!("not valid const expressions"),
            }
        }
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
                        mutable: var.mutable,
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
                        mutable: var.mutable,
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
                        mutable: var.mutable,
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
                        mutable: var.mutable,
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
                        mutable: var.mutable,
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
    }
}

fn type_ident_reg(ident: Ident, ty: &Ty) -> Reg {
    todo!()
}
