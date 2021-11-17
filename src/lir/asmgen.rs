use std::{
    fs::{create_dir_all, OpenOptions},
    io::{ErrorKind, Write},
    path::Path,
    vec,
};

use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

use crate::{
    ast::parse::symbol::Ident,
    lir::{
        asmgen::inst::{CondFlag, FloatRegister, JmpCond, USABLE_FLOAT_REGS},
        lower::{BinOp, CallExpr, Const, Expr, Func, LValue, Pat, Stmt, Ty, UnOp, Val},
        visit::Visit,
    },
};

mod inst;
use inst::{Global, Instruction, Location, Register, ARG_REGS, USABLE_REGS};

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
.bool_false: .string "false"
.bool_test: .quad 1"#;

const ZERO: Location = Location::Const { val: Val::Int(0) };
const ONE: Location = Location::Const { val: Val::Int(1) };

#[derive(Debug)]
crate struct CodeGen<'ctx> {
    asm_buf: Vec<Instruction>,
    globals: HashMap<Ident, Global>,
    used_regs: HashSet<Register>,
    used_float_regs: HashSet<FloatRegister>,
    current_stack: usize,
    total_stack: usize,
    vars: HashMap<Ident, Location>,
    path: &'ctx Path,
}

impl<'ctx> CodeGen<'ctx> {
    crate fn new(path: &'ctx Path) -> CodeGen<'ctx> {
        Self {
            asm_buf: vec![],
            globals: HashMap::default(),
            used_regs: HashSet::default(),
            used_float_regs: HashSet::default(),
            current_stack: 0,
            total_stack: 0,
            vars: HashMap::default(),
            path,
        }
    }

    #[allow(clippy::useless_format)]
    crate fn to_asm(&self, inst: &Instruction) -> String {
        const FIRST: usize = 24;
        const SECOND: usize = 20;
        const SHORT: usize = 8;
        const COMMENT: usize = 29;
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
            Instruction::Push { loc, size, comment } => {
                let mnemonic = mnemonic_from(*size);
                format!(
                    "    push{}{:a$}{:b$}# {}",
                    mnemonic,
                    loc,
                    "",
                    comment,
                    a = FIRST,
                    b = COMMENT
                )
            }
            Instruction::Pop { loc, size, comment } => {
                let mnemonic = mnemonic_from(*size);
                format!(
                    "    pop{}{:a$}{:b$}# {}",
                    mnemonic,
                    loc,
                    "",
                    comment,
                    a = FIRST + 1,
                    b = COMMENT
                )
            }
            Instruction::Call(call) => format!("    call{:>a$}", call, a = FIRST),
            Instruction::Jmp(label) => format!("    jmp{:>a$}", label, a = FIRST),
            Instruction::CondJmp { loc, cond } => {
                format!("    j{}{:>a$}", cond.to_string(), loc, a = FIRST)
            }
            Instruction::Leave => "    leave".to_owned(),
            Instruction::Ret => "    ret".to_owned(),
            Instruction::Cmp { src, dst } => {
                format!("    cmpq{:a$},{:b$}", src, dst, a = FIRST, b = SECOND)
            }
            Instruction::Mov { src, dst, comment } => {
                format!(
                    "    mov {:a$},{:b$}{:c$}# {}",
                    src,
                    dst,
                    "",
                    comment,
                    a = FIRST + 1,
                    b = SECOND,
                    c = SHORT
                )
            }
            Instruction::FloatMov { src, dst } => {
                format!("    movsd{:a$},{:b$}", src, dst, a = FIRST, b = SECOND)
            }
            Instruction::SizedMov { src, dst, size } => {
                let mnemonic = mnemonic_from(*size);
                format!("    mov{}{:a$},{:b$}", mnemonic, src, dst, a = FIRST + 1, b = SECOND)
            }
            Instruction::CondMov { src, dst, cond } => {
                format!("    cmov{}{:a$},{:b$}", cond.to_string(), src, dst, a = FIRST, b = SECOND)
            }
            Instruction::Load { src, dst, size: _ } => {
                // TODO: re-enable
                // let mnemonic = mnemonic_from(*size);
                format!("    leaq{:a$},{:b$}", src, dst, a = FIRST + 1, b = SECOND)
            }
            Instruction::Alloca { amount, reg } => {
                format!("    sub{:a$},{:b$}", format!("${}", amount), reg, a = FIRST, b = SECOND)
            }
            Instruction::Math { src, dst, op } => {
                format!(
                    "    {}{:a$},{:b$}",
                    op.as_instruction(),
                    src,
                    dst,
                    a = FIRST + 1,
                    b = SECOND
                )
            }
            Instruction::FloatMath { src, dst, op } => {
                // HACK: ewww fix
                let mut ops = op.as_instruction().to_string();
                if ops == "imul" {
                    ops = ops.replace("i", "");
                }
                format!("    {}ss{:a$},{:b$}", ops, src, dst, a = FIRST, b = SECOND)
            }
            Instruction::Idiv(loc) => format!("    idiv{:a$}", loc, a = FIRST + 1),
            Instruction::Cvt { src, dst } => {
                format!("    cvtss2sd{:a$},{:b$}", src, dst, a = FIRST - 3, b = SECOND)
            }
            Instruction::Extend => format!("    cdq"),
        }
    }

    crate fn to_global(&self, glob: &Global) -> String {
        match glob {
            Global::Text { name, content } => {
                format!("{}: .string {:?}", name, content)
            }
            Global::Int { name, content } => {
                format!("{}:   .quad {}", name, content)
            }
            Global::Char { name, content } => {
                format!("{}:   .quad {}", name, content)
            }
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
        build_dir.set_extension("s");

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
            self.asm_buf.iter().map(|inst| self.to_asm(inst)).collect::<Vec<String>>().join("\n");

        file.write_all(format!("{}\n{}\n{}\n", STATIC_PREAMBLE, globals, assembly).as_bytes())
            .map_err(|e| e.to_string())
    }

    fn free_reg(&mut self) -> Register {
        let reg = *USABLE_REGS.difference(&self.used_regs).next().expect("ran out of registers");
        self.use_reg(reg);
        reg
    }

    fn free_reg_except(&mut self, reg: Register) -> Register {
        let reg = *USABLE_REGS
            .difference(&self.used_regs)
            .find(|r| reg != **r)
            .expect("ran out of registers");
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

    // FIXME: passing in a LOT of info hmmm
    fn order_operands(&mut self, lval: &mut Location, rval: &mut Location, op: &BinOp, ty: &Ty) {
        // TODO: audit for consistency i.e. idiv may needs some flip flops too
        // YAY I made math work UGH!!
        if matches!(op, BinOp::Sub) || matches!((ty, op), (Ty::Float, BinOp::Div | BinOp::Rem)) {
            std::mem::swap(lval, rval);
        }
        match (lval, rval) {
            // Only if operation is commutative
            (_, rval @ Location::Const { .. })
                if matches!(op, BinOp::Sub | BinOp::Div | BinOp::Rem) =>
            {
                let reg = self.free_reg();
                self.asm_buf.push(Instruction::SizedMov {
                    src: rval.clone(),
                    dst: Location::Register(reg),
                    size: 8,
                });
                *rval = Location::Register(reg);
            }
            (lval, rval @ Location::Const { .. }) => {
                std::mem::swap(lval, rval);
            }
            _ => {}
        }
    }

    fn promote_to_float(&mut self, _lval: &mut Location, _rval: &mut Location) {}

    #[allow(dead_code)]
    fn deref_to_value(&self, _ptr: Location, _ty: &Ty) {}

    fn index_arr(
        &mut self,
        arr: Location,
        exprs: &'ctx [Expr],
        ele_size: usize,
        by_value: bool,
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
                let index_val = self.build_value(&exprs[0], None)?;

                let tmpidx = self.free_reg();
                let array_reg = self.free_reg();
                // lea -16(%rbp), %rxx // array
                // movq -8(%rbp), %rxy // index * ele_size
                // addq %rxy, %rxx
                // movq (%rxx), %rax
                self.asm_buf.extend_from_slice(&[
                    Instruction::SizedMov {
                        src: index_val,
                        dst: Location::Register(tmpidx),
                        size: exprs[0].type_of().size(),
                    },
                    Instruction::Math {
                        src: Location::Const { val: Val::Int(ele_size as isize) },
                        dst: Location::Register(tmpidx),
                        op: BinOp::Mul,
                    },
                    Instruction::Load {
                        src: arr,
                        dst: Location::Register(array_reg),
                        size: ele_size,
                    },
                    Instruction::Math {
                        src: Location::Register(tmpidx),
                        dst: Location::Register(array_reg),
                        op: BinOp::Add,
                    },
                    Instruction::SizedMov {
                        src: if by_value {
                            Location::NumberedOffset { offset: 0, reg: array_reg }
                        } else {
                            Location::Register(array_reg)
                        },
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

    fn call_scanf(&mut self, expr: &'ctx Expr) {
        fn format_str(ty: &Ty) -> &str {
            match ty {
                Ty::Ptr(_) | Ty::Ref(_) | Ty::Int | Ty::Bool => ".int_rformat",
                Ty::String => ".str_rformat",
                Ty::Char => ".char_rformat",
                Ty::Float => ".float_rformat",
                Ty::Array { ty, .. } => format_str(ty),
                _ => unreachable!("not valid print strings"),
            }
        }

        let mut pushed = false;
        if self.total_stack % 16 != 0 {
            self.asm_buf.push(Instruction::Push { loc: ZERO, size: 8, comment: "" });
            pushed = true;
        }

        let expr_type = expr.type_of();
        let fmt_str = format_str(&expr_type).to_string();
        let size = expr_type.size();

        let val = self.build_value(expr, None).unwrap();

        // if matches!(expr_type, Ty::String) {
        self.asm_buf.extend_from_slice(&[
            Instruction::Load { src: val, dst: Location::Register(Register::RSI), size: 8 },
            Instruction::Mov { src: ZERO, dst: Location::Register(Register::RAX), comment: "" },
            Instruction::Load {
                src: Location::NamedOffset(fmt_str),
                dst: Location::Register(Register::RDI),
                size,
            },
            Instruction::Call(Location::Label("scanf".to_owned())),
        ]);

        if pushed {
            self.asm_buf.push(Instruction::Math {
                src: Location::Const { val: Val::Int(8) },
                dst: Location::Register(Register::RSP),
                op: BinOp::Add,
            });
        }
    }

    fn call_printf(&mut self, expr: &'ctx Expr) {
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

        let mut val = self.build_value(expr, None).unwrap();

        if [
            Location::Register(Register::RSI),
            Location::Register(Register::RAX),
            Location::Register(Register::RDI),
        ]
        .contains(&val)
        {
            // TODO: make my comparator check better for free_regs_except so this works
            let free = *USABLE_REGS
                .difference(&self.used_regs)
                .find(|r| !matches!(r, Register::RAX | Register::RSI | Register::RDI))
                .expect("ran out of registers");

            self.used_regs.insert(free);
            self.asm_buf.push(Instruction::Mov {
                src: val,
                dst: Location::Register(free),
                comment: "we had to spill a printf register",
            });
            val = Location::Register(free);
        }

        if matches!(expr_type, Ty::Float) {
            if matches!(val, Location::Const { .. }) {
                if ((self.total_stack + 8) % 16) != 0 {
                    self.asm_buf.push(Instruction::Push { loc: ZERO, size: 8, comment: "" });
                }
                self.used_float_regs.insert(FloatRegister::XMM0);
                let register = self.free_float_reg();
                self.clear_float_regs_except(None);

                self.asm_buf.extend_from_slice(&[
                    // From stack pointer
                    Instruction::Push { loc: val, size: 8, comment: "" },
                    // To xmm? to store as float
                    Instruction::FloatMov {
                        src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                        dst: Location::FloatReg(register),
                    },
                    // FIXME: move back see if this helps, this may be redundant
                    Instruction::FloatMov {
                        src: Location::FloatReg(register),
                        dst: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                    },
                    Instruction::Cvt {
                        src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                        dst: Location::FloatReg(FloatRegister::XMM0),
                    },
                    Instruction::Mov {
                        src: ONE,
                        dst: Location::Register(Register::RAX),
                        comment: "",
                    },
                    Instruction::Load {
                        src: Location::NamedOffset(fmt_str),
                        dst: Location::Register(Register::RDI),
                        size,
                    },
                    Instruction::Call(Location::Label("printf".to_owned())),
                    Instruction::Math {
                        src: Location::Const { val: Val::Int(8) },
                        dst: Location::Register(Register::RSP),
                        op: BinOp::Add,
                    },
                ]);
                if ((self.total_stack + 8) % 16) != 0 {
                    self.asm_buf.push(Instruction::Math {
                        //
                        // - 16 because we push in the above seq of instructions
                        src: Location::Const { val: Val::Int(8) },
                        dst: Location::Register(Register::RSP),
                        op: BinOp::Add,
                    });
                }
            } else if let Location::Register(reg) = &val {
                if self.total_stack % 16 != 0 {
                    self.asm_buf.push(Instruction::Push {
                        loc: ZERO,
                        size: 8,
                        comment: "stack isn't 16 byte aligned",
                    });
                }
                self.asm_buf.extend_from_slice(&[
                    Instruction::Push {
                        loc: val.clone(),
                        size: 8,
                        comment: "register is not valid cvtss2sd loc",
                    },
                    Instruction::Cvt {
                        src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                        dst: Location::FloatReg(FloatRegister::XMM0),
                    },
                    Instruction::Pop { loc: val, size: 8, comment: "remove tmp from stack" },
                    Instruction::Mov {
                        src: ONE,
                        dst: Location::Register(Register::RAX),
                        comment: "",
                    },
                    Instruction::Load {
                        src: Location::NamedOffset(fmt_str),
                        dst: Location::Register(Register::RDI),
                        size,
                    },
                    Instruction::Call(Location::Label("printf".to_owned())),
                ]);
                if (self.total_stack % 16) != 0 {
                    self.asm_buf.push(Instruction::Math {
                        // `- 16` because we push in the above seq of instructions
                        src: Location::Const { val: Val::Int(8) },
                        dst: Location::Register(Register::RSP),
                        op: BinOp::Add,
                    });
                }
            } else {
                if self.total_stack % 16 != 0 {
                    self.asm_buf.push(Instruction::Push {
                        loc: ZERO,
                        size: 8,
                        comment: "stack isn't 16 byte aligned",
                    });
                }
                self.asm_buf.extend_from_slice(&[
                    Instruction::Cvt { src: val, dst: Location::FloatReg(FloatRegister::XMM0) },
                    Instruction::Mov {
                        src: ONE,
                        dst: Location::Register(Register::RAX),
                        comment: "",
                    },
                    Instruction::Load {
                        src: Location::NamedOffset(fmt_str),
                        dst: Location::Register(Register::RDI),
                        size,
                    },
                    Instruction::Call(Location::Label("printf".to_owned())),
                ]);
                if self.total_stack % 16 != 0 {
                    self.asm_buf.push(Instruction::Math {
                        //
                        // - 8 because we push only once if the stack is misaligned
                        src: Location::Const { val: Val::Int(8) },
                        dst: Location::Register(Register::RSP),
                        op: BinOp::Add,
                    });
                }
            }
        } else if matches!(expr_type, Ty::Bool) {
            // FERK! this took me hours to figure out, don't clobber registers
            // writing assembly by hand is DUMB
            let free_reg = *USABLE_REGS
                .difference(&self.used_regs)
                .find(|r| !matches!(r, Register::RAX | Register::RSI | Register::RDI))
                .expect("ran out of registers");
            self.used_regs.insert(free_reg);
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
                        src: ZERO,
                        dst: Location::Register(Register::RAX),
                        comment: "",
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
                        src: Location::NamedOffset(".bool_true".into()),
                        dst: Location::Register(Register::RSI),
                        size,
                    },
                    Instruction::Load {
                        src: Location::NamedOffset(".bool_false".into()),
                        dst: Location::Register(free_reg),
                        size: 8,
                    },
                    Instruction::Cmp { src: ZERO, dst: val },
                    Instruction::CondMov {
                        src: Location::Register(free_reg),
                        dst: Location::Register(Register::RSI),
                        cond: CondFlag::Eq,
                    },
                    Instruction::Mov {
                        src: ZERO,
                        dst: Location::Register(Register::RAX),
                        comment: "",
                    },
                    Instruction::Load {
                        src: Location::NamedOffset(".str_wformat".into()),
                        dst: Location::Register(Register::RDI),
                        size,
                    },
                    Instruction::Call(Location::Label("printf".to_owned())),
                ]);
            }
        } else if matches!(expr_type, Ty::String) {
            let first = if matches!(val, Location::NamedOffset(_)) {
                Instruction::Load { src: val, dst: Location::Register(Register::RSI), size: 8 }
            } else {
                Instruction::Mov {
                    src: val,
                    dst: Location::Register(Register::RSI),
                    comment: "str addr is stored on stack",
                }
            };
            self.asm_buf.extend_from_slice(&[
                first,
                Instruction::Mov { src: ZERO, dst: Location::Register(Register::RAX), comment: "" },
                Instruction::Load {
                    src: Location::NamedOffset(fmt_str),
                    dst: Location::Register(Register::RDI),
                    size,
                },
                Instruction::Call(Location::Label("printf".to_owned())),
            ]);
        } else {
            self.asm_buf.extend_from_slice(&[
                Instruction::Mov { src: val, dst: Location::Register(Register::RSI), comment: "" },
                Instruction::Mov { src: ZERO, dst: Location::Register(Register::RAX), comment: "" },
                Instruction::Load {
                    src: Location::NamedOffset(fmt_str),
                    dst: Location::Register(Register::RDI),
                    size,
                },
                Instruction::Call(Location::Label("printf".to_owned())),
            ]);
        }
    }

    fn push_stack(&mut self, ty: &Ty) {
        match ty {
            Ty::Array { size, ty } => {
                for _el in 0..*size {
                    // TODO: better just sub from %rsp
                    self.asm_buf.push(Instruction::Push {
                        loc: Location::Const { val: ty.null_val() },
                        size: ty.size(),
                        comment: "",
                    });
                }
            }
            Ty::Struct { ident: _, gen: _, def } => {
                for field in &def.fields {
                    self.push_stack(&field.ty)
                }
            }
            Ty::Enum { ident: _, gen: _, def } => {
                let mut largest_variant = 0;
                for var in &def.variants {
                    // Size is size of largest variant plus 8 for tag
                    // TODO: optimize tag sizes using `variants.len()`?
                    let curr = var.types.iter().map(|t| t.size()).sum::<usize>() + 8;
                    if curr > largest_variant {
                        largest_variant = curr
                    }
                }

                self.asm_buf.push(Instruction::Math {
                    src: Location::Const { val: Val::Int(largest_variant as isize) },
                    dst: Location::Register(Register::RSP),
                    op: BinOp::Sub,
                });
            }
            Ty::String | Ty::Ptr(_) | Ty::Int | Ty::Float | Ty::Char => {
                self.asm_buf.push(Instruction::Push {
                    loc: Location::Const { val: ty.null_val() },
                    size: 8,
                    comment: "",
                });
            }
            Ty::Bool => {
                self.asm_buf.push(Instruction::Push {
                    loc: Location::Const { val: ty.null_val() },
                    size: 4,
                    comment: "",
                });
            }
            _ => unreachable!(),
        }
    }

    fn alloc_stack(&mut self, name: Ident, ty: &Ty) -> Location {
        self.push_stack(ty);

        self.current_stack += ty.size();
        self.total_stack += ty.size();

        let ref_loc = Location::NumberedOffset { offset: self.current_stack, reg: Register::RBP };

        self.vars.insert(name, ref_loc.clone());
        ref_loc
    }

    fn alloc_arg(&mut self, count: usize, name: Ident, ty: &Ty) -> Location {
        let size = match ty {
            // An array is converted to a pointer like thing
            Ty::Array { .. } => 8,
            t => t.size(),
        };

        self.current_stack += size;
        self.total_stack += size;

        let ref_loc = Location::NumberedOffset { offset: self.current_stack, reg: Register::RBP };

        self.asm_buf.extend_from_slice(&[Instruction::Push {
            loc: Location::Register(ARG_REGS[count]),
            size,
            comment: "",
        }]);

        self.vars.insert(name, ref_loc.clone());
        ref_loc
    }

    fn get_pointer(&mut self, expr: &'ctx LValue) -> Option<Location> {
        Some(match expr {
            LValue::Ident { ident, ty: _ } => self.vars.get(ident)?.clone(),
            LValue::Deref { indir: _, expr: _, ty: _ } => todo!(),
            LValue::Array { ident, exprs, ty } => {
                let arr = self.vars.get(ident)?.clone();
                let ele_size = if let Ty::Array { ty, .. } = ty {
                    ty.size()
                } else {
                    unreachable!("array type must be array")
                };
                self.index_arr(arr, exprs, ele_size, false)?
            }
            LValue::FieldAccess { lhs: _, def: _, rhs: _, field_idx: _ } => todo!(),
        })
    }

    fn build_value(&mut self, expr: &'ctx Expr, assigned: Option<Ident>) -> Option<Location> {
        Some(match expr {
            Expr::Ident { ident, ty: _ } => self.vars.get(ident)?.clone(),
            Expr::Deref { indir: _, expr: _, ty: _ } => todo!(),
            Expr::AddrOf(_) => todo!(),
            Expr::Array { ident, exprs, ty } => {
                let arr = self.vars.get(ident)?.clone();
                let ele_size = if let Ty::Array { ty, .. } = ty { ty.size() } else { ty.size() };
                self.index_arr(arr, exprs, ele_size, true)?
            }
            Expr::Urnary { op, expr, ty } => {
                let val = self.build_value(expr, None)?;

                let register = self.free_reg();
                let val = if val.is_stack_offset() {
                    self.asm_buf.push(Instruction::Mov {
                        src: val,
                        dst: Location::Register(register),
                        comment: "urnary op was a memory location",
                    });
                    Location::Register(register)
                } else {
                    val
                };

                if matches!(op, UnOp::Not) {
                    if matches!(ty, Ty::Bool) {
                        let cond_reg = self.free_reg();
                        self.asm_buf.extend_from_slice(&[
                            Instruction::Mov {
                                src: ZERO,
                                dst: Location::Register(cond_reg),
                                comment: "binary compare move zero",
                            },
                            Instruction::Cmp { src: ONE, dst: val },
                            Instruction::CondMov {
                                src: Location::NamedOffset(".bool_test".into()),
                                dst: Location::Register(cond_reg),
                                cond: CondFlag::NotEq,
                            },
                        ]);
                        Location::Register(cond_reg)
                    } else {
                        let cond_reg = self.free_reg();
                        self.asm_buf.extend_from_slice(&[
                            Instruction::Mov {
                                src: ZERO,
                                dst: Location::Register(cond_reg),
                                comment: "binary compare move zero",
                            },
                            Instruction::Cmp { src: ZERO, dst: val },
                            Instruction::CondMov {
                                src: Location::NamedOffset(".bool_test".into()),
                                dst: Location::Register(cond_reg),
                                cond: CondFlag::Greater,
                            },
                        ]);
                        Location::Register(cond_reg)
                    }
                } else {
                    todo!("ones comp")
                }
            }
            Expr::Binary { op, lhs, rhs, ty } => {
                let mut lloc = self.build_value(lhs, None)?;
                let mut rloc = self.build_value(rhs, None)?;

                self.order_operands(&mut lloc, &mut rloc, op, ty);

                if matches!(ty, Ty::Float) {
                    let register = self.free_float_reg();
                    let mut lfloatloc = if let Location::Const { .. } = lloc {
                        self.asm_buf.extend_from_slice(&[
                            // This transfers the constant to the stack then we can push it to a
                            // xmm[x] reg
                            Instruction::Push { loc: lloc, size: 8, comment: "" },
                            Instruction::FloatMov {
                                src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                                dst: Location::FloatReg(register),
                            },
                            Instruction::FloatMov {
                                src: Location::FloatReg(register),
                                dst: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                            },
                        ]);
                        self.total_stack += 8;
                        Location::NumberedOffset { offset: 0, reg: Register::RSP }
                    } else {
                        lloc
                    };

                    let mut rfloatloc = if rloc.is_stack_offset() {
                        self.asm_buf.push(Instruction::FloatMov {
                            src: rloc.clone(),
                            dst: Location::FloatReg(register),
                        });
                        Location::FloatReg(register)
                    } else {
                        rloc
                    };

                    // Do type conversion
                    // TODO: REMOVE
                    self.promote_to_float(&mut lfloatloc, &mut rfloatloc);

                    self.asm_buf.extend_from_slice(&[Instruction::from_binop_float(
                        lfloatloc,
                        rfloatloc.clone(),
                        op,
                    )]);

                    self.clear_float_regs_except(Some(&rfloatloc));

                    rfloatloc
                } else {
                    if let Location::NumberedOffset { .. } = &rloc {
                        let new_reg = self.free_reg();

                        self.asm_buf.push(Instruction::Mov {
                            src: rloc.clone(),
                            dst: Location::Register(new_reg),
                            comment: "",
                        });
                        rloc = Location::Register(new_reg);
                    }

                    // If we use idiv and rax is full we gotta push and then pop it
                    let mut spilled_rax = false;
                    let mut spilled_rdx = false;

                    let inst = if op.is_cmp() {
                        let cond_reg = self.free_reg();
                        let x = Instruction::from_binop_cmp(
                            lloc.clone(),
                            rloc,
                            op,
                            Location::Register(cond_reg),
                            self,
                        );
                        rloc = Location::Register(cond_reg);
                        x
                    } else if matches!(op, BinOp::Div | BinOp::Rem) {
                        let rax_reg = Register::RAX;
                        let rdx_reg = Register::RDX;

                        // We have to spill `rax`
                        if self.used_regs.contains(&rax_reg) {
                            self.asm_buf.push(Instruction::Push {
                                loc: Location::Register(rax_reg),
                                size: 8,
                                comment: "rax used",
                            });

                            spilled_rax = true;
                            if matches!(rloc, Location::Register(Register::RAX)) {
                                let free_reg = self.free_reg_except(Register::RDX);
                                self.asm_buf.push(Instruction::Pop {
                                    loc: Location::Register(free_reg),
                                    size: 8,
                                    comment: "use new reg for rax contents right",
                                });
                                rloc = Location::Register(free_reg);
                                // Because we have "fixed" the problem by popping and moving into a
                                // non needed register I think this is "Okay-Doh'kay"
                                spilled_rax = false;
                            }
                            if matches!(lloc, Location::Register(Register::RAX)) {
                                let free_reg = self.free_reg_except(Register::RDX);
                                self.asm_buf.push(Instruction::Pop {
                                    loc: Location::Register(free_reg),
                                    size: 8,
                                    comment: "use new reg for rax contents right",
                                });
                                lloc = Location::Register(free_reg);
                                // Because we have "fixed" the problem by popping and moving into a
                                // non needed register I think this is "Okay-Doh'kay"
                                spilled_rax = false;
                            }
                        }

                        // We have to spill `rdx`
                        if self.used_regs.contains(&rdx_reg) {
                            self.asm_buf.push(Instruction::Push {
                                loc: Location::Register(rdx_reg),
                                size: 8,
                                comment: "rdx used",
                            });

                            spilled_rdx = true;
                            if matches!(lloc, Location::Register(Register::RDX)) {
                                // This is overkill but we can't have rdx here
                                let free_reg = self.free_reg_except(Register::RAX);
                                self.asm_buf.push(Instruction::Pop {
                                    loc: Location::Register(free_reg),
                                    size: 8,
                                    comment: "use new register for rdx contents left",
                                });
                                lloc = Location::Register(free_reg);
                                // see above rax comment
                                spilled_rdx = false;
                            }
                            if matches!(rloc, Location::Register(Register::RDX)) {
                                // This is overkill but we can't have rdx here
                                let free_reg = self.free_reg_except(Register::RAX);
                                self.asm_buf.push(Instruction::Pop {
                                    loc: Location::Register(free_reg),
                                    size: 8,
                                    comment: "use new register for rdx contents left",
                                });
                                rloc = Location::Register(free_reg);
                                // see above rax comment
                                spilled_rdx = false;
                            }
                        }

                        vec![
                            Instruction::Mov {
                                src: lloc.clone(),
                                dst: Location::Register(rax_reg),
                                comment: "move lhs to dividend `rdx:rax / whatever`",
                            },
                            Instruction::Extend,
                            // lloc is divided by rloc `lloc / rloc`
                            Instruction::Idiv(rloc.clone()),
                            Instruction::SizedMov {
                                src: Location::Register(rax_reg),
                                dst: rloc.clone(),
                                size: 8,
                            },
                        ]
                    } else {
                        Instruction::from_binop(lloc.clone(), rloc.clone(), op)
                    };

                    // FIXME: don't do it like this, ERROR prone (move this line after and
                    // everything breaks :( )
                    self.asm_buf.extend_from_slice(&inst);

                    // FIXME:
                    // The check for loc == register should be redundant
                    if spilled_rax && !matches!(rloc, Location::Register(Register::RAX)) {
                        self.asm_buf.push(Instruction::Pop {
                            loc: Location::Register(Register::RAX),
                            size: 8,
                            comment: "move back to rax",
                        });
                    }
                    if spilled_rdx && !matches!(lloc, Location::Register(Register::RDX)) {
                        self.asm_buf.push(Instruction::Pop {
                            loc: Location::Register(Register::RDX),
                            size: 8,
                            comment: "move back to rdx",
                        });
                    }

                    // TODO: is it ok for any mem_ref ??
                    if let Location::Register(rreg) = rloc {
                        self.clear_regs_except(Some(&Location::Register(rreg)));
                        Location::Register(rreg)
                    } else {
                        let store_reg = self.free_reg();
                        self.asm_buf.extend_from_slice(&[Instruction::Mov {
                            src: rloc,
                            dst: Location::Register(store_reg),
                            comment: "",
                        }]);
                        self.clear_regs_except(Some(&Location::Register(store_reg)));
                        Location::Register(store_reg)
                    }
                }
            }
            Expr::Parens(ex) => self.build_value(ex, assigned)?,
            Expr::Call { path, args, type_args, def: _ } => {
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
                    path.to_string()
                } else {
                    format!(
                        "{}{}",
                        path,
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
            Expr::FieldAccess { .. } => todo!(),
            Expr::StructInit { .. } => {
                todo!()
            }
            Expr::EnumInit { path: _, variant, items, def } => {
                let lval: Option<Location> = try { self.vars.get(&assigned?)?.clone() };
                let tag = def.variants.iter().position(|v| variant == &v.ident).unwrap();
                if let Some(mov_to @ Location::NumberedOffset { .. }) = &lval {
                    self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                        // Move the value on the right hand side of the `= here`
                        src: Location::Const { val: Val::Int(tag as isize) },
                        // to the left hand side of `here =`
                        // The start offset - the current item size + the tag bits which are first
                        dst: mov_to.clone(),
                        size: 8,
                    }]);
                } else {
                    todo!()
                }
                for item in items.iter() {
                    let rval = self.build_value(item, None).unwrap();

                    let ele_size = item.type_of().size();
                    if let Some(Location::NumberedOffset { offset, reg }) = lval {
                        self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                            // Move the value on the right hand side of the `= here`
                            src: rval,
                            // to the left hand side of `here =`
                            // The start offset - the current item size + the tag bits which are
                            // first
                            dst: Location::NumberedOffset {
                                offset: (offset - (ele_size + 8)),
                                reg,
                            },
                            size: ele_size,
                        }]);
                    } else {
                        self.asm_buf.extend_from_slice(&[Instruction::Push {
                            loc: rval,
                            size: ele_size,
                            comment: "",
                        }]);
                    }
                }
                if let Some(lval @ Location::NumberedOffset { .. }) = lval {
                    lval
                } else {
                    Location::NumberedOffset { offset: self.current_stack, reg: Register::RBP }
                }
            }
            Expr::ArrayInit { items, ty } => {
                let ele_size = match ty {
                    Ty::Array { ty, .. } => ty.size(),
                    t => unreachable!("not an array for array init {:?}", t),
                };

                let lval: Option<Location> = try { self.vars.get(&assigned?)?.clone() };

                // @cleanup: This is REALLY BAD don't push/movq for every ele of array at least once
                // @cleanup: This is REALLY BAD don't push/movq for every ele of array at least once
                // @cleanup: This is REALLY BAD don't push/movq for every ele of array at least once
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
                        self.asm_buf.extend_from_slice(&[Instruction::Push {
                            loc: rval,
                            size: ele_size,
                            comment: "",
                        }]);
                    }
                }
                if let Some(lval @ Location::NumberedOffset { .. }) = lval {
                    lval
                } else {
                    Location::NumberedOffset { offset: self.current_stack, reg: Register::RBP }
                }
            }
            Expr::Value(val) => match val {
                Val::Float(_) | Val::Int(_) | Val::Bool(_) => Location::Const { val: val.clone() },
                Val::Char(c) => Location::Const { val: Val::Int(*c as isize) },
                Val::Str(s) => {
                    let cleaned = s.name().replace("\"", "");
                    let name = format!(".Sstring_{}", self.asm_buf.len());
                    let x =
                        self.globals.entry(*s).or_insert(Global::Text { name, content: cleaned });
                    Location::NamedOffset(x.name().to_string())
                }
            },
        })
    }

    fn gen_statement(&mut self, stmt: &'ctx Stmt) {
        match stmt {
            Stmt::Const(var) => {
                panic!("{:?}", var);
                // TODO: deal with initializer
                self.alloc_stack(var.ident, &var.ty);
            }
            Stmt::Assign { lval, rval, is_let } => {
                if let Some(global) = self.globals.get_mut(&lval.as_ident().unwrap()) {
                    match rval {
                        Expr::Ident { ident: _, ty: _ } => todo!(),
                        Expr::StructInit { .. } => todo!(),
                        Expr::EnumInit { .. } => todo!(),
                        Expr::ArrayInit { .. } => todo!(),
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
                                Global::Char { name: _, content } => {
                                    *content = *c as u8;
                                }
                                hmm => todo!("{:?}", hmm),
                            },
                            Val::Str(s) => match global {
                                Global::Text { name: _, content } => {
                                    *content = s.name().to_string();
                                }
                                _ => todo!(),
                            },
                        },
                        _ => {}
                    }
                } else {
                    let lloc = if *is_let {
                        let ident = lval.as_ident().unwrap();
                        self.alloc_stack(ident, lval.type_of())
                    } else {
                        self.get_pointer(lval).unwrap()
                    };

                    let mut rloc = self.build_value(rval, lval.as_ident()).unwrap();

                    if let Location::Const { .. } = lloc {
                        unreachable!("ICE: assign to a constant {:?}", lloc);
                    }

                    if lloc == rloc {
                        return;
                    }

                    let ty = lval.type_of();
                    let size = match ty {
                        Ty::Array { ty, .. } => ty.size(),
                        t => t.size(),
                    };

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
                                Instruction::Push { loc: rloc, size: 8, comment: "" },
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
                                // Ugh stupid printf needs to be 16 bit aligned so we have to do
                                // our book keeping
                                Instruction::Math {
                                    src: Location::Const { val: Val::Int(8) },
                                    dst: Location::Register(Register::RSP),
                                    op: BinOp::Sub,
                                },
                            ]);
                        // Promote any non float register to float
                        // TODO: REMOVE
                        } else if matches!(
                            (&lloc, &rloc),
                            (Location::NumberedOffset { .. }, Location::Register(_))
                        ) {
                            let register = self.free_float_reg();
                            self.clear_float_regs_except(Some(&lloc));
                            self.asm_buf.extend_from_slice(&[
                                Instruction::Push {
                                    loc: rloc,
                                    size: 8,
                                    comment: "we are promoting an int to float",
                                },
                                Instruction::FloatMov {
                                    src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                                    dst: Location::FloatReg(register),
                                },
                                Instruction::FloatMov {
                                    src: Location::FloatReg(register),
                                    dst: lloc,
                                },
                                // Ugh stupid printf needs to be 16 bit aligned so we have to do
                                // our book keeping
                                Instruction::Math {
                                    src: Location::Const { val: Val::Int(8) },
                                    dst: Location::Register(Register::RSP),
                                    op: BinOp::Sub,
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
                    } else if matches!(ty, Ty::String) {
                        assert!(
                            matches!(rloc, Location::NamedOffset(_)),
                            "ICE: right hand term must be string const"
                        );

                        self.clear_regs_except(Some(&lloc));
                        if lloc.is_stack_offset() {
                            let reg = self.free_reg();
                            self.asm_buf.extend_from_slice(&[
                                Instruction::Load {
                                    src: rloc,
                                    dst: Location::Register(reg),
                                    size: 8,
                                },
                                Instruction::Mov {
                                    // Move the value on the right hand side of the `= here`
                                    src: Location::Register(reg),
                                    // to the left hand side of `here =`
                                    dst: lloc,
                                    comment: "move string addr",
                                },
                            ]);
                        } else {
                            self.asm_buf.extend_from_slice(&[Instruction::Load {
                                // Move the value on the right hand side of the `= here`
                                src: rloc,
                                // to the left hand side of `here =`
                                dst: lloc,
                                size,
                            }]);
                        }
                    } else if let (Ty::Array { .. }, Location::Register(reg)) = (ty, &lloc) {
                        self.clear_regs_except(Some(&lloc));

                        if rloc.is_stack_offset() {
                            let new = self.free_reg();
                            self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                                // Move the value on the right hand side of the `= here`
                                src: rloc,
                                // to the left hand side of `here =`
                                dst: Location::Register(new),
                                size,
                            }]);
                            rloc = Location::Register(new);
                        }

                        self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                            // Move the value on the right hand side of the `= here`
                            src: rloc,
                            // to the left hand side of `here =`
                            dst: Location::NumberedOffset { offset: 0, reg: *reg },
                            size,
                        }]);
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
                            size: 8,
                        }]);
                    }
                }
            }
            Stmt::Call { expr: CallExpr { path, args, type_args }, def } => {
                if "write" == &path.segs[0] {
                    self.call_printf(&args[0]);
                } else if "read" == &path.segs[0] {
                    self.call_scanf(&args[0]);
                } else {
                    // @copypaste this is the same as `Expr::Call`
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
                        path.to_string()
                    } else {
                        format!(
                            "{}{}",
                            path,
                            type_args.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("0"),
                        )
                    };
                    self.asm_buf.push(Instruction::Call(Location::Label(ident)));
                }
            }
            Stmt::TraitMeth { expr: _, def: _ } => todo!(),
            Stmt::If { cond, blk, els } => {
                let cond_val = self.build_value(cond, None).unwrap();
                // Check if true
                self.asm_buf.push(Instruction::Cmp {
                    src: Location::Const { val: Val::Int(1) },
                    dst: cond_val,
                });

                let name = format!(".jmpif{}", self.asm_buf.len());
                let else_or_uncond = Location::Label(name.clone());
                // Jump over the "then" block
                self.asm_buf
                    .push(Instruction::CondJmp { loc: else_or_uncond, cond: JmpCond::NotEq });

                for stmt in &blk.stmts {
                    self.gen_statement(stmt);
                }

                // Fall through or "merge" point
                let merge_label = format!(".mergeif{}", self.asm_buf.len());
                let merge_loc = Location::Label(merge_label.clone());
                self.asm_buf.push(Instruction::Jmp(merge_loc));

                self.asm_buf.push(Instruction::Label(name));
                if let Some(els) = els {
                    for stmt in &els.stmts {
                        self.gen_statement(stmt);
                    }
                }

                self.asm_buf.push(Instruction::Label(merge_label));
            }
            Stmt::While { cond, stmts } => {
                let uncond_label = format!(".uncondwhile{}", self.asm_buf.len());
                let uncond_loc = Location::Label(uncond_label.clone());
                self.asm_buf.push(Instruction::Jmp(uncond_loc));

                let name = format!(".jmpwhile{}", self.asm_buf.len());
                let loop_body = Location::Label(name.clone());
                self.asm_buf.push(Instruction::Label(name));
                // Start loop body
                for stmt in &stmts.stmts {
                    self.gen_statement(stmt);
                }

                self.asm_buf.push(Instruction::Label(uncond_label));
                let cond_val = self.build_value(cond, None).unwrap();
                // Check if true
                self.asm_buf.push(Instruction::Cmp {
                    src: Location::Const { val: Val::Int(1) },
                    dst: cond_val,
                });
                // Jump back to the loop body
                self.asm_buf.push(Instruction::CondJmp { loc: loop_body, cond: JmpCond::Eq });
            }
            Stmt::Match { expr, arms, ty: _ } => {
                let val = self.build_value(expr, None).unwrap();

                for (_idx, arm) in arms.iter().enumerate() {
                    match &arm.pat {
                        Pat::Enum { idx, items, .. } => {
                            // cmp enum tag to variant
                            self.asm_buf.push(Instruction::Cmp {
                                src: Location::Const { val: Val::Int(*idx as isize) },
                                dst: val.clone(),
                            });
                            // cmp each non wildcard pattern
                            for _item in items {
                                // TODO: recursively add cmp instructions
                                // only Binding::Values and enums actually emit cmp instructions
                            }
                        }
                        Pat::Array { size: _, items } => {
                            for _item in items {
                                // TODO: recursively add cmp instructions
                            }
                        }
                        Pat::Bind(_) => todo!(),
                    }
                }
                self.asm_buf.extend_from_slice(&[Instruction::Cmp {
                    src: Location::Const { val: Val::Int(69) },
                    dst: val,
                }]);
            }
            Stmt::Ret(expr, _ty) => {
                let val = self.build_value(expr, None).unwrap();
                self.asm_buf.extend_from_slice(&[
                    // return value is stored in %rax
                    Instruction::Mov {
                        src: val,
                        dst: Location::Register(Register::RAX),
                        comment: "",
                    },
                ]);
            }
            Stmt::Exit => {}
            Stmt::Block(_) => todo!(),
        }
    }
}

impl<'ast> Visit<'ast> for CodeGen<'ast> {
    fn visit_var(&mut self, var: &'ast Const) {
        let name = format!(".Lglobal_{}", var.ident);
        match var.ty {
            Ty::Generic { .. }
            | Ty::Array { .. }
            | Ty::Struct { .. }
            | Ty::Enum { .. }
            | Ty::Ptr(_)
            | Ty::Ref(_) => todo!(),
            Ty::String => {
                self.globals
                    .insert(var.ident, Global::Text { name: name.clone(), content: String::new() });
            }
            Ty::Char => {
                self.globals.insert(var.ident, Global::Char { name: name.clone(), content: 0xff });
            }
            Ty::Float => todo!(),
            Ty::Int => {
                self.globals.insert(var.ident, Global::Int { name: name.clone(), content: 0x0 });
            }

            Ty::Bool => {
                self.globals.insert(var.ident, Global::Int { name: name.clone(), content: 0x0 });
            }
            Ty::Void => unreachable!(),
        };
        self.vars.insert(var.ident, Location::NamedOffset(name));
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
            Instruction::Label(func.ident.name().to_string()),
            Instruction::Push { loc: Location::Register(Register::RBP), size: 8, comment: "" },
            Instruction::Mov {
                src: Location::Register(Register::RSP),
                dst: Location::Register(Register::RBP),
                comment: "",
            },
        ]);

        for (i, arg) in func.params.iter().enumerate() {
            let alloca = self.alloc_arg(i, arg.ident, &arg.ty);
            self.vars.insert(arg.ident, alloca);
        }

        for stmt in &func.stmts {
            self.gen_statement(stmt);
        }

        self.current_stack = 0;
        self.clear_regs_except(None);
        self.clear_float_regs_except(None);

        // HACK: so when other programs run our programs they don't non-zero exit
        if func.ident.name() == "main" {
            self.asm_buf.push(Instruction::SizedMov {
                src: ZERO,
                dst: Location::Register(Register::RAX),
                size: 8,
            });
        }

        self.asm_buf.extend_from_slice(&[Instruction::Leave, Instruction::Ret]);
    }
}
