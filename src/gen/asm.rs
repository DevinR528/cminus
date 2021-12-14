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
    gen::asm::inst::{
        CondFlag, FloatRegister, Global, Instruction, JmpCond, Location, Register, ARG_FLOAT_REGS,
        ARG_REGS, USABLE_FLOAT_REGS, USABLE_REGS,
    },
    lir::{
        lower::{
            BinOp, Binding, Builtin, CallExpr, Const, Expr, FieldInit, Func, LValue, MatchArm, Pat,
            Stmt, Struct, Ty, UnOp, Val,
        },
        visit::Visit,
    },
};

crate mod inst;

const STATIC_PREAMBLE: &str = r#"
.text

.bool_true: .string "true"
.bool_false: .string "false"
.bool_test: .quad 1"#;

const ZERO: Location = Location::Const { val: Val::Int(0) };
const ONE: Location = Location::Const { val: Val::Int(1) };

const RAX: Location = Location::Register(Register::RAX);
const RSP: Location = Location::Register(Register::RSP);
const RBP: Location = Location::Register(Register::RBP);
const RDX: Location = Location::Register(Register::RDX);

const XMM0: Location = Location::FloatReg(FloatRegister::XMM0);

#[derive(Clone, Copy, Debug)]
enum CanClearRegs {
    Yes,
    No,
}

#[derive(Debug)]
crate struct CodeGen<'ctx> {
    asm_buf: Vec<Instruction>,
    globals: HashMap<Ident, Global>,
    used_regs: HashSet<Register>,
    used_float_regs: HashSet<FloatRegister>,
    current_stack: usize,
    total_stack: usize,
    current_fn_params: HashSet<Ident>,
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
            current_fn_params: HashSet::default(),
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
            _ => "too_big_break",
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
            Instruction::Math { src, dst, op, cmt } => {
                format!(
                    "    {}q{:a$},{:b$}{:c$}# {}",
                    op.as_instruction(),
                    src,
                    dst,
                    "",
                    cmt,
                    a = FIRST + 1,
                    b = SECOND,
                    c = SHORT
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
        use std::fmt::Write;

        let mut buf = String::new();
        match glob {
            Global::Text { name, content, mutable } => {
                if *mutable {
                    writeln!(buf, ".global {}\n.data", name);
                }
                writeln!(buf, "{}: .string {:?}\n.text", name, content);
            }
            Global::Int { name, content, mutable } => {
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
            Global::Array { name, content, mutable } => {
                writeln!(buf, "    .global {}", name,);
                if *mutable {
                    writeln!(buf, "    .data");
                } else {
                    writeln!(buf, "    .section    .rodata");
                }
                writeln!(
                    buf,
                    "    .align 32\n    .type {}, @object\n    .size {}, {}\n{}:",
                    name,
                    name,
                    content.len() * 8,
                    name,
                );
                for item in content {
                    match item {
                        Val::Float(v) => writeln!(buf, "    .quad {}", v.to_bits()),
                        Val::Int(v) => writeln!(buf, "    .quad {}", v),
                        Val::Char(v) => writeln!(buf, "    .quad {}", (*v) as u8),
                        Val::Bool(v) => writeln!(buf, "    .quad {}", if *v { 1 } else { 0 }),
                        Val::Str(v) => writeln!(buf, "    .string {:?}", v),
                    };
                }
                writeln!(buf, ".text");
            }
        };
        buf
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

    fn clear_regs_except(&mut self, loc: Option<&Location>, can_clear: CanClearRegs) {
        if let CanClearRegs::No = can_clear {
            return;
        }
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
            | Location::NamedOffsetIndex { .. }
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

    fn clear_float_regs_except(&mut self, loc: Option<&Location>, can_clear: CanClearRegs) {
        if let CanClearRegs::No = can_clear {
            return;
        }
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
            | Location::NamedOffsetIndex { .. }
            | Location::Indexable { .. }
            | Location::NumberedOffset { .. } => {}
        };
    }

    // FIXME: passing in a LOT of info hmmm
    fn order_operands(
        &mut self,
        lval: &mut Location,
        rval: &mut Location,
        op: &BinOp,
        ty: &Ty,
    ) -> bool {
        let mut swapped = false;
        // TODO: audit for consistency i.e. idiv may needs some flip flops too
        //
        // So this flips any sub operation and the match below flips them back...
        if matches!(op, BinOp::Sub) || matches!((ty, op), (Ty::Float, BinOp::Div | BinOp::Rem)) {
            std::mem::swap(lval, rval);
            swapped = true;
        }
        match (lval, rval) {
            // Only if operation is commutative
            (_, rval @ Location::Const { .. })
                if matches!(op, BinOp::Sub | BinOp::Div | BinOp::Rem) =>
            {
                // TODO: fails for subtraction
                // assert!(!swapped);
                swapped = true;
                let reg = self.free_reg();
                self.asm_buf.push(Instruction::SizedMov {
                    src: rval.clone(),
                    dst: Location::Register(reg),
                    size: 8,
                });
                *rval = Location::Register(reg);
            }
            (lval, rval @ Location::Const { .. }) => {
                assert!(!swapped);
                swapped = true;
                std::mem::swap(lval, rval);
            }
            _ => {}
        };
        swapped
    }

    /// This will ALWAYS return the address to the indexed value in the array.
    fn index_arr(
        &mut self,
        arr: Location,
        exprs: &'ctx [Expr],
        ele_size: usize,
        is_addr: bool,
    ) -> Option<Location> {
        // println!("{}", is_addr);

        let mov_or_load = |array_reg| {
            if is_addr {
                Instruction::Load {
                    src: arr.clone(),
                    dst: Location::Register(array_reg),
                    size: ele_size,
                }
            } else {
                Instruction::SizedMov {
                    src: arr.clone(),
                    dst: Location::Register(array_reg),
                    size: ele_size,
                }
            }
        };

        Some(if let Location::NumberedOffset { offset, reg } = arr {
            if let Expr::Value(Val::Int(idx)) = exprs[0] {
                if exprs.len() == 1 {
                    let tmpidx = self.free_reg();
                    let array_reg = self.free_reg();
                    self.asm_buf.extend_from_slice(&[
                        mov_or_load(array_reg),
                        Instruction::Math {
                            src: Location::Const { val: Val::Int((ele_size as isize) * idx) },
                            dst: Location::Register(array_reg),
                            op: BinOp::Add,
                            cmt: "array index + idx * ele size",
                        },
                    ]);
                    Location::Register(array_reg)
                } else {
                    todo!("multidim arrays")
                }
            // TODO: add bounds checking maybe
            // Dynamic indexing
            } else if exprs.len() == 1 {
                let index_val = self.build_value(&exprs[0], None, CanClearRegs::No, false)?;

                let tmpidx = self.free_reg();
                let array_reg = self.free_reg();
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
                        cmt: "array index * ele size number",
                    },
                    mov_or_load(array_reg),
                    Instruction::Math {
                        src: Location::Register(tmpidx),
                        dst: Location::Register(array_reg),
                        op: BinOp::Add,
                        cmt: "array index + idx * ele size",
                    },
                ]);

                Location::Register(array_reg)
            } else {
                todo!("multidim arrays")
            }
        } else if let Location::NamedOffset(name) = &arr {
            if let Expr::Value(Val::Int(idx)) = exprs[0] {
                if exprs.len() == 1 {
                    let tmpidx = self.free_reg();
                    let array_reg = self.free_reg();
                    self.asm_buf.extend_from_slice(&[
                        mov_or_load(array_reg),
                        Instruction::Math {
                            src: Location::Const { val: Val::Int((ele_size as isize) * idx) },
                            dst: Location::Register(array_reg),
                            op: BinOp::Add,
                            cmt: "array index + idx * ele size",
                        },
                    ]);
                    Location::Register(array_reg)
                } else {
                    todo!("multidim arrays")
                }
            // TODO: add bounds checking maybe
            // Dynamic indexing
            } else if exprs.len() == 1 {
                let index_val = self.build_value(&exprs[0], None, CanClearRegs::No, false)?;

                let tmpidx = self.free_reg();
                let array_reg = self.free_reg();
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
                        cmt: "array index * ele size named",
                    },
                    mov_or_load(array_reg),
                    Instruction::Math {
                        src: Location::Register(tmpidx),
                        dst: Location::Register(array_reg),
                        op: BinOp::Add,
                        cmt: "array index + idx * ele size",
                    },
                ]);

                Location::Register(array_reg)
            } else {
                todo!("non const int index")
            }
        } else {
            unreachable!("array must be numbered offset location")
        })
    }

    fn gen_call_expr(
        &mut self,
        path: &ty::Path,
        kind: FuncKind,
        ret_ty: &Ty,
        args: &'ctx [Expr],
        type_args: &[Ty],
        can_clear: CanClearRegs,
    ) -> Location {
        let called_with_float = args.iter().any(|ex| matches!(ex.type_of(), Ty::Float));

        let mut pushed_to_align_float_stack = false;
        if (self.total_stack % 16 != 0 || self.total_stack == 0) {
            self.asm_buf.push(Instruction::Math {
                src: Location::Const { val: Val::Int((16 - self.total_stack % 16) as isize) },
                dst: RSP,
                op: BinOp::Sub,
                cmt: "printf stack misaligned",
            });

            if self.total_stack == 0 {
                self.total_stack += 16;
            }
            pushed_to_align_float_stack = true;
        }

        let mut spilled = vec![];
        // number of float arguments
        let mut float_count = 0;
        for (idx, arg) in args.iter().enumerate() {
            if self.used_regs.contains(&ARG_REGS[idx]) {
                spilled.push(ARG_REGS[idx]);
                self.asm_buf.push(Instruction::Push {
                    loc: Location::Register(ARG_REGS[idx]),
                    size: arg.type_of().size(),
                    comment: "had to spill reg for call",
                });
            }
            self.use_reg(ARG_REGS[idx]);

            let ty = arg.type_of();

            if let Ty::Array { size: _, ty } = ty {
                if matches!(arg, Expr::Array { .. }) {
                    let val = self
                        .build_value(arg, None, can_clear, true)
                        .unwrap_or_else(|| panic!("{:?}", arg));
                    if let Location::Register(reg) = val {
                        self.asm_buf.push(Instruction::SizedMov {
                            src: Location::NumberedOffset { offset: 0, reg },
                            dst: Location::Register(ARG_REGS[idx]),
                            size: ty.size(),
                        });
                    } else {
                        self.asm_buf.push(Instruction::SizedMov {
                            src: val,
                            dst: Location::Register(ARG_REGS[idx]),
                            size: ty.size(),
                        });
                    }
                } else if self.current_fn_params.contains(&arg.as_ident()) {
                    let val = self
                        .build_value(arg, None, can_clear, false)
                        .unwrap_or_else(|| panic!("{:?}", arg));
                    self.asm_buf.push(Instruction::SizedMov {
                        src: val,
                        dst: Location::Register(ARG_REGS[idx]),
                        size: ty.size(),
                    });
                } else {
                    let val = self
                        .build_value(arg, None, can_clear, false)
                        .unwrap_or_else(|| panic!("{:?}", arg));

                    // TODO: do we always want to move by ref for arrays
                    self.asm_buf.push(Instruction::Load {
                        src: val,
                        dst: Location::Register(ARG_REGS[idx]),
                        size: ty.size(),
                    });
                }
                continue;
            }

            let val = self
                .build_value(arg, None, can_clear, false)
                .unwrap_or_else(|| panic!("{:?}", arg));

            if matches!(ty, Ty::Ptr(..)) {
                // TODO: this should be ok since `CodeGen::build_value` deals with the load?
                self.asm_buf.push(Instruction::SizedMov {
                    src: val,
                    dst: Location::Register(ARG_REGS[idx]),
                    size: arg.type_of().size(),
                });
            } else if matches!(ty, Ty::Bool) {
                if let Location::Const { val: Val::Bool(b) } = val {
                    self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                        src: if b { ONE } else { ZERO },
                        dst: Location::Register(ARG_REGS[idx]),
                        size: 8,
                    }]);
                } else {
                    self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                        src: val,
                        dst: Location::Register(ARG_REGS[idx]),
                        size: 8,
                    }]);
                }
            } else if let Ty::ConstStr(..) = ty {
                if matches!(val, Location::NumberedOffset { .. } | Location::Register(..)) {
                    self.asm_buf.push(Instruction::Mov {
                        src: val,
                        dst: Location::Register(ARG_REGS[idx]),
                        comment: "move address of const str",
                    });
                } else {
                    self.asm_buf.push(Instruction::Load {
                        src: val,
                        dst: Location::Register(ARG_REGS[idx]),
                        size: ty.size(),
                    });
                }
            } else if matches!(ty, Ty::Float) {
                if matches!(val, Location::Const { .. }) {
                    self.used_float_regs.insert(ARG_FLOAT_REGS[float_count]);
                    self.clear_float_regs_except(None, can_clear);

                    self.asm_buf.extend_from_slice(&[
                        // From stack pointer
                        Instruction::Push { loc: val, size: 8, comment: "" },
                        Instruction::Cvt {
                            src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                            dst: Location::FloatReg(ARG_FLOAT_REGS[float_count]),
                        },
                        Instruction::Math {
                            src: Location::Const { val: Val::Int(8) },
                            dst: RSP,
                            op: BinOp::Add,
                            cmt: "fix above push",
                        },
                    ]);
                    // TODO: verify that these are ok, I know numbered offset is a-OK,
                    // Location::Register is questionable
                } else if matches!(val, Location::NumberedOffset { .. } | Location::Register(..)) {
                    self.asm_buf.push(Instruction::SizedMov {
                        src: val,
                        dst: Location::FloatReg(ARG_FLOAT_REGS[float_count]),
                        size: ty.size(),
                    });
                } else if val != Location::FloatReg(ARG_FLOAT_REGS[float_count]) {
                    self.asm_buf.push(Instruction::FloatMov {
                        src: val,
                        dst: Location::FloatReg(ARG_FLOAT_REGS[float_count]),
                    });
                }
                // use next float register
                float_count += 1;
            } else if let Ty::Func { .. } = ty {
                self.asm_buf.push(Instruction::SizedMov {
                    src: if let Location::Label(s) = val {
                        // When we are moving a function ptr we refer to it like a value 🤷
                        Location::Label(format!("${}", s))
                    } else {
                        val
                    },
                    dst: Location::Register(ARG_REGS[idx]),
                    size: arg.type_of().size(),
                });
            } else {
                self.asm_buf.push(Instruction::SizedMov {
                    src: val,
                    dst: Location::Register(ARG_REGS[idx]),
                    size: arg.type_of().size(),
                });
            }
        }

        self.asm_buf.push(Instruction::Mov {
            src: if called_with_float { ONE } else { ZERO },
            dst: RAX,
            comment: "set float flag",
        });

        let ident = if matches!(kind, FuncKind::Pointer) {
            if let Some(offset) = self.vars.get(&path.segs[0]).cloned() {
                let reg = self.free_reg_except(Register::RAX);
                self.asm_buf.push(Instruction::Mov {
                    src: offset,
                    dst: Location::Register(reg),
                    comment: "move fn ptr to register",
                });

                format!("*{}", reg)
            } else {
                unreachable!()
            }
        } else if type_args.is_empty() || matches!(kind, FuncKind::Linked | FuncKind::Extern) {
            path.to_string()
        } else {
            // TODO: @name-cleanup
            format!(
                "{}{}",
                path,
                type_args
                    .iter()
                    .map(|t| t.to_string().replace(" ", ""))
                    .collect::<Vec<_>>()
                    .join("0"),
            )
        };

        self.asm_buf.push(Instruction::Call(Location::Label(ident)));

        if pushed_to_align_float_stack {
            self.asm_buf.push(Instruction::Math {
                src: Location::Const { val: Val::Int((16 - (self.total_stack % 16)) as isize) },
                dst: RSP,
                op: BinOp::Add,
                cmt: "stack was larger than 16bits and misaligned",
            });
        }

        for spill in spilled.into_iter().rev() {
            self.asm_buf.push(Instruction::Pop {
                loc: Location::Register(spill),
                size: 8,
                comment: "fixing spilled register",
            });
        }

        if matches!(ret_ty, Ty::Float) {
            self.use_float_reg(FloatRegister::XMM0);
            XMM0
        } else {
            self.use_reg(Register::RAX);
            RAX
        }
    }

    fn push_stack(&mut self, ty: &Ty) {
        match ty {
            Ty::Array { size, ty } => {
                self.asm_buf.push(Instruction::Math {
                    src: Location::Const { val: Val::Int((size * ty.size()) as isize) },
                    dst: RSP,
                    op: BinOp::Sub,
                    cmt: "make stack for array",
                });
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
                        largest_variant = curr;
                    }
                }
                self.asm_buf.push(Instruction::Math {
                    src: Location::Const { val: Val::Int(largest_variant as isize) },
                    dst: RSP,
                    op: BinOp::Sub,
                    cmt: "stack for enum",
                });
            }
            Ty::ConstStr(..) | Ty::Ptr(_) | Ty::Int | Ty::Float | Ty::Char | Ty::Bool => {
                self.asm_buf.push(Instruction::Push {
                    loc: Location::Const { val: ty.null_val() },
                    size: 8,
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

    fn alloc_arg(&mut self, count: usize, f_count: &mut usize, name: Ident, ty: &Ty) -> Location {
        let size = match ty {
            // An array is converted to a pointer like thing
            Ty::Array { .. } => 8,
            t => t.size(),
        };

        self.current_stack += size;
        self.total_stack += size;

        let ref_loc = Location::NumberedOffset { offset: self.current_stack, reg: Register::RBP };

        if matches!(ty, Ty::Float) {
            self.asm_buf.extend_from_slice(&[
                Instruction::Push { loc: ZERO, size, comment: "push/mov float arg" },
                Instruction::FloatMov {
                    src: Location::FloatReg(ARG_FLOAT_REGS[*f_count]),
                    dst: ref_loc.clone(),
                },
            ]);
            *f_count += 1;
        } else {
            self.asm_buf.push(Instruction::Push {
                loc: Location::Register(ARG_REGS[count]),
                size,
                comment: "push argument to stack",
            });
        }

        self.vars.insert(name, ref_loc.clone());
        ref_loc
    }

    fn get_pointer(&mut self, expr: &'ctx LValue) -> Option<Location> {
        Some(match expr {
            LValue::Ident { ident, ty: _ } => self.vars.get(ident)?.clone(),
            LValue::Deref { indir, expr, ty } => {
                let loc = self.get_pointer(&**expr)?;
                let register = self.free_reg();
                if loc.is_stack_offset() {
                    self.asm_buf.push(Instruction::Mov {
                        src: loc,
                        dst: Location::Register(register),
                        comment: "deref",
                    });
                    Location::NumberedOffset { offset: 0, reg: register }
                } else {
                    todo!("pretty sure this is an error {:?}", loc)
                }
            }
            LValue::Array { ident, exprs, ty } => {
                let arr = self.vars.get(ident)?.clone();
                let ele_size = if let Ty::Array { ty, .. } = ty {
                    ty.size()
                } else {
                    unreachable!("array type must be array")
                };
                self.index_arr(arr, exprs, ele_size, true)?
            }
            LValue::FieldAccess { lhs, def, rhs, field_idx } => {
                let left_loc = self.vars.get(&lhs.as_ident().unwrap()).cloned();
                let lhs_ty = lhs.type_of();

                if let Some(Location::NumberedOffset { offset, reg }) = left_loc {
                    let accessor = construct_field_offset_lvalue(self, rhs, offset, reg, def)?;
                    if matches!(lhs_ty, Ty::Ptr(..)) {
                        let register = self.free_reg();
                        self.asm_buf.extend_from_slice(&[Instruction::Mov {
                            src: accessor,
                            dst: Location::Register(register),
                            comment: "deref of lvalue",
                        }]);
                        Location::NumberedOffset { offset: 0, reg: register }
                    } else {
                        accessor
                    }
                } else {
                    panic!("have not resolved field access")
                }
            }
        })
    }

    fn build_value(
        &mut self,
        expr: &'ctx Expr,
        assigned: Option<Ident>,
        can_clear: CanClearRegs,
        is_addr: bool,
    ) -> Option<Location> {
        let val = Some(match expr {
            Expr::Ident { ident, ty: _ } => {
                // panic!("{} {:?}", ident, self.vars);
                self.vars.get(ident)?.clone()
            }
            Expr::Deref { indir, expr: ex, ty } => {
                let loc = self.build_value(ex, assigned, can_clear, is_addr)?;
                let register = self.free_reg();
                if loc.is_stack_offset() {
                    self.asm_buf.extend_from_slice(&[Instruction::Mov {
                        src: loc,
                        dst: Location::Register(register),
                        comment: "deref",
                    }]);
                    Location::NumberedOffset { offset: 0, reg: register }
                } else {
                    todo!("pretty sure this is an error {:?}", loc)
                }
            }
            Expr::AddrOf(ex) => {
                let loc = self.build_value(ex, assigned, can_clear, is_addr)?;
                let register = self.free_reg();
                if loc.is_stack_offset() {
                    self.asm_buf.push(Instruction::Load {
                        src: loc,
                        dst: Location::Register(register),
                        size: 8,
                    });
                    Location::Register(register)
                } else {
                    loc
                }
            }
            Expr::Array { ident, exprs, ty } => {
                let arr = self.vars.get(ident)?.clone();
                let ele_size = if let Ty::Array { ty, .. } = ty { ty.size() } else { ty.size() };

                // TODO: my thinking, proven out by `stuff/asmgen/array/sem.cm` is that any array
                // arg is treated as an address (since it is already)
                self.index_arr(
                    arr,
                    exprs,
                    ele_size,
                    is_addr && !self.current_fn_params.contains(ident),
                )?
            }
            Expr::Urnary { op, expr, ty } => {
                let val = self.build_value(expr, None, can_clear, is_addr)?;

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
                // For `is_addr` we want it to be true when we have a local array and false if the
                // array comes from an arguemnt.
                let mut lloc = self.build_value(lhs, None, CanClearRegs::No, !matches!(&**lhs,
                    Expr::Array { .. } | Expr::Ident { ty: Ty::Array{..}, ..} if self.current_fn_params.contains(&lhs.as_ident())
                ))?;
                let mut rloc = self.build_value(rhs, None, CanClearRegs::No, !matches!(&**rhs,
                    Expr::Array { .. } | Expr::Ident { ty: Ty::Array{..}, ..} if self.current_fn_params.contains(&rhs.as_ident())
                ))?;

                // For operations where ordering doesn't matter we move consts around
                // if it matters we load it into a register.
                let swapped = self.order_operands(&mut lloc, &mut rloc, op, ty);

                if matches!(ty, Ty::Float) {
                    let register = self.free_float_reg();
                    let mut pushed = false;
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
                        pushed = true;
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

                    self.asm_buf.extend_from_slice(&[Instruction::from_binop_float(
                        lfloatloc,
                        rfloatloc.clone(),
                        op,
                    )]);

                    // Keep stack aligned to 16 bits
                    if pushed {
                        self.asm_buf.push(Instruction::Math {
                            src: Location::Const { val: Val::Int(8) },
                            dst: RSP,
                            op: BinOp::Add,
                            cmt: "bin op stack align this is probably wrong",
                        });
                    }

                    self.clear_float_regs_except(Some(&rfloatloc), can_clear);

                    rfloatloc
                } else {
                    // println!("{:?} {:?} {} {}", lhs, rloc, is_addr, swapped);
                    // TODO: remove this is hacky
                    if matches!(
                        &**lhs,
                        Expr::Array { .. } | Expr::Ident { ty: Ty::Array { .. }, .. }
                    ) {
                        let loc = if swapped { &mut rloc } else { &mut lloc };
                        match loc {
                            Location::Register(reg) => {
                                let tmp = self.free_reg();
                                self.asm_buf.push(Instruction::Mov {
                                    // Move the value on the right hand side of the `= here`
                                    src: Location::NumberedOffset { offset: 0, reg: *reg },
                                    // to the left hand side of `here =`
                                    dst: Location::Register(tmp),
                                    comment: "mov array index for binop",
                                });
                                *loc = Location::Register(tmp);
                            }
                            _ => {
                                println!("left {:?} {:?} {:?}", lhs, rhs, lloc)
                            }
                        };
                    }
                    if matches!(
                        &**rhs,
                        Expr::Array { .. } | Expr::Ident { ty: Ty::Array { .. }, .. }
                    ) {
                        let mut loc = if swapped { &mut lloc } else { &mut rloc };
                        match loc {
                            Location::Register(reg) => {
                                let tmp = self.free_reg();
                                self.asm_buf.push(Instruction::Mov {
                                    // Move the value on the right hand side of the `= here`
                                    src: Location::NumberedOffset { offset: 0, reg: *reg },
                                    // to the left hand side of `here =`
                                    dst: Location::Register(tmp),
                                    comment: "mov array index for binop",
                                });
                                *loc = Location::Register(tmp);
                            }
                            _ => {
                                println!("right {:?} {:?} {:?}", lhs, rhs, lloc)
                            }
                        };
                    }

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
                            if matches!(rloc, RAX) {
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
                            if matches!(lloc, RAX) {
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
                            if matches!(lloc, RDX) {
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
                            if matches!(rloc, RDX) {
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
                    if spilled_rax && !matches!(rloc, RAX) {
                        self.asm_buf.push(Instruction::Pop {
                            loc: RAX,
                            size: 8,
                            comment: "move back to rax",
                        });
                    }
                    if spilled_rdx && !matches!(lloc, RDX) {
                        self.asm_buf.push(Instruction::Pop {
                            loc: RDX,
                            size: 8,
                            comment: "move back to rdx",
                        });
                    }

                    // TODO: is it ok for any mem_ref ??
                    if let Location::Register(rreg) = rloc {
                        self.clear_regs_except(Some(&Location::Register(rreg)), can_clear);
                        Location::Register(rreg)
                    } else {
                        let store_reg = self.free_reg();
                        self.asm_buf.extend_from_slice(&[Instruction::Mov {
                            src: rloc,
                            dst: Location::Register(store_reg),
                            comment: "",
                        }]);
                        self.clear_regs_except(Some(&Location::Register(store_reg)), can_clear);
                        Location::Register(store_reg)
                    }
                }
            }
            Expr::Parens(ex) => self.build_value(ex, assigned, can_clear, is_addr)?,
            Expr::Call { path, args, type_args, def } => {
                let mut spilled = false;
                if !matches!(def.ret, Ty::Void | Ty::Float)
                    && self.used_regs.contains(&Register::RAX)
                {
                    spilled = true;
                    self.asm_buf.push(Instruction::Push {
                        loc: RAX,
                        size: 8,
                        comment: "had to spill rax for call",
                    });
                }

                // Generate the argument passing and calling the label or ptr
                let ret_loc =
                    self.gen_call_expr(path, def.kind, &def.ret, args, type_args, can_clear);

                if spilled {
                    let reg = self.free_reg();
                    self.asm_buf.extend_from_slice(&[
                        Instruction::Mov {
                            src: ret_loc,
                            dst: Location::Register(reg),
                            comment: "move ret val out of rax since we spilled",
                        },
                        Instruction::Pop { loc: RAX, size: 8, comment: "move back to rax" },
                    ]);
                    Location::Register(reg)
                } else {
                    ret_loc
                }
            }
            Expr::TraitMeth { trait_, args, type_args, def } => self.gen_call_expr(
                trait_,
                def.method.kind,
                &def.method.ret,
                args,
                type_args,
                can_clear,
            ),
            Expr::FieldAccess { lhs, rhs, def } => {
                let lval = self.vars.get(&lhs.as_ident()).cloned();
                if let Some(Location::NumberedOffset { offset, reg }) = &lval {
                    let accessor = construct_field_offset(self, rhs, *offset, *reg, def)?;
                    if matches!(lhs.type_of(), Ty::Ptr(..)) {
                        let register = self.free_reg();
                        self.asm_buf.extend_from_slice(&[Instruction::Mov {
                            src: accessor,
                            dst: Location::Register(register),
                            comment: "deref of lvalue",
                        }]);
                        Location::NumberedOffset { offset: 0, reg: register }
                    } else {
                        accessor
                    }
                } else {
                    panic!("have not resolved field access")
                }
            }
            Expr::StructInit { path: _, fields, .. } => {
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
                let lval: Option<Location> = try { self.vars.get(&assigned?)?.clone() };
                if let Some(Location::NumberedOffset { offset, reg }) = &lval {
                    let mut running_offset = *offset;
                    for expr in fields.iter().flat_map(flatten_struct_init) {
                        let mut rval =
                            self.build_value(expr, assigned, can_clear, is_addr).unwrap();

                        let ele_size = expr.type_of().size();

                        if rval.is_stack_offset() {
                            let tmp = self.free_reg();
                            self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                                src: rval.clone(),
                                dst: Location::Register(tmp),
                                size: ele_size,
                            }]);
                            self.used_regs.remove(&tmp);
                            rval = Location::Register(tmp)
                        }

                        self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                            // Move the value on the right hand side of the `= here`
                            src: rval,
                            // to the left hand side of `here =`
                            // The start offset - the current item size + the tag bits which are
                            // first
                            dst: Location::NumberedOffset { offset: running_offset, reg: *reg },
                            size: ele_size,
                        }]);

                        // The first field is the start of the struct so don't go to the next field
                        // until after it's been initialized
                        running_offset -= ele_size;
                    }

                    if let Some(lval @ Location::NumberedOffset { .. }) = lval {
                        lval
                    } else {
                        Location::NumberedOffset { offset: self.current_stack, reg: Register::RBP }
                    }
                } else {
                    // See enum init below
                    todo!("{:?} {:?}", expr, assigned)
                }
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
                    // This was the other branch when it was in the for loop below
                    // else {
                    //     self.asm_buf.extend_from_slice(&[Instruction::Push {
                    //         loc: rval,
                    //         size: ele_size,
                    //         comment: "we are allocating for enum payload",
                    //     }]);
                    // }
                    todo!()
                }

                let (mut running_offset, reg) =
                    if let Some(Location::NumberedOffset { offset, reg }) = lval {
                        (offset, reg)
                    } else {
                        todo!("not sure")
                    };
                for item in items.iter() {
                    let rval = self.build_value(item, None, can_clear, is_addr).unwrap();

                    let ele_size = item.type_of().size();

                    running_offset -= ele_size;
                    self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                        // Move the value on the right hand side of the `= here`
                        src: rval,
                        // to the left hand side of `here =`
                        // The start offset - the current item size + the tag bits which are
                        // first
                        dst: Location::NumberedOffset { offset: running_offset, reg },
                        size: ele_size,
                    }]);
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
                for (idx, item) in items.iter().enumerate() {
                    let rval = self.build_value(item, None, can_clear, is_addr).unwrap();

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
                    let string = s.name();
                    let cleaned = StripEscape::new(string).into_iter().collect();
                    let name = format!(".Sstring_{}", s.token());
                    let x =
                        // TODO: should this be mutable?
                        self.globals.entry(*s).or_insert(Global::Text { name, content: cleaned, mutable: false, });
                    Location::NamedOffset(x.name().to_string())
                }
            },
            Expr::Builtin(Builtin::SizeOf(ty)) => {
                Location::Const { val: Val::Int(ty.size() as isize) }
            }
            Expr::Builtin(..) => unreachable!("should be something else by now"),
        });
        val
    }

    fn gen_statement(&mut self, stmt: &'ctx Stmt) {
        match stmt {
            Stmt::Const(var) => {
                // TODO: deal with initializer
                self.alloc_stack(var.ident, &var.ty);
            }
            Stmt::Assign { lval, rval, is_let } => {
                let lloc = if *is_let {
                    let ident = lval.as_ident().unwrap();
                    self.alloc_stack(ident, lval.type_of())
                } else {
                    self.get_pointer(lval).unwrap()
                };

                let mut rloc =
                    self.build_value(rval, lval.as_ident(), CanClearRegs::Yes, true).unwrap();

                if let Location::Const { .. } = lloc {
                    unreachable!("ICE: assign to a constant {:?}", lloc);
                }

                // TODO: really this should check type, scalar values can be skipped but array,
                // struct, etc access cannot
                if lloc == rloc {
                    return;
                }

                let ty = lval.type_of();
                let size = match ty {
                    Ty::Array { ty, .. } => ty.size(),
                    t => t.size(),
                };

                if matches!(rval.type_of(), Ty::Array { .. }) {
                    // If the left hand side is something other than an array
                    if !matches!(ty, Ty::Array { .. }) {
                        if let Location::Register(reg) = rloc {
                            if lloc.is_stack_offset() {
                                let tmp = self.free_reg();
                                self.asm_buf.extend_from_slice(&[
                                    Instruction::SizedMov {
                                        src: Location::NumberedOffset { offset: 0, reg },
                                        dst: Location::Register(tmp),
                                        size: ty.size(),
                                    },
                                    Instruction::SizedMov {
                                        src: Location::Register(tmp),
                                        dst: lloc,
                                        size: ty.size(),
                                    },
                                ]);
                            } else {
                                self.asm_buf.push(Instruction::SizedMov {
                                    src: Location::NumberedOffset { offset: 0, reg },
                                    dst: lloc,
                                    size: ty.size(),
                                });
                            }
                        } else {
                            todo!()
                        }
                    } else {
                        // @copypaste look at code around bin_cmp stuff
                        // We want to swap values not addrs
                        if !*is_let {
                            self.clear_regs_except(Some(&lloc), CanClearRegs::Yes);
                            let tmp = self.free_reg();
                            let (lreg, rreg) = match (lloc, rloc) {
                                (Location::Register(lreg), Location::Register(rreg)) => {
                                    (lreg, rreg)
                                }
                                _ => todo!(),
                            };
                            self.asm_buf.extend_from_slice(&[
                                Instruction::Mov {
                                    // Move the value on the right hand side of the `= here`
                                    src: Location::NumberedOffset { offset: 0, reg: rreg },
                                    // to the left hand side of `here =`
                                    dst: Location::Register(tmp),
                                    comment: "stuff",
                                },
                                Instruction::Mov {
                                    // Move the value on the right hand side of the `= here`
                                    src: Location::Register(tmp),
                                    // to the left hand side of `here =`
                                    dst: Location::NumberedOffset { offset: 0, reg: lreg },
                                    comment: "stuff",
                                },
                            ]);
                        }
                    }

                    return;
                }

                if matches!(ty, Ty::Float) {
                    if rloc.is_float_reg() {
                        self.clear_float_regs_except(Some(&lloc), CanClearRegs::Yes);
                        self.asm_buf
                            .extend_from_slice(&[Instruction::FloatMov { src: rloc, dst: lloc }]);
                    } else if matches!(rloc, Location::Const { .. }) {
                        let register = self.free_float_reg();

                        self.clear_float_regs_except(Some(&lloc), CanClearRegs::Yes);

                        self.asm_buf.extend_from_slice(&[
                            Instruction::Push { loc: rloc, size: 8, comment: "const to stack" },
                            Instruction::Cvt {
                                src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                                dst: Location::FloatReg(register),
                            },
                            // Move xmm? value to where it is supposed to be
                            Instruction::FloatMov { src: Location::FloatReg(register), dst: lloc },
                            // Ugh stupid printf needs to be 16 bit aligned so we have to do
                            // our book keeping
                            Instruction::Math {
                                src: Location::Const { val: Val::Int(8) },
                                dst: RSP,
                                op: BinOp::Add,
                                cmt: "even out after push",
                            },
                        ]);

                    // Promote any non float register to float
                    // TODO: REMOVE
                    } else if matches!(
                        (&lloc, &rloc),
                        (Location::NumberedOffset { .. }, Location::Register(_))
                    ) {
                        let register = self.free_float_reg();
                        self.clear_float_regs_except(Some(&lloc), CanClearRegs::Yes);
                        self.asm_buf.extend_from_slice(&[
                            Instruction::Push {
                                loc: rloc,
                                size: 8,
                                comment: "FIXME look at this we are promoting an int to float",
                            },
                            Instruction::FloatMov {
                                src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                                dst: Location::FloatReg(register),
                            },
                            Instruction::FloatMov { src: Location::FloatReg(register), dst: lloc },
                            // Ugh stupid printf needs to be 16 bit aligned so we have to do
                            // our book keeping
                            Instruction::Math {
                                src: Location::Const { val: Val::Int(8) },
                                dst: RSP,
                                op: BinOp::Add,
                                cmt: "even out after push in promote",
                            },
                        ]);
                    } else {
                        let register = self.free_float_reg();
                        self.clear_float_regs_except(Some(&lloc), CanClearRegs::Yes);
                        self.asm_buf.extend_from_slice(&[
                            Instruction::FloatMov { src: rloc, dst: Location::FloatReg(register) },
                            Instruction::FloatMov { src: Location::FloatReg(register), dst: lloc },
                        ]);
                    }
                } else if matches!(ty, Ty::ConstStr(..)) {
                    // assert!(
                    //     matches!(rloc, Location::NamedOffset(_)),
                    //     "ICE: right hand term must be string const"
                    // );

                    self.clear_regs_except(Some(&lloc), CanClearRegs::Yes);
                    if lloc.is_stack_offset() {
                        let reg = if let Location::Register(reg) = &rloc {
                            *reg
                        } else {
                            let reg = self.free_reg();
                            self.asm_buf.push(Instruction::Load {
                                src: rloc,
                                dst: Location::Register(reg),
                                size: 8,
                            });
                            reg
                        };
                        self.asm_buf.push(Instruction::Mov {
                            // Move the value on the right hand side of the `= here`
                            src: Location::Register(reg),
                            // to the left hand side of `here =`
                            dst: lloc,
                            comment: "move string addr",
                        });
                    } else if matches!(rloc, Location::Register(_)) {
                        self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                            // Move the value on the right hand side of the `= here`
                            src: rloc,
                            // to the left hand side of `here =`
                            dst: lloc,
                            size,
                        }]);
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
                    self.clear_regs_except(Some(&lloc), CanClearRegs::Yes);

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
                    self.clear_regs_except(None, CanClearRegs::Yes);

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
                    self.clear_regs_except(Some(&lloc), CanClearRegs::Yes);

                    self.asm_buf.extend_from_slice(&[Instruction::SizedMov {
                        // Move the value on the right hand side of the `= here`
                        src: rloc,
                        // to the left hand side of `here =`
                        dst: lloc,
                        size: 8,
                    }]);
                }
            }
            Stmt::Call { expr: CallExpr { path, args, type_args }, def } => {
                // TODO: if we return something check rax is free

                self.gen_call_expr(path, def.kind, &def.ret, args, type_args, CanClearRegs::No);

                self.clear_float_regs_except(Some(&XMM0), CanClearRegs::Yes);
                self.clear_regs_except(Some(&RAX), CanClearRegs::Yes);
            }
            Stmt::TraitMeth { expr: _, def: _ } => todo!(),
            Stmt::If { cond, blk, els } => {
                let cond_val = self.build_value(cond, None, CanClearRegs::Yes, false).unwrap();
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
                if cond.is_const_true() {
                    self.asm_buf.push(Instruction::Jmp(loop_body));
                } else {
                    let cond_val = self.build_value(cond, None, CanClearRegs::Yes, false).unwrap();
                    // Check if true
                    self.asm_buf.push(Instruction::Cmp {
                        src: Location::Const { val: Val::Int(1) },
                        dst: cond_val,
                    });
                    // Jump back to the loop body
                    self.asm_buf.push(Instruction::CondJmp { loc: loop_body, cond: JmpCond::Eq });
                }
            }
            Stmt::Match { expr, arms, ty } => {
                let val = self.build_value(expr, None, CanClearRegs::Yes, false).unwrap();

                let match_merge = format!(".matchmerge{}", self.asm_buf.len());
                let mut jump_stream = vec![];
                for (idx, arm) in arms.iter().enumerate() {
                    jump_stream.push(vec![]);
                    self.gen_match_arm(&arm.pat, &val, None, ty, &mut jump_stream[idx]);
                    // Add the block instructions (from stmts) to the jump stream
                    std::mem::swap(&mut self.asm_buf, &mut jump_stream[idx]);
                    for stmt in &arm.blk.stmts {
                        self.gen_statement(stmt);
                    }
                    self.asm_buf.push(Instruction::Jmp(Location::Label(match_merge.clone())));

                    // Now swap the asm_buf back to where it should be
                    std::mem::swap(&mut self.asm_buf, &mut jump_stream[idx]);
                }

                for inst_buf in jump_stream {
                    self.asm_buf.extend_from_slice(&inst_buf);
                }

                self.asm_buf.push(Instruction::Label(match_merge));
            }
            // TODO: we double the `leave; ret;` instructions for functions that actually return
            Stmt::Ret(expr, ty) => {
                let expr_ty = expr.type_of();

                let val = self.build_value(expr, None, CanClearRegs::Yes, *ty == expr_ty).unwrap();
                if matches!(ty, Ty::Float) {
                    if !matches!(val, XMM0) {
                        if matches!(val, Location::Const { .. }) {
                            self.asm_buf.extend_from_slice(&[
                                Instruction::Push { loc: val, size: 8, comment: "float return" },
                                Instruction::Cvt {
                                    src: Location::NumberedOffset { offset: 0, reg: Register::RSP },
                                    dst: XMM0,
                                },
                            ]);
                        } else if matches!(val, Location::NumberedOffset { .. }) {
                            self.asm_buf.push(Instruction::Cvt { src: val, dst: XMM0 });
                        } else {
                            self.asm_buf.push(
                                // return value is stored in %xmmN
                                Instruction::FloatMov { src: val, dst: XMM0 },
                            );
                        }
                    }
                } else if !matches!(val, RAX) {
                    // println!("{:?}", ty);
                    if matches!(expr_ty, Ty::Array { .. }) {
                        if let Location::Register(reg) = val {
                            self.asm_buf.push(Instruction::SizedMov {
                                src: Location::NumberedOffset { offset: 0, reg },
                                dst: RAX,
                                size: ty.size(),
                            });
                        }
                    } else {
                        self.asm_buf.push(
                            // return value is stored in %rax
                            Instruction::Mov { src: val, dst: RAX, comment: "" },
                        );
                    }
                }
                self.asm_buf.extend_from_slice(&[
                    Instruction::Math {
                        src: Location::Const { val: Val::Int(self.current_stack as isize) },
                        dst: RSP,
                        op: BinOp::Add,
                        cmt: "fix stack size",
                    },
                    Instruction::Pop { loc: RBP, size: 8, comment: "leave function" },
                    Instruction::Ret,
                ]);
            }
            Stmt::Exit => {}
            Stmt::Block(_) => todo!(),
            Stmt::InlineAsm(asm) => {
                for inst in &asm.assembly {
                    let mut asm_str = format!("    {}  ", inst.inst);

                    if let Some(src) = &inst.src {
                        match src {
                            ty::Location::Register(reg) => asm_str.push_str(&reg.to_string()),
                            ty::Location::FloatReg(reg) => asm_str.push_str(&reg.to_string()),
                            ty::Location::NamedOffset(_) => todo!(),
                            ty::Location::Offset { amt, reg } => todo!(),
                            ty::Location::InlineVar(ident) => {
                                asm_str.push_str(&format!("{}", self.vars.get(ident).unwrap(),))
                            }
                            ty::Location::Const(v) => asm_str.push_str(&format!(
                                "{}",
                                Location::Const { val: Val::lower(v.clone()) }
                            )),
                        }
                        asm_str.push_str(",  ");
                        if let Some(src) = &inst.dst {
                            match src {
                                ty::Location::Register(reg) => asm_str.push_str(&reg.to_string()),
                                ty::Location::FloatReg(reg) => asm_str.push_str(&reg.to_string()),
                                ty::Location::NamedOffset(_) => todo!(),
                                ty::Location::Offset { amt, reg } => todo!(),
                                ty::Location::InlineVar(ident) => {
                                    asm_str.push_str(&format!("{}", self.vars.get(ident).unwrap(),))
                                }
                                ty::Location::Const(v) => asm_str.push_str(&format!(
                                    "{}",
                                    Location::Const { val: Val::lower(v.clone()) }
                                )),
                            }
                        }
                    }

                    self.asm_buf.push(Instruction::Meta(asm_str));
                }
            }
            Stmt::Builtin(b) => {
                self.asm_buf.push(Instruction::Meta(format!(" # a builtin was replaced {}", b)));
            }
        }
    }

    fn gen_match_arm(
        &mut self,
        pat: &Pat,
        tag_val: &Location,
        item_idx: Option<usize>,
        ty: &Ty,
        jump_stream: &mut Vec<Instruction>,
    ) {
        match pat {
            Pat::Enum { idx, items, .. } => {
                let var = if let Ty::Enum { def, .. } = ty {
                    &def.variants[*idx]
                } else {
                    panic!("enum match without enum")
                };
                let name = format!(".matcharm{}x{}", self.asm_buf.len(), idx);
                let match_arm = Location::Label(name.clone());

                // cmp enum tag to variant
                self.asm_buf.extend_from_slice(&[
                    Instruction::Cmp {
                        src: Location::Const { val: Val::Int(*idx as isize) },
                        dst: tag_val.clone(),
                    },
                    Instruction::CondJmp { loc: match_arm, cond: JmpCond::Eq },
                ]);

                jump_stream.push(Instruction::Label(name));
                // cmp each non wildcard pattern
                for (idx, item) in items.iter().enumerate() {
                    self.gen_match_arm(item, tag_val, Some(idx), &var.types[idx], jump_stream)
                }
            }
            Pat::Array { size: _, items } => {
                for _item in items {
                    // TODO: recursively add cmp instructions
                }
            }
            Pat::Bind(bind) => match bind {
                Binding::Wild(ident) => {
                    if let Location::NumberedOffset { offset, reg } = tag_val {
                        if let Some(idx) = item_idx {
                            let ref_loc = Location::NumberedOffset {
                                // TODO: don't assume the type is a certain size (8 bytes)
                                // We add 2 to make up for the tag and we assume the type is 8 bytes
                                offset: offset - ((idx + 1) * 8),
                                reg: *reg,
                            };

                            self.vars.insert(*ident, ref_loc);
                        }
                    }
                }
                Binding::Value(val) => {
                    let name = format!(".bindval{}", self.asm_buf.len());
                    let bind_jmp = Location::Label(name.clone());

                    // cmp enum tag to variant
                    self.asm_buf.extend_from_slice(&[
                        Instruction::Cmp {
                            src: Location::Const { val: val.clone() },
                            dst: tag_val.clone(),
                        },
                        Instruction::CondJmp { loc: bind_jmp, cond: JmpCond::Eq },
                    ]);

                    jump_stream.push(Instruction::Label(name));
                }
            },
        }
    }
}

impl<'ast> Visit<'ast> for CodeGen<'ast> {
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
                self.globals.insert(
                    var.ident,
                    Global::Array {
                        name: name.clone(),
                        content: if let Expr::ArrayInit { items, .. } = &var.init {
                            items.iter().map(convert_to_const).collect()
                        } else {
                            unreachable!("non const string value used in constant")
                        },
                        mutable: var.mutable,
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
        self.vars.insert(var.ident, Location::NamedOffset(name));
    }

    // TODO: we double the `leave; ret;` instructions for functions that actually return
    fn visit_func(&mut self, func: &'ast Func) {
        let function_name = func.ident.name().to_string();
        self.vars.insert(func.ident, Location::Label(function_name.clone()));
        if func.stmts.is_empty()
            || matches!(func.kind, FuncKind::Linked | FuncKind::EmptyTrait | FuncKind::Extern)
        {
            return;
        }

        self.asm_buf.extend_from_slice(&[
            Instruction::Meta(format!(
                ".global {name}\n.type {name},@function\n",
                name = func.ident
            )),
            Instruction::Label(function_name),
            Instruction::Push { loc: RBP, size: 8, comment: "" },
            Instruction::Mov { src: RSP, dst: RBP, comment: "" },
        ]);

        self.current_fn_params.clear();
        self.current_fn_params = func.params.iter().map(|p| p.ident).collect();

        // TODO: make this better
        let mut float_count = 0;
        for (i, arg) in func.params.iter().enumerate() {
            let alloca = self.alloc_arg(i, &mut float_count, arg.ident, &arg.ty);
            self.vars.insert(arg.ident, alloca);
        }

        for stmt in &func.stmts {
            self.gen_statement(stmt);
        }

        self.current_stack = 0;
        self.total_stack = 0;
        self.clear_regs_except(None, CanClearRegs::Yes);
        self.clear_float_regs_except(None, CanClearRegs::Yes);

        // HACK: so when other programs run our programs they don't non-zero exit
        if func.ident.name() == "main" {
            self.asm_buf.push(Instruction::SizedMov { src: ZERO, dst: RAX, size: 8 });
        }

        self.asm_buf.extend_from_slice(&[Instruction::Leave, Instruction::Ret]);
    }
}

// TODO: @copypaste this whole thing could be removed if `LValue -> Expr` worked but the lifetimes
// can't match when creating an `Expr` from a `LValue`
fn construct_field_offset_lvalue<'a>(
    gen: &mut CodeGen<'a>,
    rhs: &'a LValue,
    offset: usize,
    reg: Register,
    def: &Struct,
) -> Option<Location> {
    match rhs {
        LValue::Ident { ident, ty } => {
            let mut count = 0;
            for f in &def.fields {
                if f.ident == *ident {
                    return Some(Location::NumberedOffset { offset: offset - count, reg });
                }
                // Do this after so first field is the 0th offset
                count += f.ty.size();
            }

            // type checking missed this field somehow
            None
        }
        LValue::Deref { indir, expr, ty } => {
            todo!("follow the pointer")
        }
        LValue::Array { ident, exprs, ty } => {
            let mut count = 0;
            for f in &def.fields {
                if f.ident == *ident {
                    let arr = Location::NumberedOffset { offset: offset - count, reg };
                    let ele_size =
                        if let Ty::Array { ty, .. } = ty { ty.size() } else { ty.size() };
                    return gen.index_arr(arr, exprs, ele_size, true);
                }
                count += f.ty.size();
            }

            // type checking missed this field somehow
            None
        }
        LValue::FieldAccess { lhs, rhs: inner, def: inner_def, field_idx } => {
            let mut count = 0;
            for f in &def.fields {
                if Some(f.ident) == lhs.as_ident() {
                    return construct_field_offset_lvalue(
                        gen,
                        inner,
                        offset - count,
                        reg,
                        inner_def,
                    );
                }
                count += f.ty.size();
            }

            // type checking missed this field somehow
            None
        }
        _ => unreachable!("not a valid struct field accessor"),
    }
}

fn construct_field_offset<'a>(
    gen: &mut CodeGen<'a>,
    rhs: &'a Expr,
    offset: usize,
    reg: Register,
    def: &Struct,
) -> Option<Location> {
    match rhs {
        Expr::Ident { ident, ty } => {
            let mut count = 0;
            for f in &def.fields {
                // println!("{} = {} - {}", ident, offset, count);
                if f.ident == *ident {
                    return Some(Location::NumberedOffset { offset: offset - count, reg });
                }

                // Do this after so first field is the 0th offset
                count += f.ty.size();
            }

            // type checking missed this field somehow
            None
        }
        Expr::Deref { indir, expr, ty } => {
            todo!("follow the pointer")
        }
        Expr::Array { ident, exprs, ty } => {
            let mut count = 0;
            for f in &def.fields {
                if f.ident == *ident {
                    let arr = Location::NumberedOffset { offset: offset - count, reg };
                    let ele_size =
                        if let Ty::Array { ty, .. } = ty { ty.size() } else { ty.size() };
                    return gen.index_arr(arr, exprs, ele_size, false);
                }
                count += f.ty.size();
            }

            // type checking missed this field somehow
            None
        }
        Expr::FieldAccess { lhs, rhs: inner, def: inner_def } => {
            let mut count = 0;
            for f in &def.fields {
                if f.ident == lhs.as_ident() {
                    return construct_field_offset(gen, inner, offset - count, reg, inner_def);
                }
                count += f.ty.size();
            }

            // type checking missed this field somehow
            None
        }
        Expr::AddrOf(_) => todo!(),
        Expr::Call { path, args, type_args, def } => todo!(),
        _ => unreachable!("not a valid struct field accessor"),
    }
}
