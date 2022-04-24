use std::fmt;

use rustc_hash::FxHashSet as HashSet;

use crate::{
    gen::asm::{CodeGen, ZERO},
    lir::lower::{BinOp, Val},
};

use Register::*;

pub const ARG_REGS: [Register; 6] = [RDI, RSI, RDX, RCX, R8, R9];

lazy_static::lazy_static! { pub static ref USABLE_REGS: HashSet<Register> =
    vec![RAX, RCX, RDX, RBX, RSI, RDI, R8, R9, R10, R11, R12, R13, R14, R15]
        .into_iter()
        .collect();
}

#[rustfmt::skip]
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Register {
    RAX,
    RCX,
    RDX,
    RBX,
    RSP,
    RBP,
    RSI,
    RDI,
    R8, R9, R10, R11, R12, R13, R14, R15,
}

#[rustfmt::skip]
impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RAX => "%rax".fmt(f),
            RCX => "%rcx".fmt(f),
            RDX => "%rdx".fmt(f),
            RBX => "%rbx".fmt(f),
            RSP => "%rsp".fmt(f),
            RBP => "%rbp".fmt(f),
            RSI => "%rsi".fmt(f),
            RDI => "%rdi".fmt(f),
            R8  => "%r8".fmt(f),
            R9  => "%r9".fmt(f),
            R10 => "%r10".fmt(f),
            R11 => "%r11".fmt(f),
            R12 => "%r12".fmt(f),
            R13 => "%r13".fmt(f),
            R14 => "%r14".fmt(f),
            R15 => "%r15".fmt(f),
        }
    }
}

use FloatRegister::*;

pub const ARG_FLOAT_REGS: [FloatRegister; 7] = [XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6];

lazy_static::lazy_static! { pub static ref USABLE_FLOAT_REGS: HashSet<FloatRegister> =
    vec![XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7]
        .into_iter()
        .collect();
}

#[rustfmt::skip]
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FloatRegister {
    XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7,
}

impl fmt::Display for FloatRegister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            XMM0 => "%xmm0".fmt(f),
            XMM1 => "%xmm1".fmt(f),
            XMM2 => "%xmm2".fmt(f),
            XMM3 => "%xmm3".fmt(f),
            XMM4 => "%xmm4".fmt(f),
            XMM5 => "%xmm5".fmt(f),
            XMM6 => "%xmm6".fmt(f),
            XMM7 => "%xmm7".fmt(f),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Location {
    /// Something like this `BYTE PTR [rbp-1]`.
    RegAddr {
        reg: Register,
        offset: usize,
        size: usize,
    },
    /// Plain register.
    Register(Register),
    /// The 128 (or 32 bit) float registers.
    FloatReg(FloatRegister),
    /// Constant, like `10`.
    ///
    /// This always represents a value never a label.
    Const {
        val: Val,
    },
    /// A label to jump or call to.
    Label(String),
    /// A relative location.
    ///
    /// Accessing global variables using rip offset.
    NamedOffset(String),
    NamedOffsetIndex {
        name: String,
        plus: usize,
    },
    NumberedOffset {
        offset: usize,
        reg: Register,
    },
    Indexable {
        end: usize,
        ele_pos: usize,
        reg: Register,
    },
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = f.width().unwrap_or_default();
        match self {
            Location::RegAddr { reg: _, offset: _, size: _ } => todo!(),
            Location::Register(reg) => write!(f, "{:>width$}", reg, width = width),
            Location::FloatReg(reg) => write!(f, "{:>width$}", reg, width = width),
            Location::Const { val } => match val {
                Val::Float(v) => {
                    write!(f, "{:>width$}", format!("${}", (*v as f32).to_bits()), width = width)
                }
                Val::Int(v) => write!(f, "{:>width$}", format!("${}", v), width = width),
                Val::Char(v) => write!(f, "{:>width$}", format!("${}", v), width = width),
                Val::Bool(v) => {
                    write!(f, "{:>width$}", format!("${}", if *v { 1 } else { 0 }), width = width)
                }
                Val::Str(_, v) => write!(f, "{:>width$}", format!("${}", v), width = width),
            },
            Location::Label(label) => write!(f, "{:>width$}", label),
            Location::NamedOffset(label) => {
                write!(f, "{:>width$}", format!("{}(%rip)", label), width = width)
            }
            Location::NamedOffsetIndex { name, plus } => {
                write!(f, "{:>width$}", format!("{}+{}(%rip)", name, plus), width = width)
            }
            Location::NumberedOffset { offset, reg } => write!(
                f,
                "{:>width$}",
                format!(
                    "{}({})",
                    if *offset == 0 { "".to_owned() } else { format!("-{}", offset) },
                    reg
                ),
                width = width
            ),
            Location::Indexable { end, ele_pos, reg } => {
                assert!(ele_pos <= end, "array index is out of bounds");
                write!(
                    f,
                    "{:>width$}",
                    format!(
                        "{}({})",
                        if *ele_pos == 0 { "".to_owned() } else { format!("-{}", ele_pos) },
                        reg,
                    ),
                    width = width
                )
            }
        }
    }
}

impl Location {
    crate fn is_stack_offset(&self) -> bool {
        matches!(self, Self::NumberedOffset { .. } | Self::NamedOffset(..) | Self::Indexable { .. })
    }

    crate fn is_float_reg(&self) -> bool {
        matches!(self, Self::FloatReg(..))
    }
}

#[derive(Clone, Debug)]
pub enum Global {
    Text { name: String, content: String, mutable: bool },
    Int { name: String, content: i64, mutable: bool },
    Char { name: String, content: u8, mutable: bool },
    Array { name: String, content: Vec<Val>, mutable: bool },
}

impl Global {
    crate fn name(&self) -> &str {
        match self {
            Global::Text { name, .. } => name,
            Global::Int { name, .. } => name,
            Global::Char { name, .. } => name,
            Global::Array { name, .. } => name,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum CondFlag {
    Overflow,
    NoOverflow,
    Below,
    Carry,
    NotAboveEq,
    AboveEq,
    NotBelow,
    NotBelowEq,
    NoCarry,
    Eq,
    NotEq,
    Zero,
    Greater,
    GreaterEq,
    Less,
    LessEq,
}

impl ToString for CondFlag {
    fn to_string(&self) -> String {
        match self {
            CondFlag::Overflow => "o".into(),
            CondFlag::NoOverflow => "no".into(),
            CondFlag::Below => "b".into(),
            CondFlag::Carry => "c".into(),
            CondFlag::NotAboveEq => "nae".into(),
            CondFlag::AboveEq => "ae".into(),
            CondFlag::NotBelow => "nb".into(),
            CondFlag::NotBelowEq => "nbe".into(),
            CondFlag::NoCarry => "nc".into(),
            CondFlag::Eq => "e".into(),
            CondFlag::NotEq => "ne".into(),
            CondFlag::Zero => "z".into(),
            CondFlag::Greater => "g".into(),
            CondFlag::GreaterEq => "ge".into(),
            CondFlag::Less => "l".into(),
            CondFlag::LessEq => "le".into(),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum JmpCond {
    NotEq,
    Eq,
    Gt,
    Ge,
    Lt,
    Le,
}

impl ToString for JmpCond {
    fn to_string(&self) -> String {
        match self {
            JmpCond::NotEq => "ne".into(),
            JmpCond::Eq => "e".into(),
            JmpCond::Gt => "g".into(),
            JmpCond::Ge => "ge".into(),
            JmpCond::Lt => "l".into(),
            JmpCond::Le => "le".into(),
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum Instruction {
    /// Start a new block with the given label.
    Label(String),
    /// Instruction metadata, used for function prologue.
    Meta(String),
    /// Push `Location` to the stack.
    Push {
        loc: Location,
        size: usize,
        comment: &'static str,
    },
    /// Pop `Location` to the stack.
    Pop {
        loc: Location,
        size: usize,
        comment: &'static str,
    },
    /// Add space to the stack (alloca).
    ///
    /// This is a `subq amt, reg` instruction.
    Alloca {
        amount: i64,
        reg: Register,
    },
    /// Jump to the address saving stack info.
    ///
    /// the `Location` is most often a label but can be an address.
    Call(Location),
    /// Jump to the specified `Location`.
    Jmp(Location),
    /// Conditionally jump to the specified `Location`.
    CondJmp {
        loc: Location,
        cond: JmpCond,
    },
    /// Clean up stack before returning from a call.
    Leave,
    /// Return from a call.
    Ret,
    /// Move source to destination.
    Mov {
        src: Location,
        dst: Location,
        comment: &'static str,
    },
    /// Conditionally move source to destination.
    ///
    /// N.B. The `src` arg needs to be a non const. So far we use named rip offset.
    CondMov {
        src: Location,
        dst: Location,
        cond: CondFlag,
    },
    /// Move source float to destination float.
    FloatMov {
        src: Location,
        dst: Location,
    },
    SizedMov {
        src: Location,
        dst: Location,
        size: usize,
    },
    /// Load from the address `src` to `dst`.
    Load {
        src: Location,
        dst: Location,
        size: usize,
    },
    /// Using `src` to operate on `dst`, leaving the result in `dst`.
    Math {
        /// The left hand side of a binary operation.
        src: Location,
        /// The register that holds the value, unless div.
        dst: Location,
        /// The binary operation to apply, except division.
        op: BinOp,
        cmt: &'static str,
    },
    /// Add source to destination.
    FloatMath {
        /// The left hand side of a binary operation.
        src: Location,
        /// The register that holds the value, unless div.
        dst: Location,
        /// The binary operation to apply, except division.
        op: BinOp,
    },
    /// A `idiv` instruction, since they are funky just special case it.
    Idiv(Location),
    /// Sign extend `rax` into `rdx` `rdx:rax`.
    Extend,
    /// Convert single precision float to double for printf.
    Cvt {
        src: Location,
        dst: Location,
    },
    /// Compare `src` to `dst`.
    Cmp {
        src: Location,
        dst: Location,
    },
}

impl Instruction {
    /// The `rhs` is where the value will end up for most operations.
    crate fn from_binop(lhs: Location, rhs: Location, op: &BinOp) -> Vec<Self> {
        assert!(!op.is_cmp());
        vec![Instruction::Math { src: lhs, dst: rhs, op: op.clone(), cmt: "from binary op" }]
    }

    /// The `rhs` is where the value will end up for most operations.
    crate fn from_binop_cmp(
        mut lhs: Location,
        mut rhs: Location,
        op: &BinOp,
        cond_reg: Location,
        ctxt: &mut CodeGen,
    ) -> Vec<Self> {
        // So `1 > 2` becomes this `cmp $2, $1` then everything works so we must swap lhs which
        // would be $1 with rhs which is $2
        if !matches!(lhs, Location::Const { .. }) {
            std::mem::swap(&mut lhs, &mut rhs);
        }

        let mut inst = if rhs.is_stack_offset() {
            let tmp = ctxt.free_reg();
            let x = vec![Instruction::SizedMov { src: rhs, dst: Location::Register(tmp), size: 8 }];
            rhs = Location::Register(tmp);
            x
        } else {
            vec![]
        };

        let op_instructions = |cond| {
            vec![
                Instruction::Mov {
                    src: ZERO,
                    dst: cond_reg.clone(),
                    comment: "binary compare move zero",
                },
                Instruction::Cmp { src: lhs, dst: rhs },
                Instruction::CondMov {
                    src: Location::NamedOffset(".bool_test".into()),
                    dst: cond_reg,
                    cond,
                },
            ]
        };

        match op {
            BinOp::Lt => {
                inst.extend_from_slice(&op_instructions(CondFlag::Less));
            }
            BinOp::Le => {
                inst.extend_from_slice(&op_instructions(CondFlag::LessEq));
            }
            BinOp::Ge => {
                inst.extend_from_slice(&op_instructions(CondFlag::GreaterEq));
            }
            BinOp::Gt => {
                inst.extend_from_slice(&op_instructions(CondFlag::Greater));
            }
            BinOp::Eq => {
                inst.extend_from_slice(&op_instructions(CondFlag::Eq));
            }
            BinOp::Ne => {
                inst.extend_from_slice(&op_instructions(CondFlag::NotEq));
            }
            _ => unreachable!("not a comparison operator"),
        }
        inst
    }

    /// The `rhs` is where the value will end up for most operations.
    crate fn from_binop_float(lhs: Location, rhs: Location, op: &BinOp) -> Self {
        Instruction::FloatMath { src: lhs, dst: rhs, op: op.clone() }
    }
}
