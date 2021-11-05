use std::{collections::HashSet, fmt};

use crate::lir::lower::{BinOp, Ty, Val};

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

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RAX => "rax".fmt(f),
            RCX => "rcx".fmt(f),
            RDX => "rdx".fmt(f),
            RBX => "rbx".fmt(f),
            RSP => "rsp".fmt(f),
            RBP => "rbp".fmt(f),
            RSI => "rsi".fmt(f),
            RDI => "rdi".fmt(f),
            R8 => "r8".fmt(f),
            R9 => "r9".fmt(f),
            R10 => "r10".fmt(f),
            R11 => "r11".fmt(f),
            R12 => "r12".fmt(f),
            R13 => "r13".fmt(f),
            R14 => "r14".fmt(f),
            R15 => "r15".fmt(f),
        }
    }
}

use FloatRegister::*;

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
            XMM0 => "xmm0".fmt(f),
            XMM1 => "xmm1".fmt(f),
            XMM2 => "xmm2".fmt(f),
            XMM3 => "xmm3".fmt(f),
            XMM4 => "xmm4".fmt(f),
            XMM5 => "xmm5".fmt(f),
            XMM6 => "xmm6".fmt(f),
            XMM7 => "xmm7".fmt(f),
        }
    }
}

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
        match self {
            Location::RegAddr { reg, offset, size } => todo!(),
            Location::Register(reg) => write!(f, "%{}", reg),
            Location::FloatReg(reg) => write!(f, "%{}", reg),
            Location::Const { val } => match val {
                Val::Float(v) => write!(f, "${}", (*v as f32).to_bits()),
                Val::Int(v) => write!(f, "${}", v),
                Val::Char(v) => write!(f, "${}", v),
                Val::Bool(v) => write!(f, "${}", if *v { 1 } else { 0 }),
                Val::Str(v) => write!(f, "${}", v),
            },
            Location::Label(label) => label.fmt(f),
            Location::NamedOffset(label) => write!(f, "{}(%rip)", label),
            Location::NumberedOffset { offset, reg } => write!(
                f,
                "{}(%{})",
                if *offset == 0 { "".to_owned() } else { format!("-{}", offset) },
                reg
            ),
            Location::Indexable { end, ele_pos, reg } => {
                assert!(ele_pos <= end, "array index is out of bounds");

                write!(
                    f,
                    "{}(%{})",
                    if *ele_pos == 0 { "".to_owned() } else { format!("-{}", ele_pos) },
                    reg
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
    Text { name: String, content: String },
    Int { name: String, content: i64 },
}

#[derive(Clone, Debug)]
pub enum CondFlag {
    Overflow,
    NoOverflow,
    Below,
    Carry,
    NotAboveEq,
    AboveEq,
    NotBelow,
    NoCarry,
    Eq,
    Zero,
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
            CondFlag::NoCarry => "nc".into(),
            CondFlag::Eq => "e".into(),
            CondFlag::Zero => "z".into(),
        }
    }
}

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
    /// Clean up stack before returning from a call.
    Leave,
    /// Return from a call.
    Ret,
    /// Move source to destination.
    Mov {
        src: Location,
        dst: Location,
    },
    /// Conditionally move source to destination.
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
    /// Add source to destination.
    Math {
        /// The left hand side of a binary operation.
        src: Location,
        /// The register that holds the value, unless div.
        dst: Location,
        /// The binary operation to apply, except division.
        op: BinOp,
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
    pub fn from_binop(lhs: Location, rhs: Location, op: &BinOp) -> Self {
        Instruction::Math { src: lhs, dst: rhs, op: op.clone() }
    }

    /// The `rhs` is where the value will end up for most operations.
    pub fn from_binop_float(lhs: Location, rhs: Location, op: &BinOp) -> Self {
        Instruction::FloatMath { src: lhs, dst: rhs, op: op.clone() }
    }
}
