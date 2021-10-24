use std::fmt;

use crate::lir::lower::BinOp;

use Register::*;
pub const ARG_REGS: [Register; 6] = [RDI, RSI, RDX, RCX, R8, R9];

#[rustfmt::skip]
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Copy, Debug)]
pub enum Register {
    RAX,
    RCX,
    RDX,
    RBX,
    RSP,
    RBP,
    RSI,
    RDI,
    R8, R9, R10, R11, R12, R13, R14, R15, R16,
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
            R16 => "r16".fmt(f),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Location {
    /// Something like this `BYTE PTR [rbp-1]`.
    RegAddr {
        reg: Register,
        offset: usize,
        size: usize,
    },
    /// Plain register.
    Register(Register),
    /// Constant, like `10`.
    ///
    /// This always represents a value never a label.
    Const(String),
    /// A label to jump or call to.
    Label(String),
    /// A relative location.
    ///
    /// Accessing global variables using rip offset.
    NamedOffset(String),
    NumberedOffset(i64),
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Location::RegAddr { reg, offset, size } => todo!(),
            Location::Register(reg) => write!(f, "%{}", reg),
            Location::Const(c) => write!(f, "${}", c),
            Location::Label(label) => label.fmt(f),
            Location::NamedOffset(label) => write!(f, "{}(%rip)", label),
            Location::NumberedOffset(count) => write!(f, "-{}(%rbp)", count),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Global {
    Text { name: String, content: String },
    Int { name: String, content: i64 },
}

#[derive(Clone, Debug)]
pub enum Instruction {
    /// Start a new block with the given label.
    Label(String),
    /// Instruction metadata, used for function prologue.
    Meta(String),
    /// Push `Location` to the stack.
    Push(Location),
    /// Add space to the stack (alloca).
    ///
    /// This is a `subq amt, reg` instruction.
    Alloca { amount: i64, reg: Register },
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
    Mov { src: Location, dst: Location },
    /// Load from the address `src` to `dst`.
    Load { src: Location, dst: Location },
    /// Add source to destination.
    Math {
        /// The left hand side of a binary operation.
        src: Location,
        /// The register that holds the value, unless div.
        dst: Location,
        /// The binary operation to apply, except division.
        op: BinOp,
    },
}

impl Instruction {
    /// The `rhs` is where the value will end up for most operations.
    pub fn from_binop(lhs: Location, rhs: Location, op: &BinOp) -> Self {
        Instruction::Math { src: lhs, dst: rhs, op: op.clone() }
    }
}
