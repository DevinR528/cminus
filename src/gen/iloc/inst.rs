use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashSet},
    fmt,
    hash::{self, Hash},
    mem::discriminant,
    str::FromStr,
    usize,
};

use crate::{
    ast::parse::symbol::Ident,
    lir::lower::{BinOp, UnOp},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    BinOp(BinOp, Reg, Reg),
    UnOp(UnOp, Reg),
    Test(BinOp, Reg),
    Call(Ident),
    ImmCall(Ident),
    ArrayInit(u64),
    StructInit(u64),
    Load(Reg),
    ImmLoad(Ident),
    Store(Reg),
    CvtInt(Reg),
    CvtFloat(Reg),
    Malloc(Reg),
    Realloc(Reg, Reg),
    Free(Reg),
    FramePointer,
}

#[derive(Clone, Debug)]
pub enum Val {
    Integer(isize),
    UInteger(u32),
    Float(f64),
    Location(String),
    String(String),
    Null,
}

impl Hash for Val {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        discriminant(self).hash(state);
        match self {
            Self::Integer(int) => int.hash(state),
            Self::UInteger(int) => int.hash(state),
            Self::Float(float) => float.to_bits().hash(state),
            Self::Location(s) => s.hash(state),
            Self::String(s) => s.hash(state),
            Self::Null => {}
        }
    }
}
impl PartialEq for Val {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Integer(a), Self::Integer(b)) => a.eq(b),
            (Self::UInteger(a), Self::UInteger(b)) => a.eq(b),
            (Self::Float(a), Self::Float(b)) => a.to_bits().eq(&b.to_bits()),
            (Self::Location(a), Self::Location(b)) => a.eq(b),
            _ => false,
        }
    }
}
impl Eq for Val {}
impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::Integer(int) => int.fmt(f),
            Val::UInteger(int) => int.fmt(f),
            Val::Float(flt) => flt.fmt(f),
            Val::Location(loc) => loc.fmt(f),
            Val::String(s) => s.fmt(f),
            Val::Null => write!(f, "null"),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub enum Reg {
    Var(usize),
    Phi(usize, usize),
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reg::Var(num) | Reg::Phi(num, ..) => write!(f, "%vr{}", num),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Loc(pub String);

impl fmt::Display for Loc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[rustfmt::skip]
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub enum Instruction {
    // Integer arithmetic operations
    /// %r => %r `i2i`
    I2I { src: Reg, dst: Reg },
    /// %r + %r => %r `add`
    Add { src_a: Reg, src_b: Reg, dst: Reg },
    /// %r - %r => %r `sub`
    Sub { src_a: Reg, src_b: Reg, dst: Reg },
    /// %r * %r => %r `mult`
    Mult { src_a: Reg, src_b: Reg, dst: Reg },
    /// %r / %r => %r `div`
    Div { src_a: Reg, src_b: Reg, dst: Reg },
    /// %r << %r => %r `lshift`
    LShift { src_a: Reg, src_b: Reg, dst: Reg },
    /// %r >> %r => %r `rshift`
    RShift { src_a: Reg, src_b: Reg, dst: Reg },
    /// %r % %r => %r `mod`
    Mod { src_a: Reg, src_b: Reg, dst: Reg },
    /// %r && %r => %r `and`
    And { src_a: Reg, src_b: Reg, dst: Reg },
    /// %r || %r => %r `or`
    Or { src_a: Reg, src_b: Reg, dst: Reg },
    /// !%r => %r `not`
    Not { src: Reg, dst: Reg },

    // Immediate integer operations
    /// %r + c => %r `addI`
    ImmAdd { src: Reg, konst: Val, dst: Reg },
    /// %r - c => %r `subI`
    ImmSub { src: Reg, konst: Val, dst: Reg },
    /// %r * c => %r `multI`
    ImmMult { src: Reg, konst: Val, dst: Reg },
    /// %r << c => %r `lshiftI`
    ImmLShift { src: Reg, konst: Val, dst: Reg },
    /// %r >> c => %r `rshftI`
    ImmRShift { src: Reg, konst: Val, dst: Reg },

    // Integer memory operations
    /// c => %r `loadI`
    ImmLoad { src: Val, dst: Reg },
    /// %r => %r `load`
    Load { src: Reg, dst: Reg },
    /// (%r + c) => %r `loadAI`
    LoadAddImm { src: Reg, add: Val, dst: Reg },
    /// (%r + %r) => %r `loadAO`
    LoadAdd { src: Reg, add: Reg, dst: Reg },
    /// %r => %r `store`
    Store { src: Reg, dst: Reg },
    /// %r => (%r + c) `storeAI`
    StoreAddImm { src: Reg, add: Val, dst: Reg },
    /// %r => (%r + %r) `storeAO`
    StoreAdd { src: Reg, add: Reg, dst: Reg },

    // Comparison operations
    /// cmp_Lt %r, %r => %r
    CmpLT { a: Reg, b: Reg, dst: Reg },
    CmpLE { a: Reg, b: Reg, dst: Reg },
    CmpGT { a: Reg, b: Reg, dst: Reg },
    CmpGE { a: Reg, b: Reg, dst: Reg },
    CmpEQ { a: Reg, b: Reg, dst: Reg },
    CmpNE { a: Reg, b: Reg, dst: Reg },
    Comp { a: Reg, b: Reg, dst: Reg },
    TestEQ { test: Reg, dst: Reg },
    TestNE { test: Reg, dst: Reg },
    TestGT { test: Reg, dst: Reg },
    TestGE { test: Reg, dst: Reg },
    TestLT { test: Reg, dst: Reg },
    TestLE { test: Reg, dst: Reg },

    // Branches
    /// jump to lable `jumpI`
    ImmJump(Loc),
    /// jump %r `jump`
    Jump(Reg),
    /// Call instruction, includes arguments.
    /// `call name %r, %r
    Call { name: String, args: Vec<Reg> },
    /// Call instruction, includes arguments and return register.
    /// `call name %r, %r => %r
    ImmCall { name: String, args: Vec<Reg>, ret: Reg },
    /// Call instruction, includes arguments and return register.
    /// `call name %r, %r => %r
    ImmRCall { reg: Reg, args: Vec<Reg>, ret: Reg },
    /// `ret`
    Ret,
    /// Return a value in a register.
    /// `iret %r`
    ImmRet(Reg),
    /// cbr %r -> label `cbr` conditional break if tree
    CbrT { cond: Reg, loc: Loc },
    /// cbrne %r -> label `cbrne` conditional break if false
    CbrF { cond: Reg, loc: Loc },
    CbrLT { a: Reg, b: Reg, loc: Loc },
    CbrLE { a: Reg, b: Reg, loc: Loc },
    CbrGT { a: Reg, b: Reg, loc: Loc },
    CbrGE { a: Reg, b: Reg, loc: Loc },
    CbrEQ { a: Reg, b: Reg, loc: Loc },
    CbrNE { a: Reg, b: Reg, loc: Loc },

    // Unsigned operations
    CmpuLT { a: Reg, b: Reg, dst: Reg },
    CmpuLE { a: Reg, b: Reg, dst: Reg },
    CmpuGT { a: Reg, b: Reg, dst: Reg },
    CmpuGE { a: Reg, b: Reg, dst: Reg },
    Compu { a: Reg, b: Reg, dst: Reg },
    TestuGT { test: Reg, dst: Reg },
    TestuGE { test: Reg, dst: Reg },
    TestuLT { test: Reg, dst: Reg },
    TestuLE { test: Reg, dst: Reg },
    CbruLT { a: Reg, b: Reg, loc: Loc },
    CbruLE { a: Reg, b: Reg, loc: Loc },
    CbruGT { a: Reg, b: Reg, loc: Loc },
    CbruGE { a: Reg, b: Reg, loc: Loc },
    RShiftu { src_a: Reg, src_b: Reg, dst: Reg },
    ImmRShiftu { src: Reg, konst: Val, dst: Reg },

    // Floating point arithmetic
    /// `f2i`
    F2I { src: Reg, dst: Reg },
    /// `i2f`
    I2F { src: Reg, dst: Reg },
    /// `f2f`
    F2F { src: Reg, dst: Reg },
    /// `fadd`
    FAdd { src_a: Reg, src_b: Reg, dst: Reg },
    /// `fsub`
    FSub { src_a: Reg, src_b: Reg, dst: Reg },
    /// `fmult`
    FMult { src_a: Reg, src_b: Reg, dst: Reg },
    /// `fdiv`
    FDiv { src_a: Reg, src_b: Reg, dst: Reg },
    /// `fcomp`
    FComp { src_a: Reg, src_b: Reg, dst: Reg },
    /// `fload`
    FLoad { src: Reg, dst: Reg },
    /// `floadAI`
    FLoadAddImm { src: Reg, add: Val, dst: Reg },
    /// `floadAO`
    FLoadAdd { src: Reg, add: Reg, dst: Reg },

    // I/O operations
    /// `fread %r` where r is a float target.
    FRead(Reg),
    /// `fread %r` where r is an int target.
    IRead(Reg),
    /// `fread %r` where r is a float source.
    FWrite(Reg),
    /// `fread %r` where r is an integer source.
    IWrite(Reg),
    /// `fread %r` where r is a null terminated string source.
    SWrite(Reg),
    /// `putchar %r` where r is a int but written as ascii.
    PutChar(Reg),

    // Malloc - Free - Realloc
    /// `malloc %r => %r` where arg `r` is an int for the size and return `r` is the
    /// register that hold the address.
    Malloc { size: Reg, dst: Reg },
    /// `free %r` where `r` is the register to be freed, must be valid.
    Free(Reg),
    /// `realloc %r, %r => %r` where first `r` is the old address, second `r` is an int for the size
    /// and `r` is the register that hold the address.
    Realloc { src: Reg, size: Reg, dst: Reg },

    // Stack operations
    /// `push`
    Push(Val),
    /// `pushr`
    PushR(Reg),
    /// `pop`
    Pop,
    // Pseudo operations
    Data,
    Text,
    Frame { name: String, size: usize, params: Vec<Reg> },
    Global { name: String, size: usize, align: usize },
    Array { name: String, size: usize, content: Vec<Val> },
    String { name: String, content: String },
    Float { name: String, content: f64 },

    /// Labeled block.
    Label(String),
}

#[derive(Clone, Debug)]
pub enum Global {
    Array { name: String, content: Vec<Val> },
    Text { name: String, content: String },
    Int { name: String, content: i64 },
    Float { name: String, content: f64 },
    Char { name: String, content: u8 },
}
impl Global {
    crate fn name(&self) -> &str {
        match self {
            Global::Text { name, .. } => name,
            Global::Array { name, .. } => name,
            Global::Int { name, .. } => name,
            Global::Float { name, .. } => name,
            Global::Char { name, .. } => name,
        }
    }
}

impl Instruction {
    pub const fn inst_name(&self) -> &'static str {
        match self {
            Self::I2I { .. } => "i2i",
            Self::Add { .. } => "add",
            Self::Sub { .. } => "sub",
            Self::Mult { .. } => "mult",
            Self::Div { .. } => "div",
            Self::LShift { .. } => "lshift",
            Self::RShift { .. } => "rshift",
            // unsigned
            Self::RShiftu { .. } => "rshiftu",
            Self::Mod { .. } => "mod",
            Self::And { .. } => "and",
            Self::Or { .. } => "or",
            Self::Not { .. } => "not",
            Self::ImmAdd { .. } => "addI",
            Self::ImmSub { .. } => "subI",
            Self::ImmMult { .. } => "multI",
            Self::ImmLShift { .. } => "lshiftI",
            Self::ImmRShift { .. } => "rshiftI",
            // unsigned
            Self::ImmRShiftu { .. } => "rshiftuI",
            Self::ImmLoad { .. } => "loadI",
            Self::Load { .. } => "load",
            Self::LoadAddImm { .. } => "loadAI",
            Self::LoadAdd { .. } => "loadAO",
            Self::Store { .. } => "store",
            Self::StoreAddImm { .. } => "storeAI",
            Self::StoreAdd { .. } => "storeAO",
            Self::CmpLT { .. } => "cmp_LT",
            Self::CmpLE { .. } => "cmp_LE",
            Self::CmpGT { .. } => "cmp_GT",
            Self::CmpGE { .. } => "cmp_GE",
            Self::CmpEQ { .. } => "cmp_EQ",
            Self::CmpNE { .. } => "cmp_NE",
            Self::Comp { .. } => "comp",
            // unsigned
            Self::CmpuLT { .. } => "cmpu_LT",
            Self::CmpuLE { .. } => "cmpu_LE",
            Self::CmpuGT { .. } => "cmpu_GT",
            Self::CmpuGE { .. } => "cmpu_GE",
            Self::Compu { .. } => "compu",

            Self::TestEQ { .. } => "testeq",
            Self::TestNE { .. } => "testne",
            Self::TestGT { .. } => "testgt",
            Self::TestGE { .. } => "testge",
            Self::TestLT { .. } => "testlt",
            Self::TestLE { .. } => "testle",
            // unsigned
            Self::TestuGT { .. } => "testugt",
            Self::TestuGE { .. } => "testuge",
            Self::TestuLT { .. } => "testult",
            Self::TestuLE { .. } => "testule",

            Self::ImmJump(_) => "jumpI",
            Self::Jump(_) => "jump",
            Self::Call { .. } => "call",
            Self::ImmCall { .. } => "icall",
            Self::ImmRCall { .. } => "ircall",
            Self::Ret => "ret",
            Self::ImmRet(_) => "iret",
            Self::CbrT { .. } => "cbr",
            Self::CbrF { .. } => "cbrne",
            Self::CbrLT { .. } => "cbr_LT",
            Self::CbrLE { .. } => "cbr_LE",
            Self::CbrGT { .. } => "cbr_GT",
            Self::CbrGE { .. } => "cbr_GE",
            Self::CbrEQ { .. } => "cbr_EQ",
            Self::CbrNE { .. } => "cbr_NE",
            // unsigned
            Self::CbruLT { .. } => "cbru_LT",
            Self::CbruLE { .. } => "cbru_LE",
            Self::CbruGT { .. } => "cbru_GT",
            Self::CbruGE { .. } => "cbru_GE",

            Self::F2I { .. } => "f2i",
            Self::I2F { .. } => "i2f",
            Self::F2F { .. } => "f2f",
            Self::FAdd { .. } => "fadd",
            Self::FSub { .. } => "fsub",
            Self::FMult { .. } => "fmult",
            Self::FDiv { .. } => "fdiv",
            Self::FComp { .. } => "fcomp",
            Self::FLoad { .. } => "fload",
            Self::FLoadAddImm { .. } => "floadAI",
            Self::FLoadAdd { .. } => "floadAO",
            Self::FRead(_) => "fread",
            Self::IRead(_) => "iread",
            Self::FWrite(_) => "fwrite",
            Self::IWrite(_) => "iwrite",
            Self::SWrite(_) => "swrite",
            Self::PutChar(_) => "putchar",
            Self::Free(_) => "free",
            Self::Malloc { .. } => "malloc",
            Self::Realloc { .. } => "realloc",
            Self::Push(_) => "push",
            Self::PushR(_) => "pushr",
            Self::Pop => "pop",
            Self::Data => "data",
            Self::Text => "text",
            Self::Frame { .. } => "frame",
            Self::Global { .. } => "global",
            Self::Array { .. } => "array",
            Self::String { .. } => "string",
            Self::Float { .. } => "float",
            Self::Label(_) => "label",
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FLoad { src, dst }
            | Self::F2I { src, dst }
            | Self::I2F { src, dst }
            | Self::F2F { src, dst }
            | Self::I2I { src, dst }
            | Self::Not { src, dst }
            | Self::Load { src, dst }
            | Self::Store { src, dst } => {
                write!(f, "    {} {} => {}", self.inst_name(), src, dst)
            }
            Self::Add { src_a, src_b, dst }
            | Self::Sub { src_a, src_b, dst }
            | Self::Mult { src_a, src_b, dst }
            | Self::Div { src_a, src_b, dst }
            | Self::LShift { src_a, src_b, dst }
            | Self::RShift { src_a, src_b, dst }
            // unsigned
            | Self::RShiftu { src_a, src_b, dst }
            | Self::Mod { src_a, src_b, dst }
            | Self::And { src_a, src_b, dst }
            | Self::Or { src_a, src_b, dst } => {
                write!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Self::ImmAdd { src, konst, dst }
            | Self::ImmSub { src, konst, dst }
            | Self::ImmMult { src, konst, dst }
            | Self::ImmLShift { src, konst, dst }
            | Self::ImmRShift { src, konst, dst }
            // unsigned
            | Self::ImmRShiftu { src, konst, dst } => {
                write!(f, "    {} {}, {} => {}", self.inst_name(), src, konst, dst)
            }

            Self::ImmLoad { src, dst } => {
                write!(f, "    {} {} => {}", self.inst_name(), src, dst)
            }
            Self::LoadAddImm { src, add, dst }
            | Self::StoreAddImm { src, add, dst } => {
                write!(f, "    {} {}, {} => {}", self.inst_name(), src, add, dst)
            }
            Self::LoadAdd { src, add, dst } | Self::StoreAdd { src, add, dst } => {
                write!(f, "    {} {}, {} => {}", self.inst_name(), src, add, dst)
            }

            Self::CmpLT { a, b, dst }
            | Self::CmpLE { a, b, dst }
            | Self::CmpGT { a, b, dst }
            | Self::CmpGE { a, b, dst }
            | Self::CmpEQ { a, b, dst }
            | Self::CmpNE { a, b, dst }
            | Self::Comp { a, b, dst }
            // unsigned
            | Self::Compu { a, b, dst }
            | Self::CmpuGE { a, b, dst }
            | Self::CmpuGT { a, b, dst }
            | Self::CmpuLE { a, b, dst }
            | Self::CmpuLT { a, b, dst } => {
                write!(f, "    {} {}, {} => {}", self.inst_name(), a, b, dst)
            }

            Self::TestEQ { test, dst }
            | Self::TestNE { test, dst }
            | Self::TestGT { test, dst }
            | Self::TestGE { test, dst }
            | Self::TestLT { test, dst }
            | Self::TestLE { test, dst }
            // Unsigned
            | Self::TestuGT { test, dst }
            | Self::TestuGE { test, dst }
            | Self::TestuLT { test, dst }
            | Self::TestuLE { test, dst } => {
                write!(f, "    {} {} => {}", self.inst_name(), test, dst)
            }
            Self::ImmJump(label) => write!(f, "    {} -> {}", self.inst_name(), label),
            Self::Jump(reg) => write!(f, "    {} -> {}", self.inst_name(), reg),
            Self::Call { name, args } => write!(
                f,
                "    {} {}{} {}",
                self.inst_name(),
                name,
                if args.is_empty() { "" } else { "," },
                args.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", ")
            ),
            Self::ImmCall { name, args, ret } => write!(
                f,
                "    {} {}, {} => {}",
                self.inst_name(),
                name,
                args.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", "),
                ret
            ),
            Self::ImmRCall { reg, args, ret } => write!(
                f,
                "    {} {}, {} => {}",
                self.inst_name(),
                reg,
                args.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", "),
                ret
            ),
            Self::ImmRet(reg) => write!(f, "    {} {}", self.inst_name(), reg),
            Self::CbrT { cond, loc } | Self::CbrF { cond, loc } => {
                write!(f, "    {} {} -> {}", self.inst_name(), cond, loc)
            }
            Self::CbrLT { a, b, loc }
            | Self::CbrLE { a, b, loc }
            | Self::CbrGT { a, b, loc }
            | Self::CbrGE { a, b, loc }
            | Self::CbrEQ { a, b, loc }
            | Self::CbrNE { a, b, loc }
            // unsigned
            | Self::CbruGE { a, b, loc }
            | Self::CbruGT { a, b, loc }
            | Self::CbruLE { a, b, loc }
            | Self::CbruLT { a, b, loc } => {
                write!(f, "    {} {}, {} -> {}", self.inst_name(), a, b, loc)
            }

            Self::FAdd { src_a, src_b, dst }
            | Self::FSub { src_a, src_b, dst }
            | Self::FMult { src_a, src_b, dst }
            | Self::FDiv { src_a, src_b, dst }
            | Self::FComp { src_a, src_b, dst } => {
                write!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }

            Self::FLoadAddImm { src, add, dst } => {
                write!(f, "    {} {}, {} => {}", self.inst_name(), src, add, dst)
            }
            Self::FLoadAdd { src, add, dst } => {
                write!(f, "    {} {}, {} => {}", self.inst_name(), src, add, dst)
            }
            Self::FRead(reg)
            | Self::IRead(reg)
            | Self::FWrite(reg)
            | Self::IWrite(reg)
            | Self::SWrite(reg)
            | Self::PutChar(reg)
            | Self::Free(reg)
            | Self::PushR(reg) => write!(f, "    {} {}", self.inst_name(), reg),

            Self::Malloc { size, dst } => {
                write!(f, "    {} {} => {}", self.inst_name(), size, dst)
            }
            Self::Realloc { src, size, dst } => {
                write!(f, "    {} {}, {} => {}", self.inst_name(), src, size, dst)
            }

            Self::Push(val) => write!(f, "    {} {}", self.inst_name(), val),
            Self::Frame { name, size, params } => {
                write!(
                    f,
                    ".{} {}, {}{} {}",
                    self.inst_name(),
                    name,
                    size,
                    if params.is_empty() { "" } else { "," },
                    params.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", ")
                )
            }
            Self::Global { name, size, align } => {
                write!(f, "    .{} {}, {}, {}", self.inst_name(), name, size, align)
            }
            Self::Array { name, size, content } => {
                write!(f, "    .{} {}, {}, [", self.inst_name(), name, size)?;
                write!(f, "{}", content.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "))?;
                writeln!(f, "]")
            }
            Self::String { name, content } => {
                write!(f, "    .{} {}, {}", self.inst_name(), name, content)
            }
            Self::Float { name, content } => {
                write!(f, "    .{} {}, {}", self.inst_name(), name, content)
            }
            Self::Label(label) => {
                if label == ".L_main:"
                // Remove the labels that are added as a result of basic block construction
                    || label.chars().take(3).all(|c| c == '.' || c.is_numeric() || c == '_')
                {
                    Ok(())
                } else {
                    write!(f, "{} nop", label)
                }
            }
            Self::Text | Self::Data => write!(f, "    .{}", self.inst_name()),
            _ => write!(f, "    {}", self.inst_name()),
        }
    }
}
