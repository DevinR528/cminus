use std::{
    cmp::Ordering,
    collections::{BTreeSet, HashSet},
    fmt,
    hash::{self, Hash},
    mem::discriminant,
    str::FromStr,
    usize,
};

#[derive(Clone, Debug)]
pub enum Val {
    Integer(isize),
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
    String { name: String, content: String },
    Float { name: String, content: f64 },

    /// Labeled block.
    Label(String),
}

#[derive(Clone, Debug)]
pub enum Global {
    Text { name: String, content: String, mutable: bool },
    Int { name: String, content: i64, mutable: bool },
    Float { name: String, content: f64, mutable: bool },
    Char { name: String, content: u8, mutable: bool },
}
impl Global {
    crate fn name(&self) -> &str {
        match self {
            Global::Text { name, .. } => name,
            Global::Int { name, .. } => name,
            Global::Float { name, .. } => name,
            Global::Char { name, .. } => name,
        }
    }
}

impl Instruction {
    pub const fn inst_name(&self) -> &'static str {
        match self {
            Instruction::I2I { .. } => "i2i",
            Instruction::Add { .. } => "add",
            Instruction::Sub { .. } => "sub",
            Instruction::Mult { .. } => "mult",
            Instruction::LShift { .. } => "lshift",
            Instruction::RShift { .. } => "rshift",
            Instruction::Mod { .. } => "mod",
            Instruction::And { .. } => "and",
            Instruction::Or { .. } => "or",
            Instruction::Not { .. } => "not",
            Instruction::ImmAdd { .. } => "addI",
            Instruction::ImmSub { .. } => "subI",
            Instruction::ImmMult { .. } => "multI",
            Instruction::ImmLShift { .. } => "lshiftI",
            Instruction::ImmRShift { .. } => "rshiftI",
            Instruction::ImmLoad { .. } => "loadI",
            Instruction::Load { .. } => "load",
            Instruction::LoadAddImm { .. } => "loadAI",
            Instruction::LoadAdd { .. } => "loadAO",
            Instruction::Store { .. } => "store",
            Instruction::StoreAddImm { .. } => "storeAI",
            Instruction::StoreAdd { .. } => "storeAO",
            Instruction::CmpLT { .. } => "cmp_LT",
            Instruction::CmpLE { .. } => "cmp_LE",
            Instruction::CmpGT { .. } => "cmp_GT",
            Instruction::CmpGE { .. } => "cmp_GE",
            Instruction::CmpEQ { .. } => "cmp_EQ",
            Instruction::CmpNE { .. } => "cmp_NE",
            Instruction::Comp { .. } => "comp",
            Instruction::TestEQ { .. } => "testeq",
            Instruction::TestNE { .. } => "testne",
            Instruction::TestGT { .. } => "testgt",
            Instruction::TestGE { .. } => "testge",
            Instruction::TestLT { .. } => "testlt",
            Instruction::TestLE { .. } => "testle",
            Instruction::ImmJump(_) => "jumpI",
            Instruction::Jump(_) => "jump",
            Instruction::Call { .. } => "call",
            Instruction::ImmCall { .. } => "icall",
            Instruction::ImmRCall { .. } => "ircall",
            Instruction::Ret => "ret",
            Instruction::ImmRet(_) => "iret",
            Instruction::CbrT { .. } => "cbr",
            Instruction::CbrF { .. } => "cbrne",
            Instruction::CbrLT { .. } => "cbr_LT",
            Instruction::CbrLE { .. } => "cbr_LE",
            Instruction::CbrGT { .. } => "cbr_GT",
            Instruction::CbrGE { .. } => "cbr_GE",
            Instruction::CbrEQ { .. } => "cbr_EQ",
            Instruction::CbrNE { .. } => "cbr_NE",
            Instruction::F2I { .. } => "f2i",
            Instruction::I2F { .. } => "i2f",
            Instruction::F2F { .. } => "f2f",
            Instruction::FAdd { .. } => "fadd",
            Instruction::FSub { .. } => "fsub",
            Instruction::FMult { .. } => "fmult",
            Instruction::FDiv { .. } => "fdiv",
            Instruction::FComp { .. } => "fcomp",
            Instruction::FLoad { .. } => "fload",
            Instruction::FLoadAddImm { .. } => "floadAI",
            Instruction::FLoadAdd { .. } => "floadAO",
            Instruction::FRead(_) => "fread",
            Instruction::IRead(_) => "iread",
            Instruction::FWrite(_) => "fwrite",
            Instruction::IWrite(_) => "iwrite",
            Instruction::SWrite(_) => "swrite",
            Instruction::Push(_) => "push",
            Instruction::PushR(_) => "pushr",
            Instruction::Pop => "pop",
            Instruction::Data => "data",
            Instruction::Text => "text",
            Instruction::Frame { .. } => "frame",
            Instruction::Global { .. } => "global",
            Instruction::String { .. } => "string",
            Instruction::Float { .. } => "float",
            Instruction::Label(_) => "label",
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::I2I { src, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), src, dst)
            }
            Instruction::Add { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::Sub { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::Mult { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::LShift { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::RShift { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::Mod { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::And { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::Or { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::Not { src, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), src, dst)
            }
            Instruction::ImmAdd { src, konst, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src, konst, dst)
            }
            Instruction::ImmSub { src, konst, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src, konst, dst)
            }
            Instruction::ImmMult { src, konst, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src, konst, dst)
            }
            Instruction::ImmLShift { src, konst, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src, konst, dst)
            }
            Instruction::ImmRShift { src, konst, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src, konst, dst)
            }
            Instruction::ImmLoad { src, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), src, dst)
            }
            Instruction::Load { src, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), src, dst)
            }
            Instruction::LoadAddImm { src, add, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src, add, dst)
            }
            Instruction::LoadAdd { src, add, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src, add, dst)
            }
            Instruction::Store { src, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), src, dst)
            }
            Instruction::StoreAddImm { src, add, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src, add, dst)
            }
            Instruction::StoreAdd { src, add, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src, add, dst)
            }
            Instruction::CmpLT { a, b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), a, b, dst)
            }
            Instruction::CmpLE { a, b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), a, b, dst)
            }
            Instruction::CmpGT { a, b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), a, b, dst)
            }
            Instruction::CmpGE { a, b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), a, b, dst)
            }
            Instruction::CmpEQ { a, b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), a, b, dst)
            }
            Instruction::CmpNE { a, b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), a, b, dst)
            }
            Instruction::Comp { a, b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), a, b, dst)
            }
            Instruction::TestEQ { test, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), test, dst)
            }
            Instruction::TestNE { test, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), test, dst)
            }
            Instruction::TestGT { test, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), test, dst)
            }
            Instruction::TestGE { test, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), test, dst)
            }
            Instruction::TestLT { test, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), test, dst)
            }
            Instruction::TestLE { test, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), test, dst)
            }
            Instruction::ImmJump(label) => writeln!(f, "    {} -> {}", self.inst_name(), label),
            Instruction::Jump(reg) => writeln!(f, "    {} -> {}", self.inst_name(), reg),
            Instruction::Call { name, args } => writeln!(
                f,
                "    {} {}{} {}",
                self.inst_name(),
                name,
                if args.is_empty() { "" } else { "," },
                args.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", ")
            ),
            Instruction::ImmCall { name, args, ret } => writeln!(
                f,
                "    {} {}, {} => {}",
                self.inst_name(),
                name,
                args.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", "),
                ret
            ),
            Instruction::ImmRCall { reg, args, ret } => writeln!(
                f,
                "    {} {}, {} => {}",
                self.inst_name(),
                reg,
                args.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", "),
                ret
            ),
            Instruction::ImmRet(reg) => writeln!(f, "    {} {}", self.inst_name(), reg),
            Instruction::CbrT { cond, loc } => {
                writeln!(f, "    {} {} -> {}", self.inst_name(), cond, loc)
            }
            Instruction::CbrF { cond, loc } => {
                writeln!(f, "    {} {} -> {}", self.inst_name(), cond, loc)
            }
            Instruction::CbrLT { a, b, loc } => {
                writeln!(f, "    {} {}, {} -> {}", self.inst_name(), a, b, loc)
            }
            Instruction::CbrLE { a, b, loc } => {
                writeln!(f, "    {} {}, {} -> {}", self.inst_name(), a, b, loc)
            }
            Instruction::CbrGT { a, b, loc } => {
                writeln!(f, "    {} {}, {} -> {}", self.inst_name(), a, b, loc)
            }
            Instruction::CbrGE { a, b, loc } => {
                writeln!(f, "    {} {}, {} -> {}", self.inst_name(), a, b, loc)
            }
            Instruction::CbrEQ { a, b, loc } => {
                writeln!(f, "    {} {}, {} -> {}", self.inst_name(), a, b, loc)
            }
            Instruction::CbrNE { a, b, loc } => {
                writeln!(f, "    {} {}, {} -> {}", self.inst_name(), a, b, loc)
            }
            Instruction::F2I { src, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), src, dst)
            }
            Instruction::I2F { src, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), src, dst)
            }
            Instruction::F2F { src, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), src, dst)
            }
            Instruction::FAdd { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::FSub { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::FMult { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::FDiv { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::FComp { src_a, src_b, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src_a, src_b, dst)
            }
            Instruction::FLoad { src, dst } => {
                writeln!(f, "    {} {} => {}", self.inst_name(), src, dst)
            }
            Instruction::FLoadAddImm { src, add, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src, add, dst)
            }
            Instruction::FLoadAdd { src, add, dst } => {
                writeln!(f, "    {} {}, {} => {}", self.inst_name(), src, add, dst)
            }
            Instruction::FRead(reg) => writeln!(f, "    {} {}", self.inst_name(), reg),
            Instruction::IRead(reg) => writeln!(f, "    {} {}", self.inst_name(), reg),
            Instruction::FWrite(reg) => writeln!(f, "    {} {}", self.inst_name(), reg),
            Instruction::IWrite(reg) => writeln!(f, "    {} {}", self.inst_name(), reg),
            Instruction::SWrite(reg) => writeln!(f, "    {} {}", self.inst_name(), reg),
            Instruction::Push(val) => writeln!(f, "    {} {}", self.inst_name(), val),
            Instruction::PushR(reg) => writeln!(f, "    {} {}", self.inst_name(), reg),
            Instruction::Frame { name, size, params } => {
                writeln!(
                    f,
                    ".{} {}, {}{} {}",
                    self.inst_name(),
                    name,
                    size,
                    if params.is_empty() { "" } else { "," },
                    params.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", ")
                )
            }
            Instruction::Global { name, size, align } => {
                writeln!(f, "    .{} {}, {}, {}", self.inst_name(), name, size, align)
            }
            Instruction::String { name, content } => {
                writeln!(f, "    .{} {}, {}", self.inst_name(), name, content)
            }
            Instruction::Float { name, content } => {
                writeln!(f, "    .{} {}, {}", self.inst_name(), name, content)
            }
            Instruction::Label(label) => {
                if label == ".L_main:"
                // Remove the labels that are added as a result of basic block construction
                    || label.chars().take(3).all(|c| c == '.' || c.is_numeric() || c == '_')
                {
                    Ok(())
                } else {
                    writeln!(f, "{} nop", label)
                }
            }
            Instruction::Text | Instruction::Data => writeln!(f, "    .{}", self.inst_name()),
            _ => writeln!(f, "    {}", self.inst_name()),
        }
    }
}
