use std::{fmt, hash, ops};

crate trait TypeEquality {
    /// If the two types are considered equal.
    fn is_ty_eq(&self, other: &Self) -> bool;
}

crate trait Spany: Sized {
    /// All enums implement `Spanned` to carry span info.
    fn into_spanned<R: Into<Range>>(self, range: R) -> Spanned<Self> {
        Spanned { val: self, span: range.into() }
    }
}

#[derive(Clone, Debug)]
pub enum Val {
    Float(f64),
    Int(isize),
    Char(char),
    Bool(bool),
    Str(String),
}

impl hash::Hash for Val {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        match self {
            Val::Float(f) => format!("float{}", f).hash(state),
            Val::Int(i) => format!("int{}", i).hash(state),
            Val::Char(c) => format!("char{}", c).hash(state),
            Val::Bool(b) => format!("bool{}", b).hash(state),
            Val::Str(s) => format!("str{}", s).hash(state),
        }
    }
}

impl PartialEq for Val {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Val::Float(a), Val::Float(b)) => a.to_string().eq(&b.to_string()),
            (Val::Float(_), _) => false,
            (Val::Int(a), Val::Int(b)) => a.eq(b),
            (Val::Int(_), _) => false,
            (Val::Char(a), Val::Char(b)) => a.eq(b),
            (Val::Char(_), _) => false,
            (Val::Str(a), Val::Str(b)) => a.eq(b),
            (Val::Str(_), _) => false,
            (Val::Bool(a), Val::Bool(b)) => a.eq(b),
            (Val::Bool(_), _) => false,
        }
    }
}

impl Eq for Val {}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.val {
            Val::Float(v) => write!(f, "float {}", v),
            Val::Int(v) => write!(f, "int {}", v),
            Val::Char(v) => write!(f, "char {}", v),
            Val::Bool(v) => write!(f, "bool {}", v),
            Val::Str(v) => write!(f, "string '{}'", v),
        }
    }
}

impl Spany for Val {}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Not,
    OnesComp,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// The `*` operator.
    Mul,
    /// The `/` operator.
    Div,
    /// The `%` operator.
    Rem,
    /// The `+` operator.
    Add,
    /// The `-` operator.
    Sub,

    /// The `<<` left shift operator.
    LeftShift,
    /// The `>>` right shift operator.
    RightShift,

    /// The `<` operator.
    Lt,
    /// The `<=` operator.
    Le,
    /// The `>=` operator.
    Ge,
    /// The `>` operator.
    Gt,

    /// The `==` operator.
    Eq,
    /// The `!=` operator.
    Ne,

    /// The `&` bitwise and operator.
    BitAnd,
    /// The `^` bitwise and operator.
    BitXor,
    /// The `|` bitwise and operator.
    BitOr,

    /// The `&&` operator.
    And,
    /// The `||` operator.
    Or,

    /// The `+=` operator.
    AddAssign,
    /// The `-=` operator.
    SubAssign,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FieldInit {
    pub ident: String,
    pub init: Expression,
    pub span: Range,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    /// Access a named variable `a`.
    Ident(String),
    /// Remove indirection, follow a pointer to it's pointee.
    Deref { indir: usize, expr: Box<Expression> },
    /// Add indirection, refer to a variable by it's memory address (pointer).
    AddrOf(Box<Expression>),
    /// Access an array by index `[expr][expr]`.
    ///
    /// Each `exprs` represents an access of a dimension of the array.
    Array { ident: String, exprs: Vec<Expression> },
    /// A urnary operation `!expr`.
    Urnary { op: UnOp, expr: Box<Expression> },
    /// A binary operation `1 + 1`.
    Binary { op: BinOp, lhs: Box<Expression>, rhs: Box<Expression> },
    /// An expression wrapped in parantheses (expr).
    Parens(Box<Expression>),
    /// A function call with possible expression arguments `call(expr)`.
    Call { ident: String, args: Vec<Expression>, type_args: Vec<Type> },
    /// A call to a trait method with possible expression arguments `<<T>::trait>(expr)`.
    TraitMeth { trait_: String, args: Vec<Expression>, type_args: Vec<Type> },
    /// Access the fields of a struct `expr.expr.expr;`.
    FieldAccess { lhs: Box<Expression>, rhs: Box<Expression> },
    /// An ADT is initialized with field values.
    StructInit { name: String, fields: Vec<FieldInit> },
    /// An ADT is initialized with field values.
    EnumInit { ident: String, variant: String, items: Vec<Expression> },
    /// An array initializer `{0, 1, 2}`
    ArrayInit { items: Vec<Expression> },
    /// A literal value `1, "hello", true`
    Value(Value),
}

impl Spany for Expr {}

impl Expr {
    crate fn deref_count(&self) -> usize {
        if let Self::Deref { indir, .. } = self {
            *indir
        } else {
            0
        }
    }

    crate fn as_ident_string(&self) -> String {
        match self {
            Expr::Ident(id) => id.to_string(),
            Expr::Deref { indir, expr } => expr.val.as_ident_string(),
            Expr::AddrOf(expr) => expr.val.as_ident_string(),
            Expr::Array { ident, .. } => ident.to_string(),
            // TODO: hmm
            Expr::Call { ident, .. } => ident.to_string(),
            // TODO: hmm
            Expr::FieldAccess { lhs, rhs } => {
                let lhs = lhs.val.as_ident_string();
                let start = if lhs.starts_with('*') {
                    let mut x = lhs.replace("*", "");
                    x.push_str("->");
                    x
                } else {
                    format!("{}.", lhs)
                };
                format!("{}{}", start, rhs.val.as_ident_string())
            }
            Expr::StructInit { .. }
            | Expr::EnumInit { .. }
            | Expr::TraitMeth { .. }
            | Expr::Urnary { .. }
            | Expr::Binary { .. }
            | Expr::Parens(..)
            | Expr::ArrayInit { .. }
            | Expr::Value(..) => todo!(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    Generic {
        ident: String,
        bound: Option<String>,
    },
    Array {
        size: usize,
        ty: Box<Type>,
    },
    Struct {
        ident: String,
        gen: Vec<Type>,
    },
    Enum {
        ident: String,
        gen: Vec<Type>,
    },
    Ptr(Box<Type>),
    Ref(Box<Type>),
    String,
    Int,
    Char,
    Float,
    Bool,
    Void,
    /// This type is only used in resolving rank-1 polymorphism.
    Func {
        ident: String,
        ret: Box<Ty>,
        params: Vec<Ty>,
    },
}

impl Ty {
    /// Returns `true` if the type contains a generic parameter.
    crate fn has_generics(&self) -> bool {
        match self {
            Ty::Generic { ident, bound } => true,
            Ty::Array { size, ty } => ty.val.has_generics(),
            Ty::Struct { ident, gen } => !gen.is_empty(),
            Ty::Enum { ident, gen } => !gen.is_empty(),
            Ty::Ptr(ty) => ty.val.has_generics(),
            Ty::Ref(ty) => ty.val.has_generics(),
            Ty::Func { ident, ret, params } => {
                ret.has_generics() | params.iter().any(|t| t.has_generics())
            }
            Ty::String | Ty::Int | Ty::Char | Ty::Float | Ty::Bool | Ty::Void => false,
        }
    }

    crate fn reference(&self, mut deref: usize) -> Option<Self> {
        let mut indirection = Some(self);
        let mut stop = false;
        while !stop {
            match indirection {
                Some(Ty::Ptr(next)) if deref > 0 => {
                    deref -= 1;
                    indirection = Some(&next.val);
                }
                _ => {
                    stop = true;
                }
            }
        }
        indirection.cloned()
    }

    crate fn dereference(&self, mut indirection: usize) -> Self {
        let mut new = self.clone();
        while (indirection > 0) {
            new = Ty::Ref(box new.into_spanned(DUMMY));
            indirection -= 1;
        }
        new
    }

    crate fn resolve(&self) -> Option<Self> {
        let mut new = self.clone();
        let mut deref = 0;
        while let Ty::Ref(t) = new {
            deref += 1;
            new = t.val;
        }
        while (deref > 0) {
            new = match new {
                // peel off indirection
                Ty::Ptr(ty) => ty.val,
                Ty::Array { size, ty } => todo!("first element of array"),
                Ty::String => todo!("char??"),
                ty => return None,
            };
            deref -= 1;
        }
        Some(new)
    }

    crate fn array_dim(&self) -> usize {
        let mut dim = 0;
        let mut new = self;
        while let Ty::Array { ty, .. } = new {
            new = &ty.val;
            dim += 1;
        }
        dim
    }

    crate fn index_dim(&self, mut dim: usize) -> Self {
        let mut new = self.clone();
        while (dim > 0) {
            if let Ty::Array { ty, .. } = new {
                new = ty.val;
                dim -= 1;
            } else {
                break;
            }
        }
        new
    }
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Array { size, ty } => {
                if let Ty::Array { ty: t, size: s } = &ty.val {
                    write!(f, "{}[{}][{}]", t.val, size, s)
                } else {
                    write!(f, "{}[{}]", ty.val, size)
                }
            }
            Ty::Generic { ident, .. } => write!(f, "<{}>", ident),
            Ty::Struct { ident, gen, .. } => write!(
                f,
                "struct {}{}",
                ident,
                if gen.is_empty() {
                    "".to_owned()
                } else {
                    format!(
                        "<{}>",
                        gen.iter().map(|g| g.val.to_string()).collect::<Vec<_>>().join(", ")
                    )
                },
            ),
            Ty::Enum { ident, gen, .. } => write!(
                f,
                "enum {}{}",
                ident,
                if gen.is_empty() {
                    "".to_owned()
                } else {
                    format!(
                        "<{}>",
                        gen.iter().map(|g| g.val.to_string()).collect::<Vec<_>>().join(", ")
                    )
                }
            ),
            Ty::Ptr(t) => write!(f, "&{}", t.val),
            Ty::Ref(t) => write!(f, "*{}", t.val),
            Ty::String => write!(f, "string"),
            Ty::Int => write!(f, "int"),
            Ty::Char => write!(f, "char"),
            Ty::Float => write!(f, "float"),
            Ty::Bool => write!(f, "bool"),
            Ty::Void => write!(f, "void"),
            Ty::Func { ident, ret, params } => write!(
                f,
                "func {}({}) -> {}",
                ident,
                params.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "),
                ret
            ),
        }
    }
}

impl Spany for Ty {}

impl TypeEquality for Ty {
    fn is_ty_eq(&self, other: &Self) -> bool {
        match (self, other) {
            // TODO: does size matter? :)
            (Ty::Array { size: s1, ty: t1 }, Ty::Array { size: s2, ty: t2 }) => {
                s1.eq(s2) && t1.is_ty_eq(t2)
            }
            (Ty::Array { .. }, _) => false,
            // TODO: generic comparison
            (Ty::Struct { ident: n1, .. }, Ty::Struct { ident: n2, .. }) => n1 == n2,
            (Ty::Struct { .. }, _) => false,
            (Ty::Enum { ident: n1, .. }, Ty::Enum { ident: n2, .. }) => n1 == n2,
            (Ty::Enum { .. }, _) => false,
            (Ty::Ptr(t1), Ty::Ptr(t2)) => t1.val.is_ty_eq(&t2.val),
            (Ty::Ptr(_), _) => false,
            (Ty::Ref(t1), Ty::Ref(t2)) => t1.val.is_ty_eq(&t2.val),
            (Ty::Ref(_), _) => false,
            (Ty::String, Ty::String) => true,
            (Ty::String, _) => false,
            (Ty::Int, Ty::Int) => true,
            (Ty::Int, _) => false,
            (Ty::Char, Ty::Char) => true,
            (Ty::Char, _) => false,
            (Ty::Float, Ty::Float) => true,
            (Ty::Float, _) => false,
            (Ty::Bool, Ty::Bool) => true,
            (Ty::Bool, _) => false,
            (Ty::Void, Ty::Void) => true,
            (Ty::Void, _) => false,
            (Ty::Generic { ident: i1, .. }, Ty::Generic { ident: i2, .. }) => i1.eq(i2),
            (Ty::Generic { .. }, _) => false,
            (Ty::Func { .. }, _) => unreachable!("Func type should never be checked"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Param {
    pub ty: Type,
    pub ident: String,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Statement>,
    pub span: Range,
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "}}");
        for stmt in &self.stmts {
            write!(f, "..;")?;
        }
        write!(f, "}}")
    }
}

#[derive(Clone, Debug)]
pub enum Binding {
    Wild(String),
    Value(Value),
}

impl fmt::Display for Binding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Wild(id) => write!(f, "{}", id),
            Self::Value(id) => write!(f, "{}", id),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Pat {
    /// Match an enum variant `option::some(bind)`
    Enum {
        ident: String,
        variant: String,
        items: Vec<Pat>,
    },
    Array {
        size: usize,
        items: Vec<Pat>,
    },
    Bind(Binding),
}

impl Spany for Pat {}

impl fmt::Display for Pat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Enum { ident, variant, items } => write!(
                f,
                "{}::{}{}",
                ident,
                variant,
                if items.is_empty() {
                    "".to_owned()
                } else {
                    format!(
                        "({})",
                        items.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(", ")
                    )
                },
            ),
            Self::Array { size, items } => write!(
                f,
                "[{}]",
                items.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(", ")
            ),
            Self::Bind(b) => write!(f, "{}", b),
        }
    }
}

#[derive(Clone, Debug)]
pub struct MatchArm {
    pub pat: Pattern,
    pub blk: Block,
    pub span: Range,
}

impl fmt::Display for MatchArm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} => {}", self.pat.val, self.blk)
    }
}

#[derive(Clone, Debug)]
pub enum Stmt {
    /// Variable declaration `int x;`
    VarDecl(Vec<Var>),
    /// Assignment `lval = rval;`
    Assign { lval: Expression, rval: Expression },
    /// A call statement `call(arg1, arg2)`
    Call { ident: String, args: Vec<Expression>, type_args: Vec<Type> },
    /// A trait method call `<<T>::trait>(args)`
    TraitMeth(Expression),
    /// If statement `if (expr) { stmts }`
    If { cond: Expression, blk: Block, els: Option<Block> },
    /// While loop `while (expr) { stmts }`
    While { cond: Expression, stmt: Box<Statement> },
    /// A match statement `match expr { variant1 => { stmts }, variant2 => { stmts } }`.
    Match { expr: Expression, arms: Vec<MatchArm> },
    /// Read statment `read(ident)`
    Read(String),
    /// Write statement `write(expr)`
    Write { expr: Expression },
    /// Return statement `return expr`
    Ret(Expression),
    /// Exit statement `exit`.
    ///
    /// A void return.
    Exit,
    /// A block of statements `{ stmts }`
    Block(Block),
}

impl Spany for Stmt {}

#[derive(Clone, Debug)]
pub struct Field {
    pub ident: String,
    pub ty: Type,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub ident: String,
    pub fields: Vec<Field>,
    pub generics: Vec<Type>,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Variant {
    /// The name of the variant `some`.
    pub ident: String,
    /// The types contained in the variants "tuple".
    pub types: Vec<Type>,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Enum {
    /// The name of the enum `<option>::none`.
    pub ident: String,
    /// The variants of the enum `option::<some(type, type)>`.
    pub variants: Vec<Variant>,
    pub generics: Vec<Type>,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub enum Adt {
    Struct(Struct),
    Enum(Enum),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generic {
    pub ident: String,
    pub bound: (),
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Func {
    pub ret: Type,
    pub ident: String,
    pub generics: Vec<Type>,
    pub params: Vec<Param>,
    pub stmts: Vec<Statement>,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub enum TraitMethod {
    Default(Func),
    NoBody(Func),
}

impl TraitMethod {
    crate fn return_ty(&self) -> &Type {
        match self {
            Self::Default(func) | Self::NoBody(func) => &func.ret,
        }
    }

    crate fn function(&self) -> &Func {
        match self {
            Self::Default(func) | Self::NoBody(func) => func,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Trait {
    pub ident: String,
    pub generics: Vec<Type>,
    pub method: TraitMethod,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Impl {
    pub ident: String,
    pub type_arguments: Vec<Type>,
    pub method: Func,
    pub span: Range,
}

/// A variable declaration.
///
/// `struct foo x;` or int x[]
#[derive(Clone, Debug)]
pub struct Var {
    pub ty: Type,
    pub ident: String,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub enum Decl {
    Adt(Adt),
    Func(Func),
    Trait(Trait),
    Impl(Impl),
    Var(Var),
}

impl Spany for Decl {}

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct Spanned<T> {
    pub val: T,
    pub span: Range,
}

impl<T: TypeEquality> TypeEquality for Spanned<T> {
    fn is_ty_eq(&self, other: &Self) -> bool {
        self.val.is_ty_eq(&other.val)
    }
}

impl<T: TypeEquality> TypeEquality for Option<&T> {
    fn is_ty_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Some(a), Some(b)) => a.is_ty_eq(b),
            _ => false,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${:?}@{:?}", self.val, self.span)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Range {
    pub start: usize,
    pub end: usize,
}

impl fmt::Debug for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

impl From<ops::Range<usize>> for Range {
    fn from(other: ops::Range<usize>) -> Self {
        let (start, end) = (other.start, other.end);
        Self { start, end }
    }
}

const fn to_rng(other: ops::Range<usize>) -> Range {
    let (start, end) = (other.start, other.end);
    Range { start, end }
}

pub const DUMMY: Range = to_rng(0..0);

pub type Declaration = Spanned<Decl>;
pub type Statement = Spanned<Stmt>;
pub type Expression = Spanned<Expr>;
pub type Type = Spanned<Ty>;
pub type Value = Spanned<Val>;
pub type Pattern = Spanned<Pat>;
