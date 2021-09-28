use std::{fmt, hash, ops};

#[derive(Clone, Debug)]
pub enum Val {
    Float(f64),
    Int(isize),
    Char(char),
    Str(String),
}

impl hash::Hash for Val {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        match self {
            Val::Float(f) => format!("float{}", f).hash(state),
            Val::Int(i) => format!("int{}", i).hash(state),
            Val::Char(c) => format!("char{}", c).hash(state),
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
            Val::Str(v) => write!(f, "string '{}'", v),
        }
    }
}

impl Val {
    crate fn into_spanned(self, span: Range) -> Value {
        Spanned { val: self, span }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Not,
    // Inc,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BinOp {
    /// The `+` operator
    Add,
    /// The `-` operator
    Sub,
    /// The `*` operator
    Mul,
    /// The `/` operator
    Div,
    /// The `%` operator
    Rem,
    /// The `&&` operator
    And,
    /// The `||` operator
    Or,
    /// The `==` operator
    Eq,
    /// The `<` operator
    Lt,
    /// The `<=` operator
    Le,
    /// The `!=` operator
    Ne,
    /// The `>=` operator
    Ge,
    /// The `>` operator
    Gt,
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
    Call { ident: String, args: Vec<Expression> },
    /// Access the fields of a struct `expr.expr.expr;`.
    FieldAccess { lhs: Box<Expression>, rhs: Box<Expression> },
    /// An ADT is initialized with field values.
    StructInit { name: String, fields: Vec<FieldInit> },
    /// An array initializer `{0, 1, 2}`
    ArrayInit { items: Vec<Expression> },
    /// A literal value `1, "hello", true`
    Value(Value),
}

impl Expr {
    crate fn into_spanned<R: Into<Range>>(self, span: R) -> Expression {
        Spanned { val: self, span: span.into() }
    }

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
            Expr::Array { ident, exprs } => ident.to_string(),
            // TODO: hmm
            Expr::Call { ident, args } => ident.to_string(),
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
            | Expr::Urnary { .. }
            | Expr::Binary { .. }
            | Expr::Parens(..)
            | Expr::ArrayInit { .. }
            | Expr::Value(..) => todo!(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    Array { size: usize, ty: Box<Type> },
    Adt(String),
    Ptr(Box<Type>),
    Ref(Box<Type>),
    String,
    Int,
    Char,
    Float,
    Bool,
    Void,
}

impl Ty {
    crate fn into_spanned<R: Into<Range>>(self, span: R) -> Type {
        Spanned { val: self, span: span.into() }
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

    crate fn derfreference(&self, mut indirection: usize) -> Self {
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
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Array { size, ty } => write!(f, "{}[{}]", ty.val, size),
            Ty::Adt(n) => write!(f, "struct {}", n),
            Ty::Ptr(t) => write!(f, "&{}", t.val),
            Ty::Ref(t) => write!(f, "*{}", t.val),
            Ty::String => write!(f, "string"),
            Ty::Int => write!(f, "int"),
            Ty::Char => write!(f, "char"),
            Ty::Float => write!(f, "float"),
            Ty::Bool => write!(f, "bool"),
            Ty::Void => write!(f, "void"),
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

#[derive(Clone, Debug)]
pub enum Stmt {
    /// Variable declaration `int x;`
    VarDecl(Vec<Var>),
    /// Assignment `lval = rval;`
    Assign { deref: usize, lval: Expression, rval: Expression },
    /// A call statement `call(arg1, arg2)`
    Call { ident: String, args: Vec<Expression> },
    /// If statement `if (expr) { stmts }`
    If { cond: Expression, blk: Block, els: Option<Block> },
    /// While loop `while (expr) { stmts }`
    While { cond: Expression, stmt: Box<Statement> },
    /// Read statment `read(ident | "string")`
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

impl Stmt {
    crate fn into_spanned<R: Into<Range>>(self, span: R) -> Statement {
        Spanned { val: self, span: span.into() }
    }
}

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
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Func {
    pub ret: Type,
    pub ident: String,
    pub params: Vec<Param>,
    pub stmts: Vec<Statement>,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Var {
    pub ty: Type,
    pub ident: String,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub enum Decl {
    Adt(Struct),
    Func(Func),
    Var(Var),
}

impl Decl {
    crate fn into_spanned<R: Into<Range>>(self, span: R) -> Declaration {
        Spanned { val: self, span: span.into() }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    pub val: T,
    pub span: Range,
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
