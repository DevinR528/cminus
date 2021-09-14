use std::{fmt, hash, ops::Range};

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
    crate fn into_spanned(self, span: Range<usize>) -> Value {
        Spanned { val: self, span }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Not,
    Inc,
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
pub enum Expr {
    /// Access a named variable `a`.
    Ident(String),
    /// Access an array by index `[expr]`.
    Array { ident: String, expr: Box<Expression> },
    /// A urnary operation `!expr`.
    Urnary { op: UnOp, expr: Box<Expression> },
    /// A binary operation `1 + 1`.
    Binary { op: BinOp, lhs: Box<Expression>, rhs: Box<Expression> },
    /// An expression wrapped in parantheses (expr).
    Parens(Box<Expression>),
    /// A function call with possible expression arguments `call(expr)`.
    Call { ident: String, args: Vec<Expression> },
    /// A literal value `1, "hello", true`
    Value(Value),
}

impl Expr {
    crate fn into_spanned(self, span: Range<usize>) -> Expression {
        Spanned { val: self, span }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    Int,
    Char,
    String,
    Float,
    Array { size: usize, ty: Box<Type> },
    Bool,
    Void,
}

impl Ty {
    crate fn into_spanned(self, span: Range<usize>) -> Type {
        Spanned { val: self, span }
    }
}

#[derive(Clone, Debug)]
pub struct Param {
    pub ty: Type,
    pub ident: String,
    pub span: Range<usize>,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Statement>,
    pub span: Range<usize>,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    VarDecl(Vec<Var>),
    Assign { ident: String, expr: Expression },
    ArrayAssign { ident: String, expr: Expression },
    Call { ident: String, args: Vec<Expression> },
    If { cond: Expression, blk: Block, els: Option<Block> },
    While { cond: Expression, stmt: Box<Statement> },
    Read(String),
    Write { expr: Expression },
    Ret(Expression),
    Exit,
    Block(Block),
}

impl Stmt {
    crate fn into_spanned(self, span: Range<usize>) -> Statement {
        Spanned { val: self, span }
    }
}

#[derive(Clone, Debug)]
pub struct Func {
    pub ret: Type,
    pub ident: String,
    pub params: Vec<Param>,
    pub stmts: Vec<Statement>,
    pub span: Range<usize>,
}

#[derive(Clone, Debug)]
pub struct Var {
    pub ty: Type,
    pub ident: String,
    pub span: Range<usize>,
}

#[derive(Clone, Debug)]
pub enum Decl {
    Func(Func),
    Var(Var),
}

impl Decl {
    crate fn into_spanned(self, span: Range<usize>) -> Declaration {
        Spanned { val: self, span }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    pub val: T,
    pub span: Range<usize>,
}

impl<T: fmt::Debug> fmt::Debug for Spanned<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "${:?}@{:?}", self.val, self.span)
    }
}

pub const DUMMY: Range<usize> = 0..0;

pub type Declaration = Spanned<Decl>;
pub type Statement = Spanned<Stmt>;
pub type Expression = Spanned<Expr>;
pub type Type = Spanned<Ty>;
pub type Value = Spanned<Val>;
