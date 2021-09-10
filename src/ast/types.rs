use std::fmt;

#[derive(Clone, Debug)]
pub enum Val {
    Float(f64),
    Int(isize),
    Char(char),
    Str(String),
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::Float(v) => write!(f, "float {}", v),
            Val::Int(v) => write!(f, "int {}", v),
            Val::Char(v) => write!(f, "char {}", v),
            Val::Str(v) => write!(f, "string '{}'", v),
        }
    }
}

#[derive(Clone, Debug)]
pub enum UnOp {
    Not,
    Inc,
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
pub enum Expr {
    /// Access a named variable `a`.
    Ident(String),
    /// Access an array by index `[expr]`.
    Array { ident: String, expr: Box<Expr> },
    /// A urnary operation `!expr`.
    Urnary { op: UnOp, expr: Box<Expr> },
    /// A binary operation `1 + 1`.
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    /// An expression wrapped in parantheses (expr).
    Parens(Box<Expr>),
    /// A function call with possible expression arguments `call(expr)`.
    Call { ident: String, args: Vec<Expr> },
    /// A literal value `1, "hello", true`
    Value(Val),
}

#[derive(Clone, Debug)]
pub enum Ty {
    Int,
    Char,
    Float,
    Array { size: usize, ty: Box<Ty> },
    Void,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub ty: Ty,
    pub ident: String,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    VarDecl(Vec<Var>),
    Assign { ident: String, expr: Expr },
    ArrayAssign { ident: String, expr: Expr },
    Call { ident: String, args: Vec<Expr> },
    If { cond: Expr, blk: Block, els: Option<Block> },
    While { cond: Expr, stmt: Box<Stmt> },
    Read(String),
    Write { expr: Expr },
    Ret(Expr),
    Exit,
    Block(Block),
}

#[derive(Clone, Debug)]
pub struct Func {
    pub ret: Ty,
    pub ident: String,
    pub params: Vec<Param>,
    pub stmts: Vec<Stmt>,
}

#[derive(Clone, Debug)]
pub struct Var {
    pub ty: Ty,
    pub ident: String,
}

#[derive(Clone, Debug)]
pub enum Decl {
    Func(Func),
    Var(Var),
}
