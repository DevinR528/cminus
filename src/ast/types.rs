#[derive(Clone, Debug)]
pub enum Val {
    Float(f64),
    Int(isize),
    Char(char),
    Str(String),
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
    Ident(String),
    Urnary { op: UnOp, expr: Box<Expr> },
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Parens(Box<Expr>),
    Call { ident: String, args: Vec<Expr> },
    Value(Val),
}

#[derive(Clone, Debug)]
pub enum Ty {
    Int,
    Char,
    Float,
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
