#[derive(Debug)]
pub enum Val {
    Float(f64),
    Int(isize),
    Char(char),
    Str(String),
}

#[derive(Debug)]
pub enum UnOp {
    Not,
    Inc,
}

#[derive(Debug)]
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

#[derive(Debug)]
pub enum Expr {
    Ident(String),
    Urnary { op: UnOp, expr: Box<Expr> },
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr> },
    Parens(Box<Expr>),
    Call { ident: String, args: Vec<Expr> },
    Value(Val),
}
