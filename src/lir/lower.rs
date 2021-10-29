use std::{collections::HashMap, fmt, slice::SliceIndex};

use crate::{
    ast::types as ty, error::Error, lir::const_fold::Folder, typeck::TyCheckRes, visit::Visit,
};

#[derive(Clone, Debug)]
pub enum Val {
    Float(f64),
    Int(isize),
    Char(char),
    Bool(bool),
    Str(String),
}

impl Val {
    fn lower(val: ty::Val) -> Self {
        match val {
            ty::Val::Float(v) => Val::Float(v),
            ty::Val::Int(v) => Val::Int(v),
            ty::Val::Char(v) => Val::Char(v),
            ty::Val::Bool(v) => Val::Bool(v),
            ty::Val::Str(v) => Val::Str(v),
        }
    }

    fn type_of(&self) -> Ty {
        match self {
            Val::Float(_) => Ty::Float,
            Val::Int(_) => Ty::Int,
            Val::Char(_) => Ty::Char,
            Val::Bool(_) => Ty::Bool,
            Val::Str(_) => Ty::String,
        }
    }

    crate fn size_of(&self) -> usize {
        match self {
            Val::Float(_) => 8,
            Val::Int(_) => 8,
            Val::Char(_) => 4,
            Val::Bool(_) => 1,
            Val::Str(s) => 8,
        }
    }
}

impl fmt::Display for Val {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Val::Float(v) => v.fmt(f),
            Val::Int(v) => v.fmt(f),
            Val::Char(v) => v.fmt(f),
            Val::Bool(v) => v.fmt(f),
            Val::Str(v) => v.fmt(f),
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

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UnOp {
    Not,
    OnesComp,
}

impl UnOp {
    fn lower(op: ty::UnOp) -> Self {
        match op {
            ty::UnOp::Not => UnOp::Not,
            ty::UnOp::OnesComp => UnOp::OnesComp,
        }
    }
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

impl BinOp {
    fn lower(op: ty::BinOp) -> Self {
        match op {
            ty::BinOp::Mul => BinOp::Mul,
            ty::BinOp::Div => BinOp::Div,
            ty::BinOp::Rem => BinOp::Rem,
            ty::BinOp::Add => BinOp::Add,
            ty::BinOp::Sub => BinOp::Sub,
            ty::BinOp::LeftShift => BinOp::LeftShift,
            ty::BinOp::RightShift => BinOp::RightShift,
            ty::BinOp::Lt => BinOp::Lt,
            ty::BinOp::Le => BinOp::Le,
            ty::BinOp::Ge => BinOp::Ge,
            ty::BinOp::Gt => BinOp::Gt,
            ty::BinOp::Eq => BinOp::Eq,
            ty::BinOp::Ne => BinOp::Ne,
            ty::BinOp::BitAnd => BinOp::BitAnd,
            ty::BinOp::BitXor => BinOp::BitXor,
            ty::BinOp::BitOr => BinOp::BitOr,
            ty::BinOp::And => BinOp::And,
            ty::BinOp::Or => BinOp::Or,
            ty::BinOp::AddAssign => BinOp::AddAssign,
            ty::BinOp::SubAssign => BinOp::SubAssign,
        }
    }

    crate fn as_instruction(&self) -> &'static str {
        match self {
            BinOp::Mul => "imul",
            BinOp::Add => "add",
            BinOp::Sub => "sub",
            BinOp::LeftShift => "shl",
            BinOp::RightShift => "shr",
            BinOp::And => "and",
            BinOp::Or => "or",
            BinOp::BitAnd => "and",
            BinOp::BitXor => "xor",
            BinOp::BitOr => "or",
            _ => unreachable!("handle differently {:?}", self),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldInit {
    pub ident: String,
    pub init: Expr,
    pub ty: Ty,
}

impl FieldInit {
    fn lower(tyctx: &TyCheckRes<'_, '_>, fold: &Folder, def: &ty::Field, f: ty::FieldInit) -> Self {
        FieldInit {
            ident: f.ident,
            init: Expr::lower(tyctx, fold, f.init),
            ty: Ty::lower(tyctx, &def.ty.val),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expr {
    /// Access a named variable `a`.
    Ident { ident: String, ty: Ty },
    /// Remove indirection, follow a pointer to it's pointee.
    Deref { indir: usize, expr: Box<Expr>, ty: Ty },
    /// Add indirection, refer to a variable by it's memory address (pointer).
    AddrOf(Box<Expr>),
    /// Access an array by index `[expr][expr]`.
    ///
    /// Each `exprs` represents an access of a dimension of the array.
    Array { ident: String, exprs: Vec<Expr>, ty: Ty },
    /// A urnary operation `!expr`.
    Urnary { op: UnOp, expr: Box<Expr>, ty: Ty },
    /// A binary operation `1 + 1`.
    Binary { op: BinOp, lhs: Box<Expr>, rhs: Box<Expr>, ty: Ty },
    /// An expression wrapped in parantheses (expr).
    Parens(Box<Expr>),
    /// A function call with possible expression arguments `call(expr)`.
    Call { ident: String, args: Vec<Expr>, type_args: Vec<Ty>, def: Func },
    /// A call to a trait method with possible expression arguments `<<T>::trait>(expr)`.
    TraitMeth { trait_: String, args: Vec<Expr>, type_args: Vec<Ty>, def: Impl },
    /// Access the fields of a struct `expr.expr.expr;`.
    FieldAccess { lhs: Box<Expr>, def: Struct, rhs: Box<Expr>, field_idx: u32 },
    /// An ADT is initialized with field values.
    StructInit { name: String, fields: Vec<FieldInit>, def: Struct },
    /// An ADT is initialized with field values.
    EnumInit { ident: String, variant: String, items: Vec<Expr>, def: Enum },
    /// An array initializer `{0, 1, 2}`
    ArrayInit { items: Vec<Expr>, ty: Ty },
    /// A literal value `1, "hello", true`
    Value(Val),
}

impl Expr {
    fn lower(tyctx: &TyCheckRes<'_, '_>, fold: &Folder, mut ex: ty::Expression) -> Self {
        let ty = Ty::lower(
            tyctx,
            tyctx.expr_ty.get(&ex).expect(&format!("could not find this expression {:?}", ex)),
        );

        let mut lowered = match ex.val {
            ty::Expr::Ident(ident) => Expr::Ident { ident, ty },
            ty::Expr::AddrOf(expr) => Expr::AddrOf(box Expr::lower(tyctx, fold, *expr)),
            ty::Expr::Deref { indir, expr } => {
                Expr::Deref { indir, expr: box Expr::lower(tyctx, fold, *expr), ty }
            }
            ty::Expr::Array { ident, exprs } => Expr::Array {
                ident,
                ty,
                exprs: exprs.into_iter().map(|e| Expr::lower(tyctx, fold, e)).collect(),
            },
            ty::Expr::FieldAccess { lhs, rhs } => {
                let left = Expr::lower(tyctx, fold, *lhs);
                let right = Expr::lower(tyctx, fold, *rhs);

                let def = if let Expr::Ident { ty: Ty::Struct { def, .. }, .. } = &left {
                    def.clone()
                } else {
                    unreachable!("lhs of field access must be struct")
                };
                let field_idx = if let Expr::Ident { ident, .. } | Expr::Array { ident, .. } =
                    &right
                {
                    def.fields
                        .iter()
                        .enumerate()
                        .find_map(|(i, f)| if f.ident == *ident { Some(i as u32) } else { None })
                        .expect("field access of unknown field")
                } else {
                    unreachable!("rhs of field access must be struct field")
                };

                Expr::FieldAccess { lhs: box left, def, rhs: box right, field_idx }
            }
            ty::Expr::Urnary { op, expr } => {
                Expr::Urnary { op: UnOp::lower(op), expr: box Expr::lower(tyctx, fold, *expr), ty }
            }
            ty::Expr::Binary { op, lhs, rhs } => Expr::Binary {
                op: BinOp::lower(op),
                lhs: box Expr::lower(tyctx, fold, *lhs),
                rhs: box Expr::lower(tyctx, fold, *rhs),
                ty,
            },
            ty::Expr::Parens(expr) => Expr::Parens(box Expr::lower(tyctx, fold, *expr)),
            ty::Expr::Call { ident, args, type_args } => {
                let func = tyctx.var_func.name_func.get(&ident).expect("function is defined");
                Expr::Call {
                    ident,
                    args: args.into_iter().map(|a| Expr::lower(tyctx, fold, a)).collect(),
                    type_args: type_args.into_iter().map(|t| Ty::lower(tyctx, &t.val)).collect(),
                    def: Func::lower(tyctx, fold, func),
                }
            }
            ty::Expr::TraitMeth { trait_, args, type_args } => {
                let func = tyctx
                    .trait_solve
                    .impls
                    .get(&trait_)
                    .expect("function is defined")
                    .get(&type_args.iter().map(|t| &t.val).collect::<Vec<_>>())
                    .cloned()
                    .expect("types have impl");
                Expr::TraitMeth {
                    trait_,
                    args: args.into_iter().map(|a| Expr::lower(tyctx, fold, a)).collect(),
                    type_args: type_args.into_iter().map(|t| Ty::lower(tyctx, &t.val)).collect(),
                    def: Impl::lower(tyctx, fold, func),
                }
            }
            ty::Expr::StructInit { name, fields } => {
                let struc = tyctx.name_struct.get(&name).expect("struct is defined");
                Expr::StructInit {
                    name,
                    fields: fields
                        .into_iter()
                        .zip(&struc.fields)
                        .map(|(finit, fdef)| FieldInit::lower(tyctx, fold, fdef, finit))
                        .collect(),
                    def: Struct::lower(tyctx, (*struc).clone()),
                }
            }
            ty::Expr::EnumInit { ident, variant, items } => {
                let enu = tyctx.name_enum.get(&ident).expect("struct is defined");
                Expr::EnumInit {
                    ident,
                    variant,
                    items: items.into_iter().map(|f| Expr::lower(tyctx, fold, f)).collect(),
                    def: Enum::lower(tyctx, (*enu).clone()),
                }
            }
            ty::Expr::ArrayInit { items } => Expr::ArrayInit {
                items: items.into_iter().map(|f| Expr::lower(tyctx, fold, f)).collect(),
                ty,
            },
            ty::Expr::Value(v) => Expr::Value(Val::lower(v.val)),
        };
        // Evaluate any constant expressions, since this is the lowered Expr we don't have to worry
        // about destroying spans or hashes since we gather types for everything
        lowered.const_fold();
        lowered
    }

    crate fn type_of(&self) -> Ty {
        match self {
            Expr::Ident { ident, ty } => ty.clone(),
            Expr::Deref { indir, expr, ty } => ty.clone(),
            Expr::AddrOf(expr) => expr.type_of(),
            Expr::Array { ident, exprs, ty } => ty.clone(),
            Expr::Urnary { op, expr, ty } => ty.clone(),
            Expr::Binary { op, lhs, rhs, ty } => ty.clone(),
            Expr::Parens(expr) => expr.type_of(),
            Expr::Call { ident, args, type_args, def } => def.ret.clone(),
            Expr::TraitMeth { trait_, args, type_args, def } => def.method.ret.clone(),
            Expr::FieldAccess { lhs, def, rhs, field_idx } => {
                def.fields[*field_idx as usize].ty.clone()
            }
            Expr::StructInit { name, fields, def } => {
                Ty::Struct { ident: def.ident.clone(), gen: def.generics.clone(), def: def.clone() }
            }
            Expr::EnumInit { ident, variant, items, def } => {
                Ty::Enum { ident: def.ident.clone(), gen: def.generics.clone(), def: def.clone() }
            }
            Expr::ArrayInit { items, ty } => ty.clone(),
            Expr::Value(v) => v.type_of(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CallExpr {
    pub ident: String,
    pub args: Vec<Expr>,
    pub type_args: Vec<Ty>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraitMethExpr {
    pub trait_: String,
    pub args: Vec<Expr>,
    pub type_args: Vec<Ty>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LValue {
    /// Access a named variable `a`.
    Ident { ident: String, ty: Ty },
    /// Remove indirection, follow a pointer to it's pointee.
    Deref { indir: usize, expr: Box<LValue> },
    /// Access an array by index `[expr][expr]`.
    ///
    /// Each `exprs` represents an access of a dimension of the array.
    Array { ident: String, exprs: Vec<Expr>, ty: Ty },
    /// Access the fields of a struct `expr.expr.expr;`.
    FieldAccess { lhs: Box<LValue>, rhs: Box<LValue> },
}

impl LValue {
    fn lower(tyctx: &TyCheckRes<'_, '_>, fold: &Folder, ex: ty::Expression) -> Self {
        match ex.val {
            ty::Expr::Ident(ident) => {
                let ty = Ty::lower(
                    tyctx,
                    &tyctx.type_of_ident(&ident, ex.span).expect("type checking missed ident"),
                );
                LValue::Ident { ident, ty }
            }
            ty::Expr::Deref { indir, expr } => {
                LValue::Deref { indir, expr: box LValue::lower(tyctx, fold, *expr) }
            }
            ty::Expr::Array { ident, exprs } => {
                let ty = Ty::lower(
                    tyctx,
                    &tyctx.type_of_ident(&ident, ex.span).expect("type checking missed ident"),
                );
                LValue::Array {
                    ident,
                    exprs: exprs.into_iter().map(|expr| Expr::lower(tyctx, fold, expr)).collect(),
                    ty,
                }
            }
            ty::Expr::FieldAccess { lhs, rhs } => LValue::FieldAccess {
                lhs: box LValue::lower(tyctx, fold, *lhs),
                rhs: box LValue::lower(tyctx, fold, *rhs),
            },
            _ => unreachable!("not valid lvalue made it all the way to lowering"),
        }
    }

    crate fn as_ident(&self) -> Option<&str> {
        Some(match self {
            LValue::Ident { ident, ty } => ident,
            LValue::Deref { indir, expr } => expr.as_ident()?,
            LValue::Array { ident, .. } => ident,
            LValue::FieldAccess { lhs, rhs } => lhs.as_ident()?,
        })
    }

    crate fn type_of(&self) -> &Ty {
        match self {
            LValue::Ident { ident, ty } => ty,
            LValue::Deref { indir, expr } => expr.type_of(),
            LValue::Array { ident, exprs, ty } => ty,
            // TODO: do we want the final value this would affect array too
            LValue::FieldAccess { lhs, rhs } => rhs.type_of(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ty {
    /// A generic type parameter `<T>`.
    ///
    /// N.B. This may be used as a type argument but should not be.
    Generic { ident: String, bound: Option<String> },
    /// A static array of `size` containing item of `ty`.
    Array { size: usize, ty: Box<Ty> },
    /// A struct defined by the user.
    ///
    /// The `ident` is the name of the "type" and there are 'gen' generics.
    Struct { ident: String, gen: Vec<Ty>, def: Struct },
    /// An enum defined by the user.
    ///
    /// The `ident` is the name of the "type" and there are 'gen' generics.
    Enum { ident: String, gen: Vec<Ty>, def: Enum },
    /// A pointer to a type.
    ///
    /// This is equivalent to indirection, for each layer of `Ty::Ptr(..)` we have
    /// to follow a reference to get at the value.
    Ptr(Box<Ty>),
    /// This represents the number of times a pointer has been followed.
    ///
    /// The number of dereferences represented as layers.
    Ref(Box<Ty>),
    /// A string of `char`'s.
    ///
    /// `"hello, world"`
    String,
    /// A positive or negative number.
    Int,
    /// An ascii character.
    ///
    /// todo: Could be bound to between 0-255
    Char,
    /// A positive or negative number with a fractional component.
    Float,
    /// A single bit representing true and false.
    Bool,
    /// The empty/never/uninhabited type.
    Void,
}

impl Ty {
    crate fn size(&self) -> usize {
        match self {
            Ty::Array { size, ty } => ty.size() * size,
            Ty::Struct { ident, gen, def } => def.fields.iter().map(|f| f.ty.size()).sum(),
            Ty::Enum { ident, gen, def } => {
                def.variants.iter().map(|v| v.types.iter().map(|t| t.size()).sum::<usize>()).sum()
            }
            Ty::Ptr(_) | Ty::Ref(_) | Ty::String => 8,
            Ty::Int => 8,
            Ty::Char => 4,
            Ty::Float => 8,
            Ty::Bool => 4,
            Ty::Void => 0,
            _ => unreachable!("generic type should be monomorphized"),
        }
    }

    crate fn null_val(&self) -> Val {
        match self {
            Ty::Ptr(_) | Ty::Ref(_) | Ty::String | Ty::Int | Ty::Float => Val::Int(1),
            Ty::Char | Ty::Bool => Val::Char(0 as char),
            _ => unreachable!("generic type should be monomorphized cannot create null value"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Param {
    pub ty: Ty,
    pub ident: String,
}

impl Param {
    fn lower(tyctx: &TyCheckRes<'_, '_>, v: ty::Param) -> Self {
        Param { ident: v.ident, ty: Ty::lower(tyctx, &v.ty.val) }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

impl Block {
    fn lower(tyctx: &TyCheckRes<'_, '_>, fold: &Folder, b: ty::Block) -> Self {
        Block { stmts: b.stmts.into_iter().map(|s| Stmt::lower(tyctx, fold, s)).collect() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Binding {
    Wild(String),
    Value(Val),
}

#[derive(Clone, Debug, PartialEq, Eq)]
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

impl Pat {
    fn lower(tyctx: &TyCheckRes<'_, '_>, fold: &Folder, pat: ty::Pat) -> Self {
        match pat {
            ty::Pat::Enum { ident, variant, items } => Pat::Enum {
                ident,
                variant,
                items: items.into_iter().map(|p| Pat::lower(tyctx, fold, p)).collect(),
            },
            ty::Pat::Array { size, items } => Pat::Array {
                size,
                items: items.into_iter().map(|p| Pat::lower(tyctx, fold, p)).collect(),
            },
            ty::Pat::Bind(b) => Pat::Bind(match b {
                ty::Binding::Wild(w) => Binding::Wild(w),
                ty::Binding::Value(v) => Binding::Value(Val::lower(v.val)),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchArm {
    pub pat: Pat,
    pub blk: Block,
}

impl MatchArm {
    fn lower(tyctx: &TyCheckRes<'_, '_>, fold: &Folder, arm: ty::MatchArm) -> Self {
        MatchArm {
            pat: Pat::lower(tyctx, fold, arm.pat.val),
            blk: Block::lower(tyctx, fold, arm.blk),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Stmt {
    /// Variable declaration `int x;`
    VarDecl(Vec<Var>),
    /// Assignment `lval = rval;`
    Assign { lval: LValue, rval: Expr },
    /// A call statement `call(arg1, arg2)`
    Call { expr: CallExpr, def: Func },
    /// A trait method call `<<T>::trait>(args)`
    TraitMeth { expr: TraitMethExpr, def: Impl },
    /// If statement `if (expr) { stmts }`
    If { cond: Expr, blk: Block, els: Option<Block> },
    /// While loop `while (expr) { stmts }`
    While { cond: Expr, stmt: Box<Stmt> },
    /// A match statement `match expr { variant1 => { stmts }, variant2 => { stmts } }`.
    Match { expr: Expr, arms: Vec<MatchArm> },
    /// Read statment `read(ident)`
    Read(Expr),
    /// Write statement `write(expr)`
    Write { expr: Expr },
    /// Return statement `return expr`
    Ret(Expr, Ty),
    /// Exit statement `exit`.
    ///
    /// A void return.
    Exit,
    /// A block of statements `{ stmts }`
    Block(Block),
}

impl Stmt {
    fn lower(tyctx: &TyCheckRes<'_, '_>, fold: &Folder, s: ty::Statement) -> Self {
        match s.val {
            ty::Stmt::VarDecl(var) => Stmt::VarDecl(
                var.iter()
                    .map(|var| Var {
                        ty: Ty::lower(tyctx, &var.ty.val),
                        ident: var.ident.clone(),
                        is_global: false,
                    })
                    .collect(),
            ),
            ty::Stmt::Assign { lval, rval } => Stmt::Assign {
                lval: LValue::lower(tyctx, fold, lval),
                rval: Expr::lower(tyctx, fold, rval),
            },
            ty::Stmt::Call(ty::Spanned {
                val: ty::Expr::Call { ident, args, type_args }, ..
            }) => {
                let func = tyctx.var_func.name_func.get(&ident).expect("function is defined");
                Stmt::Call {
                    expr: CallExpr {
                        ident,
                        args: args.into_iter().map(|a| Expr::lower(tyctx, fold, a)).collect(),
                        type_args: type_args
                            .into_iter()
                            .map(|a| Ty::lower(tyctx, &a.val))
                            .collect(),
                    },
                    def: Func::lower(tyctx, fold, func),
                }
            }
            ty::Stmt::Call(_) => unreachable!("call statement without call expression"),
            ty::Stmt::TraitMeth(ty::Spanned {
                val: ty::Expr::TraitMeth { trait_, args, type_args },
                ..
            }) => {
                let func = tyctx
                    .trait_solve
                    .impls
                    .get(&trait_)
                    .expect("function is defined")
                    .get(&type_args.iter().map(|t| &t.val).collect::<Vec<_>>())
                    .cloned()
                    .expect("types have impl");
                Stmt::TraitMeth {
                    expr: TraitMethExpr {
                        trait_,
                        args: args.into_iter().map(|a| Expr::lower(tyctx, fold, a)).collect(),
                        type_args: type_args
                            .into_iter()
                            .map(|a| Ty::lower(tyctx, &a.val))
                            .collect(),
                    },
                    def: Impl::lower(tyctx, fold, func),
                }
            }
            ty::Stmt::TraitMeth(_) => {
                unreachable!("trait method call statement without call expression")
            }
            ty::Stmt::If { cond, blk, els } => Stmt::If {
                cond: Expr::lower(tyctx, fold, cond),
                blk: Block::lower(tyctx, fold, blk),
                els: els.map(|e| Block::lower(tyctx, fold, e)),
            },
            ty::Stmt::While { cond, stmt } => Stmt::While {
                cond: Expr::lower(tyctx, fold, cond),
                stmt: box Stmt::lower(tyctx, fold, *stmt),
            },
            ty::Stmt::Match { expr, arms } => Stmt::Match {
                expr: Expr::lower(tyctx, fold, expr),
                arms: arms.into_iter().map(|a| MatchArm::lower(tyctx, fold, a)).collect(),
            },
            ty::Stmt::Read(expr) => Stmt::Read(Expr::lower(tyctx, fold, expr)),
            ty::Stmt::Write { expr } => Stmt::Write { expr: Expr::lower(tyctx, fold, expr) },
            ty::Stmt::Ret(expr) => {
                let expr = Expr::lower(tyctx, fold, expr);
                let ty = expr.type_of();
                Stmt::Ret(expr, ty)
            }
            ty::Stmt::Exit => Stmt::Exit,
            ty::Stmt::Block(ty::Block { stmts, .. }) => Stmt::Block(Block {
                stmts: stmts.into_iter().map(|s| Stmt::lower(tyctx, fold, s)).collect(),
            }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub ident: String,
    pub ty: Ty,
}

impl Field {
    fn lower(tyctx: &TyCheckRes<'_, '_>, v: ty::Field) -> Self {
        Field { ident: v.ident, ty: Ty::lower(tyctx, &v.ty.val) }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Struct {
    pub ident: String,
    pub fields: Vec<Field>,
    pub generics: Vec<Ty>,
}

impl Struct {
    fn lower(tyctx: &TyCheckRes<'_, '_>, s: ty::Struct) -> Self {
        Struct {
            ident: s.ident,
            fields: s.fields.into_iter().map(|v| Field::lower(tyctx, v)).collect(),
            // TODO: any generic needs to be gone by this point
            generics: s.generics.into_iter().map(|t| Ty::lower(tyctx, &t.val)).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Variant {
    /// The name of the variant `some`.
    pub ident: String,
    /// The types contained in the variants "tuple".
    pub types: Vec<Ty>,
}

impl Variant {
    fn lower(tyctx: &TyCheckRes<'_, '_>, v: ty::Variant) -> Self {
        Variant {
            ident: v.ident,
            types: v.types.into_iter().map(|t| Ty::lower(tyctx, &t.val)).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Enum {
    /// The name of the enum `<option>::none`.
    pub ident: String,
    /// The variants of the enum `option::<some(ty, type)>`.
    pub variants: Vec<Variant>,
    pub generics: Vec<Ty>,
}

impl Enum {
    fn lower(tyctx: &TyCheckRes<'_, '_>, e: ty::Enum) -> Self {
        Enum {
            ident: e.ident,
            variants: e.variants.into_iter().map(|v| Variant::lower(tyctx, v)).collect(),
            // TODO: any generic needs to be gone by this point
            generics: e.generics.into_iter().map(|t| Ty::lower(tyctx, &t.val)).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Adt {
    Struct(Struct),
    Enum(Enum),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generic {
    pub ident: String,
    pub bound: (),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Func {
    /// The return type `int name() { stmts }`
    pub ret: Ty,
    /// Name of the function.
    pub ident: String,
    /// The generic parameters listed for a function.
    pub generics: Vec<Ty>,
    /// the type and identifier of each parameter.
    pub params: Vec<Param>,
    /// All the crap the function does.
    pub stmts: Vec<Stmt>,
}

impl Func {
    fn lower(tyctx: &TyCheckRes<'_, '_>, fold: &Folder, func: &ty::Func) -> Self {
        Func {
            ret: Ty::lower(tyctx, &func.ret.val),
            ident: func.ident.clone(),
            params: func.params.iter().map(|p| Param::lower(tyctx, p.clone())).collect(),
            generics: vec![], // TODO
            stmts: func.stmts.iter().map(|s| Stmt::lower(tyctx, fold, s.clone())).collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraitMethod {
    Default(Func),
    NoBody(Func),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trait {
    pub ident: String,
    pub generics: Vec<Ty>,
    pub method: TraitMethod,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Impl {
    pub ident: String,
    pub type_arguments: Vec<Ty>,
    pub method: Func,
}

impl Impl {
    fn lower(tyctx: &TyCheckRes<'_, '_>, fold: &Folder, imp: &ty::Impl) -> Self {
        Impl {
            ident: imp.ident.clone(),
            type_arguments: imp.type_arguments.iter().map(|t| Ty::lower(tyctx, &t.val)).collect(),
            method: Func::lower(tyctx, fold, &imp.method),
        }
    }
}

/// A variable declaration.
///
/// `struct foo x;` or int x[]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Var {
    pub ty: Ty,
    pub ident: String,
    pub is_global: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    Adt(Adt),
    Func(Func),
    Trait(Trait),
    Impl(Impl),
    Var(Var),
}

impl Ty {
    fn lower(tyctx: &TyCheckRes<'_, '_>, ty: &ty::Ty) -> Self {
        match ty {
            ty::Ty::Array { size, ty: t } => {
                Ty::Array { ty: box Ty::lower(tyctx, &t.val), size: *size }
            }
            ty::Ty::Struct { ident, gen } => Ty::Struct {
                ident: ident.clone(),
                gen: gen.iter().map(|t| Ty::lower(tyctx, &t.val)).collect(),
                def: tyctx
                    .name_struct
                    .get(ident)
                    .map(|e| Struct::lower(tyctx, (*e).clone()))
                    .unwrap(),
            },
            ty::Ty::Enum { ident, gen } => Ty::Enum {
                ident: ident.clone(),
                gen: gen.iter().map(|t| Ty::lower(tyctx, &t.val)).collect(),
                def: tyctx.name_enum.get(ident).map(|e| Enum::lower(tyctx, (*e).clone())).unwrap(),
            },
            ty::Ty::Ptr(t) => Ty::Ptr(box Ty::lower(tyctx, &t.val)),
            ty::Ty::Ref(t) => Ty::Ref(box Ty::lower(tyctx, &t.val)),
            ty::Ty::String => Ty::String,
            ty::Ty::Int => Ty::Int,
            ty::Ty::Char => Ty::Char,
            ty::Ty::Float => Ty::Float,
            ty::Ty::Bool => Ty::Bool,
            ty::Ty::Void => Ty::Void,
            ty::Ty::Generic { ident, bound } => {
                Ty::Generic { ident: ident.clone(), bound: bound.clone() }
            }
            ty::Ty::Func { .. } => {
                todo!("pretty sure this is an error")
            }
        }
    }
}

crate fn lower_items(items: &[ty::Declaration], tyctx: TyCheckRes<'_, '_>) -> Vec<Item> {
    let fold = Folder::default();
    let mut lowered = vec![];
    for item in items {
        match &item.val {
            ty::Decl::Adt(adt) => {}
            ty::Decl::Func(func) => {
                lowered.push(Item::Func(Func::lower(&tyctx, &fold, func)));
            }
            ty::Decl::Impl(imp) => todo!(),
            ty::Decl::Var(var) => lowered.push(Item::Var(Var {
                ty: Ty::lower(&tyctx, &var.ty.val),
                ident: var.ident.clone(),
                is_global: true,
            })),
            _ => {}
        }
    }
    lowered
}
