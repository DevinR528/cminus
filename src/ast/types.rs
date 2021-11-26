use std::{
    fmt,
    hash::{self, Hash, Hasher},
    ops,
};

use crate::{
    ast::parse::symbol::Ident,
    data_struc::{rawptr::RawPtr, rawvec::RawVec},
    error::Error,
    typeck::{check::fold_ty, TyCheckRes},
};

crate trait TypeEquality<T = Self> {
    /// If the two types are considered equal.
    fn is_ty_eq(&self, other: &T) -> bool;
}

crate trait Spany: Sized {
    /// All enums implement `Spanned` to carry span info.
    fn into_spanned(self, span: Range) -> Spanned<Self> {
        Spanned { val: self, span }
    }
}

#[derive(Clone, Debug)]
pub enum Val {
    Float(f64),
    Int(isize),
    Char(char),
    Bool(bool),
    Str(Ident),
}

impl Val {
    crate fn to_type(&self) -> Ty {
        match self {
            Val::Float(_) => Ty::Float,
            Val::Int(_) => Ty::Int,
            Val::Char(_) => Ty::Char,
            Val::Bool(_) => Ty::Bool,
            Val::Str(s) => Ty::ConstStr(s.name().len()),
        }
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    pub ident: Ident,
    pub init: Expression,
    pub span: Range,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Expr {
    /// Access a named variable `a`.
    Ident(Ident),
    /// Remove indirection, follow a pointer to it's pointee.
    Deref { indir: usize, expr: Box<Expression> },
    /// Add indirection, refer to a variable by it's memory address (pointer).
    AddrOf(Box<Expression>),
    /// Access an array by index `[expr][expr]`.
    ///
    /// Each `exprs` represents an access of a dimension of the array.
    Array { ident: Ident, exprs: Vec<Expression> },
    /// A urnary operation `!expr`.
    Urnary { op: UnOp, expr: Box<Expression> },
    /// A binary operation `1 + 1`.
    Binary { op: BinOp, lhs: Box<Expression>, rhs: Box<Expression> },
    /// An expression wrapped in parantheses (expr).
    Parens(Box<Expression>),
    /// A function call with possible expression arguments `call(expr)`.
    Call { path: Path, args: Vec<Expression>, type_args: RawVec<Type> },
    /// A call to a trait method with possible expression arguments `<<T>::trait>(expr)`.
    TraitMeth { trait_: Path, args: Vec<Expression>, type_args: Vec<Type> },
    /// Access the fields of a struct `expr.expr.expr;`.
    FieldAccess { lhs: Box<Expression>, rhs: Box<Expression> },
    /// An ADT is initialized with field values.
    StructInit { path: Path, fields: Vec<FieldInit> },
    /// An ADT is initialized with field values.
    EnumInit { path: Path, variant: Ident, items: Vec<Expression> },
    /// An array initializer `{0, 1, 2}`
    ArrayInit { items: Vec<Expression> },
    /// A literal value `1, "hello", true`
    Value(Value),
}

impl Spany for Expr {}

impl Expr {
    // crate fn deref_count(&self) -> usize {
    //     if let Self::Deref { indir, .. } = self {
    //         *indir
    //     } else {
    //         0
    //     }
    // }

    crate fn as_ident(&self) -> Ident {
        match self {
            Expr::Ident(id) => *id,
            Expr::Deref { expr, .. } => expr.val.as_ident(),
            Expr::AddrOf(expr) => expr.val.as_ident(),
            Expr::Array { ident, .. } => *ident,
            // TODO: hmm
            Expr::Call { path: ident, .. } => ident.segs[0],
            // TODO: hmm
            Expr::FieldAccess { .. }
            | Expr::StructInit { .. }
            | Expr::EnumInit { .. }
            | Expr::TraitMeth { .. }
            | Expr::Urnary { .. }
            | Expr::Binary { .. }
            | Expr::Parens(..)
            | Expr::ArrayInit { .. }
            | Expr::Value(..) => todo!(),
        }
    }

    crate fn type_of(&self) -> Option<Ty> {
        match self {
            Expr::Array { ident, exprs } => exprs[0].val.type_of(),
            Expr::Urnary { op, expr } => expr.val.type_of(),
            Expr::Binary { op, lhs, rhs } => {
                let lty = lhs.val.type_of();
                let rty = rhs.val.type_of();
                if lty.as_ref().is_ty_eq(&rty.as_ref()) {
                    lty
                } else {
                    None
                }
            }
            Expr::Parens(ex) => ex.val.type_of(),
            Expr::FieldAccess { lhs, rhs } => rhs.val.type_of(),
            Expr::ArrayInit { items } => Some(Ty::Array {
                size: items.len(),
                ty: box items[0].val.type_of().unwrap().into_spanned(DUMMY),
            }),
            Expr::Value(v) => Some(v.val.to_type()),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Eq)]
pub struct Path {
    pub segs: Vec<Ident>,
    pub span: Range,
}

impl Path {
    // This will use a `DUMMY` span. DO NOT USE until after type checking.
    crate fn single(seg: Ident) -> Self {
        Self { segs: vec![seg], span: DUMMY }
    }

    /// Return the file local identifier for this declaration.
    ///
    /// ## Panics
    /// If the `Path` is empty.
    ///
    /// If we are in `lib.cm` and have `struct foo` our path is `lib::foo` and the local ident is
    /// `foo`.
    crate fn local_ident(&self) -> Ident {
        self.segs.last().copied().unwrap()
    }
}

impl Hash for Path {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.segs.hash(state);
    }
}
impl PartialEq for Path {
    fn eq(&self, other: &Self) -> bool {
        self.segs.eq(&other.segs)
    }
}
impl fmt::Display for Path {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.segs.iter().map(|id| id.name()).collect::<Vec<_>>().join("::").fmt(f)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Ty {
    /// A generic type parameter `<T>`.
    ///
    /// N.B. This may be used as a type argument but should not be.
    Generic { ident: Ident, bound: Option<Path> },
    /// A static array of `size` containing item of `ty`.
    Array { size: usize, ty: Box<Type> },
    /// A struct defined by the user.
    ///
    /// The `ident` is the name of the "type" and there are 'gen' generics.
    Struct { ident: Ident, gen: Vec<Type> },
    /// An enum defined by the user.
    ///
    /// The `ident` is the name of the "type" and there are 'gen' generics.
    Enum { ident: Ident, gen: Vec<Type> },
    /// Any kind of path, this could be a type name or an import path
    Path(Path),
    /// A pointer to a type. From either taking the address of `&x` or passed as argument `*x`.
    ///
    /// This is equivalent to indirection, for each layer of `Ty::Ptr(..)` we have
    /// to follow a reference to get at the value.
    Ptr(Box<Type>),
    /// This represents the number of times a pointer has been followed.
    ///
    /// The number of dereferences represented as layers. A deref is `*` on a type that is an
    /// address i.e. `&x`
    Ref(Box<Type>),
    /// A constant array of `char`'s, this has a compile time known size.
    ///
    /// `"hello, world"`
    ConstStr(usize),
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
    /// This type is only used in resolving rank-1 polymorphism.
    Func { ident: Ident, ret: Box<Ty>, params: Vec<Ty> },
}

impl Ty {
    /// Returns `Ty::Generic { .. }` as a string `T`.
    ///
    /// ## Panics
    /// if `Self` is not a `Ty::Generic`.
    crate fn generic(&self) -> Ident {
        if let Self::Generic { ident, .. } = self {
            *ident
        } else {
            panic!("type was not a Generic {:?}", self)
        }
    }

    /// Returns iterator of all generic parameters [`T`, `U`, ..].
    crate fn generics(&self) -> Vec<&Ident> {
        match self {
            Ty::Generic { ident, .. } => vec![ident],
            Ty::Array { ty, .. } => ty.val.generics(),
            Ty::Struct { gen, .. } => gen.iter().map(|t| t.val.generics()).flatten().collect(),
            Ty::Enum { gen, .. } => gen.iter().map(|t| t.val.generics()).flatten().collect(),
            Ty::Ptr(ty) => ty.val.generics(),
            Ty::Ref(ty) => ty.val.generics(),
            Ty::Func { ret, params, .. } => {
                params.iter().map(|p| p.generics()).flatten().chain(ret.generics()).collect()
            }
            Ty::Path(p) => todo!("{}", p),
            Ty::ConstStr(..) | Ty::Int | Ty::Char | Ty::Float | Ty::Bool | Ty::Void => {
                vec![]
            }
        }
    }

    /// Returns `true` if the type contains a generic parameter.
    crate fn has_generics(&self) -> bool {
        match self {
            Ty::Generic { .. } => true,
            Ty::Array { ty, .. } => ty.val.has_generics(),
            Ty::Struct { gen, .. } => !gen.is_empty(),
            Ty::Enum { gen, .. } => !gen.is_empty(),
            Ty::Ptr(ty) => ty.val.has_generics(),
            Ty::Ref(ty) => ty.val.has_generics(),
            Ty::Func { ret, params, .. } => {
                ret.has_generics() | params.iter().any(|t| t.has_generics())
            }
            Ty::Path(_) => false,
            Ty::ConstStr(..) | Ty::Int | Ty::Char | Ty::Float | Ty::Bool | Ty::Void => false,
        }
    }

    /// Substitute a generic parameter with a concrete type.
    crate fn subst_generic(&mut self, generic: Ident, subs: &Ty) {
        match self {
            t @ Ty::Generic { .. } if generic == t.generic() => {
                *t = subs.clone();
            }
            Ty::Array { size: _, ty } => ty.val.subst_generic(generic, subs),
            Ty::Struct { ident: _, gen } => {
                for t in gen {
                    t.val.subst_generic(generic, subs)
                }
            }
            Ty::Enum { ident: _, gen } => {
                for t in gen {
                    t.val.subst_generic(generic, subs)
                }
            }
            Ty::Ptr(ty) => ty.val.subst_generic(generic, subs),
            Ty::Ref(ty) => ty.val.subst_generic(generic, subs),
            Ty::Func { ident: _, ret: _, params: _ } => {
                todo!()
            }
            _ => {}
        }
    }

    // crate fn reference(&self, mut deref: usize) -> Option<Self> {
    //     let mut indirection = Some(self);
    //     let mut stop = false;
    //     while !stop {
    //         match indirection {
    //             Some(Ty::Ptr(next)) if deref > 0 => {
    //                 deref -= 1;
    //                 indirection = Some(&next.val);
    //             }
    //             _ => {
    //                 stop = true;
    //             }
    //         }
    //     }
    //     indirection.cloned()
    // }

    crate fn dereference(&self, mut indirection: usize) -> Self {
        let mut new = self.clone();
        while indirection > 0 {
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
        while deref > 0 {
            new = match new {
                // peel off indirection
                Ty::Ptr(ty) => ty.val,
                Ty::Array { size: _, ty: _ } => todo!("first element of array"),
                Ty::ConstStr(..) => todo!("char??"),
                _ty => return None,
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

    crate fn index_dim(
        &self,
        tcxt: &TyCheckRes<'_, '_>,
        exprs: &[Expression],
        span: Range,
    ) -> Option<Self> {
        let mut new = self.clone();
        for expr in exprs {
            if let Ty::Array { ty, ref size } = &new {
                if let Expr::Value(Spanned { val: Val::Int(i), .. }) = &expr.val {
                    if i >= &(*size as isize) {
                        tcxt.errors.push_error(Error::error_with_span(
                            tcxt,
                            span,
                            "out of bound of static array",
                        ));
                        tcxt.errors.poisoned(true);
                    }
                }
                new = ty.val.clone();
            } else {
                break;
            }
        }
        Some(new)
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
            Ty::Path(p) => {
                write!(f, "{}", p.segs.iter().map(|i| i.name()).collect::<Vec<_>>().join("::"))
            }
            Ty::ConstStr(..) => write!(f, "string"),
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
            // TODO: does size matter for all uses
            (Ty::Array { size: s1, ty: t1 }, Ty::Array { size: s2, ty: t2 }) => {
                s1.eq(s2) && t1.is_ty_eq(t2)
            }
            // TODO: generic comparison
            (Ty::Struct { ident: n1, .. }, Ty::Struct { ident: n2, .. }) => n1 == n2,
            (Ty::Enum { ident: n1, gen: g1 }, Ty::Enum { ident: n2, gen: g2 }) => {
                n1 == n2 && g1.iter().zip(g2).all(|(a, b)| a.is_ty_eq(b))
            }
            (Ty::Ptr(t1), Ty::Ptr(t2)) => t1.val.is_ty_eq(&t2.val),
            (Ty::Ref(t1), Ty::Ref(t2)) => t1.val.is_ty_eq(&t2.val),
            // TODO: we don't want/need the size to be ==
            (Ty::ConstStr(..), Ty::ConstStr(..))
            | (Ty::Int, Ty::Int)
            | (Ty::Char, Ty::Char)
            | (Ty::Float, Ty::Float)
            | (Ty::Bool, Ty::Bool)
            | (Ty::Void, Ty::Void) => true,
            (Ty::Generic { ident: i1, .. }, Ty::Generic { ident: i2, .. }) => i1.eq(i2),
            (Ty::Func { .. }, _) => unreachable!("Func type should never be checked"),
            _ => false,
        }
    }
}

impl TypeEquality<Ty> for Generic {
    fn is_ty_eq(&self, other: &Ty) -> bool {
        match other {
            Ty::Generic { ident, bound } => self.ident.eq(ident) && self.bound.eq(bound),
            _ => false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Param {
    pub ty: RawPtr<Type>,
    pub ident: Ident,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Block {
    pub stmts: RawVec<Statement>,
    pub span: Range,
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "}}")?;
        for _stmt in self.stmts.iter() {
            write!(f, "..;")?;
        }
        write!(f, "}}")
    }
}

#[derive(Clone, Debug)]
pub enum Binding {
    Wild(Ident),
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
        path: Path,
        variant: Ident,
        items: Vec<Pattern>,
    },
    Array {
        size: usize,
        items: Vec<Pattern>,
    },
    Bind(Binding),
}

impl Spany for Pat {}

impl fmt::Display for Pat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Enum { path, variant, items, .. } => write!(
                f,
                "{}::{}{}",
                path,
                variant,
                if items.is_empty() {
                    "".to_owned()
                } else {
                    format!(
                        "({})",
                        items.iter().map(|b| b.val.to_string()).collect::<Vec<_>>().join(", ")
                    )
                },
            ),
            Self::Array { size: _, items } => write!(
                f,
                "[{}]",
                items.iter().map(|b| b.val.to_string()).collect::<Vec<_>>().join(", ")
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
    Const(Const),
    /// Assignment `lval = rval;`
    Assign { lval: Expression, rval: Expression, ty: Option<Type>, is_let: bool },
    /// Assignment operations `lval += rval;`
    AssignOp { lval: Expression, rval: Expression, op: BinOp },
    /// A call statement `call(arg1, arg2)`
    Call(Expression),
    /// A trait method call `<<T>::trait>(args)`
    TraitMeth(Expression),
    /// If statement `if (expr) { stmts }`
    If { cond: Expression, blk: Block, els: Option<Block> },
    /// While loop `while (expr) { stmts }`
    While { cond: Expression, blk: Block },
    /// A match statement `match expr { variant1 => { stmts }, variant2 => { stmts } }`.
    Match { expr: Expression, arms: Vec<MatchArm> },
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
    pub ident: Ident,
    pub ty: RawPtr<Type>,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Struct {
    pub ident: Ident,
    pub fields: Vec<Field>,
    pub generics: Vec<Generic>,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Variant {
    /// The name of the variant `some`.
    pub ident: Ident,
    /// The types contained in the variants "tuple".
    pub types: RawVec<Type>,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Enum {
    /// The name of the enum `<option>::none`.
    pub ident: Ident,
    /// The variants of the enum `option::<some(type, type)>`.
    pub variants: Vec<Variant>,
    pub generics: Vec<Generic>,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub enum Adt {
    Struct(Struct),
    Enum(Enum),
}

impl Adt {
    fn ident(&self) -> Ident {
        match self {
            Adt::Struct(it) => it.ident,
            Adt::Enum(it) => it.ident,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generic {
    pub ident: Ident,
    pub bound: Option<Path>,
    pub span: Range,
}

impl Generic {
    crate fn to_type(&self) -> Ty {
        Ty::Generic { ident: self.ident, bound: self.bound.clone() }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuncKind {
    Normal,
    Linked,
    Extern,
    EmptyTrait,
}

#[derive(Clone, Debug)]
pub struct Func {
    /// The return type `int name() { stmts }`
    pub ret: RawPtr<Type>,
    /// Name of the function.
    pub ident: Ident,
    /// The generic parameters listed for a function.
    pub generics: Vec<Generic>,
    /// the type and identifier of each parameter.
    pub params: Vec<Param>,
    /// All the crap the function does.
    pub stmts: Block,
    pub kind: FuncKind,
    pub span: Range,
}

impl Default for Func {
    fn default() -> Self {
        Self {
            ret: crate::rawptr!(Ty::Void.into_spanned(DUMMY)),
            ident: Ident::dummy(),
            generics: vec![],
            params: vec![],
            stmts: Block { stmts: crate::raw_vec![], span: DUMMY },
            kind: FuncKind::Normal,
            span: DUMMY,
        }
    }
}

#[derive(Clone, Debug)]
pub enum TraitMethod {
    Default(Func),
    NoBody(Func),
}

impl TraitMethod {
    crate fn return_ty(&self) -> &Type {
        match self {
            Self::Default(func) | Self::NoBody(func) => func.ret.get(),
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
    pub path: Path,
    pub generics: Vec<Generic>,
    pub method: TraitMethod,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub struct Impl {
    pub path: Path,
    pub type_arguments: Vec<Type>,
    pub method: Func,
    pub span: Range,
}

/// A const declaration.
///
/// `const foo: type = expr;`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Const {
    pub ty: Type,
    pub ident: Ident,
    pub init: Expression,
    pub span: Range,
}

#[derive(Clone, Debug)]
pub enum Decl {
    Adt(Adt),
    Func(Func),
    Trait(Trait),
    Impl(Impl),
    Const(Const),
    Import(Path),
}

impl Decl {
    crate fn name(&self) -> Ident {
        match self {
            Decl::Adt(it) => it.ident(),
            Decl::Func(it) => it.ident,
            Decl::Trait(it) => it.path.segs[0],
            Decl::Impl(it) => it.path.segs[0],
            Decl::Const(it) => it.ident,
            Decl::Import(it) => it.segs[0],
        }
    }
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
        // f.debug_struct("Spanned").field("span", &self.span).field("val", &self.val).finish()
        f.debug_tuple("Spanned").field(&self.val).finish()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Range {
    pub start: usize,
    pub end: usize,
    pub file_id: u64,
}

impl fmt::Debug for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}..{}", self.start, self.end)
    }
}

crate const fn to_rng(other: ops::Range<usize>, file_id: u64) -> Range {
    let (start, end) = (other.start, other.end);
    Range { start, end, file_id }
}

pub const DUMMY: Range = to_rng(0..0, 0);

pub type Declaration = Spanned<Decl>;
pub type Statement = Spanned<Stmt>;
pub type Expression = Spanned<Expr>;
pub type Type = Spanned<Ty>;
pub type Value = Spanned<Val>;
pub type Pattern = Spanned<Pat>;
