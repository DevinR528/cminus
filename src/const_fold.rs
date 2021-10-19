use crate::{
    ast::types::{
        Adt, BinOp, Binding, Block, Decl, Expr, Expression, Field, FieldInit, Func, Generic, Impl,
        MatchArm, Param, Pat, Range, Spanned, Spany, Statement, Stmt, Struct, Trait, Ty, Type,
        TypeEquality, UnOp, Val, Value, Var, Variant, DUMMY,
    },
    error::Error,
    visit::VisitMut,
};

#[derive(Debug, Default)]
crate struct Folder;

impl<'ast> VisitMut<'ast> for Folder {
    fn visit_expr(&mut self, expr: &'ast mut Expression) {
        // crate::visit::walk_mut_expr(self, expr);

        expr.val.const_fold();
    }
}

crate trait Foldable {
    fn const_fold(&mut self);
}

impl Foldable for Expr {
    fn const_fold(&mut self) {
        match self {
            Expr::Ident(_) => {
                // TODO: this could be checked if it was instantiated with a const value
            }
            Expr::AddrOf(expr) | Expr::Deref { expr, .. } => {
                expr.val.const_fold();
            }
            Expr::Parens(expr) => {
                expr.val.const_fold();
                if let ex @ Expr::Value(_) = &expr.val {
                    *self = ex.clone();
                }
            }
            Expr::Urnary { expr, op } => {
                if let Expr::Value(val) = &mut expr.val {
                    match op {
                        UnOp::Not => match val.val {
                            Val::Int(i) => {
                                // TODO: hmmm is this right ??
                                val.val = Val::Bool(!(i != 0));
                            }
                            Val::Bool(b) => {
                                val.val = Val::Bool(!b);
                            }
                            _ => {}
                        },
                        UnOp::OnesComp => match val.val {
                            Val::Int(i) => {
                                // TODO: hmmm is this right ??
                                *self = Expr::Value(Val::Int(!i).into_spanned(val.span));
                            }
                            Val::Float(f) => {
                                // TODO: hmmm is this right ??
                                val.val = Val::Float(f64::from_bits(!f.to_bits()));
                            }
                            Val::Bool(b) => {
                                val.val = Val::Bool(!b);
                            }
                            _ => {}
                        },
                    }
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                lhs.val.const_fold();
                rhs.val.const_fold();
                if let Some(folded) = eval_binop(op, lhs, rhs) {
                    *self = folded;
                }
            }
            Expr::Array { exprs, .. } => {
                for expr in exprs {
                    expr.val.const_fold();
                }
            }
            Expr::Call { ident, args, type_args } => {
                for expr in args {
                    expr.val.const_fold();
                }
            }
            Expr::TraitMeth { trait_, args, type_args } => {
                for expr in args {
                    expr.val.const_fold();
                }
            }
            Expr::StructInit { name, fields } => {
                for expr in fields {
                    expr.init.val.const_fold();
                }
            }
            Expr::EnumInit { items, .. } => {
                for expr in items {
                    expr.val.const_fold();
                }
            }
            Expr::ArrayInit { items } => {
                for expr in items {
                    expr.val.const_fold();
                }
            }
            Expr::Value(_) | Expr::FieldAccess { .. } => {}
        }
    }
}

// TODO: identity folding `x & 0` or `x * 0` is always `0`
fn eval_binop(op: &BinOp, lhs: &Expression, rhs: &Expression) -> Option<Expr> {
    let lval = if let Expr::Value(val) = &lhs.val {
        &val.val
    } else {
        return None;
    };
    let rval = if let Expr::Value(val) = &rhs.val {
        &val.val
    } else {
        return None;
    };
    Some(Expr::Value(lval.evaluate(op, &rval).into_spanned(lhs.span)))
}

impl Val {
    fn evaluate(&self, op: &BinOp, other: &Val) -> Val {
        match (self, other) {
            (Val::Float(f1), Val::Float(f2)) => float_op(*f1, *f2, op),
            (Val::Int(i1), Val::Int(i2)) => int_op(*i1, *i2, op),
            (Val::Char(c1), Val::Char(c2)) => char_op(*c1, *c2, op),
            (Val::Bool(b1), Val::Bool(b2)) => bool_op(*b1, *b2, op),
            (Val::Str(s1), Val::Str(s2)) => str_op(s1, s2, op),
            _ => todo!("coercion and const fold"),
        }
    }
}

fn float_op(a: f64, b: f64, op: &BinOp) -> Val {
    match op {
        BinOp::Mul => Val::Float(a * b),
        BinOp::Div => Val::Float(a / b),
        BinOp::Rem => Val::Float(a % b),
        BinOp::Add => Val::Float(a + b),
        BinOp::Sub => Val::Float(a - b),
        BinOp::LeftShift => Val::Float(f64::from_bits(a.to_bits() << b.to_bits())),
        BinOp::RightShift => Val::Float(f64::from_bits(a.to_bits() >> b.to_bits())),
        BinOp::Lt => Val::Bool(a < b),
        BinOp::Le => Val::Bool(a <= b),
        BinOp::Ge => Val::Bool(a >= b),
        BinOp::Gt => Val::Bool(a > b),
        BinOp::Eq => Val::Bool(a == b),
        BinOp::Ne => Val::Bool(a != b),
        BinOp::And => Val::Bool((a.to_bits() != 0) && (b.to_bits() != 0)),
        BinOp::Or => Val::Bool((a.to_bits() != 0) || (b.to_bits() != 0)),
        _ => {
            unreachable!("ICE illegal float operation")
        }
    }
}

fn int_op(a: isize, b: isize, op: &BinOp) -> Val {
    match op {
        BinOp::Mul => Val::Int(a * b),
        BinOp::Div => Val::Int(a / b),
        BinOp::Rem => Val::Int(a % b),
        BinOp::Add => Val::Int(a + b),
        BinOp::Sub => Val::Int(a - b),
        BinOp::LeftShift => Val::Int(a << b),
        BinOp::RightShift => Val::Int(a >> b),
        BinOp::Lt => Val::Bool(a < b),
        BinOp::Le => Val::Bool(a <= b),
        BinOp::Ge => Val::Bool(a >= b),
        BinOp::Gt => Val::Bool(a > b),
        BinOp::Eq => Val::Bool(a == b),
        BinOp::Ne => Val::Bool(a != b),
        BinOp::BitAnd => Val::Int(a & b),
        BinOp::BitXor => Val::Int(a ^ b),
        BinOp::BitOr => Val::Int(a | b),
        BinOp::And => Val::Bool((a != 0) && (b != 0)),
        BinOp::Or => Val::Bool((a != 0) || (b != 0)),
        BinOp::AddAssign | BinOp::SubAssign => {
            unreachable!("ICE assign operations should be lowered by const folding")
        }
    }
}

fn char_op(a: char, b: char, op: &BinOp) -> Val {
    match op {
        BinOp::Lt => Val::Bool(a < b),
        BinOp::Le => Val::Bool(a <= b),
        BinOp::Ge => Val::Bool(a >= b),
        BinOp::Gt => Val::Bool(a > b),
        BinOp::Eq => Val::Bool(a == b),
        BinOp::Ne => Val::Bool(a != b),
        _ => {
            unreachable!("ICE assign operations should be lowered by const folding")
        }
    }
}

fn bool_op(a: bool, b: bool, op: &BinOp) -> Val {
    match op {
        BinOp::Lt => Val::Bool(a < b),
        BinOp::Le => Val::Bool(a <= b),
        BinOp::Ge => Val::Bool(a >= b),
        BinOp::Gt => Val::Bool(a > b),
        BinOp::Eq => Val::Bool(a == b),
        BinOp::Ne => Val::Bool(a != b),
        BinOp::BitAnd => Val::Bool(a & b),
        BinOp::BitXor => Val::Bool(a ^ b),
        BinOp::BitOr => Val::Bool(a | b),
        BinOp::And => Val::Bool(a && b),
        BinOp::Or => Val::Bool(a || b),
        _ => {
            unreachable!("ICE assign operations should be lowered by const folding")
        }
    }
}

fn str_op(a: &str, b: &str, op: &BinOp) -> Val {
    match op {
        BinOp::Lt => Val::Bool(a < b),
        BinOp::Le => Val::Bool(a <= b),
        BinOp::Ge => Val::Bool(a >= b),
        BinOp::Gt => Val::Bool(a > b),
        BinOp::Eq => Val::Bool(a == b),
        BinOp::Ne => Val::Bool(a != b),
        _ => {
            unreachable!("ICE assign operations should be lowered by const folding")
        }
    }
}

#[test]
fn float_logical_ops() {
    println!("{}", 1.2 + 2.1);
    println!("{} | {}", (1.2_f64).to_bits(), (2.1_f64).to_bits());
    println!("{}", (1.2_f64).to_bits() ^ (2.1_f64).to_bits());
    println!("{}", f64::from_bits((1.2_f64).to_bits() ^ (2.1_f64).to_bits()));
    // println!("{}", 0.1_f64.to_bits() != 0);
    println!("{}", (1.1_f64.to_bits() != 0) && (0.0_f64.to_bits() != 0));
    println!("{}", !1_u8);
    println!("{}", f64::from_bits(!(1.1_f64).to_bits()));
}

macro_rules! expr {
    ($ex:tt + $($rest:tt)*) => {
        Expr::Binary {
            op: BinOp::Add,
            lhs: box Expr::Value(Val::Int($ex).into_spanned(DUMMY)).into_spanned(DUMMY),
            rhs: box expr!($($rest)*),
        }.into_spanned(DUMMY)
    };
    ($ex:tt * $($rest:tt)*) => {
        Expr::Binary {
            op: BinOp::Mul,
            lhs: box Expr::Value(Val::Int($ex).into_spanned(DUMMY)).into_spanned(DUMMY),
            rhs: box expr!($($rest)*),
        }.into_spanned(DUMMY)
    };
    ($ex:tt - $($rest:tt)*) => {
        Expr::Binary {
            op: BinOp::Sub,
            lhs: box Expr::Value(Val::Int($ex).into_spanned(DUMMY)).into_spanned(DUMMY),
            rhs: box expr!($($rest)*),
        }.into_spanned(DUMMY)
    };
    ($ex:expr) => {
        Expr::Value(Val::Int($ex).into_spanned(DUMMY)).into_spanned(DUMMY)
    };
}

#[test]
fn fold_expr() {
    let mut ex = expr!(5 + 9 * 9 - 3).val;
    ex.const_fold();

    assert!(matches!(
        ex,
        // Need the parenthesis because macro has no precedence
        Expr::Value(Spanned { val: Val::Int(i), .. }) if i == (5 + (9 * (9 - 3)))
    ));
}
