use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fmt,
};

use crate::{
    ast::types::{Expr, Spany, Ty, Type, Var, DUMMY},
    typeck::TyCheckRes,
};

#[derive(Clone)]
crate enum TyRegion<'ast> {
    Expr(&'ast Expr),
    VarDecl(&'ast Var),
}

impl fmt::Debug for TyRegion<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Expr(e) => write!(f, "Expr(..)"),
            Self::VarDecl(e) => write!(f, "VarDecl({})", e.ident),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
crate enum Node {
    Func(String),
    Trait(String),
    Enum(String),
    Struct(String),
}

impl Node {
    crate fn type_parent(ty: &Ty) -> Option<Node> {
        match ty {
            Ty::Generic { ident, bound } => None,
            Ty::Array { size, ty } => todo!(),
            Ty::Struct { ident, gen } => Some(Node::Struct(ident.clone())),
            Ty::Enum { ident, gen } => Some(Node::Enum(ident.clone())),
            Ty::Func { ident, .. } => Some(Node::Func(ident.clone())),
            Ty::Ptr(_)
            | Ty::Ref(_)
            | Ty::String
            | Ty::Int
            | Ty::Char
            | Ty::Float
            | Ty::Bool
            | Ty::Void => None,
        }
    }
}

#[derive(Debug)]
crate struct GenericArgument<'ast> {
    crate ty: Ty,
    exprs: Vec<TyRegion<'ast>>,
}

#[derive(Debug, Default)]
struct GenericParam {
    /// Generic type name `T` to possible bounds `T: add`.
    generics: BTreeMap<String, Option<String>>,
    /// Any dependent generic types. When monomorphizing these will be walked to create
    /// mono variants of each type.
    children: BTreeMap<Node, GenericParam>,
}

impl GenericParam {
    fn insert_generic(&mut self, id: &str, bound: Option<String>) {
        self.generics.insert(id.to_owned(), bound);
    }
}

#[derive(Debug, Default)]
crate struct GenericResolver<'ast> {
    /// Mapping of region name (function or struct/enum) to the generic arguments.
    ///
    /// These are the "resolved" types.
    node_resolved: BTreeMap<Node, Vec<GenericArgument<'ast>>>,
    /// Mapping of declaration (function or struct or enum) to the generic parameter.
    ///
    /// If a function defines a dependent statement that relationship is preserved.
    /// ```c
    /// enum option<T> foo<T>(T x) {
    ///     enum option<T> abc;
    ///     abc = option::some(x);
    ///     return abc;
    /// }
    /// ```
    item_generics: BTreeMap<Node, GenericParam>,
}

impl<'ast> GenericResolver<'ast> {
    crate fn get_resolved(&self, node: &Node, idx: usize) -> Option<&GenericArgument<'_>> {
        if let Some(res) = self.node_resolved.get(node) {
            res.get(idx)
        } else {
            None
        }
    }

    crate fn has_generics(&self, node: &Node) -> bool {
        self.item_generics.contains_key(node)
    }

    crate fn collect_generic_params(&mut self, node: &Node, ty: &Ty) {
        match ty {
            Ty::Generic { ident, bound } => {
                self.item_generics
                    .entry(node.clone())
                    .or_default()
                    .insert_generic(ident, bound.clone());
            }
            Ty::Array { size, ty } => todo!(),
            Ty::Struct { ident, gen } => {
                for t in gen {
                    self.collect_generic_params(node, &t.val);
                }
            }
            Ty::Enum { ident, gen } => {
                for t in gen {
                    self.collect_generic_params(node, &t.val);
                }
            }
            Ty::Func { ident, ret, params } => {
                if let Ty::Generic { .. } = &**ret {
                    self.collect_generic_params(node, ret);
                }
                for t in params {
                    self.collect_generic_params(node, t);
                }
            }
            _ => {
                panic!("walk {:?}", ty);
            }
        }
    }

    fn push_generic_child(
        &mut self,
        stack: &[Node],
        expr: &[TyRegion<'ast>],
        id: &str,
        bound: Option<String>,
    ) -> Option<GenericParam> {
        println!("GEN STACK {:?} {:?}\n", stack, expr);
        let mut iter = stack.iter();
        let mut gp = self.item_generics.get_mut(iter.next()?)?;

        let mut generics = BTreeMap::new();
        generics.insert(id.to_owned(), bound);
        gp.children
            .insert(iter.next()?.clone(), GenericParam { generics, children: BTreeMap::default() })
    }

    crate fn push_resolved_child(&mut self, stack: &[Node], ty: &Ty, exprs: Vec<TyRegion<'ast>>) {
        for node in stack.iter().rev() {
            self.node_resolved
                .entry(node.clone())
                .or_default()
                .push(GenericArgument { ty: ty.clone(), exprs: exprs.clone() });
        }
    }
}

crate fn check_type_arg(tcxt: &mut TyCheckRes<'_, '_>, id: &str, bound: &Option<String>) -> Ty {
    // TODO: make <int[3]> work
    match id {
        "bool" => Ty::Bool,
        "int" => Ty::Int,
        "char" => Ty::Char,
        "float" => Ty::Float,
        "string" => Ty::String,
        s => tcxt
            .struct_fields
            .get(s)
            .map(|(generics, _fields)| Ty::Struct { ident: s.to_owned(), gen: generics.clone() })
            .or_else(|| {
                tcxt.enum_fields.get(s).map(|(generics, _variants)| Ty::Enum {
                    ident: s.to_owned(),
                    gen: generics.clone(),
                })
            })
            .unwrap_or(Ty::Generic { ident: s.to_string(), bound: bound.clone() }),
    }
}

/// Collect all the generics to track resolved and dependent sites/uses.
///
/// This also converts any type arguments to their correct type.
crate fn collect_generic_usage<'ast>(
    tcxt: &mut TyCheckRes<'ast, '_>,
    ty: &Ty,
    exprs: &[TyRegion<'ast>],
    stack: &mut Vec<Node>,
) -> Ty {
    println!("collect {:?} {:?}", ty, stack);
    match &ty {
        Ty::Generic { ident: outer_name, bound } => {
            let res = check_type_arg(tcxt, outer_name, bound);
            match &res {
                Ty::Generic { ident, bound } => {
                    tcxt.generic_res.push_generic_child(stack, exprs, ident, bound.clone());
                }
                Ty::Array { size, ty } => todo!(),
                Ty::Struct { ident, gen } => {
                    // TODO: whaaaaat hmm what do I do.
                    assert!(gen.is_empty());
                    tcxt.generic_res.push_resolved_child(stack, &res, exprs.to_vec());
                }
                Ty::Enum { ident: inner_name, gen } => {
                    // TODO: whaaaaat hmm what do I do.
                    assert!(gen.is_empty());
                    tcxt.generic_res.push_resolved_child(stack, &res, exprs.to_vec());
                }
                Ty::Ptr(_) => todo!(),
                Ty::Ref(_) => todo!(),
                Ty::String | Ty::Int | Ty::Char | Ty::Float | Ty::Bool => {
                    tcxt.generic_res.push_resolved_child(stack, &res, exprs.to_vec());
                }
                Ty::Void => todo!(),
                Ty::Func { ident, ret, params } => todo!(),
            }
            res
        }
        Ty::Array { size, ty } => Ty::Array {
            size: *size,
            ty: box collect_generic_usage(tcxt, &ty.val, exprs, stack).into_spanned(DUMMY),
        },
        Ty::Struct { ident, gen } => {
            stack.push(Node::Struct(ident.clone()));

            let struc = Ty::Struct {
                ident: ident.clone(),
                gen: gen
                    .iter()
                    .map(|t| collect_generic_usage(tcxt, &t.val, exprs, stack).into_spanned(DUMMY))
                    .collect(),
            };

            stack.pop();
            struc
        }
        Ty::Enum { ident, gen } => {
            stack.push(Node::Enum(ident.clone()));

            let en = Ty::Enum {
                ident: ident.clone(),
                gen: gen
                    .iter()
                    .map(|t| collect_generic_usage(tcxt, &t.val, exprs, stack).into_spanned(DUMMY))
                    .collect(),
            };

            stack.pop();
            en
        }
        Ty::Func { ident, ret, params } => {
            stack.push(Node::Func(ident.clone()));
            todo!()
        }
        Ty::Ptr(t) => {
            Ty::Ptr(box collect_generic_usage(tcxt, &t.val, exprs, stack).into_spanned(DUMMY))
        }
        Ty::Ref(t) => {
            Ty::Ref(box collect_generic_usage(tcxt, &t.val, exprs, stack).into_spanned(DUMMY))
        }
        _ => {
            println!("{:?}", ty);
            tcxt.generic_res.push_resolved_child(stack, ty, exprs.to_vec());
            ty.clone()
        }
    }
}

struct Foo<T> {
    it: T,
}

enum Bar<T> {
    Var(T),
    Other,
}

fn func<T>(it: T) {}

fn func2<T>(it: T) -> Bar<T> {
    let x: Bar<T> = Bar::Var(it);
    let y = func2(10);
    x
}

/*
fn main() {
    // This includes enums, structs, functions
    res.collect_generic_declarations();

    // direct use so `enum option<int> opt;` and `enum option<T> opt;` where T is
    // inferred from a parent function.
    res.collect_use_stmts();
}
*/
