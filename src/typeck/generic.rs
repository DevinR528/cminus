use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::{
    ast::types::{Expr, Ty, Type, Var, DUMMY},
    typeck::TyCheckRes,
};

#[derive(Clone, Debug)]
crate enum TyRegion<'ast> {
    Expr(&'ast Expr),
    VarDecl(&'ast Var),
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
crate enum Node {
    Func(String),
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
crate enum SubsTy<'ast> {
    /// The parent type will always be `Ty::Generic`
    Parent(Ty, Vec<SubsTy<'ast>>),
    /// The `Ty` is always a `Ty::Generic`.
    UnSubed(Ty, TyRegion<'ast>),
    /// This is always a resolved type, never `Ty::Generic`.
    Resolved(Ty),
}

#[derive(Debug, Default)]
struct GenericParam {
    generics: BTreeSet<String>,
    children: BTreeMap<Node, GenericParam>,
}

impl GenericParam {
    fn insert_generic(&mut self, id: &str) {
        self.generics.insert(id.to_owned());
    }
}

#[derive(Debug, Default)]
crate struct GenericResolver<'ast> {
    /// Mapping of region name (function or struct/enum) to the substitution types that
    /// represent each generic.
    node_generic: BTreeMap<Node, BTreeMap<String, SubsTy<'ast>>>,
    node_resolved: BTreeMap<Node, Vec<Ty>>,
    item_generics: BTreeMap<Node, GenericParam>,
}

impl<'ast> GenericResolver<'ast> {
    crate fn get_resolved(&self, node: &Node, idx: usize) -> Option<&Ty> {
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
                self.item_generics.entry(node.clone()).or_default().insert_generic(ident);
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
                println!("walk {:?}", ty);
            }
        }
    }

    fn push_generic_child(
        &mut self,
        stack: &[Node],
        expr: &TyRegion<'ast>,
        id: &str,
    ) -> Option<GenericParam> {
        println!("GEN STACK {:?} {:?}\n", stack, expr);
        let mut iter = stack.iter();
        let mut gp = self.item_generics.get_mut(iter.next()?)?;

        let mut generics = BTreeSet::new();
        generics.insert(id.to_owned());
        gp.children
            .insert(iter.next()?.clone(), GenericParam { generics, children: BTreeMap::default() })
    }

    crate fn push_resolved_child(&mut self, stack: &[Node], ty: &Ty) {
        for node in stack.iter().rev() {
            self.node_resolved.entry(node.clone()).or_default().push(ty.clone());
        }
    }
}

crate fn check_type_arg(tcxt: &mut TyCheckRes<'_, '_>, id: &str) -> Ty {
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
            .map(|fields| Ty::Struct {
                ident: s.to_owned(),
                gen: fields.iter().map(|f| f.ty.clone()).collect(),
            })
            .or_else(|| {
                tcxt.enum_fields.get(s).map(|(generics, _variants)| Ty::Enum {
                    ident: s.to_owned(),
                    gen: generics.clone(),
                })
            })
            .unwrap_or(Ty::Generic { ident: s.to_string(), bound: () }),
    }
}

crate fn collect_generic_usage<'ast>(
    tcxt: &mut TyCheckRes<'ast, '_>,
    ty: &Ty,
    expr: &TyRegion<'ast>,
    stack: &mut Vec<Node>,
) -> Ty {
    println!("collect {:?} {:?}", ty, stack);
    match &ty {
        Ty::Generic { ident: outer_name, bound } => {
            let res = check_type_arg(tcxt, outer_name);
            match &res {
                Ty::Generic { ident, bound } => {
                    tcxt.generic_res.push_generic_child(stack, expr, ident);
                }
                Ty::Array { size, ty } => todo!(),
                Ty::Struct { ident, gen } => todo!(),
                Ty::Enum { ident: inner_name, gen } => {
                    // panic!("{:?}", stack);

                    // TODO: whaaaaat hmm what do I do.
                    assert!(gen.is_empty());
                    tcxt.generic_res.push_resolved_child(stack, &res);
                }
                Ty::Ptr(_) => todo!(),
                Ty::Ref(_) => todo!(),
                Ty::String => todo!(),
                Ty::Int => todo!(),
                Ty::Char => todo!(),
                Ty::Float => todo!(),
                Ty::Bool => todo!(),
                Ty::Void => todo!(),
                Ty::Func { ident, ret, params } => todo!(),
            }
            // TODO: add to generic resolver
            res
        }
        Ty::Array { size, ty } => Ty::Array {
            size: *size,
            ty: box collect_generic_usage(tcxt, &ty.val, expr, stack).into_spanned(DUMMY),
        },
        Ty::Struct { ident, gen } => todo!(),
        Ty::Enum { ident, gen } => {
            stack.push(Node::Enum(ident.clone()));

            let en = Ty::Enum {
                ident: ident.clone(),
                gen: gen
                    .iter()
                    .map(|t| collect_generic_usage(tcxt, &t.val, expr, stack).into_spanned(DUMMY))
                    .collect(),
            };

            stack.pop();
            en
        }
        Ty::Func { ident, ret, params } => {
            stack.push(Node::Func(ident.clone()));
            todo!()
        }
        Ty::Ptr(t) => collect_generic_usage(tcxt, &t.val, expr, stack),
        Ty::Ref(t) => collect_generic_usage(tcxt, &t.val, expr, stack),
        _ => ty.clone(),
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
