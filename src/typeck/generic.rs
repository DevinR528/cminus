use std::collections::{BTreeMap, HashSet};

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
crate struct GenericParam {
    ident: String,
    owner: Ty,
}

#[derive(Debug)]
crate struct GenericParamStack {
    stack: Vec<Ty>,
    owner: Ty,
    name: String,
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
crate struct GenericResolver<'ast> {
    /// Mapping of region name (function or struct/enum) to the substitution types that
    /// represent each generic.
    node_generic: BTreeMap<Node, BTreeMap<String, SubsTy<'ast>>>,
    node_resolved: BTreeMap<Node, Vec<Ty>>,
    item_generics: Vec<GenericParamStack>,
}

impl<'ast> GenericResolver<'ast> {
    crate fn get_resolved(&self, node: &Node, idx: usize) -> Option<&Ty> {
        if let Some(res) = self.node_resolved.get(node) {
            res.get(idx)
        } else {
            None
        }
    }

    crate fn has_generics(&self, id: &str) -> bool {
        self.item_generics.iter().any(|g| g.name == id)
    }

    crate fn walk_generic_arg_type(&mut self, node: &Node, ty: &Ty) {
        match ty {
            Ty::Generic { ident, bound } => {
                if let Some(item) = self.item_generics.last_mut() {
                    item.stack.push(ty.clone())
                }
            }
            Ty::Array { size, ty } => todo!(),
            Ty::Struct { ident, gen } => {
                let stack =
                    GenericParamStack { stack: vec![], owner: ty.clone(), name: ident.clone() };
                self.item_generics.push(stack);
                for t in gen {
                    self.walk_generic_arg_type(node, &t.val);
                }
            }
            Ty::Enum { ident, gen } => {
                let stack =
                    GenericParamStack { stack: vec![], owner: ty.clone(), name: ident.clone() };
                self.item_generics.push(stack);
                for t in gen {
                    self.walk_generic_arg_type(node, &t.val);
                }
            }
            Ty::Func { ident, ret, params } => {
                let mut stack =
                    GenericParamStack { stack: vec![], owner: ty.clone(), name: ident.clone() };
                if let Ty::Generic { .. } = &**ret {
                    stack.stack.push((**ret).clone());
                }
                self.item_generics.push(stack);
                for t in params {
                    self.walk_generic_arg_type(node, t);
                }
            }
            _ => {
                println!("walk {:?}", ty);
            }
        }
    }

    crate fn insert_generic(&mut self, node: Node, gen: Ty) {
        self.walk_generic_arg_type(&node, &gen);
    }

    crate fn push_generic_child(&mut self, stack: &[Node], expr: &TyRegion<'ast>) {
        // if let Some(linked) = self.node_generic.get_mut(node) {
        //     if let Some(sub) = linked.get_mut(id) {
        //         match sub {
        //             SubsTy::Parent(gen, kids) => {
        //                 // TODO: type error
        //                 assert!(gen.generic_id() == id);

        //                 kids.push(SubsTy::UnSubed(
        //                     Ty::Generic { ident: id.to_owned(), bound: () },
        //                     expr.clone(),
        //                 ))
        //             }
        //             SubsTy::UnSubed(_, ex) => todo!(),
        //             SubsTy::Resolved(_) => todo!(),
        //         }
        //     }
        // }
    }

    crate fn push_resolved_child(&mut self, stack: &[Node], ty: &Ty) {
        for node in stack.iter().rev() {
            self.node_resolved.entry(node.clone()).or_default().push(ty.clone());
        }
    }

    crate fn push_resolved(&mut self, node: &Node, gen: Ty) -> bool {
        if let Some(res) = self.node_resolved.get_mut(node) {
            res.push(gen);
            return true;
        }
        false
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

crate fn generic_usage(
    tcxt: &mut TyCheckRes<'_, '_>,
    ty: Option<&Ty>,
    expr: &Expr,
    expected: Option<&Ty>,
) -> Ty {
    todo!()
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
                    tcxt.generic_res.push_generic_child(stack, expr);
                }
                Ty::Array { size, ty } => todo!(),
                Ty::Struct { ident, gen } => todo!(),
                Ty::Enum { ident: inner_name, gen } => {
                    // panic!("{:?}", stack);

                    // TODO: whaaaaat hmm what do I do.
                    assert!(gen.is_empty());
                    tcxt.generic_res.push_resolved_child(stack, ty);
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
