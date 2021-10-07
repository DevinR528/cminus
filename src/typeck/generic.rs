use std::collections::{BTreeMap, HashSet};

use crate::{
    ast::types::{Ty, Type, DUMMY},
    typeck::TyCheckRes,
};

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

#[derive(Debug, Default)]
crate struct GenericResolver {
    node_generic: BTreeMap<Node, Vec<Ty>>,
    node_resolved: BTreeMap<Node, Vec<Ty>>,
}

impl GenericResolver {
    crate fn get_resolved(&self, node: &Node, idx: usize) -> Option<&Ty> {
        if let Some(res) = self.node_resolved.get(node) {
            res.get(idx)
        } else {
            None
        }
    }

    crate fn insert_generic(&mut self, node: Node, gen: Ty) {
        self.node_generic.entry(node).or_default().push(gen);
    }

    crate fn push_resolved(&mut self, node: &Node, gen: Ty) -> bool {
        if let Some(res) = self.node_resolved.get_mut(node) {
            res.push(gen);
            return true;
        }
        false
    }
}

crate fn check_type_arg(tcxt: &mut TyCheckRes<'_, '_>, id: &str) -> Option<Ty> {
    Some(match id {
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
                tcxt.enum_fields.get(s).map(|variants| Ty::Enum {
                    ident: s.to_owned(),
                    gen: {
                        let set: HashSet<Ty> = variants
                            .iter()
                            .map(|v| v.types.iter().map(|t| t.val.clone()))
                            .flatten()
                            .collect();
                        // TODO: this could be out of order
                        set.into_iter().map(|t| t.into_spanned(DUMMY)).collect()
                    },
                })
            })?,
    })
}

crate fn collect_generics(tcxt: &mut TyCheckRes<'_, '_>, ty: &Type, parent: Option<Node>) -> Ty {
    match &ty.val {
        Ty::Generic { ident, bound } => {
            let res = check_type_arg(tcxt, ident).expect("no type found for generic argument");
            // TODO: add to generic resolver
            res
        }
        Ty::Array { size, ty } => Ty::Array {
            size: *size,
            ty: box collect_generics(tcxt, ty, parent).into_spanned(DUMMY),
        },
        Ty::Struct { ident, gen } => todo!(),
        Ty::Enum { ident, gen } => Ty::Enum {
            ident: ident.clone(),
            gen: gen
                .iter()
                .map(|t| collect_generics(tcxt, t, parent.clone()).into_spanned(DUMMY))
                .collect(),
        },
        Ty::Ptr(t) => collect_generics(tcxt, t, parent),
        Ty::Ref(t) => collect_generics(tcxt, t, parent),
        _ => ty.val.clone(),
    }
}
