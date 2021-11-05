use std::collections::HashSet;

use proc_macro::TokenStream as StdTokenStream;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::{
    parse_macro_input, parse_quote, spanned::Spanned, Attribute, Data, DeriveInput, Fields,
    Generics, Type, Variant, WherePredicate,
};

/// Generates an implementation of `ruma_events::EventContent`.
#[proc_macro_derive(Debug, attributes(dbg_ignore, with_fmt))]
pub fn derive_event_content(input: StdTokenStream) -> StdTokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = &input.ident;
    let mut generics = input.generics.clone();
    let fields = match input.data {
        Data::Struct(s) => fmt_field(&s.fields, name, &mut generics),
        Data::Enum(e) => {
            let arms = e
                .variants
                .iter()
                .filter_map(|v| {
                    if has_ignore(&v.attrs) {
                        None
                    } else {
                        Some(variant_to_arm(v, &mut generics))
                    }
                })
                .collect::<TokenStream>();

            quote! {
                match self {
                    #arms
                    _ => {
                        format!("{}::<ignored>", stringify!(#name)).fmt(fmtr)
                    }
                }
            }
        }
        Data::Union(_) => {
            return syn::Error::new(Span::call_site(), "union type not supported")
                .into_compile_error()
                .into();
        }
    };
    let (impl_gen, ty_gen, where_clause) = generics.split_for_impl();

    quote! {
        #[automatically_derived]
        #[allow(unused_qualifications)]
        impl #impl_gen ::std::fmt::Debug for #name #ty_gen #where_clause {
            fn fmt(&self, fmtr: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
                #fields
            }
        }
    }
    .into()
}

fn variant_to_arm(variant: &Variant, gen: &mut Generics) -> TokenStream {
    let gen_types = gen.type_params().into_iter().map(|t| t.ident.clone()).collect::<Vec<_>>();
    let mut done_gen = HashSet::new();

    let name = &variant.ident;
    let mut is_struct_like = false;
    let (fields, debug_names): (TokenStream, TokenStream) = variant
        .fields
        .iter()
        .enumerate()
        .filter_map(|(idx, f)| {
            for path in peel_path(&f.ty) {
                if !done_gen.insert(path.get_ident()) {
                    continue;
                }
                if let Some(ty_name) = gen_types.iter().find(|id| Some(*id) == path.get_ident()) {
                    let bound: WherePredicate = parse_quote! { #ty_name: ::core::fmt::Debug };
                    gen.make_where_clause().predicates.extend([bound].into_iter())
                }
            }

            if has_ignore(&f.attrs) {
                None
            } else {
                Some(if let Some(field_name) = &f.ident {
                    is_struct_like = true;
                    (
                        quote! { #field_name, },
                        quote! {
                            let _ =
                                ::core::fmt::DebugStruct::field(
                                    dbg_builder,
                                    stringify!(#field_name),
                                    #field_name
                                );
                        },
                    )
                } else {
                    let field_name = Ident::new(&format!("_self_{}", idx), f.span());
                    let idx = syn::Index::from(idx);
                    (
                        quote! {
                            #idx: #field_name,
                        },
                        quote! {
                            let _ =
                                ::core::fmt::DebugTuple::field(
                                    dbg_builder,
                                    #field_name
                                );
                        },
                    )
                })
            }
        })
        .unzip();
    let (start_fmtr, end_fmtr) = if is_struct_like {
        (
            quote! {
                let dbg_builder = &mut ::core::fmt::Formatter::debug_struct(fmtr, stringify!(#name));
            },
            quote! {
                ::core::fmt::DebugStruct::finish(dbg_builder)
            },
        )
    } else {
        (
            quote! {
                let dbg_builder = &mut ::core::fmt::Formatter::debug_tuple(fmtr, stringify!(#name));
            },
            quote! {
                ::core::fmt::DebugTuple::finish(dbg_builder)
            },
        )
    };
    quote! {
        Self::#name { #fields .. } => {
            #start_fmtr
            #debug_names
            #end_fmtr
        }
    }
}

fn fmt_field(fields: &Fields, name: &Ident, gen: &mut Generics) -> TokenStream {
    let gen_types = gen.type_params().into_iter().map(|t| t.ident.clone()).collect::<Vec<_>>();
    let mut done_gen = HashSet::new();

    let mut is_struct_like = false;
    let (fields, debug_names): (TokenStream, TokenStream) = fields
        .iter()
        .enumerate()
        .filter_map(|(idx, f)| {
            for path in peel_path(&f.ty) {
                if !done_gen.insert(path.get_ident()) {
                    continue;
                }
                if let Some(ty_name) = gen_types.iter().find(|id| Some(*id) == path.get_ident()) {
                    let bound: WherePredicate = parse_quote! { #ty_name: ::core::fmt::Debug };
                    gen.make_where_clause().predicates.extend([bound].into_iter())
                }
            }

            if has_ignore(&f.attrs) {
                None
            } else {
                Some(if let Some(field_name) = &f.ident {
                    is_struct_like = true;
                    (
                        quote! { #field_name, },
                        quote! {
                            let _ =
                                ::core::fmt::DebugStruct::field(
                                    dbg_builder,
                                    stringify!(#field_name),
                                    #field_name
                                );
                        },
                    )
                } else {
                    let field_name = Ident::new(&format!("_self_{}", idx), f.span());
                    let idx = syn::Index::from(idx);
                    (
                        quote! {
                            #idx: #field_name,
                        },
                        quote! {
                            let _ =
                                ::core::fmt::DebugTuple::field(
                                    dbg_builder,
                                    #field_name
                                );
                        },
                    )
                })
            }
        })
        .unzip();

    let start_fmtr = quote! {
        let dbg_builder = &mut ::core::fmt::Formatter::debug_struct(fmtr, stringify!(#name));
    };
    let end_fmtr = quote! {
        ::core::fmt::DebugStruct::finish(dbg_builder)
    };
    quote! {
        let #name { #fields .. } = &self;
        #start_fmtr
        #debug_names
        #end_fmtr
    }
}

fn peel_path(ty: &Type) -> Vec<&syn::Path> {
    match ty {
        Type::Array(at) => peel_path(&*at.elem),
        Type::Group(gt) => peel_path(&*gt.elem),
        Type::Paren(pt) => peel_path(&*pt.elem),
        Type::Path(path) => vec![&path.path],
        Type::Ptr(pt) => peel_path(&*pt.elem),
        Type::Reference(rt) => peel_path(&*rt.elem),
        Type::Slice(st) => peel_path(&*st.elem),
        Type::Tuple(t) => t.elems.iter().flat_map(peel_path).collect(),
        _ => vec![],
    }
}

fn has_ignore(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| a.path.is_ident("dbg_ignore"))
}
