use std::collections::HashSet;

use proc_macro::TokenStream as StdTokenStream;
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::{
	parse_macro_input, parse_quote, spanned::Spanned, Attribute, Data, DeriveInput, Field, Fields,
	Generics, Meta, NestedMeta, Type, Variant, WherePredicate,
};

/// Generates an implementation of `ruma_events::EventContent`.
#[proc_macro_derive(Debug, attributes(dbg_ignore, dbg_with))]
pub fn derive_event_content(input: StdTokenStream) -> StdTokenStream {
	let input = parse_macro_input!(input as DeriveInput);

	let name = &input.ident;
	let mut generics = input.generics.clone();
	let fields = match input.data {
		Data::Struct(s) => fmt_field(&s.fields, name, &mut generics),
		Data::Enum(e) => {
			let arms = match e
				.variants
				.iter()
				.map(|v| {
					Ok(match has_attrs(&v.attrs)? {
						Attrs::Ignore => TokenStream::new(),
						Attrs::With(call) => variant_to_arm(v, Some(call), &mut generics),
						Attrs::None => variant_to_arm(v, None, &mut generics),
					})
				})
				.collect::<syn::Result<TokenStream>>()
				.map_err(|e| e.into_compile_error().into())
			{
				Ok(a) => a,
				Err(tkns) => return tkns,
			};

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

fn variant_to_arm(
	variant: &Variant,
	with_call: Option<NestedMeta>,
	gen: &mut Generics,
) -> TokenStream {
	let name = &variant.ident;

	// So, `into_compile_error` just emits compile_error!($msg) so the surroundings have to be valid
	// syntax...
	let error = if with_call.is_some() && variant.fields.len() > 1 {
		let err =
			syn::Error::new(variant.span(), "can only call debug_with on one \"field\" at a time")
				.into_compile_error();
		return quote! {
			Self::#name {.. } => {
				#err
			}
		};
	} else {
		TokenStream::new()
	};

	let gen_types = gen.type_params().into_iter().map(|t| t.ident.clone()).collect::<Vec<_>>();
	let mut done_gen = HashSet::new();

	let mut has_named_fields = false;
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

			let attrs = match has_attrs(&f.attrs).map_err(|e| e.into_compile_error()) {
				Ok(a) => a,
				Err(e) => {
					return Some((e.clone(), e));
				}
			};

			if with_call.is_some() {
				Some(construct_debug_calls(f, idx, &mut has_named_fields, with_call.as_ref()))
			} else {
				match attrs {
					Attrs::Ignore => None,
					Attrs::With(expr) => {
						Some(construct_debug_calls(f, idx, &mut has_named_fields, Some(&expr)))
					}
					Attrs::None => Some(construct_debug_calls(f, idx, &mut has_named_fields, None)),
				}
			}
		})
		.unzip();

	let (start_fmtr, end_fmtr) = formatter_type(has_named_fields, name);
	quote! {
		Self::#name { #fields .. } => {
			#error
			#start_fmtr
			#debug_names
			#end_fmtr
		}
	}
}

fn fmt_field(fields: &Fields, name: &Ident, gen: &mut Generics) -> TokenStream {
	let gen_types = gen.type_params().into_iter().map(|t| t.ident.clone()).collect::<Vec<_>>();
	let mut done_gen = HashSet::new();

	let mut has_named_fields = false;
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

			let attrs = match has_attrs(&f.attrs).map_err(|e| e.into_compile_error()) {
				Ok(a) => a,
				Err(e) => {
					return Some((e.clone(), e));
				}
			};

			match attrs {
				Attrs::Ignore => None,
				Attrs::With(expr) => {
					Some(construct_debug_calls(f, idx, &mut has_named_fields, Some(&expr)))
				}
				Attrs::None => Some(construct_debug_calls(f, idx, &mut has_named_fields, None)),
			}
		})
		.unzip();

	let (start_fmtr, end_fmtr) = formatter_type(has_named_fields, name);
	quote! {
		let #name { #fields .. } = &self;
		#start_fmtr
		#debug_names
		#end_fmtr
	}
}

fn construct_debug_calls(
	f: &Field,
	idx: usize,
	is_struct_like: &mut bool,
	with_call: Option<&NestedMeta>,
) -> (TokenStream, TokenStream) {
	if let Some(field_name) = &f.ident {
		let field_or_call = if let Some(call) = with_call {
			quote! { & #call (#field_name) }
		} else {
			quote! { #field_name }
		};
		*is_struct_like = true;
		(
			quote! { #field_name, },
			quote! {
				let _ =
					::core::fmt::DebugStruct::field(
						dbg_builder,
						stringify!(#field_name),
						#field_or_call
					);
			},
		)
	} else {
		let field_name = Ident::new(&format!("_self_{}", idx), f.span());
		let idx = syn::Index::from(idx);
		let field_or_call = if let Some(call) = with_call {
			quote! { & #call (#field_name) }
		} else {
			quote! { #field_name }
		};
		(
			quote! {
				#idx: #field_name,
			},
			quote! {
				let _ =
					::core::fmt::DebugTuple::field(
						dbg_builder,
						#field_or_call
					);
			},
		)
	}
}

fn formatter_type(has_named_fields: bool, name: &Ident) -> (TokenStream, TokenStream) {
	if has_named_fields {
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

enum Attrs {
	Ignore,
	With(NestedMeta),
	None,
}

fn has_attrs(attrs: &[Attribute]) -> syn::Result<Attrs> {
	let mut ours = Attrs::None;

	for attr in attrs {
		if attr.path.is_ident("dbg_with") {
			match attr.parse_meta()? {
				Meta::List(meta) => match ours {
					Attrs::None => {
						ours = Attrs::With(meta.nested[0].clone());
					}
					_ => {
						return Err(syn::Error::new_spanned(
							meta,
							"invalid combination of attributes",
						));
					}
				},
				bad => {
					return Err(syn::Error::new_spanned(bad, "unrecognized attribute"));
				}
			}
		// TODO: catch error if this is already set
		} else if attr.path.is_ident("dbg_ignore") {
			ours = Attrs::Ignore;
		}
	}
	Ok(ours)
}
