use pest::iterators::{Pair, Pairs};

use crate::{
    ast::{
        precedence::{Assoc, Operator, PrecClimber},
        types::{
            Adt, BinOp, Binding, Block, Decl, Declaration, Enum, Expr, Expression, Field,
            FieldInit, Func, Generic, Impl, MatchArm, Param, Pat, Pattern, Range, Spany, Statement,
            Stmt, Struct, Trait, TraitMethod, Ty, Type, UnOp, Val, Var, Variant,
        },
    },
    Rule,
};

crate fn parse_decl(pair: Pair<'_, Rule>) -> Vec<Declaration> {
    pair.into_inner()
        .flat_map(|decl| {
            let span = to_span(&decl);
            match decl.as_rule() {
                Rule::func_decl => {
                    vec![Decl::Func(parse_func(decl.into_inner())).into_spanned(span)]
                }
                Rule::var_decl => parse_var_decl(decl.into_inner())
                    .into_iter()
                    .map(|var| Decl::Var(var).into_spanned(span))
                    .collect(),
                Rule::adt_decl => {
                    vec![Decl::Adt(parse_adt(decl.into_inner(), span)).into_spanned(span)]
                }
                Rule::trait_decl => {
                    vec![Decl::Trait(parse_trait(decl.into_inner(), span)).into_spanned(span)]
                }
                Rule::trait_impl => {
                    vec![Decl::Impl(parse_impl(decl.into_inner(), span)).into_spanned(span)]
                }
                _ => unreachable!("malformed declaration"),
            }
        })
        .collect()
}

fn parse_adt(struct_: Pairs<Rule>, span: Range) -> Adt {
    match struct_.clone().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        [(Rule::STRUCT, _), (Rule::ident, ident), (Rule::generic, gen), (Rule::LBR, _), fields @ .., (Rule::RBR, _)] => {
            Adt::Struct(Struct {
                ident: ident.as_str().to_string(),
                fields: fields
                    .iter()
                    .map(|(_, p)| {
                        match p
                            .clone()
                            .into_inner()
                            .map(|p| (p.as_rule(), p))
                            .collect::<Vec<_>>()
                            .as_slice()
                        {
                            [(Rule::param, param), (Rule::SC, _)] => {
                                parse_struct_field_decl(param.clone())
                            }
                            _ => unreachable!("malformed struct fields"),
                        }
                    })
                    .collect(),
                generics: parse_generics(gen.clone()),
                span,
            })
        }
        [(Rule::ENUM, _), (Rule::ident, ident), (Rule::generic, gen), (Rule::LBR, _), fields @ .., (Rule::RBR, _)] => {
            Adt::Enum(Enum {
                ident: ident.as_str().to_string(),
                variants: fields
                    .iter()
                    .filter_map(|(r, p)| match r {
                        Rule::variant => Some(parse_variant_decl(p.clone())),
                        Rule::CM => None,
                        _ => unreachable!("malformed variant {}", p.to_json()),
                    })
                    .collect(),
                generics: parse_generics(gen.clone()),
                span,
            })
        }
        _ => unreachable!("malformed function parameter {}", struct_.to_json()),
    }
}

fn parse_struct_field_decl(param: Pair<Rule>) -> Field {
    match param.into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        [(Rule::type_, ty), (Rule::var_name, var)] => {
            let ty = parse_ty(ty.clone());
            let span = to_span(var);
            match var.clone().into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice()
            {
                [(Rule::addrof, addr), (Rule::ident, id), array_parts @ ..] => {
                    let inner_span = (addr.as_span().start()..id.as_span().end()).into();
                    let ty = {
                        let indirection = addr.as_str().matches('*').count();
                        build_recursive_pointer_ty(indirection, ty, inner_span)
                    };
                    if !array_parts.is_empty() {
                        Field {
                            ty: build_recursive_ty(
                                array_parts.iter().filter_map(|(r, p)| match r {
                                    Rule::integer => {
                                        Some(p.as_str().parse().expect("invalid integer"))
                                    }
                                    Rule::LBK | Rule::RBK => None,
                                    _ => unreachable!("malformed array access"),
                                }),
                                ty,
                            ),
                            ident: id.as_str().to_string(),
                            span,
                        }
                    } else {
                        Field { ty, ident: id.as_str().to_string(), span: inner_span }
                    }
                }
                _ => unreachable!("malformed variable"),
            }
        }
        _ => unreachable!("malformed function parameter"),
    }
}

fn parse_variant_decl(variant: Pair<Rule>) -> Variant {
    let span = to_span(&variant);
    match variant.clone().into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        [(Rule::ident, id), content_tuple @ ..] => Variant {
            ident: id.as_str().to_string(),
            types: if content_tuple.is_empty() {
                vec![]
            } else {
                match content_tuple {
                    [(Rule::LP, _), (Rule::type_, ty), rest @ ..] => vec![parse_ty(ty.clone())]
                        .into_iter()
                        .chain(rest.iter().filter_map(|(r, n)| match r {
                            Rule::type_ => Some(parse_ty(n.clone())),
                            Rule::CM | Rule::RP => None,
                            _ => {
                                unreachable!("malformed variant tuple declaration {}", n.to_json())
                            }
                        }))
                        .collect(),
                    _ => unreachable!("malformed variant declaration"),
                }
            },
            span,
        },
        _ => unreachable!("malformed enum variant"),
    }
}

fn parse_trait(trait_: Pairs<Rule>, span: Range) -> Trait {
    match trait_.clone().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        [(Rule::TRAIT, _), (Rule::ident, ident), (Rule::generic, gen), (Rule::LBR, _), (Rule::trait_item, item), (Rule::RBR, _)] => {
            Trait {
                ident: ident.as_str().to_string(),
                method: match item
                    .clone()
                    .into_inner()
                    .map(|p| (p.as_rule(), p))
                    .collect::<Vec<_>>()
                    .as_slice()
                {
                    [(Rule::type_, ty), (Rule::ident, ident), (Rule::generic, gen), (Rule::LP, _), (Rule::param_list, params), (Rule::RP, _), (Rule::SC, sc)] => {
                        TraitMethod::NoBody(Func {
                            ret: parse_ty(ty.clone()),
                            ident: ident.as_str().to_string(),
                            params: params
                                .clone()
                                .into_inner()
                                .filter_map(|param| match param.as_rule() {
                                    Rule::param => Some(parse_param(param)),
                                    Rule::CM => None,
                                    _ => unreachable!("malformed call statement"),
                                })
                                .collect(),
                            stmts: vec![],
                            generics: parse_generics(gen.clone()),
                            span: (ty.as_span().start()..sc.as_span().end()).into(),
                        })
                    }
                    [(Rule::func_decl, func)] => {
                        TraitMethod::Default(parse_func(func.clone().into_inner()))
                    }
                    _ => unreachable!("malformed trait item"),
                },
                generics: parse_generics(gen.clone()),
                span,
            }
        }
        _ => unreachable!("malformed trait declaration {}", trait_.to_json()),
    }
}

fn parse_impl(trait_: Pairs<Rule>, span: Range) -> Impl {
    match trait_.clone().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        [(Rule::IMPL, _), (Rule::ident, ident), (Rule::generic, gen), (Rule::LBR, _), (Rule::func_decl, func), (Rule::RBR, _)] =>
        {
            let type_arguments = parse_generics(gen.clone());
            Impl {
                ident: ident.as_str().to_string(),
                method: {
                    let mut f = parse_func(func.clone().into_inner());
                    f.ident.push_str(&format!(
                        "<{}>",
                        type_arguments
                            .iter()
                            .map(|t| t.val.to_string())
                            .collect::<Vec<_>>()
                            .join(",")
                    ));
                    f
                },
                type_arguments,
                span,
            }
        }
        _ => unreachable!("malformed trait declaration {}", trait_.to_json()),
    }
}

fn parse_param(param: Pair<Rule>) -> Param {
    let span = to_span(&param);
    let Var { ty, ident, span } = parse_var_decl(param.into_inner()).remove(0);
    Param { ty, ident, span }
}

#[rustfmt::skip]
fn parse_func(func: Pairs<Rule>) -> Func {
    // println!("func = {}", func.to_json());
    match func.into_iter().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // int foo(int a, int b) { stmts }
        [(Rule::type_, ty), (Rule::ident, ident), (Rule::generic, gen),
         (Rule::LP, _), (Rule::param_list, params), (Rule::RP, _), (Rule::LBR, _),
         action @ ..,
         (Rule::RBR, rbr)
        ] => {
            Func {
                ret: parse_ty(ty.clone()),
                ident: ident.as_str().to_string(),
                params: params.clone().into_inner().filter_map(|param| match param.as_rule() {
                    Rule::param => Some(parse_param(param)),
                    Rule::CM => None,
                    _ => unreachable!("malformed call statement"),
                }).collect(),
                stmts: action.iter().map(|(r, s)| match r {
                    Rule::stmt => parse_stmt(s.clone()),
                    Rule::var_decl => Stmt::VarDecl(
                        parse_var_decl(s.clone().into_inner())).into_spanned(to_span(s)
                    ),
                    _ => unreachable!("malformed statement"),
                }).collect(),
                generics: parse_generics(gen.clone()),
                span: (ty.as_span().start()..rbr.as_span().end()).into(),
            }
        }
        // int foo() { stmts }
        [(Rule::type_, ty), (Rule::ident, ident), (Rule::generic, gen),
         (Rule::LP, _), (Rule::RP, _), (Rule::LBR, _),
         action @ ..,
         (Rule::RBR, rbr)
        ] => {
            Func {
                ret: parse_ty(ty.clone()),
                ident: ident.as_str().to_string(),
                params: vec![],
                stmts: action.iter().map(|(r, s)| match r {
                    Rule::stmt => parse_stmt(s.clone()),
                    Rule::var_decl => Stmt::VarDecl(
                        parse_var_decl(s.clone().into_inner())
                    ).into_spanned(to_span(s)),
                    _ => unreachable!("malformed statement"),
                }).collect(),
                generics: parse_generics(gen.clone()),
                span: (ty.as_span().start()..rbr.as_span().end()).into(),
            }
        }
        _ => unreachable!("malformed function"),
    }
}

fn parse_stmt(stmt: Pair<Rule>) -> Statement {
    let stmt = stmt.into_inner().next().unwrap();
    let span = to_span(&stmt);
    #[rustfmt::skip]
    match stmt.as_rule() {
        Rule::math_assign => {
            match stmt.clone()
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // x += 1;
                [(Rule::expr, expr), (Rule::SC, _)] => {
                    match parse_expr(expr.clone()).val {
                        Expr::Binary { op, lhs, rhs } => {
                            let rhs_span = rhs.span;
                            Stmt::Assign {
                                lval: *lhs.clone(),
                                rval: Expr::Binary {
                                    op: match op {
                                        BinOp::AddAssign => BinOp::Add,
                                        BinOp::SubAssign => BinOp::Sub,
                                        _ => unreachable!("invalid expression statement {}", stmt.to_json())
                                    },
                                    lhs,
                                    rhs,
                                }.into_spanned(rhs_span),
                            }
                        },
                        meth @ Expr::TraitMeth { .. } => Stmt::TraitMeth(meth.into_spanned(to_span(expr))),
                        _ => todo!("{}", stmt.to_json()),
                    }.into_spanned(span)
                }
                _ => unreachable!("malformed expression statement {}", stmt.to_json()),
            }
        }
        Rule::assign => {
            match stmt.clone()
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // [*]var = [*]expr;
                [(Rule::expr, var), (Rule::ASSIGN, _),
                    (Rule::expr | Rule::struct_assign | Rule::arr_init, expr), (Rule::SC, _)
                ] => {
                    parse_lvalue(var.clone(), expr.clone(), span)
                }
                _ => unreachable!("malformed assignment {}", stmt.to_json()),
            }
        }
        Rule::call_stmt => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // foo(x,y); or foo();
                [(Rule::ident, name), (Rule::type_args, type_args), (Rule::LP, _),
                    args @ ..,
                (Rule::RP, _), (Rule::SC, _)] => {
                    Stmt::Call(
                        Expr::Call {
                            ident: name.as_str().to_string(),
                            args: args.iter().map(|(_, p)| p.clone().into_inner()
                                .filter_map(|arg| match arg.as_rule() {
                                    Rule::expr => Some(parse_expr(arg)),
                                    Rule::CM => None,
                                    _ => unreachable!("malformed call statement"),
                                })
                                .collect::<Vec<_>>()).flatten().collect(),
                            type_args: parse_type_arguments(type_args.clone()),
                        }.into_spanned(span)
                    ).into_spanned(span)
                }
                _ => unreachable!("malformed call statement"),
            }
        }
        Rule::if_stmt => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // if expr { stmts } [ else { stmts }]
                [(Rule::IF, _), (Rule::LP, _), (Rule::expr, expr), (Rule::RP, _),
                    (Rule::block_stmt, block), else_blk @ ..
                ] => {
                    Stmt::If {
                        cond: parse_expr(expr.clone()),
                        blk: parse_block(block.clone()),
                        els: match else_blk {
                            [(Rule::ELSE, _), (Rule::block_stmt, blk)] => {
                                Some(parse_block(blk.clone()))
                            }
                            [] => None,
                            _ => unreachable!("malformed if statement"),
                        },
                    }.into_spanned(span)
                }
                [
                    (Rule::IF, _), (Rule::LP, _), (Rule::expr, expr), (Rule::RP, _), (Rule::SC, sc)
                ] => {
                    Stmt::If {
                        cond: parse_expr(expr.clone()),
                        blk: Block { span: to_span(sc), stmts: vec![]},
                        els: None,
                    }.into_spanned(span)
                }
                _ => unreachable!("malformed assignment"),
            }
        }
        Rule::match_stmt => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // match expr { arms }
                [(Rule::MATCH, _), (Rule::expr, expr), (Rule::LBR, _), arms @ .., (Rule::RBR, _),] => {
                    Stmt::Match {
                        expr: parse_expr(expr.clone()),
                        arms: arms.iter().filter_map(|(_, a)| match a.clone().into_inner()
                            .map(|p| (p.as_rule(), p))
                            .collect::<Vec<_>>()
                            .as_slice() {
                                [
                                    (Rule::expr, pat) | (Rule::arr_init, pat), (Rule::ARROW, _),
                                    (Rule::block_stmt, blk), opt_comma @ ..
                                ] if opt_comma.len() <= 1 => Some(MatchArm {
                                    pat: parse_match_arm_pat(parse_expr(pat.clone()).val, to_span(pat)),
                                    blk: parse_block(blk.clone()),
                                    span: to_span(a),
                                }),
                                _ => unreachable!("malformed match arm {}", a.to_json())
                            })
                            .collect(),
                    }.into_spanned(span)
                }
                _ => unreachable!("malformed assignment"),
            }
        }
        Rule::while_stmt => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // while (expr) { stmts }
                [(Rule::WHILE, _), (Rule::LP, _), (Rule::expr, expr), (Rule::RP, _),
                    (Rule::stmt, stmt),
                ] => {
                    Stmt::While {
                        cond: parse_expr(expr.clone()),
                        // This will mostly be `Stmt::Block(..)` but sometimes just
                        // assignment i = i + 1;
                        stmt: box parse_stmt(stmt.clone()),
                    }.into_spanned(span)
                }
                _ => unreachable!("malformed assignment"),
            }
        }
        Rule::io_stmt => {
            let span = to_span(&stmt);
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // read(var);
                [(Rule::READ, _), (Rule::LP, _),
                    (Rule::expr, expr),
                (Rule::RP, _), (Rule::SC, _)] => {
                    Stmt::Read(parse_expr(expr.clone()))
                }
                // write(expr);
                [(Rule::WRITE, _), (Rule::LP, _),
                    (arg_rule, arg),
                (Rule::RP, _), (Rule::SC, _)] => {
                    Stmt::Write {
                        expr: match arg_rule {
                            Rule::expr => parse_expr(arg.clone()),
                            Rule::string => Expr::Value(
                                Val::Str(arg.as_str().replace("\"", "")).into_spanned(to_span(arg))
                            ).into_spanned(to_span(arg)),
                            _ => unreachable!("malformed write statement")
                        },
                    }
                }
                _ => unreachable!("malformed IO statement"),
            }.into_spanned(span)
        }
        Rule::ret_stmt => {
            match stmt
                .into_inner()
                .map(|p| (p.as_rule(), p))
                .collect::<Vec<_>>()
                .as_slice()
            {
                // return expr;
                [(Rule::RETURN, _), (Rule::expr, expr), (Rule::SC, _)] => {
                    Stmt::Ret(parse_expr(expr.clone())).into_spanned(span)
                }
                _ => unreachable!("malformed return statement"),
            }
        }
        // exit;
        Rule::exit_stmt => Stmt::Exit.into_spanned(span),
        // { stmts }
        Rule::block_stmt => Stmt::Block(parse_block(stmt)).into_spanned(span),
        _ => unreachable!("malformed statement"),
    }
}

fn parse_match_arm_pat(pat: Expr, span: Range) -> Pattern {
    match pat {
        Expr::Value(v) => Pat::Bind(Binding::Value(v)),
        Expr::Ident(id) => Pat::Bind(Binding::Wild(id)),
        Expr::EnumInit { ident, variant, items } => Pat::Enum {
            ident,
            variant,
            items: items
                .into_iter()
                .map(|e| {
                    let inner_span = e.span;
                    parse_match_arm_pat(e.val, inner_span).val
                })
                .collect(),
        },
        Expr::StructInit { name, fields } => todo!(),
        Expr::ArrayInit { items } => Pat::Array {
            size: items.len(),
            items: items
                .into_iter()
                .map(|e| {
                    let inner_span = e.span;
                    parse_match_arm_pat(e.val, inner_span).val
                })
                .collect(),
        },
        Expr::Deref { .. }
        | Expr::AddrOf(_)
        | Expr::Array { .. }
        | Expr::Urnary { .. }
        | Expr::Binary { .. }
        | Expr::Parens(_)
        | Expr::Call { .. }
        | Expr::TraitMeth { .. }
        | Expr::FieldAccess { .. } => todo!(),
    }
    .into_spanned(span)
}

fn parse_generics(generics: Pair<Rule>) -> Vec<Type> {
    let span = to_span(&generics);
    generics
        .into_inner()
        .map(|g| (g.as_rule(), g))
        .filter_map(|(r, g)| match r {
            Rule::CM => None,
            Rule::type_ => Some(parse_ty(g)),
            _ => unreachable!("malformed generic type in declaration"),
        })
        .collect()
}

fn parse_ty(ty: Pair<Rule>) -> Type {
    let span = to_span(&ty);
    match ty.clone().into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // int x,y,z;
        [(rule, ty), (Rule::addrof, addr)] => {
            let indirection = addr.as_str().matches('*').count();
            let t = match rule {
                Rule::INT => Ty::Int,
                Rule::FLOAT => Ty::Float,
                Rule::CHAR => Ty::Char,
                Rule::BOOL => Ty::Bool,
                Rule::VOID if addr.as_str().is_empty() => Ty::Void,
                Rule::ident => Ty::Generic { ident: ty.as_str().to_string(), bound: None },
                Rule::bound => match ty.clone().into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
                    [(Rule::ident, gen), (Rule::ident, bound)] => Ty::Generic { ident: gen.as_str().to_string(), bound: Some(bound.as_str().to_string()) },
                    _ => unreachable!("malformed generic parameter bound")
                },
                _ => unreachable!("malformed addrof type {}", ty.to_json()),
            }
            .into_spanned(to_span(ty));
            build_recursive_pointer_ty(indirection, t, span).val
        }
        [(kw @ Rule::STRUCT | kw @ Rule::ENUM, s), (Rule::ident, ident), (Rule::generic, gen), (Rule::addrof, addr)] => {
            let indirection = addr.as_str().matches('*').count();
            build_recursive_pointer_ty(
                indirection,
                if Rule::STRUCT == *kw {
                    Ty::Struct { ident: ident.as_str().to_string(), gen: parse_generics(gen.clone()) }
                    .into_spanned(s.as_span().start()..addr.as_span().end())
                } else {
                    Ty::Enum { ident: ident.as_str().to_string(), gen: parse_generics(gen.clone()) }
                    .into_spanned(s.as_span().start()..addr.as_span().end())
                },
                (s.as_span().start()..addr.as_span().end()).into(),
            )
            .val
        }
        _ => unreachable!("malformed type {}", ty.to_json()),
    }
    .into_spanned(span)
}

fn parse_block(blk: Pair<Rule>) -> Block {
    let span = to_span(&blk);
    match blk.into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // { stmt* }
        [(Rule::LBR, _), stmts @ .., (Rule::RBR, _)] => {
            Block { stmts: stmts.iter().map(|(_, s)| parse_stmt(s.clone())).collect(), span }
        }
        _ => unreachable!("malformed function"),
    }
}

fn parse_lvalue(var: Pair<Rule>, expr: Pair<'_, Rule>, span: Range) -> Statement {
    let lval = parse_expr(var);
    valid_lval(&lval.val).unwrap();
    Stmt::Assign { lval, rval: parse_expr(expr) }.into_spanned(span)
}

fn valid_lval(ex: &Expr) -> Result<(), String> {
    let mut stack = vec![ex];

    // validate lValue
    while let Some(val) = stack.pop() {
        match val {
            Expr::Parens(expr) | Expr::Deref { expr, .. } | Expr::AddrOf(expr) => {
                stack.push(&expr.val)
            }
            Expr::FieldAccess { lhs, rhs } => {
                stack.push(&lhs.val);
                stack.push(&rhs.val);
            }
            // Valid expressions that are not recursive
            Expr::Ident(..) | Expr::Array { .. } => {}
            Expr::Call { .. } => {
                return Err("call expression is not a valid lvalue".to_owned());
            }
            Expr::TraitMeth { .. } => {
                return Err("call expression is not a valid lvalue".to_owned());
            }
            Expr::Urnary { .. } => {
                return Err("urnary expression is not a valid lvalue".to_owned());
            }
            Expr::Binary { .. } => {
                return Err("binary expression is not a valid lvalue".to_owned());
            }
            Expr::StructInit { .. } => {
                return Err("struct init expression is not a valid lvalue".to_owned());
            }
            Expr::EnumInit { .. } => {
                return Err("enum init expression is not a valid lvalue".to_owned());
            }
            Expr::ArrayInit { .. } => {
                return Err("array init expression is not a valid lvalue".to_owned());
            }
            Expr::Value(_) => {
                return Err("value is not a valid lvalue".to_owned());
            }
        }
    }
    Ok(())
}

fn build_recursive_ty<I: Iterator<Item = usize>>(mut dims: I, base_ty: Type) -> Type {
    if let Some(size) = dims.next() {
        let span = base_ty.span;
        Ty::Array { size, ty: box build_recursive_ty(dims, base_ty) }.into_spanned(span)
    } else {
        base_ty
    }
}

fn build_recursive_pointer_ty(mut indir: usize, base_ty: Type, outer_span: Range) -> Type {
    if indir > 0 {
        let span = base_ty.span;
        Ty::Ptr(box build_recursive_pointer_ty(indir - 1, base_ty, outer_span))
            .into_spanned(outer_span)
    } else {
        base_ty
    }
}

fn parse_var_decl(var: Pairs<Rule>) -> Vec<Var> {
    match var.clone().into_iter().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        // int x,y,z;
        [(Rule::type_, ty), (Rule::var_name, var_name), names @ ..] => {
            let ty = parse_ty(ty.clone());
            let parse_var_array = |var: &Pair<Rule>| {
                let span = to_span(var);
                match var
                    .clone()
                    .into_inner()
                    .map(|p| (p.as_rule(), p))
                    .collect::<Vec<_>>()
                    .as_slice()
                {
                    [(Rule::addrof, addr), (Rule::ident, id), array_parts @ ..] => {
                        let inner_span = (addr.as_span().start()..id.as_span().end()).into();
                        let ty = {
                            let indirection = addr.as_str().matches('*').count();
                            build_recursive_pointer_ty(indirection, ty.clone(), inner_span)
                        };
                        if !array_parts.is_empty() {
                            Var {
                                ty: build_recursive_ty(
                                    array_parts.iter().filter_map(|(r, p)| match r {
                                        Rule::integer => {
                                            Some(p.as_str().parse().expect("invalid integer"))
                                        }
                                        Rule::LBK | Rule::RBK => None,
                                        _ => unreachable!("malformed array access"),
                                    }),
                                    ty,
                                ),
                                ident: id.as_str().to_string(),
                                span,
                            }
                        } else {
                            Var { ty, ident: id.as_str().to_string(), span: inner_span }
                        }
                    }
                    _ => unreachable!("malformed variable"),
                }
            };

            vec![parse_var_array(var_name)]
                .into_iter()
                .chain(names.iter().filter_map(|(r, n)| match r {
                    Rule::var_name => Some(parse_var_array(n)),
                    Rule::CM | Rule::SC => None,
                    _ => unreachable!("malformed variable declaration"),
                }))
                .collect()
        }
        _ => unreachable!("malformed function {:?}", var.map(|p| p.as_rule()).collect::<Vec<_>>()),
    }
}

fn parse_expr(expr: Pair<Rule>) -> Expression {
    // println!("{}", expr.to_json());
    let climber = PrecClimber::new(vec![
        // +=, -=
        Operator::new(Rule::ADDASSIGN, Assoc::Left) | Operator::new(Rule::SUBASSIGN, Assoc::Left),
        // ||
        Operator::new(Rule::OR, Assoc::Left),
        // &&
        Operator::new(Rule::AND, Assoc::Left),
        // |
        Operator::new(Rule::BOR, Assoc::Left),
        // ^
        Operator::new(Rule::BXOR, Assoc::Left),
        // &
        Operator::new(Rule::BAND, Assoc::Left),
        // ==, !=
        Operator::new(Rule::EQ, Assoc::Left) | Operator::new(Rule::NE, Assoc::Left),
        //  >, >=, <, <=
        Operator::new(Rule::GT, Assoc::Left)
            | Operator::new(Rule::GE, Assoc::Left)
            | Operator::new(Rule::LT, Assoc::Left)
            | Operator::new(Rule::LE, Assoc::Left),
        // <<, >>
        Operator::new(Rule::BLSF, Assoc::Left) | Operator::new(Rule::BRSF, Assoc::Left),
        // +, -
        Operator::new(Rule::PLUS, Assoc::Left) | Operator::new(Rule::SUB, Assoc::Left),
        // *, /
        Operator::new(Rule::MUL, Assoc::Left)
            | Operator::new(Rule::DIV, Assoc::Left)
            | Operator::new(Rule::REM, Assoc::Left),
        // ., ->
        Operator::new(Rule::DOT, Assoc::Left) | Operator::new(Rule::ARROW, Assoc::Left),
    ]);

    consume(expr, &climber, true)
}

fn consume(expr: Pair<'_, Rule>, climber: &PrecClimber<Rule>, first: bool) -> Expression {
    let primary = |p: Pair<'_, _>| consume(p, climber, false);
    let infix = |lhs: Expression, op: Pair<Rule>, rhs: Expression| {
        let span = lhs.span.start..rhs.span.end;
        let expr = match op.as_rule() {
            Rule::DOT => match &lhs.val {
                Expr::Deref { indir, expr: inner } => Expr::Deref {
                    indir: *indir,
                    expr: box Expr::FieldAccess { lhs: inner.clone(), rhs: box rhs }
                        .into_spanned(inner.span.start..span.end),
                },
                _ => Expr::FieldAccess { lhs: box lhs, rhs: box rhs },
            },
            Rule::ARROW => match &lhs.val {
                Expr::Deref { indir, expr: inner } => Expr::Deref {
                    indir: *indir,
                    expr: box Expr::FieldAccess {
                        lhs: box Expr::Deref { indir: 1, expr: inner.clone() }
                            .into_spanned(span.start..rhs.span.start),
                        rhs: box rhs,
                    }
                    .into_spanned(inner.span.start..span.end),
                },
                _ => Expr::FieldAccess {
                    lhs: box Expr::Deref { indir: 1, expr: box lhs }
                        .into_spanned(span.start..rhs.span.start),
                    rhs: box rhs,
                },
            },

            Rule::MUL => Expr::Binary { op: BinOp::Mul, lhs: box lhs, rhs: box rhs },
            Rule::DIV => Expr::Binary { op: BinOp::Div, lhs: box lhs, rhs: box rhs },
            Rule::REM => Expr::Binary { op: BinOp::Rem, lhs: box lhs, rhs: box rhs },

            Rule::PLUS => Expr::Binary { op: BinOp::Add, lhs: box lhs, rhs: box rhs },
            Rule::SUB => Expr::Binary { op: BinOp::Sub, lhs: box lhs, rhs: box rhs },

            Rule::BLSF => Expr::Binary { op: BinOp::LeftShift, lhs: box lhs, rhs: box rhs },
            Rule::BRSF => Expr::Binary { op: BinOp::RightShift, lhs: box lhs, rhs: box rhs },

            Rule::GT => Expr::Binary { op: BinOp::Gt, lhs: box lhs, rhs: box rhs },
            Rule::GE => Expr::Binary { op: BinOp::Ge, lhs: box lhs, rhs: box rhs },
            Rule::LT => Expr::Binary { op: BinOp::Lt, lhs: box lhs, rhs: box rhs },
            Rule::LE => Expr::Binary { op: BinOp::Le, lhs: box lhs, rhs: box rhs },

            Rule::EQ => Expr::Binary { op: BinOp::Eq, lhs: box lhs, rhs: box rhs },
            Rule::NE => Expr::Binary { op: BinOp::Ne, lhs: box lhs, rhs: box rhs },

            Rule::BAND => Expr::Binary { op: BinOp::BitAnd, lhs: box lhs, rhs: box rhs },
            Rule::BXOR => Expr::Binary { op: BinOp::BitXor, lhs: box lhs, rhs: box rhs },
            Rule::BOR => Expr::Binary { op: BinOp::BitOr, lhs: box lhs, rhs: box rhs },

            Rule::AND => Expr::Binary { op: BinOp::And, lhs: box lhs, rhs: box rhs },
            Rule::OR => Expr::Binary { op: BinOp::Or, lhs: box lhs, rhs: box rhs },

            Rule::ADDASSIGN => Expr::Binary { op: BinOp::AddAssign, lhs: box lhs, rhs: box rhs },
            Rule::SUBASSIGN => Expr::Binary { op: BinOp::SubAssign, lhs: box lhs, rhs: box rhs },
            _ => unreachable!(),
        };
        expr.into_spanned(span)
    };

    let span = to_span(&expr);
    enum ExprKind {
        Deref(usize),
        AddrOf,
        None,
    }
    let mut wrapper = ExprKind::None;
    let ex = match expr.as_rule() {
        Rule::expr => climber.climb(expr.into_inner(), primary, infix),
        Rule::op => climber.climb(expr.into_inner(), primary, infix),
        Rule::term => {
            match expr.into_inner().map(|r| (r.as_rule(), r)).collect::<Vec<_>>().as_slice() {
                // x,y[],z.z
                [(Rule::variable, var)] => {
                    let span = to_span(var);
                    match var
                        .clone()
                        .into_inner()
                        .map(|p| (p.as_rule(), p))
                        .collect::<Vec<_>>()
                        .as_slice()
                    {
                        [(Rule::deref, deref), (Rule::ident, ident)] => {
                            let indir = deref.as_str();
                            if indir == "&" {
                                wrapper = ExprKind::AddrOf;
                            } else {
                                let indirection = indir.matches('*').count();
                                if indirection > 0 {
                                    wrapper = ExprKind::Deref(indirection);
                                }
                            }
                            Expr::Ident(ident.as_str().to_string()).into_spanned(to_span(ident))
                        }
                        [(Rule::deref, deref), (Rule::ident, ident), (Rule::LBK, _), (Rule::expr, expr), (Rule::RBK, _), rest @ ..] =>
                        {
                            let indir = deref.as_str();

                            let arr = Expr::Array {
                                ident: ident.as_str().to_string(),
                                exprs: vec![parse_expr(expr.clone())]
                                    .into_iter()
                                    .chain(rest.iter().filter_map(|(r, p)| match r {
                                        Rule::LBK | Rule::RBK => None,
                                        Rule::expr => Some(parse_expr(p.clone())),
                                        _ => unreachable!("malformed multi-dim array"),
                                    }))
                                    .collect(),
                            };
                            if indir == "&" {
                                wrapper = ExprKind::AddrOf;
                            } else {
                                let indirection = indir.matches('*').count();
                                if indirection > 0 {
                                    wrapper = ExprKind::Deref(indirection);
                                }
                            }

                            arr.into_spanned(ident.as_span().start()..span.end)
                        }
                        _ => unreachable!("malformed variable name {}", var.to_json()),
                    }
                }
                // 1,true,'a', "string"
                [(Rule::const_, konst)] => parse_const(konst.clone()).into_spanned(span),
                // call() | call::<type>()
                [(Rule::ident, ident), (Rule::type_args, type_args), (Rule::LP, _), arg_list @ .., (Rule::RP, _)] =>
                {
                    Expr::Call {
                        ident: ident.as_str().to_string(),
                        args: if let [(Rule::arg_list, args)] = arg_list {
                            args.clone()
                                .into_inner()
                                .filter_map(|p| match p.as_rule() {
                                    Rule::expr => Some(
                                        // if there are expressions as arguments they are
                                        // ordered on their own `1 + call(2*3)`  the 2*3
                                        // has nothing to do with 1 + whatever
                                        climber.climb(p.clone().into_inner(), primary, infix),
                                    ),
                                    Rule::arr_init => Some(parse_expr(p)),
                                    Rule::struct_assign => Some(parse_expr(p)),
                                    Rule::CM => None,
                                    _ => unreachable!("malformed arguments in call {:?}", arg_list),
                                })
                                .collect::<Vec<_>>()
                        } else {
                            vec![]
                        },
                        type_args: parse_type_arguments(type_args.clone()),
                    }
                    .into_spanned(span)
                }
                // <<T>::trait>()
                [(Rule::generic, gen_args), (Rule::ident, trait_), (Rule::LP, _), arg_list @ .., (Rule::RP, _)] =>
                {
                    Expr::TraitMeth {
                        trait_: trait_.as_str().to_string(),
                        args: if let [(Rule::arg_list, args)] = arg_list {
                            args.clone()
                                .into_inner()
                                .filter_map(|p| match p.as_rule() {
                                    Rule::expr => Some(
                                        // if there are expressions as arguments they are
                                        // ordered on their own `1 + call(2*3)`  the 2*3
                                        // has nothing to do with 1 + whatever
                                        climber.climb(p.clone().into_inner(), primary, infix),
                                    ),
                                    Rule::CM => None,
                                    _ => unreachable!("malformed arguments in call {:?}", arg_list),
                                })
                                .collect::<Vec<_>>()
                        } else {
                            vec![]
                        },
                        type_args: parse_generics(gen_args.clone()),
                    }
                    .into_spanned(span)
                }
                // [! | ~]expr
                [(Rule::logic_bit, logbit), (Rule::expr, expr)] => {
                    let inner = climber.climb(expr.clone().into_inner(), primary, infix);
                    Expr::Urnary {
                        op: if logbit.as_str() == "!" { UnOp::Not } else { UnOp::OnesComp },
                        expr: box inner,
                    }
                    .into_spanned(span)
                }
                // (1 + 1)
                [(Rule::LP, _), (Rule::expr, expr), (Rule::RP, _)] => {
                    let inner = climber.climb(expr.clone().into_inner(), primary, infix);
                    Expr::Parens(box inner).into_spanned(span)
                }
                [(Rule::enum_init, expr)] => {
                    let span = to_span(expr);
                    match expr
                        .clone()
                        .into_inner()
                        .map(|r| (r.as_rule(), r))
                        .collect::<Vec<_>>()
                        .as_slice()
                    {
                        [(Rule::ident, enum_name), (Rule::ident, variant), items @ ..] => {
                            Expr::EnumInit {
                                ident: enum_name.as_str().to_string(),
                                variant: variant.as_str().to_string(),
                                items: items
                                    .iter()
                                    .filter_map(|(r, p)| match r {
                                        Rule::expr => Some(parse_expr(p.clone())),
                                        Rule::CM | Rule::LP | Rule::RP => None,
                                        _ => unreachable!(
                                            "malformed item in enum initializer {}",
                                            p.to_json()
                                        ),
                                    })
                                    .collect(),
                            }
                            .into_spanned(span)
                        }
                        _ => unreachable!("malformed enum assignment {}", expr.to_json()),
                    }
                }
                _ => unreachable!("malformed expression"),
            }
        }
        Rule::struct_assign => {
            let span = to_span(&expr);
            match expr.into_inner().map(|r| (r.as_rule(), r)).collect::<Vec<_>>().as_slice() {
                [(Rule::ident, name), (Rule::LBR, _), fields @ .., (Rule::RBR, _)] => {
                    Expr::StructInit {
                        name: name.as_str().to_string(),
                        fields: fields
                            .iter()
                            .filter_map(|(r, p)| match r {
                                Rule::field_expr => Some(parse_field_init(p.clone())),
                                Rule::CM => None,
                                _ => unreachable!("malformed struct initializer"),
                            })
                            .collect(),
                    }
                    .into_spanned(span)
                }
                _ => unreachable!("malformed struct assignment"),
            }
        }
        Rule::arr_init => {
            let span = to_span(&expr);
            match expr.into_inner().map(|r| (r.as_rule(), r)).collect::<Vec<_>>().as_slice() {
                [(Rule::LBR, _), fields @ .., (Rule::RBR, _)] => Expr::ArrayInit {
                    items: fields
                        .iter()
                        .filter_map(|(r, p)| match r {
                            Rule::arr_init | Rule::expr => Some(parse_expr(p.clone())),
                            Rule::CM => None,
                            _ => unreachable!("malformed array initializer"),
                        })
                        .collect(),
                }
                .into_spanned(span),
                _ => unreachable!("malformed array assignment"),
            }
        }
        err => unreachable!("{:?}", err),
    };
    match wrapper {
        ExprKind::Deref(indir) => Expr::Deref { indir, expr: box ex }.into_spanned(span),
        ExprKind::AddrOf => Expr::AddrOf(box ex).into_spanned(span),
        ExprKind::None => ex,
    }
}

fn parse_const(konst: Pair<'_, Rule>) -> Expr {
    let inner_span = to_span(&konst);
    match konst.clone().into_inner().next().unwrap().as_rule() {
        Rule::integer => {
            Expr::Value(Val::Int(konst.as_str().parse().unwrap()).into_spanned(inner_span))
        }
        Rule::decimal => {
            Expr::Value(Val::Float(konst.as_str().parse().unwrap()).into_spanned(inner_span))
        }
        Rule::charstr => {
            let ch = konst.as_str().replace('\'', "").chars().collect::<Vec<_>>();
            if let [c] = ch.as_slice() {
                Expr::Value(Val::Char(*c).into_spanned(inner_span))
            } else {
                unreachable!("multiple char char is not allowed")
            }
        }
        Rule::string => Expr::Value(Val::Str(konst.as_str().to_string()).into_spanned(inner_span)),
        Rule::TRUE => Expr::Value(Val::Bool(true).into_spanned(inner_span)),
        Rule::FALSE => Expr::Value(Val::Bool(false).into_spanned(inner_span)),
        r => unreachable!("malformed const expression {:?}", r),
    }
}

fn parse_field_init(field: Pair<Rule>) -> FieldInit {
    match field.clone().into_inner().map(|p| (p.as_rule(), p)).collect::<Vec<_>>().as_slice() {
        [(Rule::ident, ident), (Rule::COLON, _), (Rule::expr, expr)] => FieldInit {
            ident: ident.as_str().to_string(),
            init: parse_expr(expr.clone()),
            span: (ident.as_span().start()..expr.as_span().end()).into(),
        },
        [(Rule::ident, ident), (Rule::COLON, _), (Rule::arr_init, expr)] => FieldInit {
            ident: ident.as_str().to_string(),
            init: parse_expr(expr.clone()),
            span: (ident.as_span().start()..expr.as_span().end()).into(),
        },
        [(Rule::ident, ident), (Rule::COLON, _), (Rule::struct_assign, expr)] => FieldInit {
            ident: ident.as_str().to_string(),
            init: parse_expr(expr.clone()),
            span: (ident.as_span().start()..expr.as_span().end()).into(),
        },
        _ => unreachable!("malformed struct fields {}", field.to_json()),
    }
}

fn parse_type_arguments(args: Pair<Rule>) -> Vec<Type> {
    parse_generics(args)
}

fn to_span(p: &Pair<Rule>) -> Range {
    let sp = p.as_span();
    (sp.start()..sp.end()).into()
}
