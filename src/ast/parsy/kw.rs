macro_rules! keywords {
    ($($tkn:ident: $rep:expr,)*) => {
        pub fn is_keyword(s: &str) -> bool {
            match s {
                // These are not actual keywords so overwrite them
                // to return false
                "auto" => false,
                "catch" => false,
                "default" => false,
                "macro_rules" => false,
                "raw" => false,
                "union" => false,
                $(
                    $rep => true,
                )*
                _ => false,
            }
        }
        pub enum Keywords {
            $($tkn),*
        }
        impl std::convert::TryFrom<&str> for Keywords {
            type Error = $crate::ast::parsy::error::ParseError;

            fn try_from(s: &str) -> std::result::Result<Self, Self::Error> {
                std::result::Result::Ok(match s {
                    $(
                        $rep => Self::$tkn,
                    )*
                    _ => std::result::Result::Err($crate::ast::parsy::error::ParseError::IncorrectToken)?
                })
            }
        }
        impl Keywords {
            pub fn text(&self) -> &'static str {
                match self {
                    $(
                        Self::$tkn => $rep,
                    )*
                }
            }
        }
    };
}

// After modifying this list adjust `is_special`, `is_used_keyword`/`is_unused_keyword`,
// this should be rarely necessary though if the keywords are kept in alphabetic order.
keywords! {
    // Special reserved identifiers used internally for elided lifetimes,
    // unnamed method parameters, crate root module, error recovery etc.
    Empty:              "",
    PathRoot:           "{{root}}",
    DollarCrate:        "$crate",
    Underscore:         "_",

    As:                 "as",
    Break:              "break",
    Const:              "const",
    Continue:           "continue",
    Else:               "else",
    Enum:               "enum",
    False:              "false",
    Fn:                 "fn",
    For:                "for",
    If:                 "if",
    Impl:               "impl",
    In:                 "in",
    Let:                "let",
    Loop:               "loop",
    Match:              "match",
    Mod:                "mod",
    Move:               "move",
    Mut:                "mut",
    Pub:                "pub",
    Ref:                "ref",
    Return:             "return",
    Exit:               "exit",
    Struct:             "struct",
    Trait:              "trait",
    True:               "true",
    Use:                "use",
    While:              "while",
}
