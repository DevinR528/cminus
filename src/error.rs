use std::fmt;

use crate::{ast::types::Range, typeck::TyCheckRes};

#[derive(Debug)]
crate struct Error<'input> {
    name: &'input str,
    input: &'input str,
    span: Range,
    msg: String,
    help: Option<String>,
}

impl<'input> Error<'input> {
    crate fn error_with_span(tctx: &TyCheckRes<'_, 'input>, span: Range, msg: &str) -> Self {
        Self { name: tctx.name, input: tctx.input, span, msg: msg.to_owned(), help: None }
    }
}

impl fmt::Display for Error<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (row, col) = calc_line_col(self.span, self.input);
        write!(
            f,
            "error: {}\n  --> {}:{}:{}\n{}",
            self.msg,
            self.name,
            row,
            col,
            calc_snippet_around(self.span, self.input, row)
        )
    }
}

/// Returns the `(row, column) of the span based on `input`.
fn calc_line_col(span: Range, input: &str) -> (usize, usize) {
    let mut col = 1;
    let mut row = 1;
    for c in input.chars().take(span.start) {
        if c == '\n' {
            col = 1;
            row += 1;
            continue;
        }
        // skip windows lines line feed char
        if c == '\r' {
            continue;
        }
        col += 1;
    }
    (row, col)
}

fn calc_snippet_around(span: Range, input: &str, row: usize) -> String {
    let mut first = false;
    let pre = input[0..span.start]
        .chars()
        .rev()
        .take_while(|c| {
            if *c == '\n' && !first {
                first = true;
                true
            } else {
                *c != '\n'
            }
        })
        .collect::<String>();
    // flip it back
    let pre = pre.chars().rev().collect::<String>();
    // reset flag
    first = false;
    let post = input[span.end..]
        .chars()
        .take_while(|c| {
            if *c == '\n' && !first {
                first = true;
                true
            } else {
                *c != '\n'
            }
        })
        .collect::<String>();
    let area = &input[span.start..span.end];
    let error_area = if area.lines().count() <= 1 {
        let pre_pad = pre.chars().rev().take_while(|c| *c != '\n').count();
        let pad = area.chars().filter(|c| c.is_whitespace()).count() + pre_pad;
        let underline = span.end - span.start;
        format!("{}\n{}{}", area, " ".repeat(pre_pad), "^".repeat(underline))
    } else {
        area.to_owned()
    };
    // TODO: bad/wrong algorithm
    let mut adjusted = row - 1;
    format!("{}{}{}", pre, error_area, post)
        .lines()
        .enumerate()
        .map(|(i, l)| {
            if row + 1 == adjusted + i {
                format!("  |{}\n", l)
            } else {
                format!("{} |{}\n", i + adjusted, l)
            }
        })
        .collect::<String>()
}
