use std::{
    cell::Cell,
    fmt::{self, Formatter},
    io::Write,
};

use parking_lot::{RwLock, RwLockReadGuard};
use termcolor::{Ansi, BufferWriter, Color, ColorChoice, ColorSpec, WriteColor};

use crate::{ast::types::Range, typeck::TyCheckRes};

#[derive(Default, Debug)]
crate struct ErrorReport<'input> {
    // TODO: SHIT this is ugly
    /// Errors collected during parsing and type checking.
    errors: RwLock<Vec<Error<'input>>>,
    error_in_current_expr_tree: Cell<bool>,
}

impl<'input> ErrorReport<'input> {
    crate fn push_error(&self, e: Error<'input>) {
        if !self.error_in_current_expr_tree.get() {
            let mut list = self.errors.write();
            list.push(e);
            list.sort_by(|a, b| a.span.cmp(&b.span))
        }
    }

    crate fn is_poisoned(&self) -> bool {
        self.error_in_current_expr_tree.get()
    }

    crate fn poisoned(&self, poisoned: bool) {
        self.error_in_current_expr_tree.set(poisoned);
    }

    crate fn is_empty(&self) -> bool {
        self.errors.read().is_empty()
    }

    crate fn errors(&self) -> RwLockReadGuard<'_, Vec<Error<'input>>> {
        self.errors.read()
    }
}

#[derive(Debug)]
crate struct Error<'input> {
    name: &'input str,
    input: &'input str,
    crate span: Range,
    msg: String,
    help: Option<String>,
}

impl<'input> Error<'input> {
    crate fn error_with_span(tctx: &TyCheckRes<'_, 'input>, span: Range, msg: &str) -> Self {
        Self {
            name: tctx.file_names.get(&span.file_id).expect("error for non existent file"),
            input: tctx.inputs.get(&span.file_id).expect("error for non existent file"),
            span,
            msg: msg.to_owned(),
            help: None,
        }
    }

    crate fn error_from_parts(
        name: &'input str,
        input: &'input str,
        span: Range,
        msg: &str,
    ) -> Self {
        Self { name, input, span, msg: msg.to_owned(), help: None }
    }
}

impl fmt::Display for Error<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (row, col) = calc_line_col(self.span, self.input);

        write!(
            f,
            "{}{}\n  --> {}:{}:{}\n{}{}",
            colorize(Color::Red, "Error: ")?,
            self.msg,
            self.name,
            row,
            col,
            calc_snippet_around(self.span, self.input, row),
            self.help.as_deref().unwrap_or(""), // TODO:
        )
    }
}

fn colorize(color: Color, msg: &str) -> Result<String, fmt::Error> {
    let mut s = vec![];

    let mut colorbuf = Ansi::new(&mut s);
    colorbuf.set_color(ColorSpec::new().set_fg(Some(color))).map_err(|e| fmt::Error)?;
    write!(colorbuf, "{}", msg);
    colorbuf.reset();

    // These are always static strings basically
    Ok(unsafe { String::from_utf8_unchecked(s) })
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
    // This is everything before the snippet starts we walk backwards
    // to take 1 line before the snippet starts
    let pre = input[0..span.start]
        .chars()
        .rev()
        .take_while(|c| {
            // The first new line char
            if *c == '\n' && !first {
                first = true;
                return true;
            // The second
            } else if *c != '\n' {
                return true;
            }
            false
        })
        .collect::<String>();
    // flip it back
    let mut pre = pre.chars().rev().collect::<String>();

    let error_seg = &input[span.start..span.end];
    if error_seg.lines().count() <= 1 {
        pre.push_str(&input[span.start..span.end]);
    }

    // reset flag
    first = false;
    let mut added_to_pre = 0;
    let post = input[span.end..]
        .chars()
        .take_while(|c| {
            if *c == '\n' && !first {
                first = true;
                true
            } else if *c != '\n' && !first {
                pre.push(*c);
                added_to_pre += 1;
                true
            } else {
                *c != '\n'
            }
        })
        .collect::<String>();
    let post = &post[added_to_pre..];

    let error_area = if error_seg.lines().count() <= 1 {
        // Don't pad the underline if the line starts at 0, we used to take the line before
        // which worked accidentally most of the time if the line was just newline or had any space
        // in it
        let pad = if input[span.start.saturating_sub(1)..].starts_with('\n') {
            0
        } else {
            input[..span.start].lines().last().map_or(0, |l| l.len())
        };
        let underline = span.end - span.start;
        format!("\n{}{}", " ".repeat(pad), colorize(Color::Blue, &"^".repeat(underline)).unwrap())
    } else {
        error_seg.to_owned()
    };

    // TODO: bad/wrong algorithm
    let mut adjusted = row - 1;
    let mut past_problem = false;
    let s = format!("{}{}{}", pre, error_area, post);
    let num_pad = (adjusted + 2).to_string().len() + 1;
    s.lines()
        .enumerate()
        .map(|(i, l)| {
            if (row + 1 == adjusted + i) && !past_problem {
                adjusted = adjusted.saturating_sub(1);
                past_problem = true;
                format!("{}|{}\n", " ".repeat(num_pad), l)
            } else {
                let line = adjusted + i;
                let pad = num_pad - line.to_string().len();
                format!("{}{}|{}\n", line, " ".repeat(pad), l)
            }
        })
        .collect::<String>()
}
