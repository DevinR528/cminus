use std::fmt::{self, Display, Write};

use crate::ast::types::{Block, Decl, Expr, Func, Param, Stmt, Ty, Var};

pub trait Visit: Sized {
    fn visit_prog(&mut self, items: &[Decl]) {
        walk_items(self, items)
    }

    fn visit_decl(&mut self, item: &Decl) {
        walk_decl(self, item)
    }

    fn visit_func(&mut self, func: &Func) {
        walk_func(self, func)
    }

    fn visit_var(&mut self, var: &Var) {
        walk_var(self, var)
    }

    fn visit_params(&mut self, params: &[Param]) {
        walk_params(self, params)
    }

    fn visit_ty(&mut self, ty: &Ty) {
        // done
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        walk_stmt(self, stmt)
    }

    fn visit_expr(&mut self, expr: &Expr) {
        walk_expr(self, expr)
    }
}

crate fn walk_items<V: Visit>(visit: &mut V, items: &[Decl]) {
    for item in items {
        visit.visit_decl(item);
    }
}

crate fn walk_decl<V: Visit>(visit: &mut V, item: &Decl) {
    match item {
        Decl::Func(func) => {
            visit.visit_func(func);
        }
        Decl::Var(var) => {
            visit.visit_var(var);
        }
    }
}

crate fn walk_func<V: Visit>(visit: &mut V, func: &Func) {
    let Func { ident, params, stmts, ret } = func;
    // visit.visit_ident(ident);
    visit.visit_params(params);
    visit.visit_ty(ret);
    for stmt in stmts {
        visit.visit_stmt(stmt);
    }
}

crate fn walk_var<V: Visit>(visit: &mut V, var: &Var) {
    // visit.visit_ident(&var.ident);
    visit.visit_ty(&var.ty);
}

crate fn walk_params<V: Visit>(visit: &mut V, params: &[Param]) {
    for Param { ident, ty } in params {
        visit.visit_ty(ty);
    }
}

crate fn walk_stmt<V: Visit>(visit: &mut V, stmt: &Stmt) {
    match stmt {
        Stmt::VarDecl(vars) => {
            for var in vars {
                visit.visit_var(var)
            }
        }
        Stmt::Assign { ident, expr } => {
            // visit.visit_ident(ident);
            visit.visit_expr(expr);
        }
        Stmt::ArrayAssign { ident, expr } => {
            // visit.visit_ident(ident);
            visit.visit_expr(expr);
        }
        Stmt::Call { ident, args } => {
            // visit.visit_ident(ident);
            for expr in args {
                visit.visit_expr(expr);
            }
        }
        Stmt::If { cond, blk: Block { stmts }, els } => {
            visit.visit_expr(cond);
            for stmt in stmts {
                visit.visit_stmt(stmt);
            }
            if let Some(Block { stmts }) = els {
                for stmt in stmts {
                    visit.visit_stmt(stmt);
                }
            }
        }
        Stmt::While { cond, stmt } => {
            visit.visit_expr(cond);
            visit.visit_stmt(stmt);
        }
        Stmt::Read(_) => {
            // variable ident
        }
        Stmt::Write { expr } => visit.visit_expr(expr),
        Stmt::Ret(expr) => visit.visit_expr(expr),
        Stmt::Exit => {}
        Stmt::Block(Block { stmts }) => {
            for stmt in stmts {
                visit.visit_stmt(stmt);
            }
        }
    }
}

crate fn walk_expr<V: Visit>(visit: &mut V, expr: &Expr) {
    match expr {
        Expr::Ident(id) => {
            // visit.visit_ident(id)
        }
        Expr::Array { ident, expr } => {
            // visit.visit_ident(ident);
            visit.visit_expr(expr);
        }
        Expr::Urnary { op, expr } => {
            visit.visit_expr(expr);
        }
        Expr::Binary { op, lhs, rhs } => {
            visit.visit_expr(lhs);
            visit.visit_expr(rhs)
        }
        Expr::Parens(expr) => visit.visit_expr(expr),
        Expr::Call { ident, args } => {
            for expr in args {
                visit.visit_expr(expr);
            }
        }
        Expr::Value(_) => {
            // visit.visit_value(val);
        }
    }
}

#[derive(Default, Debug)]
crate struct DotWalker {
    buf: String,
    node_id: usize,
    prev_id: usize,
}

impl DotWalker {
    crate fn new() -> Self {
        Self { buf: String::from("digraph ast {\n"), ..DotWalker::default() }
    }

    fn walk_deeper<P: Fn(&mut Self), F: Fn(&mut Self)>(&mut self, mut pre: P, mut calls: F) {
        self.node_id += 1;
        pre(self);
        let tmp = self.prev_id;
        self.prev_id = self.node_id;
        calls(self);
        self.prev_id = tmp;
    }
}

impl Display for DotWalker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}}}", self.buf)
    }
}

impl Visit for DotWalker {
    fn visit_prog(&mut self, items: &[Decl]) {
        writeln!(&mut self.buf, "{}[label = PGM, shape = ellipse]", self.node_id);
        for item in items {
            match item {
                Decl::Func(func) => {
                    self.visit_func(func);
                }
                Decl::Var(var) => {
                    self.visit_var(var);
                }
            }
        }
    }

    fn visit_func(&mut self, func: &Func) {
        self.walk_deeper(
            |this| {
                writeln!(
                    this.buf,
                    "{}[label = \"func {}\", shape = ellipse]",
                    this.node_id, func.ident
                );
                writeln!(this.buf, "{} -> {}", this.prev_id, this.node_id);
            },
            |this| {
                for stmt in &func.stmts {
                    this.visit_stmt(stmt);
                }
            },
        );
    }

    fn visit_var(&mut self, var: &Var) {
        writeln!(&mut self.buf, "{}[label = \"var {}\", shape = ellipse]", self.node_id, var.ident);
        writeln!(&mut self.buf, "{} -> {}", self.prev_id, self.node_id);
    }

    fn visit_params(&mut self, params: &[Param]) {
        walk_params(self, params);
    }

    fn visit_ty(&mut self, ty: &Ty) {
        // done
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::VarDecl(vars) => {
                for var in vars {
                    self.node_id += 1;
                    self.visit_var(var);
                }
            }
            Stmt::Assign { ident, expr } => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"assign {}\", shape = ellipse]",
                            this.node_id, ident
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| this.visit_expr(expr),
                );
            }
            Stmt::ArrayAssign { ident, expr } => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"array assign {}\", shape = ellipse]",
                            this.node_id, ident
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| this.visit_expr(expr),
                );
            }
            Stmt::Call { ident, args } => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"call {}\", shape = ellipse]",
                            this.node_id, ident
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| {
                        for expr in args {
                            this.visit_expr(expr);
                        }
                    },
                );
            }
            Stmt::If { cond, blk, els } => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"if call\", shape = ellipse]",
                            this.node_id
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| {
                        this.visit_expr(cond);
                        for stmt in &blk.stmts {
                            this.visit_stmt(stmt);
                        }
                        if let Some(Block { stmts }) = els {
                            for stmt in stmts {
                                this.visit_stmt(stmt);
                            }
                        }
                    },
                );
            }
            Stmt::While { cond, stmt } => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"while loop\", shape = ellipse]",
                            this.node_id
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| {
                        this.visit_expr(cond);
                        this.visit_stmt(stmt);
                    },
                );
            }
            Stmt::Read(ident) => {
                self.node_id += 1;
                writeln!(
                    &mut self.buf,
                    "{}[label = \"read({})\", shape = ellipse]",
                    self.node_id, ident
                );
                writeln!(&mut self.buf, "{} -> {}", self.prev_id, self.node_id);
            }
            Stmt::Write { expr } => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"write call\", shape = ellipse]",
                            this.node_id
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| this.visit_expr(expr),
                );
            }
            Stmt::Ret(expr) => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"return\", shape = ellipse]",
                            this.node_id
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| this.visit_expr(expr),
                );
            }
            Stmt::Exit => {
                self.node_id += 1;
                writeln!(&mut self.buf, "{}[label = \"exit\", shape = ellipse]", self.node_id);
                writeln!(&mut self.buf, "{} -> {}", self.prev_id, self.node_id);
            }
            Stmt::Block(Block { stmts }) => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"block\", shape = ellipse]",
                            this.node_id
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| {
                        for stmt in stmts {
                            this.visit_stmt(stmt);
                        }
                    },
                );
            }
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Ident(name) => {
                self.node_id += 1;
                writeln!(
                    &mut self.buf,
                    "{}[label = \"ident {}\", shape = ellipse]",
                    self.node_id, name
                );
                writeln!(&mut self.buf, "{} -> {}", self.prev_id, self.node_id);
            }
            Expr::Array { ident, expr } => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"array {}\", shape = ellipse]",
                            this.node_id, ident
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| this.visit_expr(expr),
                );
            }
            Expr::Urnary { op, expr } => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"expr UrnOp {:?}\", shape = ellipse]",
                            this.node_id, op
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| this.visit_expr(expr),
                );
            }
            Expr::Binary { op, lhs, rhs } => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"expr BinOp {:?}\", shape = ellipse]",
                            this.node_id, op
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| {
                        this.visit_expr(lhs);
                        this.visit_expr(rhs);
                    },
                );
            }
            Expr::Parens(expr) => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"parenthesis\", shape = ellipse]",
                            this.node_id,
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| this.visit_expr(expr),
                );
            }
            Expr::Call { ident, args } => {
                self.walk_deeper(
                    |this| {
                        writeln!(
                            &mut this.buf,
                            "{}[label = \"call {}\", shape = ellipse]",
                            this.node_id, ident
                        );
                        writeln!(&mut this.buf, "{} -> {}", this.prev_id, this.node_id);
                    },
                    |this| {
                        for expr in args {
                            this.visit_expr(expr);
                        }
                    },
                );
            }
            Expr::Value(val) => {
                self.node_id += 1;
                writeln!(
                    &mut self.buf,
                    "{}[label = \"value {}\", shape = ellipse]",
                    self.node_id, val
                );
                writeln!(&mut self.buf, "{} -> {}", self.prev_id, self.node_id);
            }
        }
    }
}
