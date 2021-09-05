use std::{collections::HashMap, iter::Peekable, ops::BitOr};

use pest::{iterators::Pair, RuleType};

/// Associativity of an [`Operator`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Assoc {
    /// Left `Operator` associativity
    Left,
    /// Right `Operator` associativity
    Right,
}

/// Infix operator used in [`PrecClimber`].
#[derive(Debug)]
pub struct Operator<R: RuleType> {
    rule: R,
    assoc: Assoc,
    next: Option<Box<Operator<R>>>,
}

impl<R: RuleType> Operator<R> {
    pub fn new(rule: R, assoc: Assoc) -> Operator<R> {
        Operator { rule, assoc, next: None }
    }
}

impl<R: RuleType> BitOr for Operator<R> {
    type Output = Self;

    fn bitor(mut self, rhs: Self) -> Self {
        fn assign_next<R: RuleType>(op: &mut Operator<R>, next: Operator<R>) {
            if let Some(ref mut child) = op.next {
                assign_next(child, next);
            } else {
                op.next = Some(Box::new(next));
            }
        }
        assign_next(&mut self, rhs);
        self
    }
}

pub struct PrecClimber<R: RuleType> {
    ops: HashMap<R, (u32, Assoc)>,
}

impl<R: RuleType> PrecClimber<R> {
    pub fn new(ops: Vec<Operator<R>>) -> PrecClimber<R> {
        let ops = ops.into_iter().zip(1..).fold(HashMap::new(), |mut map, (op, prec)| {
            let mut next = Some(op);

            while let Some(op) = next.take() {
                let Operator { rule, assoc, next: op_next } = op;
                {
                    map.insert(rule, (prec, assoc));
                    next = op_next.map(|op| *op);
                }
            }

            map
        });

        PrecClimber { ops }
    }

    pub fn climb<'i, P, F, G, T>(&self, mut pairs: P, mut primary: F, mut infix: G) -> T
    where
        P: Iterator<Item = Pair<'i, R>>,
        F: FnMut(Pair<'i, R>) -> T,
        G: FnMut(T, Pair<'i, R>, T) -> T,
    {
        let lhs = primary(
            pairs.next().expect("precedence climbing requires a non-empty Pairs"),
        );
        self.climb_rec(lhs, 0, &mut pairs.peekable(), &mut primary, &mut infix)
    }

    pub fn climb_rec<'i, P, F, G, T>(
        &self,
        mut lhs: T,
        min_prec: u32,
        pairs: &mut Peekable<P>,
        primary: &mut F,
        infix: &mut G,
    ) -> T
    where
        P: Iterator<Item = Pair<'i, R>>,
        F: FnMut(Pair<'i, R>) -> T,
        G: FnMut(T, Pair<'i, R>, T) -> T,
    {
        while pairs.peek().is_some() {
            let rule =
                pairs.peek().unwrap().clone().into_inner().next().unwrap().as_rule();
            if let Some((op, prec)) = self.ops.get(&rule).and_then(|(prec, _)| {
                if prec >= &min_prec {
                    pairs.next().unwrap().into_inner().next().map(|op| (op, prec))
                } else {
                    None
                }
            }) {
                let mut rhs = primary(pairs.next().expect(
                    "infix operator must be followed by \
                         a primary expression",
                ));

                while pairs.peek().is_some() {
                    let rule = pairs
                        .peek()
                        .unwrap()
                        .clone() // We need the next child
                        // so this long chain is unavoidable unless the parse tree shape changes
                        .into_inner()
                        .next()
                        .unwrap()
                        .as_rule();
                    if let Some(&(new_prec, assoc)) = self.ops.get(&rule) {
                        if new_prec > *prec || assoc == Assoc::Right && new_prec == *prec
                        {
                            rhs = self.climb_rec(rhs, new_prec, pairs, primary, infix);
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                lhs = infix(lhs, op, rhs);
            } else {
                break;
            }
        }

        lhs
    }
}
