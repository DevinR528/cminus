crate struct StripEscape<'a> {
    s: std::str::Chars<'a>,
}

impl<'a> StripEscape<'a> {
    crate fn new(s: &'a str) -> Self {
        Self { s: s.chars() }
    }
}

impl<'a> Iterator for StripEscape<'a> {
    type Item = char;

    fn next(&mut self) -> Option<Self::Item> {
        self.s.next().and_then(|c| match c {
            '\\' => Some(match self.s.next() {
                Some('n') => '\n',
                Some('t') => '\t',
                Some('f') => '\r',
                Some('0') => '\0',
                Some('\\') => '\\',
                c => unreachable!("{:?}", c),
            }),
            '"' => self.next(),
            c => Some(c),
        })
    }
}
