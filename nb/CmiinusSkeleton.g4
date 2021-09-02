grammar Cminus;

options {
	language = ;
}


@header { 
}

program: ;


CHAR: 'char';
ELSE: 'else';
EXIT: 'exit';
FLOAT: 'float';
IF: 'if';
INT: 'int';
READ: 'read';
RETURN: 'return';
VOID: 'void';
WHILE: 'while';
WRITE: 'write';

AND: '&&';
ASSIGN: '=';
CM: ',';
DIVIDE: '/';
DOT: '.';
DQ: '"';
EQ: '==';
GE: '>=';
GT: '>';
LBK: '[';
LBR: '{';
LE: '<=';
LP: '(';
LT: '<';
MINUS: '-';
NE: '!=';
NOT: '!';
OR: '||';
PLUS: '+';
RBK: ']';
RBR: '}';
RP: ')';
SC: ';';
SQ: '\'';
TIMES: '*';

fragment
LETTER: ('a'..'z' | 'A'..'Z');

fragment
DIGIT: '0'..'9';

ID: LETTER (LETTER | DIGIT)*;

fragment
POSITIVE: '1' ..'9';

INTCON: (POSITIVE DIGIT*) | '0';

FLOATCON: INTCON DOT DIGIT*;

CHARCON: SQ .? SQ;

COMMENT: '/*' .*? '*/' -> channel(HIDDEN);

WS: ( ' ' | '\t' | '\r' | '\n') -> skip;

STRING: DQ .*? DQ;
