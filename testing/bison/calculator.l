%{

#include "parser.hpp"
#include <iostream>

#define YY_USER_ACTION yylloc.first_line = yylloc.last_line = yylineno;
%}

%option noyywrap

%%

"+" { return ADD; }
"-" { return SUB; }
"*" { return MUL; }
"/" { return DIV; }
"abs" { return ABS; }
"(" { return OPEN_PAREN; }
")" { return CLOSE_PAREN; }
(\.[0-9]+)|([0-9]+(\.[0-9]*)?([eE][+-]?[0-9]+)?)  { yylval.dval = atof(yytext); return NUMBER; }
\n { return EOL; }
[ \t]+ {}
. { std::cerr << "Unrecognized symbol " << yytext << " at line " << yylloc.first_line << "\n"; exit(1); }

%%