/// for header
%code requires{
    class Data;
}

/// for cpp source
%{

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include "lexer.hpp"

void yyerror(const char* msg);

class Data {
    public:
        int data;
        Data() : data(0) {}
        ~Data() {}
};

%}

%union{
    double dval;
    int ival;
    Data* cval;
}

%locations

%start cal
%token ADD SUB MUL DIV ABS OPEN_PAREN CLOSE_PAREN EOL
%token <dval> NUMBER
%type <dval> expr
%type <cval> cal
%type <dval> ufactor
%type <dval> factor
%type <dval> rfactor

%%

cal : { $$ = new Data(); }
    | cal expr EOL { $$ = $1; $1->data++; std::cout << $2 << "\n"; }
    ;

expr : ufactor { $$ = $1; }
    | expr ADD ufactor { $$ = $1 + $3; }
    | expr SUB ufactor { $$ = $1 - $3; }
    ;

ufactor : factor { $$ = $1; }
    | SUB factor { $$ = -$2; }
    | ABS factor { $$ = $2 >= 0 ? $2 : -$2; }
    ;

factor : rfactor { $$ = $1; }
    | factor MUL rfactor { $$ = $1 * $3; }
    | factor DIV rfactor { if ($3 > -1e-10 && $3 < 1e-10) yyerror("Divided by zero."); $$ = $1 / $3; }
    ;

rfactor : NUMBER { $$ = $1; }
    | OPEN_PAREN expr CLOSE_PAREN { $$ = $2; }
    ;



%%

int main(int argc, char **argv) {
    /* if (argc > 1) {
        std::ifstream file;
        file.open(std::string(argv[1]));

    } */
    yyparse();
    return 0;
}

void yyerror(const char *msg) {
    std::cout << msg << "\n";
}