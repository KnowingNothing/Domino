%{
    #include <string>
    #include <iostream>

    int count_char = 0;
    int count_word = 0;
    int count_line = 1;

    enum MyToken {
        WORD = 258,
        RETURN = 259,
        OTHER = 260
    };

    std::string word_text;
%}

%option noyywrap


%%

[^ \t\v\n]+ { count_word += 1; count_char += strlen(yytext); word_text = std::string(yytext); return WORD; }
\n { count_char += 1; count_line += 1; return RETURN; }
. { count_char += 1; return OTHER; }

%%

int main(int argc, char** argv) {
    int token;
    yyFlexLexer yylexer(std::cin, std::cout);
    while (token = yylexer.yylex()) {
        std::cout << "TOKEN " << token;
        if (token == WORD) {
            std::cout << " " << word_text;
        }
        std::cout << "\n";
    }

    std::cout << "Total lines: " << count_line << ", words: " << count_word << ", characters: " << count_char << "\n";

    return 0;
}