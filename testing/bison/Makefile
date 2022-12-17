all: calculator.l calculator.y
	bison -d -o parser.cpp calculator.y
	flex --header-file=lexer.hpp -o lexer.cpp calculator.l
	g++ lexer.cpp parser.cpp -o run

clean:
	rm -rf run calculator.tab.* lex.yy.c* lexer.hpp lexer.cpp parser.cpp parser.hpp
