
#include "parser.hh"

int main()
{
    std::string s = "2 + -3^x - 2*(3*y - -4*z^g^u)";

    ast::calculator_grammar<std::string::const_iterator> g;

    auto tree = ast_parse<std::string::const_iterator>(s.begin(), s.end(), g);


}
