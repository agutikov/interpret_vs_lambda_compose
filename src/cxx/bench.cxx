
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <any>
#include <algorithm>
#include <memory>
#include <utility>
#include <exception>
#include <cmath>
#include <chrono>
#include <ratio>

#include "parser.hh"
#include "test.hxx"
#include "lambda_vs_interpret.hxx"
#include "lambda_vs_interpret_fast.hxx"
#include "lambda_vs_interpret_stack.hxx"
#include "lambda_vs_interpret_fast_packed_env.hxx"
#include "lambda_vs_interpret_fastest.hxx"

template<typename Ops_T, typename Env_T, typename Compiler_T, typename Interpreter_T>
void run_tests(const Ops_T &ops)
{
    ast::calculator_grammar<std::string::const_iterator> g;

    std::string text = "((z * y) - 4096 + 999) - (x * -1) / 0.1 - 999 - (4096 - -1 + (10 - 4096) * ((999 + x) * (z + 4096))) / ( -z / x / x - -1 + (4096 * y - z - -1)) - (999 + -1 / (0.1 + 10)) - ( -(4096 / -1) / ( -y +  -0.1))";

    while (text.size() < 10000) {
        text += " + " + text;
    }

    test<Ops_T, Env_T, Compiler_T, Interpreter_T>
    (ops, g, text, {{"x", 1.0}, {"y", 10.0}, {"z", 2.0}}, 0.0, false, true);
}

void print_delim(const char* name)
{
    printf("\n===============================================================================\n");
    printf("%s\n", name);
}

int main()
{
    print_delim("V1");
    run_tests<li_v1::ops_t, li_v1::env_t, li_v1::ast_node_compiler, li_v1::interpreter>(li_v1::ops);

    print_delim("Stack");
    run_tests<li_stack::ops_t, li_stack::env_t, li_stack::ast_node_compiler, li_stack::interpreter>(li_stack::ops);

    print_delim("Fast");
    run_tests<li_fast::ops_t, li_fast::env_t, li_fast::ast_node_compiler, li_fast::interpreter>(li_fast::ops);

    print_delim("Fast with packed env");
    run_tests<li_fast_packed_env::ops_t, li_fast_packed_env::env_t, li_fast_packed_env::ast_node_compiler, li_fast_packed_env::interpreter>(li_fast_packed_env::ops);

    print_delim("Fastest");
    run_tests<li_fastest::ops_t, li_fastest::env_t, li_fastest::ast_node_compiler, li_fastest::interpreter>(li_fastest::ops);

    printf("\n");
    return 0;
}




