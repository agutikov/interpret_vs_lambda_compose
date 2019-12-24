#pragma once

#include <chrono>
#include <ratio>
#include <any>

#include "parser.hh"

double us(std::chrono::steady_clock::duration d)
{
    return double(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count()) / 1000.0;
}

template<typename F, typename Env_T>
auto apply(F &f, const Env_T &e)
{
    return f(e);
}

template<typename Ops_T, typename Env_T, typename Compiler_T, typename Interpreter_T>
void test(
    const Ops_T &ops,
    const ast::grammar<std::string::const_iterator> &g,
    const std::string &text,
    const Env_T &env,
    double r,
    bool verbose=false,
    bool debug=false)
{
    printf("\n");
    if (verbose) {
        printf("\n%s\n", text.c_str());
    }

    auto cops = Compiler_T::generate_compiler_ops(ops);

    auto start_parse = std::chrono::steady_clock::now();
    auto tree = ast_parse(text, g);
    auto elapsed_parse = std::chrono::steady_clock::now() - start_parse;

    if (verbose) {
        print_tree(tree);
    }

    auto counters = count_nodes(tree);
    printf("chars: %ld, nodes: %d, subtrees: %d, leafs: %d, max_depth: %d\n",
        text.size(),
        std::get<0>(counters),
        std::get<1>(counters),
        std::get<2>(counters),
        std::get<3>(counters)
    );

    auto start_compile = std::chrono::steady_clock::now();
    auto f = Compiler_T::compile_tree(cops, tree);
    auto elapsed_compile = std::chrono::steady_clock::now() - start_compile;

    auto start_exec = std::chrono::steady_clock::now();
    auto result_exec = apply(f, env);
    auto elapsed_exec = std::chrono::steady_clock::now() - start_exec;

    Interpreter_T interpreter(ops);
    auto start_interpret = std::chrono::steady_clock::now();
    auto result_interpret = interpreter.interpret_tree(tree, env);
    auto elapsed_interpret = std::chrono::steady_clock::now() - start_compile;

    printf("parse: %.3f us, compile: %.3f us, exec: %.3f us, interpret: %.3f us, speedup: %.2f\n",
        us(elapsed_parse),
        us(elapsed_compile),
        us(elapsed_exec),
        us(elapsed_interpret),
        us(elapsed_interpret)/us(elapsed_exec));

    if (verbose) {
        printf("result: %f\n", std::any_cast<double>(result_exec));
    }
}

