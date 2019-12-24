#pragma once

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

namespace li_v1 {

typedef std::function<std::any(const std::vector<std::any>&)> op_f;

typedef std::map<std::string, std::any> env_t;

typedef std::function<std::any(const env_t&)> compiled_f;

typedef std::function<compiled_f(const ast::ast_node &)> compile_node_f;

typedef std::map<std::string, std::pair<op_f, compile_node_f>> ops_t;



struct interpreter
{
    interpreter(const ops_t &ops) :
        ops(ops)
    {}

    std::any interpret_tree(const ast::ast_tree &tree, const env_t &env)
    {
        // get op functions
        auto op = ops.find(tree.name)->second;
        auto func = op.first;
        auto compile_token = op.second;

        std::vector<std::any> args;
        if (func == nullptr) {
            // compile_token returns function compiled from token
            auto f = compile_token(tree.children[0]);
            std::any value = f(env);
            return value;
        } else {
            // interpret args
            BOOST_FOREACH(ast::ast_node const& node, tree.children) {
                std::any arg = interpret_tree(boost::get<ast::ast_tree>(node), env);
                args.push_back(arg);
            }

            // interpret function call
            return func(args);
        }

    }

    const ops_t &ops;
};


struct ast_node_compiler : boost::static_visitor<compiled_f>
{
    typedef std::map<std::string, std::shared_ptr<ast_node_compiler>> compiler_visitors_t;

    static std::string get_node_name(const ast::ast_node &tree)
    {
        struct node_name : boost::static_visitor<std::string>
        {
            std::string operator()(const ast::ast_tree &ast) { return ast.name; }
            std::string operator()(const std::string &value) { return value; }
            std::string operator()(double value) { throw std::invalid_argument("ERROR: node_name::operator()(double) called"); }
        };
        static node_name get_name;

        return boost::apply_visitor(get_name, tree);
    }

    static compiled_f compile_tree(std::weak_ptr<compiler_visitors_t> ops, const ast::ast_node &tree)
    {
        // first apply visitor that returns string
        std::string name = get_node_name(tree);

        // then select compile visitor from ops
        std::shared_ptr<ast_node_compiler> op = ops.lock()->find(name)->second;

        // then apply compile visitor to this node
        return boost::apply_visitor(*(op.get()), tree);
    }

    static std::shared_ptr<compiler_visitors_t> generate_compiler_ops(const ops_t &ops)
    {
        auto m = std::make_shared<compiler_visitors_t>();

        for (const auto &op : ops) {
            auto c = std::make_shared<ast_node_compiler>(op.second.first, op.second.second, m);
            m->emplace(op.first, c);
        }

        return m;
    }

    ast_node_compiler(op_f func, compile_node_f compile_token, std::shared_ptr<compiler_visitors_t> ops) :
        func(func),
        compile_token(compile_token),
        ops(ops)
    {}

    compiled_f operator()(ast::ast_tree const& ast)
    {
        // compile args
        std::vector<compiled_f> args;
        BOOST_FOREACH(ast::ast_node const& node, ast.children) {
            compiled_f arg_f = (compile_token != nullptr) ? compile_token(node) : compile_tree(ops, node);
            args.push_back(arg_f);
        }

        // return compiled argument
        if (func == nullptr) {
            return args[0];
        }
        
        // compile function call
        return [_func = func, args] (const env_t& e) -> std::any {
            std::vector<std::any> a;

            // calculate args
            std::transform(args.begin(), args.end(), std::back_inserter(a), 
                [&e](compiled_f f){ return f(e); });

            // return result of calling func on calculated args
            return _func(a);
        };
    }

    compiled_f operator()(std::string const& value)
    {
        throw std::invalid_argument("ERROR: ast_node_compiler::operator()(std::string const&) called");
    }

    compiled_f operator()(double value)
    {
        throw std::invalid_argument("ERROR: ast_node_compiler::operator()(double) called");
    }

    op_f func;
    compile_node_f compile_token;

    std::weak_ptr<compiler_visitors_t> ops;
};




compile_node_f compile_number = [](const ast::ast_node &node) -> compiled_f
{
    auto value = boost::get<double>(node);
    return [value](const env_t&){
        return std::any(value);
    };
};

compile_node_f compile_const = [](const ast::ast_node &node) -> compiled_f
{
    auto value = boost::get<std::string>(node);
    return [value](const env_t& e){
        return e.find(value)->second;
    };
};

ops_t ops = {
    {"number", {nullptr, compile_number}},
    {"const", {nullptr, compile_const}},
    {"pow", {[](const std::vector<std::any>& args){ return std::any(pow(std::any_cast<double>(args[0]), std::any_cast<double>(args[1]))); }, nullptr}},
    {"neg", {[](const std::vector<std::any>& args){ return std::any(- std::any_cast<double>(args[0])); }, nullptr}},
    {"mul", {[](const std::vector<std::any>& args){ return std::any(std::any_cast<double>(args[0]) * std::any_cast<double>(args[1])); }, nullptr}},
    {"div", {[](const std::vector<std::any>& args){ return std::any(std::any_cast<double>(args[0]) / std::any_cast<double>(args[1])); }, nullptr}},
    {"add", {[](const std::vector<std::any>& args){ return std::any(std::any_cast<double>(args[0]) + std::any_cast<double>(args[1])); }, nullptr}},
    {"sub", {[](const std::vector<std::any>& args){ return std::any(std::any_cast<double>(args[0]) - std::any_cast<double>(args[1])); }, nullptr}},
};


} // namespace li_v1












