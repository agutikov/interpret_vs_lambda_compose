/**
 * 
 * 
 * 
 * 
 * 
 * 
 */

#pragma once

#include <cstdio>
#include <functional>
#include <map>
#include <unordered_map>
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

namespace li_fastest {

typedef std::any op_f;

typedef std::vector<double> _env_t;

typedef std::map<std::string, double> env_t;

typedef std::function<double(const _env_t&)> compiled_f;

typedef std::function<compiled_f(const ast::ast_node &)> compile_node_f;

typedef std::map<std::string, std::pair<op_f, compile_node_f>> ops_t;



//TODO: Can generic lambdas and curring help to improve performance or flexibility?
// https://stackoverflow.com/questions/25885893/how-to-create-a-variadic-generic-lambda


std::map<std::string, size_t> const_name_2_index;

void reset_const_names()
{
    const_name_2_index = {};
}

_env_t convert_env(const std::map<std::string, double> &e)
{
    _env_t v;
    for (const auto& [key, value] : e) {
        auto it = const_name_2_index.find(key);
        if (v.size() <= it->second) {
            v.resize(it->second + 1);
        }
        v[it->second] = value;
    }
    return v;
}


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

    static auto compile_tree(std::weak_ptr<compiler_visitors_t> ops, const ast::ast_node &tree)
    {
        reset_const_names();
        auto f = _compile_tree(ops, tree);
        return [f](const env_t &e)
            {
                return f(convert_env(e));
            };
    }

    static compiled_f _compile_tree(std::weak_ptr<compiler_visitors_t> ops, const ast::ast_node &tree)
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
    {
        if (!func.has_value() && compile_token == nullptr) {
            throw std::invalid_argument("ERROR: ast_node_compiler: At least func or compile_token must be provided.");
        }
    }

    compiled_f operator()(ast::ast_tree const& ast)
    {
        // compile args
        std::vector<compiled_f> args;
        BOOST_FOREACH(ast::ast_node const& node, ast.children) {
            compiled_f arg_f = (compile_token != nullptr) ? compile_token(node) : _compile_tree(ops, node);
            args.push_back(arg_f);
        }

        // return compiled argument
        if (!func.has_value()) {
            return args[0];
        }

        // compile function call
        // The fastest way would be to use call stack directly.
        // different implementations of lambdas depending on number of args (with or without checking at compile time)
        if (args.size() == 1) {
            return
            [
                _func = std::any_cast<std::function<double(double)>>(func),
                arg0 = args[0]
            ]
            (const _env_t& e) -> double
            {
                return _func(arg0(e));
            };
        } else if (args.size() == 2) {
            return
            [
                _func = std::any_cast<std::function<double(double, double)>>(func),
                arg0 = args[0],
                arg1 = args[1]
            ]
            (const _env_t& e) -> double
            {
                return _func(arg0(e), arg1(e));
            };
        } else {
            throw std::invalid_argument("ERROR: ast_node_compiler::operator()(ast::ast_tree const&) not supported number of function arguments");
        }
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


struct interpreter
{
    interpreter(const ops_t &ops) :
        ops(ops)
    {}

    double interpret_tree(const ast::ast_tree &tree, const env_t &env)
    {
        reset_const_names();
        return _interpret_tree(tree, convert_env(env));
    }

    double _interpret_tree(const ast::ast_tree &tree, const _env_t &env)
    {
        // get op functions
        auto op = ops.find(tree.name)->second;
        auto func = op.first;
        auto compile_token = op.second;

        if (!func.has_value()) {
            // compile_token returns function compiled from token
            return compile_token(tree.children[0])(env);
        } else {
            // interpret function call
            if (tree.children.size() == 1) {
                auto arg0 = _interpret_tree(boost::get<ast::ast_tree>(tree.children[0]), env);
                return std::any_cast<std::function<double(double)>>(func)(arg0);
            } else if (tree.children.size() == 2) {
                auto arg0 = _interpret_tree(boost::get<ast::ast_tree>(tree.children[0]), env);
                auto arg1 = _interpret_tree(boost::get<ast::ast_tree>(tree.children[1]), env);
                return std::any_cast<std::function<double(double, double)>>(func)(arg0, arg1);
            } else {
                throw std::invalid_argument("ERROR: interpreter::_interpret_tree(const ast::ast_tree &, const _env_t &) not supported number of function arguments");
            }
        }
    }

    const ops_t &ops;
};



compile_node_f compile_number = [](const ast::ast_node &node) -> compiled_f
{
    auto value = boost::get<double>(node);
    return [value](const _env_t&){
        return value;
    };
};


compile_node_f compile_const = [](const ast::ast_node &node) -> compiled_f
{
    auto value = boost::get<std::string>(node);

    auto it = const_name_2_index.find(value);
    size_t index = const_name_2_index.size();
    if (it == const_name_2_index.end()) {
        const_name_2_index[value] = index;
    }

    return [index](const _env_t& e){
        return e[index];
    };
};


ops_t ops = {
    {"number", {{}, compile_number}},
    {"const", {{}, compile_const}},

    {"pow", {
        std::function<double(double, double)>(
            [](double a, double b){ return pow(a, b); }
        ), nullptr}
    },

    {"neg", {
        std::function<double(double)>(
            [](double a){ return -a; }
        ), nullptr}
    },

    {"mul", {
        std::function<double(double, double)>(
            [](double a, double b){ return a * b; }
        ), nullptr}
    },

    {"div", {
        std::function<double(double, double)>(
            [](double a, double b){ return a / b; }
        ), nullptr}
    },

    {"add", {
        std::function<double(double, double)>(
            [](double a, double b){ return a + b; }
        ), nullptr}
    },

    {"sub", {
        std::function<double(double, double)>(
            [](double a, double b){ return a - b; }
        ), nullptr}
    },
};

} // namespace li_fastest









