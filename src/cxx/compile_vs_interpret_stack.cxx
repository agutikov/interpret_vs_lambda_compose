
#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include <stack>
#include <any>
#include <algorithm>
#include <memory>
#include <utility>
#include <exception>
#include <cmath>
#include <chrono>
#include <ratio>

#include "parser.hh"


typedef std::stack<std::any> stack_t;

typedef std::function<std::any(stack_t&)> op_f;

typedef std::map<std::string, std::any> env_t;

typedef std::function<std::any(stack_t&, const env_t&)> compiled_f;

typedef std::function<compiled_f(const ast::ast_node &)> compile_node_f;

typedef std::map<std::string, std::pair<op_f, compile_node_f>> ops_t;


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
        return [_func = func, args] (stack_t& stack, const env_t& e) -> std::any {
            // calculate args
            std::for_each(args.begin(), args.end(), 
                [&stack, &e](compiled_f f){ stack.push(f(stack, e)); });

            // return result of calling func on calculated args
            return _func(stack);
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

/*============================================================================================
 * 
 * 
 *============================================================================================
 */

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

        if (func == nullptr) {
            // compile_token returns function compiled from token
            return compile_token(tree.children[0])(stack, env);
        } else {
            // interpret args
            BOOST_FOREACH(ast::ast_node const& node, tree.children) {
                stack.push(interpret_tree(boost::get<ast::ast_tree>(node), env));
            }

            // interpret function call
            return func(stack);
        }
    }

    const ops_t &ops;
    stack_t stack;
};




/*============================================================================================
 * 
 * 
 *============================================================================================
 */

double us(std::chrono::steady_clock::duration d)
{
    return double(std::chrono::duration_cast<std::chrono::nanoseconds>(d).count()) / 1000.0;
}

void test(
    const ops_t &ops,
    const ast::grammar<std::string::const_iterator> &g,
    const std::string &text,
    const env_t &env,
    double r,
    bool verbose=false,
    bool debug=false)
{
    printf("\n");
    if (verbose) {
        printf("\n%s\n", text.c_str());
    }

    auto cops = ast_node_compiler::generate_compiler_ops(ops);

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
    compiled_f f = ast_node_compiler::compile_tree(cops, tree);
    auto elapsed_compile = std::chrono::steady_clock::now() - start_compile;

    auto start_exec = std::chrono::steady_clock::now();
    stack_t stack;
    std::any result_exec = f(stack, env);
    auto elapsed_exec = std::chrono::steady_clock::now() - start_exec;

    interpreter interpreter(ops);
    auto start_interpret = std::chrono::steady_clock::now();
    std::any result_interpret = interpreter.interpret_tree(tree, env);
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

/*============================================================================================
 * 
 * 
 *============================================================================================
 */



compile_node_f compile_number = [](const ast::ast_node &node) -> compiled_f
{
    auto value = boost::get<double>(node);
    return [value](stack_t&, const env_t&){
        return std::any(value);
    };
};

compile_node_f compile_const = [](const ast::ast_node &node) -> compiled_f
{
    auto value = boost::get<std::string>(node);
    return [value](stack_t&, const env_t& e){
        return e.find(value)->second;
    };
};

ops_t ops = {
    {"number", {nullptr, compile_number}},
    {"const", {nullptr, compile_const}},

    {"pow", {[](stack_t& stack)
    {
        double arg1 = std::any_cast<double>(stack.top());
        stack.pop();
        double arg0 = std::any_cast<double>(stack.top());
        stack.pop();
        return std::any(pow(arg0, arg1));
    }, nullptr}},
    
    {"neg", {[](stack_t& stack)
    {
        double arg0 = std::any_cast<double>(stack.top());
        stack.pop();
        return std::any(-arg0); 
    }, nullptr}},
    
    {"mul", {[](stack_t& stack)
    {
        double arg1 = std::any_cast<double>(stack.top());
        stack.pop();
        double arg0 = std::any_cast<double>(stack.top());
        stack.pop();
        return std::any(arg0 * arg1);
    }, nullptr}},
    
    {"div", {[](stack_t& stack)
    {
        double arg1 = std::any_cast<double>(stack.top());
        stack.pop();
        double arg0 = std::any_cast<double>(stack.top());
        stack.pop();
        return std::any(arg0 / arg1);
    }, nullptr}},
    
    {"add", {[](stack_t& stack)
    {
        double arg1 = std::any_cast<double>(stack.top());
        stack.pop();
        double arg0 = std::any_cast<double>(stack.top());
        stack.pop();
        return std::any(arg0 + arg1);
    }, nullptr}},
    
    {"sub", {[](stack_t& stack)
    {
        double arg1 = std::any_cast<double>(stack.top());
        stack.pop();
        double arg0 = std::any_cast<double>(stack.top());
        stack.pop();
        return std::any(arg0 - arg1);
    }, nullptr}},
};


/*============================================================================================
 * 
 * 
 *============================================================================================
 */

int main()
{
    ast::calculator_grammar<std::string::const_iterator> g;

    test(ops, g, "x * 2 + -y", {{"x", 1.0}, {"y", 2.0}}, 0.0);
    test(ops, g, "x / 2 - 1 / y", {{"x", 1.0}, {"y", 2.0}}, 0.0);
    test(ops, g, "x ^ y - 1", {{"x", 1.0}, {"y", 2.0}}, 0.0);
    test(ops, g, "2 + -3^x - 2*(3*y - -4*z^g^u)", {{"x", 1.0}, {"y", 10.0}, {"z", 2.0}, {"g", 2.0}, {"u", 3.0}}, -2109.0, false);

    std::string text = "((z * y) - 4096 + 999) - (x * -1) / 0.1 - 999 - (4096 - -1 + (10 - 4096) * ((999 + x) * (z + 4096))) / ( -z / x / x - -1 + (4096 * y - z - -1)) - (999 + -1 / (0.1 + 10)) - ( -(4096 / -1) / ( -y +  -0.1))";
    
    test(ops, g, text, {{"x", 1.0}, {"y", 10.0}, {"z", 2.0}}, 0.0, false, true);


    while (text.size() < 5000) {
        text += " + " + text;
    }

    test(ops, g, text, {{"x", 1.0}, {"y", 10.0}, {"z", 2.0}}, 0.0, false, true);


    return 0;
}














