#pragma once

#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <boost/foreach.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <exception>
#include <tuple>


namespace ast
{
    namespace fusion = boost::fusion;
    namespace phoenix = boost::phoenix;
    namespace qi = boost::spirit::qi;
    namespace ascii = boost::spirit::ascii;

    struct ast_tree;

    typedef
        boost::variant<
            boost::recursive_wrapper<ast_tree>,  // subtree
            std::string,                         // or value in leaf node
            double
        >
    ast_node;

    struct ast_tree
    {
        std::string name = "default";            // tag name
        std::vector<ast_node> children;          // children
    };
}

BOOST_FUSION_ADAPT_STRUCT(
    ast::ast_tree,
    (std::string, name)
    (std::vector<ast::ast_node>, children)
)

namespace ast
{
    int const tabsize = 2;

    void tab(int indent)
    {
        for (int i = 0; i < indent; ++i)
            std::cout << ' ';
    }

    struct ast_tree_printer
    {
        ast_tree_printer(int indent = 0)
          : indent(indent)
        {
        }

        void operator()(ast_tree const& ast) const;

        int indent;
    };

    struct ast_node_printer : boost::static_visitor<>
    {
        ast_node_printer(int indent = 0)
          : indent(indent)
        {
        }

        void operator()(ast_tree const& ast) const
        {
            ast_tree_printer(indent+tabsize)(ast);
        }

        void operator()(std::string const& value) const
        {
            std::cout << "\t" << value;
        }

        void operator()(double value) const
        {
            std::cout << "\t" << value;
        }

        int indent;
    };

    void ast_tree_printer::operator()(ast_tree const& ast) const
    {
        std::cout << std::endl;
        tab(indent);
        std::cout << ast.name;
        
        BOOST_FOREACH(ast_node const& node, ast.children)
        {
            boost::apply_visitor(ast_node_printer(indent), node);
        }

    }

    template <typename Iterator>
    struct grammar
      : qi::grammar<Iterator, ast_tree(), ascii::space_type>
    {
        grammar() : grammar::base_type(start)
        {}

        qi::rule<Iterator, ast_tree(), ascii::space_type> start;
    };


    template <typename Iterator>
    struct calculator_grammar : ast::grammar<Iterator>
    {
        calculator_grammar(bool dbg=false)
        {
            using qi::lit;
            using qi::lexeme;
            using qi::raw;
            using ascii::char_;
            using ascii::alnum;
            using ascii::alpha;
            using ascii::digit;
            using qi::double_;
            using qi::real_parser;
            using ascii::string;
            using namespace qi::labels;
            using phoenix::at_c;
            using phoenix::push_back;
            using qi::debug;

            CNAME = raw[lexeme[alpha >> *(alnum | '_')]];
            NUMBER = double_;

            number = NUMBER                             [at_c<0>(_val) = "number", push_back(at_c<1>(_val), _1)];
            const_ref = CNAME                           [at_c<0>(_val) = "const", push_back(at_c<1>(_val), _1)];

            value = number                              [_val = _1]
                | const_ref                             [_val = _1]
                | '(' >> sum                            [_val = _1]
                      >> ')';

            pow = value                                 [push_back(at_c<1>(_val), _1)]
                  >> '^' >> pow                         [at_c<0>(_val) = "pow", push_back(at_c<1>(_val), _1)]
                | value                                 [_val = _1];

            neg = '-' >> pow                            [at_c<0>(_val) = "neg", push_back(at_c<1>(_val), _1)]
                | pow                                   [_val = _1];

            product = neg                               [push_back(at_c<1>(_val), _1)]
                >>
                (   ('*' >> product                     [at_c<0>(_val) = "mul", push_back(at_c<1>(_val), _1)]  )
                  | ('/' >> product                     [at_c<0>(_val) = "div", push_back(at_c<1>(_val), _1)]  )
                )
                | neg                                   [_val = _1];

            sum = product                               [push_back(at_c<1>(_val), _1)]
                >>
                (   ('+' >> sum                         [at_c<0>(_val) = "add", push_back(at_c<1>(_val), _1)]  )
                  | ('-' >> sum                         [at_c<0>(_val) = "sub", push_back(at_c<1>(_val), _1)]  )
                )
                | product                               [_val = _1];

            if (dbg) {
                sum.name("sum");
                product.name("product");
                neg.name("neg");
                pow.name("pow");
                value.name("value");
                number.name("number");
                const_ref.name("const");

                debug(number);
                debug(const_ref);
                debug(value);
                debug(pow);
                debug(neg);
                debug(product);
                debug(sum);
            }

           this->start = sum[_val = _1];
        }

        qi::rule<Iterator, std::string(), ascii::space_type> CNAME;
        qi::rule<Iterator, double, ascii::space_type> NUMBER;

        qi::rule<Iterator, ast_tree(), ascii::space_type> number;
        qi::rule<Iterator, ast_tree(), ascii::space_type> const_ref;

        qi::rule<Iterator, ast_tree(), ascii::space_type> value;

        qi::rule<Iterator, ast_tree(), ascii::space_type> pow;

        qi::rule<Iterator, ast_tree(), ascii::space_type> neg;

        qi::rule<Iterator, ast_tree(), ascii::space_type> product;

        qi::rule<Iterator, ast_tree(), ascii::space_type> sum;
    };

}

template<typename Iterator>
ast::ast_tree
ast_parse(
    Iterator begin,
    Iterator end,
    const ast::grammar<Iterator> &g)
{
    ast::ast_tree tree;
    using boost::spirit::ascii::space;

    bool r = phrase_parse(begin, end, g, space, tree);

    if (r && begin == end) {
        return tree;
    } else {
        std::string::const_iterator some = begin+30;
        std::string context(begin, (some>end)?end:some);
        std::cout << "stopped at: \"" << context << "\"\n";
        
        throw std::invalid_argument(std::string("parser failed at: \"") + context + std::string("\""));
    }
}

ast::ast_tree ast_parse(const std::string &text, const ast::grammar<std::string::const_iterator> &g)
{
    return ast_parse(text.begin(), text.end(), g);
}

void print_tree(const ast::ast_tree &tree)
{
    ast::ast_tree_printer printer;
    printer(tree);
    std::cout << std::endl;
}



std::tuple<int, int, int, int> count_nodes(const ast::ast_tree &tree)
{
    struct node_counter : boost::static_visitor<>
    {
        void operator()(const ast::ast_tree &ast)
        {
            nodes++;
            subtrees++;
            depth++;
            if (max_depth < depth) {
                max_depth = depth;
            }
            BOOST_FOREACH(ast::ast_node const& node, ast.children)
            {
                boost::apply_visitor(*this, node);
            }
            depth--;
        }

        void operator()(const std::string &value)
        {
            count_leaf();
        }

        void operator()(double value)
        {
            count_leaf();
        }

        void count_leaf() {
            nodes++;
            leafs++;
            if (max_depth < depth + 1) {
                max_depth = depth + 1;
            }
        }

        size_t nodes = 0;
        size_t subtrees = 0;
        size_t leafs = 0;
        size_t depth = 0;
        size_t max_depth = 0;
    };
    node_counter counter;

    counter(tree);

    return {counter.nodes, counter.subtrees, counter.leafs, counter.max_depth};
}

