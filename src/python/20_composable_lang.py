
from typing import List, Dict, Tuple, Callable, Any
import lark
from pprint import pprint
from inspect import signature, Parameter
import time
import random
import functools
from enum import Enum
from collections import namedtuple
from copy import deepcopy

# :) id - main function of all functions
_id = lambda x: x

def value_closure(value):
    return lambda *xs: value

# lang -> (grammar -> parser, compiler, interpreter, generator)
# lang: [op],
#   op: "name", "sign", arity, function, grammar_rules
#     grammar_rules: priority, grammar_order, grammar_accociativity, grammar_parantheses
#       priority: non-negative number, root element has min priority
#       grammar_order: infix, prefix, postfix
#       grammar_accociativity: left, right
#       grammar_parantheses: required, available, restricted
#
# For example function call, like f(x) is prefix with parantheses required with accociativity not applicable.
# 
# Compose languages by matching terminal and root nodes of grammar and shifting priority
# Compose predefined languages, like:
#   MYLANG = LOGIC on COMPARE on ARYTHMETIC on (What goes here? Vars and numbers. How to make function with arg, like (_ + 3))
# Compose one language over multiple langs, like:
#   MYLANG = LOGIC on (COMPARE on ARYTHMETIC on (ENV_VARS and NUMBERS) and TYPE_PREDICATES)
#
# Composition of languages and types (signatures) of functions, Category Theory graph for that functions and types.
#
# Ops dict is complete definition of lang grammar and of operational semantics.
#
# Let's think about semantics of functions themself.
# For every function we can define true statements using args and return value and other operations.
# Example: If z == x + y then z > x and z > y and z - x == y and z - y == x.
# Definition of function: if z is the result of add(x, y) then z == x + y.
# Definition of multiplication with addition, foldl and repeat:
#   z == x * y => z == foldl (+) [repeat x times y]
# Also can add other axioms, or properties of operations, like transitivity:
#   transitivity of <: x < y and y < z => x < z
#   transitivity of any op: x op y and y op z => x op z
# So having all this properties with every op compiler can infer properties of composed functions.
# Or if can answer questions like: if (many equations) then if (one equation) is true?
# Anyway what is the meaning of understanding of function semantics?
# It is ability to predict result of function, and explain properties of result.
#

def compile_tree(ops, tree):
    op_name = tree.data if isinstance(tree, lark.Tree) else tree
    return ops[op_name](ops, tree)


def compile_func_call(func, compile_arg, ops, tree):
    """
        Returns lambda that call func from results of compiled functions of tree.children
    """
    sig = signature(func)

    #TODO: 2 types of leafs (Token) handling:
    # 1-st - find it in ops and return function directly
    # 2-nd - compile function from token value, like get var value or return number
    if isinstance(tree, lark.Token):
        return func
    
    # tree without children is Token
    if len(tree.children) == 0:
        return func

    arity = len(sig.parameters)
    check_arity = True
    if arity >= 1:
        if list(sig.parameters.values())[-1].kind == Parameter.VAR_POSITIONAL:
            check_arity = False

    if check_arity and arity != len(tree.children):
        raise Exception(f'ERROR: compile_func_call: function "{tree.data}" reqires {arity} arguments, provided {len(tree.children)}')

    arg_funcs = [compile_arg(ops, arg) for arg in tree.children]

    if func == _id or func is None:
        # optimization :)
        return arg_funcs[0]

    if sig.return_annotation == Callable:
        # Functor
        return func(*arg_funcs)

    return (lambda f, funcs: lambda *xs: f(*(g(*xs) for g in funcs)))(func, arg_funcs)


def generate_compiler_ops(ops_table: Dict[str, Callable]):
    """
        Generates functions that will be called from compile_tree() for compilation of tree nodes.
    """
    ops = {}
    for name, value in ops_table.items():
        if callable(value[0]):
            func = value[0]
        else:
            func = lambda *xs: func

        if value[1] is not None:
            compile_arg = value[1]
        else:
            compile_arg = compile_tree
    
        ops[name] = (lambda f, c_arg: lambda ops, tree: compile_func_call(f, c_arg, ops, tree))(func, compile_arg)
    return ops


def compile(compiler, text):
    compile_ops, parser = compiler
    tree = parser.parse(text)
    print(tree.pretty())
    return compile_tree(compile_ops, tree)


#
# ====================================================================================================
# 
# ====================================================================================================
#


class SyntacticOrder(Enum):
    NOT_APPLICABLE = None
    PREFIX = -1
    INFIX = 0
    POSTFIX = 1

class SyntacticAssociativity(Enum):
    NOT_APPLICABLE = None
    LEFT = -1
    NOT_ASSOCIATIVE = 0,
    RIGHT = 1,
    FULLY_ASSOCIATIVE = 2,

class SyntacticParantheses(Enum):
    NOT_APPLICABLE = None
    REQUIRED = True

Syntax = namedtuple('Syntax', ['token', 'priority', 'order', 'associativity', 'parantheses', 'arity'], defaults=[
    0,
    SyntacticOrder.NOT_APPLICABLE,
    SyntacticAssociativity.NOT_APPLICABLE,
    SyntacticParantheses.NOT_APPLICABLE,
    None,
])

LangOp = namedtuple('LangOp', ['syntax', 'op', 'compile_token'], defaults=[None, None])

def is_associative(op):
    return op.syntax.associativity in set([SyntacticAssociativity.LEFT, SyntacticAssociativity.RIGHT])

def isnamedtupleinstance(x):
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple: return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple): return False
    return all(type(n)==str for n in f)

def namedtuple2dict(nt):
    d = {}
    for k, v in nt._asdict().items():
        if isnamedtupleinstance(v):
            d[k] = namedtuple2dict(v)
        elif isinstance(v, Enum):
            d[k] = v.name
        else:
            d[k] = v
    return d

def ops2dict(ops):
    d = {}
    for k, v in ops.items():
        d[k] = namedtuple2dict(v)
    return d



#
# ====================================================================================================
# 
# ====================================================================================================
#



def generate_grammar_rule(ops, inner_rule_name, rule_name, alias_prefix):
    """
        Generates one grammar rule for ops with same priority.
    """
    # every op produce a match for non-terminal 
    # op key used as alias
    matches = []
    plain_ops = {}

    if inner_rule_name is not None:
        matches.append(([inner_rule_name], None))

    for op_name, op in ops.items():
        alias = f'f_{alias_prefix}_{op_name}'

        if op.syntax.arity is not None:
            arity = op.syntax.arity
        elif op.op is None or not callable(op.op):
            arity = 0
        else:
            sig = signature(op.op)
            arity = len(sig.parameters)
            multiarity = arity >= 1 and list(sig.parameters.values())[-1].kind == Parameter.VAR_POSITIONAL
            multiarity_suffix = ["+" if multiarity else ""]

        token = op.syntax.token

        if arity == 0:
            if '|' in token:
                match = [f'( {token} )']
            else:
                match = [token]
        else:
            #TODO: Parantheses of function args

            if arity == 2 and op.syntax.order == SyntacticOrder.INFIX:
                if op.syntax.associativity == SyntacticAssociativity.RIGHT:
                    match = [inner_rule_name, token, rule_name]
                elif op.syntax.associativity == SyntacticAssociativity.LEFT:
                    match = [rule_name, token, inner_rule_name]
                elif op.syntax.associativity == SyntacticAssociativity.NOT_ASSOCIATIVE:
                    match = [inner_rule_name, token, inner_rule_name]
                elif op.syntax.associativity == SyntacticAssociativity.FULLY_ASSOCIATIVE:
                    match = [rule_name, token, rule_name]
                else:
                    raise Exception(f'ERROR: generate_grammar_rule: invalid "{op_name}" in {ops2dict(ops)}')
            else:
                if op.syntax.order == SyntacticOrder.INFIX:
                    raise Exception(f'ERROR: generate_grammar_rule: invalid "{op_name}" in {ops2dict(ops)}')

                if op.syntax.order == SyntacticOrder.PREFIX:
                    if op.syntax.associativity == SyntacticAssociativity.RIGHT:
                        match = [token] + [rule_name]*arity + multiarity_suffix
                    else:
                        match = [token] + [inner_rule_name]*arity + multiarity_suffix
                elif op.syntax.order == SyntacticOrder.POSTFIX:
                    if op.syntax.associativity == SyntacticAssociativity.LEFT:
                        match = [rule_name]*arity + multiarity_suffix + [token]
                    else:
                        match = [inner_rule_name]*arity + multiarity_suffix + [token]
                else:
                    raise Exception(f'ERROR: generate_grammar_rule: invalid "{op_name}" in {ops2dict(ops)}')

        matches.append((match, alias))
        plain_ops[alias] = (_id if op.op is None else op.op, op.compile_token)

    return matches, plain_ops


def join_grammar_rule(inner_rule_names):
    return [([name], None) for name in inner_rule_names]


def _path2str(path):
    return '_'.join(map(str, path))


def group_ops_prio(ops):
    d = {}
    for k, v in ops.items():
        if v.syntax.priority not in d:
            d[v.syntax.priority] = {}
        d[v.syntax.priority][k] = v
    return d


def _generate_grammar_and_ops(ops, inner_rule_name, path):
    """
        Generates grammar rules for complete dict of ops.
    """
    prefix = _path2str(path)
    rules = {}
    plain_ops = {}
    
    ops_by_prio = group_ops_prio(ops)
    first_rule_name = None
    for prio in sorted(ops_by_prio):
        # every priority produce a grammar rule
        # e.g. ops of same priority compose one grammar rule
        rule_name = f'm_{prefix}_rule_{prio}'
        if first_rule_name is not None:
            first_rule_name = rule_name
        rules[rule_name], rule_plain_ops = generate_grammar_rule(ops_by_prio[prio], inner_rule_name, rule_name, prefix)
        plain_ops = {**plain_ops, **rule_plain_ops}
        inner_rule_name = rule_name
 
    return rules, rule_name, plain_ops


def ops_are_associative(ops_tree):
    if isinstance(ops_tree, list):
        ops_tree = ops_tree[0]
    return functools.reduce(lambda x,y: x or y, [is_associative(op) for _, op in ops_tree.items()], False)


def generate_grammar_and_ops(ops_tree, path = [], parenthesised_rule_name=None):
    """
        Recursively generates grammar rules for entire subtree represented by ops_tree.
        Head ops can be only ops dict, not tree.
    """
    if not isinstance(ops_tree, list):
        # ops_tree is a simple ops dict
        return _generate_grammar_and_ops(ops_tree, None, path)

    rules = {}
    ops = {}

    # ops_tree is a composition
    head = ops_tree[0]
    tail = ops_tree[1:]
    join_rule_name = f'j_{_path2str(path + [0])}_rule'

    # First create head rules and get head root rule name - start
    head_rules, head_root_rule_name, head_ops = _generate_grammar_and_ops(head, join_rule_name, path + [0])

    if not ops_are_associative(head):
        parenthesised_rule_name = None
    elif parenthesised_rule_name is None:
        parenthesised_rule_name = head_root_rule_name

    # Process tail items
    tail_rules = []
    parentheses_passed = True
    for index, ops_tree_item in reversed(list(enumerate(tail))):
        # 0 is head
        r, s, o = generate_grammar_and_ops(ops_tree_item, path + [index + 1], parenthesised_rule_name)
        tail_rules.append(s)
        rules = {**rules, **r}
        ops = {**ops, **o}
        parentheses_passed = parentheses_passed and ops_are_associative(ops_tree_item)

    # Join tail items
    rules[join_rule_name] = join_grammar_rule(tail_rules)

    # Add parentheses match if it was not passed through every subtree
    if parenthesised_rule_name is not None and not parentheses_passed:
        rules[join_rule_name] += [([f'"(" {parenthesised_rule_name} ")"'], None)]
        parenthesised_rule_name = None

    # Finally wrap with head rules
    rules = {**rules, **head_rules}
    ops = {**ops, **head_ops}

    return rules, head_root_rule_name, ops



grammar_footer = """
%import common.ESCAPED_STRING
%import common.CNAME
%import common.NUMBER

%import common.WS_INLINE
%import common.NEWLINE
%ignore WS_INLINE
%ignore NEWLINE
"""

def grammar2str(rules):
    grammar = grammar_footer
    for rule_name, rule in rules.items():
        rule_str = '\n  | '.join([' '.join(match) + (' -> ' + alias if alias is not None else '') for match, alias in rule])
        grammar = "?" + rule_name + ": " + rule_str + "\n\n" + grammar

    return '\n' + grammar


def make_compiler(ops_tree):
    grammar, start_symbol, ops = generate_grammar_and_ops(ops_tree)
    pprint(ops)

    g_str = grammar2str(grammar)
    print(g_str)

    parser = lark.Lark(g_str, start=start_symbol)

    cops = generate_compiler_ops(ops)

    return (cops, parser)

#
# ====================================================================================================
# Test
# ====================================================================================================
#

class NodeCounter:
    def __init__(self):
        self.node_counter = 0
        self.subtree_counter = 0
        self.leaf_counter = 0
        self.depth = 0
        self.max_depth = 0

    def visit(self, tree):
        self.depth += 1
        if self.max_depth < self.depth:
            self.max_depth = self.depth

        self.node_counter += 1
        if isinstance(tree, lark.Tree):
            self.subtree_counter += 1
            for node in tree.children:
                self.visit(node)
        else:
            self.leaf_counter += 1

        self.depth -= 1

def count_nodes(tree):
    nc = NodeCounter()
    nc.visit(tree)
    return nc.node_counter, nc.subtree_counter, nc.leaf_counter, nc.max_depth
 
def test(compiler, text, env, result, verbose=False, debug=False):
    print()
    if verbose:
        print(text)
        print()
        print(env)

    start = time.process_time()
    tree = compiler[1].parse(text)
    parse_elapsed = time.process_time() - start
    if verbose:
        print()
        print(tree.pretty())

    nodes, subtrees, leafs, max_depth = count_nodes(tree)
    print(f'chars: {len(text)}, nodes: {nodes}, subtrees: {subtrees}, leafs: {leafs}, max_depth: {max_depth}')

    start = time.process_time()
    f = compile_tree(compiler[0], tree)
    compile_elapsed = time.process_time() - start

    start = time.process_time()
    r = f(env)
    exec_elapsed = time.process_time() - start

    if not debug:
        assert(result == r)

    print(f"parse: {parse_elapsed}, " 
          f"compile: {compile_elapsed}, "
          f"exec: {exec_elapsed}")

#
# ====================================================================================================
# 
# ====================================================================================================
#


#TODO: CNAME vs. Reserved Literals

logic_literals = {
    "false": LangOp(
        op=False,
        syntax=Syntax('"False" | "FALSE" | "false" | "F" | "f"')),
    
    "true": LangOp(
        op=True,
        syntax=Syntax('"True" | "TRUE" | "true" | "T" | "t"')),
}


def compile_const(_, token):
    return (lambda const_name: lambda x: x[const_name])(token) 

constants = {
    "constant": LangOp(
        compile_token=compile_const,
        syntax=Syntax('CNAME')),
}

logic = {
    "or": LangOp(
        op=lambda x, y: x or y,
        syntax=Syntax(
            token='"or"',
            priority=2,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.LEFT,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),

    "and": LangOp(
        op=lambda x, y: x and y,
        syntax=Syntax(
            token='"and"',
            priority=1,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.LEFT,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),

    "not": LangOp(
        op=lambda x: not x,
        syntax=Syntax(
            token='"not"',
            priority=0,
            order=SyntacticOrder.PREFIX,
            associativity=SyntacticAssociativity.RIGHT,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),
}

logic_comparison = {
    "eq": LangOp(
        op=lambda x, y: x == y,
        syntax=Syntax(
            token='"=="',
            priority=3,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.NOT_ASSOCIATIVE,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),

    "neq": LangOp(
        op=lambda x, y: x != y,
        syntax=Syntax(
            token='"!="',
            priority=3,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.NOT_ASSOCIATIVE,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),
}


# it works because constant_ops goes before logic_literals_ops
# if not - parser will consider "False" and other literals as CNAME
c1 = make_compiler([logic, constants, logic_literals])

f = compile(c1, 'x and y or not z or F')

print(f({'x': True, 'y': False, 'z': False}))



#
# ====================================================================================================
# 
# ====================================================================================================
#


number_literals = {
    "number": LangOp(
        compile_token=lambda _, token: value_closure(float(token)),
        syntax=Syntax('NUMBER'))
}

number_arithmetic = {
    "sub": LangOp(
        op=lambda x, y: x - y,
        syntax=Syntax(
            token='"-"',
            priority=3,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.LEFT,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),

    "add": LangOp(
        op=lambda x, y: x + y,
        syntax=Syntax(
            token='"+"',
            priority=3,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.LEFT,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),

    "div": LangOp(
        op=lambda x, y: x / y,
        syntax=Syntax(
            token='"/"',
            priority=2,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.LEFT,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),

    "mul": LangOp(
        op=lambda x, y: x * y,
        syntax=Syntax(
            token='"*"',
            priority=2,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.LEFT,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),
    
    "neg": LangOp(
        op=lambda x: -x,
        syntax=Syntax(
            token='"-"',
            priority=1,
            order=SyntacticOrder.PREFIX,
            associativity=SyntacticAssociativity.RIGHT,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),
    
    "pow": LangOp(
        op=lambda x, y: x ** y,
        syntax=Syntax(
            token='"^"',
            priority=0,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.RIGHT,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),
}


string_literals = {
    "string": LangOp(
        compile_token=lambda _, token: value_closure(str(token)),
        syntax=Syntax('ESCAPED_STRING'))
}

string_arithmetic = {
    "add": LangOp(
        op=lambda x, y: x + y,
        syntax=Syntax(
            token='"+"',
            priority=0,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.LEFT,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    )
}

comparison = {
    "eq": LangOp(
        op=lambda x, y: x == y,
        syntax=Syntax(
            token='"=="',
            priority=0,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.NOT_ASSOCIATIVE,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),

    "neq": LangOp(
        op=lambda x, y: x != y,
        syntax=Syntax(
            token='"!="',
            priority=0,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.NOT_ASSOCIATIVE,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),

    "gt": LangOp(
        op=lambda x, y: x > y,
        syntax=Syntax(
            token='">"',
            priority=0,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.NOT_ASSOCIATIVE,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),

    "gte": LangOp(
        op=lambda x, y: x >= y,
        syntax=Syntax(
            token='">="',
            priority=0,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.NOT_ASSOCIATIVE,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),

    "lt": LangOp(
        op=lambda x, y: x < y,
        syntax=Syntax(
            token='"<"',
            priority=0,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.NOT_ASSOCIATIVE,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),

    "lte": LangOp(
        op=lambda x, y: x <= y,
        syntax=Syntax(
            token='"<="',
            priority=0,
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.NOT_ASSOCIATIVE,
            parantheses=SyntacticParantheses.NOT_APPLICABLE
        )
    ),
}


c2 = make_compiler([logic,
                        [comparison,
                            [number_arithmetic,
                                number_literals,
                                constants]], 
                        [comparison,
                            [string_arithmetic,
                                string_literals,
                                constants]],
                        logic_literals,
                        constants,
                        ])


text = '(f * g + e > d and a < b and a == b or a * c >= b ^ 2 or "AA" <= "Aa") and (a != b) and (1 == 1) and ((2+2)*2 == 6)'
env = {'a': 100, 'b': 200, 'c': 3, 'd': 9768, 'e': 2334, 'g': -1, 'f': -5.5}

f = compile(c2, text)

print(f(env))


#
# ====================================================================================================
# 
# ====================================================================================================
#

def compile_getarg(ops, tree):
    return (lambda N: lambda *xs: [x for x in xs][N])(int(tree))

args = {
    "arg": LangOp(
        op=lambda *xs: [x for x in xs][0],
        syntax=Syntax(
            token='"_"',
            arity=0,
        )
    ),

    "getarg": LangOp(
        compile_token=compile_getarg,
        syntax=Syntax(
            token='"$" NUMBER',
        )
    ),
}

def _map(f) -> Callable:
    return lambda x: [f(el) for el in x]

def mapf(*funcs) -> Callable:
    return lambda x: [f(x) for f in funcs]

def foldl(f, init_val) -> Callable:
    return lambda x: functools.reduce(f, x, init_val())

# This bind requires all arguments
def bind(f, *fargs) -> Callable:
    return lambda *xs: f(*(g(*xs) for g in fargs))

def compose_inv(g, f) -> Callable:
    return lambda x: f(g(x))

list_functions = {
    "count": LangOp(
        op=lambda x: len(x),
        syntax=Syntax(
            token='"count"',
            arity=0,
        )
    ),

    "sum": LangOp(
        op=lambda x: sum(x),
        syntax=Syntax(
            token='"sum"',
            arity=0,
        )
    ),
}

functors = {
    "pipeline": LangOp(
        op=compose_inv,
        syntax=Syntax(
            token='"|"',
            order=SyntacticOrder.INFIX,
            associativity=SyntacticAssociativity.FULLY_ASSOCIATIVE,
        )
    ),

    "map": LangOp(
        op=_map,
        syntax=Syntax(
            token='"map"',
            order=SyntacticOrder.PREFIX,
            associativity=SyntacticAssociativity.RIGHT,
        )
    ),

    "mapf": LangOp(
        op=mapf,
        syntax=Syntax(
            token='"mapf"',
            order=SyntacticOrder.PREFIX,
            associativity=SyntacticAssociativity.RIGHT,
        )
    ),

    "bind": LangOp(
        op=bind,
        syntax=Syntax(
            token='"bind"',
            order=SyntacticOrder.PREFIX,
            associativity=SyntacticAssociativity.RIGHT,
        )
    ),

    "foldl": LangOp(
        op=foldl,
        syntax=Syntax(
            token='"foldl"',
            order=SyntacticOrder.PREFIX,
            associativity=SyntacticAssociativity.RIGHT,
        )
    ),
    
}



c3 = make_compiler([{**list_functions, **functors}, [number_arithmetic, number_literals, args]])

test_arithmetic_and_functors = lambda *args, **kvargs: test(c3, verbose=False, *args, **kvargs)

def vtest_af(tests):
    for text, input, output in tests:
        test_arithmetic_and_functors(text, input, output)

vtest_af([
    ("10", None, 10),
    ("10 + 10", None, 20),
    ("(_ + 1) | (_ * 2) | (10 / _) | (_ / 5)", 0, 1),
    ("_ + 1 | _ * 2 | 10 / _ | _ / 5", 0, 1),
    ("_ + 1 | ((_ * 2) | (10 / _)) | _ / 5", 0, 1),
    ("_ + 1 | ((_ * 2) | 10 / _) | _ / 5", 0, 1),
    ("_ + 1 | (_ * 2 | 10 / _) | _ / 5", 0, 1),
    ("_ + 1 | (_ * 2 | (10 / _)) | _ / 5", 0, 1),
    ("count | _ * 2", [0, 1, 2], 6),
    ("map (mapf (_) (_) (_) | count) | sum", [{}, {}, {}], 9),
    ("map (2 * _ | _ + 1)", [0, 1, 2], [1, 3, 5]),
    ("mapf (_ + 1) (_ - 1) (_ + 1 | _ * 2)", 0, [1, -1, 2]),
    ("map (_ + _) | sum", [0, 1, 2], 6),
    ("map ($0 + $0) | sum", [0, 1, 2], 6),
    ("bind ($0 + $0) 1", None, 2),
    ("bind ($0 + $1) 1 2", None, 3),
    ("bind ($0 + $1) 1 _", 2, 3),
    ("bind ($0 + $1) _ 1", 2, 3),
    ("bind ($0 + 1 + $1) 1 _", 1, 3),
    ("bind ($0 + 1) 1", None, 2),
    ("bind (_ + 1) 1", None, 2),
    ("foldl ($0*2 + $1 + 1) 0", [1, 2], 7),
    ("foldl ($0 * $1) 1", [1, 2, 3], 6),
    ("bind ($0 - $1) sum foldl ($0 + $1) 0", [1, 2, 3], 0),
    ("bind count (0 | mapf _ _ _)", None, 3),
    ("bind (mapf _ _ _) _ | count", 0, 3),
    ("bind count (0 | mapf _ _ _)", None, 3),
    ("bind ($0 - $1) (foldl ($0 + $1) 0) sum", [1, 2, 3], 0),
    ("foldl (bind ($0 ^ $1) $1 $0) 1", [1, 2], 2),
    ("foldl ($0 + $1) 0", list(range(10000)), sum(range(10000))),
    ("sum", list(range(10000)), sum(range(10000))),
    ("(sum) - (foldl ($0 + $1) 0)", [1, 2, 3], 0),
    ("map (map (_ ^ 2) | sum)", [[1,2], [0,1], [-2,-1]], [5, 1, 5]),
])


