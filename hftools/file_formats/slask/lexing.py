# ------------------------------------------------------------
# calclex.py
#
# tokenizer for a simple expression evaluator for
# numbers and +,-,*,/
# ------------------------------------------------------------
import ply.lex as lex

# List of token names.   This is always required
tokens = (
   'NUMBER',
   'NEWLINE',
   'INFOLINE',
   'COMMENT',
)

# A regular expression rule with some action code
def t_NUMBER(t):
    r'[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
    t.value = float(t.value)    
    return t


def t_INFOLINE(t):
    r'[ \t]*\#[^!]*'
    return t

def t_COMMENT(t):
    r'[ \t]*!.*'
    return t


# Define a rule so we can track line numbers and newlines
def t_NEWLINE(t):
    r'\n+'
    t.lexer.lineno += len(t.value)
    return t

# A string containing ignored characters (spaces and tabs)
t_ignore  = ' \t'

# Error handling rule
def t_error(t):
    raise lex.LexError("Illegal character '%s' on line %s" % (t.value[0], t.lexer.lineno), t)
# Build the lexer
lexer = lex.lex()

data = open("../../data/active-numbers.s2p").read()


def p_touchstone(t):
    """touchstone : touchstone1
                  | comments touchstone1
    """
    if len(t) == 3:
        t[0] = (t[1],) + t[2]
    else:
        t[0] = ([],) + t[2]


def p_touchstone1(t):
    """touchstone1 : infoline data_lines
                   | infoline data_lines comments
    """
    t[0] = (t[1], t[2])


def p_infoline(t):
    """infoline : INFOLINE NEWLINE
                | INFOLINE COMMENT NEWLINE
    """
    t[0] = t[1]


def p_comments(t):
    """comments : comment
                | comments comment"""
    if len(t) == 2:
        t[0] = t[1]
    else:
        t[0] = t[1] + t[2]


def p_comment(t):
    """comment : COMMENT
               | COMMENT NEWLINE"""
    t[0] = [t[1]]


def p_data_lines(t):
    """data_lines : data_line
                  | data_lines data_line
    """
    if len(t) == 2:
        t[0] = [t[1]]
    else:
        t[1].append(t[2])
        t[0] = t[1]


def p_data_line(t):
    """data_line : numbers
                 | numbers NEWLINE
                 | numbers COMMENT NEWLINE
    """
    t[0] = t[1]


def p_numbers(t):
    """numbers : NUMBER
               | numbers NUMBER
    """
    if len(t) == 2:
        t[0] = [t[1]]
    else:
        t[1].append(t[2])
        t[0] = t[1]


def p_error(t):
    print("Syntax error at '%s'" % t.value)

import ply.yacc as yacc
yacc.yacc(debug=0)
touch = """!kalle:10
!kula:20
#asjdkljads ljk asdl !asdlkjasd
10 20  !asdau
20 30
30 40,12
!alskdlöaksd
"""

numbers = """10 20
20 30
30 40"""

print yacc.parse(touch)

