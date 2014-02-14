# ------------------------------------------------------------
# calclex.py
#
# tokenizer for a simple expression evaluator for
# numbers and +,-,*,/
# ------------------------------------------------------------
import ply.lex as lex
import itertools
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




def p_freq_data_lines(t):
    """freq_data_lines : freq_data
                       | freq_data_lines freq_data
    """
    if len(t) == 2:
        t[0] = [t[1]]
    else:
        t[1].append(t[2])
        t[0] = t[1]

def p_freq_data(t):
    """freq_data : NUMBER data_lines
    """
    t[0] = (float(t[1]), t[2])

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
    """data_line : data NEWLINE
                 | data COMMENT NEWLINE
    """
    t[0] = t[1]

def p_data(t):
    """data : number_pair
            | number_pair number_pair
            | number_pair number_pair number_pair
            | number_pair number_pair number_pair number_pair
    """
    t[0] = tuple(itertools.chain(*t[1:]))


def p_number_pair(t):
    """number_pair : NUMBER NUMBER
    """
    t[0] = (float(t[1]), float(t[2]))


def p_error(t):
    print("Syntax error at '%s' on line: %s" % (t.value, t.lineno))

import ply.yacc as yacc
yacc.yacc(debug=1)
touch = """10 20 30
30 40
40.12 50
110 120 130
130 140
140.12 150
"""

numbers = """10 20
20 30
30 40"""

print yacc.parse(touch)

