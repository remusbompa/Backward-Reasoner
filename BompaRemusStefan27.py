#!/usr/bin/env python
# coding: utf-8

#    # Backward Reasoner

# ## Reprezentare prin formule logice

# - pentru reprezentarea atomilor, am folosit reprezentarea din laboratorul 5: tupluri avand
# ca prin element numele tipului (Constant, Var, Fun, Atom, Neg, And, Or) iar ca al doilea
# element valoarea simbolului (valoarea in cazul constantei, numele variabilei) sau un 
# tuplu in cazul in care sibolul ia argumente (cazul functie, atomului, and si or)
# - am folosit functii de creare formule (make), de verificare formule (is), de extragere valoare,
# nume, head (tip), argumente, inlocuire argumente, de afisare formula in format lizibil (
# print_formula)

# In[1]:


import sys
### Reprezentare - construcție

# întoarce un termen constant, cu valoarea specificată.
def make_const(value):
    return ("Constant", value)

# întoarce un termen care este o variabilă, cu numele specificat.
def make_var(name):
    return ("Var", name)

# întoarce un termen care este un apel al funcției specificate, pe restul argumentelor date.
# E.g. pentru a construi termenul add[1, 2, 3] vom apela
#  make_function_call(add, make_const(1), make_const(2), make_const(3))
# !! ATENȚIE: python dă args ca tuplu cu restul argumentelor, nu ca listă. Se poate converti la listă cu list(args)
def make_function_call(function, *args):
    return ("Fun", (function, args))

# întoarce o formulă formată dintr-un atom care este aplicarea predicatului dat pe restul argumentelor date.
# !! ATENȚIE: python dă args ca tuplu cu restul argumentelor, nu ca listă. Se poate converti la listă cu list(args)
def make_atom(predicate, *args):
    return ("Atom", (predicate, tuple(args)))

# întoarce o formulă care este negarea propoziției date.
# get_args(make_neg(s1)) va întoarce [s1]
def make_neg(sentence):
    return ("Neg", sentence)

# întoarce o formulă care este conjuncția propozițiilor date (2 sau mai multe).
# e.g. apelul make_and(s1, s2, s3, s4) va întoarce o structură care este conjuncția s1 ^ s2 ^ s3 ^ s4
#  și get_args pe această structură va întoarce [s1, s2, s3, s4]
def make_and(sentence1, sentence2, *others):
    l = [sentence1, sentence2]
    l += list(others)
    return ("And", tuple(l))

# întoarce o formulă care este disjuncția propozițiilor date.
# e.g. apelul make_or(s1, s2, s3, s4) va întoarce o structură care este disjuncția s1 V s2 V s3 V s4
#  și get_args pe această structură va întoarce [s1, s2, s3, s4]
def make_or(sentence1, sentence2, *others):
    l = [sentence1, sentence2]
    l += list(others)
    return ("Or", tuple(l))

# întoarce o copie a formulei sau apelul de funcție date, în care argumentele au fost înlocuite
#  cu cele din lista new_args.
# e.g. pentru formula p(x, y), înlocuirea argumentelor cu lista [1, 2] va rezulta în formula p(1, 2).
# Noua listă de argumente trebuie să aibă aceeași lungime cu numărul de argumente inițial din formulă.
def replace_args(formula, new_args):
    if is_function_call(formula) or is_atom(formula):
        head, (t, args) = formula
        if len(args) != len(new_args):
            return formula
        return (head, (t, new_args))
    
    if(is_sentence(formula)):
        t, body = formula
        if t == "Neg":
            if len(new_args) != 1:
                return formula
            return (t, new_args[0])
        args = []
        for sentence in body:
            args.append(sentence)
        if len(args) != len(new_args):
            return formula
        return (t, tuple(new_args))
    return formula
### Reprezentare - verificare

# întoarce adevărat dacă f este un termen.
def is_term(f):
    return is_constant(f) or is_variable(f) or is_function_call(f)

# întoarce adevărat dacă f este un termen constant.
def is_constant(f):
    if len(f) != 2:
        return False
    t, val = f
    if t == "Constant":
        return True
    return False

# întoarce adevărat dacă f este un termen ce este o variabilă.
def is_variable(f):
    if len(f) != 2:
        return False
    t, val = f
    if t == "Var":
        return True
    return False

# întoarce adevărat dacă f este un apel de funcție.
def is_function_call(f):
    if len(f) != 2:
        return False
    t, val = f
    if t == "Fun":
        return True
    return False

# întoarce adevărat dacă f este un atom (aplicare a unui predicat).
def is_atom(f):
    if len(f) != 2:
        return False
    t, val = f
    if t == "Atom":
        return True
    return False

# întoarce adevărat dacă f este o propoziție validă.
def is_sentence(f):
    if len(f) != 2:
        return False
    if is_atom(f):
        return True
    (t, val) = f
    if t == "And" or t == "Neg" or t == "Or":
        return True
    return False

# întoarce adevărat dacă formula f este ceva ce are argumente.
def has_args(f):
    return is_function_call(f) or is_sentence(f)


### Reprezentare - verificare

# pentru constante (de verificat), se întoarce valoarea constantei; altfel, None.
def get_value(f):
    if is_constant(f):
        (t, value) = f
        return value
    return None

# pentru variabile (de verificat), se întoarce numele variabilei; altfel, None.
def get_name(f):
    if is_variable(f):
        (t, var) = f
        return var
    return None

# pentru apeluri de funcții, se întoarce funcția;
# pentru atomi, se întoarce numele predicatului; 
# pentru propoziții compuse, se întoarce un șir de caractere care reprezintă conectorul logic (e.g. ~, A sau V);
# altfel, None
def get_head(f):
    if(is_function_call(f) or is_atom(f)):
        (head, body) = f
        (t, args) = body
        return t
    if is_sentence(f):
        (head, body) = f
        if head == "And":
            return "A"
        elif head == "Or":
            return "V"
        else:
            return "~"
    return None

# pentru propoziții sau apeluri de funcții, se întoarce lista de argumente; altfel, None.
# Vezi și "Important:", mai sus.
def get_args(f):
    if is_function_call(f) or is_atom(f):
        head, (t, args) = f
        return list(args)
    
    if(is_sentence(f)):
        t, body = f
        if t == "Neg":
            return [body]
        args = []
        for sentence in body:
            args.append(sentence)
        return args
    return []

def is_positive_literal(L):
    return is_atom(L)
def is_negative_literal(L):
    return get_head(L) == "~" and is_positive_literal(get_args(L)[0])
def is_literal(L):
    return is_positive_literal(L) or is_negative_literal(L)

def get_opposite(L):
    if is_positive_literal(L):
        return make_neg(L)
    return get_args(L)[0]

PREFIX_VAR = "prefix_"

# Afișează formula f. Dacă argumentul return_result este True, rezultatul nu este afișat la consolă, ci întors.
def print_formula(f, dict_name_fct, return_result = False):
    ret = ""
    if is_term(f):
        if is_constant(f):
            ret += str(get_value(f))
        elif is_variable(f):
            name = get_name(f)
            if(name[:len(PREFIX_VAR)] == PREFIX_VAR):
                name = name[len(PREFIX_VAR):]
            ret += "?" + name
        elif is_function_call(f):
            ret += str(dict_name_fct[get_head(f)]) + "[" + "".join([print_formula(arg, dict_name_fct, True) + "," for arg in get_args(f)])[:-1] + "]"
        else:
            ret += "???"
    elif is_atom(f):
        ret += str(get_head(f)) + "(" + "".join([print_formula(arg, dict_name_fct, True) + ", " for arg in get_args(f)])[:-2] + ")"
    elif is_sentence(f):
        # negation, conjunction or disjunction
        args = get_args(f)
        if len(args) == 1:
            ret += str(get_head(f)) + print_formula(args[0], dict_name_fct, True)
        else:
            ret += "(" + str(get_head(f)) + "".join([" " + print_formula(arg, dict_name_fct, True) for arg in get_args(f)]) + ")"
    else:
        ret += "???"
    if return_result:
        return ret
    print(ret)
    return


# Pentru unificarea a doua formule am folosit functia unify:
# - compara fiecare subformula din cadrul formulei: daca formulele au argumente compara header,
# simbol (functie/ predicat) si apoi argumentele pe rand (sunt puse in stiva perechile de 
# argumente), daca se ajunge la 2 constante diferite se opreste (intoarce False), daca se ajunge
# la 2 variabile cea din formula a 2-a va indica catre cea din prima formula, iar daca se ajunge
# la o variabila si o formula (variabila nu se gaseste in fomula, verificare folosind 
# occur_check), variabila va fi substituita cu formula
# - intoarce substitutia construita subst prin efect lateral
# - deoarece variabilele din a 2-a formula vor fi intotdeauna diferite de cele din prima
# formula (a 2-a formula este un fapt, variabilele nu sunt instantiate), daca se ajunge la
# o variabila din a 2-a formula, aceasta va fi prefixata cu PREFIX_VAR = "prefix_". Acest prefix
# va disparea la apelul substitutie, cand cele care impreuna cu prefixul nu se gasesc in
# substitutie vor fi pastrate fara prefix.
# - intoarce ca rezultat substitutia gasita pentru variabilele din prima formula (data ca 
# parametru): pentru fiecare variabila din setul de variabile a primei formule se aplica
# functia substf1_not_in_f2, care intoarce o formula formata numai din constante sau None daca
# s-a ajuns la o formula ce contine cel putin o variabila.
# -cele 2 rezultate vor fi folosite:
# 	*) subst e folosita pentru a substitui formula a 2-a (concluzia/faptul gasit) cu variabile
# 	din prima formula (teorema, premisa) si constante
# 	*)subst_f1 (rezultatul) e folosit pentru a updata dictionarul de substitutii 
# 	(new_subst.update(crt_subst)), astfel incat sa se tina evidenta variabilelor substituite
# 	din fiecare teorema/premisa

# In[2]:


def substitute(f, substitution, firstVar = True):
    if substitution is None:
        return None
    if is_variable(f):
        new_f = f
        if firstVar:
            new_f = make_var(PREFIX_VAR + get_name(f))
        if get_name(new_f) in substitution:
            return substitute(substitution[get_name(new_f)], substitution, False)
    if has_args(f):
        return replace_args(f, tuple(substitute(arg, substitution, firstVar) for arg in get_args(f)))
    return f

def occur_check(v, t, subst):
    if v == t:
        return True
    if is_variable(t) and t[1] in subst:
        return occur_check(v, subst[t[1]], subst)
    if is_function_call(t):
        [head, [f, args]] = t
        for sentence in args:
            if occur_check(v, sentence, subst):
                return True
    return False

def get_variables(f, res):
    if is_variable(f):
        res.add(get_name(f))
    elif has_args(f):
        for arg in get_args(f):
            get_variables(arg, res)

def substf1_not_in_f2(f, substitution):
    if substitution is None:
        return None
    if is_variable(f) and (get_name(f) in substitution):
        return substf1_not_in_f2(substitution[get_name(f)], substitution)
    if has_args(f):
        new_args = []
        for arg in get_args(f):
            new_arg = substf1_not_in_f2(arg, substitution)
            if new_arg == None:
                return None
            new_args.append(new_arg)
        return replace_args(f, tuple(new_args))
    if is_variable(f):
        return None
    return f

def unify(f1, f2, subst = None):
    if subst is None:
        subst = {}
            
    set_vars_f1 = set()
    
    stack = []
    stack.append((f1, f2))
    while stack:
        (f1, f2) = stack.pop()
        if is_variable(f2):
            f2 = make_var(PREFIX_VAR + get_name(f2))
        while is_variable(f1) and f1[1] in subst:
            f1 = subst[f1[1]]
            set_vars_f1.add(f1)
        while is_variable(f2) and f2[1] in subst:
            f2 = subst[f2[1]]
        if is_variable(f1) and is_variable(f2):
            if occur_check(f2, f1, subst):
                return False
            else:
                subst[f2[1]] = f1
                set_vars_f1.add(f1)
        elif is_variable(f1):
            if occur_check(f1, f2, subst):
                return False
            else:
                subst[f1[1]] = f2
                set_vars_f1.add(f1)
        elif is_variable(f2):
            if occur_check(f2, f1, subst):
                return False
            else:
                subst[f2[1]] = f1

        elif is_constant(f1) and is_constant(f2):
            if f1[1] == f2[1]:
                continue
            else:
                return False

        elif f1[0] == f2[0] and len(f1[1]) == len(f2[1]):
            if is_atom(f1) and f1[1][0] == f2[1][0] and len(f1[1][1]) == len(f2[1][1]):
                for i in range(len(f1[1][1])):
                    stack.append((f1[1][1][i], f2[1][1][i]))
            elif get_head(f1) == 'A' or get_head(f1) == "V":
                for i in range(len(f1[1])):
                    stack.append((f1[1][i], f2[1][i]))
            else:
                stack.append((f1[1], f2[1]))
        else:
            return False
    
    subst_f1 = {}
    for var in set_vars_f1:
        res = substf1_not_in_f2(var, subst)
        if res != None:
            subst_f1[var[1]] = res
    return subst_f1


# In[3]:


from copy import deepcopy

def get_premises(formula):
    if get_head(formula) == "V":
        premises = []
        elements = get_args(formula)
        for element in elements:
            if is_negative_literal(element):
                premises.append(get_args(element)[0])
        return premises
    return []

def get_conclusion(formula):
    if get_head(formula) == "V":
        elements = get_args(formula)
        for element in elements:
            if is_positive_literal(element):
                return element
    return None

def is_fact(formula):
    return is_positive_literal(formula)

def is_rule(formula):
    return len(get_premises(formula)) >= 1

def equal_terms(t1, t2):
    if is_constant(t1) and is_constant(t2):
        return get_value(t1) == get_value(t2)
    if is_variable(t1) and is_variable(t2):
            return True
    if is_function_call(t1) and is_function(t2):
        if get_head(t1) != get_head(t2):
            return all([equal_terms(get_args(t1)[i], get_args(t2)[i]) for i in range(len(get_args(t1)))])
    return False

def is_equal_to(a1, a2):
    # verificăm atomi cu același nume de predicat și același număr de argumente
    if not (is_atom(a1) and is_atom(a2) and get_head(a1) == get_head(a2) and len(get_args(a1)) == len(get_args(a2))):
        return False
    return all([equal_terms(get_args(a1)[i], get_args(a2)[i]) for i in range(len(get_args(a1)))])


# Functii de inserat, cautat, iterat prin fapte:
# - faptele sunt organizate intr-un dictionar list_facts avand dret cheie predicatul atomului.
# Valoarea unei chei este o lista de atomi avand predicatul respectiv. Aceasta organizare ajuta
# la cautarea eficienta a unei fapte care sa unifice cu teorema (trebuie sa aiba acelasi predicat)

# In[4]:


def insert_facts(facts, fact):
    predicate = get_head(fact)
    if predicate not in facts:
        facts[predicate] = []
    facts[predicate].append(fact)
    
def find_facts(facts, fact):
    predicate = get_head(fact)
    if predicate not in facts:
        return False
    return fact in facts[predicate]

def get_possible_facts(facts, fact):
    predicate = get_head(fact)
    if predicate not in facts:
        return []
    return facts[predicate]

def print_facts(facts, fout):
    list_facts = []
    for predicate in facts:
        list_facts += [prFormula(fact) for fact in facts[predicate]]
    print(list_facts, file = fout)


# # Cerinta1 (Citire fisier intrare)

# - citirea din fisier s-a facut parsand fiecare linie in functie de gramatica data, folosind o 
# stiva din care se pun sau se scot, se unesc si se pun termeni in functie de tipul de paranteza
# la care s-a ajuns sau de caracterul la care s-a ajuns. Faptele se pun in dictionarul de fapte
# list_facts, regulile in lista list_rules, interogarile in lista list_queries iar coeficientii
# in dictionarul dict_coef avand drept chei fapte/reguli.

# In[5]:


def parseLine(line, list_facts, list_rules, list_quries, dict_coef, dict_fct):
    start = 0
    my_list = list_facts
    coef = 1.0
    stack = []
    for i in range(len(line) + 1):
        if i == 0 and line[0] == '?':
            my_list = list_queries
            start = 1
            continue
        term_str = line[start:i].strip()
        if i == len(line) or line[i] == ',' or                 line[i] == '#' or (len(stack) == 0 and line[i] == ':'):
            if not term_str:
                if i == len(line) or line[i] == '#' or line[i] == ':':
                    break
                start = i + 1
                continue #a fost un apel de functie
            if term_str[0] == '?':       
                stack.append(make_var(term_str[1:]))
            else:
                stack.append(make_const(term_str))
            if i == len(line) or line[i] == '#' or line[i] == ':':
                break
            start = i + 1
        elif line[i] == '(':
            if term_str in dict_fct:
                stack.append((dict_fct[term_str], "Fun"))
            else:
                stack.append((term_str, "Atom"))
            stack.append('(')
            start = i + 1
        elif line[i] == ')':
            if term_str: #nu a fost un apel de functie
                if term_str[0] == '?':       
                    stack.append(make_var(term_str[1:]))
                else:
                    stack.append(make_const(term_str))
           
            list_args = []
            while True:
                top = stack.pop()
                if top == '(':
                    break
                list_args.insert(0, top)
            (name, type_name) = stack.pop()
            if type_name == "Fun":
                stack.append(make_function_call(name, *list_args))
            elif type_name == "Atom":
                stack.append(make_atom(name, *list_args))
            start = i + 1
        elif line[i] == ':':
            my_list = list_rules
            start = i + 1
        elif line[i] == '%':
            start = len(line)
            term_str = line[i+1:].strip()
            coef = float(term_str)
            
    if len(stack) == 0:
        return
    
    if my_list == list_facts:
        fact = stack.pop()
        insert_facts(list_facts, fact)
        dict_coef[fact] = coef
    elif my_list == list_rules:
        hypotheses = []
        while len(stack) > 1:
            hypotheses.insert(0, stack.pop())
        conclusion = stack.pop()
        rule = make_or(*([make_neg(s) for s in hypotheses] + [conclusion]))
        list_rules.append(rule)
        dict_coef[rule] = coef
    else:
        query = stack.pop()
        list_queries.append(query)
    
def readFile(filename, list_facts, list_rules, list_quries, dict_coef, dict_fct):
    with open(filename) as file:
        for line in file:
            parseLine(line, list_facts, list_rules, list_quries, dict_coef, dict_fct)


# In[6]:


dict_fct = {}
dict_name_fct = {}

def get(T, L):
    L = int(L)
    if L == 0:
        return T[0:2]
    elif L == 1:
        return T[1:3]
    elif L == 2:
        return T[0] + T[2]
    return None
dict_fct['get'] = get
dict_name_fct[get] = 'get'
def compute_triangle(L1, L2, L3):
    L1 = int(L1)
    L2 = int(L2)
    L3 = int(L3)
    Lmax = max([L1, L2, L3])
    return L1 + L2 + L3 - 2 * Lmax
dict_fct['compute_triangle'] = compute_triangle
dict_name_fct[compute_triangle] = 'compute_triangle'
def getShortest(L1, L2, L3):
    L1 = int(L1)
    L2 = int(L2)
    L3 = int(L3)
    return min([L1, L2, L3])
dict_fct['getShortest'] = getShortest
dict_name_fct[getShortest] = 'getShortest'
def getLongest(L1, L2, L3):
    L1 = int(L1)
    L2 = int(L2)
    L3 = int(L3)
    return max([L1, L2, L3])
dict_fct['getLongest'] = getLongest
dict_name_fct[getLongest] = 'getLongest'
def getMiddle(L1, L2, L3):
    L1 = int(L1)
    L2 = int(L2)
    L3 = int(L3)
    if (L1 <= L2 and L2 <= L3) or (L3 <= L2 and L2 <= L1):
        return L2
    elif (L1 <= L3 and L3 <= L2) or (L2 <= L3 and L3 <= L1):
        return L3
    else:
        return L1
dict_fct['getMiddle'] = getMiddle
dict_name_fct[getMiddle] = 'getMiddle'
def compute_pitagoras(LS, LM, LL):
    LS = int(LS)
    LM = int(LM)
    LL = int(LL)
    return LS*LS + LM*LM - LL*LL
dict_fct['compute_pitagoras'] = compute_pitagoras
dict_name_fct[compute_pitagoras] = 'compute_pitagoras'

prFormula = lambda fact: print_formula(fact, dict_name_fct, True)

def evaluate(formula, indent = 0, fout = sys.stdout, insideFunction = False):
    if is_constant(formula):
        return formula
    if is_variable(formula):
        return False if insideFunction else formula
    new_args = []
    for arg in get_args(formula):
        if is_function_call(formula):
            new_arg = evaluate(arg, indent, fout, True)
        else:
            new_arg = evaluate(arg, indent, fout, insideFunction)
        if not new_arg:
            return False
        new_args.append(new_arg)
    if is_function_call(formula):
        result = make_const(str(get_head(formula)(*[get_value(arg) for arg in new_args])))
        print(INDENT * indent + prFormula(formula) + " => " + prFormula(result), file = fout)
        return result
    return replace_args(formula, tuple(new_args))


# In[7]:


INDENT = "  "
SOL_INDENT ="**" 

#functii de printare: lista de formule, substitutie, lista de substitutii
def print_formula_list(formula_list):
    str_formula_list = ""
    for formula in formula_list:
        str_formula_list += prFormula(formula) + ", "
    return str_formula_list[:-2]

def print_subst(subst, theorem = None):
    print_sbst = ""
    if not subst:
        return "{}"
    
    set_vars = set()
    if theorem != None :
        get_variables(theorem, set_vars)
    for var in sorted(subst, key= str.lower):
        if theorem != None :
            if var not in set_vars:
                continue
            else:
                print_sbst += "?" + var + ": " + prFormula(subst[var]) + ", "
        else:
            print_sbst += "?" + var + ": " + prFormula(subst[var]) + ", "
    if not print_sbst:
        return "{}"
    return print_sbst[:-2]

def print_list_subst(list_subst, theorem = None):
    text = ""

    set_subst = set()
    set_vars = set()
    if theorem != None :
        get_variables(theorem, set_vars)
        
    for (subst, coef) in list_subst:
        if theorem != None:
            for key in list(subst):
                if key not in set_vars:
                    del subst[key]
        if (str(subst), coef) in set_subst:
            continue
        set_subst.add((str(subst), coef))
        text += print_subst(subst, theorem) + " % " + str(coef) + "\n"
    return text[:-1]


# # Cerinta 2 (Backward chaining)

# In[8]:


def get_subst_bkt(i, premises, facts, conclusion, rules, crt_subst, list_sol, rule_coef,                   dict_coef, min_coef, indent, fout):
    if i == len(premises):
        coef = rule_coef * min_coef
        if not find_facts(facts, conclusion):
            insert_facts(facts, conclusion)
        print(INDENT * (indent - 1) + "Solutie: " + print_subst(crt_subst) + " % " + str(coef), file = fout)
        list_sol.append((crt_subst, coef))
        dict_coef[conclusion] = coef
        return True
        
    eval_premise = evaluate(premises[i], indent-1, fout)
    if eval_premise != False and eval_premise != premises[i]:
        premises[i] = eval_premise
        print(INDENT * (indent-1) + "Scopuri de demonstrat: " +  print_formula_list(premises[i:]), file = fout)

    unified = False
    ret = False
    for fact in get_possible_facts(facts, premises[i]):
        subst = {}
        new_subst = unify(premises[i], fact, subst)
        if new_subst != False:
            found_fact = substitute(fact, subst)
            if not find_facts(facts, found_fact):
                insert_facts(facts, found_fact)
                dict_coef[found_fact] = dict_coef[fact]
            print(INDENT * indent + prFormula(found_fact) + " % " + str(dict_coef[found_fact]) +                   " este un fapt", file = fout)             

            new_conclusion = substitute(conclusion, new_subst, False)
            new_premises = [substitute(premise, new_subst, False) for premise in premises]
            new_subst.update(crt_subst)
            new_min_coef = min(min_coef, dict_coef[fact])
            if get_subst_bkt(i+1, new_premises, facts, new_conclusion, rules, new_subst, list_sol,                              rule_coef, dict_coef, new_min_coef, indent + 1, fout):
                ret = True
            unified = True
    #daca premisa premises[i] nu unifica cu niciun fapt, incerc sa-i aflu valoarea
    if not unified:
        if backward_chaining(facts, rules, premises[i], dict_coef, fout, indent, crt_subst):
            return get_subst_bkt(i, premises, facts, conclusion, rules, crt_subst, list_sol,                                 rule_coef, dict_coef, min_coef, indent, fout)
        else:
            return False
    return ret

def backward_chaining(facts, list_rules, theorem, dict_coef, fout, indent = 0, crt_subst = {}):
    initial_indent = indent
    ret = False
    list_sol = []
    if initial_indent == 0:
        print(INDENT * indent + "Scopuri de demonstrat: " + prFormula(theorem), file = fout)
        indent += 1
    for fact in get_possible_facts(facts, theorem):
        subst = unify(theorem, fact)
        if subst != False:
            print(INDENT * indent + prFormula(fact) + " % " + str(dict_coef[fact]) +                       " este un fapt" + (" dat" if initial_indent == 0 else ""), file = fout)
            if subst:
                print(INDENT * indent + "Solutie: " + print_subst(subst, theorem) + " % " + str(dict_coef[fact]), file = fout)
            list_sol.append((subst, dict_coef[fact]))
            ret = True
            
    for rule in list_rules:
        # Pentru fiecare regulă
        premises = get_premises(rule)
        conclusion = get_conclusion(rule)
        conclusion_subst = {}
        crt_subst = unify(theorem, conclusion, conclusion_subst)
        #daca regula nu ajunge la teorema incerc alta
        if crt_subst == False:
            continue
            
        new_conclusion = substitute(conclusion, conclusion_subst)
        new_premises = [substitute(premise, conclusion_subst) for premise in premises]
            
        print_premises = print_formula_list(new_premises)
        print(INDENT * indent + "Incercam " + prFormula(new_conclusion) + ": " +               print_premises + " % " + str(dict_coef[rule]), file = fout)

        #obtin substitutiile posibile pentru ca fiecare premisa sa unifice cu un fapt
        if get_subst_bkt(0, new_premises, facts, new_conclusion,                          list_rules, crt_subst, list_sol, dict_coef[rule], dict_coef, 1.0, indent + 1, fout):       
            ret = True
    
    if initial_indent == 0:
        print(INDENT * initial_indent + "Gata.", file = fout)
    if ret:
        new_coef = 0
        for (_, coef) in list_sol:
            new_coef = new_coef + coef - new_coef * coef
        dict_coef[theorem] = new_coef
        if(initial_indent == 0):
            print(SOL_INDENT + "The theorem " + prFormula(theorem) + " is TRUE!" + " % " + str(new_coef), file = fout)
            print(print_list_subst(list_sol, theorem), file = fout)
            print(file = fout)
        else:
            print(INDENT * initial_indent + "The theorem " + prFormula(theorem) + " is TRUE!", file = fout)
        return ret
    if(initial_indent == 0):
        print(SOL_INDENT + "The theorem " + prFormula(theorem) + " is FALSE!", file = fout)
        print(file = fout)
    else:
        print(INDENT * initial_indent + "The theorem " + prFormula(theorem) + " is FALSE!", file = fout)
    return False


# In[9]:


import os
import time
from datetime import timedelta

input_path = './teste/'
output_path = './output/'
prefix_output_file = "output_"

if not os.path.exists(output_path):
    os.makedirs(output_path)
    
for file in os.listdir(input_path):
    list_facts = {}
    list_rules = []
    list_queries = []
    dict_coef = {}
    start_time = time.time()
    readFile(input_path + file, list_facts, list_rules, list_queries, dict_coef, dict_fct)
    fout = open(output_path + prefix_output_file + file, 'w')
    print("List facts:", file = fout)
    print_facts(list_facts, fout)
    print(file = fout)
    print("List rules:", file = fout)
    print([prFormula(rule) for rule in list_rules], file = fout)
    print(file = fout)
    print("List queries:", file = fout)
    print([prFormula(query) for query in list_queries], file = fout)
    print(file = fout)
    #print("List coef:", file = fout)
    #print([prFormula(formula) + ": " + str(dict_coef[formula]) for formula in dict_coef], file = fout)
    print(file = fout)
    for query in list_queries:
        backward_chaining(deepcopy(list_facts), list_rules, query, deepcopy(dict_coef), fout)
    fout.close()
    
    elapsed_time = time.time() - start_time
    print(file + "-"*20 + " " + str(timedelta(seconds=elapsed_time)))

