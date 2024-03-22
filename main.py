#opParser
import networkx as nx
import numpy as np
from networkx.algorithms.shortest_paths.generic import shortest_path_length

def check_string():
    operators = list(input("Enter the operators used in the given grammar: "))
    operators.append('$')
    print(operators)

    terminals = list("abcdefghijklmnopqrstuvwxyz")
    symbols = list('(/*%+-)')
    precedence_table = np.empty([len(operators) + 1, len(operators) + 1], dtype=str, order="F")

    for j in range(1, len(operators) + 1):
        precedence_table[0][j] = operators[j - 1]
        precedence_table[j][0] = operators[j - 1]

    for i in range(1, len(operators) + 1):
        for j in range(1, len(operators) + 1):
            if (precedence_table[i][0] in terminals) and (precedence_table[0][j] in terminals):
                precedence_table[i][j] = ""
            elif (precedence_table[i][0] in terminals):
                precedence_table[i][j] = ">"
            elif (precedence_table[i][0] in symbols) and (precedence_table[0][j] in symbols):
                if (symbols.index(precedence_table[i][0]) <= symbols.index(precedence_table[0][j])):
                    precedence_table[i][j] = ">"
                else:
                    precedence_table[i][j] = "<"
            elif (precedence_table[i][0] in symbols) and precedence_table[0][j] in terminals:
                precedence_table[i][j] = "<"
            elif precedence_table[i][0] == "$" and precedence_table[0][j] != "$":
                precedence_table[i][j] = "<"
            elif precedence_table[0][j] == "$" and precedence_table[i][0] != "$":
                 precedence_table[i][j] = ">"
            else:
                break

    print("The Operator Precedence Relational Table:")
    print(precedence_table)

    string_to_check = list(input("Enter the string to be checked: "))
    string_to_check.append("$")
    stack = [None] * len(string_to_check)
    stack_index = 0
    stack.insert(stack_index, "$")
    row_labels = [row[0] for row in precedence_table]
    column_labels = list(precedence_table[0])
    string_index = 0

    while stack[0] != stack[1]:
        if string_to_check[len(string_to_check) - 2] in symbols:
            break
        elif (stack[stack_index] in row_labels) and (string_to_check[string_index] in column_labels):
            if precedence_table[row_labels.index(stack[stack_index])][column_labels.index(string_to_check[string_index])] == "<":
                stack_index += 1
                stack.insert(stack_index, string_to_check[string_index])
                string_index += 1
            elif precedence_table[row_labels.index(stack[stack_index])][column_labels.index(string_to_check[string_index])] == ">":
                stack.pop(stack_index)
                stack_index -= 1
            elif (precedence_table[row_labels.index(stack[stack_index])][column_labels.index(string_to_check[string_index])] == '') and ((stack[stack_index] == "$") and (string_to_check[string_index] == "$")):
                stack[1] = stack[0]
        else:
            break

    if stack[0] != stack[1]:
        return False
    else:
        return True

def check_grammar(i):
    print("Enter production ", str(i + 1))
    production_rule = list(input().split("->"))
    terminals = list("abcdefghijklmnopqrstuvwxyz")

    if (production_rule[0] == " " or production_rule[0] == "" or production_rule[0] in terminals or len(production_rule) == 1):
        return False
    else:
        production_rule.pop(0)
        production_rule = list(production_rule[0])
        non_terminals = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        symbols = list("(abcdefghijklmnopqrstuvwxyz^/*+-|),")
        special_characters = ['!', '@', '#', '$', '?', '~', '`', ',', ';', ':', '"', '=', '_', '&', "'", "", " "]

        for i in range(0, len(production_rule), 2):
            if (production_rule[i] == " "):
                is_valid = False
            elif (production_rule[i] in special_characters):
                is_valid = False
                break
            elif (production_rule[len(production_rule) - 1] in symbols) and ((production_rule[0] == "(" and production_rule[len(production_rule) - 1] == ")") or (production_rule.count("(") == production_rule.count(")"))):
                is_valid = True
            elif (production_rule[i] in terminals):
                is_valid = True
            elif (production_rule[len(production_rule) - 1] in symbols):
                is_valid = False
            elif ((i == len(production_rule) - 1) and (production_rule[i] in non_terminals)):
                is_valid = True
            elif ((i == len(production_rule) - 1) and (production_rule[i] not in non_terminals) and (production_rule[i] in symbols) and production_rule[i - 1] in symbols):
                is_valid = True
            elif ((production_rule[i] in non_terminals) and(production_rule[i + 1] in symbols)):
                is_valid = True
            elif ((production_rule[i] in non_terminals) and (production_rule[i + 1] in non_terminals)):
                is_valid = False
                break
            else:
                is_valid = False
                break

        if (is_valid == True):
            return True
        else:
            return False

num_lhs_variables = int(input("Enter the number of productions: "))

for i in range(num_lhs_variables):
    if check_grammar(i):
        grammar_accepted = True
    else:
        grammar_accepted = False
        break

if grammar_accepted:
    print("Grammar is accepted")

    if check_string():
        print("String is accepted")
    else:
        print("String is not accepted")
else:
    print("Grammar is not accepted")


def create_edges(precedence_table):
    edges = []

    for i in range(1, len(precedence_table)):
        for j in range(1, len(precedence_table[i])):
            if precedence_table[i][j] == '<':
                source = f'f{precedence_table[i][0]}'
                destination = f'g{precedence_table[0][j]}'
                edges.append((source, destination))
            elif precedence_table[i][j] == '>':
                source = f'g{precedence_table[0][j]}'
                destination = f'f{precedence_table[i][0]}'
                edges.append((source, destination))

    return edges

precedence_table = [
    ['', '+', '*', 'a', '$'],
    ['+', '', '', '<', '>'],
    ['*', '', '', '>', '>'],
    ['a', '>', '', '', '>'],
    ['$', '<', '<', '<', '']
]

edges = create_edges(precedence_table)
for edge in edges:
    print(f'Edge: {edge[0]} -> {edge[1]}')

print(edges)

G = nx.DiGraph()
G.add_edges_from(edges)

f_nodes = [node for node in G.nodes if node.startswith('f')]
g_nodes = [node for node in G.nodes if node.startswith('g')]

max_lengths_f = {}
max_lengths_g = {}

for node in f_nodes:
    max_lengths_f[node] = max(shortest_path_length(G, node).values())

for node in g_nodes:
    max_lengths_g[node] = max(shortest_path_length(G, node).values())

print("Maximum path lengths from 'f' nodes:")
for node, length in max_lengths_f.items():
    print(f"{node}: {length}")

print("\nMaximum path lengths from 'g' nodes:")
for node, length in max_lengths_g.items():
    print(f"{node}: {length}")
#------------------------op parser-------------------

#------------sr parser--------------------------------
'''
# example 2
gram = {
	"S":["S+S","S*S","i"]
}
starting_terminal = "S"
inp = "i+i*i"
'''
gram = {
	"E":["E+T", "T"],
    "T":["T*F", "F"],
    "F":["(E)", "i"]
}
starting_terminal = "E"
inp = "i+i*i"

stack = "$"
print(f'{"Stack": <15}'+"|"+f'{"Input Buffer": <15}'+"|"+f'Parsing Action')
print(f'{"-":-<50}')

while True:
	action = True
	i = 0
	while i<len(gram[starting_terminal]):
		if gram[starting_terminal][i] in stack:
			stack = stack.replace(gram[starting_terminal][i],starting_terminal)
			print(f'{stack: <15}'+"|"+f'{inp: <15}'+"|"+f'Reduce S->{gram[starting_terminal][i]}')
			i=-1
			action = False
		i+=1
	if len(inp)>1:
		stack+=inp[0]
		inp=inp[1:]
		print(f'{stack: <15}'+"|"+f'{inp: <15}'+"|"+f'Shift')
		action = False

	if inp == "$" and stack == ("$"+starting_terminal):
		print(f'{stack: <15}'+"|"+f'{inp: <15}'+"|"+f'Accepted')
		break

	if action:
		print(f'{stack: <15}'+"|"+f'{inp: <15}'+"|"+f'Rejected')
		break
#------------------------------sr parser-------------------------

#------------------------pr parser----------------------------
#Some helper functions
def print_iter(Matched,Stack,Input,Action,verbose=True):
    if verbose==True:
        print(".".join(Matched).ljust(30)," | ",".".join(Stack).ljust(25)," | ",".".join(Input).ljust(30)," | ",Action)
#The predictive parsing algorithm
def predictive_parsing(sentence,parsingtable,terminals,start_state="S",verbose=True):      #Set verbose to false to not see the stages of the algorithm
    status = None
    match = []
    stack = [start_state,"$"]
    Inp = sentence.split(".")
    if verbose==True:
        print_iter(["Matched"],["Stack"],["Input"],"Action")
    print_iter(match,stack,Inp,"Initial",verbose)
    action=[]
    while(len(sentence)>0 and status!=False):
        top_of_input = Inp[0]
        pos = top_of_input
        if stack[0] =="$" and pos == "$" :
            print_iter(match,stack,Inp,"Accepted",verbose)
            return "Accepted"
        if stack[0] == pos:
            print_iter(match,stack,Inp,"Pop",verbose)
            match.append(stack[0])
            del(stack[0])
            del(Inp[0])
            continue
        if stack[0]=="epsilon":
            print_iter(match,stack,Inp,"Poping Epsilon",verbose)
            del(stack[0])
            continue
        try:
            production=parsingtable[stack[0]][pos]
            print_iter(match,stack,Inp,stack[0]+" -> "+production,verbose)
        except:
            return "error for "+str(stack[0])+" on "+str(pos),"Not Accepted"

        new = production.split(".")   
        stack=new+stack[1:]
    return "Not Accepted"

if __name__=="__main__":
    parsingtable = {
    "E" : {"id" : "T.E1", "(" : "T.E1"},
    "E1" : {"+":"+.T.E1", ")":"epsilon", "$" : "epsilon"},
    "T" : {"id" : "F.T1", "(" : "F.T1" },
    "T1" : {"+" : "epsilon", "*" : "*.F.T1", ")" : "epsilon", "$" : "epsilon"},
    "F":{"id":"id","(":"(.E.)"}
    }
    terminals = ["id","(",")","+","*"]
    print(predictive_parsing(sentence="id.+.(.id.+.id.).$",parsingtable=parsingtable,terminals=terminals,start_state="E",verbose=True))
    #Another Example done in class:-
    #print(predictive_parsing(sentence="c.c.c.c.d.d.$",parsingtable={"S" : {"c":"C.C","d":"C.C"},"C":{"c":"c.C","d":"d"}},terminals=["c,d"],start_state="S"))

#-------------------------------------------------
# #example for direct left recursion
# gram = {"A":["Aa","Ab","c","d"]
# }
#example for indirect left recursion
gram = {
	"E":["E+T","T"],
	"T":["T*F","F"],
	"F":["(E)","i"]
}

def removeDirectLR(gramA, A):   #remove direct left recursion
	"""gramA is dictonary"""
	temp = gramA[A]
	tempCr = []
	tempInCr = []
	for i in temp:
		if i[0] == A:
			#tempInCr.append(i[1:])
			tempInCr.append(i[1:]+[A+"'"])
		else:
			#tempCr.append(i)
			tempCr.append(i+[A+"'"])
	tempInCr.append(["e"])
	gramA[A] = tempCr
	gramA[A+"'"] = tempInCr
	return gramA


def checkForIndirect(gramA, a, ai):
	if ai not in gramA:
		return False 
	if a == ai:
		return True
	for i in gramA[ai]:
		if i[0] == ai:
			return False
		if i[0] in gramA:
			return checkForIndirect(gramA, a, i[0])
	return False

def rep(gramA, A):   
	temp = gramA[A]
	newTemp = []
	for i in temp:
		if checkForIndirect(gramA, A, i[0]):
			t = []
			for k in gramA[i[0]]:
				t=[]
				t+=k
				t+=i[1:]
				newTemp.append(t)

		else:
			newTemp.append(i)
	gramA[A] = newTemp
	return gramA

def rem(gram):      #remove indirect left recursion
	c = 1
	conv = {}
	gramA = {}
	revconv = {}
	for j in gram:
		conv[j] = "A"+str(c)
		gramA["A"+str(c)] = []
		c+=1

	for i in gram:
		for j in gram[i]:
			temp = []	
			for k in j:
				if k in conv:
					temp.append(conv[k])
				else:
					temp.append(k)
			gramA[conv[i]].append(temp)


	print(gramA)
	for i in range(c-1,0,-1):
		ai = "A"+str(i)
		for j in range(0,i):
			aj = gramA[ai][0][0]
			if ai!=aj :
				if aj in gramA and checkForIndirect(gramA,ai,aj):
					gramA = rep(gramA, ai)

	for i in range(1,c):
		ai = "A"+str(i)
		for j in gramA[ai]:
			if ai==j[0]:
				gramA = removeDirectLR(gramA, ai)
				break

	op = {}
	for i in gramA:
		a = str(i)
		for j in conv:
			a = a.replace(conv[j],j)
		revconv[i] = a

	for i in gramA:
		l = []
		for j in gramA[i]:
			k = []
			for m in j:
				if m in revconv:
					k.append(m.replace(m,revconv[m]))
				else:
					k.append(m)
			l.append(k)
		op[revconv[i]] = l

	return op

result = rem(gram)
terminals = []
for i in result:
	for j in result[i]:
		for k in j:
			if k not in result:
				terminals+=[k]
terminals = list(set(terminals))
#print(terminals)

def first(gram, term):
	a = []
	if term not in gram:
		return [term]
	for i in gram[term]:
		if i[0] not in gram:
			a.append(i[0])
		elif i[0] in gram:
			a += first(gram, i[0])
	return a

firsts = {}
for i in result:
	firsts[i] = first(result,i)
	print(f'First({i}):',firsts[i])

def follow(gram, term):
	a = []
	for rule in gram:
		for i in gram[rule]:
			if term in i:
				temp = i
				indx = i.index(term)
				if indx+1!=len(i):
					if i[-1] in firsts:
						a+=firsts[i[-1]]
					else:
						a+=[i[-1]]
				else:
					a+=["e"]
				if rule != term and "e" in a:
					a+= follow(gram,rule)
	return a

follows = {}
for i in result:
	follows[i] = list(set(follow(result,i)))
	if "e" in follows[i]:
		follows[i].pop(follows[i].index("e"))
	follows[i]+=["$"]
	print(f'Follow({i}):',follows[i])

resMod = {}
for i in result:
	l = []
	for j in result[i]:
		temp = ""
		for k in j:
			temp+=k
		l.append(temp)
	resMod[i] = l

# create predictive parsing table
tterm = list(terminals)
tterm.pop(tterm.index("e"))
tterm+=["$"]
pptable = {}
for i in result:
	for j in tterm:
		if j in firsts[i]:
			pptable[(i,j)]=resMod[i][0]
		else:
			pptable[(i,j)]=""
	if "e" in firsts[i]:
		for j in tterm:
			if j in follows[i]:
				pptable[(i,j)]="e" 	
pptable[("F","i")] = "i"
toprint = f'{"": <10}'
for i in tterm:
	toprint+= f'|{i: <10}'
print(toprint)
for i in result:
	toprint = f'{i: <10}'
	for j in tterm:
		if pptable[(i,j)]!="":
			toprint+=f'|{i+"->"+pptable[(i,j)]: <10}'
		else:
			toprint+=f'|{pptable[(i,j)]: <10}'
	print(f'{"-":-<76}')
	print(toprint)
#--------------------------------pr parser--------------------------

#----------------------------rec descent-------------------------------
print("Recursive Desent Parsing For following grammar\n")
print("E->TE'\nE'->+TE'/@\nT->FT'\nT'->*FT'/@\nF->(E)/i\n")

global s
global i
i=0
s=list(input("Enter the string want to be checked: \n"))

def match(a):
    global s
    global i
    if(i>=len(s)):
        return False
    elif(s[i]==a):
        i+=1
        return True
    else:
        return False
    
def F():
    if(match("(")):
        if(E()):
            if(match(")")):
                return True
            else:
                return False
        else:
            return False
    elif(match("i")):
        return True
    else:
        return False
    
def Tx():
    if(match("*")):
        if(F()):
            if(Tx()):
                return True
            else:
                return False
        else:
            return False
    else:
        return True
    
def T():
    if(F()):
        if(Tx()):
            return True
        else:
            return False
    else:
        return False
    
def Ex():
    if(match("+")):
        if(T()):
            if(Ex()):
                return True
            else:
                return False
        else:
            return False
    else:
        return True
    
def E():
    if(T()):
        if(Ex()):
            return True
        else:
            return False
    else:
        return False
    
if(E()):
    if(i==len(s)):
        print("String is accepted")
    else:
         print("String is not accepted")
    
else:
    print("string is not accepted")
#-----------------------------rec descent--------------------------------------

#---------------------leading trailing----------------------------
a = ["E=E+T",
     "E=T",
     "T=T*F",
     "T=F",
     "F=(E)",
     "F=i"]

rules = {}   #to store grammar rules
NT = []      #to store NTs

for i in a:
    temp = i.split("=")
    NT.append(temp[0])
    try:
        rules[temp[0]] += [temp[1]]
    except:
        rules[temp[0]] = [temp[1]]

NT = list(set(NT))   #to remove duplicates
print(rules,NT)

def leading(gram, rules, term, start):
    s = []
    if gram[0] not in NT:
        return gram[0]
    elif len(gram) == 1:
        return [0]
    elif gram[1] not in NT and gram[-1] is not start:
        for i in rules[gram[-1]]:
            s+= leading(i, rules, gram[-1], start)
            s+= [gram[1]]
        return s

def trailing(gram, rules, term, start):
    s = []
    if gram[-1] not in NT:
        return gram[-1]
    elif len(gram) == 1:
        return [0]
    elif gram[-2] not in NT and gram[-1] is not start:

        for i in rules[gram[-1]]:
            s+= trailing(i, rules, gram[-1], start)
            s+= [gram[-2]]
        return s

leads = {}
trails = {}
for i in NT:
    s = [0]
    for j in rules[i]:
        s+=leading(j,rules,i,i)
    s = set(s)
    s.remove(0)
    leads[i] = s
    s = [0]
    for j in rules[i]:
        s+=trailing(j,rules,i,i)
    s = set(s)
    s.remove(0)
    trails[i] = s

for i in NT:
    print("LEADING("+i+"):",leads[i])
for i in NT:
    print("TRAILING("+i+"):",trails[i])
#--------------------------------leading trailing-----------------------

#-------------------------------first follow--------------------------
# #example for direct left recursion
# gram = {"A":["Aa","Ab","c","d"]
# }
#example for indirect left recursion
gram = {
	"E":["E+T","T"],
	"T":["T*F","F"],
	"F":["(E)","i"]
}

def removeDirectLR(gramA, A):
	"""gramA is dictonary"""
	temp = gramA[A]
	tempCr = []
	tempInCr = []
	for i in temp:
		if i[0] == A:
			#tempInCr.append(i[1:])
			tempInCr.append(i[1:]+[A+"'"])
		else:
			#tempCr.append(i)
			tempCr.append(i+[A+"'"])
	tempInCr.append(["e"])
	gramA[A] = tempCr
	gramA[A+"'"] = tempInCr
	return gramA


def checkForIndirect(gramA, a, ai):
	if ai not in gramA:
		return False 
	if a == ai:
		return True
	for i in gramA[ai]:
		if i[0] == ai:
			return False
		if i[0] in gramA:
			return checkForIndirect(gramA, a, i[0])
	return False

def rep(gramA, A):
	temp = gramA[A]
	newTemp = []
	for i in temp:
		if checkForIndirect(gramA, A, i[0]):
			t = []
			for k in gramA[i[0]]:
				t=[]
				t+=k
				t+=i[1:]
				newTemp.append(t)

		else:
			newTemp.append(i)
	gramA[A] = newTemp
	return gramA

def rem(gram):
	c = 1
	conv = {}
	gramA = {}
	revconv = {}
	for j in gram:
		conv[j] = "A"+str(c)
		gramA["A"+str(c)] = []
		c+=1

	for i in gram:
		for j in gram[i]:
			temp = []	
			for k in j:
				if k in conv:
					temp.append(conv[k])
				else:
					temp.append(k)
			gramA[conv[i]].append(temp)


	#print(gramA)
	for i in range(c-1,0,-1):
		ai = "A"+str(i)
		for j in range(0,i):
			aj = gramA[ai][0][0]
			if ai!=aj :
				if aj in gramA and checkForIndirect(gramA,ai,aj):
					gramA = rep(gramA, ai)

	for i in range(1,c):
		ai = "A"+str(i)
		for j in gramA[ai]:
			if ai==j[0]:
				gramA = removeDirectLR(gramA, ai)
				break

	op = {}
	for i in gramA:
		a = str(i)
		for j in conv:
			a = a.replace(conv[j],j)
		revconv[i] = a

	for i in gramA:
		l = []
		for j in gramA[i]:
			k = []
			for m in j:
				if m in revconv:
					k.append(m.replace(m,revconv[m]))
				else:
					k.append(m)
			l.append(k)
		op[revconv[i]] = l

	return op

result = rem(gram)


def first(gram, term):
	a = []
	if term not in gram:
		return [term]
	for i in gram[term]:
		if i[0] not in gram:
			a.append(i[0])
		elif i[0] in gram:
			a += first(gram, i[0])
	return a

firsts = {}
for i in result:
	firsts[i] = first(result,i)
	print(f'First({i}):',firsts[i])
# 	temp = follow(result,i,i)
# 	temp = list(set(temp))
# 	temp = [x if x != "e" else "$" for x in temp]
# 	print(f'Follow({i}):',temp)

def follow(gram, term):
	a = []
	for rule in gram:
		for i in gram[rule]:
			if term in i:
				temp = i
				indx = i.index(term)
				if indx+1!=len(i):
					if i[-1] in firsts:
						a+=firsts[i[-1]]
					else:
						a+=[i[-1]]
				else:
					a+=["e"]
				if rule != term and "e" in a:
					a+= follow(gram,rule)
	return a

follows = {}
for i in result:
	follows[i] = list(set(follow(result,i)))
	if "e" in follows[i]:
		follows[i].pop(follows[i].index("e"))
	follows[i]+=["$"]
	print(f'Follow({i}):',follows[i])
#-----------------------first follow----------------------

#-------------------------left recursion----------------
def remove_left_recursion(grammar):
    non_terminals = list(grammar.keys())
    updated_grammar = {}

    for A in non_terminals:
        productions = grammar[A]
        updated_productions = []
        new_A = A + "'"
        alpha_productions = []
        beta_productions = []

        for production in productions:
            if production[0] == A:
                alpha_productions.append(production[1:])
            else:
                beta_productions.append(production)

        if len(alpha_productions) > 0:
            updated_productions.extend([production + new_A for production in beta_productions])
            updated_productions.extend([production + new_A for production in alpha_productions])
            updated_productions.append('$')

            updated_grammar[A] = [production for production in updated_productions if production != '$']
            updated_grammar[new_A] = [production for production in beta_productions] + ['$']
        else:
            updated_grammar[A] = productions

    return updated_grammar


def eliminate_indirect_recursion(grammar):
    non_terminals = list(grammar.keys())
    updated_grammar = grammar.copy()

    for i in range(len(non_terminals)):
        A = non_terminals[i]

        for j in range(i):
            B = non_terminals[j]
            productions_A = updated_grammar[A]
            productions_B = updated_grammar[B]

            new_productions = []

            for production_A in productions_A:
                if production_A[0] == B:
                    for production_B in productions_B:
                        new_productions.append(production_B + production_A[1:])
                else:
                    new_productions.append(production_A)

            updated_grammar[A] = new_productions

    return updated_grammar


def main():
    grammar = {
        'S': ['a', '^', '(T)'],
        'T': ['T,S', 'S']
    }

    print("Original Grammar:")
    for non_terminal, productions in grammar.items():
        print(non_terminal, '->', '|'.join(productions))

    grammar = remove_left_recursion(grammar)
    grammar = eliminate_indirect_recursion(grammar)

    print("\nGrammar after removing left recursion:")
    for non_terminal, productions in grammar.items():
        print(non_terminal, '->', '|'.join(productions))


if __name__ == "__main__":
    main()
#-----------------------left recursion--------------------------

#------------------------left factoring-----------------------
from itertools import takewhile

s= "S->iEtS|iEtSeS|a"

def groupby(ls):
    d = {}
    ls = [ y[0] for y in rules ]
    initial = list(set(ls))
    for y in initial:
        for i in rules:
            if i.startswith(y):
                if y not in d:
                    d[y] = []
                d[y].append(i)
    return d

def prefix(x):
    return len(set(x)) == 1


starting=""
rules=[]
common=[]
alphabetset=["A'","B'","C'","D'","E'","F'","G'","H'","I'","J'","K'","L'","M'","N'","O'","P'","Q'","R'","S'","T'","U'","V'","W'","X'","Y'","Z'"]
s = s.replace(" ", "").replace("	", "").replace("\n", "")

while(True):
    rules=[]
    common=[]
    split=s.split("->")
    starting=split[0]
    for i in split[1].split("|"):
        rules.append(i)

#logic for taking commons out
    for k, l in groupby(rules).items():
        r = [l[0] for l in takewhile(prefix, zip(*l))]
        common.append(''.join(r))
#end of taking commons
    for i in common:
        newalphabet=alphabetset.pop()
        print(starting+"->"+i+newalphabet)
        index=[]
        for k in rules:
            if(k.startswith(i)):
                index.append(k)
        print(newalphabet+"->",end="")
        for j in index[:-1]:
            stringtoprint=j.replace(i,"", 1)+"|"
            if stringtoprint=="|":
                print("\u03B5","|",end="")
            else:
                print(j.replace(i,"", 1)+"|",end="")
        stringtoprint=index[-1].replace(i,"", 1)+"|"
        if stringtoprint=="|":
            print("\u03B5","",end="")
        else:
            print(index[-1].replace(i,"", 1)+"",end="")
        print("")
    break

#! curl https://raw.githubusercontent.com/samyuktaakannan/all/main/main.py | clip
#-----------------------------------left Factoring---------------------------

