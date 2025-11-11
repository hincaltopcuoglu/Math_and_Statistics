"""
Definition 1.4.1

Example 1.4.1:
Rolling a Die. When a six-sided die is rolled, 
the sample space can be regarded as
containing the six numbers 1, 2, 3, 4, 5, 6, 
each representing a possible side of the die
that shows after the roll. Symbolically, we write
S = {1, 2, 3, 4, 5, 6}.
A-) Find even numbers using this set
B-) Find numbers greater than 2 using this set
"""

s = [1,2,3,4,5,6]

# A-)
def create_even_subset(s:list)->list:
    d = []
    for i in s:
        if i % 2 == 0:
            d.append(i)
    return d

#print(create_even_subset(s))


# B-)
def find_greater(s:list, target:int)->list:
    lst = []
    for i in s:
        if i > target:
            lst.append(i)
    return lst

#print(find_greater(s,2))

###########################################################################

"""
Definition 1.4.2

Containment:
It is said that a set A is contained in another set B if every element
of the set A also belongs to the set B. 
This relation between two events is expressed
symbolically by the expression A ⊂ B, 
which is the set-theoretic expression for saying
that A is a subset of B. 
Equivalently, if A ⊂ B, we may say that B contains A and may
write B ⊃ A

Problem 2: Write Containment function
"""

def is_containment(a,b)->bool:
    a_in_b = all(elem in b for elem in a)
    b_in_a = all(elem in a for elem in b)
    return {
        "A⊂B": a_in_b,
        "B⊂A": b_in_a
    }


#print(is_containment(a=[2,4,6], b=[2,3,4,5,6]))
#print(is_containment(a=[1,2,3,4,5],b=[1,2,3]))


###########################################################################


"""
Theorem 1.4.1

Let A, B, and C be events. Then A ⊂ S. If A ⊂ B and B ⊂ A, then A = B. If A ⊂ B
and B ⊂ C, then A ⊂ C

Problem 3: Proof containtments above
"""

def proof_containments(a,b,c):
    a_in_b = all(elem in b for elem in a)
    b_in_a = all(elem in a for elem in b)
    b_in_c = all(elem in c for elem in b)

    if a_in_b and b_in_a:
        return a,b
    elif a_in_b and b_in_c:
        return a,c
    else:
        return False



#print(proof_containments(a=[1,2,3],b=[3,1,2],c=[1,2,3,4,5,6,7,8,9]))
#print(proof_containments(a=[1,2,3],b=[1,2,3,4,5,6],c=[1,2,3,4,5,6,7,8,9]))
#print(proof_containments(a=[1,2,3,4,5],b=[1,2],c=[1,2,3,4,5,6,7,8,9]))

###########################################################################

"""
Definition 1.4.5

Complement. The complement of a set A is defined to be the set that contains all
elements of the sample space S that do not belong to A. The notation for the
complement of A is A_c

Problem 4: Proof Complement

"""

def is_complement(a,s):
    return [elem for elem in s if elem not in a]
        

#print(is_complement(a=[1,2,3,4],s=[1,2,3,4,5,9]))


###########################################################################

"""
Theorem 1.4.3

Problem 5:
Proof that Let A be an event. Then(A_c)_c = A -> complement of complement

"""

def comp_of_comp(a,s):
    comp = [elem for elem in s if elem not in a]
    return [c for c in s if c not in comp]



##print(comp_of_comp(a=[1,2,3,4],s=[1,2,3,4,5,9]))


###########################################################################


"""
Definition 1.4.6

Union of Two Sets. If A and B are any two sets, the union of A and B is defined to be
the set containing all outcomes that belong to A alone, to B alone, or to both A and
B. The notation for the union of A and B is A ∪ B
Problem 6: write code of it
"""

def proof_union_v1(l1,l2)->list:
    return list(set(l1) | set(l2))


def proof_union_v2(l1,l2)->list:
    seen = set()
    lst = []
    
    for elem in l1 + l2:
        if elem not in seen:
            lst.append(elem)
            seen.add(elem)
    
    return lst


#print(proof_union_v1(l1=[1,2,3,4],l2=[1,2,3,4,5,9]))
#print(proof_union_v2(l1=[1,2,3,4],l2=[1,2,3,4,5,9]))

###########################################################################

"""
Theorem 1.4.6

Associative Property. For every three events A, B, and C, the following associative
relations are satisfied:
A ∪ B ∪ C = (A ∪ B) ∪ C = A ∪ (B ∪ C)

Problem 7: write code of this proof
"""

def associative(a,b,c):
    a_union_b = list(set(a) | set(b))
    b_union_c = list(set(b) | set(c))
    ab_union_c = list(set(a_union_b) | set(c))
    a_union_bc = list(set(a) | set(b_union_c))
    direct = list(set(a) | set(b) | set(c))
    return sorted(a_union_bc) == sorted(ab_union_c) == sorted(direct)
    
    

#print(associative(a=[1,2,3,4,5],b=[1,2],c=[1,2,3,4,5,6,7,8,9]))
#print(associative(a=[7,9],b=[1,2],c=[4,5,6,7,8]))

###########################################################################



"""
Definition 1.4.8

Intersection of Two Sets. If A and B are any two sets, the intersection of A and B is
defined to be the set that contains all outcomes that belong both to A and to B. The
notation for the intersection of A and B is A ∩ B.

Problem 8: Prove that definition

"""

def intersect(a,b):

    intersect_ab = [elem for elem in a if elem in b]
    intersect_ba = [elem for elem in b if elem in a]
    
    if intersect_ab:
        return sorted(intersect_ab) == sorted(intersect_ba)


#print(intersect(a=[1,2,3,4],b=[1,2,3,4,5,9]))
#print(intersect(a=[1,2,3,4],b=[5,6,7,8,9]))


###########################################################################

"""
Theorem 1.4.8

Associative Property. For every three events A, B, and C, the following associative
relations are satisfied:
A ∩ B ∩ C = (A ∩ B) ∩ C = A ∩ (B ∩ C)

Problem 9: Prove this theorem
"""

def intersect_assc(a,b,c):
    a_int_b = list(set(a) & set(b))
    b_int_c = list(set(b) & set(c))
    a_int_c = list(set(a) & set(c))
    ab_int_c = list(set(a_int_b) & set(c))
    a_int_bc = list(set(a) & set(b_int_c)) 
    b_int_ac = list(set(b) & set(a_int_c))
    direct = list(set(a) & set(b) & set(c))

    return ab_int_c == a_int_bc == b_int_ac == direct, direct



#print(intersect_assc(a=[1,2,3,4,5],b=[1,2],c=[1,2,3,4,5,6,7,8,9]))


###########################################################################