import ast
import re
from math import sin, log, cos, exp

class DerivativeVisitor(ast.NodeVisitor):
    def __init__(self, diff_var):
        self.diff_var = diff_var
        self.terms = []

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            self.visit(node.left)
            self.visit(node.right) 

        elif isinstance(node.op, ast.Sub):
            self.visit(node.left)
            self.terms.append("NEGATE_NEXT")  # or a marker to negate next term
            self.visit(node.right)

        
        elif isinstance(node.op, ast.Pow):
            if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Constant):
                var = node.left.id 
                exponent = node.right.value

                if var == self.diff_var:
                    new_exponent = exponent -1
                    derivative = f"{exponent}*{var}**{new_exponent}"
                    print(f"Found {var}**{exponent} -> Derivative: {derivative}")
                else:
                    derivative = "0"

                self.terms.append(derivative)

        elif isinstance(node.op, ast.Mult):
            if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Name):
                var = node.right.id
                if var == self.diff_var:
                    self.terms.append(f"{node.left.value}*1")
            
            elif isinstance(node.left, ast.Name) and isinstance(node.right, ast.Constant):
                var = node.left.id
                if var == self.diff_var:
                    self.terms.append(f"{node.right.value}*1")
                
                else:
                    self.terms.append("0")

            elif isinstance(node.left, ast.Constant) and isinstance(node.right, ast.BinOp):
                if isinstance(node.right.op, ast.Pow):
                    if isinstance(node.right.left, ast.Name) and isinstance(node.right.right, ast.Constant):
                        var = node.right.left.id
                        power = node.right.right.value
                    if var == self.diff_var:
                        new_power = power - 1
                        coeff = node.left.value * power
                        self.terms.append(f"{coeff}*{var}**{new_power}")
                    else:
                        self.terms.append("0")
            else:
                self.terms.append("0")
        else:
            self.terms.append("0")
            

        #if isinstance(node, ast.BinOp):
        #    self.generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Only support one argument for now
            if len(node.args) != 1:
                self.terms.append("0")  # or raise an error
                return

            arg = node.args[0]
            if isinstance(arg, ast.Name):
                var = arg.id
                if var == self.diff_var:
                    # Basic rules
                    if func_name == "sin":
                        self.terms.append(f"cos({var})")
                    elif func_name == "cos":
                        self.terms.append(f"-sin({var})")
                    elif func_name == "log":
                        self.terms.append(f"1/{var}")
                    elif func_name == "exp":
                        self.terms.append(f"exp({var})")
                    else:
                        self.terms.append("0")  # unknown function
                else:
                    self.terms.append("0")
            else:
                self.terms.append("0")



    def visit_Name(self, node):
        if node.id == self.diff_var:
            self.terms.append("1")
        else:
            self.terms.append("0")

    def visit_Constant(self, node):
        self.terms.append("0")



def pretty(term):
    # Superscripts dictionary
    superscripts = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³',
        '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷',
        '8': '⁸', '9': '⁹'
    }

    # 1. Replace x**n → xⁿ
    term = re.sub(
        r'([a-zA-Z])\*\*(\d+)',
        lambda m: m.group(1) + ''.join(superscripts.get(ch, ch) for ch in m.group(2)),
        term
    )

    # 2. Replace "*1" safely → remove only if it's at end
    term = re.sub(r'\*1\b', '', term)

    # 3. Replace "1*var" → "var"
    term = re.sub(r'\b1([a-zA-Z])', r'\1', term)

    # 4. Remove remaining "*" only if between var/const: e.g., "5*y" → "5y"
    term = term.replace("*", "")

    return term




func_str = input("Enter a function of x (e.g. x**2 + 1): ")

diff_var = input("Differantiate with respect to: ")

tree = ast.parse(func_str,mode='eval')

visitor = DerivativeVisitor(diff_var)
visitor.visit(tree)

#filtered_terms = [t for t in visitor.terms if t!="0"]

filtered_terms = []
negate = False
for t in visitor.terms:
    if t == "0":
        continue
    elif t == "NEGATE_NEXT":
        negate = True
    else:
        if negate:
            if t.startswith("-"):
                filtered_terms.append(t[1:])  # --2 becomes +2
            else:
                filtered_terms.append(f"-{t}")
            negate = False
        else:
            filtered_terms.append(t)




pretty_terms = [pretty(t) for t in filtered_terms]
result = " + ".join(pretty_terms) if pretty_terms else "0"
print(f"d/d{diff_var} ({pretty(func_str)}) = {result}")


