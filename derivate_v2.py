import ast
import re


class DerivativeVisitor(ast.NodeVisitor):

    def __init__(self, func_str, diff_var):
        self.diff_var = diff_var
        self.terms = []

        if "=" in func_str:
            self.func_name, self.args, self.body = self._parse_function_definition(func_str)
            
        else:
            self.func_name = None
            self.args = [diff_var]
            self.body = func_str

        self.tree = ast.parse(self.body,mode="eval") # parse the body now


    def _parse_function_definition(self,def_str):
        left,right = def_str.split("=")
        left = left.strip()
        right = right.strip()

        name = left[:left.index("(")]
        args_str = left[left.index("(")+1 : left.index(")")]
        args = [arg.strip() for arg in args_str.split(",")]

        return name ,args, right
    

    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Add):
            self.visit(node.left)
            self.visit(node.right)

        elif isinstance(node.op, ast.Sub):
            self.visit(node.left)
            self.terms.append("NEGATE_NEXT")
            self.visit(node.right)
        
        elif isinstance(node.op, ast.Pow):
            if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Constant):
                var = node.left.id
                exp = node.right.value
                if var == self.diff_var:
                    self.terms.append(f"{exp}*{var}**{exp -1}")
                else:
                    self.terms.append("0")

        elif isinstance(node.op, ast.Div):
            u = ast.unparse(node.left)
            v = ast.unparse(node.right)

            du = DerivativeVisitor.__new__(DerivativeVisitor)
            du.diff_var = self.diff_var
            du.terms = []
            du.visit(node.left)
            du_term = " + ".join(du.terms) if du.terms else "0"

            dv = DerivativeVisitor.__new__(DerivativeVisitor)
            dv.diff_var = self.diff_var
            dv.terms = []
            dv.visit(node.right)
            dv_term = " + ".join(dv.terms) if dv.terms else "0"

            numerator = f"{v}*({du_term}) - {u}*({dv_term})"
            denominator = f"{v}**2"
            self.terms.append(f"({numerator})/({denominator})")

        elif isinstance(node.op, ast.FloorDiv):
            print("⚠️  Floor division (//) is not differentiable. Returning 0.")
            self.terms.append("0")

        elif isinstance(node.op, ast.Mod):
            print("⚠️  Modulo (%) is not differentiable in symbolic math. Returning 0.")
            self.terms.append("0")


        
        elif isinstance(node.op, ast.Mult):
            left = node.left
            right = node.right

            # Symbolic product rule
            if isinstance(left, (ast.Name, ast.Call)) and isinstance(right, (ast.Name, ast.Call)):
                # Differentiate both sides
                u = ast.unparse(left)
                v = ast.unparse(right)

                # Manually visit left and right to get derivative terms
                du = DerivativeVisitor.__new__(DerivativeVisitor)
                du.diff_var = self.diff_var
                du.terms = []
                du.visit(left)

                dv = DerivativeVisitor.__new__(DerivativeVisitor)
                dv.diff_var = self.diff_var
                dv.terms = []
                dv.visit(right)

                du_term = " + ".join(du.terms) if du.terms else "0"
                dv_term = " + ".join(dv.terms) if dv.terms else "0"

                left_term = f"({du_term})*{v}" if du_term != "0" else ""
                right_term = f"{u}*({dv_term})" if dv_term != "0" else ""

                combined = " + ".join(filter(None, [left_term, right_term]))
                self.terms.append(combined if combined else "0")

            if isinstance(left, ast.Constant) and isinstance(right, ast.Name):
                if right.id == self.diff_var:
                    self.terms.append(f"{left.value}")
                else:
                    self.terms.append("0")

            elif isinstance(left, ast.Name) and isinstance(right, ast.Name):
                if left.id == self.diff_var:
                    self.terms.append(f"{right.id}")
                elif right.id == self.diff_var:
                    self.terms.append(f"{left.id}")
                else:
                    self.terms.append("0")

            else:
                self.terms.append("0")

    def visit_Name(self, node):
        if node.id == self.diff_var:
            print(f"🚀 Found variable match: {node.id} == {self.diff_var} → 1")
            self.terms.append("1")
        else:
            print(f"🧩 Variable {node.id} is not diff target → 0")
            self.terms.append("0")




    def visit_Call(self,node):
        func_name = node.func.id
        #print("Function name is: ", func_name)
        rules = {
            'sin': lambda var: f"cos({var})",
            'cos': lambda var: f"-sin({var})",
            'tan': lambda var: f"1/cos({var})**2",      # or f"sec({var})**2"
            'log': lambda var: f"1/{var}",
            'exp': lambda var: f"exp({var})",
            'sqrt': lambda var: f"1/(2*sqrt({var}))",
            'sec': lambda var: f"sec({var})*tan({var})",
            'csc': lambda var: f"-csc({var})*cot({var})",
            'cot': lambda var: f"-1/sin({var})**2",
            'abs': lambda var: f"{var}/abs({var})"  # assuming var ≠ 0
        }


        if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
            var = node.args[0].id
            if var == self.diff_var:
                self.terms.append(rules.get(func_name, lambda v: "0")(var))
            else:
                self.terms.append("0")
        else:
            self.terms.append("0")
            
        


def pretty(term):
    superscripts = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³',
        '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷',
        '8': '⁸', '9': '⁹'
    }

    # Basic simplifications
    term = re.sub(r'\*1\b', '', term)     # *1 → nothing
    term = re.sub(r'\b1\*', '', term)     # 1* → nothing
    term = re.sub(r'\*0\b', '0', term)    # *0 → 0

    # Remove unnecessary +0 or 0+
    term = re.sub(r'\+ *0\b', '', term)
    term = re.sub(r'\b0 *\+', '', term)

    # Join constants with variables: 2*x → 2x
    term = re.sub(r'\b(\d+)\*([a-zA-Z(])', r'\1\2', term)

    # Remove **1 (e.g., x**1 → x)
    term = term.replace("**1", "")

    # Special simplification: (y*(1) - x*(0))/(y**2) → 1/y
    term = re.sub(r"\((\w+)\*\(1\) - \w+\*\(0\)\)/\(\1\*\*2\)", r"1/\1", term)

    # Convert **2, **3, etc. to superscripts
    term = re.sub(r'([a-zA-Z])\*\*(\d+)', 
                  lambda m: m.group(1) + ''.join(superscripts.get(ch, ch) for ch in m.group(2)), 
                  term)

    # Final cleanup: remove trailing *
    term = re.sub(r'\*$', '', term)

    return term








expr_str = input("Enter a function or expression: ")
diff_var = input("Differentiate with respect to: ")

if "=" in expr_str:
    # User entered a full function definition
    visitor = DerivativeVisitor(expr_str, diff_var)
    visitor.visit(visitor.tree)
    pretty_expr = visitor.body
else:
    # User entered just an expression
    tree = ast.parse(expr_str, mode="eval")
    visitor = DerivativeVisitor(expr_str,diff_var)
    visitor.visit(tree)
    pretty_expr = expr_str

# Shared formatting
filtered_terms = [t for t in visitor.terms if t != "0"]
pretty_terms = [pretty(t) for t in filtered_terms]
result = " + ".join(pretty_terms) if pretty_terms else "0"
print(f"d/d{diff_var} ({pretty(pretty_expr)}) = {result}")

