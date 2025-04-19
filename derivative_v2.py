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

            # Check whether either side contains the variable
            u = ast.unparse(left)
            v = ast.unparse(right)

            # Differentiate both sides
            du = DerivativeVisitor.__new__(DerivativeVisitor)
            du.diff_var = self.diff_var
            du.terms = []
            du.visit(left)
            du_term = " + ".join(du.terms) if du.terms else "0"

            dv = DerivativeVisitor.__new__(DerivativeVisitor)
            dv.diff_var = self.diff_var
            dv.terms = []
            dv.visit(right)
            dv_term = " + ".join(dv.terms) if dv.terms else "0"

            # Only apply product rule if BOTH terms involve diff_var
            if du_term != "0" and dv_term != "0":
                left_term = f"({du_term})*{v}" if du_term != "0" else ""
                right_term = f"{u}*({dv_term})" if dv_term != "0" else ""
                combined = " + ".join(filter(None, [left_term, right_term]))
                self.terms.append(combined if combined else "0")
            elif du_term != "0":
                self.terms.append(f"{du_term}*{v}")
            elif dv_term != "0":
                self.terms.append(f"{u}*{dv_term}")
            else:
                self.terms.append("0")

        
        

    def visit_Name(self, node):
        if node.id == self.diff_var:
            print(f"🚀 Found variable match: {node.id} == {self.diff_var} → 1")
            self.terms.append("1")
        else:
            print(f"🧩 Variable {node.id} is not diff target → 0")
            self.terms.append("0")

    @staticmethod
    def pretty(term):
        superscripts = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³',
            '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷',
            '8': '⁸', '9': '⁹'
        }

        # 1. Clean *1, 1*, *0
        term = re.sub(r'\*1\b', '', term)
        term = re.sub(r'\b1\*', '', term)
        term = re.sub(r'\*0\b', '0', term)

        # 2. Remove +0 and 0+
        term = re.sub(r'\+ *0\b', '', term)
        term = re.sub(r'\b0 *\+', '', term)

        # 3. Convert 2*x → 2x
        term = re.sub(r'\b(\d+)\*([a-zA-Z(])', r'\1\2', term)

        # 4. Remove dangling * at end or before closing parenthesis
        term = re.sub(r'\*$', '', term)
        term = re.sub(r'\*(\))', r'\1', term)
        term = re.sub(r'\(\s*([^\)]+)\*\)', r'(\1)', term)  # handles (2x*) too

        # 5. Special simplification: (y*(1) - x*(0))/(y**2) → 1/y
        term = re.sub(r"\((\w+)\*\(1\) - \w+\*\(0\)\)/\(\1\*\*2\)", r"1/\1", term)

        # 6. Convert **n to superscripts: x**2 → x²
        term = re.sub(r'([a-zA-Z])\s*\*\*\s*(\d+)', 
                  lambda m: m.group(1) + ''.join(superscripts.get(ch, ch) for ch in m.group(2)),
                  term)

        return term
   


    def visit_Call(self, node):
        func_name = node.func.id

        # Outer function rules
        rules = {
            'sin': lambda g: f"cos({g})",
            'cos': lambda g: f"-sin({g})",
            'tan': lambda g: f"1/cos({g})**2",
            'log': lambda g: f"1/({g})",
            'exp': lambda g: f"exp({g})",
            'sqrt': lambda g: f"1/(2*sqrt({g}))",
            'sec': lambda g: f"sec({g})*tan({g})",
            'csc': lambda g: f"-csc({g})*cot({g})",
            'cot': lambda g: f"-1/sin({g})**2",
            'abs': lambda g: f"{g}/abs({g})"
        }

        if len(node.args) != 1:
            self.terms.append("0")  # Unsupported multi-arg function
            return

        # Get inner expression (g(x))
        inner_node = node.args[0]
        g_expr = ast.unparse(inner_node)

        # Compute g'(x): visit inner node using a fresh visitor
        inner_visitor = DerivativeVisitor.__new__(DerivativeVisitor)
        inner_visitor.diff_var = self.diff_var
        inner_visitor.terms = []
        inner_visitor.visit(inner_node)
        g_prime = " + ".join(inner_visitor.terms) if inner_visitor.terms else "0"

        # Compute f'(g(x))
        #outer = rules.get(func_name, lambda g: "0")(g_expr)
        if func_name not in rules:
            print(f"⚠️ Unknown function '{func_name}' — treating derivative as 0.")
            self.terms.append("0")
            return

        outer_raw = rules[func_name](g_expr)
        outer = self.pretty(outer_raw)
        g_prime_pretty = self.pretty(g_prime)

        # Apply chain rule: f'(g(x)) * g'(x)
        if g_prime == "1":
            self.terms.append(outer)
        elif g_prime == "0":
            self.terms.append("0")
        else:
            self.terms.append(f"{outer}*({g_prime_pretty})")


            
        
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
#pretty_terms = [pretty(t) for t in filtered_terms]
pretty_terms = [DerivativeVisitor.pretty(t) for t in filtered_terms]
result = " + ".join(pretty_terms) if pretty_terms else "0"

print(f"d/d{diff_var} ({DerivativeVisitor.pretty(pretty_expr)}) = {result}")


