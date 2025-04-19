import ast
import re
import math
from tabulate import tabulate
import operator
from itertools import combinations_with_replacement

class DerivativeVisitor(ast.NodeVisitor):

    def __init__(self, func_str, diff_var, chain_expand=False):
        self.chain_expand = chain_expand
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
            left_visitor = DerivativeVisitor.__new__(DerivativeVisitor)
            left_visitor.diff_var = self.diff_var
            left_visitor.terms = []
            left_visitor.visit(node.left)
            left_expr = " + ".join(left_visitor.terms) if left_visitor.terms else "0"

            right_visitor = DerivativeVisitor.__new__(DerivativeVisitor)
            right_visitor.diff_var = self.diff_var
            right_visitor.terms = []
            right_visitor.visit(node.right)
            right_expr = " + ".join(right_visitor.terms) if right_visitor.terms else "0"

            self.terms.append(f"({left_expr}) - ({right_expr})")


        elif isinstance(node.op, ast.Pow):
            if isinstance(node.left, ast.Name) and isinstance(node.right, ast.Constant):
                var = node.left.id
                exp = node.right.value
                if var == self.diff_var:
                    if exp - 1 == 0:
                        self.terms.append(f"{exp}")
                    else:
                        self.terms.append(f"{exp}*{var}**{exp - 1}")
                else:
                    self.terms.append("0")  # ← this was missing!

        

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


    def visit_Constant(self, node):
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

        term = re.sub(r'\b[a-zA-Z]\*\*0\b', '1', term)
        term = re.sub(r'[a-zA-Z]0\b', '', term)  

        # 6. Convert **n to superscripts: x**2 → x²
        term = re.sub(r'([a-zA-Z])\s*\*\*\s*(\d+)', 
                  lambda m: m.group(1) + ''.join(superscripts.get(ch, ch) for ch in m.group(2)),
                  term)

        return term
   
    def visit_Call(self, node):
        func_name = node.func.id
        if len(node.args) != 1:
            self.terms.append("0")
            return

        g_node = node.args[0]
        g_expr = ast.unparse(g_node)

        # Recursively get derivative of inner expression g(x)
        #g_prime_visitor = DerivativeVisitor.__init__(g_prime_visitor, f"f({', '.join(self.args)}) = {g_expr}", self.diff_var, self.chain_expand)
        #g_prime_visitor = DerivativeVisitor(f"f({', '.join(self.args)}) = {g_expr}", self.diff_var, chain_expand=self.chain_expand)
        g_prime_visitor = DerivativeVisitor(f"f({', '.join(self.args)}) = {g_expr}", self.diff_var, chain_expand=self.chain_expand)


        #g_prime_visitor.diff_var = self.diff_var
        #g_prime_visitor.terms = []
        #g_prime_visitor.chain_expand = self.chain_expand
        #g_prime_visitor.args = self.args
        #g_prime_visitor.body = g_expr
        #g_prime_visitor.tree = ast.parse(g_expr, mode="eval")
        g_prime_visitor.visit(g_prime_visitor.tree)
        g_prime_expr = " + ".join(g_prime_visitor.terms)

        chain_rules = {
            'sin': lambda g: f"cos({g})",
            'cos': lambda g: f"-sin({g})",
            'log': lambda g: f"1/({g})",
            'exp': lambda g: f"exp({g})"
        }

        if func_name not in chain_rules:
            print(f"⚠️ Unknown function '{func_name}' — treating derivative as 0.")
            self.terms.append("0")
            return

        outer_prime = chain_rules[func_name](g_expr)

        if self.chain_expand:
            self.terms.append(f"({outer_prime}) * ({g_prime_expr})")
        else:
            if g_prime_expr == "1":
                self.terms.append(outer_prime)
            elif g_prime_expr == "0":
                self.terms.append("0")
            else:
                self.terms.append(f"{outer_prime} * ({g_prime_expr})")

    def symbolic_chain_expand(self):
        print("\n🔗 Symbolic Chain Rule Expansion:")
        for var in self.args:
            expanded = DerivativeVisitor(f"f({', '.join(self.args)}) = {self.body}", var, chain_expand=True)
            expanded.visit(expanded.tree)
            expr = " + ".join(expanded.terms)
            print(f"∂f/∂{var} = {expr}")

       


    

    def validate_domain(self, context):

        safe_globals = {name: getattr(math, name) for name in dir(math) if not name.startswith("__")}
        has_issue = False

        class DomainValidator(ast.NodeVisitor):
            def visit_Call(self, node):
                nonlocal has_issue
                func_name = node.func.id if isinstance(node.func, ast.Name) else None
                if func_name in ("log", "sqrt"):
                    expr_str = ast.unparse(node.args[0])
                    try:
                        value = eval(expr_str, safe_globals, context)
                        if func_name == "log" and value <= 0:
                            print(f"⚠️ log({expr_str}) = {value:.4f} → undefined (≤ 0)")
                            has_issue = True
                        elif func_name == "sqrt" and value < 0:
                            print(f"⚠️ sqrt({expr_str}) = {value:.4f} → undefined (< 0)")
                            has_issue = True
                    except Exception as e:
                        print(f"⚠️ Could not evaluate {func_name}({expr_str}): {e}")
                        has_issue = True
                self.generic_visit(node)

            def visit_BinOp(self, node):
                nonlocal has_issue
                if isinstance(node.op, ast.Div):
                    denom_str = ast.unparse(node.right)
                    try:
                        denom_val = eval(denom_str, safe_globals, context)
                        if denom_val == 0:
                            print(f"⚠️ Division by zero: denominator {denom_str} = 0")
                            has_issue = True
                    except Exception as e:
                        print(f"⚠️ Could not evaluate denominator {denom_str}: {e}")
                        has_issue = True
                self.generic_visit(node)

        DomainValidator().visit(self.tree)
        return not has_issue



    def evaluate_limit_theorem(self, symbolic_expr, var, context, deltas):

        safe_globals = {name: getattr(math, name) for name in dir(math) if not name.startswith("__")}

        print("\n📐 Limit-based Approximation of Derivative:")
        print(f"Variable: {var}, Point: {context[var]}")

        print(f"Symbolic derivative at {var} = {context[var]}: ", end="")

        # Evaluate symbolic derivative
        try:
            
            symbolic_value = eval(symbolic_expr, safe_globals, context)
            print(f"{symbolic_value:.10f}\n")
        except Exception as e:
            print(f"❌ Error evaluating symbolic expression: {e}")
            return

        # Sort deltas
        deltas = sorted(deltas, reverse=True)

        print(f"{'Δ':>10} | {'(f(x+Δ) - f(x))/Δ':>25}")
        print("-" * 40)

        for delta in deltas:
            try:
                # Copy the context
                ctx_plus = context.copy()
                ctx_base = context.copy()

                # Only modify the variable we're differentiating with respect to
                ctx_plus[var] += delta

                # Try evaluating f(x + δ) and f(x)
                debug_expr = f"sin({ctx_plus[var]} ** 2 + 1)"
                debug_val = eval(debug_expr, safe_globals, ctx_plus)
                #print(f"Δ = {delta}, sin(...) = {debug_val}")
                
                f_plus = eval(self.body, safe_globals, ctx_plus)
                f_base = eval(self.body, safe_globals, ctx_base)

                # Ensure they are numbers
                if not isinstance(f_plus, (int, float)) or not isinstance(f_base, (int, float)):
                    raise ValueError("Non-numeric result")

                # Check for NaN or domain-sensitive outputs
                if math.isnan(f_plus) or math.isnan(f_base):
                    raise ValueError("Result is NaN")
                if math.isinf(f_plus) or math.isinf(f_base):
                    raise ValueError("Result is infinite")

                approx = (f_plus - f_base) / delta
                print(f"{delta:10.6f} | {approx:25.10f}")

            except Exception as e:
                print(f"{delta:10.6f} | ❌ Error: {e}")


    def evaluate_central_difference(self, var, context, deltas):
    
        safe_globals = {name: getattr(math, name) for name in dir(math) if not name.startswith("__")}

        print("\n📐 Central Difference Approximation of Derivative:")
        print(f"Variable: {var}, Point: {context[var]}")
        print(f"{'Δ':>10} | {'(f(x+Δ) - f(x-Δ)) / 2Δ':>30}")
        print("-" * 45)

        for delta in sorted(deltas, reverse=True):
            try:
                ctx_plus = context.copy()
                ctx_minus = context.copy()

                ctx_plus[var] += delta
                ctx_minus[var] -= delta

                f_plus = eval(self.body, safe_globals, ctx_plus)
                f_minus = eval(self.body, safe_globals, ctx_minus)

                approx = (f_plus - f_minus) / (2 * delta)
                print(f"{delta:10.6f} | {approx:30.10f}")
            except Exception as e:
                print(f"{delta:10.6f} | ❌ Error: {e}")


    def evaluate_both_differences(self, symbolic_value, var, context, deltas):

        safe_globals = {name: getattr(math, name) for name in dir(math) if not name.startswith("__")}

        print("\n📐 Derivative Approximations")
        print(f"Variable: {var}, Point: {context[var]}")
        print(f"True (Symbolic): {symbolic_value:.10f}\n")

        print(f"{'Δ':>10} | {'Forward Diff':>16} | {'Central Diff':>16}")
        print("-" * 47)

        for delta in sorted(deltas, reverse=True):
            try:
                ctx_plus = context.copy()
                ctx_minus = context.copy()

                ctx_plus[var] += delta
                ctx_minus[var] -= delta

                f_plus = eval(self.body, safe_globals, ctx_plus)
                f_base = eval(self.body, safe_globals, context)
                f_minus = eval(self.body, safe_globals, ctx_minus)

                forward = (f_plus - f_base) / delta
                central = (f_plus - f_minus) / (2 * delta)

                print(f"{delta:10.6f} | {forward:16.10f} | {central:16.10f}")

            except Exception as e:
                print(f"{delta:10.6f} | ❌ Error: {e}")


    def evaluate_gradient(self, context):
        print("\n📐 Gradient Evaluation:")

        # Extract all variable names from the context
        vars_to_diff = list(context.keys())
        gradient_vector = []

        for var in vars_to_diff:
            partial = DerivativeVisitor(f"f({', '.join(vars_to_diff)}) = {self.body}", var)
            partial.visit(partial.tree)

            filtered_terms = [t for t in partial.terms if t != "0"]
            symbolic_expr = " + ".join(filtered_terms) if filtered_terms else "0"
            symbolic_pretty = self.pretty(symbolic_expr)

            print(f"∂f/∂{var} = {symbolic_pretty}")

            # Evaluate this partial derivative
        
            safe_globals = {name: getattr(math, name) for name in dir(math) if not name.startswith("__")}
            try:
                val = eval(symbolic_expr, safe_globals, context)
                print(f"  → Evaluated at {context}: {val}")
                gradient_vector.append(val)
            except Exception as e:
                print(f"  ❌ Error evaluating ∂f/∂{var}: {e}")
                gradient_vector.append(None)

        print(f"\n🔼 Gradient vector: {gradient_vector}")


    def evaluate_full_gradient_with_comparison(self, context, deltas):

        safe_globals = {name: getattr(math, name) for name in dir(math) if not name.startswith("__")}

        print("\n📐 Gradient Components with Numerical Comparison")

        for var in self.args:
            # Symbolic derivative for this variable
            partial = DerivativeVisitor(f"f({', '.join(self.args)}) = {self.body}", var)
            partial.visit(partial.tree)

            filtered_terms = [t for t in partial.terms if t != "0"]
            symbolic_expr = " + ".join(filtered_terms)
            try:
                true_val = eval(symbolic_expr, safe_globals, context)
            except Exception as e:
                print(f"\n∂f/∂{var} - ❌ Error evaluating symbolic: {e}")
                continue

            print(f"\nGradient component ∂f/∂{var}:")
            print(f"True (Symbolic) = {true_val:.10f}\n")
            print(f"{'Δ':>10} | {'Forward Diff':>16} | {'Central Diff':>16}")
            print("-" * 47)

            for delta in sorted(deltas, reverse=True):
                try:
                    ctx_plus = context.copy()
                    ctx_minus = context.copy()

                    ctx_plus[var] += delta
                    ctx_minus[var] -= delta

                    f_plus = eval(self.body, safe_globals, ctx_plus)
                    f_base = eval(self.body, safe_globals, context)
                    f_minus = eval(self.body, safe_globals, ctx_minus)

                    forward = (f_plus - f_base) / delta
                    central = (f_plus - f_minus) / (2 * delta)

                    print(f"{delta:10.6f} | {forward:16.10f} | {central:16.10f}")
                except Exception as e:
                    print(f"{delta:10.6f} | ❌ Error: {e}")



    @staticmethod
    def evaluate_jacobian():
    
        num_funcs = int(input("🔢 Enter number of functions: "))

        func_exprs = []
        func_names = []
        variables = []

        for i in range(num_funcs):
            func_line = input(f"f{i+1}(x₁, x₂, ...) = ")
            func_exprs.append(func_line)

            # Extract function name and variables from the first one
            if i == 0:
                left, _ = func_line.split("=")
                left = left.strip()
                args_str = left[left.index("(")+1 : left.index(")")]
                variables = [arg.strip() for arg in args_str.split(",")]

        # Build Jacobian matrix: list of rows, each is a list of ∂fi/∂xj
        jacobian = []

        for expr_str in func_exprs:
            row = []
            for var in variables:
                visitor = DerivativeVisitor(expr_str, var)
                visitor.visit(visitor.tree)
                terms = [t for t in visitor.terms if t != "0"]
                pretty_term = DerivativeVisitor.pretty(" + ".join(terms)) if terms else "0"
                row.append(pretty_term)
            jacobian.append(row)

        # Build and print table
        headers = ["∂fᵢ/∂xⱼ →"] + variables
        table = [[f"f{i+1}"] + row for i, row in enumerate(jacobian)]

        print("\n📐 Jacobian Matrix:")
        print(tabulate(table, headers=headers, tablefmt="grid"))




    @staticmethod
    def simplify_hessian_expression(expr: str) -> str:
    
        # First basic regex cleanup (keep your existing ones if needed)
        expr = expr.strip()

        # ✅ Detect and evaluate if the entire expr is numeric-only
        try:
            # Parse safely using AST
            node = ast.parse(expr, mode="eval")

            # Only allow safe arithmetic nodes
            allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Constant,
                    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)

            if all(isinstance(n, allowed) for n in ast.walk(node)):
                result = eval(expr, {"__builtins__": {}})
                return str(result)
        except:
            pass

        return expr

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.USub):  # Unary minus
            inner = DerivativeVisitor.__new__(DerivativeVisitor)
            inner.diff_var = self.diff_var
            inner.terms = []
            inner.visit(node.operand)
            expr = " + ".join(inner.terms) if inner.terms else "0"
            self.terms.append(f"-({expr})")
        else:
            self.generic_visit(node)



    def evaluate_hessian(self, context):
        print("\n📐 Hessian Matrix Evaluation:")

        variables = self.args
        hessian = []

        # Prepare safe evaluation
        safe_globals = {name: getattr(math, name) for name in dir(math) if not name.startswith("__")}

        for var1 in variables:
            row = []
            for var2 in variables:
                # First derivative w.r.t. var2
                d1 = DerivativeVisitor(f"f({', '.join(variables)}) = {self.body}", var2)
                d1.visit(d1.tree)
                first_terms = [t for t in d1.terms if t != "0"]
                inner_expr = " + ".join(first_terms) if first_terms else "0"

                # Second derivative w.r.t. var1 of the first result
                d2 = DerivativeVisitor(f"f({', '.join(variables)}) = {inner_expr}", var1)
                d2.visit(d2.tree)
                second_terms = [t for t in d2.terms if t != "0"]
                raw_expr = " + ".join(second_terms) if second_terms else "0"

                try:
                    # Evaluate numerically using the same context used earlier
                    value = eval(raw_expr, safe_globals, context)
                    row.append(round(value, 6))  # Rounded for nice display
                except Exception:
                    row.append("❌")
            hessian.append(row)

        headers = ["∂²f/∂xi∂xj →"] + variables
        table = [[v] + row for v, row in zip(variables, hessian)]
        print(tabulate(table, headers=headers, tablefmt="grid"))



    def evaluate_laplacian(self, context):
        print("\n📐 Laplacian Evaluation:")

        variables = self.args
        second_derivatives = []

        safe_globals = {name: getattr(math, name) for name in dir(math) if not name.startswith("__")}

        total = 0.0

        for var in variables:
            # First derivative
            d1 = DerivativeVisitor(f"f({', '.join(variables)}) = {self.body}", var)
            d1.visit(d1.tree)
            first_terms = [t for t in d1.terms if t != "0"]
            first_expr = " + ".join(first_terms) if first_terms else "0"

            # Second derivative
            d2 = DerivativeVisitor(f"f({', '.join(variables)}) = {first_expr}", var)
            d2.visit(d2.tree)
            second_terms = [t for t in d2.terms if t != "0"]
            second_expr = " + ".join(second_terms) if second_terms else "0"

            # Pretty + evaluate
            symbolic_pretty = self.pretty(second_expr)
            print(f"∂²f/∂{var}² = {symbolic_pretty}")

            try:
                val = eval(second_expr, safe_globals, context)
                print(f"  → Evaluated at {var}={context[var]}: {val}\n")
                second_derivatives.append(val)
                total += val
            except Exception as e:
                print(f"  ❌ Error evaluating: {e}")
                second_derivatives.append(None)

        print(f"🔄 Laplacian (∇²f): {total}")


    def evaluate_taylor_expansion(self, context, order=2):
        print("\n📐 Taylor Series Expansion:")

        variables = self.args
        expansion_point = context
        f0 = eval(self.body, {**math.__dict__}, expansion_point)

        print(f"f({expansion_point}) = {f0}")
        terms = [str(f0)]

        for n in range(1, order + 1):
            for var_combo in combinations_with_replacement(variables, n):
                term_expr = self.body

                for var in var_combo:
                    d = DerivativeVisitor(f"f({', '.join(variables)}) = {term_expr}", var)
                    d.visit(d.tree)
                    partial_terms = [t for t in d.terms if t != "0"]
                    term_expr = " + ".join(partial_terms) if partial_terms else "0"

                try:
                    safe_globals = {name: getattr(math, name) for name in dir(math) if not name.startswith("__")}
                    deriv_val = eval(term_expr, safe_globals, expansion_point)
                except Exception as e:
                    deriv_val = f"❌ ({e})"

                if isinstance(deriv_val, (int, float)) and abs(deriv_val) > 1e-12:
                    pieces = []
                    for var in variables:
                        exp = var_combo.count(var)
                        if exp > 0:
                            pieces.append(f"({var} - {expansion_point[var]})**{exp}" if exp > 1 else f"({var} - {expansion_point[var]})")
                    poly = " * ".join(pieces)
                    if poly:
                        term_str = f"{deriv_val} * {poly}" if abs(deriv_val) != 1 else poly
                    else:
                        term_str = f"{deriv_val}"
                    if n > 1:
                        term_str += f" / {math.factorial(n)}"
                    terms.append(term_str)

        print("\n📦 Taylor Expansion (up to order", order, "):")
        print("  ", " + ".join(terms))
    


    def evaluate_directional_derivative(self, context):
        print("\n📐 Directional Derivative Evaluation:")

        variables = self.args
        print(f"Variables: {variables}")
        raw_dir = input(f"Enter direction vector as comma-separated values (e.g., 1,2,...): ")
        direction = [float(x.strip()) for x in raw_dir.split(",")]

        if len(direction) != len(variables):
            print("❌ Direction vector length must match number of variables.")
            return

        # Normalize the direction vector
        norm = math.sqrt(sum(x**2 for x in direction))
        if norm == 0:
            print("❌ Direction vector cannot be zero.")
            return
        direction_unit = [x / norm for x in direction]
        print(f"🔄 Normalized direction vector: {direction_unit}")

        # Compute gradient
        gradient_vector = []
        for var in variables:
            partial = DerivativeVisitor(f"f({', '.join(variables)}) = {self.body}", var)
            partial.visit(partial.tree)
            filtered_terms = [t for t in partial.terms if t != "0"]
            symbolic_expr = " + ".join(filtered_terms) if filtered_terms else "0"
            try:
                val = eval(symbolic_expr, {**math.__dict__}, context)
                gradient_vector.append(val)
            except Exception as e:
                print(f"❌ Error evaluating ∂f/∂{var}: {e}")
                return

        print(f"🧮 Gradient at {context}: {gradient_vector}")

        # Compute dot product
        directional_deriv = sum(g * v for g, v in zip(gradient_vector, direction_unit))
        print(f"📈 Directional Derivative = ∇f ⋅ v̂ = {directional_deriv:.10f}")




    
def evaluate_symbolic_derivative():

    # Step 1: get user inputs
    expr_str = input("Enter a function: ")  # e.g., f(x, y) = x*y + log(x + y**2)
    diff_var = input("Differentiate with respect to: ")  # e.g., x

    # Step 2: Ask for evaluation point
    raw_input = input("Enter values for variables (e.g. x=2, y=3): ")

    context = {}
    for item in raw_input.split(","):
        key, value = item.strip().split("=")
        context[key.strip()] = eval(value.strip())

    # Step 3: Symbolic differentiation
    if "=" in expr_str:
        visitor = DerivativeVisitor(expr_str, diff_var)
        parsed_expr = visitor.body
    else:
        vars_found = sorted(set(re.findall(r'\b[a-zA-Z]\w*\b', expr_str)))
        arglist = ", ".join(vars_found)
        full_func = f"f({arglist}) = {expr_str}"
        visitor = DerivativeVisitor(full_func, diff_var)
        parsed_expr = expr_str

    # ✅ Ask for missing variables if needed
    for var in visitor.args:
        if var not in context:
            val = input(f"🧠 Enter value for missing variable '{var}': ")
            context[var] = eval(val)

    # ✅ Add domain check
    is_domain_safe = visitor.validate_domain(context)
    visitor.visit(visitor.tree)

    # Step 4: Get symbolic expression string
    filtered_terms = [t for t in visitor.terms if t != "0"]
    evaluable_expr = " + ".join(filtered_terms)
    symbolic_pretty = DerivativeVisitor.pretty(evaluable_expr)
    print(f"\n📘 Symbolic derivative: {symbolic_pretty}")

    # Step 5: Evaluate symbolic derivative at the given point
    try:
        safe_globals = {name: getattr(math, name) for name in dir(math) if not name.startswith("__")}
        print("🧪 Evaluable expression (for eval):", evaluable_expr)
        result = eval(evaluable_expr, safe_globals, context)
        print(f"📈 Evaluated at {context}: {result}")
        return expr_str, diff_var, context, evaluable_expr, result, is_domain_safe
    except Exception as e:
        print(f"❌ Error evaluating: {e}")
        return None, None, None, None, None, False


            
        
if __name__ == "__main__":
    print("📌 Choose what you want to calculate:")
    print("1. Derivative / Gradient / Hessian / Limit Comparisons / Laplace / Taylor Series / Directional Derivative / Symbolic Chain Rule Expansion")
    print("2. Jacobian Matrix")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("\n🎯 STEP 1: Symbolic Derivative + Evaluation")
        expr_str, diff_var, context, symbolic_expr, symbolic_val, is_domain_safe = evaluate_symbolic_derivative()

        if expr_str and is_domain_safe:
            deltas = [0.03, 0.02, 0.01, 0.005, 0.0001]
            visitor = DerivativeVisitor(expr_str, diff_var)

            while True:
                print("\n🧪 Select an operation to perform:")
                print("1. Compare Derivative Approximations (Limit-based)")
                print("2. Compute Gradient")
                print("3. Compare Gradient with Numerical Derivatives")
                print("4. Compute Hessian Matrix")
                print("5. Compute Laplacian")
                print("6. Taylor Series")
                print("7. Compute Directional Derivative")
                print("8. Symbolic Chain Rule Expansion")
                print("9. Exit")
                sub_choice = input("Your choice (1-9): ").strip()

                if sub_choice == "1":
                    visitor.evaluate_both_differences(symbolic_val, diff_var, context, deltas)
                elif sub_choice == "2":
                    visitor.evaluate_gradient(context)
                elif sub_choice == "3":
                    visitor.evaluate_full_gradient_with_comparison(context, deltas)
                elif sub_choice == "4":
                    visitor.evaluate_hessian(context)
                elif sub_choice == "5":
                    visitor.evaluate_laplacian(context)
                elif sub_choice == "6":
                    visitor.evaluate_taylor_expansion(context,order=2)
                elif sub_choice == "7":
                    visitor.evaluate_directional_derivative(context)
                elif sub_choice == "8":
                    visitor.symbolic_chain_expand()
                elif sub_choice == "9":
                    break
                else:
                    print("⚠️ Invalid selection.")

        elif expr_str:
            print("⛔ Skipping derivative-related operations due to domain issues.")

    elif choice == "2":
        DerivativeVisitor.evaluate_jacobian()
    else:
        print("⚠️ Invalid choice. Program terminated.")
