import ast
import re
import math

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
        g_prime_raw = g_prime
        #outer = self.pretty(outer_raw)
        #g_prime_pretty = self.pretty(g_prime)

        # Apply chain rule: f'(g(x)) * g'(x)
        if g_prime == "1":
            self.terms.append(outer_raw)
        elif g_prime == "0":
            self.terms.append("0")
        else:
            self.terms.append(f"{outer_raw}*({g_prime_raw})")

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
            symbolic_expr = " + ".join(filtered_terms)
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
    print("🎯 STEP 1: Symbolic Derivative + Evaluation")
    expr_str, diff_var, context, symbolic_expr, symbolic_val, is_domain_safe = evaluate_symbolic_derivative()

    if expr_str and is_domain_safe:
        deltas = [0.03, 0.02, 0.01, 0.005, 0.0001]
        visitor = DerivativeVisitor(expr_str, diff_var)
        visitor.evaluate_both_differences(symbolic_val, diff_var, context, deltas)
        visitor.evaluate_gradient(context)


    elif expr_str:
        print("⛔ Skipping limit-based approximation due to domain issues.")
    


    


