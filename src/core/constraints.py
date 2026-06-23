import re
import numpy as np

def check_constraints(row_dict, numerical_rules, disallowed_rules, features_config):
    """
    Evaluates whether a candidate row (dict) satisfies all specified numerical
    equations and disallowed categorical combinations.
    """
    # 1. Check disallowed combinations (categorical)
    if disallowed_rules:
        for rule in disallowed_rules:
            if not rule:
                continue
            violation = True
            for cond in rule:
                feat_name = cond.get('feature')
                feat_val = cond.get('value')
                if feat_name in row_dict:
                    if str(row_dict[feat_name]).strip() != str(feat_val).strip():
                        violation = False
                        break
                else:
                    violation = False
                    break
            if violation:
                return False  # Violates this disallowed combination

    # 2. Check numerical constraints
    if numerical_rules:
        for expr in numerical_rules:
            if not expr.strip():
                continue
            
            # Map single "=" to "==" for evaluation, but keep "<=" and ">=" intact
            clean_expr = expr.strip()
            clean_expr = re.sub(r'(?<![<>=!])=(?![=])', '==', clean_expr)
            
            # Replace feature names with their values
            eval_expr = clean_expr
            sorted_feats = sorted(features_config, key=lambda x: len(x.get('name', '')), reverse=True)
            for f in sorted_feats:
                feat_name = f.get('name')
                if feat_name in row_dict:
                    val = row_dict[feat_name]
                    # Use word boundaries to only replace exact feature name
                    eval_expr = re.sub(r'\b' + re.escape(feat_name) + r'\b', str(val), eval_expr)
            
            try:
                # Find relation operator
                ops = ['<=', '>=', '==', '<', '>', '!=']
                op_found = None
                for op in ops:
                    if op in eval_expr:
                        op_found = op
                        break
                
                if op_found:
                    parts = eval_expr.split(op_found)
                    lhs = eval(parts[0], {"__builtins__": {}}, {})
                    rhs = eval(parts[1], {"__builtins__": {}}, {})
                    
                    if op_found == '<=':
                        res = (lhs <= rhs)
                    elif op_found == '>=':
                        res = (lhs >= rhs)
                    elif op_found == '==':
                        res = (lhs == rhs)
                    elif op_found == '<':
                        res = (lhs < rhs)
                    elif op_found == '>':
                        res = (lhs > rhs)
                    elif op_found == '!=':
                        res = (lhs != rhs)
                else:
                    res = bool(eval(eval_expr, {"__builtins__": {}}, {}))
                
                if not res:
                    return False
            except Exception as e:
                print(f"Error evaluating numerical constraint '{expr}' (parsed as '{eval_expr}'): {e}")
                return False
                
    return True
