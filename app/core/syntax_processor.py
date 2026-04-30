"""KALESS Engine — Syntax Processor Module."""

import re

def parse_syntax_command(syntax_text: str) -> dict:
    """Parse a classic SPSS syntax command into a structured execution plan.
    
    Example input:
    REGRESSION 
      /DEPENDENT Score 
      /VARIABLES Age Height.
      
    Output:
    {
        "analysis_type": "linear_regression",
        "params": {
            "dependent": "Score",
            "variables": ["Age", "Height"]
        }
    }
    """
    syntax_text = syntax_text.strip()
    if not syntax_text:
        raise ValueError("Empty syntax command.")
        
    # Remove trailing period if present
    if syntax_text.endswith("."):
        syntax_text = syntax_text[:-1]
        
    lines = [line.strip() for line in syntax_text.split('\n') if line.strip()]
    if not lines:
        raise ValueError("Empty syntax command.")
        
    # First line is usually the command
    command_parts = lines[0].split(maxsplit=1)
    command = command_parts[0].upper()
    
    # Reconstruct the full string for regex parsing
    full_text = " ".join(lines)
    
    # Helper to extract a slash subcommand
    def extract_slash_params(text, subcommand):
        pattern = rf"/{subcommand}\s*=?\s*([^/]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return [v.strip() for v in match.group(1).replace(',', ' ').split() if v.strip()]
        return []

    if command == "REGRESSION":
        dep = extract_slash_params(full_text, "DEPENDENT")
        if not dep:
            raise ValueError("Syntax Error: REGRESSION requires /DEPENDENT")
        
        # Method /ENTER is default in SPSS, we assume VARIABLES are independent
        # Usually SPSS uses /METHOD=ENTER var1 var2 OR /VARIABLES=var1 var2
        vars_enter = extract_slash_params(full_text, "METHOD=ENTER")
        if not vars_enter:
            vars_enter = extract_slash_params(full_text, "VARIABLES")
            
        return {
            "analysis_type": "linear_regression",
            "params": {
                "dependent": dep[0],
                "variables": [v for v in vars_enter if v != dep[0]] # ensure dependent isn't in independent
            }
        }
        
    elif command == "ONEWAY":
        # ONEWAY Score BY Group
        pattern = r"ONEWAY\s+([^\s]+)\s+BY\s+([^\s]+)"
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            dep = match.group(1).strip()
            factor = match.group(2).strip()
            return {
                "analysis_type": "one_way_anova",
                "params": {
                    "dependent": dep,
                    "factor": factor
                }
            }
        else:
            raise ValueError("Syntax Error: ONEWAY requires format 'ONEWAY Dependent BY Factor'")
            
    elif command == "T-TEST":
        # T-TEST GROUPS=Group(1 2) /VARIABLES=Score
        groups = extract_slash_params(full_text, "GROUPS")
        variables = extract_slash_params(full_text, "VARIABLES")
        
        if groups and variables:
            # GROUPS=Group(1 2) -> we extract "Group"
            group_var_match = re.match(r"([^\(]+)", groups[0])
            group_var = group_var_match.group(1).strip() if group_var_match else groups[0]
            
            return {
                "analysis_type": "independent_t",
                "params": {
                    "variables": [variables[0], group_var]
                }
            }
        else:
            # Maybe paired? T-TEST PAIRS=Var1 WITH Var2
            pairs = extract_slash_params(full_text, "PAIRS")
            if pairs:
                # PAIRS=Var1 WITH Var2
                pairs_str = " ".join(pairs)
                if "WITH" in pairs_str.upper():
                    p_split = re.split(r'\s+WITH\s+', pairs_str, flags=re.IGNORECASE)
                    return {
                        "analysis_type": "paired_t",
                        "params": {
                            "variables": [p_split[0].strip(), p_split[1].strip()]
                        }
                    }
            raise ValueError("Syntax Error: Unrecognized T-TEST format.")
            
    elif command == "FREQUENCIES":
        variables = extract_slash_params(full_text, "VARIABLES")
        if not variables:
            # In SPSS, it's often FREQUENCIES VARIABLES=Var1
            # Or just FREQUENCIES Var1
            if len(command_parts) > 1:
                parts = command_parts[1].replace("VARIABLES=", "").replace("VARIABLES =", "")
                variables = [v.strip() for v in parts.split()]
                
        if not variables:
            raise ValueError("Syntax Error: FREQUENCIES requires VARIABLES")
            
        return {
            "analysis_type": "frequencies",
            "params": {
                "variables": [variables[0]]
            }
        }
        
    elif command == "DESCRIPTIVES":
        variables = extract_slash_params(full_text, "VARIABLES")
        if not variables:
            if len(command_parts) > 1:
                parts = command_parts[1].replace("VARIABLES=", "").replace("VARIABLES =", "")
                variables = [v.strip() for v in parts.split()]
        return {
            "analysis_type": "descriptives",
            "params": {
                "variables": variables
            }
        }
        
    elif command == "CORRELATIONS":
        variables = extract_slash_params(full_text, "VARIABLES")
        if not variables:
            if len(command_parts) > 1:
                parts = command_parts[1].replace("VARIABLES=", "").replace("VARIABLES =", "")
                variables = [v.strip() for v in parts.split()]
        return {
            "analysis_type": "correlation",
            "params": {
                "variables": variables
            }
        }
        
    elif command == "RELIABILITY":
        variables = extract_slash_params(full_text, "VARIABLES")
        return {
            "analysis_type": "reliability",
            "params": {
                "variables": variables
            }
        }

    elif command == "NPAR" or (command == "NPAR" and len(command_parts) > 1):
        # NPAR TESTS /CHISQUARE=var1 var2
        # NPAR TESTS /M-W=var1 BY group(1 2)
        # NPAR TESTS /WILCOXON=var1 WITH var2
        chi_vars = extract_slash_params(full_text, "CHISQUARE")
        mw_vars = extract_slash_params(full_text, "M-W")
        wilcoxon_vars = extract_slash_params(full_text, "WILCOXON")

        if chi_vars:
            return {
                "analysis_type": "nonparametric",
                "params": {
                    "test_type": "chi_square",
                    "variables": chi_vars
                }
            }
        elif mw_vars:
            # M-W=Score BY Group
            mw_str = " ".join(mw_vars)
            by_match = re.split(r'\s+BY\s+', mw_str, flags=re.IGNORECASE)
            if len(by_match) == 2:
                test_var = by_match[0].strip()
                group_var = by_match[1].strip().split("(")[0].strip()
                return {
                    "analysis_type": "nonparametric",
                    "params": {
                        "test_type": "mann_whitney",
                        "variables": [test_var],
                        "grouping_var": group_var
                    }
                }
            raise ValueError("Syntax Error: NPAR TESTS /M-W requires 'variable BY grouping_variable'")
        elif wilcoxon_vars:
            # WILCOXON=var1 WITH var2
            w_str = " ".join(wilcoxon_vars)
            with_match = re.split(r'\s+WITH\s+', w_str, flags=re.IGNORECASE)
            if len(with_match) == 2:
                return {
                    "analysis_type": "nonparametric",
                    "params": {
                        "test_type": "wilcoxon",
                        "variables": [with_match[0].strip(), with_match[1].strip()]
                    }
                }
            raise ValueError("Syntax Error: NPAR TESTS /WILCOXON requires 'var1 WITH var2'")
        else:
            raise ValueError("Syntax Error: NPAR TESTS requires /CHISQUARE, /M-W, or /WILCOXON subcommand.")

    elif command == "FACTOR":
        # FACTOR /VARIABLES=var1 var2 var3
        #   /EXTRACTION=PC
        #   /ROTATION=VARIMAX.
        variables = extract_slash_params(full_text, "VARIABLES")
        if not variables:
            if len(command_parts) > 1:
                variables = [v.strip() for v in command_parts[1].replace("VARIABLES=", "").split() if v.strip()]

        rotation = "varimax"
        rot_params = extract_slash_params(full_text, "ROTATION")
        if rot_params:
            rot_val = rot_params[0].upper()
            if rot_val == "NOROTATE":
                rotation = "none"
            elif rot_val in ("VARIMAX", "PROMAX", "OBLIMIN"):
                rotation = rot_val.lower()

        return {
            "analysis_type": "factor_analysis",
            "params": {
                "variables": variables,
                "rotation": rotation
            }
        }

    elif command in ("GLM", "UNIANOVA"):
        # GLM Score BY Group1 Group2 WITH Covariate1
        # UNIANOVA Score BY Gender WITH Age
        remainder = command_parts[1] if len(command_parts) > 1 else ""
        full_remainder = " ".join([remainder] + lines[1:])

        # Parse: Dependent BY Factor1 Factor2 WITH Covariate1 Covariate2
        by_match = re.split(r'\s+BY\s+', full_remainder, maxsplit=1, flags=re.IGNORECASE)
        if len(by_match) < 2:
            raise ValueError("Syntax Error: GLM/UNIANOVA requires 'Dependent BY Factor(s)'")

        dependent = by_match[0].strip()
        rest = by_match[1].strip()

        # Split on WITH if present
        with_match = re.split(r'\s+WITH\s+', rest, maxsplit=1, flags=re.IGNORECASE)
        factors_str = with_match[0].strip()
        covariates_str = with_match[1].strip() if len(with_match) > 1 else ""

        # Remove any /subcommands from factors and covariates
        factors_str = re.split(r'\s*/', factors_str)[0]
        covariates_str = re.split(r'\s*/', covariates_str)[0] if covariates_str else ""

        fixed_factors = [f.strip() for f in factors_str.split() if f.strip()]
        covariates = [c.strip() for c in covariates_str.split() if c.strip()] if covariates_str else []

        return {
            "analysis_type": "glm_univariate",
            "params": {
                "dependent": dependent,
                "fixed_factors": fixed_factors,
                "covariates": covariates
            }
        }

    elif command in ("QUICK", "CLUSTER"):
        # QUICK CLUSTER var1 var2 var3 /CRITERIA CLUSTERS(3)
        # or: CLUSTER var1 var2 /METHOD=KMEANS /CLUSTERS=3
        variables = []
        if len(command_parts) > 1:
            vars_str = command_parts[1]
            # Remove /subcommands
            vars_str = re.split(r'\s*/', vars_str)[0]
            variables = [v.strip() for v in vars_str.split() if v.strip() and v.upper() != "CLUSTER"]

        # Extract number of clusters
        n_clusters = 3
        cluster_match = re.search(r'CLUSTERS?\s*\(?(\d+)\)?', full_text, re.IGNORECASE)
        if cluster_match:
            n_clusters = int(cluster_match.group(1))

        if not variables:
            raise ValueError("Syntax Error: QUICK CLUSTER requires variable names.")

        return {
            "analysis_type": "kmeans_cluster",
            "params": {
                "variables": variables,
                "n_clusters": n_clusters,
                "save_membership": True
            }
        }

    else:
        raise ValueError(f"Syntax Error: Unsupported command '{command}'")
