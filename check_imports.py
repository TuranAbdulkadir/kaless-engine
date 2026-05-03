
import sys
import os

# Add the current directory to sys.path
sys.path.append(os.getcwd())

try:
    print("Checking app.analysis.registry...")
    import app.analysis.registry
    print("Checking app.utils.interpretation...")
    import app.utils.interpretation
    print("Checking app.analysis.descriptives...")
    import app.analysis.descriptives
    print("Checking app.analysis.anova...")
    import app.analysis.anova
    print("Checking app.analysis.regression...")
    import app.analysis.regression
    print("Checking app.analysis.factor...")
    import app.analysis.factor
    print("Checking app.analysis.classify...")
    import app.analysis.classify
    print("Checking app.analysis.neural_net...")
    import app.analysis.neural_net
    print("All imports successful!")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
