import subprocess
import os

def get_tax_data_from_java():
    java_folder = r'C:\Users\cahar\OneDrive\Documents\GitHub\StockAnalyser\StockMarketData\JavaPog\src'
    java_file = "TaxCalculator.java"
    
    # Step 1: Compile the Java file
    compile_result = subprocess.run(
        ["javac", java_file],
        cwd=java_folder,
        capture_output=True,
        text=True
    )

    if compile_result.returncode != 0:
        print("Compilation Error:", compile_result.stderr)
        return None

    # Step 2: Run the compiled Java class
    run_result = subprocess.run(
        ["java", "-cp", ".", "TaxCalculator"],
        cwd=java_folder,
        capture_output=True,
        text=True
    )

    if run_result.returncode == 0:
        return run_result.stdout.strip()
    else:
        print("Execution Error:", run_result.stderr)
        return None

# Example usage
output = get_tax_data_from_java()
if output:
    print("Received from Java:", output)
