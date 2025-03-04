
import os
import subprocess

def get_hello_world_from_java():
    java_folder = r'C:\Users\cahar\Documents\GitHub\StockAnalyser\StockMarketData\JavaPog\src'

    # Run the Java program
    result = subprocess.run(
        ["java", "-cp", java_folder, "TaxCalculator"],  # -cp specifies the classpath
        capture_output=True,
        text=True
    )
    # Return the output or handle errors
    if result.returncode == 0:
        return result.stdout.strip()  # Return output without newline
    else:
        print("Error:", result.stderr)
        return None
    
# Example usage
hello_world = get_hello_world_from_java()
if hello_world:
    print("Received from Java:", hello_world)
