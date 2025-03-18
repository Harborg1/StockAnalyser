import subprocess

def get_tax_data_from_java():
    java_folder = r'C:\Users\cahar\OneDrive\Documents\GitHub\StockAnalyser\StockMarketData\JavaPog\src'

    # Run the Java program with correct classpath
    result = subprocess.run(
        ["java", "-cp", ".", "TaxCalculator"],  # Ensure classpath is set correctly
        cwd=java_folder,               # Set working directory
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        return result.stdout.strip()
    else:
        print("Execution Error:", result.stderr)
        return None
    
# Example usage
output = get_tax_data_from_java()
if output:
    print("Received from Java:", output)
