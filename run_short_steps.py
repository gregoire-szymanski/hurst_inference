import subprocess
import sys

# List of scripts to run in order
scripts = [
    "parameters.py",
    "preparations.py",
    "volatility_patterns.py",
    "volatility_increments.py",
    "quadratic_variations.py",
]

def run_scripts(scripts):
    for script in scripts:
        print(f"Running {script}...")
        try:
            result = subprocess.run([sys.executable, script], check=True, text=True, capture_output=True)
            print(f"Output of {script}:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running {script}:\n{e.stderr}")
            break

if __name__ == "__main__":
    run_scripts(scripts)