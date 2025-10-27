import subprocess

model_name = "gemma3:1b"
new_model_name = "gemma3-finetuned"
modelfile_path = "Modelfile"

try:
    print("🚀 Starting customization process...")
    subprocess.run(
        ["ollama", "create", new_model_name, "-f", modelfile_path],
        check=True
    )
    print(f"✅ Model '{new_model_name}' created successfully!")

except subprocess.CalledProcessError as e:
    print("❌ Error during model creation:")
    print(e.stderr if e.stderr else e)
