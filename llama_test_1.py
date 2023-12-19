import os 
from llama_cpp import Llama

model_path = os.getenv("MODEL_PATH")

# Create a Llama instance with a specified model
llm = Llama(
    model_path= model_path,
    model_kwargs={"n_gpu_layers": 1}
    )

# Define the prompt and the generation parameters
prompt = "Quelle est la puissance de llama2 ?"
max_tokens = 400  # Generate up to 400 tokens

# Generate a completion based on the prompt and parameters
output = llm(prompt,
             max_tokens=max_tokens,
             echo=True)

# Print the generated output
print(output)
