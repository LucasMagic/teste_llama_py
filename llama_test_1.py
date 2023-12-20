import threading
import time
import psutil
import GPUtil
from llama_cpp import Llama
import os

# Fonction pour surveiller l'utilisation du CPU et du GPU
def monitor_usage(stop_event, interval=1):
    while not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=None)
        gpu_usage = [gpu.load for gpu in GPUtil.getGPUs()]
        print(f"CPU Usage: {cpu_usage}%, GPU Usage: {gpu_usage}")
        time.sleep(interval)

# Créer une instance de Llama
model_path = os.getenv("MODEL_PATH")
llm = Llama(model_path=model_path, model_kwargs={"n_gpu_layers": 1})

# Définir l'invite et les paramètres de génération
prompt = "Quelle est la puissance de llama2 ?"
max_tokens = 20

# Initialiser le thread de surveillance
stop_event = threading.Event()
monitor_thread = threading.Thread(target=monitor_usage, args=(stop_event,))

# Démarrer le thread de surveillance
monitor_thread.start()

# Exécuter le modèle
output = llm(prompt, max_tokens=max_tokens, echo=True)

# Arrêter le thread de surveillance
stop_event.set()
monitor_thread.join()

# Afficher la sortie du modèle
print(output)
