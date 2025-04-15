import pickle
import torch

import pickletools


def test1():
    class CustomUnpickler(pickle.Unpickler):
        def __init__(self, file, custom_param=None):
            self.s =  pickle.Unpickler(file)
            super().__init__(file)
            self.persistent_load_safe = super().persistent_load
            
        def persistent_load(self, pid):
            # Füge deine Logik hier ein, um 'pid' zu behandeln.
            print(f"Persistent ID gefunden: {pid}")
            return super().persistent_load(pid)

    def custom_load(file_path):
        with open(file_path, 'rb') as f:
            return CustomUnpickler(f).load()
            
    custom_load("data.pkl")
            
    with open("data.pkl", "rb") as f:
        unpickler = pickle.Unpickler(f)
        unpickler.persistent_load = dummy_persistent_load
        try:
            data = unpickler.load()
            print("Laden erfolgreich.")
        except Exception as e:
            print("Fehler beim Laden:", e)
            
            
    quit()

#import numpy as np
#np.loadtxt("data.pkl")
data = torch.load('data.pkl')

torch.save(data, "converted_ckpt.pth")

quit()
def test2():
    def dummy_persistent_load(id):
        print(f"PID received {id}")
        return 0
        
    with open("data.pkl", "rb") as f:
        unpickler = pickle.Unpickler(f)
        unpickler.persistent_load = dummy_persistent_load
        try:
            data = unpickler.load()
            print("Laden erfolgreich.")
        except Exception as e:
            print("Fehler beim Laden:", e)
num_gpus = torch.cuda.device_count()
print(f"Anzahl der verfügbaren GPUs: {num_gpus}")

model_data = torch.load("data.pkl", map_location="cpu")
    
test2()
# Öffne die Datei und lese den Inhalt als Byte-String
with open("data.pkl", "rb") as f:
    byte_data = f.read()


# Disassembliert das Pickle-Objekt – zeigt dir den "Bytecode"
#pickletools.dis(byte_data)  # begrenzt auf die ersten 500 Bytes, sonst riesig

#print(byte_data)

# Jetzt versuche, daraus zu deserialisieren
model_data = torch.load("data.pkl", map_location="cuda:0")

# Guck dir mal an, was du bekommen hast
print(f"Typ: {type(model_data)}")
if hasattr(model_data, "keys"):
    print("Keys:", model_data.keys())
else:
    print("Kein dict, sondern:", type(model_data))

# Speichere den deserialisierten Inhalt im Torch-kompatiblen Format
torch.save(model_data, "converted_ckpt.pth")

