import requests

BASE_URL = 'http://127.0.0.1:5000'

payload = {
    'id_producto': 1,
    'semanas': 4,
}

respuesta = requests.post(f'{BASE_URL}/predecir-demanda', json=payload, timeout=20)
print(respuesta.status_code)
print(respuesta.json())
