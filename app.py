from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)  # permite que tu app web (frontend) llame a este backend desde otro dominio

# ---------------------------------------------------------------------------
# Carga del modelo (una sola vez, al iniciar el servidor)
# ---------------------------------------------------------------------------
with open('modelo_demanda_semanal.pkl', 'rb') as f:
    modelo_demanda_semanal = pickle.load(f)

pipeline = modelo_demanda_semanal['pipeline']
columnas_entrada = modelo_demanda_semanal['columnas_entrada']
historico_reciente = modelo_demanda_semanal['historico_reciente']
metadata = modelo_demanda_semanal['metadata']

semana_max_entrenamiento = metadata['semana_max_entrenamiento']


# ---------------------------------------------------------------------------
# Endpoint de salud (útil para verificar que el servicio de Render está vivo)
# ---------------------------------------------------------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'servicio': 'Predicción de demanda semanal - Dulcería Angelitos',
        'estado': 'activo',
        'metadata_modelo': metadata
    })


# ---------------------------------------------------------------------------
# Endpoint principal de predicción
# Body esperado (JSON): { "id_producto": 37 }
# Opcional: { "id_producto": 37, "semana": 16 }
# ---------------------------------------------------------------------------
@app.route('/predecir-demanda', methods=['POST'])
def predecir_demanda():
    datos = request.get_json(silent=True)

    if not datos or 'id_producto' not in datos:
        return jsonify({'error': "Falta el campo 'id_producto' en el body"}), 400

    try:
        id_producto = int(datos['id_producto'])
    except (TypeError, ValueError):
        return jsonify({'error': "'id_producto' debe ser un número entero"}), 400

    # Si no mandan semana, se asume que se quiere predecir la siguiente
    # semana después de la última semana con la que se entrenó el modelo.
    semana = int(datos.get('semana', semana_max_entrenamiento + 1))

    # Buscar la última venta conocida de ese producto para calcular
    # 'ventas_semana' (la venta de la semana anterior).
    fila_reciente = historico_reciente[historico_reciente['id_producto'] == id_producto]

    if fila_reciente.empty:
        return jsonify({
            'error': f'No hay histórico para id_producto={id_producto}. '
                     f'Verifica que el producto exista y tenga ventas registradas.'
        }), 404

    ventas_pasadas = float(fila_reciente['unidades_vendidas'].values[0])
    
    datos_modelo = {
        'id_producto': id_producto,
        'semana': semana
    }
    
    for col in columnas_entrada:
        if col not in datos_modelo:
            if col in fila_reciente.columns:
                datos_modelo[col] = float(fila_reciente[col].values[0])
            else:
                datos_modelo[col] = ventas_pasadas

    X_nuevo = pd.DataFrame([datos_modelo])[columnas_entrada]

    prediccion = pipeline.predict(X_nuevo)[0]
    prediccion = max(0, round(float(prediccion)))  # no tiene sentido predecir ventas negativas

    return jsonify({
        'id_producto': id_producto,
        'semana_predicha': semana,
        'unidades_predichas': prediccion,
        'ventas_semana_anterior_usada': ventas_semana,
        'metadata_modelo': {
            'r2': metadata['r2'],
            'mae': metadata['mae']
        }
    })


# ---------------------------------------------------------------------------
# Endpoint para predecir varios productos de un jalón
# Body esperado (JSON): { "productos": [37, 43, 12], "semana": 16 }
# ---------------------------------------------------------------------------
@app.route('/predecir-demanda-lote', methods=['POST'])
def predecir_demanda_lote():
    datos = request.get_json(silent=True)

    if not datos or 'productos' not in datos or not isinstance(datos['productos'], list):
        return jsonify({'error': "Falta el campo 'productos' (lista de id_producto) en el body"}), 400

    semana = int(datos.get('semana', semana_max_entrenamiento + 1))
    resultados = []

    for id_producto in datos['productos']:
        try:
            id_producto = int(id_producto)
        except (TypeError, ValueError):
            resultados.append({'id_producto': id_producto, 'error': 'id_producto inválido'})
            continue

        fila_reciente = historico_reciente[historico_reciente['id_producto'] == id_producto]

        if fila_reciente.empty:
            resultados.append({'id_producto': id_producto, 'error': 'sin histórico'})
            continue

        ventas_pasadas = float(fila_reciente['unidades_vendidas'].values[0])
        
        datos_modelo = {
            'id_producto': id_producto,
            'semana': semana
        }
        
        for col in columnas_entrada:
            if col not in datos_modelo:
                if col in fila_reciente.columns:
                    datos_modelo[col] = float(fila_reciente[col].values[0])
                else:
                    datos_modelo[col] = ventas_pasadas

        X_nuevo = pd.DataFrame([datos_modelo])[columnas_entrada]

        prediccion = pipeline.predict(X_nuevo)[0]
        prediccion = max(0, round(float(prediccion)))

        resultados.append({
            'id_producto': id_producto,
            'unidades_predichas': prediccion
        })

    return jsonify({
        'semana_predicha': semana,
        'resultados': resultados
    })


if __name__ == '__main__':
    app.run(debug=True)
