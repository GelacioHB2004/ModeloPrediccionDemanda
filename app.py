# app.py
#
# Microservicio Flask (SEPARADO del de recomendación) para servir el modelo
# de predicción de demanda (Random Forest Regressor).
# Se despliega en Render como un Web Service nuevo e independiente.
#
# Endpoints:
#   GET /                                -> healthcheck
#   GET /predecir-demanda/<id_producto>  -> predicción de demanda para los próximos N días

from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import os
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  # permite que el frontend/backend en otro dominio lo consuma

MODELO_PATH = os.path.join(os.path.dirname(__file__), 'modelo_demanda.pkl')

with open(MODELO_PATH, 'rb') as f:
    modelo_demanda = pickle.load(f)

print(f"Modelo de demanda cargado. R2: {modelo_demanda['metadata']['r2']}, MAE: {modelo_demanda['metadata']['mae']}")


def predecir_demanda(id_producto, dias=7):
    m = modelo_demanda['modelo']
    feats = modelo_demanda['features']
    hist = modelo_demanda['historico_reciente']
    hist_producto = hist[hist['id_producto'] == id_producto].sort_values('fecha')

    if hist_producto.empty:
        ultimos_valores = [0] * 7
        ultima_fecha = pd.Timestamp.today().normalize()
    else:
        ultimos_valores = hist_producto['unidades_vendidas'].tolist()
        ultima_fecha = hist_producto['fecha'].max()

    predicciones = []
    for i in range(1, dias + 1):
        fecha_futura = ultima_fecha + pd.Timedelta(days=i)
        ventas_dia_anterior = ultimos_valores[-1] if ultimos_valores else 0
        promedio_7 = np.mean(ultimos_valores[-7:]) if ultimos_valores else 0

        fila = pd.DataFrame([{
            'id_producto': id_producto,
            'dia_semana': fecha_futura.dayofweek,
            'es_fin_semana': int(fecha_futura.dayofweek in [5, 6]),
            'mes': fecha_futura.month,
            'dia_mes': fecha_futura.day,
            'ventas_dia_anterior': ventas_dia_anterior,
            'promedio_ultimos_7_dias': promedio_7
        }])[feats]

        pred = max(0, round(float(m.predict(fila)[0])))
        predicciones.append({'fecha': str(fecha_futura.date()), 'unidades_predichas': pred})
        ultimos_valores.append(pred)

    return predicciones


@app.route('/', methods=['GET'])
def healthcheck():
    return jsonify({
        'status': 'ok',
        'modelo': 'prediccion_demanda',
        'metadata': modelo_demanda['metadata']
    })


@app.route('/predecir-demanda/<int:id_producto>', methods=['GET'])
def predecir_demanda_endpoint(id_producto):
    dias = request.args.get('dias', default=7, type=int)
    predicciones = predecir_demanda(id_producto, dias)
    total_predicho = sum(p['unidades_predichas'] for p in predicciones)

    return jsonify({
        'id_producto': id_producto,
        'dias_predichos': dias,
        'total_unidades_predichas': total_predicho,
        'predicciones_por_dia': predicciones
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
