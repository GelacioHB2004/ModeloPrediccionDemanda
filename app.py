from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = Path(__file__).resolve().parent
MODELO_PATH = BASE_DIR / 'modelo_demanda_semanal.pkl'

with open(MODELO_PATH, 'rb') as f:
    paquete = pickle.load(f)

pipeline = paquete['pipeline']
columnas_entrada = paquete.get('columnas_entrada', paquete.get('features'))
historico_reciente = pd.DataFrame(paquete['historico_reciente'])
productos = pd.DataFrame(paquete['productos'])
metadata = paquete['metadata']
fecha_min = pd.to_datetime(metadata['fecha_min_entrenamiento'])
semana_max_entrenamiento = int(metadata['semana_max_entrenamiento'])

app = Flask(__name__)


def _producto_info(id_producto):
    fila = productos[productos['id_producto'] == id_producto]
    if fila.empty:
        return {'nombre': 'Producto no registrado', 'stock': 0, 'stock_minimo': 0}
    fila = fila.iloc[0]
    return {
        'nombre': str(fila.get('nombre', 'Producto sin nombre')),
        'stock': int(0 if pd.isna(fila.get('stock', 0)) else fila.get('stock', 0)),
        'stock_minimo': int(0 if pd.isna(fila.get('stock_minimo', 0)) else fila.get('stock_minimo', 0)),
    }


def _historial_producto(id_producto):
    hist = historico_reciente[historico_reciente['id_producto'] == id_producto].sort_values('semana')
    if hist.empty:
        return [0], semana_max_entrenamiento
    return hist['unidades_vendidas'].astype(float).tolist(), int(hist['semana'].max())


def _predecir_producto(id_producto, semanas=4):
    info = _producto_info(id_producto)
    ultimos_valores, ultima_semana = _historial_producto(id_producto)
    resultados = []

    for i in range(1, semanas + 1):
        semana_futura = ultima_semana + i
        fecha_semana = fecha_min + pd.to_timedelta((semana_futura - 1) * 7, unit='D')
        ventas_semana_anterior = ultimos_valores[-1] if ultimos_valores else 0

        entrada = pd.DataFrame([{
            'id_producto': id_producto,
            'semana': semana_futura,
            'mes': int(fecha_semana.month),
            'semana_del_anio': int(fecha_semana.isocalendar().week),
            'ventas_semana_anterior': ventas_semana_anterior,
        }])[columnas_entrada]

        unidades_predichas = max(0, int(round(float(pipeline.predict(entrada)[0]))))
        stock_seguridad = max(info['stock_minimo'], int(round(unidades_predichas * 0.20)))
        inventario_recomendado = unidades_predichas + stock_seguridad
        cantidad_sugerida = max(0, inventario_recomendado - info['stock'])

        resultados.append({
            'id_producto': int(id_producto),
            'nombre': info['nombre'],
            'semana_predicha': int(semana_futura),
            'mes': int(fecha_semana.month),
            'semana_del_anio': int(fecha_semana.isocalendar().week),
            'ventas_semana_anterior_usada': int(round(ventas_semana_anterior)),
            'unidades_predichas': unidades_predichas,
            'stock_actual': info['stock'],
            'stock_seguridad': stock_seguridad,
            'inventario_recomendado': inventario_recomendado,
            'cantidad_sugerida_a_comprar': cantidad_sugerida,
        })
        ultimos_valores.append(unidades_predichas)

    return resultados


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'servicio': 'Prediccion de demanda semanal - Dulceria Angelitos',
        'estado': 'activo',
        'endpoints': {
            'productos': 'GET /productos',
            'prediccion_individual': 'POST /predecir-demanda',
            'prediccion_lote': 'POST /predecir-demanda-lote',
        },
        'metadata_modelo': metadata,
    })


@app.route('/productos', methods=['GET'])
def listar_productos():
    cols = ['id_producto', 'nombre', 'stock', 'stock_minimo']
    return jsonify(productos[cols].fillna(0).to_dict(orient='records'))


@app.route('/predecir-demanda', methods=['POST'])
def predecir_demanda():
    datos = request.get_json(silent=True) or {}
    if 'id_producto' not in datos:
        return jsonify({'error': 'Falta id_producto en el JSON'}), 400

    id_producto = int(datos['id_producto'])
    semanas = int(datos.get('semanas', 4))
    semanas = max(1, min(semanas, 12))

    return jsonify({
        'id_producto': id_producto,
        'predicciones': _predecir_producto(id_producto, semanas),
        'metadata_modelo': metadata,
    })


@app.route('/predecir-demanda-lote', methods=['POST'])
def predecir_demanda_lote():
    datos = request.get_json(silent=True) or {}
    ids = datos.get('productos', datos.get('ids_productos'))
    if not ids:
        return jsonify({'error': 'Falta la lista productos o ids_productos en el JSON'}), 400

    semanas = int(datos.get('semanas', 4))
    semanas = max(1, min(semanas, 12))
    resultados = []
    for id_producto in ids:
        resultados.append({
            'id_producto': int(id_producto),
            'predicciones': _predecir_producto(int(id_producto), semanas),
        })

    return jsonify({'resultados': resultados, 'metadata_modelo': metadata})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
