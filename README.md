# Modelo de prediccion de demanda semanal

Microservicio Flask para Dulceria Angelitos. Carga `modelo_demanda_semanal.pkl` y predice demanda semanal por producto.

## Ejecutar localmente

```bash
pip install -r requirements.txt
python app.py
```

## Endpoints

- `GET /`: estado del servicio y metricas del modelo.
- `GET /productos`: catalogo basico de productos disponibles.
- `POST /predecir-demanda`: predice uno o mas semanas para un producto.
- `POST /predecir-demanda-lote`: predice demanda para varios productos.

Ejemplo:

```json
{
  "id_producto": 1,
  "semanas": 4
}
```

La respuesta incluye `unidades_predichas`, `stock_actual`, `stock_seguridad`, `inventario_recomendado` y `cantidad_sugerida_a_comprar`.

## Metricas guardadas

- MAE Random Forest: 2.4
- RMSE Random Forest: 3.11
- R2 Random Forest: 0.0824
- MAE linea base: 2.52
