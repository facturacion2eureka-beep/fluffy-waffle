from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import time
import io
from scipy.optimize import linear_sum_assignment

app = FastAPI(title="Procesador de marcas - Attendance API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------- CONFIG ---------
objetivos = [
    ("Fecha Inicial", time(8, 0)),
    ("Fecha Inicio Almuerzo", time(13, 0)),
    ("Fecha Fin Almuerzo", time(14, 0)),
    ("Fecha Final", time(18, 0)),
]

ventanas = {
    "Fecha Inicial": (time(5, 0), time(10, 30)),
    "Fecha Inicio Almuerzo": (time(10, 30), time(14, 30)),
    "Fecha Fin Almuerzo": (time(12, 30), time(16, 0)),
    "Fecha Final": (time(15, 0), time(23, 59, 59)),
}

BIG = 10**9
# --------------------------------------

def en_ventana(ts, ventana):
    if pd.isna(ts):
        return False
    t = ts.time()
    hmin, hmax = ventana
    return (t >= hmin) and (t <= hmax)

def formatear_con_apostrofo(dt):
    if pd.isna(dt) or dt == "" or str(dt) == "nan":
        return ""
    try:
        hora_12 = dt.strftime("%I").lstrip("0")
        if hora_12 == "":
            hora_12 = "12"
        texto = f"{dt.strftime('%d/%m/%Y')} {hora_12}:{dt.strftime('%M:%S')} {dt.strftime('%p')}"
        texto = texto.replace("AM", "a. m.").replace("PM", "p. m.")
        return "'" + texto
    except:
        return ""

def procesar_dataframe(df_filtrado):
    df = df_filtrado.copy()

    # parsear "10/12/2025 7:04:00 p. m."
    df["Fecha/Hora"] = pd.to_datetime(
        df["Fecha/Hora"].astype(str).str.replace("a. m.", "AM").str.replace("p. m.", "PM"),
        format="%d/%m/%Y %I:%M:%S %p",
        errors="coerce"
    )

    # Filtrar filas donde no se pudo parsear la fecha
    df = df.dropna(subset=["Fecha/Hora"])
    
    if df.empty:
        # Si no hay datos válidos, retornar DataFrame vacío con estructura correcta
        return pd.DataFrame(columns=[
            "Nombre y Apellido",
            "Fecha Inicial",
            "Fecha Inicio Almuerzo",
            "Fecha Fin Almuerzo",
            "Fecha Final",
            "Sin Clasificar"
        ])

    df["Fecha"] = df["Fecha/Hora"].dt.date

    rows = []
    for (nombre, fecha), grupo in df.groupby(["Nombre y Apellido", "Fecha"]):
        fila = {
            "Nombre y Apellido": nombre,
            "Fecha Inicial": None,
            "Fecha Inicio Almuerzo": None,
            "Fecha Fin Almuerzo": None,
            "Fecha Final": None,
            "Sin Clasificar": ""
        }

        candidatos = list(grupo["Fecha/Hora"].sort_values())
        n_t = len(objetivos)
        n_c = len(candidatos)

        if n_c == 0:
            rows.append(fila)
            continue

        cost_matrix = np.full((n_t, n_c), BIG, dtype=float)

        for i, (campo, hora_obj) in enumerate(objetivos):
            ventana = ventanas[campo]
            for j, cand in enumerate(candidatos):
                if en_ventana(cand, ventana):
                    objetivo_dt = pd.to_datetime(f"{cand.date()} {hora_obj.strftime('%H:%M')}")
                    cost_matrix[i, j] = abs((cand - objetivo_dt).total_seconds())

        # Hungarian
        m = max(n_t, n_c)
        square = np.full((m, m), BIG, dtype=float)
        square[:n_t, :n_c] = cost_matrix
        row_ind, col_ind = linear_sum_assignment(square)

        asignados = set()
        for r, c in zip(row_ind, col_ind):
            if r < n_t and c < n_c:
                if square[r, c] < BIG:
                    campo = objetivos[r][0]
                    fila[campo] = candidatos[c]
                    asignados.add(c)

        # Sin clasificar
        restantes = [candidatos[i] for i in range(n_c) if i not in asignados]
        restantes_fmt = [formatear_con_apostrofo(x) for x in restantes]
        fila["Sin Clasificar"] = ", ".join(restantes_fmt)

        rows.append(fila)

    # Verificar si hay rows antes de crear DataFrame
    if not rows:
        df_final = pd.DataFrame(columns=[
            "Nombre y Apellido",
            "Fecha Inicial",
            "Fecha Inicio Almuerzo",
            "Fecha Fin Almuerzo",
            "Fecha Final",
            "Sin Clasificar"
        ])
    else:
        df_final = pd.DataFrame(rows)
        
        # convertir todas las columnas de fecha a texto formateado
        for col in ["Fecha Inicial", "Fecha Inicio Almuerzo", "Fecha Fin Almuerzo", "Fecha Final"]:
            if col in df_final.columns:
                df_final[col] = df_final[col].apply(formatear_con_apostrofo)

    df_final = df_final.astype(str)
    return df_final


@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    export_format: str = Form(default="xlsx")
):
    """
    Procesa el archivo de asistencia y exporta en el formato especificado.
    
    Parámetros:
    - file: Archivo Excel a procesar
    - export_format: Formato de exportación ("xlsx" o "xls"), por defecto "xlsx"
    """
    
    # Validar formato de exportación
    export_format = export_format.lower()
    if export_format not in ["xlsx", "xls"]:
        raise HTTPException(
            status_code=400, 
            detail="Formato inválido. Use 'xlsx' o 'xls'"
        )
    
    # Validar archivo de entrada
    if not file.filename.lower().endswith((".xls", ".xlsx")):
        raise HTTPException(status_code=400, detail="Archivo inválido")

    data = await file.read()

    try:
        df = pd.read_excel(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo Excel: {str(e)}")

    # Verificar que el DataFrame no esté vacío
    if df.empty:
        raise HTTPException(status_code=400, detail="El archivo Excel está vacío")

    columnas = ["Nombre y Apellido", "Fecha/Hora"]
    if not all(c in df.columns for c in columnas):
        raise HTTPException(
            status_code=400, 
            detail=f"Faltan columnas. Se esperan: {columnas}. Columnas encontradas: {list(df.columns)}"
        )

    # Filtrar columnas necesarias y eliminar filas completamente vacías
    df_filtrado = df[columnas].copy()
    df_filtrado = df_filtrado.dropna(how='all')
    
    if df_filtrado.empty:
        raise HTTPException(
            status_code=400,
            detail="No hay datos válidos en las columnas requeridas"
        )

    # Procesar datos
    try:
        df_final = procesar_dataframe(df_filtrado)
    except KeyError as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando datos - columna faltante: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando datos: {str(e)}"
        )

    # Verificar que el resultado tiene datos
    if df_final.empty:
        raise HTTPException(
            status_code=400,
            detail="No se pudieron procesar los datos. Verifica el formato de las fechas."
        )

    # Exportar en el formato especificado
    out = io.BytesIO()
    
    try:
        if export_format == "xlsx":
            # Usar openpyxl para .xlsx
            with pd.ExcelWriter(out, engine="openpyxl") as wr:
                df_final.to_excel(wr, index=False, sheet_name="Asistencia")
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = "procesado.xlsx"
        else:  # xls
            # Usar xlsxwriter para compatibilidad
            with pd.ExcelWriter(out, engine="xlsxwriter") as wr:
                df_final.to_excel(wr, index=False, sheet_name="Asistencia")
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = "procesado.xlsx"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exportando archivo: {str(e)}")
    
    out.seek(0)

    return StreamingResponse(
        out,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )


@app.get("/")
async def root():
    return {
        "message": "API de Procesamiento de Asistencia - ETL by Juancai",
        "version": "2.0",
        "endpoint": "/process",
        "formatos_soportados": ["xlsx", "xls"],
        "columnas_requeridas": ["Nombre y Apellido", "Fecha/Hora"],
        "formato_fecha": "dd/mm/yyyy h:mm:ss a. m./p. m."
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "attendance-processor"}