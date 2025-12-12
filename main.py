from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import numpy as np
from datetime import time
import io
from scipy.optimize import linear_sum_assignment

app = FastAPI(title="Procesador de marcas - Attendance API")

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
    hora_12 = dt.strftime("%I").lstrip("0")
    if hora_12 == "":
        hora_12 = "12"
    texto = f"{dt.strftime('%d/%m/%Y')} {hora_12}:{dt.strftime('%M:%S')} {dt.strftime('%p')}"
    texto = texto.replace("AM", "a. m.").replace("PM", "p. m.")
    return "'" + texto

def procesar_dataframe(df_filtrado):
    df = df_filtrado.copy()

    # parsear "25/11/2025 7:34:48 a. m."
    df["Fecha/Hora"] = pd.to_datetime(
        df["Fecha/Hora"].astype(str).str.replace("a. m.", "AM").str.replace("p. m.", "PM"),
        format="%d/%m/%Y %I:%M:%S %p",
        errors="coerce"
    )

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

    df_final = pd.DataFrame(rows)

    # convertir todas las columnas de fecha a texto formateado
    for col in ["Fecha Inicial", "Fecha Inicio Almuerzo", "Fecha Fin Almuerzo", "Fecha Final"]:
        df_final[col] = df_final[col].apply(formatear_con_apostrofo)

    df_final = df_final.astype(str)
    return df_final


@app.post("/process")
async def process_file(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".xls", ".xlsx")):
        raise HTTPException(status_code=400, detail="Archivo inválido")

    data = await file.read()

    try:
        df = pd.read_excel(io.BytesIO(data))
    except:
        raise HTTPException(status_code=400, detail="Error leyendo Excel")

    columnas = ["Nombre y Apellido", "Fecha/Hora"]
    if not all(c in df.columns for c in columnas):
        raise HTTPException(status_code=400, detail=f"Faltan columnas: {columnas}")

    df_final = procesar_dataframe(df[columnas])

    # exportar Excel
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as wr:
        df_final.to_excel(wr, index=False)
    out.seek(0)

    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=procesado.xlsx"}
    )
