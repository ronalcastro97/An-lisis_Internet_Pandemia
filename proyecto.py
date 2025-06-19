
# Dashboard Profesional - Tráfico de Internet en Colombia
# Estructura mejorada para portafolio

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="📡 Dashboard Tráfico Internet - Colombia", layout="wide")

# Encabezado visual atractivo
st.markdown("""
# 📊 Dashboard de Tráfico de Internet en Colombia
Explora el volumen de datos a nivel nacional por año, proveedor y tipo de tráfico, incluyendo una predicción automática del consumo de septiembre 2021.
""")
st.markdown("___")

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("Monitoreo de Tráfico de Internet - Trafico Diario_133 - Monitoreo de Tráfico de Internet - Trafico Diario_133.csv", sep=",", encoding="latin1")
    mapping = {
        'Mes de trÃ¡fico': 'Mes de trafico',
        'TrÃ¡fico Datos: Internacional (GB)': 'Tráfico Datos: Internacional (GB)',
        'TrÃ¡fico Datos:  NAPs - Colombia (GB)': 'Tráfico Datos: NAPs - Colombia (GB)',
        'TrÃ¡fico Datos: Acuerdos de trÃ¡nsito o peering directo (GB)': 'Tráfico Datos: Acuerdos de tránsito o peering directo (GB)',
        'TrÃ¡fico Datos: Local (GB)': 'Tráfico Datos: Local (GB)',
        'TrÃ¡fico Datos: Total Mes (GB)': 'Trafico Datos: Total Mes (GB)'
    }
    df.rename(columns=mapping, inplace=True)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace('á', 'a')
        .str.replace('é', 'e')
    )
    cols = [
        'trafico datos: internacional (gb)',
        'trafico datos: naps - colombia (gb)',
        'trafico datos: acuerdos de transito o peering directo (gb)',
        'trafico datos: local (gb)'
    ]
    df[cols] = (
        df[cols]
        .astype(str)
        .replace(r'\.', '', regex=True)
        .replace(',', '.', regex=True)
        .replace('', pd.NA)
        .astype('float64')
    )
    df.drop(columns=['trafico datos: total mes (gb)'], inplace=True)
    df['mes de trafico'] = pd.to_datetime(df['mes de trafico'], format='%d/%m/%Y', dayfirst=True)
    df['año'] = df['mes de trafico'].dt.year
    df['mes'] = df['mes de trafico'].dt.month
    df['dia'] = df['mes de trafico'].dt.day
    return df

df = cargar_datos()

# Navegación
st.sidebar.header("🧭 Navegación del dashboard")
opcion = st.sidebar.radio("Selecciona vista:", (
    "📶 Volumen por Año",
    "📡 Por Proveedor",
    "🌐 Tráfico Internacional",
    "📍 Tráfico Local",
    "🔄 Peering",
    "🔮 Predicción Septiembre"
))

# Función para formato legible
def formatear_valor(v):
    return f"{v/1e9:.1f}B" if v >= 1e9 else f"{v/1e6:.1f}M"

# Visualizaciones según opción
if opcion == "📶 Volumen por Año":
    resumen = df.groupby(['año'], as_index=False)['trafico datos: naps - colombia (gb)'].sum()
    resumen.rename(columns={'trafico datos: naps - colombia (gb)': 'total_naps_gb'}, inplace=True)
    st.sidebar.markdown("### 🎛️ Filtro")
    seleccion = st.sidebar.multiselect("Año:", resumen['año'].unique(), default=list(resumen['año'].unique()))
    resumen = resumen[resumen['año'].isin(seleccion)]
    fig = px.bar(resumen, x="año", y="total_naps_gb", text="total_naps_gb",
                 title="📶 Volumen anual de tráfico NAPs",
                 labels={"año": "Año", "total_naps_gb": "Tráfico NAPs (GB)"},
                 color="total_naps_gb", color_continuous_scale="Blues")
    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

elif opcion == "📡 Por Proveedor":
    resumen = df.groupby('proveedor', as_index=False)['trafico datos: naps - colombia (gb)'].sum()
    resumen['texto'] = resumen['trafico datos: naps - colombia (gb)'].apply(formatear_valor)
    fig = px.bar(resumen.sort_values(by='trafico datos: naps - colombia (gb)', ascending=False),
                 x="proveedor", y="trafico datos: naps - colombia (gb)", text="texto",
                 title="📡 Tráfico NAPs por proveedor",
                 color="trafico datos: naps - colombia (gb)", color_continuous_scale="Blues")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

elif opcion == "🌐 Tráfico Internacional":
    resumen = df.groupby('proveedor', as_index=False)['trafico datos: internacional (gb)'].sum()
    resumen['texto'] = resumen['trafico datos: internacional (gb)'].apply(formatear_valor)
    fig = px.bar(resumen.sort_values(by='trafico datos: internacional (gb)', ascending=False),
                 x="proveedor", y="trafico datos: internacional (gb)", text="texto",
                 title="🌐 Tráfico internacional por proveedor",
                 color="trafico datos: internacional (gb)", color_continuous_scale="Blues")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

elif opcion == "📍 Tráfico Local":
    resumen = df.groupby('proveedor', as_index=False)['trafico datos: local (gb)'].sum()
    resumen['texto'] = resumen['trafico datos: local (gb)'].apply(formatear_valor)
    fig = px.bar(resumen.sort_values(by='trafico datos: local (gb)', ascending=False),
                 x="proveedor", y="trafico datos: local (gb)", text="texto",
                 title="📍 Tráfico local por proveedor",
                 color="trafico datos: local (gb)", color_continuous_scale="Blues")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

elif opcion == "🔄 Peering":
    resumen = df.groupby('proveedor', as_index=False)['trafico datos: acuerdos de transito o peering directo (gb)'].sum()
    resumen['texto'] = resumen['trafico datos: acuerdos de transito o peering directo (gb)'].apply(formatear_valor)
    fig = px.bar(resumen.sort_values(by='trafico datos: acuerdos de transito o peering directo (gb)', ascending=False),
                 x="proveedor", y="trafico datos: acuerdos de transito o peering directo (gb)", text="texto",
                 title="🔄 Acuerdos de tránsito / peering directo",
                 color="trafico datos: acuerdos de transito o peering directo (gb)", color_continuous_scale="Blues")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

elif opcion == "🔮 Predicción Septiembre":
    trafico_columnas = [
        'trafico datos: internacional (gb)',
        'trafico datos: naps - colombia (gb)',
        'trafico datos: acuerdos de transito o peering directo (gb)',
        'trafico datos: local (gb)'
    ]
    df_2020 = df[df['año'] == 2020]
    df_2021 = df[df['año'] == 2021]
    meses_2020 = df_2020.groupby('mes', as_index=False)[trafico_columnas].sum()
    meses_2021 = df_2021.groupby('mes', as_index=False)[trafico_columnas].sum()
    meses_2020["año"] = 2020
    meses_2021["año"] = 2021
    meses_2020["total"] = meses_2020[trafico_columnas].sum(axis=1)
    meses_2021["total"] = meses_2021[trafico_columnas].sum(axis=1)
    nombres_meses = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    meses_2020["nombre"] = meses_2020["mes"].map(nombres_meses)
    meses_2021["nombre"] = meses_2021["mes"].map(nombres_meses)
    todos = pd.concat([meses_2020, meses_2021], ignore_index=True)
    todos = todos[~((todos['año']==2021) & (todos['mes']==10))].sort_values(['año','mes']).reset_index(drop=True)
    todos['t'] = range(1, len(todos)+1)
    train = todos.iloc[:-1]
    test = todos.iloc[-1:]
    X = train['t'].values.reshape(-1,1)
    y = train['total'].values
    model = LinearRegression()
    model.fit(X, y)
    t_pred = np.array([[len(X)+1]])
    y_pred = model.predict(t_pred)[0]
    y_true = test['total'].values[0]
    mes, año = test['nombre'].iloc[0], test['año'].iloc[0]

    st.subheader("🔮 Predicción consumo datos - Septiembre 2021")
    st.markdown(f"📌 Mes predicho: **{mes} {año}**")
    st.markdown(f"📊 **Predicción:** `{y_pred:.3e}` GB")
    st.markdown(f"📊 **Valor real:** `{y_true:.3e}` GB")
    st.markdown(f"❗ **Error:** `{(y_pred - y_true):.3e}` GB  `{(y_pred - y_true)/y_true*100:.1f}%`")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train['t'], train['total'], marker='o', label='Histórico')
    ax.scatter(t_pred, y_pred, s=120, c='red', label='Predicción')
    ax.scatter(t_pred, y_true, s=120, c='green', label='Real')
    ax.set_xticks(list(train['t']) + [t_pred[0][0]])
    ax.set_xticklabels(list(train['nombre'] + ' ' + train['año'].astype(str)) + [f"{mes} {año}"], rotation=45)
    ax.set_ylabel("Consumo (GB)")
    ax.set_title("📈 Consumo mensual histórico vs predicción")
    ax.legend()
    st.pyplot(fig)
