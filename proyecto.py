import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title="Dashboard de Tráfico NAPs", layout="wide")
st.title("📈 Dashboard de Tráfico de Internet en Colombia")

# Cargar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv("Monitoreo de Tráfico de Internet - Trafico Diario_133 - Monitoreo de Tráfico de Internet - Trafico Diario_133.csv",
                     sep=',', encoding='latin1')
  
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

# Barra lateral de navegación
st.sidebar.title("📊 Navegación del dashboard")
opcion = st.sidebar.radio("Selecciona una visualización:", (
    "Volumen NAPs por Año",
    "NAPs por Proveedor",
    "Tráfico Local",
    "Tráfico Internacional",
    "Tráfico por Peering",
    "Predicción de Septiembre 2021"
))


# Función para formatear valores
def formatear_valor(v):
    return f"{v/1e9:.1f}B" if v >= 1e9 else f"{v/1e6:.1f}M"

# Visualización condicional
if opcion == "Volumen NAPs por Año":
    resumen = (
        df.groupby(['año'], as_index=False)['trafico datos: naps - colombia (gb)']
        .sum()
        .rename(columns={'trafico datos: naps - colombia (gb)': 'total_naps_gb'})
    )
    años_disponibles = resumen['año'].unique()
    años_seleccionados = st.sidebar.multiselect("Filtrar por año", años_disponibles, default=años_disponibles)
    resumen = resumen[resumen['año'].isin(años_seleccionados)]

    fig = px.bar(
        resumen, x='año', y='total_naps_gb', text='total_naps_gb',
        title='📶 Volumen anual de tráfico NAPs – Colombia',
        labels={'año': 'Año', 'total_naps_gb': 'Tráfico NAPs (GB)'},
        color='total_naps_gb', color_continuous_scale='Blues'
    )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True, key="nap_anual")

elif opcion == "NAPs por Proveedor":
    resumen = (
        df.groupby(['proveedor'], as_index=False)['trafico datos: naps - colombia (gb)']
        .sum().rename(columns={'trafico datos: naps - colombia (gb)': 'total_naps_gb'})
    ).sort_values(by='total_naps_gb', ascending=False)
    resumen['text'] = resumen['total_naps_gb'].apply(formatear_valor)
    fig = px.bar(
        resumen, x='proveedor', y='total_naps_gb', text='text',
        title='📡 Volumen de tráfico NAPs por proveedor',
        labels={'proveedor': 'Proveedor', 'total_naps_gb': 'Tráfico NAPs (GB)'},
        color='total_naps_gb', color_continuous_scale='Blues'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True, key="nap_proveedor")

elif opcion == "Tráfico Local":
    resumen = (
        df.groupby(['proveedor'], as_index=False)['trafico datos: local (gb)']
        .sum().rename(columns={'trafico datos: local (gb)': 'trafico_local'})
    ).sort_values(by='trafico_local', ascending=False)
    resumen['text'] = resumen['trafico_local'].apply(formatear_valor)
    fig = px.bar(
        resumen, x='proveedor', y='trafico_local', text='text',
        title='📡 Volumen de tráfico local por proveedor',
        labels={'proveedor': 'Proveedor', 'trafico_local': 'Tráfico Local (GB)'},
        color='trafico_local', color_continuous_scale='Blues'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True, key="local")

elif opcion == "Tráfico Internacional":
    resumen = (
        df.groupby(['proveedor'], as_index=False)['trafico datos: internacional (gb)']
        .sum().rename(columns={'trafico datos: internacional (gb)': 'trafico_internacional'})
    ).sort_values(by='trafico_internacional', ascending=False)
    resumen['text'] = resumen['trafico_internacional'].apply(formatear_valor)
    fig = px.bar(
        resumen, x='proveedor', y='trafico_internacional', text='text',
        title='🌐 Volumen de tráfico internacional por proveedor',
        labels={'proveedor': 'Proveedor', 'trafico_internacional': 'Tráfico Internacional (GB)'},
        color='trafico_internacional', color_continuous_scale='Blues'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True, key="internacional")


elif opcion == "Tráfico por Peering":
    resumen = (
        df.groupby(['proveedor'], as_index=False)['trafico datos: acuerdos de transito o peering directo (gb)']
        .sum().rename(columns={'trafico datos: acuerdos de transito o peering directo (gb)': 'trafico_peering'})
    ).sort_values(by='trafico_peering', ascending=False)
    resumen['text'] = resumen['trafico_peering'].apply(formatear_valor)
    fig = px.bar(
        resumen, x='proveedor', y='trafico_peering', text='text',
        title='🔄 Volumen de tráfico por acuerdos de tránsito o peering',
        labels={'proveedor': 'Proveedor', 'trafico_peering': 'Tráfico Peering (GB)'},
        color='trafico_peering', color_continuous_scale='Blues'
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True, key="peering")


elif opcion == "Predicción de Septiembre 2021":
    st.header("🔮 Predicción del consumo de datos - Septiembre 2021")

    # Columnas que vamos a sumar
    trafico_columnas = [
        'trafico datos: internacional (gb)',
        'trafico datos: naps - colombia (gb)',
        'trafico datos: acuerdos de transito o peering directo (gb)',
        'trafico datos: local (gb)'
    ]

    # Filtrar y agrupar por año y mes
    df_2020 = df[df['año'] == 2020]
    df_2021 = df[df['año'] == 2021]

    meses_2020 = df_2020.groupby('mes', as_index=False)[trafico_columnas].sum()
    meses_2021 = df_2021.groupby('mes', as_index=False)[trafico_columnas].sum()

    # Agregar columna año
    meses_2020["año"] = 2020
    meses_2021["año"] = 2021

    # Sumar columnas de tráfico total por mes
    meses_2020["Suma_mensual_total"] = meses_2020[trafico_columnas].sum(axis=1)
    meses_2021["Suma_mensual_total"] = meses_2021[trafico_columnas].sum(axis=1)

    # Agregar nombre del mes
    nombres_meses = {1:"Ene",2:"Feb",3:"Mar",4:"Abr",5:"May",6:"Jun",7:"Jul",8:"Ago",9:"Sep",10:"Oct",11:"Nov",12:"Dic"}
    meses_2020["nombre_mes"] = meses_2020["mes"].map(nombres_meses)
    meses_2021["nombre_mes"] = meses_2021["mes"].map(nombres_meses)

    # Unir ambos años
    meses_todos = pd.concat([meses_2020, meses_2021], ignore_index=True)
    meses_todos = meses_todos.sort_values(['año', 'mes']).reset_index(drop=True)
    meses_todos['t'] = range(1, len(meses_todos) + 1)

    # Separar datos
    # Eliminar octubre 2021 si existe
    meses_todos_filtrado = meses_todos[~((meses_todos['año'] == 2021) & (meses_todos['mes'] == 10))]

    # Asignar train y test
    train = meses_todos_filtrado.iloc[:-1].copy()
    test = meses_todos_filtrado.iloc[-1:].copy()



    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    import numpy as np

    X_train = train['t'].values.reshape(-1, 1)
    y_train = train['Suma_mensual_total'].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    t_test = np.array([[len(X_train)]])
    y_pred = model.predict(t_test)[0]
    y_true = test["Suma_mensual_total"].values[0]
    mes, año = test['nombre_mes'].iloc[0], test['año'].iloc[0]

    # Mostrar qué mes se va a predecir
    st.markdown(f"📌 Mes predicho: **{test['nombre_mes'].iloc[0]} {test['año'].iloc[0]}**")

    st.markdown(f"📅 **Predicción para {mes} {año}:** `{y_pred:.3e}` GB")
    st.markdown(f"📊 **Valor real para {mes} {año}:** `{y_true:.3e}` GB")
    st.markdown(f"❗ **Error:** `{(y_pred - y_true):.3e}` GB (`{(y_pred - y_true)/y_true*100:.1f}%`)")


    # Gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train['t'], train['Suma_mensual_total'], marker='o', linestyle='-', color='C0', label='Histórico')
    ax.scatter(t_test, y_true, color='green', s=100, label=f'Real: {mes} {año}')
    ax.scatter(t_test, y_pred, color='red', s=100, label=f'Predicción: {mes} {año}')
    ax.legend()
    ax.set_xlabel('Mes')
    ax.set_ylabel('Uso total de datos (GB)')
    ax.set_title('📈 Consumo mensual: histórico vs predicción')
    ticks = list(train['t']) + [t_test[0][0]]
    labels = list(train['nombre_mes'] + ' ' + train['año'].astype(str)) + [f'{mes} {año}']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)
    st.pyplot(fig)






