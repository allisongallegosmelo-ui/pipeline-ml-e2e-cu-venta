"""Dashboard Streamlit opcional para el extra de la tarea."""

import glob
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard Pipeline CU Venta", layout="wide")
st.title("Dashboard Pipeline CU Venta")

files = sorted(glob.glob("Datos/Output/final/output_tlv_*.csv"))
if not files:
    st.warning("No se encontró output TLV. Ejecuta primero el pipeline.")
    st.stop()

selected = st.selectbox("Archivo TLV", files, index=len(files) - 1)
df = pd.read_csv(selected)

col1, col2, col3 = st.columns(3)
col1.metric("Clientes", f"{len(df):,}")
col2.metric("Score promedio", f"{df['prob'].mean():.4f}")
col3.metric("TLV promedio", f"{df['puntuacion_tlv'].mean():.6f}")

st.subheader("Distribución de grupos de ejecución")
st.bar_chart(df["grupo_ejec_tlv"].value_counts().sort_index())

st.subheader("Top 20 clientes por puntuación TLV")
cols = ["key_value", "prob", "puntuacion_tlv", "grupo_ejec_tlv", "monto"]
st.dataframe(df[cols].sort_values("puntuacion_tlv", ascending=False).head(20), use_container_width=True)

st.subheader("Monto promedio por grupo")
st.bar_chart(df.groupby("grupo_ejec_tlv")["monto"].mean().sort_index())
