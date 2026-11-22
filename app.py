"""
Dashboard Streamlit - Proyecto Integrador U4 (INC-2505)
Optimizacion Inferencial del Rendimiento Agricola y Eficiencia de Riego
en Cultivos de Invernadero.

Ejecutar localmente:  streamlit run app.py
"""
import streamlit as st
import analisis as an

st.set_page_config(page_title="Invernadero inteligente - Dashboard inferencial",
                   page_icon="🌱", layout="wide")


@st.cache_data(show_spinner="Generando telemetria con DuckDB...")
def cargar_datos():
    return an.generar_telemetria()


df_total = cargar_datos()

# ------------------------------------------------------------------ Sidebar
st.sidebar.header("Filtros dinamicos")
trats = st.sidebar.multiselect("Tipo de tratamiento", an.ORDEN_TRAT, default=an.ORDEN_TRAT)

r_lo, r_hi = int(df_total["volumen_riego"].min()), int(df_total["volumen_riego"].max()) + 1
f_lo, f_hi = int(df_total["fertilizante_kg"].min()), int(df_total["fertilizante_kg"].max()) + 1
rango_riego = st.sidebar.slider("Volumen de riego (m³/ha)", r_lo, r_hi, (r_lo, r_hi))
rango_fert = st.sidebar.slider("Fertilizante (kg/ha)", f_lo, f_hi, (f_lo, f_hi))
incluir_trat = st.sidebar.checkbox("Incluir tratamiento como variable del modelo (extra)", value=False,
                                   help="Por defecto el modelo es Rendimiento ~ Riego + Fertilizante, como pide el proyecto.")
st.sidebar.caption(f"Semilla DuckDB: {an.SEMILLA} · N = {an.N_PARCELAS} parcelas")

df = df_total[
    df_total["tipo_tratamiento"].isin(trats)
    & df_total["volumen_riego"].between(*rango_riego)
    & df_total["fertilizante_kg"].between(*rango_fert)
]

# ------------------------------------------------------------------ Encabezado
st.title("🌱 Optimización inferencial del rendimiento agrícola y eficiencia de riego")
st.caption("Invernaderos inteligentes de jitomate/aguacate · Análisis y Exploración de Datos (INC-2505) · "
           "Subtemas 4.2, 4.3 y Unidad 6")

if len(df) < 10:
    st.warning("Con los filtros actuales quedan menos de 10 parcelas. Amplía los filtros para ver el análisis.")
    st.stop()

modelo = an.ajustar_ols(df, incluir_trat)
met = an.metricas_ols(modelo)
t = an.prueba_t(df)

# ------------------------------------------------------------------ KPIs
k1, k2, k3 = st.columns(3)
media = df["rendimiento_ton"].mean()
k1.metric("Rendimiento promedio global", f"{media:.2f} ton/ha",
          delta=f"{media - df_total['rendimiento_ton'].mean():+.2f} vs. total", delta_color="off")
k2.metric("R² ajustado del modelo", f"{met['r2_aj'] * 100:.1f} %",
          help="Porcentaje de la variabilidad del rendimiento explicado por el modelo OLS.")
k3.metric("p-valor de la prueba t (cola derecha)",
          "N/D" if t is None else f"{t['p_valor']:.2e}",
          help="H0: μB − μA ≤ 0  vs  H1: μB − μA > 0 · α = 0.05")

if t is not None and t["rechaza_h0"]:
    st.success("**Veredicto de ingeniería:** " + an.veredicto(t))
elif t is not None:
    st.info("**Veredicto de ingeniería:** " + an.veredicto(t))
else:
    st.warning(an.veredicto(t))

st.divider()

# ------------------------------------------------------------------ Cuadrícula 2x2
fila1 = st.columns(2)
with fila1[0]:
    st.subheader("1 · Mapa de calor de correlación")
    st.pyplot(an.grafica_heatmap(df), clear_figure=True)
    st.caption("Identifica el factor dominante: el de mayor |r| con el rendimiento.")
with fila1[1]:
    st.subheader("2 · Dispersión con ajuste OLS")
    st.pyplot(an.grafica_dispersion(df), clear_figure=True)
    st.caption("Recta de mínimos cuadrados con banda de confianza al 95 %.")

fila2 = st.columns(2)
with fila2[0]:
    st.subheader("3 · Boxplot comparativo de tratamientos")
    st.pyplot(an.grafica_boxplot(df), clear_figure=True)
    st.caption("Soporte visual de la prueba t: medianas, cuartiles, atípicos y media con IC 95 %.")
with fila2[1]:
    st.subheader("4 · Histograma de residuos")
    st.pyplot(an.grafica_residuos(modelo), clear_figure=True)
    st.caption("Validación del supuesto de normalidad de los errores.")

st.divider()

# ------------------------------------------------------------------ Detalle técnico
tab1, tab2, tab3, tab4 = st.tabs(["Modelo OLS", "Prueba t", "Datos", "Hito 1: SQL (DuckDB)"])

with tab1:
    st.markdown(f"**RMSE:** {met['rmse']:.3f} ton/ha · **R²:** {met['r2']:.4f} · "
                f"**R² ajustado:** {met['r2_aj']:.4f}")
    st.write(an.interpretar_r2(met["r2_aj"]))
    st.code(modelo.summary().as_text())

with tab2:
    if t is None:
        st.warning(an.veredicto(t))
    else:
        c1, c2 = st.columns(2)
        c1.markdown(f"""
**Hipótesis**  
H₀: μB − μA ≤ 0 · H₁: μB − μA > 0 (α = 0.05)

| | Tratamiento A | Tratamiento B |
|---|---|---|
| n | {t['n_A']} | {t['n_B']} |
| Media (ton/ha) | {t['media_A']:.3f} | {t['media_B']:.3f} |
""")
        c2.markdown(f"""
**Resultados**  
- Diferencia de medias (B − A): **{t['dif_medias']:.3f} ton/ha**
- Estadístico t = {t['t']:.3f} (gl = {t['gl']})
- p-valor (cola derecha) = {t['p_valor']:.3e}
- IC 95 % de la diferencia: [{t['ic_inf']:.3f}, {t['ic_sup']:.3f}]
- d de Cohen = {t['d_cohen']:.2f}
- Robustez: p (Welch) = {t['p_welch']:.3e} · Levene p = {t['levene_p']:.3f}
""")
        st.write(an.veredicto(t))

with tab3:
    st.dataframe(df)
    st.download_button("Descargar datos filtrados (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name="telemetria_filtrada.csv", mime="text/csv")

with tab4:
    st.write(f"Semilla fija `setseed({an.SEMILLA})` y generación de N = {an.N_PARCELAS} parcelas.")
    st.code(an.SQL_CREAR_TABLA, language="sql")
    st.code("\n".join(an.SQL_ANADIR_TEMPERATURA), language="sql")
    st.code(an.SQL_EXTRAER, language="sql")
