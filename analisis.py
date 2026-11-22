"""
Proyecto Integrador U4 - Optimizacion Inferencial del Rendimiento Agricola
y Eficiencia de Riego en Cultivos de Invernadero.

Modulo de analisis (Hitos 1, 2 y 3 + funciones de graficado del Hito 4).
La interfaz Streamlit vive en app.py y solo llama a las funciones de aqui.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# --------------------------------------------------------------------------
# Constantes del experimento
# --------------------------------------------------------------------------
SEMILLA = 0.42          # semilla global de DuckDB (setseed)
N_PARCELAS = 100        # N = 100 parcelas
ALPHA = 0.05            # nivel de significancia
ORDEN_TRAT = ["Tratamiento_A", "Tratamiento_B"]
COLORES = {"Tratamiento_A": "#4C72B0", "Tratamiento_B": "#55A868"}

# --------------------------------------------------------------------------
# HITO 1 - Telemetria en DuckDB (SQL)
# --------------------------------------------------------------------------
# Orden de las llamadas random(): riego -> fertilizante -> grupo -> ruido
SQL_CREAR_TABLA = f"""
CREATE OR REPLACE TABLE telemetria AS
WITH base AS (
    SELECT
        i AS parcela_id,
        ROUND(100.0 + random() * 200.0, 2) AS volumen_riego,      -- U(100, 300) m3/ha
        ROUND(50.0  + random() * 100.0, 2) AS fertilizante_kg,    -- U(50, 150) kg/ha
        random()                           AS p_trat,             -- U(0, 1)
        (random() * 1.6 - 0.8)             AS ruido               -- U(-0.8, 0.8) ton/ha
    FROM range(1, {N_PARCELAS + 1}) AS t(i)
),
calc AS (
    SELECT
        parcela_id, volumen_riego, fertilizante_kg,
        CASE WHEN p_trat < 0.50 THEN 'Tratamiento_A' ELSE 'Tratamiento_B' END AS tipo_tratamiento,
        5.0                                   -- beta0
        + 0.035 * volumen_riego               -- beta1
        + 0.020 * fertilizante_kg             -- beta2
        + CASE WHEN p_trat >= 0.50 THEN 1.8 ELSE 0.0 END   -- efecto del bio-estimulante
        + ruido AS y
    FROM base
)
SELECT parcela_id, volumen_riego, fertilizante_kg, tipo_tratamiento,
       ROUND(y, 2) AS rendimiento_ton
FROM calc;
"""

# Temperatura: variable ambiental extra, requerida por el heatmap (Grafica 1).
# Se genera DESPUES de la tabla base para no alterar el flujo aleatorio original
# y NO influye en el rendimiento (su correlacion esperada es ~0).
SQL_ANADIR_TEMPERATURA = [
    "ALTER TABLE telemetria ADD COLUMN temperatura_c DOUBLE;",
    "UPDATE telemetria SET temperatura_c = ROUND(18.0 + random() * 14.0, 2);",
]

# Hito 1.3 - limpieza / extraccion previa al motor analitico de Python
SQL_EXTRAER = """
SELECT parcela_id, volumen_riego, fertilizante_kg, temperatura_c,
       tipo_tratamiento, rendimiento_ton
FROM telemetria
WHERE volumen_riego    IS NOT NULL
  AND fertilizante_kg  IS NOT NULL
  AND rendimiento_ton  IS NOT NULL
  AND tipo_tratamiento IN ('Tratamiento_A', 'Tratamiento_B')
ORDER BY parcela_id;
"""

SQL_VALIDACION = """
SELECT COUNT(*)                                        AS total_parcelas,
       COUNT(*) FILTER (WHERE tipo_tratamiento = 'Tratamiento_A') AS n_A,
       COUNT(*) FILTER (WHERE tipo_tratamiento = 'Tratamiento_B') AS n_B,
       MIN(rendimiento_ton) AS rend_min,
       MAX(rendimiento_ton) AS rend_max,
       COUNT(*) FILTER (WHERE rendimiento_ton IS NULL) AS rend_nulos
FROM telemetria;
"""


def generar_telemetria(con=None):
    """Hito 1: siembra la BD, crea la tabla (N=100) y devuelve un DataFrame."""
    import duckdb  # import local: solo se necesita al generar los datos
    propia = con is None
    if propia:
        con = duckdb.connect(":memory:")
    try:
        con.execute(f"SELECT setseed({SEMILLA})")      # Paso 1.1
        con.execute(SQL_CREAR_TABLA)                   # Paso 1.2
        for sentencia in SQL_ANADIR_TEMPERATURA:
            con.execute(sentencia)
        df = con.execute(SQL_EXTRAER).df()             # Paso 1.3
    finally:
        if propia:
            con.close()
    return df


# --------------------------------------------------------------------------
# HITO 2 - Correlacion de Pearson y regresion lineal OLS (Subtema 4.2)
# --------------------------------------------------------------------------
ETIQUETAS = {
    "volumen_riego": "Riego",
    "fertilizante_kg": "Fertilizante",
    "temperatura_c": "Temperatura",
    "rendimiento_ton": "Rendimiento",
}


def matriz_correlacion(df):
    return df[list(ETIQUETAS)].rename(columns=ETIQUETAS).corr(method="pearson")


def ajustar_ols(df, incluir_tratamiento=False):
    """Rendimiento = b0 + b1*Riego + b2*Fertilizante (+ b3*TratamientoB opcional)."""
    X = df[["volumen_riego", "fertilizante_kg"]].copy()
    if incluir_tratamiento:
        X["tratamiento_B"] = (df["tipo_tratamiento"] == "Tratamiento_B").astype(float)
    X = sm.add_constant(X, has_constant="add")
    return sm.OLS(df["rendimiento_ton"], X).fit()


def metricas_ols(modelo):
    resid = np.asarray(modelo.resid)
    return {
        "r2": float(modelo.rsquared),
        "r2_aj": float(modelo.rsquared_adj),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "coef": modelo.params,
        "pvalores": modelo.pvalues,
    }


# --------------------------------------------------------------------------
# HITO 3 - Inferencia: t de Student de dos muestras independientes (4.3)
# H0: mu_B - mu_A <= 0   vs   H1: mu_B - mu_A > 0   (cola derecha)
# --------------------------------------------------------------------------
def prueba_t(df, alpha=ALPHA):
    a = df.loc[df["tipo_tratamiento"] == "Tratamiento_A", "rendimiento_ton"].to_numpy()
    b = df.loc[df["tipo_tratamiento"] == "Tratamiento_B", "rendimiento_ton"].to_numpy()
    if len(a) < 2 or len(b) < 2:
        return None
    na, nb = len(a), len(b)
    t_stat, p_valor = stats.ttest_ind(b, a, equal_var=True, alternative="greater")
    _, p_welch = stats.ttest_ind(b, a, equal_var=False, alternative="greater")
    diff = b.mean() - a.mean()
    gl = na + nb - 2
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / gl
    se = np.sqrt(sp2 * (1 / na + 1 / nb))
    t_crit = stats.t.ppf(1 - 0.05 / 2, gl)           # IC bilateral al 95 %
    return {
        "n_A": na, "n_B": nb,
        "media_A": float(a.mean()), "media_B": float(b.mean()),
        "dif_medias": float(diff),
        "t": float(t_stat), "gl": int(gl),
        "p_valor": float(p_valor), "p_welch": float(p_welch),
        "ic_inf": float(diff - t_crit * se), "ic_sup": float(diff + t_crit * se),
        "d_cohen": float(diff / np.sqrt(sp2)),
        "levene_p": float(stats.levene(a, b).pvalue),
        "rechaza_h0": bool(p_valor < alpha),
    }


def veredicto(t):
    if t is None:
        return "Selecciona ambos tratamientos (con al menos 2 parcelas cada uno) para ejecutar la prueba t."
    if t["rechaza_h0"]:
        return (f"p = {t['p_valor']:.3g} < {ALPHA}: se RECHAZA H0. El Tratamiento_B incrementa "
                f"significativamente el rendimiento medio (diferencia = {t['dif_medias']:.2f} ton/ha, "
                f"IC 95 % [{t['ic_inf']:.2f}, {t['ic_sup']:.2f}]). Se recomienda autorizar la inversion "
                f"para escalar el Tratamiento_B.")
    return (f"p = {t['p_valor']:.3g} >= {ALPHA}: NO se rechaza H0. No hay evidencia suficiente de que el "
            f"Tratamiento_B incremente el rendimiento medio; no se recomienda escalar la inversion todavia.")


def interpretar_r2(r2_aj):
    return (f"El {r2_aj * 100:.1f} % de la variabilidad del rendimiento es explicada por el riego y el "
            f"fertilizante (R2 ajustado); el resto corresponde a otros factores no incluidos en el modelo.")


# --------------------------------------------------------------------------
# HITO 4 - Las 4 graficas del dashboard (matplotlib / seaborn)
# --------------------------------------------------------------------------
sns.set_theme(style="whitegrid")


def _orden_presente(df):
    return [t for t in ORDEN_TRAT if t in set(df["tipo_tratamiento"])]


def grafica_heatmap(df):
    """Grafica 1: mapa de calor de la correlacion de Pearson."""
    fig, ax = plt.subplots(figsize=(6, 4.4))
    sns.heatmap(matriz_correlacion(df), annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, square=True, linewidths=.5,
                cbar_kws={"label": "r de Pearson"}, ax=ax)
    ax.set_title("Correlacion de Pearson")
    fig.tight_layout()
    return fig


def grafica_dispersion(df):
    """Grafica 2: dispersion Riego vs Rendimiento + recta OLS + banda de confianza 95 %."""
    fig, ax = plt.subplots(figsize=(6, 4.4))
    orden = _orden_presente(df)
    sns.scatterplot(data=df, x="volumen_riego", y="rendimiento_ton",
                    hue="tipo_tratamiento", hue_order=orden, palette=COLORES,
                    alpha=.85, edgecolor="white", ax=ax)
    X = sm.add_constant(df[["volumen_riego"]], has_constant="add")
    m = sm.OLS(df["rendimiento_ton"], X).fit()
    xs = np.linspace(df["volumen_riego"].min(), df["volumen_riego"].max(), 100)
    Xs = sm.add_constant(pd.DataFrame({"volumen_riego": xs}), has_constant="add")
    sf = m.get_prediction(Xs).summary_frame(alpha=0.05)
    b0, b1 = m.params.iloc[0], m.params.iloc[1]
    ax.plot(xs, sf["mean"], color="black", lw=2, label=f"OLS: y = {b0:.2f} + {b1:.4f}x")
    ax.fill_between(xs, sf["mean_ci_lower"], sf["mean_ci_upper"], color="gray", alpha=.25,
                    label="Banda de confianza 95 %")
    ax.set_xlabel(r"Volumen de riego ($\mathrm{m^3/ha}$)")
    ax.set_ylabel(r"Rendimiento ($\mathrm{ton/ha}$)")
    ax.set_title("Regresion lineal OLS (Riego vs Rendimiento)")
    ax.legend(title="", fontsize=8)
    fig.tight_layout()
    return fig


def grafica_boxplot(df):
    """Grafica 3: boxplot por tratamiento con media e IC 95 % de cada grupo."""
    fig, ax = plt.subplots(figsize=(6, 4.4))
    orden = _orden_presente(df)
    sns.boxplot(data=df, x="tipo_tratamiento", y="rendimiento_ton", order=orden,
                hue="tipo_tratamiento", palette=COLORES, legend=False,
                width=.5, showmeans=False, ax=ax)
    sns.stripplot(data=df, x="tipo_tratamiento", y="rendimiento_ton", order=orden,
                  color="black", alpha=.35, size=3, ax=ax)
    for i, trat in enumerate(orden):
        y = df.loc[df["tipo_tratamiento"] == trat, "rendimiento_ton"]
        if len(y) > 1:
            media = y.mean()
            semi = stats.t.ppf(0.975, len(y) - 1) * y.std(ddof=1) / np.sqrt(len(y))
            ax.errorbar(i, media, yerr=semi, fmt="D", color="red", capsize=6, zorder=5,
                        label="Media e IC 95 %" if i == 0 else None)
    ax.set_xlabel("Tipo de tratamiento")
    ax.set_ylabel(r"Rendimiento ($\mathrm{ton/ha}$)")
    ax.set_title("Rendimiento por tratamiento")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def grafica_residuos(modelo):
    """Grafica 4: histograma de residuos + KDE + curva normal teorica."""
    resid = np.asarray(modelo.resid)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    sns.histplot(resid, kde=True, stat="density", bins=15, color="#8172B2", ax=ax,
                 label="Residuos (hist. + KDE)")
    xs = np.linspace(resid.min() - 0.5, resid.max() + 0.5, 200)
    ax.plot(xs, stats.norm.pdf(xs, 0, resid.std(ddof=1)), color="black", lw=2, ls="--",
            label=r"Normal $N(0,\hat{\sigma}^2)$")
    p_sw = stats.shapiro(resid).pvalue if len(resid) >= 3 else float("nan")
    ax.set_xlabel(r"Residuo = $Y_{real} - \hat{Y}$ ($\mathrm{ton/ha}$)")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Diagnostico de residuos (Shapiro-Wilk p = {p_sw:.3f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
