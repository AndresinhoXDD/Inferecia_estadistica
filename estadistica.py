import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode, probplot
from sklearn.cluster import KMeans

# ----------------------------------------------------------
# 1. GENERACIÓN DE DATOS FICTICIOS CON NUEEVAS VARIABLES
# ----------------------------------------------------------
np.random.seed(42)
n = 100
# Variables cualitativas
genero = np.random.choice(['Masculino', 'Femenino'], size=n)
repetidor = np.random.choice(['Sí', 'No'], size=n, p=[0.2, 0.8])
# Variables cuantitativas
edad = np.random.randint(17, 30, size=n)            # Edad en años
horas_estudio = np.round(np.random.uniform(0, 20, size=n), 1)
corte1 = np.random.uniform(0, 100, size=n)
corte2 = np.random.uniform(0, 100, size=n)
corte3 = np.random.uniform(0, 100, size=n)
# Nota final en escala 0-100 y luego 0-5
nota_final = corte1 * 0.3 + corte2 * 0.3 + corte3 * 0.4
nota_5 = (nota_final / 100) * 5
# Notas semestrales (6 semestres)
notas_semestrales = np.random.uniform(0, 5, size=(n, 6))

# Construcción de DataFrame
cols = {
    'Género': genero,
    'Repetidor': repetidor,
    'Edad': edad,
    'Horas_Estudio': horas_estudio,
    'Corte1': np.round(corte1, 1),
    'Corte2': np.round(corte2, 1),
    'Corte3': np.round(corte3, 1),
    'Nota_Final_100': np.round(nota_final, 1),
    'Nota_Final_5': np.round(nota_5, 2)
}
df = pd.DataFrame(cols)
for i in range(6):
    df[f'Semestre_{i+1}'] = np.round(notas_semestrales[:, i], 2)

# ----------------------------------------------------------
# 2. CATEGORÍAS DERIVADAS
# ----------------------------------------------------------
# Categoría de rendimiento (ordinal)
bins_cat = [0, 2.5, 3.5, 4.5, 5.0]
labels_cat = ['Malo', 'Bueno', 'Sobresaliente', 'Excelente']
df['Categoría'] = pd.cut(df['Nota_Final_5'], bins=bins_cat, labels=labels_cat, include_lowest=True)
# Cluster de desempeño (KMeans)
kmeans = KMeans(n_clusters=3, random_state=42).fit(df[['Nota_Final_5']])
df['Cluster'] = ['Grupo ' + str(label+1) for label in kmeans.labels_]

# ----------------------------------------------------------
# 3. TABLA: ESTADÍSTICAS DESCRIPTIVAS FORMATEADA
# ----------------------------------------------------------
desc = df['Nota_Final_5'].describe(percentiles=[.1, .25, .5, .75, .9])
desc['mode'] = mode(df['Nota_Final_5'], keepdims=True).mode[0]
# Formato de valores
txt_vals = []
for key, val in desc.items():
    if key == 'count':
        txt_vals.append(f"{int(val)}")
    else:
        txt_vals.append(f"{val:.2f}")
# Fig y tabla
fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('off')
cell_cols = [['#f2f2f2'] * len(txt_vals)]
tbl = ax.table(
    cellText=[txt_vals],
    colLabels=[k.capitalize() for k in desc.index],
    cellLoc='center', colLoc='center', loc='center',
    cellColours=cell_cols
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2)
# Bordes y negrita encabezados
for (i, j), cell in tbl.get_celld().items():
    cell.set_edgecolor('black')
    if i == 0:
        cell.set_text_props(weight='bold')
plt.title('Estadísticas Descriptivas de Nota Final (0-5)', pad=20)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 4. TABLA: DISTRIBUCIÓN DE FRECUENCIAS FORMATEADA
# ----------------------------------------------------------
tf = df['Nota_Final_5'].value_counts(bins=6).sort_index().reset_index()
tf.columns = ['Rango', 'Frecuencia']
tf['Frecuencia_Rel'] = (tf['Frecuencia'] / n).round(2)
tf['Acumulada'] = tf['Frecuencia'].cumsum().astype(int)
# Fig y tabla
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
tbl2 = ax.table(
    cellText=tf.values,
    colLabels=tf.columns,
    cellLoc='center', colLoc='center', loc='center'
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(9)
tbl2.scale(1, 1.5)
# Estilo encabezados y bordes
for (i, j), cell in tbl2.get_celld().items():
    cell.set_edgecolor('black')
    if i == 0:
        cell.set_facecolor('#f2f2f2')
        cell.set_text_props(weight='bold')
plt.title('Distribución de Frecuencias de Nota Final', pad=20)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 5. TABLA: DISTRIBUCIÓN DE PROBABILIDAD AGRUPADA
# ----------------------------------------------------------
# Agrupar redondeando a 1 decimal
tp = df['Nota_Final_5'].round(1).value_counts(normalize=True).sort_index().reset_index()
tp.columns = ['Promedio', 'Probabilidad']
tp['Probabilidad'] = tp['Probabilidad'].round(2)
# Fig y tabla (top 10)
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')
tbl3 = ax.table(
    cellText=tp.head(10).values,
    colLabels=tp.columns,
    cellLoc='center', colLoc='center', loc='center'
)
tbl3.auto_set_font_size(False)
tbl3.set_fontsize(10)
tbl3.scale(1, 2)
for (i, j), cell in tbl3.get_celld().items():
    cell.set_edgecolor('black')
    if i == 0:
        cell.set_facecolor('#f2f2f2')
        cell.set_text_props(weight='bold')
plt.title('Distribución de Probabilidad (Nota Final redondeada)', pad=20)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# 6. VISUALIZACIONES DE VARIABLES
# ----------------------------------------------------------
# a) Cualitativas
g = sns.set_theme()
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
sns.countplot(x='Género', data=df, ax=axs[0, 0]).set_title('Género')
sns.countplot(x='Repetidor', data=df, ax=axs[0, 1]).set_title('Repetidor')
sns.countplot(x='Categoría', data=df, order=labels_cat, ax=axs[1, 0]).set_title('Categoría')
sns.countplot(x='Cluster', data=df, ax=axs[1, 1]).set_title('Cluster')
plt.tight_layout()
plt.show()

# b) Cuantitativas
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['Edad'], kde=True, ax=axs[0]).set_title('Edad (años)')
sns.histplot(df['Horas_Estudio'], kde=True, ax=axs[1]).set_title('Horas de Estudio')
plt.tight_layout()
plt.show()

# Boxplot de notas
temp_cols = ['Corte1', 'Corte2', 'Corte3', 'Nota_Final_5']
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[temp_cols])
plt.title('Boxplot de Notas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# c) Tendencia semestral promedio
prom_sem = df[[f'Semestre_{i+1}' for i in range(6)]].mean()
plt.figure(figsize=(8, 4))
plt.plot(range(1, 7), prom_sem, marker='o')
plt.title('Evolución Promedio Semestral')
plt.xlabel('Semestre')
plt.ylabel('Nota Media')
plt.grid(True)
plt.tight_layout()
plt.show()

# d) QQ-plot de nota final
plt.figure(figsize=(6, 6))
probplot(df['Nota_Final_5'], dist='norm', plot=plt)
plt.title('QQ-plot Nota Final')
plt.tight_layout()
plt.show()
