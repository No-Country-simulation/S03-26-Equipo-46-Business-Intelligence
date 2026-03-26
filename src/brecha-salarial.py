import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. CARGA Y CONSOLIDACIÓN (ETL)
# Cargar datos
empleados = pd.read_csv('./data/empleados.csv')
puestos = pd.read_csv('./data/puestos.csv')
salarios = pd.read_csv('./data/salarios.csv')

# Cambiar nombre 'sueldo' a 'salario'
salarios = salarios.rename(columns={'sueldo': 'salario'})

# Quedarse con el último salario por empleado
df_actual = salarios.sort_values('mes').drop_duplicates('id_empleado', keep='last')

# Unir tablas
df_master = pd.merge(empleados, df_actual, on='id_empleado', how='inner')
df_master = pd.merge(df_master, puestos, on='id_puesto', how='inner')

# 2. GENERACIÓN DE VARIABLE SINTÉTICA "COLOR DE PIEL" (Escala PERLA 1-11)
# Simulamos una distribución basada en la etnia para que el análisis de colorismo tenga sentido
# 1 = Muy claro, 11 = Muy oscuro
np.random.seed(42)  # Para reproducibilidad

def generate_piel(etnia):
    if etnia == 'blanco':
        return np.random.randint(1, 4)
    elif etnia == 'griego':
        return np.random.randint(2, 5)
    elif etnia == 'indio':
        return np.random.randint(5, 9)
    else:
        return np.random.randint(1, 12)

df_master['tono_piel'] = df_master['etnia'].apply(generate_piel)

# Guardar Dataset Maestro
df_master.to_csv('./data/dataset_maestro.csv', index=False)

# 3. ANÁLISIS DE IMPACTO (REGRESIÓN MÚLTIPLE)
# Crear dummies para variables categóricas
df_dummies = pd.get_dummies(df_master, columns=['genero', 'etnia', 'seniority', 'departamento'], drop_first=True)

# Identificar columnas de interés para el modelo
# Buscamos capturar género, discapacidad, tono de piel (colorismo) y factores de control (seniority, edad)
columnas_x = [col for col in df_dummies.columns if any(x in col for x in ['genero_', 'discapacidad', 'tono_piel', 'seniority_', 'edad'])]

X = df_dummies[columnas_x]
y = df_dummies['salario']

# Entrenar el modelo
model = LinearRegression().fit(X, y)

# 4. RESULTADOS Y VISUALIZACIÓN
print("--- COEFICIENTES DE IMPACTO SALARIAL ---")
for feature, coef in zip(X.columns, model.coef_):
    print(f"Impacto de {feature}: {coef:.2f} soles")

# Visualización 1: Boxplot Salario por Género
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_master, x='genero', y='salario')
plt.title('Distribución Salarial por Género')
plt.savefig('./data/boxplot_genero.png')

# Visualización 2: Correlación Tono de Piel vs Salario
plt.figure(figsize=(10, 6))
sns.regplot(data=df_master, x='tono_piel', y='salario', scatter_kws={'alpha':0.5})
plt.title('Correlación: Tono de Piel (PERLA 1-11) vs Salario')
plt.savefig('./data/correlacion_piel.png')

# Visualización 3: Mapa de Calor por Departamento
pivot_gap = df_master.groupby(['departamento', 'genero'])['salario'].mean().unstack()
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_gap, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title('Promedio Salarial por Departamento y Género')
plt.savefig('./data/heatmap_departamento.png')

print("\n--- PROCESO COMPLETADO ---")
print("Dataset Maestro guardado en ./data/dataset_maestro.csv")
print("Gráficos guardados en ./data/")