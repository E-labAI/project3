import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # Nueva importación
import random
import numpy as np
from matplotlib.gridspec import GridSpec

# Configuración de estilo (cambiada)
sns.set_theme(style="whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

def leer_archivo_excel(ruta):
    """Lee y procesa el archivo Excel"""
    try:
        df = pd.read_excel(ruta, sheet_name='Lista Lepidopteras')
        
        # Verificar y normalizar nombres de columnas
        if 'Nombre' not in df.columns or '% Coincidencia' not in df.columns:
            df.columns = ['Nombre', '1', '2', '3', '4', '5', '% Coincidencia']
        
        # Convertir porcentajes a valores numéricos si es necesario
        if df['% Coincidencia'].dtype == 'object':
            df['% Coincidencia'] = df['% Coincidencia'].str.replace('=', '').astype(float)
        
        return df
    
    except Exception as e:
        print(f"Error al leer el archivo: {str(e)}")
        return None

def crear_visualizacion(df):
    """Crea la visualización completa con subplots"""
    if df is None or df.empty:
        print("No hay datos para visualizar")
        return
    
    # Seleccionar 3 grupos aleatorios de 10 especies cada uno
    random.seed(42)  # Para reproducibilidad
    grupos = [df.sample(n=10, replace=False) for _ in range(3)]
    
    # Crear figura con diseño personalizado
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 0.4], height_ratios=[1, 1, 1])
    
    # Gráficas de barras para cada grupo
    axs_barras = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[2, 0])
    ]
    
    # Gráficas de distribución para cada grupo
    axs_dist = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 1])
    ]
    
    # Gráfica resumen general
    ax_resumen = fig.add_subplot(gs[:, 2])
    
    # Paleta de colores de Seaborn
    colors = sns.color_palette("husl", 3)
    
    # Procesar cada grupo
    stats_grupos = []
    for i, (grupo, color) in enumerate(zip(grupos, colors)):
        # Gráfico de barras horizontal
        grupo = grupo.sort_values('% Coincidencia', ascending=True)
        nombres = [nombre[:12] + '...' if len(nombre) > 15 else nombre for nombre in grupo['Nombre']]
        
        axs_barras[i].barh(
            y=nombres,
            width=grupo['% Coincidencia'],
            color=color,
            alpha=0.7
        )
        axs_barras[i].set_title(f'Grupo {i+1} - Coincidencias por especie')
        axs_barras[i].set_xlabel('% Coincidencia')
        axs_barras[i].set_xlim(0, 1)
        
        # Histograma de distribución
        axs_dist[i].hist(
            grupo['% Coincidencia'],
            bins=np.arange(0, 1.1, 0.1),
            color=color,
            alpha=0.7,
            edgecolor='white'
        )
        axs_dist[i].set_title(f'Grupo {i+1} - Distribución')
        axs_dist[i].set_xlabel('% Coincidencia')
        axs_dist[i].set_ylabel('Frecuencia')
        axs_dist[i].set_xlim(0, 1)
        
        # Estadísticas para el resumen
        stats = {
            'Grupo': i+1,
            'Media': grupo['% Coincidencia'].mean(),
            'Mediana': grupo['% Coincidencia'].median(),
            'Máximo': grupo['% Coincidencia'].max(),
            'Mínimo': grupo['% Coincidencia'].min(),
            'Color': color
        }
        stats_grupos.append(stats)
    
    # Gráfico resumen comparativo
    for stats in stats_grupos:
        ax_resumen.plot(
            ['Media', 'Mediana', 'Máximo', 'Mínimo'],
            [stats['Media'], stats['Mediana'], stats['Máximo'], stats['Mínimo']],
            'o-',
            color=stats['Color'],
            label=f'Grupo {stats["Grupo"]}',
            alpha=0.7
        )
    
    ax_resumen.set_title('Comparación entre grupos')
    ax_resumen.set_ylabel('% Coincidencia')
    ax_resumen.legend()
    ax_resumen.set_ylim(0, 1)
    
    # Ajustar layout y mostrar
    plt.tight_layout()
    fig.suptitle('Análisis de Coincidencias en Lepidópteros', y=1.02, fontsize=16)
    plt.show()

# Ejecutar el análisis
df = leer_archivo_excel('Lista-Lepidopteras.xlsx')
crear_visualizacion(df)