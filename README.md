# La rana estadística en mundos paralelos — Caminatas aleatorias 1D, 2D y 3D

Punto 2 del Taller Primer 50% de **Simulación por Computador** (UPTC).

Simulación de una rana que parte del origen y da **1.000.000 de saltos** aleatorios de una unidad en una, dos o tres dimensiones. Los números pseudoaleatorios se generan con el módulo propio [`pseudo_gen/`](pseudo_gen/) (desarrollado desde cero, sin `random` ni `numpy.random`). Una secuencia **solo se usa en la simulación si aprueba las 6 pruebas estadísticas** del módulo.

---

## Estructura del proyecto

```text
punto2/
├── pseudo_gen/              # Módulo de generadores y pruebas estadísticas (ver su README)
│   └── venv/                # Entorno virtual con numpy, scipy y matplotlib
├── validacion.py            # Genera 1.000.000 de R_i y los somete a las 6 pruebas
├── rana_1d.py               # Caminata en la recta (Z)
├── rana_2d.py               # Caminata en el plano (Z²)
├── rana_3d.py               # Caminata en el espacio (Z³)
├── semillas_100.txt         # Archivo de ejemplo con 100 semillas
├── semillas_5.txt           # Archivo de ejemplo con 5 semillas (pruebas rápidas)
└── resultados/              # Gráficas (.png) y tablas (.csv) generadas
```

---

## Requisitos

- Python 3.10 o superior (probado con Python 3.14).
- `numpy`, `scipy` y `matplotlib`, que ya vienen instalados en `pseudo_gen/venv`.

Todos los comandos se ejecutan desde la carpeta `punto2`. En Windows se usa el intérprete del entorno virtual:

```
pseudo_gen\venv\Scripts\python.exe rana_1d.py --semilla 12345
```

Si se usa otro intérprete, primero se instalan las dependencias:

```
pip install numpy scipy matplotlib
```

---

## Uso

Los tres programas (`rana_1d.py`, `rana_2d.py`, `rana_3d.py`) tienen la misma interfaz. El número de saltos es **siempre 1.000.000** (constante `N_PASOS` en `rana_1d.py`).

### Una simulación con una semilla

```
pseudo_gen\venv\Scripts\python.exe rana_1d.py --semilla 12345
pseudo_gen\venv\Scripts\python.exe rana_2d.py --semilla 12345
pseudo_gen\venv\Scripts\python.exe rana_3d.py --semilla 12345
```

Muestra el resultado de cada prueba estadística y, si todas se aprueban, simula la caminata, imprime sus métricas y grafica la trayectoria.

### Varias simulaciones con semillas desde archivo

```
pseudo_gen\venv\Scripts\python.exe rana_1d.py --archivo semillas_100.txt
pseudo_gen\venv\Scripts\python.exe rana_2d.py --archivo semillas_5.txt
```

Ejecuta una simulación por semilla, imprime el progreso, las estadísticas de las posiciones finales comparadas con los valores teóricos y grafica los histogramas de frecuencia de las posiciones finales.

### Menú interactivo

Sin argumentos, el programa pregunta si se desea usar una semilla o un archivo:

```
pseudo_gen\venv\Scripts\python.exe rana_3d.py
```

### Opciones

| Opción | Descripción | Valor por defecto |
| :--- | :--- | :--- |
| `--semilla N` | Una sola simulación con la semilla `N` | — |
| `--archivo RUTA` | Archivo `.txt` o `.csv` con las semillas | — |
| `--generador {lcg,mult}` | Generador de `pseudo_gen` a utilizar | `lcg` |
| `--alpha A` | Nivel de significancia de las pruebas | `0.05` |
| `--salida CARPETA` | Carpeta donde se guardan gráficas y tablas | `resultados` |
| `--no-mostrar` | Guarda las gráficas sin abrir ventanas | desactivado |

### Formato del archivo de semillas

Números enteros, uno por línea o separados por coma, punto y coma o tabulador. Las líneas vacías y las que empiezan con `#` se ignoran.

```text
# Semillas para la simulación
12345
67890, 2024; 31415
```

Rango válido de las semillas: `0 ≤ x0 < 2^32` para `lcg`, y `1 ≤ x0 < 2^31 − 1` para `mult`.

---

## Flujo de la simulación

```text
semilla ─► generador (pseudo_gen) ─► R_1 … R_1.000.000 ─► 6 pruebas estadísticas
                                                              │
                        ┌─────────── alguna falla ◄───────────┤
                        ▼                                     ▼ todas aprueban
              semilla rechazada                   simulación de la caminata
             (se reporta el motivo)          ─► métricas ─► gráficas y tabla CSV
```

1. **Generación** ([validacion.py](validacion.py)): se crean 1.000.000 de números `R_i ∈ [0, 1)`.
2. **Validación**: se aplican las pruebas de medias, varianza, chi-cuadrado, Kolmogorov-Smirnov, póker y rachas (α = 0.05 por defecto).
3. **Simulación**: cada `R_i` se convierte en un salto y la trayectoria se obtiene como suma acumulada de los saltos.
4. **Análisis**: métricas por simulación, estadísticas del conjunto y gráficas.

### Generadores disponibles

| Clave | Generador | Parámetros | Período |
| :--- | :--- | :--- | :--- |
| `lcg` | Congruencial lineal (Numerical Recipes) | `a = 1664525, c = 1013904223, m = 2^32` | 2^32 ≈ 4,3 × 10⁹ |
| `mult` | Congruencial multiplicativo (MINSTD, Park & Miller) | `a = 16807, m = 2^31 − 1` | 2^31 − 2 ≈ 2,1 × 10⁹ |

En ambos casos el período es mucho mayor que el millón de números que consume cada simulación.

---

## Modelos

Cada salto consume **un** número `R_i`. En todas las dimensiones cada salto mueve **una sola coordenada** una unidad.

### 1D — `rana_1d.py`

| Condición | Salto |
| :--- | :--- |
| `R_i > 0.5` | +1 (adelante) |
| `R_i ≤ 0.5` | −1 (atrás) |

### 2D — `rana_2d.py`

| Condición | Salto |
| :--- | :--- |
| `R_i ≤ 0.25` | Arriba (0, +1) |
| `R_i ≤ 0.50` | Abajo (0, −1) |
| `R_i ≤ 0.75` | Derecha (+1, 0) |
| `R_i > 0.75` | Izquierda (−1, 0) |

### 3D — `rana_3d.py`

| Condición | Salto |
| :--- | :--- |
| `R_i ≤ 1/6` | +X (derecha) |
| `R_i ≤ 2/6` | −X (izquierda) |
| `R_i ≤ 3/6` | +Y (adelante) |
| `R_i ≤ 4/6` | −Y (atrás) |
| `R_i ≤ 5/6` | +Z (arriba) |
| `R_i > 5/6` | −Z (abajo) |

### Valores teóricos (n = 1.000.000)

| Magnitud | 1D | 2D | 3D |
| :--- | :--- | :--- | :--- |
| Media de cada coordenada | 0 | 0 | 0 |
| Varianza de cada coordenada | n | n/2 | n/3 |
| Desviación estándar por coordenada | 1000 | ≈ 707 | ≈ 577 |
| E[D²] (distancia al cuadrado) | n | n | n |
| Distribución de cada coordenada final | N(0, n) | N(0, n/2) | N(0, n/3) |
| Distribución de la distancia final D | \|N(0, n)\| | Rayleigh(σ = √(n/2)) | Maxwell(σ = √(n/3)) |
| E[D] | √(2n/π) ≈ 798 | √(πn)/2 ≈ 886 | 2σ√(2/π) ≈ 921 |

> **Nota.** En 2D y 3D la varianza por coordenada es n/2 y n/3 (no n), porque en cada salto solo una de las coordenadas cambia. En consecuencia E[D²] = n en todas las dimensiones, y la distancia típica al origen es √n = 1000.

---

## Resultados que se generan

Todos los archivos se guardan en la carpeta `resultados/` (o la indicada con `--salida`).

### Modo una semilla

| Dimensión | Archivo | Contenido |
| :--- | :--- | :--- |
| 1D | `trayectoria_1d_semilla_N.png` | Posición vs. iteración, con banda ±√n, inicio y fin |
| 2D | `trayectoria_2d_semilla_N.png` | Trayectoria en el plano, con circunferencia de radio √n |
| 3D | `trayectoria_3d_semilla_N.png` | Trayectoria 3D y proyecciones ortogonales XY, XZ, YZ |

En consola se muestran: resultado de cada prueba, tiempos de generación, pruebas y simulación, posición final, distancia al origen (máxima y promedio), posición en el paso 1000, número de regresos al origen y, en 2D y 3D, la frecuencia de cada dirección.

> En 3D la gráfica dibuja 1 de cada 6 posiciones (unos 167.000 puntos) para que la ventana de Matplotlib sea manejable. La simulación y todas las métricas usan la trayectoria completa.

### Modo archivo

| Dimensión | Gráfica | Tabla |
| :--- | :--- | :--- |
| 1D | `histograma_posiciones_finales_1d.png`: histograma de X_n con la normal teórica | `resultados_1d.csv` |
| 2D | `posiciones_finales_2d.png`: dispersión en el plano, histogramas de X_n, Y_n y de la distancia (Rayleigh) | `resultados_2d.csv` |
| 3D | `posiciones_finales_3d.png`: dispersión 3D, histogramas de X_n, Y_n, Z_n y de la distancia (Maxwell) | `resultados_3d.csv` |

Cada tabla CSV tiene una fila por semilla con: resultado de cada prueba, posición final, posición en el paso 1000, distancias, regresos al origen y tiempo de ejecución. La posición en el paso 1000 permite estimar la probabilidad de retorno al origen tras 1.000 pasos en cada dimensión.

En consola se imprime una tabla que compara las estadísticas observadas (media, desviación estándar, asimetría, curtosis, porcentaje de simulaciones dentro de √n y 2√n) con los valores teóricos.

---

## Semillas rechazadas

Si una secuencia no aprueba alguna de las 6 pruebas, la semilla se descarta, no se simula y se reporta junto con las pruebas que fallaron.

Con α = 0.05 y 6 pruebas, se espera que alrededor del 26 % de las secuencias fallen al menos una prueba aunque el generador sea adecuado (1 − 0,95⁶ ≈ 0,26). En la ejecución con `semillas_100.txt` se rechazaron 20 de 100 semillas. Para obtener al menos 100 simulaciones válidas se recomienda incluir unas 130 semillas en el archivo. El programa avisa cuando hay menos de 100 simulaciones válidas.

---

## Tiempos de ejecución de referencia

Medidos en el equipo de desarrollo (Windows 11, Python 3.14), por semilla:

| Etapa | Tiempo aproximado |
| :--- | :--- |
| Generación de 1.000.000 de números | 0,2 – 0,4 s |
| 6 pruebas estadísticas | 2,7 – 4,5 s (la de póker es la más lenta) |
| Simulación 1D / 2D / 3D | 0,03 s / 0,06 s / 0,06 s |

La ejecución completa con 100 semillas en 1D tardó unos 458 s. El tiempo depende principalmente de la velocidad de un núcleo del procesador, porque las semillas se procesan una tras otra y la validación está escrita en Python puro. La simulación usa operaciones vectorizadas de `numpy` (`cumsum`), por lo que representa menos del 2 % del tiempo total.

---

## Uso de herramientas de inteligencia artificial

De acuerdo con los lineamientos del taller:

- **Módulo `pseudo_gen/`** (generadores y pruebas): desarrollado por el grupo.
- **Programas de simulación** (`validacion.py`, `rana_1d.py`, `rana_2d.py`, `rana_3d.py`) y este README: escritos con apoyo de un asistente de código (Claude, de Anthropic), a partir de los requisitos del grupo, los apuntes de clase y el módulo `pseudo_gen`.
- **Decisiones técnicas del grupo**: número fijo de saltos, uso obligatorio de secuencias validadas, carga de semillas desde archivo, tipo de gráficas por modo y estilo de las gráficas tomado de los ejemplos de clase.
