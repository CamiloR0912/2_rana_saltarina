"""
Punto 2 - La rana estadística: simulación computacional en tres dimensiones.

Modelo (caminata aleatoria simple en Z³, solo movimientos ortogonales, sin diagonales):
    (X_0, Y_0, Z_0) = (0, 0, 0)
    En cada salto se usa un R_i para escoger uno de los 6 vecinos (P = 1/6 cada uno),
    con intervalos acumulados como en ProbabilityMatrix3D vista en clase:
        R_i ≤ 1/6 → +X (Derecha)     R_i ≤ 2/6 → -X (Izquierda)
        R_i ≤ 3/6 → +Y (Adelante)    R_i ≤ 4/6 → -Y (Atrás)
        R_i ≤ 5/6 → +Z (Arriba)      R_i >  5/6 → -Z (Abajo)
    n = 1.000.000 saltos (fijo).

Propiedades teóricas (cada salto mueve solo UNA coordenada, con prob. 1/3 cada una):
    E[X_n] = E[Y_n] = E[Z_n] = 0
    Var(X_n) = Var(Y_n) = Var(Z_n) = n/3    → σ_eje = √(n/3) ≈ 577
    E[X_n² + Y_n² + Z_n²] = n                → distancia típica √n = 1000
    Distancia al origen D ≈ Maxwell(σ = √(n/3)):  E[D] = 2σ√(2/π) ≈ 921

Los R_i provienen del módulo pseudo_gen y solo se usan si la secuencia aprueba
las 6 pruebas estadísticas (ver validacion.py).

Modos de uso:
    python rana_3d.py --semilla 12345              # 1 simulación → trayectoria 3D + proyecciones
    python rana_3d.py --archivo semillas_100.txt   # N simulaciones → histogramas de posiciones finales
    python rana_3d.py                              # menú interactivo

Opciones: --generador {lcg,mult}  --alpha 0.05  --salida resultados  --no-mostrar
"""
import argparse
import csv
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats

from pseudo_gen.utilidades.archivo_manager import ArchivoManager
from rana_1d import N_PASOS, PASO_CONTROL, _finalizar, imprimir_pruebas, menu_interactivo
from validacion import GENERADORES, generar_secuencia_validada

# 6 direcciones ortogonales, en el mismo orden que la matriz de probabilidades de clase
DIRECCIONES = np.array([
    (1, 0, 0),    # +X (Derecha)
    (-1, 0, 0),   # -X (Izquierda)
    (0, 1, 0),    # +Y (Adelante)
    (0, -1, 0),   # -Y (Atrás)
    (0, 0, 1),    # +Z (Arriba)
    (0, 0, -1),   # -Z (Abajo)
], dtype=np.int32)
NOMBRES_DIRECCIONES = ["+X (DERECHA)", "-X (IZQUIERDA)", "+Y (ADELANTE)",
                       "-Y (ATRÁS)", "+Z (ARRIBA)", "-Z (ABAJO)"]
CORTES = np.cumsum([1 / 6] * 5)   # 1/6, 2/6, ..., 5/6 (intervalos acumulados)

SIGMA_EJE = math.sqrt(N_PASOS / 3)          # desviación teórica de cada coordenada
MAX_PUNTOS_GRAFICA = 200_000                # puntos dibujados de la trayectoria (ver graficar_trayectoria)


# ── Modelo ───────────────────────────────────────────────────────────────

def simular_caminata_3d(ri):
    """
    Convierte la secuencia R_i en la trayectoria de la rana en el espacio.

    Args:
        ri (list[float]): N_PASOS números validados en [0, 1).

    Returns:
        tuple:
            - np.ndarray[int32] de forma (n + 1, 3): posiciones (X_k, Y_k, Z_k), iniciando en el origen.
            - np.ndarray[int64] de 6 elementos: cantidad de saltos en cada dirección.
    """
    # searchsorted(side="left") equivale a "el primer intervalo acumulado con R ≤ cum_prob"
    indices = np.searchsorted(CORTES, np.asarray(ri), side="left")
    pasos = DIRECCIONES[indices]                       # (ξ_i, η_i, ζ_i) de cada salto

    trayectoria = np.empty((len(pasos) + 1, 3), dtype=np.int32)
    trayectoria[0] = (0, 0, 0)
    np.cumsum(pasos, axis=0, out=trayectoria[1:])      # posición k = Σ_{i≤k} salto_i
    return trayectoria, np.bincount(indices, minlength=6)


def ejecutar_semilla(semilla, generador, alpha):
    """Genera y valida la secuencia de una semilla y, si es válida, simula la caminata 3D."""
    val = generar_secuencia_validada(generador, semilla, N_PASOS, alpha)
    registro = {
        "semilla": semilla,
        "aprobada": val["aprobada"],
        "resultados": val["resultados"],
        "t_generacion": val["t_generacion"],
        "t_pruebas": val["t_pruebas"],
        "trayectoria": None,
        "conteo_direcciones": None,
        "t_simulacion": 0.0,
    }
    if val["aprobada"]:
        t0 = time.perf_counter()
        registro["trayectoria"], registro["conteo_direcciones"] = simular_caminata_3d(val["ri"])
        registro["t_simulacion"] = time.perf_counter() - t0
    return registro


def resumen_trayectoria(tray):
    """Métricas de una trayectoria individual en 3D."""
    x_f, y_f, z_f = (int(v) for v in tray[-1])
    distancias = np.sqrt(np.sum(tray.astype(np.float64) ** 2, axis=1))
    en_origen = np.all(tray[1:] == 0, axis=1)
    return {
        "x_final": x_f,
        "y_final": y_f,
        "z_final": z_f,
        "distancia_final": round(math.sqrt(x_f ** 2 + y_f ** 2 + z_f ** 2), 3),
        "distancia_maxima": round(float(distancias.max()), 3),
        "distancia_promedio": round(float(distancias.mean()), 3),
        "x_paso_1000": int(tray[PASO_CONTROL, 0]),
        "y_paso_1000": int(tray[PASO_CONTROL, 1]),
        "z_paso_1000": int(tray[PASO_CONTROL, 2]),
        "visitas_origen": int(np.count_nonzero(en_origen)),
    }


# ── Gráficas ─────────────────────────────────────────────────────────────

def graficar_trayectoria(tray, semilla, ruta, mostrar=True):
    """
    Trayectoria 3D (estilo de la gráfica de clase: línea azul, inicio verde, fin rojo)
    más sus tres proyecciones ortogonales (XY, XZ, YZ).

    Dibujar el millón de puntos en 3D hace muy lenta la ventana de Matplotlib, así que se
    grafica 1 de cada `paso` posiciones (máx. MAX_PUNTOS_GRAFICA). La simulación y todas
    las métricas usan la trayectoria completa; el submuestreo es solo visual.
    """
    n = len(tray) - 1
    paso = max(1, math.ceil(len(tray) / MAX_PUNTOS_GRAFICA))
    vis = tray[::paso]
    if not np.array_equal(vis[-1], tray[-1]):
        vis = np.vstack([vis, tray[-1]])
    inicio, fin = tray[0], tray[-1]

    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[2.2, 1])

    # Vista 3D principal
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax.plot(vis[:, 0], vis[:, 1], vis[:, 2], color="tab:blue", linewidth=0.4, alpha=0.7,
            label="Trayectoria")
    ax.scatter(*inicio, color="green", s=80, depthshade=False, label=f"Inicio {tuple(int(v) for v in inicio)}")
    ax.scatter(*fin, color="red", marker="s", s=80, depthshade=False, label=f"Fin {tuple(int(v) for v in fin)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Semilla: {semilla}", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)

    # Proyecciones ortogonales
    for fila, (i, j, nombre) in enumerate([(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]):
        axp = fig.add_subplot(gs[fila, 1])
        axp.plot(vis[:, i], vis[:, j], color="tab:blue", linewidth=0.3, alpha=0.7)
        axp.plot(inicio[i], inicio[j], "go", markersize=7)
        axp.plot(fin[i], fin[j], "rs", markersize=7)
        axp.axhline(0, color="black", linewidth=0.8)
        axp.axvline(0, color="black", linewidth=0.8)
        axp.grid(True, linestyle="--", alpha=0.6)
        axp.set_aspect("equal", adjustable="datalim")
        axp.set_xlabel(nombre[0])
        axp.set_ylabel(nombre[1])
        axp.set_title(f"Proyección {nombre}", fontsize=10)

    nota = f" (se grafica 1 de cada {paso} posiciones)" if paso > 1 else ""
    fig.suptitle(f"Caminata aleatoria 3D - {n:,} saltos".replace(",", ".") + nota,
                 fontsize=14, fontweight="bold")
    _finalizar(fig, ruta, mostrar)


def _histograma_normal(ax, datos, eje):
    """Histograma de una coordenada final con la normal teórica N(0, n/3)."""
    k = len(datos)
    _, bordes, _ = ax.hist(datos, bins="sturges", color="#4A90E2", edgecolor="#1C3F75",
                           alpha=0.8, label="Frecuencia observada")
    ancho = bordes[1] - bordes[0]
    lim = max(4 * SIGMA_EJE, np.abs(datos).max() * 1.1)
    x = np.linspace(-lim, lim, 400)
    ax.plot(x, k * ancho * stats.norm.pdf(x, 0, SIGMA_EJE), "r-", linewidth=2,
            label=f"N(0, n/3), σ = {SIGMA_EJE:.0f}")
    ax.axvline(datos.mean(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Media = {datos.mean():.1f}")
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel(f"Posición final en {eje}")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Histograma de {eje} final")
    ax.legend(fontsize=8)


def graficar_resultados_multiples(finales, ruta, mostrar=True):
    """
    Panel con las posiciones finales de todas las simulaciones:
        - Dispersión 3D de (X_n, Y_n, Z_n).
        - Histograma de la distancia al origen (~ Maxwell).
        - Histogramas de X_n, Y_n y Z_n (cada uno ~ N(0, n/3)).
    """
    finales = np.asarray(finales, dtype=float)
    x, y, z = finales[:, 0], finales[:, 1], finales[:, 2]
    dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    k = len(finales)

    fig = plt.figure(figsize=(17, 11))
    gs = GridSpec(2, 6, figure=fig)

    # 1. Posiciones finales en el espacio
    ax = fig.add_subplot(gs[0, 0:3], projection="3d")
    ax.scatter(x, y, z, s=25, color="#4A90E2", edgecolor="#1C3F75", alpha=0.8, label="Posición final")
    ax.scatter(0, 0, 0, color="green", s=90, depthshade=False, label="Origen")
    ax.scatter(x.mean(), y.mean(), z.mean(), color="red", marker="+", s=150, depthshade=False,
               label=f"Centroide ({x.mean():.0f}, {y.mean():.0f}, {z.mean():.0f})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Posiciones finales en el espacio")
    ax.legend(fontsize=8, loc="upper left")

    # 2. Distancia al origen con la densidad de Maxwell
    ax = fig.add_subplot(gs[0, 3:6])
    _, bordes, _ = ax.hist(dist, bins="sturges", color="#8E44AD", edgecolor="#4A235A",
                           alpha=0.7, label="Frecuencia observada")
    ancho = bordes[1] - bordes[0]
    r = np.linspace(0, max(4 * SIGMA_EJE, dist.max() * 1.1), 400)
    ax.plot(r, k * ancho * stats.maxwell.pdf(r, scale=SIGMA_EJE), "r-", linewidth=2,
            label=f"Maxwell(σ = {SIGMA_EJE:.0f})")
    ax.axvline(dist.mean(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Media = {dist.mean():.1f}")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Distancia final al origen  √(X_n² + Y_n² + Z_n²)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Histograma de distancia final al origen")
    ax.legend(fontsize=8)

    # 3, 4 y 5. Histogramas de cada coordenada
    for col, (datos, eje) in enumerate([(x, "X"), (y, "Y"), (z, "Z")]):
        _histograma_normal(fig.add_subplot(gs[1, 2 * col:2 * col + 2]), datos, eje)

    fig.suptitle(f"Caminata 3D - {k} simulaciones de {N_PASOS:,} saltos".replace(",", "."),
                 fontsize=15, fontweight="bold")
    _finalizar(fig, ruta, mostrar)


# ── Modos de ejecución ───────────────────────────────────────────────────

def modo_una_semilla(semilla, generador, alpha, salida, mostrar):
    print(f"\nSemilla {semilla} | {GENERADORES[generador]['nombre']} | {N_PASOS:,} saltos | α = {alpha}")
    print("Generando y validando la secuencia...")
    reg = ejecutar_semilla(semilla, generador, alpha)

    imprimir_pruebas(reg["resultados"])
    print(f"  Tiempo generación: {reg['t_generacion']:.2f} s | pruebas: {reg['t_pruebas']:.2f} s")

    if not reg["aprobada"]:
        print("\nLa secuencia NO aprobó todas las pruebas: no se usa en la simulación. Pruebe otra semilla.")
        return 1

    m = resumen_trayectoria(reg["trayectoria"])
    e_d = 2 * SIGMA_EJE * math.sqrt(2 / math.pi)
    print(f"\nSecuencia validada. Simulación completada en {reg['t_simulacion']:.3f} s")
    print(f"  Posición final:             ({m['x_final']}, {m['y_final']}, {m['z_final']})")
    print(f"  Distancia al origen:        {m['distancia_final']:.2f}  "
          f"(√n = {math.sqrt(N_PASOS):.0f}; E[D] ≈ {e_d:.0f})")
    print(f"  Distancia máxima alcanzada: {m['distancia_maxima']:.2f}")
    print(f"  Distancia promedio:         {m['distancia_promedio']:.2f}")
    print(f"  Posición en el paso {PASO_CONTROL}:   "
          f"({m['x_paso_1000']}, {m['y_paso_1000']}, {m['z_paso_1000']})")
    print(f"  Regresos al origen:         {m['visitas_origen']}")
    print("\n=== ESTADÍSTICAS DE DIRECCIONES ===")
    for nombre, c in zip(NOMBRES_DIRECCIONES, reg["conteo_direcciones"]):
        print(f"  {nombre:<16}: {c:>8} veces ({c / N_PASOS:.4f}, esperado: {1 / 6:.4f})")

    graficar_trayectoria(reg["trayectoria"], semilla,
                         os.path.join(salida, f"trayectoria_3d_semilla_{semilla}.png"), mostrar)
    return 0


def modo_archivo(ruta, generador, alpha, salida, mostrar):
    semillas = ArchivoManager.leer_semillas(ruta)
    if not semillas:
        print(f"No se encontraron semillas en {ruta}")
        return 1
    if len(set(semillas)) != len(semillas):
        print("ADVERTENCIA: el archivo contiene semillas repetidas; esas simulaciones no son independientes.")

    print(f"\n{len(semillas)} semillas leídas de {ruta} | {GENERADORES[generador]['nombre']} | "
          f"{N_PASOS:,} saltos | α = {alpha}\n")

    filas, finales, rechazadas = [], [], []
    t_inicio = time.perf_counter()
    for i, semilla in enumerate(semillas, start=1):
        try:
            reg = ejecutar_semilla(semilla, generador, alpha)
        except ValueError as e:   # semilla fuera del rango válido del generador
            print(f"[{i:>3}/{len(semillas)}] Semilla {semilla}: inválida ({e})")
            rechazadas.append((semilla, "semilla inválida"))
            continue

        fila = {"semilla": semilla, "aprobada": reg["aprobada"]}
        fila.update({f"prueba_{n}": bool(r.aprobada) for n, r in reg["resultados"]})

        if reg["aprobada"]:
            m = resumen_trayectoria(reg["trayectoria"])
            fila.update(m)
            finales.append((m["x_final"], m["y_final"], m["z_final"]))
            estado = (f"({m['x_final']:>6}, {m['y_final']:>6}, {m['z_final']:>6})  "
                      f"D = {m['distancia_final']:8.1f}")
        else:
            fallidas = [n for n, r in reg["resultados"] if not r.aprobada]
            rechazadas.append((semilla, ", ".join(fallidas)))
            estado = f"RECHAZADA ({', '.join(fallidas)})"

        fila["tiempo_s"] = round(reg["t_generacion"] + reg["t_pruebas"] + reg["t_simulacion"], 3)
        filas.append(fila)
        print(f"[{i:>3}/{len(semillas)}] Semilla {semilla:>12}: {estado}   ({fila['tiempo_s']:.1f} s)")

    t_total = time.perf_counter() - t_inicio

    ruta_csv = os.path.join(salida, "resultados_3d.csv")
    columnas = list(dict.fromkeys(k for f in filas for k in f))
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)

    print(f"\nTiempo total: {t_total:.1f} s | Tabla por semilla guardada en: {ruta_csv}")
    print(f"Simulaciones válidas: {len(finales)} de {len(semillas)}")
    if rechazadas:
        print("Semillas rechazadas (no se usaron en la simulación):")
        for s, motivo in rechazadas:
            print(f"  - {s}: {motivo}")
    if len(finales) < 100:
        print("ADVERTENCIA: hay menos de 100 simulaciones válidas. Agregue más semillas al archivo.")
    if len(finales) < 2:
        return 1

    p = np.asarray(finales, dtype=float)
    d = np.sqrt(np.sum(p ** 2, axis=1))
    r1 = math.sqrt(N_PASOS)
    print("\nEstadísticas de las posiciones finales")
    print(f"  {'':<30}{'Observado':>14}{'Teórico':>12}")
    for col, eje in enumerate("XYZ"):
        c = p[:, col]
        print(f"  {f'Media / Desv. estándar {eje}':<30}{f'{c.mean():.1f} / {c.std(ddof=1):.1f}':>14}"
              f"{f'0 / {SIGMA_EJE:.1f}':>12}")
    print(f"  {'Media de D² = X² + Y² + Z²':<30}{np.mean(d ** 2):>14.0f}{N_PASOS:>12}")
    print(f"  {'Distancia media E[D]':<30}{d.mean():>14.2f}{2 * SIGMA_EJE * math.sqrt(2 / math.pi):>12.2f}")
    print(f"  {'% con D ≤ √n':<30}{100 * np.mean(d <= r1):>13.1f}%"
          f"{100 * stats.maxwell.cdf(r1, scale=SIGMA_EJE):>11.1f}%")
    print(f"  {'% con D ≤ 2√n':<30}{100 * np.mean(d <= 2 * r1):>13.1f}%"
          f"{100 * stats.maxwell.cdf(2 * r1, scale=SIGMA_EJE):>11.1f}%")

    graficar_resultados_multiples(finales, os.path.join(salida, "posiciones_finales_3d.png"), mostrar)
    return 0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Simulación 3D de la rana estadística (1.000.000 de saltos).")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--semilla", type=int, help="Una sola simulación con esta semilla")
    grupo.add_argument("--archivo", help="Archivo .txt/.csv con las semillas (una simulación por semilla)")
    parser.add_argument("--generador", choices=GENERADORES, default="lcg",
                        help="Generador de pseudo_gen a usar (por defecto: lcg)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Nivel de significancia de las pruebas")
    parser.add_argument("--salida", default="resultados", help="Carpeta para gráficas y tablas")
    parser.add_argument("--no-mostrar", action="store_true", help="Solo guardar las gráficas, sin abrir ventanas")
    args = parser.parse_args()

    semilla, archivo = args.semilla, args.archivo
    if semilla is None and archivo is None:
        eleccion = menu_interactivo("3D")
        semilla, archivo = eleccion["semilla"], eleccion["archivo"]

    os.makedirs(args.salida, exist_ok=True)
    mostrar = not args.no_mostrar

    try:
        if archivo:
            return modo_archivo(archivo, args.generador, args.alpha, args.salida, mostrar)
        return modo_una_semilla(semilla, args.generador, args.alpha, args.salida, mostrar)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
