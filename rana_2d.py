"""
Punto 2 - La rana estadística: simulación computacional en dos dimensiones.

Modelo (caminata aleatoria simple en Z²):
    (X_0, Y_0) = (0, 0)
    En cada salto se usa un R_i para escoger uno de los 4 vecinos (P = 1/4 cada uno),
    con los mismos cortes vistos en clase:
        R_i ≤ 0.25 → Arriba    ( 0, +1)
        R_i ≤ 0.50 → Abajo     ( 0, -1)
        R_i ≤ 0.75 → Derecha   (+1,  0)
        R_i >  0.75 → Izquierda (-1,  0)
    n = 1.000.000 saltos (fijo).

Propiedades teóricas (cada salto mueve solo UNA coordenada, con prob. 1/2 cada una):
    E[X_n] = E[Y_n] = 0
    Var(X_n) = Var(Y_n) = n/2          → σ_x = σ_y = √(n/2) ≈ 707
    E[X_n² + Y_n²] = n                 → distancia típica √n = 1000
    Distancia al origen D ≈ Rayleigh(σ = √(n/2)):  E[D] = √(πn)/2 ≈ 886

Los R_i provienen del módulo pseudo_gen y solo se usan si la secuencia aprueba
las 6 pruebas estadísticas (ver validacion.py).

Modos de uso:
    python rana_2d.py --semilla 12345              # 1 simulación → trayectoria en el plano
    python rana_2d.py --archivo semillas_100.txt   # N simulaciones → histogramas de posiciones finales
    python rana_2d.py                              # menú interactivo

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
from scipy import stats

from pseudo_gen.utilidades.archivo_manager import ArchivoManager
from rana_1d import N_PASOS, PASO_CONTROL, _finalizar, imprimir_pruebas, menu_interactivo
from validacion import GENERADORES, generar_secuencia_validada

# Direcciones en el mismo orden que los cortes 0.25 / 0.50 / 0.75
DIRECCIONES = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)], dtype=np.int32)
NOMBRES_DIRECCIONES = ["Arriba (+Y)", "Abajo (-Y)", "Derecha (+X)", "Izquierda (-X)"]
CORTES = np.array([0.25, 0.50, 0.75])

SIGMA_EJE = math.sqrt(N_PASOS / 2)   # desviación teórica de X_n y de Y_n


# ── Modelo ───────────────────────────────────────────────────────────────

def simular_caminata_2d(ri):
    """
    Convierte la secuencia R_i en la trayectoria de la rana en el plano.

    Args:
        ri (list[float]): N_PASOS números validados en [0, 1).

    Returns:
        tuple:
            - np.ndarray[int32] de forma (n + 1, 2): posiciones (X_k, Y_k), con (X_0, Y_0) = (0, 0).
            - np.ndarray[int64] de 4 elementos: cantidad de saltos en cada dirección.
    """
    # searchsorted(side="left") reproduce los cortes de clase: R ≤ 0.25 → 0, R ≤ 0.50 → 1, ...
    indices = np.searchsorted(CORTES, np.asarray(ri), side="left")
    pasos = DIRECCIONES[indices]                       # (ξ_i, η_i) de cada salto

    trayectoria = np.empty((len(pasos) + 1, 2), dtype=np.int32)
    trayectoria[0] = (0, 0)
    np.cumsum(pasos, axis=0, out=trayectoria[1:])      # (X_k, Y_k) = Σ_{i≤k} (ξ_i, η_i)
    return trayectoria, np.bincount(indices, minlength=4)


def ejecutar_semilla(semilla, generador, alpha):
    """Genera y valida la secuencia de una semilla y, si es válida, simula la caminata 2D."""
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
        registro["trayectoria"], registro["conteo_direcciones"] = simular_caminata_2d(val["ri"])
        registro["t_simulacion"] = time.perf_counter() - t0
    return registro


def resumen_trayectoria(tray):
    """Métricas de una trayectoria individual en 2D."""
    x_f, y_f = int(tray[-1, 0]), int(tray[-1, 1])
    distancias = np.hypot(tray[:, 0], tray[:, 1])
    en_origen = (tray[1:, 0] == 0) & (tray[1:, 1] == 0)
    return {
        "x_final": x_f,
        "y_final": y_f,
        "distancia_final": round(math.hypot(x_f, y_f), 3),
        "distancia_maxima": round(float(distancias.max()), 3),
        "distancia_promedio": round(float(distancias.mean()), 3),
        "x_paso_1000": int(tray[PASO_CONTROL, 0]),
        "y_paso_1000": int(tray[PASO_CONTROL, 1]),
        "visitas_origen": int(np.count_nonzero(en_origen)),
    }


# ── Gráficas ─────────────────────────────────────────────────────────────

def graficar_trayectoria(tray, semilla, ruta, mostrar=True):
    """Trayectoria de la rana en el plano (estilo de plot_random_walk visto en clase)."""
    n = len(tray) - 1
    xf, yf = tray[:, 0], tray[:, 1]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(xf, yf, "b-", linewidth=0.4, alpha=0.7, label="Trayectoria")

    # Circunferencia de radio √n (distancia cuadrática media teórica)
    t = np.linspace(0, 2 * np.pi, 300)
    r = math.sqrt(n)
    ax.plot(r * np.cos(t), r * np.sin(t), color="gray", linestyle=":", linewidth=1.5,
            label=f"Radio √n = {r:.0f}")

    ax.plot(xf[0], yf[0], "go", markersize=12, label=f"Inicio ({xf[0]}, {yf[0]})")
    ax.plot(xf[-1], yf[-1], "rs", markersize=12, label=f"Fin ({xf[-1]}, {yf[-1]})")

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=1, alpha=0.8)
    ax.axvline(0, color="black", linewidth=1, alpha=0.8)
    ax.set_xlabel("Posición en X", fontsize=12)
    ax.set_ylabel("Posición en Y", fontsize=12)
    ax.set_title(f"Caminata aleatoria 2D - Semilla {semilla} ({n:,} saltos)".replace(",", "."),
                 fontsize=14, fontweight="bold")
    ax.set_aspect("equal", adjustable="datalim")   # misma escala en X y Y para no deformar el plano
    ax.legend(fontsize=10, loc="best")
    _finalizar(fig, ruta, mostrar)


def _histograma_normal(ax, datos, eje):
    """Histograma de una coordenada final con la normal teórica N(0, n/2)."""
    k = len(datos)
    frec, bordes, _ = ax.hist(datos, bins="sturges", color="#4A90E2", edgecolor="#1C3F75",
                              alpha=0.8, label="Frecuencia observada")
    ancho = bordes[1] - bordes[0]
    lim = max(4 * SIGMA_EJE, np.abs(datos).max() * 1.1)
    x = np.linspace(-lim, lim, 400)
    ax.plot(x, k * ancho * stats.norm.pdf(x, 0, SIGMA_EJE), "r-", linewidth=2,
            label=f"N(0, n/2), σ = {SIGMA_EJE:.0f}")
    ax.axvline(datos.mean(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Media = {datos.mean():.1f}")
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel(f"Posición final en {eje}")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Histograma de {eje} final")
    ax.legend(fontsize=8)


def graficar_resultados_multiples(x_fin, y_fin, ruta, mostrar=True):
    """
    Panel 2x2 con las posiciones finales de todas las simulaciones:
        - Dispersión (X_n, Y_n) en el plano.
        - Histograma de X_n y de Y_n (cada uno ~ N(0, n/2)).
        - Histograma de la distancia al origen (~ Rayleigh).
    """
    x_fin, y_fin = np.asarray(x_fin), np.asarray(y_fin)
    dist = np.hypot(x_fin, y_fin)
    k = len(x_fin)

    fig, ejes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Posiciones finales en el plano
    ax = ejes[0, 0]
    ax.scatter(x_fin, y_fin, s=25, color="#4A90E2", edgecolor="#1C3F75", alpha=0.8,
               label="Posición final")
    t = np.linspace(0, 2 * np.pi, 300)
    for mult, estilo in ((1, ":"), (2, "--")):
        r = mult * math.sqrt(N_PASOS)
        ax.plot(r * np.cos(t), r * np.sin(t), color="gray", linestyle=estilo, linewidth=1.2,
                label=f"Radio {'' if mult == 1 else mult}√n = {r:.0f}")
    ax.plot(0, 0, "go", markersize=10, label="Origen")
    ax.plot(x_fin.mean(), y_fin.mean(), "r+", markersize=14, mew=2,
            label=f"Centroide ({x_fin.mean():.0f}, {y_fin.mean():.0f})")
    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Posición final en X")
    ax.set_ylabel("Posición final en Y")
    ax.set_title("Posiciones finales en el plano")
    ax.legend(fontsize=8, loc="upper right")

    # 2 y 3. Histogramas de cada coordenada
    _histograma_normal(ejes[0, 1], x_fin, "X")
    _histograma_normal(ejes[1, 0], y_fin, "Y")

    # 4. Distancia al origen con la densidad de Rayleigh
    ax = ejes[1, 1]
    frec, bordes, _ = ax.hist(dist, bins="sturges", color="#8E44AD", edgecolor="#4A235A",
                              alpha=0.7, label="Frecuencia observada")
    ancho = bordes[1] - bordes[0]
    r = np.linspace(0, max(4 * SIGMA_EJE, dist.max() * 1.1), 400)
    ax.plot(r, k * ancho * stats.rayleigh.pdf(r, scale=SIGMA_EJE), "r-", linewidth=2,
            label=f"Rayleigh(σ = {SIGMA_EJE:.0f})")
    ax.axvline(dist.mean(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Media = {dist.mean():.1f}")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Distancia final al origen  √(X_n² + Y_n²)")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Histograma de distancia final al origen")
    ax.legend(fontsize=8)

    fig.suptitle(f"Caminata 2D - {k} simulaciones de {N_PASOS:,} saltos".replace(",", "."),
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
    print(f"\nSecuencia validada. Simulación completada en {reg['t_simulacion']:.3f} s")
    print(f"  Posición final:             ({m['x_final']}, {m['y_final']})")
    print(f"  Distancia al origen:        {m['distancia_final']:.2f}  "
          f"(√n = {math.sqrt(N_PASOS):.0f}; E[D] ≈ √(πn)/2 = {math.sqrt(math.pi * N_PASOS) / 2:.0f})")
    print(f"  Distancia máxima alcanzada: {m['distancia_maxima']:.2f}")
    print(f"  Distancia promedio:         {m['distancia_promedio']:.2f}")
    print(f"  Posición en el paso {PASO_CONTROL}:   ({m['x_paso_1000']}, {m['y_paso_1000']})")
    print(f"  Regresos al origen:         {m['visitas_origen']}")
    print("  Saltos por dirección (esperado: 0.25 cada una):")
    for nombre, c in zip(NOMBRES_DIRECCIONES, reg["conteo_direcciones"]):
        print(f"    {nombre:<16}{c:>9}  ({c / N_PASOS:.4f})")

    graficar_trayectoria(reg["trayectoria"], semilla,
                         os.path.join(salida, f"trayectoria_2d_semilla_{semilla}.png"), mostrar)
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

    filas, x_fin, y_fin, rechazadas = [], [], [], []
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
            x_fin.append(m["x_final"])
            y_fin.append(m["y_final"])
            estado = f"({m['x_final']:>6}, {m['y_final']:>6})  D = {m['distancia_final']:8.1f}"
        else:
            fallidas = [n for n, r in reg["resultados"] if not r.aprobada]
            rechazadas.append((semilla, ", ".join(fallidas)))
            estado = f"RECHAZADA ({', '.join(fallidas)})"

        fila["tiempo_s"] = round(reg["t_generacion"] + reg["t_pruebas"] + reg["t_simulacion"], 3)
        filas.append(fila)
        print(f"[{i:>3}/{len(semillas)}] Semilla {semilla:>12}: {estado}   ({fila['tiempo_s']:.1f} s)")

    t_total = time.perf_counter() - t_inicio

    ruta_csv = os.path.join(salida, "resultados_2d.csv")
    columnas = list(dict.fromkeys(k for f in filas for k in f))
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)

    print(f"\nTiempo total: {t_total:.1f} s | Tabla por semilla guardada en: {ruta_csv}")
    print(f"Simulaciones válidas: {len(x_fin)} de {len(semillas)}")
    if rechazadas:
        print("Semillas rechazadas (no se usaron en la simulación):")
        for s, motivo in rechazadas:
            print(f"  - {s}: {motivo}")
    if len(x_fin) < 100:
        print("ADVERTENCIA: hay menos de 100 simulaciones válidas. Agregue más semillas al archivo.")
    if len(x_fin) < 2:
        return 1

    x, y = np.asarray(x_fin, dtype=float), np.asarray(y_fin, dtype=float)
    d = np.hypot(x, y)
    r1 = math.sqrt(N_PASOS)
    print("\nEstadísticas de las posiciones finales")
    print(f"  {'':<30}{'Observado':>12}{'Teórico':>12}")
    print(f"  {'Media X / Media Y':<30}{f'{x.mean():.1f} / {y.mean():.1f}':>12}{'0 / 0':>12}")
    print(f"  {'Desv. estándar X':<30}{x.std(ddof=1):>12.2f}{SIGMA_EJE:>12.2f}")
    print(f"  {'Desv. estándar Y':<30}{y.std(ddof=1):>12.2f}{SIGMA_EJE:>12.2f}")
    print(f"  {'Asimetría X / Y':<30}{f'{stats.skew(x):.3f} / {stats.skew(y):.3f}':>12}{'0 / 0':>12}")
    print(f"  {'Correlación X-Y':<30}{np.corrcoef(x, y)[0, 1]:>12.3f}{0:>12}")
    print(f"  {'Media de D² = X² + Y²':<30}{np.mean(d ** 2):>12.0f}{N_PASOS:>12}")
    print(f"  {'Distancia media E[D]':<30}{d.mean():>12.2f}{math.sqrt(math.pi * N_PASOS) / 2:>12.2f}")
    print(f"  {'% con D ≤ √n':<30}{100 * np.mean(d <= r1):>11.1f}%{100 * (1 - math.exp(-1)):>11.1f}%")
    print(f"  {'% con D ≤ 2√n':<30}{100 * np.mean(d <= 2 * r1):>11.1f}%{100 * (1 - math.exp(-4)):>11.1f}%")

    graficar_resultados_multiples(x_fin, y_fin, os.path.join(salida, "posiciones_finales_2d.png"), mostrar)
    return 0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Simulación 2D de la rana estadística (1.000.000 de saltos).")
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
        eleccion = menu_interactivo("2D")
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
