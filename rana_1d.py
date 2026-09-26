"""
Punto 2 - La rana estadística: simulación computacional unidimensional.

Modelo (caminata aleatoria simple simétrica en Z):
    X_0 = 0
    ξ_i = +1 si R_i > 0.5  (salto hacia adelante), -1 en otro caso (salto hacia atrás)
    X_n = Σ ξ_i,           n = 1.000.000 (fijo)

Propiedades teóricas: E[X_n] = 0,  Var(X_n) = n  →  σ = √n = 1000.

Los R_i provienen del módulo pseudo_gen y solo se usan si la secuencia aprueba
las 6 pruebas estadísticas (medias, varianza, χ², KS, póker, rachas).

Modos de uso:
    python rana_1d.py --semilla 12345                  # 1 simulación → gráfica de la trayectoria
    python rana_1d.py --archivo semillas_100.txt       # N simulaciones → histograma de posiciones finales
    python rana_1d.py                                  # menú interactivo

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
from validacion import GENERADORES, generar_secuencia_validada

N_PASOS = 1_000_000   # Siempre un millón de saltos
PASO_CONTROL = 1000   # Se registra la posición en el paso 1000 (retorno al origen, sección 2D/3D)


# ── Modelo ───────────────────────────────────────────────────────────────

def simular_caminata_1d(ri):
    """
    Convierte la secuencia R_i en la trayectoria de la rana.

    Args:
        ri (list[float]): N_PASOS números validados en [0, 1).

    Returns:
        np.ndarray[int32]: posiciones X_0..X_n (len = n + 1), con X_0 = 0.
    """
    ri = np.asarray(ri)
    pasos = np.where(ri > 0.5, 1, -1).astype(np.int32)   # ξ_i ∈ {+1, -1}, P = 1/2 cada uno

    trayectoria = np.empty(len(pasos) + 1, dtype=np.int32)
    trayectoria[0] = 0
    np.cumsum(pasos, out=trayectoria[1:])                  # X_k = Σ_{i≤k} ξ_i
    return trayectoria


def ejecutar_semilla(semilla, generador, alpha):
    """Genera y valida la secuencia de una semilla y, si es válida, simula la caminata."""
    val = generar_secuencia_validada(generador, semilla, N_PASOS, alpha)
    registro = {
        "semilla": semilla,
        "aprobada": val["aprobada"],
        "resultados": val["resultados"],
        "t_generacion": val["t_generacion"],
        "t_pruebas": val["t_pruebas"],
        "trayectoria": None,
        "t_simulacion": 0.0,
    }
    if val["aprobada"]:
        t0 = time.perf_counter()
        registro["trayectoria"] = simular_caminata_1d(val["ri"])
        registro["t_simulacion"] = time.perf_counter() - t0
    return registro


# ── Consola ──────────────────────────────────────────────────────────────

def imprimir_pruebas(resultados):
    print(f"  {'Prueba':<20}{'Estadístico':>16}{'Referencia':>16}   Resultado")
    for nombre, res in resultados:
        estado = "APROBADA" if res.aprobada else "RECHAZADA"
        print(f"  {nombre:<20}{res.estadistico:>16.6f}{res.valor_critico:>16.6f}   {estado}")


def resumen_trayectoria(tray):
    """Métricas de una trayectoria individual."""
    return {
        "posicion_final": int(tray[-1]),
        "posicion_paso_1000": int(tray[PASO_CONTROL]),
        "maximo": int(tray.max()),
        "minimo": int(tray.min()),
        "saltos_adelante": int((N_PASOS + tray[-1]) // 2),
        "saltos_atras": int((N_PASOS - tray[-1]) // 2),
        "visitas_origen": int(np.count_nonzero(tray[1:] == 0)),
    }


# ── Gráficas ─────────────────────────────────────────────────────────────

def _finalizar(fig, ruta, mostrar):
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    print(f"  Gráfica guardada en: {ruta}")
    if mostrar:
        plt.show()
    else:
        plt.close(fig)


def graficar_trayectoria(tray, semilla, ruta, mostrar=True):
    """Posición de la rana vs. iteración (estilo de la gráfica vista en clase)."""
    n = len(tray) - 1
    raiz_n = math.sqrt(n)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(n + 1), tray, color="g", linewidth=0.6, label="Trayectoria")

    # Banda de ±√n (una desviación estándar teórica de X_n)
    ax.axhline(raiz_n, color="gray", linestyle=":", linewidth=1.2, label=f"±√n = ±{raiz_n:.0f}")
    ax.axhline(-raiz_n, color="gray", linestyle=":", linewidth=1.2)

    ax.plot(0, tray[0], "go", markersize=9, label="Inicio (0)")
    ax.plot(n, tray[-1], "rs", markersize=9, label=f"Fin ({tray[-1]})")

    ax.grid(True, linestyle="--", alpha=0.7)
    ax.axhline(0, color="black", linewidth=1)   # Eje X en el origen
    ax.axvline(0, color="black", linewidth=1)   # Eje Y en el origen
    ax.set_xlabel("Iteración (salto)")
    ax.set_ylabel("Posición")
    ax.set_title(f"Gráfica de movimientos de la rana - Semilla {semilla} ({n:,} saltos)".replace(",", "."))
    ax.ticklabel_format(axis="x", style="plain")
    ax.legend(loc="best")
    _finalizar(fig, ruta, mostrar)


def graficar_histograma(finales, ruta, mostrar=True):
    """Histograma de frecuencias de las posiciones finales con la normal teórica N(0, n)."""
    finales = np.asarray(finales)
    k = len(finales)
    sigma = math.sqrt(N_PASOS)

    fig, ax = plt.subplots(figsize=(10, 6))
    frec, bordes, _ = ax.hist(finales, bins="sturges", color="#4A90E2",
                              edgecolor="#1C3F75", alpha=0.8, label="Frecuencia observada")

    # Curva normal teórica escalada a frecuencias: k · ancho_clase · f(x)
    ancho = bordes[1] - bordes[0]
    lim = max(4 * sigma, np.abs(finales).max() * 1.1)
    x = np.linspace(-lim, lim, 400)
    ax.plot(x, k * ancho * stats.norm.pdf(x, 0, sigma), "r-", linewidth=2,
            label=f"N(0, n) teórica, σ = √n = {sigma:.0f}")

    media = finales.mean()
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(media, color="orange", linestyle="--", linewidth=1.5,
               label=f"Media observada = {media:.1f}")

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Posición final de la rana (X_n)")
    ax.set_ylabel("Frecuencia")
    ax.set_title(f"Histograma de posiciones finales - {k} simulaciones de {N_PASOS:,} saltos".replace(",", "."))
    ax.legend(loc="best")
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

    tray = reg["trayectoria"]
    m = resumen_trayectoria(tray)
    print(f"\nSecuencia validada. Simulación completada en {reg['t_simulacion']:.3f} s")
    print(f"  Posición final (X_n):       {m['posicion_final']}")
    print(f"  Distancia al origen:        {abs(m['posicion_final'])}  (σ = √n = {math.sqrt(N_PASOS):.0f}; E|X_n| ≈ √(2n/π) = {math.sqrt(2 * N_PASOS / math.pi):.0f})")
    print(f"  Saltos adelante / atrás:    {m['saltos_adelante']} / {m['saltos_atras']}")
    print(f"  Posición máxima / mínima:   {m['maximo']} / {m['minimo']}")
    print(f"  Posición en el paso {PASO_CONTROL}:   {m['posicion_paso_1000']}")
    print(f"  Regresos al origen:         {m['visitas_origen']}")

    graficar_trayectoria(tray, semilla, os.path.join(salida, f"trayectoria_1d_semilla_{semilla}.png"), mostrar)
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
            finales.append(m["posicion_final"])
            estado = f"X_n = {m['posicion_final']:>6}"
        else:
            fallidas = [n for n, r in reg["resultados"] if not r.aprobada]
            rechazadas.append((semilla, ", ".join(fallidas)))
            estado = f"RECHAZADA ({', '.join(fallidas)})"

        fila["tiempo_s"] = round(reg["t_generacion"] + reg["t_pruebas"] + reg["t_simulacion"], 3)
        filas.append(fila)
        print(f"[{i:>3}/{len(semillas)}] Semilla {semilla:>12}: {estado}   ({fila['tiempo_s']:.1f} s)")

    t_total = time.perf_counter() - t_inicio

    # Tabla de resultados por semilla
    ruta_csv = os.path.join(salida, "resultados_1d.csv")
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

    x = np.asarray(finales, dtype=float)
    sigma = math.sqrt(N_PASOS)
    print("\nEstadísticas de las posiciones finales")
    print(f"  {'':<28}{'Observado':>12}{'Teórico':>12}")
    print(f"  {'Media':<28}{x.mean():>12.2f}{0:>12}")
    print(f"  {'Desviación estándar':<28}{x.std(ddof=1):>12.2f}{sigma:>12.0f}")
    print(f"  {'Asimetría':<28}{stats.skew(x):>12.3f}{0:>12}")
    print(f"  {'Curtosis (exceso)':<28}{stats.kurtosis(x):>12.3f}{0:>12}")
    print(f"  {'% dentro de ±√n':<28}{100 * np.mean(np.abs(x) <= sigma):>11.1f}%{68.3:>11.1f}%")
    print(f"  {'% dentro de ±2√n':<28}{100 * np.mean(np.abs(x) <= 2 * sigma):>11.1f}%{95.4:>11.1f}%")
    print(f"  {'Mínimo / Máximo':<28}{f'{x.min():.0f} / {x.max():.0f}':>12}")

    graficar_histograma(finales, os.path.join(salida, "histograma_posiciones_finales_1d.png"), mostrar)
    return 0


def menu_interactivo(dimension="1D"):
    print(f"=== La rana estadística - Caminata aleatoria {dimension} ===")
    print("1) Una simulación con una semilla")
    print("2) Varias simulaciones con semillas desde archivo (.txt / .csv)")
    opcion = input("Opción: ").strip()
    if opcion == "1":
        return {"semilla": int(input("Semilla: ").strip()), "archivo": None}
    if opcion == "2":
        return {"semilla": None, "archivo": input("Ruta del archivo de semillas: ").strip().strip('"')}
    print("Opción no válida")
    sys.exit(1)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Simulación 1D de la rana estadística (1.000.000 de saltos).")
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
        eleccion = menu_interactivo()
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
