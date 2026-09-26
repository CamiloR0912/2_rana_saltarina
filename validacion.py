"""
Obtención de secuencias pseudoaleatorias validadas para las simulaciones del punto 2.

Flujo:
    semilla → generador (pseudo_gen) → R_1..R_n → 6 pruebas estadísticas
            → si TODAS se aprueban, la secuencia se entrega a la simulación;
              si alguna falla, la semilla se rechaza.

Se reutiliza en las caminatas 1D, 2D y 3D.
"""
import time

import pseudo_gen as pg

# Parámetros de los generadores (ver pseudo_gen/README.md, "Parámetros recomendados").
# El período (~2^32 y ~2^31) es mucho mayor que el millón de números que usa cada caminata.
GENERADORES = {
    "lcg": {
        "nombre": "Congruencial Lineal (Numerical Recipes)",
        "crear": lambda s: pg.CongruencialLineal(x0=s, a=1664525, c=1013904223, m=2**32),
    },
    "mult": {
        "nombre": "Congruencial Multiplicativo (MINSTD, Park & Miller)",
        "crear": lambda s: pg.CongruencialMultiplicativo(x0=s, a=16807, m=2**31 - 1),
    },
}

# Las 6 pruebas del módulo, en el orden en que se ejecutan
PRUEBAS = [
    ("Medias", pg.PruebaMedias),
    ("Varianza", pg.PruebaVarianza),
    ("Chi-cuadrado", pg.PruebaChi2),
    ("Kolmogorov-Smirnov", pg.PruebaKS),
    ("Póker", pg.PruebaPoker),
    ("Rachas", pg.PruebaRachas),
]


def generar_secuencia_validada(tipo_generador, semilla, cantidad, alpha=0.05):
    """
    Genera `cantidad` números R_i con el generador indicado y los somete a las 6 pruebas.

    Args:
        tipo_generador (str): Clave de GENERADORES ("lcg" o "mult").
        semilla (int): Semilla x0 del generador.
        cantidad (int): Cantidad de números R_i a generar.
        alpha (float): Nivel de significancia de las pruebas.

    Returns:
        dict con:
            - "ri": lista de R_i (None si la secuencia fue rechazada, para liberar memoria).
            - "aprobada": True si pasó las 6 pruebas.
            - "resultados": lista de (nombre_prueba, ResultadoPrueba).
            - "t_generacion", "t_pruebas": tiempos en segundos.
    """
    gen = GENERADORES[tipo_generador]["crear"](semilla)

    t0 = time.perf_counter()
    _, ri = gen.generar(cantidad)
    t_generacion = time.perf_counter() - t0

    t0 = time.perf_counter()
    resultados = [(nombre, Prueba(ri, alpha=alpha).ejecutar()) for nombre, Prueba in PRUEBAS]
    t_pruebas = time.perf_counter() - t0

    aprobada = all(bool(res.aprobada) for _, res in resultados)
    return {
        "ri": ri if aprobada else None,
        "aprobada": aprobada,
        "resultados": resultados,
        "t_generacion": t_generacion,
        "t_pruebas": t_pruebas,
    }
