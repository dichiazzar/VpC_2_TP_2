"""Video demo de la presentación (máximo 2 minutos): dos partidas reales con placas, leyendas y tramos acelerados.

Dos pasos:
  1. Reprocesa las partidas crudas del registro (crudo_9 y crudo_12) con el sistema final, igual que
     verificar_partidas.py (detector A con los pesos por defecto, seguimiento, reglas y gestos), y guarda los videos
     anotados en videos/demo_trabajo/. Se hace una sola vez (con --rehacer se repite).
  2. Arma el video: placa de título, un cuadro fijo que explica cómo se lee la mesa, la partida 9 completa
     (pide apuntando, se planta con la palma, gana 20 a 17), los momentos clave de la partida 12 (un 5 girado,
     21, gana 21 a 20) y una placa con los resultados. Los momentos clave van a velocidad real y los tiempos
     muertos, acelerados. Las leyendas van en la franja negra derecha, debajo del panel del juego.
El video no tiene audio: se narra en vivo durante la presentación. El FPS del panel es el de la grabación en vivo.

Uso (desde la carpeta del repo):
    python Experimentos/video_demo.py                  # -> videos/demo.mp4 (H.264, 1280x720, ~83 s)
    python Experimentos/video_demo.py --sin-leyendas   # la misma versión sin leyendas
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "Identificador_Cartas"))
sys.path.insert(0, str(REPO / "Gestos"))
from video import GrabadorVideo  # noqa: E402

PARTIDAS = REPO / "Experimentos" / "partidas_reales"
TRABAJO = REPO / "videos" / "demo_trabajo"
W, H, FPS = 1280, 720, 15
VERDE, CREMA, DORADO, GRIS = (16, 38, 31), (247, 244, 236), (217, 174, 82), (201, 216, 209)

# Tramos de cada partida: (video, desde s, hasta s, velocidad, título, subtítulo de la leyenda)
PARTIDA_9 = [
    ("crudo_9", 4.5, 20.5, 2.0, "Reparto", "2 cartas al Jugador y 1 visible a la Casa"),
    ("crudo_9", 20.5, 37.4, 3.0, "Turno del Jugador", "6 + 8 = 14 contra una K"),
    ("crudo_9", 37.4, 41.5, 1.0, "Pide carta", "Gesto: apuntar con el índice"),
    ("crudo_9", 41.5, 50.2, 2.5, "El crupier reparte", "Mientras espera la carta, los gestos se ignoran"),
    ("crudo_9", 50.2, 53.0, 1.0, "Recibe un 6", "6 + 8 + 6 = 20"),
    ("crudo_9", 53.0, 58.5, 1.0, "Se planta con 20", "Gesto: palma abierta"),
    ("crudo_9", 58.5, 77.5, 2.5, "Turno de la Casa", "Da vuelta la carta oculta y pide hasta llegar a 17"),
    ("crudo_9", 77.5, 82.5, 1.0, "Gana el Jugador", "20 a 17"),
]
PARTIDA_12 = [
    ("crudo_12", 4.5, 17.0, 3.0, "Otra partida", "Reparto: A + 5 contra una K"),
    ("crudo_12", 17.0, 25.5, 3.0, "Turno del Jugador", "A + 5 = 16 (el As vale 11)"),
    ("crudo_12", 25.5, 28.0, 1.0, "Pide carta", "Gesto: apuntar con el índice"),
    ("crudo_12", 28.0, 32.0, 2.0, "El crupier reparte", "Una carta girada"),
    ("crudo_12", 32.0, 36.0, 1.0, "Un 5 girado", "Llega a 21 y se planta solo"),
    ("crudo_12", 36.0, 40.5, 1.5, "Turno de la Casa", "K + Q = 20"),
    ("crudo_12", 40.5, 45.0, 1.0, "Gana el Jugador", "21 a 20"),
]


# ---------------------------------------------------------------- paso 1: reprocesar las partidas crudas

def renderizar(nombre):
    """Pasa la partida cruda por el sistema completo y guarda el video anotado (mp4v, intermedio)."""
    from detectores import crear_detector
    from gestos import DetectorDeManos, ReconocedorDeGestos
    from juego import Partida

    detector = crear_detector("A", conf=0.3, pesos="auto")
    partida = Partida(detector, gestos=ReconocedorDeGestos(DetectorDeManos(num_manos=2, modo="video")))
    cap = cv2.VideoCapture(str(PARTIDAS / f"{nombre}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    TRABAJO.mkdir(parents=True, exist_ok=True)
    writer = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vista = partida.procesar(frame, info=f"{detector.nombre} | {fps:.1f} FPS")
        if writer is None:
            writer = cv2.VideoWriter(str(TRABAJO / f"{nombre}_anotado.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     vista.shape[1::-1])
        writer.write(vista)
    cap.release()
    writer.release()
    for e in partida.juego.eventos:
        print(f"  [{e.cuadro:4d}] {e.cuadro / fps:6.1f} s  {e.tipo:13s} {e.texto}")


def cargar(nombre):
    """Cuadros anotados en memoria, comprimidos en JPEG (unas decenas de MB por partida)."""
    cap = cv2.VideoCapture(str(TRABAJO / f"{nombre}_anotado.mp4"))
    fps, cuadros = cap.get(cv2.CAP_PROP_FPS), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cuadros.append(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1])
    return fps, cuadros


# ---------------------------------------------------------------- paso 2: placas y leyendas

def _ruta_fuente(negrita):
    """Segoe UI en Windows; si no está, la DejaVu Sans que trae matplotlib."""
    win = Path(r"C:\Windows\Fonts") / ("segoeuib.ttf" if negrita else "segoeui.ttf")
    if win.exists():
        return str(win)
    import matplotlib
    return str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" /
               ("DejaVuSans-Bold.ttf" if negrita else "DejaVuSans.ttf"))


def fuente(tam, negrita=False):
    return ImageFont.truetype(_ruta_fuente(negrita), tam)


def a_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def a_bgr(pil):
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def partir(texto, f, ancho, d):
    """Parte el texto en líneas que entren en el ancho dado (en píxeles)."""
    lineas, actual = [], ""
    for p in texto.split():
        prueba = (actual + " " + p).strip()
        if d.textlength(prueba, font=f) <= ancho:
            actual = prueba
        else:
            lineas.append(actual)
            actual = p
    return lineas + ([actual] if actual else [])


def recuadro(d, x0, y0, ancho, alto):
    d.rounded_rectangle([x0, y0, x0 + ancho, y0 + alto], 14, fill=VERDE + (238,))
    d.rectangle([x0, y0 + 14, x0 + 6, y0 + alto - 14], fill=DORADO + (255,))


def leyenda(img, titulo, sub):
    """Recuadro en la franja negra derecha, debajo del panel del juego."""
    pil = a_pil(img).convert("RGBA")
    capa = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    x0, y0, ancho = 905, 470, 355
    lineas = partir(sub, fuente(22), ancho - 40, d)
    recuadro(d, x0, y0, ancho, 94 + 30 * len(lineas))
    d.text((x0 + 24, y0 + 22), titulo, font=fuente(34, True), fill=CREMA + (255,))
    for i, l in enumerate(lineas):
        d.text((x0 + 24, y0 + 72 + 30 * i), l, font=fuente(22), fill=GRIS + (255,))
    return a_bgr(Image.alpha_composite(pil, capa).convert("RGB"))


def explicacion(cuadro):
    """Cuadro fijo del reparto con la explicación de los colores del sistema."""
    pil = a_pil(cuadro).convert("RGBA")
    capa = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(capa)
    x0, y0, ancho = 905, 250, 355
    recuadro(d, x0, y0, ancho, 425)
    d.text((x0 + 24, y0 + 20), "Cómo lee la mesa", font=fuente(30, True), fill=CREMA + (255,))
    y = y0 + 78
    for color, texto in [("Verde", "cada esquina detectada, con su valor"),
                         ("Amarillo", "la diagonal que une las dos esquinas de una carta"),
                         ("Azul", "la carta armada"),
                         ("Línea roja", "arriba la Casa, abajo el Jugador")]:
        d.text((x0 + 24, y), color, font=fuente(22, True), fill=CREMA + (255,))
        y += 29
        for l in partir(texto, fuente(22), ancho - 44, d):
            d.text((x0 + 24, y), l, font=fuente(22), fill=GRIS + (255,))
            y += 29
        y += 12
    return a_bgr(Image.alpha_composite(pil, capa).convert("RGB"))


def placa(lineas):
    """Placa de fondo verde: lineas = [(texto, tamaño, negrita, color, y)]."""
    pil = Image.new("RGB", (W, H), VERDE)
    d = ImageDraw.Draw(pil)
    for texto, tam, neg, color, y in lineas:
        d.text((96, y), texto, font=fuente(tam, neg), fill=color)
    return a_bgr(pil)


# ---------------------------------------------------------------- armado

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-leyendas", action="store_true", help="la misma versión, sin leyendas ni explicación")
    ap.add_argument("--rehacer", action="store_true", help="volver a reprocesar las partidas crudas")
    ap.add_argument("--salida", default=str(REPO / "videos" / "demo.mp4"))
    args = ap.parse_args()

    for nombre in ("crudo_9", "crudo_12"):
        if args.rehacer or not (TRABAJO / f"{nombre}_anotado.mp4").exists():
            print(f"reprocesando {nombre} ...", flush=True)
            renderizar(nombre)
    videos = {n: cargar(n) for n in ("crudo_9", "crudo_12")}
    salida = []

    def fijo(img, segundos):
        salida.extend([img] * int(round(segundos * FPS)))

    fijo(placa([("VISIÓN POR COMPUTADORA II · CEIA-FIUBA", 22, True, DORADO, 150),
                ("Blackjack con visión por computadora", 54, True, CREMA, 220),
                ("Demo: dos partidas reales, jugadas con cámara y gestos", 30, False, GRIS, 300),
                ("Cristhian Pettico · Rodolfo Di Chiazza · Julian Blanco", 26, False, CREMA, 520)]), 5)

    fps9, c9 = videos["crudo_9"]
    reparto = cv2.imdecode(c9[int(20.0 * fps9)], cv2.IMREAD_COLOR)
    fijo(reparto if args.sin_leyendas else explicacion(reparto), 7)

    cortes = [len(salida) - 7 * FPS, len(salida)]           # fundidos: título -> explicación -> partida 9
    for tramos in (PARTIDA_9, PARTIDA_12):
        for nombre, t0, t1, vel, tit, sub in tramos:
            fps, cuadros = videos[nombre]
            ultimo, img = None, None
            for k in range(int(round((t1 - t0) / vel * FPS))):
                i = min(int((t0 + k * vel / FPS) * fps), len(cuadros) - 1)
                if i != ultimo:
                    img = cv2.imdecode(cuadros[i], cv2.IMREAD_COLOR)
                    img = img if args.sin_leyendas else leyenda(img, tit, sub)
                    ultimo = i
                salida.append(img)
        cortes.append(len(salida))                          # partida 9 -> 12 -> placa final

    fijo(placa([("RESULTADOS", 22, True, DORADO, 130),
                ("4 / 4 partidas reales con el resultado correcto", 40, True, CREMA, 190),
                ("F1 por esquina 0,96 en nuestra mesa (A ajustado con fotos propias)", 28, False, GRIS, 270),
                ("7–10 cuadros por segundo en CPU, todo el sistema", 28, False, GRIS, 320),
                ("github.com/dichiazzar/VpC_2_TP_2", 30, True, CREMA, 520)]), 6)

    for c in cortes:                                        # fundido cruzado de 0,4 s en cada cambio
        for j in range(1, 6):
            if c + j < len(salida):
                salida[c - 6 + j] = cv2.addWeighted(salida[c - 6 + j], 1 - j / 6, salida[c], j / 6, 0)

    g = GrabadorVideo(args.salida, FPS, (W, H))
    for f in salida:
        g.write(f if f.shape[:2] == (H, W) else cv2.resize(f, (W, H)))
    g.release()
    print(f"{args.salida}: {len(salida) / FPS:.1f} s")


if __name__ == "__main__":
    main()
