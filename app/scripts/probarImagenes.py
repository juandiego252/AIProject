"""
Script para probar el reconocimiento facial en tiempo real.
Usa el modelo entrenado para identificar personas a través de la cámara.

Uso: python probarImagenes.py
"""

import os
import sys

import cv2


def inicializar_camara(indice_camara: int = 0) -> cv2.VideoCapture:
    """
    Inicializa la cámara con el backend apropiado según el sistema operativo.
    """
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(indice_camara, cv2.CAP_V4L2)
    elif sys.platform == "win32":
        cap = cv2.VideoCapture(indice_camara, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(indice_camara)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con índice {indice_camara}")

    return cap


def cargar_modelo(ruta_modelo: str, directorio_datos: str) -> tuple:
    """
    Carga el modelo de reconocimiento facial y la lista de personas.

    Args:
        ruta_modelo: Ruta al archivo XML del modelo.
        directorio_datos: Ruta al directorio con las carpetas de personas.

    Returns:
        Tupla con (reconocedor, lista_personas)
    """
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"No se encontró el modelo: {ruta_modelo}")

    if not os.path.exists(directorio_datos):
        raise FileNotFoundError(f"No se encontró el directorio: {directorio_datos}")

    # Crear reconocedor LBPH y cargar modelo
    reconocedor = cv2.face.LBPHFaceRecognizer_create()
    reconocedor.read(ruta_modelo)

    # Obtener lista de personas (debe coincidir con el orden del entrenamiento)
    lista_personas = sorted(os.listdir(directorio_datos))

    return reconocedor, lista_personas


def dibujar_resultado(
    frame,
    x: int,
    y: int,
    w: int,
    h: int,
    nombre: str,
    confianza: float,
    umbral: float = 70,
):
    """
    Dibuja el rectángulo y el nombre de la persona detectada.

    Args:
        frame: Frame de video.
        x, y, w, h: Coordenadas del rostro.
        nombre: Nombre de la persona detectada.
        confianza: Valor de confianza del reconocimiento.
        umbral: Umbral de confianza (menor = más confianza).
    """
    if confianza < umbral:
        # Persona reconocida (verde)
        color = (0, 255, 0)
        texto = nombre
    else:
        # Persona desconocida (rojo)
        color = (0, 0, 255)
        texto = "Desconocido"

    # Dibujar rectángulo
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Dibujar nombre
    cv2.putText(
        frame, texto, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA
    )

    # Dibujar confianza
    cv2.putText(
        frame,
        f"Conf: {confianza:.1f}",
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )


def reconocer_rostros(
    ruta_modelo: str = "modeloLBPHFace.xml",
    directorio_datos: str = "images",
    indice_camara: int = 0,
    umbral_confianza: float = 70,
):
    """
    Ejecuta el reconocimiento facial en tiempo real.

    Args:
        ruta_modelo: Ruta al archivo del modelo entrenado.
        directorio_datos: Ruta al directorio con las carpetas de personas.
        indice_camara: Índice de la cámara a usar.
        umbral_confianza: Umbral de confianza (menor = más estricto).
    """
    print("=" * 50)
    print("   RECONOCIMIENTO FACIAL EN TIEMPO REAL")
    print("=" * 50)
    print(f"Modelo: {ruta_modelo}")
    print(f"Umbral de confianza: {umbral_confianza}")
    print("Presiona ESC para salir.\n")

    # Cargar modelo y lista de personas
    reconocedor, lista_personas = cargar_modelo(ruta_modelo, directorio_datos)
    print(f"Personas registradas: {lista_personas}\n")

    # Inicializar cámara
    cap = inicializar_camara(indice_camara)

    # Cargar clasificador de rostros
    clasificador_rostros = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostros
        rostros = clasificador_rostros.detectMultiScale(
            frame_gris, scaleFactor=1.3, minNeighbors=5
        )

        for x, y, w, h in rostros:
            # Recortar y redimensionar el rostro
            rostro = frame_gris[y : y + h, x : x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            # Predecir identidad
            etiqueta, confianza = reconocedor.predict(rostro)

            # Obtener nombre de la persona
            nombre = (
                lista_personas[etiqueta]
                if etiqueta < len(lista_personas)
                else "Desconocido"
            )

            # Dibujar resultado
            dibujar_resultado(frame, x, y, w, h, nombre, confianza, umbral_confianza)

        cv2.imshow("Reconocimiento Facial - ESC para salir", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado.")


def main():
    """Función principal."""
    try:
        reconocer_rostros(
            ruta_modelo="modeloLBPHFace.xml",
            directorio_datos="images",
            indice_camara=0,
            umbral_confianza=70,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
