"""
Script para capturar imágenes de rostros usando la cámara.
Las imágenes se guardan en la carpeta 'images/<nombre_persona>'.
"""

import os
import sys

import cv2
import imutils


def crear_directorio_persona(nombre: str, directorio_base: str = "images") -> str:
    """Crea el directorio para guardar las imágenes de la persona."""
    ruta_persona = os.path.join(directorio_base, nombre)
    if not os.path.exists(ruta_persona):
        os.makedirs(ruta_persona)
        print(f"Carpeta creada: {ruta_persona}")
    else:
        print(f"Carpeta existente: {ruta_persona}")
    return ruta_persona


def inicializar_camara(indice_camara: int = 0) -> cv2.VideoCapture:
    """Inicializa la cámara con el backend apropiado para el sistema operativo."""
    # En Linux usar CAP_V4L2, en Windows CAP_DSHOW, o dejar que OpenCV elija automáticamente
    if sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(indice_camara, cv2.CAP_V4L2)
    elif sys.platform == "win32":
        cap = cv2.VideoCapture(indice_camara, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(indice_camara)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con índice {indice_camara}")

    return cap


def capturar_rostros(
    nombre_persona: str, max_imagenes: int = 300, indice_camara: int = 0
) -> int:
    """
    Captura imágenes de rostros de una persona.

    Args:
        nombre_persona: Nombre de la persona para crear la carpeta.
        max_imagenes: Número máximo de imágenes a capturar.
        indice_camara: Índice de la cámara a usar.

    Returns:
        Número de imágenes capturadas.
    """
    ruta_persona = crear_directorio_persona(nombre_persona)

    try:
        cap = inicializar_camara(indice_camara)
    except RuntimeError as e:
        print(f"Error: {e}")
        return 0

    # Cargar el clasificador de rostros
    clasificador_rostros = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
    )

    contador = 0
    print(f"Capturando rostros de '{nombre_persona}'. Presiona ESC para salir.")
    print(f"Se capturarán hasta {max_imagenes} imágenes.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo leer el frame de la cámara.")
            break

        frame = imutils.resize(frame, width=640)
        frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_original = frame.copy()

        # Detectar rostros
        rostros = clasificador_rostros.detectMultiScale(
            frame_gris, scaleFactor=1.3, minNeighbors=5
        )

        for x, y, w, h in rostros:
            # Dibujar rectángulo en el rostro detectado
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Recortar y redimensionar el rostro
            rostro = frame_original[y : y + h, x : x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            # Guardar la imagen
            nombre_archivo = os.path.join(ruta_persona, f"rostro_{contador}.jpg")
            cv2.imwrite(nombre_archivo, rostro)
            contador += 1

            # Mostrar progreso
            cv2.putText(
                frame,
                f"Capturadas: {contador}/{max_imagenes}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Captura de Rostros - ESC para salir", frame)

        # Salir con ESC o al alcanzar el máximo de imágenes
        if cv2.waitKey(1) == 27 or contador >= max_imagenes:
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Captura finalizada. Total de imágenes: {contador}")
    return contador


def main():
    """Función principal del programa."""
    print("=== Captura de Rostros ===")
    print("Escribe 'salir' para terminar el programa.\n")

    while True:
        nombre = input("Nombre de la persona: ").strip()

        if nombre.lower() in ("salir", "exit", "none", ""):
            print("Saliendo del programa...")
            break

        capturar_rostros(nombre)
        print()  # Línea en blanco para separar


if __name__ == "__main__":
    main()
