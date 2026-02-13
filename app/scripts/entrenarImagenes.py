"""
Script para entrenar un modelo de reconocimiento facial.
Lee las imágenes de la carpeta 'images/<persona>' y genera un modelo XML.

Uso: python entrenarImagenes.py
"""

import os
import sys
from enum import Enum

import cv2
import numpy as np


class TipoReconocedor(Enum):
    """Tipos de reconocedores faciales disponibles en OpenCV."""

    EIGEN = "EigenFace"
    FISHER = "FisherFace"
    LBPH = "LBPHFace"


def obtener_reconocedor(tipo: TipoReconocedor):
    """
    Crea y retorna el reconocedor facial según el tipo especificado.

    Args:
        tipo: Tipo de reconocedor a crear.

    Returns:
        Instancia del reconocedor facial.
    """
    reconocedores = {
        TipoReconocedor.EIGEN: cv2.face.EigenFaceRecognizer_create,
        TipoReconocedor.FISHER: cv2.face.FisherFaceRecognizer_create,
        TipoReconocedor.LBPH: cv2.face.LBPHFaceRecognizer_create,
    }
    return reconocedores[tipo]()


def cargar_imagenes(
    directorio_datos: str, mostrar_progreso: bool = True
) -> tuple[list, list, list]:
    """
    Carga las imágenes de rostros desde el directorio de datos.

    Args:
        directorio_datos: Ruta al directorio con las carpetas de personas.
        mostrar_progreso: Si True, muestra cada imagen mientras se carga.

    Returns:
        Tupla con (datos_rostros, etiquetas, lista_personas)
    """
    if not os.path.exists(directorio_datos):
        raise FileNotFoundError(f"No se encontró el directorio: {directorio_datos}")

    lista_personas = os.listdir(directorio_datos)

    if not lista_personas:
        raise ValueError(f"No hay carpetas de personas en: {directorio_datos}")

    print(f"Personas encontradas: {lista_personas}")
    print(f"Total: {len(lista_personas)} personas\n")

    etiquetas = []
    datos_rostros = []

    for etiqueta, nombre_persona in enumerate(lista_personas):
        ruta_persona = os.path.join(directorio_datos, nombre_persona)

        if not os.path.isdir(ruta_persona):
            continue

        archivos = os.listdir(ruta_persona)
        print(f"Leyendo imágenes de '{nombre_persona}' ({len(archivos)} archivos)...")

        for nombre_archivo in archivos:
            ruta_imagen = os.path.join(ruta_persona, nombre_archivo)

            # Leer imagen en escala de grises
            imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

            if imagen is None:
                print(f"  Advertencia: No se pudo leer {nombre_archivo}")
                continue

            etiquetas.append(etiqueta)
            datos_rostros.append(imagen)

            # Mostrar progreso visual
            if mostrar_progreso:
                cv2.imshow(f"Cargando: {nombre_persona}", imagen)
                cv2.waitKey(3)

    cv2.destroyAllWindows()

    return datos_rostros, etiquetas, lista_personas


def entrenar_modelo(
    directorio_datos: str = "images",
    tipo_reconocedor: TipoReconocedor = TipoReconocedor.LBPH,
    mostrar_progreso: bool = True,
) -> str:
    """
    Entrena el modelo de reconocimiento facial y lo guarda en un archivo XML.

    Args:
        directorio_datos: Ruta al directorio con las imágenes.
        tipo_reconocedor: Tipo de reconocedor a usar.
        mostrar_progreso: Si True, muestra las imágenes mientras se cargan.

    Returns:
        Ruta al archivo del modelo guardado.
    """
    print("=" * 50)
    print("   ENTRENAMIENTO DE MODELO FACIAL")
    print("=" * 50)
    print(f"Reconocedor: {tipo_reconocedor.value}")
    print(f"Directorio: {directorio_datos}\n")

    # Cargar imágenes
    datos_rostros, etiquetas, lista_personas = cargar_imagenes(
        directorio_datos, mostrar_progreso
    )

    print(f"\nTotal de imágenes cargadas: {len(datos_rostros)}")

    # Mostrar estadísticas por persona
    for i, persona in enumerate(lista_personas):
        cantidad = np.count_nonzero(np.array(etiquetas) == i)
        print(f"  - {persona}: {cantidad} imágenes")

    # Crear y entrenar el reconocedor
    print(f"\nEntrenando modelo {tipo_reconocedor.value}...")
    reconocedor = obtener_reconocedor(tipo_reconocedor)
    reconocedor.train(datos_rostros, np.array(etiquetas))

    # Guardar el modelo
    nombre_modelo = f"modelo{tipo_reconocedor.value}.xml"
    reconocedor.write(nombre_modelo)

    print("Modelo guardado exitosamente")
    print(f"Archivo: {nombre_modelo}")
    print("=" * 50)

    return nombre_modelo


def main():
    """Función principal."""
    try:
        entrenar_modelo(
            directorio_datos="images",
            tipo_reconocedor=TipoReconocedor.LBPH,
            mostrar_progreso=True,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
