"""
Script para probar el reconocimiento facial en tiempo real con Supabase.
Usa el modelo entrenado para identificar personas a través de la cámara.
Registra todos los intentos de acceso en Supabase.

Uso: python probarImagenes.py
"""

import os
import sys

import cv2
import numpy as np

# Importar el gestor de sesiones
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from app.db.supabase_session import SupabaseSessionManager

    SUPABASE_ENABLED = True
    print("✅ Módulo supabase_session importado correctamente al probar imágenes")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    SUPABASE_ENABLED = False
except Exception as e:
    print(f"❌ Error al cargar Supabase: {e}")
    SUPABASE_ENABLED = False


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
    logged: bool = False,
):
    """
    Dibuja el rectángulo y el nombre de la persona detectada.

    Args:
        frame: Frame de video.
        x, y, w, h: Coordenadas del rostro.
        nombre: Nombre de la persona detectada.
        confianza: Valor de confianza del reconocimiento.
        umbral: Umbral de confianza (menor = más confianza).
        logged: Si el evento fue registrado en la base de datos.
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

    # Indicador de registro en DB
    if logged and SUPABASE_ENABLED:
        cv2.putText(
            frame,
            "DB: OK",
            (x + w - 60, y + h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def reconocer_rostros(
    ruta_modelo: str = "modeloLBPHFace.xml",
    directorio_datos: str = "images",
    indice_camara: int = 0,
    umbral_confianza: float = 70,
    log_interval: int = 30,  # Registrar cada 30 frames para el mismo rostro
):
    """
    Ejecuta el reconocimiento facial en tiempo real con integración Supabase.

    Args:
        ruta_modelo: Ruta al archivo del modelo entrenado.
        directorio_datos: Ruta al directorio con las carpetas de personas.
        indice_camara: Índice de la cámara a usar.
        umbral_confianza: Umbral de confianza (menor = más estricto).
        log_interval: Frames entre cada registro para evitar spam.
    """
    print("=" * 50)
    print("   RECONOCIMIENTO FACIAL EN TIEMPO REAL")
    print("=" * 50)
    print(f"Modelo: {ruta_modelo}")
    print(f"Umbral de confianza: {umbral_confianza}")

    # Inicializar gestor de sesiones
    session_manager = None
    if SUPABASE_ENABLED:
        try:
            session_manager = SupabaseSessionManager()
            print("✅ Supabase conectado - Registros activos")
        except Exception as e:
            print(f"⚠️ Error al conectar Supabase: {e}")
            print("Continuando sin registro de sesiones...")
    else:
        print("⚠️ Modo offline - Sin registro de sesiones")

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

    # Control de registro para evitar spam
    frame_counter = 0
    last_logged_person = None
    last_logged_frame = -log_interval
    last_logged_event = None

    print("🎥 Iniciando reconocimiento...\n")

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

        logged_this_frame = False

        if len(rostros) == 0:
            # No se detectó ningún rostro
            # Solo registrar si ha pasado suficiente tiempo
            if session_manager and (frame_counter - last_logged_frame >= log_interval):
                if last_logged_event != "no_face":
                    # Solo si el evento anterior no fue "no_face"
                    # Evitamos spam de "no face detected"
                    pass
                last_logged_event = "no_face"
        else:
            for x, y, w, h in rostros:
                # Recortar y redimensionar el rostro
                rostro = frame_gris[y : y + h, x : x + w]
                rostro_color = frame[y : y + h, x : x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

                # Predecir identidad
                etiqueta, confianza = reconocedor.predict(rostro)

                # Obtener nombre de la persona
                nombre = (
                    lista_personas[etiqueta]
                    if etiqueta < len(lista_personas)
                    else "Desconocido"
                )

                # Determinar si el acceso es exitoso
                access_granted = confianza < umbral_confianza

                # Registrar en Supabase si es necesario
                should_log = session_manager and (
                    frame_counter - last_logged_frame >= log_interval
                    or last_logged_person != nombre
                )

                if should_log:
                    try:
                        if access_granted:
                            # Acceso exitoso
                            session_manager.log_successful_access(
                                person_name=nombre,
                                confidence=confianza,
                                face_image=rostro_color,
                                additional_data={
                                    "threshold": umbral_confianza,
                                    "face_position": {
                                        "x": int(x),
                                        "y": int(y),
                                        "w": int(w),
                                        "h": int(h),
                                    },
                                },
                            )
                            logged_this_frame = True
                            last_logged_event = "success"
                        else:
                            # Acceso fallido - persona desconocida o baja confianza
                            reason = (
                                "unknown_person"
                                if nombre == "Desconocido"
                                else "low_confidence"
                            )
                            session_manager.log_failed_access(
                                confidence=confianza,
                                face_image=rostro_color,
                                reason=reason,
                                additional_data={
                                    "threshold": umbral_confianza,
                                    "predicted_name": nombre
                                    if nombre != "Desconocido"
                                    else None,
                                    "face_position": {
                                        "x": int(x),
                                        "y": int(y),
                                        "w": int(w),
                                        "h": int(h),
                                    },
                                },
                            )
                            logged_this_frame = True
                            last_logged_event = "failed"

                        last_logged_frame = frame_counter
                        last_logged_person = nombre
                    except Exception as e:
                        print(f"❌ Error al registrar en DB: {e}")

                # Dibujar resultado
                dibujar_resultado(
                    frame,
                    x,
                    y,
                    w,
                    h,
                    nombre,
                    confianza,
                    umbral_confianza,
                    logged_this_frame,
                )

        # Mostrar contador de frames y estado
        cv2.putText(
            frame,
            f"Frame: {frame_counter}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if SUPABASE_ENABLED and session_manager:
            status_text = "DB: Conectado"
            status_color = (0, 255, 0)
        else:
            status_text = "DB: Offline"
            status_color = (0, 0, 255)

        cv2.putText(
            frame,
            status_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Reconocimiento Facial - ESC para salir", frame)

        frame_counter += 1

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Mostrar estadísticas finales
    if session_manager:
        print("\n" + "=" * 50)
        print("   ESTADÍSTICAS DE LA SESIÓN")
        print("=" * 50)
        stats = session_manager.get_statistics()
        print(f"Accesos exitosos: {stats['total_successful']}")
        print(f"Accesos fallidos: {stats['total_failed']}")
        if stats["by_person"]:
            print("\nPor persona:")
            for person, count in stats["by_person"].items():
                print(f"  - {person}: {count} accesos")

    print("\n✅ Programa finalizado.")


def main():
    """Función principal."""
    try:
        reconocer_rostros(
            ruta_modelo="modeloLBPHFace.xml",
            directorio_datos="images",
            indice_camara=0,
            umbral_confianza=70,
            log_interval=30,  # Registrar cada 30 frames
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
