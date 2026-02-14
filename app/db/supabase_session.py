"""
Gestor de sesiones de reconocimiento facial con Supabase.
Maneja el registro de intentos de acceso exitosos y fallidos.
"""

import os
import base64
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv


@dataclass
class RecognitionResult:
    """Resultado de un intento de reconocimiento facial."""

    success: bool
    person_name: Optional[str]
    confidence: float
    timestamp: datetime
    image_path: Optional[str] = None
    image_base64: Optional[str] = None


class SupabaseSessionManager:
    """Gestiona las sesiones de reconocimiento facial en Supabase."""

    def __init__(self):
        """Inicializa el cliente de Supabase."""
        print(f"🔗 Cargando configuración de Supabase desde .env")
        load_dotenv()

        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")

        print(f"🔗 Conectando a Supabase: {self.supabase_url}")

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "SUPABASE_URL y SUPABASE_KEY deben estar definidos en .env"
            )

        self.client: Client = create_client(self.supabase_url, self.supabase_key)
        self.failed_attempts_dir = "failed_attempts"

        # Crear directorio para intentos fallidos
        if not os.path.exists(self.failed_attempts_dir):
            os.makedirs(self.failed_attempts_dir)

    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convierte una imagen de OpenCV a base64."""
        _, buffer = cv2.imencode(".jpg", image)
        return base64.b64encode(buffer).decode("utf-8")

    def _save_failed_attempt_image(self, image: np.ndarray, timestamp: datetime) -> str:
        """Guarda la imagen de un intento fallido localmente."""
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"failed_{timestamp_str}.jpg"
        filepath = os.path.join(self.failed_attempts_dir, filename)

        cv2.imwrite(filepath, image)
        return filepath

    def log_successful_access(
        self,
        person_name: str,
        confidence: float,
        face_image: Optional[np.ndarray] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Registra un acceso exitoso en Supabase.

        Args:
            person_name: Nombre de la persona reconocida.
            confidence: Nivel de confianza del reconocimiento.
            face_image: Imagen del rostro detectado (opcional).
            additional_data: Datos adicionales a almacenar.

        Returns:
            Respuesta de Supabase con los datos insertados.
        """
        timestamp = datetime.now()

        data = {
            "person_name": person_name,
            "confidence": float(confidence),
            "access_timestamp": timestamp.isoformat(),
            "access_granted": True,
            "event_type": "successful_access",
        }

        # Añadir imagen en base64 si se proporciona
        if face_image is not None:
            data["face_image_base64"] = self._image_to_base64(face_image)

        # Añadir datos adicionales
        if additional_data:
            data["additional_data"] = additional_data

        try:
            response = self.client.table("access_logs").insert(data).execute()
            print(
                f"✅ Acceso exitoso registrado: {person_name} (confianza: {confidence:.2f})"
            )
            return response.data[0] if response.data else {}
        except Exception as e:
            print(f"❌ Error al registrar acceso exitoso: {e}")
            return {"error": str(e)}

    def log_failed_access(
        self,
        confidence: float,
        face_image: np.ndarray,
        reason: str = "unknown_person",
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Registra un intento de acceso fallido en Supabase.

        Args:
            confidence: Nivel de confianza del reconocimiento.
            face_image: Imagen del rostro detectado.
            reason: Razón del fallo (unknown_person, low_confidence, etc.).
            additional_data: Datos adicionales a almacenar.

        Returns:
            Respuesta de Supabase con los datos insertados.
        """
        timestamp = datetime.now()

        # Guardar imagen localmente
        image_path = self._save_failed_attempt_image(face_image, timestamp)

        data = {
            "person_name": None,
            "confidence": float(confidence),
            "access_timestamp": timestamp.isoformat(),
            "access_granted": False,
            "event_type": "failed_access",
            "failure_reason": reason,
            "image_path": image_path,
            "face_image_base64": self._image_to_base64(face_image),
        }

        # Añadir datos adicionales
        if additional_data:
            data["additional_data"] = additional_data

        try:
            response = self.client.table("access_logs").insert(data).execute()
            print(
                f"⚠️ Acceso fallido registrado: {reason} (confianza: {confidence:.2f})"
            )
            print(f"   Imagen guardada: {image_path}")
            return response.data[0] if response.data else {}
        except Exception as e:
            print(f"❌ Error al registrar acceso fallido: {e}")
            return {"error": str(e)}

    def log_no_face_detected(
        self, frame_image: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Registra un intento donde no se detectó ningún rostro.

        Args:
            frame_image: Imagen del frame completo (opcional).

        Returns:
            Respuesta de Supabase con los datos insertados.
        """
        timestamp = datetime.now()

        data = {
            "person_name": None,
            "confidence": 0.0,
            "access_timestamp": timestamp.isoformat(),
            "access_granted": False,
            "event_type": "no_face_detected",
            "failure_reason": "no_face_detected",
        }

        if frame_image is not None:
            image_path = self._save_failed_attempt_image(frame_image, timestamp)
            data["image_path"] = image_path
            data["face_image_base64"] = self._image_to_base64(frame_image)

        try:
            response = self.client.table("access_logs").insert(data).execute()
            print(f"⚠️ No se detectó rostro en el intento")
            return response.data[0] if response.data else {}
        except Exception as e:
            print(f"❌ Error al registrar intento sin rostro: {e}")
            return {"error": str(e)}

    def get_access_history(
        self,
        person_name: Optional[str] = None,
        limit: int = 100,
        access_granted: Optional[bool] = None,
    ) -> list:
        """
        Obtiene el historial de accesos.

        Args:
            person_name: Filtrar por nombre de persona (opcional).
            limit: Número máximo de registros.
            access_granted: Filtrar por acceso exitoso/fallido (opcional).

        Returns:
            Lista de registros de acceso.
        """
        try:
            query = self.client.table("access_logs").select("*")

            if person_name:
                query = query.eq("person_name", person_name)

            if access_granted is not None:
                query = query.eq("access_granted", access_granted)

            response = query.order("access_timestamp", desc=True).limit(limit).execute()
            return response.data
        except Exception as e:
            print(f"❌ Error al obtener historial: {e}")
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de los accesos.

        Returns:
            Diccionario con estadísticas.
        """
        try:
            # Total de accesos exitosos
            successful = (
                self.client.table("access_logs")
                .select("*", count="exact")
                .eq("access_granted", True)
                .execute()
            )

            # Total de accesos fallidos
            failed = (
                self.client.table("access_logs")
                .select("*", count="exact")
                .eq("access_granted", False)
                .execute()
            )

            # Accesos por persona
            by_person = (
                self.client.table("access_logs")
                .select("person_name")
                .eq("access_granted", True)
                .execute()
            )

            person_counts = {}
            if by_person.data:
                for record in by_person.data:
                    name = record.get("person_name")
                    if name:
                        person_counts[name] = person_counts.get(name, 0) + 1

            return {
                "total_successful": successful.count
                if hasattr(successful, "count")
                else 0,
                "total_failed": failed.count if hasattr(failed, "count") else 0,
                "by_person": person_counts,
            }
        except Exception as e:
            print(f"❌ Error al obtener estadísticas: {e}")
            return {"total_successful": 0, "total_failed": 0, "by_person": {}}

    def register_training_session(
        self, person_name: str, images_count: int, model_type: str, success: bool = True
    ) -> Dict[str, Any]:
        """
        Registra una sesión de entrenamiento del modelo.

        Args:
            person_name: Nombre de la persona entrenada.
            images_count: Número de imágenes capturadas.
            model_type: Tipo de modelo entrenado.
            success: Si el entrenamiento fue exitoso.

        Returns:
            Respuesta de Supabase.
        """
        data = {
            "person_name": person_name,
            "images_count": images_count,
            "model_type": model_type,
            "training_timestamp": datetime.now().isoformat(),
            "success": success,
        }

        try:
            response = self.client.table("training_sessions").insert(data).execute()
            print(
                f"✅ Sesión de entrenamiento registrada en training_sessions: {person_name} ({images_count} imgs)"
            )
            return response.data[0] if response.data else {}
        except Exception as e:
            print(f"❌ Error al registrar sesión de entrenamiento: {e}")
            return {"error": str(e)}


# Función auxiliar para facilitar el uso
def create_session_manager() -> SupabaseSessionManager:
    """Crea y retorna una instancia del gestor de sesiones."""
    return SupabaseSessionManager()
