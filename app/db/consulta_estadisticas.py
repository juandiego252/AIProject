"""
Script de ejemplo para consultar estadísticas de Supabase.
Demuestra cómo usar el gestor de sesiones para análisis de datos.

Uso: python consultar_estadisticas.py
"""

import os
import sys
from datetime import datetime, timedelta
from tabulate import tabulate

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.db.supabase_session import SupabaseSessionManager


def mostrar_estadisticas_generales(manager: SupabaseSessionManager):
    """Muestra estadísticas generales del sistema."""
    print("\n" + "=" * 70)
    print("📊 ESTADÍSTICAS GENERALES")
    print("=" * 70)

    stats = manager.get_statistics()

    print(f"\n✅ Accesos exitosos: {stats['total_successful']}")
    print(f"❌ Accesos fallidos: {stats['total_failed']}")
    print(f"📈 Total de eventos: {stats['total_successful'] + stats['total_failed']}")

    if stats["total_successful"] + stats["total_failed"] > 0:
        tasa_exito = (
            stats["total_successful"]
            / (stats["total_successful"] + stats["total_failed"])
        ) * 100
        print(f"✨ Tasa de éxito: {tasa_exito:.2f}%")

    print("\n" + "-" * 70)
    print("👥 ACCESOS POR PERSONA")
    print("-" * 70)

    if stats["by_person"]:
        data = [[nombre, count] for nombre, count in stats["by_person"].items()]
        headers = ["Persona", "Accesos Exitosos"]
        print(tabulate(data, headers=headers, tablefmt="grid"))
    else:
        print("No hay datos disponibles.")


def mostrar_historial_reciente(manager: SupabaseSessionManager, limit: int = 20):
    """Muestra el historial reciente de accesos."""
    print("\n" + "=" * 70)
    print(f"📜 HISTORIAL RECIENTE (últimos {limit} eventos)")
    print("=" * 70)

    historial = manager.get_access_history(limit=limit)

    if not historial:
        print("No hay eventos registrados.")
        return

    data = []
    for evento in historial:
        timestamp = datetime.fromisoformat(
            evento["access_timestamp"].replace("Z", "+00:00")
        )
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        estado = "✅" if evento["access_granted"] else "❌"
        persona = evento.get("person_name", "Desconocido") or "Desconocido"
        confianza = evento.get("confidence", 0)
        tipo = evento.get("event_type", "N/A")

        data.append([timestamp_str, estado, persona, f"{confianza:.2f}", tipo])

    headers = ["Timestamp", "Estado", "Persona", "Confianza", "Tipo"]
    print(tabulate(data, headers=headers, tablefmt="grid"))


def mostrar_accesos_por_persona(manager: SupabaseSessionManager, persona: str):
    """Muestra el historial de una persona específica."""
    print("\n" + "=" * 70)
    print(f"👤 HISTORIAL DE: {persona}")
    print("=" * 70)

    historial = manager.get_access_history(person_name=persona, limit=50)

    if not historial:
        print(f"No hay eventos registrados para {persona}.")
        return

    exitosos = sum(1 for e in historial if e["access_granted"])
    fallidos = sum(1 for e in historial if not e["access_granted"])

    print(f"\n📊 Total de intentos: {len(historial)}")
    print(f"✅ Exitosos: {exitosos}")
    print(f"❌ Fallidos: {fallidos}")

    if len(historial) > 0:
        confianzas = [e["confidence"] for e in historial if e["confidence"] > 0]
        if confianzas:
            print(f"📈 Confianza promedio: {sum(confianzas) / len(confianzas):.2f}")
            print(f"🎯 Mejor confianza: {min(confianzas):.2f}")

    print("\n" + "-" * 70)
    print("ÚLTIMOS 10 EVENTOS:")
    print("-" * 70)

    data = []
    for evento in historial[:10]:
        timestamp = datetime.fromisoformat(
            evento["access_timestamp"].replace("Z", "+00:00")
        )
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        estado = "✅" if evento["access_granted"] else "❌"
        confianza = evento.get("confidence", 0)
        tipo = evento.get("event_type", "N/A")

        data.append([timestamp_str, estado, f"{confianza:.2f}", tipo])

    headers = ["Timestamp", "Estado", "Confianza", "Tipo"]
    print(tabulate(data, headers=headers, tablefmt="grid"))


def mostrar_fallos_recientes(manager: SupabaseSessionManager, limit: int = 10):
    """Muestra los intentos fallidos recientes."""
    print("\n" + "=" * 70)
    print(f"⚠️ INTENTOS FALLIDOS RECIENTES (últimos {limit})")
    print("=" * 70)

    historial = manager.get_access_history(access_granted=False, limit=limit)

    if not historial:
        print("No hay fallos registrados. ¡Excelente!")
        return

    data = []
    for evento in historial:
        timestamp = datetime.fromisoformat(
            evento["access_timestamp"].replace("Z", "+00:00")
        )
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

        razon = evento.get("failure_reason", "N/A")
        confianza = evento.get("confidence", 0)
        imagen = "Sí" if evento.get("image_path") else "No"

        data.append([timestamp_str, razon, f"{confianza:.2f}", imagen])

    headers = ["Timestamp", "Razón", "Confianza", "Imagen Guardada"]
    print(tabulate(data, headers=headers, tablefmt="grid"))


def mostrar_entrenamientos(manager: SupabaseSessionManager):
    """Muestra el historial de entrenamientos."""
    print("\n" + "=" * 70)
    print("🧠 HISTORIAL DE ENTRENAMIENTOS")
    print("=" * 70)

    try:
        response = (
            manager.client.table("training_sessions")
            .select("*")
            .order("training_timestamp", desc=True)
            .limit(20)
            .execute()
        )

        if not response.data:
            print("No hay entrenamientos registrados.")
            return

        data = []
        for sesion in response.data:
            timestamp = datetime.fromisoformat(
                sesion["training_timestamp"].replace("Z", "+00:00")
            )
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

            persona = sesion["person_name"]
            imagenes = sesion["images_count"]
            modelo = sesion["model_type"]
            estado = "✅" if sesion.get("success", True) else "❌"

            data.append([timestamp_str, persona, imagenes, modelo, estado])

        headers = ["Timestamp", "Persona", "Imágenes", "Modelo", "Estado"]
        print(tabulate(data, headers=headers, tablefmt="grid"))

    except Exception as e:
        print(f"Error al obtener entrenamientos: {e}")


def menu_principal():
    """Menú interactivo principal."""
    try:
        manager = SupabaseSessionManager()
        print("\n✅ Conectado a Supabase exitosamente!")
    except Exception as e:
        print(f"\n❌ Error al conectar con Supabase: {e}")
        print("Verifica tu archivo .env y la configuración de Supabase.")
        return

    while True:
        print("\n" + "=" * 70)
        print("🎭 SISTEMA DE CONSULTA DE ESTADÍSTICAS")
        print("=" * 70)
        print("\nOpciones:")
        print("  1. Ver estadísticas generales")
        print("  2. Ver historial reciente")
        print("  3. Ver historial de una persona específica")
        print("  4. Ver intentos fallidos recientes")
        print("  5. Ver historial de entrenamientos")
        print("  6. Salir")

        opcion = input("\nSelecciona una opción (1-6): ").strip()

        if opcion == "1":
            mostrar_estadisticas_generales(manager)

        elif opcion == "2":
            try:
                limit = int(
                    input("¿Cuántos eventos mostrar? (por defecto 20): ").strip()
                    or "20"
                )
                mostrar_historial_reciente(manager, limit)
            except ValueError:
                print("❌ Valor inválido. Usando 20 por defecto.")
                mostrar_historial_reciente(manager, 20)

        elif opcion == "3":
            persona = input("Nombre de la persona: ").strip()
            if persona:
                mostrar_accesos_por_persona(manager, persona)
            else:
                print("❌ Nombre inválido.")

        elif opcion == "4":
            try:
                limit = int(
                    input("¿Cuántos fallos mostrar? (por defecto 10): ").strip() or "10"
                )
                mostrar_fallos_recientes(manager, limit)
            except ValueError:
                print("❌ Valor inválido. Usando 10 por defecto.")
                mostrar_fallos_recientes(manager, 10)

        elif opcion == "5":
            mostrar_entrenamientos(manager)

        elif opcion == "6":
            print("\n👋 ¡Hasta luego!")
            break

        else:
            print("❌ Opción inválida. Intenta de nuevo.")

        input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    # Verificar si tabulate está instalado
    try:
        import tabulate
    except ImportError:
        print("⚠️ Instalando tabulate para mejor visualización...")
        os.system("pip install tabulate")
        import tabulate

    menu_principal()
