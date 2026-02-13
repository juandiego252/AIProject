"""
Interfaz gráfica para el sistema de reconocimiento facial.
Integra captura, entrenamiento y reconocimiento en una sola aplicación.

Uso: python gui.py
"""

import os
import sys
import threading
import tkinter as tk
from datetime import datetime
from enum import Enum
from tkinter import messagebox, scrolledtext, ttk

import cv2
import imutils
import numpy as np


class TipoReconocedor(Enum):
    """Tipos de reconocedores faciales disponibles en OpenCV."""

    EIGEN = "EigenFace"
    FISHER = "FisherFace"
    LBPH = "LBPHFace"


class AplicacionReconocimientoFacial:
    """Aplicación principal de reconocimiento facial con interfaz gráfica."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Sistema de Reconocimiento Facial")
        self.root.geometry("800x700")
        self.root.resizable(True, True)

        self.directorio_imagenes = "images"
        self.captura_activa = False
        self.reconocimiento_activo = False
        self.hilo_actual = None

        self.configurar_estilo()
        self.crear_interfaz()

        self.log("Sistema de Reconocimiento Facial iniciado.")
        self.log(f"Directorio de imágenes: {os.path.abspath(self.directorio_imagenes)}")
        self.actualizar_lista_personas()

    def configurar_estilo(self):
        """Configura el estilo visual de la aplicación."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=8)
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"))
        style.configure("Subheader.TLabel", font=("Segoe UI", 11, "bold"))

    def crear_interfaz(self):
        """Crea todos los elementos de la interfaz gráfica."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        titulo = ttk.Label(
            main_frame,
            text="🎭 Sistema de Reconocimiento Facial",
            style="Header.TLabel",
        )
        titulo.grid(row=0, column=0, columnspan=2, pady=(0, 15))

        self.crear_panel_izquierdo(main_frame)
        self.crear_panel_derecho(main_frame)
        self.crear_area_log(main_frame)

    def crear_panel_izquierdo(self, parent):
        """Crea el panel izquierdo con captura y entrenamiento."""
        panel = ttk.LabelFrame(parent, text="Captura y Entrenamiento", padding="10")
        panel.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=5)

        # === Sección de Captura ===
        ttk.Label(panel, text="📷 Captura de Rostros", style="Subheader.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        ttk.Label(panel, text="Nombre:").grid(row=1, column=0, sticky="w", pady=5)
        self.entry_nombre = ttk.Entry(panel, width=25)
        self.entry_nombre.grid(row=1, column=1, sticky="ew", pady=5, padx=(5, 0))

        ttk.Label(panel, text="Máx. imágenes:").grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.spin_max_imagenes = ttk.Spinbox(
            panel, from_=50, to=500, width=10, increment=50
        )
        self.spin_max_imagenes.set(300)
        self.spin_max_imagenes.grid(row=2, column=1, sticky="w", pady=5, padx=(5, 0))

        # Frame para botones de captura
        frame_botones_cap = ttk.Frame(panel)
        frame_botones_cap.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        frame_botones_cap.columnconfigure(0, weight=1)
        frame_botones_cap.columnconfigure(1, weight=1)

        self.btn_capturar = ttk.Button(
            frame_botones_cap, text="▶ Iniciar", command=self.iniciar_captura
        )
        self.btn_capturar.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.btn_detener_cap = ttk.Button(
            frame_botones_cap,
            text="⏹ Detener",
            command=self.detener_captura,
            state="disabled",
        )
        self.btn_detener_cap.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        ttk.Separator(panel, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=15
        )

        # === Sección de Entrenamiento ===
        ttk.Label(panel, text="🧠 Entrenamiento", style="Subheader.TLabel").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        ttk.Label(panel, text="Algoritmo:").grid(row=6, column=0, sticky="w", pady=5)
        self.combo_reconocedor = ttk.Combobox(
            panel, values=[t.value for t in TipoReconocedor], state="readonly", width=20
        )
        self.combo_reconocedor.set(TipoReconocedor.LBPH.value)
        self.combo_reconocedor.grid(row=6, column=1, sticky="w", pady=5, padx=(5, 0))

        self.var_mostrar_progreso = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            panel,
            text="Mostrar imágenes durante entrenamiento",
            variable=self.var_mostrar_progreso,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=5)

        self.btn_entrenar = ttk.Button(
            panel, text="🎓 Entrenar Modelo", command=self.iniciar_entrenamiento
        )
        self.btn_entrenar.grid(row=8, column=0, columnspan=2, sticky="ew", pady=10)

    def crear_panel_derecho(self, parent):
        """Crea el panel derecho con reconocimiento y estado."""
        panel = ttk.LabelFrame(parent, text="Reconocimiento y Estado", padding="10")
        panel.grid(row=1, column=1, sticky="nsew", padx=(5, 0), pady=5)

        # === Sección de Reconocimiento ===
        ttk.Label(
            panel, text="🔍 Reconocimiento Facial", style="Subheader.TLabel"
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Label(panel, text="Umbral confianza:").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.spin_umbral = ttk.Spinbox(panel, from_=30, to=150, width=10, increment=5)
        self.spin_umbral.set(70)
        self.spin_umbral.grid(row=1, column=1, sticky="w", pady=5, padx=(5, 0))

        ttk.Label(panel, text="Índice cámara:").grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.spin_camara = ttk.Spinbox(panel, from_=0, to=5, width=10)
        self.spin_camara.set(0)
        self.spin_camara.grid(row=2, column=1, sticky="w", pady=5, padx=(5, 0))

        # Frame para botones de reconocimiento
        frame_botones_rec = ttk.Frame(panel)
        frame_botones_rec.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)
        frame_botones_rec.columnconfigure(0, weight=1)
        frame_botones_rec.columnconfigure(1, weight=1)

        self.btn_reconocer = ttk.Button(
            frame_botones_rec, text="▶ Iniciar", command=self.iniciar_reconocimiento
        )
        self.btn_reconocer.grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.btn_detener_rec = ttk.Button(
            frame_botones_rec,
            text="⏹ Detener",
            command=self.detener_reconocimiento,
            state="disabled",
        )
        self.btn_detener_rec.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        ttk.Separator(panel, orient="horizontal").grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=15
        )

        # === Estado del Sistema ===
        ttk.Label(panel, text="📊 Estado del Sistema", style="Subheader.TLabel").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(0, 10)
        )

        ttk.Label(panel, text="Personas registradas:").grid(
            row=6, column=0, sticky="nw", pady=5
        )

        lista_frame = ttk.Frame(panel)
        lista_frame.grid(row=6, column=1, sticky="nsew", pady=5, padx=(5, 0))

        self.listbox_personas = tk.Listbox(lista_frame, height=5, width=20)
        scrollbar = ttk.Scrollbar(
            lista_frame, orient="vertical", command=self.listbox_personas.yview
        )
        self.listbox_personas.configure(yscrollcommand=scrollbar.set)

        self.listbox_personas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Button(
            panel, text="🔄 Actualizar Lista", command=self.actualizar_lista_personas
        ).grid(row=7, column=0, columnspan=2, sticky="ew", pady=5)

        self.label_modelo = ttk.Label(panel, text="Modelo: No cargado")
        self.label_modelo.grid(row=8, column=0, columnspan=2, sticky="w", pady=5)

    def crear_area_log(self, parent):
        """Crea el área de log en la parte inferior."""
        log_frame = ttk.LabelFrame(parent, text="📝 Registro de Actividad", padding="5")
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))

        parent.rowconfigure(2, weight=1)

        self.text_log = scrolledtext.ScrolledText(
            log_frame, height=10, font=("Consolas", 9), wrap=tk.WORD
        )
        self.text_log.pack(fill="both", expand=True)
        self.text_log.configure(state="disabled")

        ttk.Button(log_frame, text="🗑️ Limpiar Log", command=self.limpiar_log).pack(
            anchor="e", pady=(5, 0)
        )

    def log(self, mensaje: str):
        """Agrega un mensaje al área de log."""
        self.text_log.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.text_log.insert(tk.END, f"[{timestamp}] {mensaje}\n")
        self.text_log.see(tk.END)
        self.text_log.configure(state="disabled")
        self.root.update_idletasks()

    def limpiar_log(self):
        """Limpia el área de log."""
        self.text_log.configure(state="normal")
        self.text_log.delete(1.0, tk.END)
        self.text_log.configure(state="disabled")

    def actualizar_lista_personas(self):
        """Actualiza la lista de personas registradas."""
        self.listbox_personas.delete(0, tk.END)

        if os.path.exists(self.directorio_imagenes):
            personas = sorted(os.listdir(self.directorio_imagenes))
            for persona in personas:
                ruta = os.path.join(self.directorio_imagenes, persona)
                if os.path.isdir(ruta):
                    num_imagenes = len(os.listdir(ruta))
                    self.listbox_personas.insert(
                        tk.END, f"{persona} ({num_imagenes} imgs)"
                    )

            self.log(f"Lista actualizada: {len(personas)} personas registradas.")
        else:
            self.log("No existe el directorio de imágenes.")

        modelo_path = "modeloLBPHFace.xml"
        if os.path.exists(modelo_path):
            self.label_modelo.config(text=f"Modelo: ✅ {modelo_path}")
        else:
            self.label_modelo.config(text="Modelo: ❌ No encontrado")

    # === CAPTURA DE ROSTROS ===
    def iniciar_captura(self):
        """Inicia la captura de rostros en un hilo separado."""
        nombre = self.entry_nombre.get().strip()

        if not nombre:
            messagebox.showwarning("Advertencia", "Por favor ingrese un nombre.")
            return

        if self.captura_activa or self.reconocimiento_activo:
            messagebox.showwarning("Advertencia", "Ya hay una operación en curso.")
            return

        max_imagenes = int(self.spin_max_imagenes.get())
        indice_camara = int(self.spin_camara.get())

        self.captura_activa = True
        self.btn_capturar.configure(state="disabled")
        self.btn_detener_cap.configure(state="normal")

        self.hilo_actual = threading.Thread(
            target=self._capturar_rostros_thread,
            args=(nombre, max_imagenes, indice_camara),
            daemon=True,
        )
        self.hilo_actual.start()

    def detener_captura(self):
        """Detiene la captura de rostros."""
        if self.captura_activa:
            self.captura_activa = False
            self.log("Deteniendo captura...")

    def _capturar_rostros_thread(
        self, nombre: str, max_imagenes: int, indice_camara: int
    ):
        """Hilo de captura de rostros."""
        try:
            ruta_persona = os.path.join(self.directorio_imagenes, nombre)
            if not os.path.exists(ruta_persona):
                os.makedirs(ruta_persona)
                self.log(f"Carpeta creada: {ruta_persona}")

            cap = self._inicializar_camara(indice_camara)
            if cap is None:
                return

            clasificador = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
            )

            contador = 0
            self.log(
                f"Iniciando captura para '{nombre}'. Máximo: {max_imagenes} imágenes."
            )

            while self.captura_activa:
                ret, frame = cap.read()
                if not ret:
                    self.log("Error: No se pudo leer el frame.")
                    break

                frame = imutils.resize(frame, width=640)
                frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_original = frame.copy()

                rostros = clasificador.detectMultiScale(
                    frame_gris, scaleFactor=1.3, minNeighbors=5
                )

                for x, y, w, h in rostros:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    rostro = frame_original[y : y + h, x : x + w]
                    rostro = cv2.resize(
                        rostro, (150, 150), interpolation=cv2.INTER_CUBIC
                    )

                    nombre_archivo = os.path.join(
                        ruta_persona, f"rostro_{contador}.jpg"
                    )
                    cv2.imwrite(nombre_archivo, rostro)
                    contador += 1

                    cv2.putText(
                        frame,
                        f"Capturadas: {contador}/{max_imagenes}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow(f"Captura: {nombre} - ESC para salir", frame)

                if cv2.waitKey(1) == 27 or contador >= max_imagenes:
                    break

            cap.release()
            cv2.destroyAllWindows()

            self.log(f"Captura finalizada. Total: {contador} imágenes de '{nombre}'.")

        except Exception as e:
            self.log(f"Error en captura: {str(e)}")
        finally:
            self.captura_activa = False
            self.root.after(0, self._restaurar_boton_captura)
            self.root.after(0, self.actualizar_lista_personas)

    def _restaurar_boton_captura(self):
        """Restaura los botones de captura."""
        self.btn_capturar.configure(state="normal")
        self.btn_detener_cap.configure(state="disabled")

    # === ENTRENAMIENTO ===
    def iniciar_entrenamiento(self):
        """Inicia el entrenamiento en un hilo separado."""
        if self.captura_activa or self.reconocimiento_activo:
            messagebox.showwarning("Advertencia", "Ya hay una operación en curso.")
            return

        if not os.path.exists(self.directorio_imagenes):
            messagebox.showerror("Error", "No existe el directorio de imágenes.")
            return

        tipo_str = self.combo_reconocedor.get()
        tipo = next(t for t in TipoReconocedor if t.value == tipo_str)
        mostrar_progreso = self.var_mostrar_progreso.get()

        self.btn_entrenar.configure(text="⏳ Entrenando...", state="disabled")

        self.hilo_actual = threading.Thread(
            target=self._entrenar_modelo_thread,
            args=(tipo, mostrar_progreso),
            daemon=True,
        )
        self.hilo_actual.start()

    def _entrenar_modelo_thread(self, tipo: TipoReconocedor, mostrar_progreso: bool):
        """Hilo de entrenamiento."""
        try:
            self.log(f"Iniciando entrenamiento con {tipo.value}...")

            lista_personas = sorted(os.listdir(self.directorio_imagenes))
            lista_personas = [
                p
                for p in lista_personas
                if os.path.isdir(os.path.join(self.directorio_imagenes, p))
            ]

            if not lista_personas:
                self.log("Error: No hay personas registradas.")
                return

            self.log(f"Personas encontradas: {lista_personas}")

            datos_rostros = []
            etiquetas = []

            for etiqueta, nombre in enumerate(lista_personas):
                ruta = os.path.join(self.directorio_imagenes, nombre)
                archivos = os.listdir(ruta)
                self.log(f"Cargando {len(archivos)} imágenes de '{nombre}'...")

                for archivo in archivos:
                    imagen = cv2.imread(
                        os.path.join(ruta, archivo), cv2.IMREAD_GRAYSCALE
                    )
                    if imagen is not None:
                        datos_rostros.append(imagen)
                        etiquetas.append(etiqueta)

                        if mostrar_progreso:
                            cv2.imshow(f"Cargando: {nombre}", imagen)
                            cv2.waitKey(3)

            if mostrar_progreso:
                cv2.destroyAllWindows()

            self.log(f"Total imágenes cargadas: {len(datos_rostros)}")

            reconocedores = {
                TipoReconocedor.EIGEN: cv2.face.EigenFaceRecognizer_create,  # type: ignore
                TipoReconocedor.FISHER: cv2.face.FisherFaceRecognizer_create,  # type: ignore
                TipoReconocedor.LBPH: cv2.face.LBPHFaceRecognizer_create,  # type: ignore
            }

            reconocedor = reconocedores[tipo]()
            reconocedor.train(datos_rostros, np.array(etiquetas))

            nombre_modelo = f"modelo{tipo.value}.xml"
            reconocedor.write(nombre_modelo)

            self.log(f"✅ Modelo guardado: {nombre_modelo}")

        except Exception as e:
            self.log(f"Error en entrenamiento: {str(e)}")
        finally:
            self.root.after(0, self._restaurar_boton_entrenar)
            self.root.after(0, self.actualizar_lista_personas)

    def _restaurar_boton_entrenar(self):
        """Restaura el botón de entrenamiento."""
        self.btn_entrenar.configure(text="🎓 Entrenar Modelo", state="normal")

    # === RECONOCIMIENTO ===
    def iniciar_reconocimiento(self):
        """Inicia el reconocimiento facial."""
        if self.captura_activa or self.reconocimiento_activo:
            messagebox.showwarning("Advertencia", "Ya hay una operación en curso.")
            return

        modelo_path = "modeloLBPHFace.xml"
        if not os.path.exists(modelo_path):
            messagebox.showerror("Error", "No se encontró el modelo. Entrena primero.")
            return

        umbral = float(self.spin_umbral.get())
        indice_camara = int(self.spin_camara.get())

        self.reconocimiento_activo = True
        self.btn_reconocer.configure(state="disabled")
        self.btn_detener_rec.configure(state="normal")

        self.hilo_actual = threading.Thread(
            target=self._reconocer_thread,
            args=(modelo_path, umbral, indice_camara),
            daemon=True,
        )
        self.hilo_actual.start()

    def detener_reconocimiento(self):
        """Detiene el reconocimiento facial."""
        if self.reconocimiento_activo:
            self.reconocimiento_activo = False
            self.log("Deteniendo reconocimiento...")

    def _reconocer_thread(self, modelo_path: str, umbral: float, indice_camara: int):
        """Hilo de reconocimiento."""
        try:
            self.log("Iniciando reconocimiento facial...")

            reconocedor = cv2.face.LBPHFaceRecognizer_create()  # type: ignore
            reconocedor.read(modelo_path)

            lista_personas = sorted(
                [
                    p
                    for p in os.listdir(self.directorio_imagenes)
                    if os.path.isdir(os.path.join(self.directorio_imagenes, p))
                ]
            )
            self.log(f"Personas registradas: {lista_personas}")

            cap = self._inicializar_camara(indice_camara)
            if cap is None:
                return

            clasificador = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
            )

            while self.reconocimiento_activo:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rostros = clasificador.detectMultiScale(
                    frame_gris, scaleFactor=1.3, minNeighbors=5
                )

                for x, y, w, h in rostros:
                    rostro = frame_gris[y : y + h, x : x + w]
                    rostro = cv2.resize(
                        rostro, (150, 150), interpolation=cv2.INTER_CUBIC
                    )

                    etiqueta, confianza = reconocedor.predict(rostro)

                    if confianza < umbral and etiqueta < len(lista_personas):
                        nombre = lista_personas[etiqueta]
                        color = (0, 255, 0)
                    else:
                        nombre = "Desconocido"
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame,
                        nombre,
                        (x, y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Conf: {confianza:.1f}",
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )

                cv2.imshow("Reconocimiento Facial - ESC para salir", frame)

                if cv2.waitKey(1) == 27:
                    break

            cap.release()
            cv2.destroyAllWindows()
            self.log("Reconocimiento finalizado.")

        except Exception as e:
            self.log(f"Error en reconocimiento: {str(e)}")
        finally:
            self.reconocimiento_activo = False
            self.root.after(0, self._restaurar_boton_reconocer)

    def _restaurar_boton_reconocer(self):
        """Restaura los botones de reconocimiento."""
        self.btn_reconocer.configure(state="normal")
        self.btn_detener_rec.configure(state="disabled")

    def _inicializar_camara(self, indice: int):
        """Inicializa la cámara según el sistema operativo."""
        try:
            if sys.platform.startswith("linux"):
                cap = cv2.VideoCapture(indice, cv2.CAP_V4L2)
            elif sys.platform == "win32":
                cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(indice)

            if not cap.isOpened():
                self.log(f"Error: No se pudo abrir la cámara {indice}.")
                return None

            return cap
        except Exception as e:
            self.log(f"Error inicializando cámara: {str(e)}")
            return None


def main():
    """Función principal."""
    root = tk.Tk()
    app = AplicacionReconocimientoFacial(root)
    root.mainloop()


if __name__ == "__main__":
    main()
