# Proyecto de Reconocimiento Facial

Sistema de reconocimiento facial usando OpenCV con Python.

## Requisitos

- Python 3.10 o superior
- Cámara web

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd AIProject
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
```

### 3. Activar entorno virtual

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

### 4. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 5. (Solo Linux) Verificar permisos de cámara

```bash
sudo usermod -aG video $USER
# Cerrar sesión y volver a iniciar
```

## Uso

### 1. Capturar rostros

```bash
cd app/scripts
python capturarOpciones.py
```

Ingresa el nombre de la persona y mira a la cámara. Presiona ESC para terminar.

### 2. Entrenar modelo

```bash
python entrenarImagenes.py
```

Esto generará el archivo `modeloLBPHFace.xml`.

### 3. Probar reconocimiento

```bash
python probarImagenes.py
```

Presiona ESC para salir.

## Estructura del proyecto

```
AIProject/
├── app/
│   └── scripts/
│       ├── capturarOpciones.py   # Captura imágenes de rostros
│       ├── entrenarImagenes.py   # Entrena el modelo
│       └── probarImagenes.py     # Prueba el reconocimiento
├── images/                        # Carpeta con rostros (se crea automáticamente)
├── requirements.txt
└── README.md
```

## Dependencias

| Paquete | Descripción |
|---------|-------------|
| `opencv-contrib-python` | OpenCV con módulos extras (reconocimiento facial) |
| `imutils` | Utilidades para procesamiento de imágenes |
| `numpy` | Operaciones numéricas |

## Solución de problemas

### Error: "No se pudo abrir la cámara"

1. Verifica que la cámara esté conectada:
   ```bash
   ls -la /dev/video*
   ```

2. Verifica que tu usuario tenga permisos:
   ```bash
   groups $USER
   ```
   Debe aparecer `video` en la lista.

### Error: "module 'cv2' has no attribute 'face'"

Asegúrate de instalar `opencv-contrib-python` y no `opencv-python`:

```bash
pip uninstall opencv-python -y
pip install opencv-contrib-python
```

## Autor

Juan Diego

## Licencia

MIT