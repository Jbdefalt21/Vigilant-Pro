# Vigilant Pro

Repositorio oficial:
https://github.com/Jbdefalt21/Vigilant-Pro

Video demostrativo:
https://www.youtube.com/watch?v=eW4B6X5ENaA

## Descripción

Vigilant Pro (MEcs Sentinel) es un sistema inteligente de vigilancia y seguridad perimetral ejecutado completamente de forma local (on-premise), sin dependencia de APIs comerciales ni servicios externos. El sistema integra técnicas de visión por computadora mediante modelos YOLO y seguimiento multiobjeto (ByteTrack) para la detección automática de eventos de interés en tiempo real, incluyendo intrusión en zonas restringidas, permanencia y merodeo, personas corriendo o inmóviles, caídas, objetos abandonados y objetos movidos, además de la detección de armas y elementos de protección personal (EPP).

El sistema fue desarrollado con fines académicos y de investigación para la validación de técnicas de analítica de video aplicadas a entornos institucionales e industriales.

---

## Categoría del concurso

**Categoría:** Analítica de Video

**Equipo:** MEcs Sentinel

**Institución:** Universidad Tecnológica Paso del Norte, Ciudad Juárez, Chihuahua, México.

---

## Tabla de contenidos

1. Requisitos del sistema
2. Instalación
3. Configuración
4. Ejecución
5. Modelos y dataset
6. Cómo probar el sistema
7. Estructura del proyecto
8. Tecnologías utilizadas
9. Métricas principales
10. Limitaciones conocidas
11. Créditos y licencia

---

## Requisitos del sistema

* Sistema operativo: Windows 11 (64 bits).
* Python 3.10.9.
* 16 GB de memoria RAM.
* GPU NVIDIA compatible con CUDA (recomendada para inferencia en tiempo real).
* Cámaras USB o cámaras IP mediante RTSP.

### FFmpeg

FFmpeg se encuentra incluido dentro del proyecto en la carpeta:

```text
ffmpeg-8.1-essentials_build/
```

No es necesario instalarlo por separado ni agregarlo al PATH del sistema.

---

## Instalación

1. Clonar o copiar el proyecto en cualquier ubicación del equipo.

2. Instalar Python 3.10.9.

3. Instalar las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

4. Verificar que los modelos:

```text
models/yolo26n.pt
models/best.pt
```

se encuentren presentes dentro del proyecto.

5. Verificar que la carpeta:

```text
ffmpeg-8.1-essentials_build/
```

esté incluida dentro del directorio del proyecto.

---

## Configuración

* Iniciar sesión utilizando las credenciales del sistema.
* Configurar cámaras USB o cámaras RTSP desde la sección **Cámaras**.
* Configurar zonas y parámetros de inteligencia artificial desde **Configurar IA**.
* Registrar y administrar usuarios desde la sección correspondiente.
* Ajustar parámetros operativos según las necesidades del entorno.

---

## Ejecución

Ejecutar la aplicación mediante:

```bash
python Mecs_sentinel.py
```

---

## Modelos y dataset

### Modelos utilizados

#### yolo26n.pt

Modelo preentrenado utilizado para la detección general de personas y objetos.

#### best.pt

Modelo especializado ajustado (fine-tuning) por el equipo de desarrollo para la detección de armas y elementos de protección personal (EPP), utilizando un dataset mixto curado específicamente para el proyecto.

Los modelos requeridos para la demostración se incluyen dentro del repositorio:

```text
models/yolo26n.pt
models/best.pt
```

### Dataset

El dataset utilizado para el ajuste del modelo especializado no se distribuye dentro del repositorio debido a restricciones de tamaño y privacidad. Su composición, proceso de curación y métricas asociadas se encuentran documentadas en la ficha técnica del proyecto.

---

## Cómo probar el sistema

### Ejecución de pruebas

Las validaciones funcionales pueden ejecutarse siguiendo las instrucciones descritas en:

```text
tests/test_smoke.md
```

Debido a la naturaleza interactiva del sistema y al uso de dispositivos físicos (cámaras USB y RTSP), las pruebas corresponden a **smoke tests manuales** orientados a validar el funcionamiento integral del sistema.

Las pruebas incluyen:

* Inicio del sistema.
* Apertura de cámaras USB.
* Recepción de cámaras RTSP.
* Ejecución de inferencia mediante inteligencia artificial.
* Generación de evidencias y eventos.

---

## Estructura del proyecto

```text
Vigilant-Pro/
│
├── Mecs_sentinel.py
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── tests/
├── scripts/
├── docs/
├── models/
│   ├── yolo26n.pt
│   └── best.pt
├── ffmpeg-8.1-essentials_build/
├── _icons/
└── ...
```

---

## Tecnologías utilizadas

* Python 3.10.9
* OpenCV
* Ultralytics
* PyTorch
* NumPy
* Pillow
* Tkinter
* ByteTrack
* FFmpeg
* PyGrabber
* Matplotlib
* fpdf2
* psutil
* pynvml / nvidia-ml-py

---

## Métricas principales

El sistema fue evaluado mediante pruebas de desempeño y carga ejecutadas sobre una estación de trabajo equipada con GPU NVIDIA RTX 4050 Laptop GPU. Los resultados obtenidos incluyen métricas de utilización de CPU, RAM, GPU, throughput y percentiles de latencia.

El detalle completo de dichas mediciones se encuentra documentado en la ficha técnica y en el Reporte de Pruebas de Carga del proyecto.

---

## Limitaciones conocidas

* La precisión de detección disminuye bajo condiciones adversas de iluminación y calidad de imagen.
* La detección de armas y EPP presentó episodios de alta precisión alternados con episodios de detección deficiente durante las pruebas realizadas.
* La identificación de operadores dentro de zonas definidas depende de la presencia de una persona en la región configurada, sin validación biométrica del individuo.
* El desempeño en tiempo real depende de la capacidad del hardware disponible.

---

## Créditos

**Equipo:** MEcs Sentinel

**Líder:** María Cristina Guevara Neri

**Co-líder:** Uriel Reyes Fraire

**Integrantes:**

* Gerardo Javier Soto Ulloa
* Melchor Mariscal Jesús Noé
* David Rodríguez Gausin

**Institución:**

Universidad Tecnológica Paso del Norte, Ciudad Juárez, Chihuahua, México.

---

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulte el archivo `LICENSE` para más información.
