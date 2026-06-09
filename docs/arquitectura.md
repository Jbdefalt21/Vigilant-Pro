# Arquitectura de Vigilant Pro

## Flujo general del sistema

```text
Cámaras USB / RTSP
        ↓
Captura y procesamiento de video
        ↓
Inferencia mediante YOLO
        ↓
Seguimiento multiobjeto (ByteTrack)
        ↓
Detección y clasificación de eventos
        ↓
Generación de evidencias (snapshots y grabaciones)
        ↓
Registro y auditoría de eventos
        ↓
Interfaz gráfica de usuario
```

## Descripción de módulos

* **Cámaras USB / RTSP:** adquisición de video desde cámaras locales o cámaras IP.
* **Captura y procesamiento:** obtención y preparación de cuadros para su análisis.
* **YOLO:** detección de personas, objetos y elementos de interés.
* **ByteTrack:** seguimiento de objetos detectados conservando identidades temporales.
* **Detección de eventos:** evaluación de reglas para identificar intrusiones, merodeo, caídas, objetos abandonados y otros eventos configurados.
* **Evidencias:** almacenamiento automático de snapshots y grabaciones asociadas a eventos.
* **Registro y auditoría:** persistencia local de eventos y acciones relevantes del sistema.
* **Interfaz gráfica:** monitoreo en tiempo real y configuración operativa del sistema.

```
```
