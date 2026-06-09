# Smoke Test Manual – Vigilant-Pro

Objetivo:
Verificar el funcionamiento básico del sistema antes de una demostración o despliegue.

## Pruebas realizadas

1. Inicio del sistema
- Ejecutar:
  python Mecs_sentinel.py
- Resultado esperado:
  La interfaz gráfica inicia correctamente.

2. Detección mediante cámara USB
- Seleccionar una cámara USB disponible.
- Resultado esperado:
  El video se visualiza correctamente.

3. Detección mediante cámara RTSP
- Configurar una dirección RTSP válida.
- Resultado esperado:
  El flujo de video se recibe sin errores.

4. Inferencia de IA
- Activar la analítica sobre una cámara.
- Resultado esperado:
  El sistema detecta personas y genera eventos.

5. Generación de evidencias
- Provocar un evento.
- Resultado esperado:
  Se generan snapshots y registros correspondientes.

Resultado:
Si todas las verificaciones anteriores son exitosas, el sistema se considera apto para operación demostrativa.