import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import json
import os
import cv2
import hashlib
import ctypes
import threading
import subprocess
import uuid
import string
from datetime import datetime
import numpy as np
import time
import collections
import urllib.parse
from pygrabber.dshow_graph import FilterGraph
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "_icons", "camaralogo.png")
BIG_LOGO_PATH = os.path.join(BASE_DIR, "_icons", "logocam.png")
FFMPEG_BIN = os.path.join(
    BASE_DIR, "ffmpeg-8.1-essentials_build", "ffmpeg-8.1-essentials_build", "bin", "ffmpeg.exe"
)

# =========================
# DATA (PERSISTENTE)
# =========================
DATA_FILE = os.path.join(BASE_DIR, "users.json")
CAMERA_NAMES_FILE = os.path.join(BASE_DIR, "camera_names.json")
ROOT_KEY_FILE = os.path.join(BASE_DIR, "root_key.json")
AUDIT_LOG_FILE = os.path.join(BASE_DIR, "audit_log.json")
RTSP_CAMERAS_FILE = os.path.join(BASE_DIR, "rtsp_cameras.json")
AI_CONFIG_FILE = os.path.join(BASE_DIR, "ai_config.json")
SHIFTS_CONFIG_FILE = os.path.join(BASE_DIR, "shifts_config.json")
EVENTS_LOG_FILE = os.path.join(BASE_DIR, "events_log.json")
REGISTRY_LOG_FILE = os.path.join(BASE_DIR, "registry_log.jsonl")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")

# =========================
# YOLO
# =========================
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolo26n.pt")
TRACKER_CONFIG = os.path.join(BASE_DIR, "bytetrack.yaml")

# =========================
# AI CONFIG
# =========================
def load_ai_config():

    if not os.path.exists(AI_CONFIG_FILE):

        with open(AI_CONFIG_FILE, "w") as f:
            json.dump({}, f)

        return {}

    with open(AI_CONFIG_FILE, "r") as f:
        return json.load(f)

def save_ai_config():

    with open(AI_CONFIG_FILE, "w") as f:
        json.dump(ai_config, f, indent=4)

ai_config = load_ai_config()

# =========================
# SHIFT CONFIG
# =========================
_SHIFTS_DEFAULT = {
    "mañana": {"nombre": "Mañana",  "icono": "☀",  "inicio": "06:00", "fin": "14:00"},
    "tarde":  {"nombre": "Tarde",   "icono": "🌤", "inicio": "14:00", "fin": "22:00"},
    "noche":  {"nombre": "Noche",   "icono": "🌙", "inicio": "22:00", "fin": "06:00"},
}

def load_shifts_config():
    if not os.path.exists(SHIFTS_CONFIG_FILE):
        with open(SHIFTS_CONFIG_FILE, "w") as f:
            json.dump(_SHIFTS_DEFAULT, f, indent=4)
        return {k: dict(v) for k, v in _SHIFTS_DEFAULT.items()}
    try:
        with open(SHIFTS_CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {k: dict(v) for k, v in _SHIFTS_DEFAULT.items()}

def save_shifts_config():
    with open(SHIFTS_CONFIG_FILE, "w") as f:
        json.dump(shifts_config, f, indent=4)

def get_active_shift(cfg=None):
    if cfg is None:
        cfg = shifts_config
    now_min = datetime.now().hour * 60 + datetime.now().minute
    for key, sh in cfg.items():
        try:
            h0, m0 = map(int, sh["inicio"].split(":"))
            h1, m1 = map(int, sh["fin"].split(":"))
            start = h0 * 60 + m0
            end   = h1 * 60 + m1
            if start < end:
                if start <= now_min < end:
                    return key
            else:  # overnight span (e.g., 22:00 - 06:00)
                if now_min >= start or now_min < end:
                    return key
        except Exception:
            pass
    return "noche"

shifts_config = load_shifts_config()
_active_shift  = get_active_shift()
_shift_lbl_ref = [None]   # sidebar label widget reference

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def generate_root_token():
    raw = str(uuid.uuid4()) + "VIGILANT_PRO"
    return hashlib.sha256(raw.encode()).hexdigest()

# =========================
# AUDIT LOGGER
# =========================
def load_audit_log():

    if not os.path.exists(AUDIT_LOG_FILE):

        with open(AUDIT_LOG_FILE, "w") as f:
            json.dump([], f)

        return []

    with open(AUDIT_LOG_FILE, "r") as f:
        return json.load(f)


def save_audit_log(logs):

    with open(AUDIT_LOG_FILE, "w") as f:
        json.dump(logs, f, indent=4)


def register_event(
    user,
    action,
    status="OK",
    details=""
):

    from datetime import datetime

    logs = load_audit_log()

    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user,
        "action": action,
        "status": status,
        "details": details
    }

    logs.append(event)

    # =========================
    # SOLO 30 DÍAS
    # =========================
    logs = logs[-5000:]

    save_audit_log(logs)

# ── SISTEMA DE EVENTOS Y REGISTRO TÉCNICO ────────────────────────────────────

def _ensure_snapshots_dir():
    os.makedirs(os.path.join(SNAPSHOTS_DIR, "analitica"), exist_ok=True)

def _save_event_snapshot(cam_id, frame, event_type, ts_str):
    """Saves AI-annotated event snapshot to snapshots/analitica/ with per-event cooldown."""
    key = (str(cam_id), event_type)
    now = time.time()
    if now - _snap_cooldown_ts.get(key, 0) < _SNAP_COOLDOWN_S:
        return None
    _snap_cooldown_ts[key] = now
    try:
        _ensure_snapshots_dir()
        safe   = event_type.replace(" ", "_").replace("/", "_")[:25]
        fname  = f"snap_c{cam_id}_{safe}_{ts_str}.jpg"
        c_name = _cam_name_for(cam_id)
        ai_path = os.path.join(SNAPSHOTS_DIR, "analitica", fname)
        cv2.imwrite(ai_path, _burn_timestamp(frame, c_name),
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        return ai_path
    except Exception:
        return None

def _cam_name_for(cam_id):
    """Returns display name for a camera ID (USB or RTSP)."""
    n = camera_names.get(str(cam_id), "")
    if n:
        return n
    for c in rtsp_cameras:
        if c.get("id") == cam_id:
            return c.get("name", f"CAM {cam_id}")
    return f"CAM {cam_id}"

def _draw_cam_overlay(frame, cam_name, online=True, cam_id=None):
    """Draws a professional VMS-style OSD overlay onto a COPY of frame (display only)."""
    out = frame.copy()
    h, w = out.shape[:2]
    bar_h = max(40, min(52, h // 7))

    # Semi-transparent dark top bar
    roi = out[0:bar_h, 0:w]
    cv2.addWeighted(roi, 0.30, np.zeros_like(roi), 0.70, 0, roi)
    out[0:bar_h, 0:w] = roi

    font       = cv2.FONT_HERSHEY_SIMPLEX
    scale_name = max(0.38, min(0.52, w / 1400.0))
    scale_sm   = scale_name * 0.72

    # Camera name — top-left
    cv2.putText(out, cam_name.upper(), (8, 16),
                font, scale_name, (255, 255, 255), 1, cv2.LINE_AA)

    # Status dot + text — below name
    dot_color  = (50, 210, 80) if online else (50, 80, 210)
    status_txt = "  EN LINEA" if online else "  SIN SENAL"
    cv2.circle(out, (10, bar_h - 10), 4, dot_color, -1)
    cv2.putText(out, status_txt, (8, bar_h - 5),
                font, scale_sm, dot_color, 1, cv2.LINE_AA)

    # Date / time — top-right
    ts_txt = datetime.now().strftime("%Y-%m-%d  |  %H:%M:%S")
    (tw, _), _ = cv2.getTextSize(ts_txt, font, scale_sm, 1)
    cv2.putText(out, ts_txt, (w - tw - 8, 16),
                font, scale_sm, (180, 200, 220), 1, cv2.LINE_AA)

    # Occupancy counter — bottom-right of top bar (only when AI person counting active)
    if cam_id is not None:
        occ = _person_occupancy.get(str(cam_id))
        if occ is not None:
            cnt = occ.current
            occ_txt = f"Personas: {cnt}"
            occ_color = (0, 220, 130) if cnt > 0 else (120, 120, 120)
            (ow, _), _ = cv2.getTextSize(occ_txt, font, scale_sm, 1)
            cv2.putText(out, occ_txt, (w - ow - 8, bar_h - 5),
                        font, scale_sm, occ_color, 1, cv2.LINE_AA)

    return out

def _burn_timestamp(frame, cam_name):
    """Burns a minimal timestamp bar onto a COPY of frame for saved evidence."""
    out = frame.copy()
    h, w = out.shape[:2]
    bar_h = 22
    roi = out[h - bar_h:h, 0:w]
    cv2.addWeighted(roi, 0.20, np.zeros_like(roi), 0.80, 0, roi)
    out[h - bar_h:h, 0:w] = roi
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(out, f"{cam_name}   {ts}", (6, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
    return out

def _append_registry(entry):
    try:
        with open(REGISTRY_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

def load_registry_log(limit=800):
    if not os.path.exists(REGISTRY_LOG_FILE):
        return []
    try:
        with open(REGISTRY_LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        entries = []
        for line in reversed(lines[-limit:]):
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
        return entries
    except Exception:
        return []

def log_event(cam_id, event_type, severity="INFO", details="",
              track_id=None, conf=None, coords=None, frame=None, frame_num=None):
    now = datetime.now()
    ts_full = now.strftime("%Y-%m-%d %H:%M:%S")
    ts_time = now.strftime("%H:%M")
    snap = None
    if frame is not None and severity in ("ALERTA", "CRÍTICO"):
        ts_file = ts_full.replace(":", "-").replace(" ", "_")
        snap = _save_event_snapshot(cam_id, frame, event_type, ts_file)
    event = {
        "ts": ts_full, "ts_display": ts_time,
        "cam": str(cam_id), "type": event_type,
        "severity": severity, "details": details,
        "track_id": track_id,
        "conf": round(conf, 2) if conf is not None else None,
        "snapshot": snap,
    }
    _events_list.appendleft(event)
    _append_registry({
        "ts": ts_full, "cam": str(cam_id), "type": event_type,
        "severity": severity, "track_id": track_id,
        "conf": round(conf, 2) if conf is not None else None,
        "coords": coords, "frame_num": frame_num,
        "user": current_user or "system", "details": details, "snapshot": snap,
    })
    # Contadores de diagnóstico — solo para detecciones reales (no eventos del sistema)
    if event_type not in SYSTEM_EVENT_KEYWORDS:
        _diag_cam_det_count[str(cam_id)] += 1
        _diag_det_ts.append(time.time())
    if any(kw in event_type for kw in INCIDENT_KEYWORDS):
        try:
            root.after(0, _update_events_badge)
        except Exception:
            pass
    return event

def _confirm_detection(cam_id, key, required=3, window=4.0):
    """Returns True exactly when consecutive detection count hits `required` within `window` s."""
    now = time.time()
    k = (str(cam_id), key)
    e = _detection_confirm.get(k)
    if e is None or (now - e["last_ts"]) > window:
        _detection_confirm[k] = {"count": 1, "last_ts": now}
        return False
    e["count"] += 1
    e["last_ts"] = now
    return e["count"] == required

def _update_track_timeline(cam_id, track_id, zone=None, behavior=None):
    key = (str(cam_id), track_id)
    if key not in _track_timeline:
        _track_timeline[key] = {
            "appeared": datetime.now().strftime("%H:%M:%S"),
            "last_seen": datetime.now().strftime("%H:%M:%S"),
            "zones": [], "behaviors": [],
        }
    e = _track_timeline[key]
    e["last_seen"] = datetime.now().strftime("%H:%M:%S")
    if zone and zone not in e["zones"]:
        e["zones"].append(zone)
    if behavior and behavior not in e["behaviors"]:
        e["behaviors"].append(behavior)

def load_users():
    if not os.path.exists(DATA_FILE):
        default = [
            {"user": "admin", "name": "Administrador del Sistema", "password": hash_password("1234"), "pin": hash_password("1234"), "role": "Administrador", "status": "Activo"},
            {"user": "operador", "name": "Operador de Cámaras", "password": hash_password("1234"), "pin": hash_password("1234"), "role": "Operador", "status": "Activo"},
            {"user": "visualizador", "name": "Usuario de Visualización", "password": hash_password("1234"), "pin": hash_password("1234"), "role": "Visualizador", "status": "Activo"}
        ]
        with open(DATA_FILE, "w") as f:
            json.dump(default, f, indent=4)
        return default

    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_users():
    with open(DATA_FILE, "w") as f:
        json.dump(users_data, f, indent=4)

users_data = load_users()
current_user = None
session_start_time = None
preview_camera = None
preview_label = None
active_preview_id = None
preview_after_id = None
ffmpeg_process = None
current_ffmpeg = None
preview_running = False
notification_frame = None
camera_checkboxes = []

SELECTED_CAMERAS_FILE = os.path.join(BASE_DIR, "selected_cameras.json")

def load_selected_cameras():
    if not os.path.exists(SELECTED_CAMERAS_FILE):
        return []
    try:
        with open(SELECTED_CAMERAS_FILE, "r") as f:
            data = json.load(f)
        # Permitir IDs USB (0-19) e IDs RTSP (≥100)
        return [c for c in data if isinstance(c, int) and (0 <= c < 20 or c >= 100)]
    except Exception:
        return []

def save_selected_cameras():
    with open(SELECTED_CAMERAS_FILE, "w") as f:
        json.dump(selected_cameras, f)

selected_cameras = load_selected_cameras()
notification_after_id = None
_live_cameras   = []   # cv2.VideoCapture handles for live view
_live_after_ids = []   # after() ids for live view loops

# IA threading — doble buffer para no bloquear tkinter
_ai_frame_buffer    = {}   # {cam_id: frame procesado listo}
_ai_frame_buffer_ts = {}   # {cam_id: float} — timestamp del último frame procesado
_ai_raw_buffer      = {}   # {cam_id: frame crudo a procesar}
_ai_thread_active   = {}   # {cam_id: bool}
_ai_lock            = {}   # {cam_id: threading.Lock}
_ai_models_ready    = {}   # {cam_id: threading.Event} — set cuando modelos YOLO están listos
_notification_queue = collections.deque(maxlen=20)  # notificaciones desde threads
_AI_PROCESS_INTERVAL = 0.25  # procesar IA máximo cada 250 ms (4 fps de IA)
_ai_last_process     = {}   # {cam_id: timestamp último proceso}

# ── OPTIMIZACIÓN DE RENDIMIENTO ───────────────────────────────────────────────
_MOVED_OBJ_INTERVAL  = 2.0  # s — mínimo entre invocaciones de _check_moved_objects
_OPERATORS_INTERVAL  = 1.5  # s — mínimo entre el fallback YOLO de operadores
_USB_CAP_WIDTH       = 1280 # resolución de captura USB por defecto (cambiar a 1920 si se necesita)
_USB_CAP_HEIGHT      = 720
_moved_obj_last      = {}   # {cam_id: float} — timestamp último _check_moved_objects
_operators_last      = {}   # {cam_id: float} — timestamp último fallback YOLO operadores
_operators_boxes_cache = {} # {cam_id: list}  — últimos effective_boxes del fallback (evita flickering entre llamadas throttled)

# ── MONITOREO DE RENDIMIENTO (logs [PERF]) ────────────────────────────────────
_perf_cap_ts    = collections.defaultdict(lambda: collections.deque(maxlen=90))
_perf_ai_ts     = collections.defaultdict(lambda: collections.deque(maxlen=30))
_perf_raw_shape = {}   # {cam_id: (h, w)} — resolución del frame crudo en IA
_perf_drops     = collections.defaultdict(int)   # frames descartados por cámara
_perf_last_log  = {}   # {cam_id: float} — timestamp último log [PERF]
_perf_t_gen    = collections.defaultdict(lambda: collections.deque(maxlen=30))  # tiempo YOLO general (s)
_perf_t_sec    = collections.defaultdict(lambda: collections.deque(maxlen=30))  # tiempo YOLO seguridad (s)
_perf_t_ops    = collections.defaultdict(lambda: collections.deque(maxlen=30))  # tiempo operadores (s)
_perf_t_abnd   = collections.defaultdict(lambda: collections.deque(maxlen=30))  # tiempo obj. abandonados (s)
_perf_t_moved  = collections.defaultdict(lambda: collections.deque(maxlen=30))  # tiempo obj. movidos (s)
_perf_t_render = collections.defaultdict(lambda: collections.deque(maxlen=30))  # tiempo render UI (s)
_perf_t_record = collections.defaultdict(lambda: collections.deque(maxlen=30))  # tiempo grabación (s)
_perf_t_total  = collections.defaultdict(lambda: collections.deque(maxlen=30))  # tiempo total process_frame (s)

# Control explícito de IA desde Vista en Vivo
_ai_started     = {}   # {cam_id_str: bool} — IA activada manualmente
_ai_pending_bg  = {}   # {cam_id_str: bool} — capturar fondo al 1er frame
_ai_pending_obj = {}   # {cam_id_str: bool} — escanear objetos al 1er frame

# ── EVENTOS, REGISTRO Y ESTADO ───────────────────────────────────────────────
_events_list       = collections.deque(maxlen=500)   # timeline rápido (más reciente primero)
_detection_confirm = {}   # {(cam_id, key): {"count":int, "last_ts":float}}
_track_timeline    = {}   # {(cam_id, track_id): {appeared, last_seen, zones, behaviors}}
_ai_fps_ts         = collections.defaultdict(lambda: collections.deque(maxlen=30))
_cam_last_frame    = {}   # {cam_id: float} — timestamp último frame recibido

# ── DIAGNÓSTICO Y RENDIMIENTO ─────────────────────────────────────────────────
_diag_ai_start_time = [None]          # float — epoch cuando se activó la IA por 1ª vez
_diag_latency_all   = collections.deque(maxlen=500)   # latencias globales (segundos)
_diag_cam_lat       = collections.defaultdict(lambda: collections.deque(maxlen=50))
_diag_panel_after   = [None]          # after() ID del panel (se cancela al salir)
_diag_mon_active    = [False]         # True = monitoreo de carga activo
_diag_mon_log       = []              # [{ts, cpu, ram, gpu, fps, events, cams}]
_diag_mon_after     = [None]          # after() ID del monitoreo de carga
_diag_cam_det_count = collections.defaultdict(int)              # detecciones reales por cámara (no system events)
_diag_det_ts        = collections.deque(maxlen=2000)            # timestamps de detecciones (para det/min)
_diag_fps_history   = collections.deque(maxlen=5000)            # historial global FPS para avg/min/max

# ── BENCHMARK OFICIAL ─────────────────────────────────────────────────────────
_benchmark_running      = [False]
_benchmark_cancel_flag  = [False]
_benchmark_results      = [None]
_benchmark_panel_after  = [None]
_benchmark_thread       = [None]   # tracks active scenario thread for auto-recovery

# pynvml importado una vez al inicio del módulo (disponible vía torch/ultralytics)
try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _PYNVML_OK = True
except Exception:
    _pynvml  = None
    _PYNVML_OK = False
_watchdog_active   = [False]
_cam_preview_frame = {}   # {cid_str: np.ndarray} — last decoded frame for dashboard
_events_badge_lbl  = [None]  # tk.Label reference for sidebar incident counter
_events_acked      = set()   # {event_key} — acknowledged events

# Keyword sets for classification
INCIDENT_KEYWORDS = (
    "Intrusión", "Objeto Abandonado", "Objeto Movido", "Operador Ausente",
    "Persona Corriendo", "Persona Inmóvil", "Arma", "Zona Roja",
    "Permanencia Excedida", "Detección Seguridad", "Zona Amarilla",
    "Persona Caída",
)
SYSTEM_EVENT_KEYWORDS = (
    "IA Iniciada", "IA Detenida", "Grabación Iniciada", "Grabación Finalizada",
    "Cámara Congelada", "Worker IA Reiniciado", "Cámara Sin Señal",
    "Cámara Conectada", "RTSP Iniciando", "RTSP Conectada", "FFMPEG Error",
    "RTSP Señal Perdida", "Error YOLO", "Persona Detectada", "Objeto Detectado",
    "Salida Zona", "Zona Verde",
)

SEVERITY_COLORS = {
    "INFO":    "#3b82f6",
    "WARNING": "#facc15",
    "ALERTA":  "#f97316",
    "CRÍTICO": "#ef4444",
}

# =========================
# IA / DETECCIÓN
# =========================
yolo_enabled = True

# =========================
# MODELOS IA (carga diferida)
# =========================
_model_general = None
_model_seguridad = None
_track_models_gen = {}   # {cam_id: YOLO instance dedicada para tracking general}
_track_models_sec = {}   # {cam_id: YOLO instance dedicada para tracking seguridad}

def get_model_general():
    global _model_general
    if _model_general is None:
        _model_general = YOLO(os.path.join(BASE_DIR, "models", "yolo26n.pt")).to("cuda")
        print(f"[CUDA] YOLO General -> {next(_model_general.model.parameters()).device}")
    return _model_general

def get_model_seguridad():
    global _model_seguridad
    if _model_seguridad is None:
        _model_seguridad = YOLO(os.path.join(BASE_DIR, "models", "best.pt")).to("cuda")
        print(f"[CUDA] YOLO Seguridad -> {next(_model_seguridad.model.parameters()).device}")
        print('=== CLASES best.pt ===')
        for i, n in _model_seguridad.names.items():
            print(f'  {i}: {n}')
        print('======================')
    return _model_seguridad

def _get_track_gen(cam_id):
    """Dedicated YOLO instance per camera so tracker state never leaks between cameras."""
    if cam_id not in _track_models_gen:
        _track_models_gen[cam_id] = YOLO(os.path.join(BASE_DIR, "models", "yolo26n.pt")).to("cuda")
        print(f"[CUDA] YOLO General cam={cam_id} -> {next(_track_models_gen[cam_id].model.parameters()).device}")
    return _track_models_gen[cam_id]

def _get_track_sec(cam_id):
    if cam_id not in _track_models_sec:
        _track_models_sec[cam_id] = YOLO(os.path.join(BASE_DIR, "models", "best.pt")).to("cuda")
        print(f"[CUDA] YOLO Seguridad cam={cam_id} -> {next(_track_models_sec[cam_id].model.parameters()).device}")
    return _track_models_sec[cam_id]

# =========================
# ALIAS DE CÁMARAS
# =========================
def load_camera_names():

    if not os.path.exists(CAMERA_NAMES_FILE):

        with open(CAMERA_NAMES_FILE, "w") as f:
            json.dump({}, f)

        return {}

    with open(CAMERA_NAMES_FILE, "r") as f:
        return json.load(f)


def save_camera_names():

    with open(CAMERA_NAMES_FILE, "w") as f:
        json.dump(camera_names, f, indent=4)


camera_names = load_camera_names()
# =========================
# RTSP CAMERAS
# =========================
def load_rtsp_cameras():

    if not os.path.exists(RTSP_CAMERAS_FILE):

        with open(RTSP_CAMERAS_FILE, "w") as f:
            json.dump([], f)

        return []

    with open(RTSP_CAMERAS_FILE, "r") as f:
        return json.load(f)


def save_rtsp_cameras(data):

    with open(RTSP_CAMERAS_FILE, "w") as f:
        json.dump(data, f, indent=4)


rtsp_cameras = load_rtsp_cameras()

# =========================
# DETECCIÓN DE CÁMARAS USB
# =========================
def scan_usb_cameras():
    """Lista cámaras USB usando DirectShow (sin abrir cv2 — rápido, no bloquea UI)."""
    cameras = []
    try:
        graph   = FilterGraph()
        devices = graph.get_input_devices()
    except Exception:
        devices = []

    for i, name in enumerate(devices):
        cameras.append({
            "id":     i,
            "name":   name,
            "alias":  camera_names.get(str(i), ""),
            "status": "Funcional",
        })
    return cameras

# =========================
# CYBERPUNK NOTIFICATIONS
# =========================
def show_notification(title, message, color="#22d3ee"):
    import threading as _threading
    if _threading.current_thread() is not _threading.main_thread():
        _notification_queue.append((title, message, color))
        return

    global notification_frame
    global notification_after_id

    # =========================
    # LIMPIAR NOTIFICACIÓN ANTERIOR
    # =========================
    if notification_frame is not None:

        try:
            notification_frame.destroy()
        except:
            pass

    if notification_after_id is not None:

        try:
            root.after_cancel(notification_after_id)
        except:
            pass

    # =========================
    # FRAME PRINCIPAL
    # =========================
    notification_frame = tk.Frame(
        root,
        bg="#050816",
        highlightbackground=color,
        highlightthickness=2
    )

    notification_frame.place(
        relx=0.5,
        y=30,
        anchor="n"
    )

    # =========================
    # TÍTULO
    # =========================
    tk.Label(
        notification_frame,
        text=title,
        fg=color,
        bg="#050816",
        font=("Segoe UI", 12, "bold")
    ).pack(anchor="w", padx=20, pady=(12, 0))

    # =========================
    # MENSAJE
    # =========================
    tk.Label(
        notification_frame,
        text=message,
        fg="#e5e7eb",
        bg="#050816",
        font=("Segoe UI", 10)
    ).pack(anchor="w", padx=20, pady=(2, 12))

    # =========================
    # AUTO CLOSE
    # =========================
    def close_notification():

        global notification_frame

        if notification_frame is not None:

            try:
                notification_frame.destroy()
            except:
                pass

            notification_frame = None

    notification_after_id = root.after(
        3500,
        close_notification
    )

# =========================
# RTSP URL BUILDER
# =========================
def build_rtsp_url(ip, user, password):

    endpoint = "/onvif1"

    # Normalizar primero (unquote) para evitar doble codificación si el usuario
    # pegó credenciales ya codificadas (p.ej. %23 desde VLC → no se vuelve %2523)
    encoded_user     = urllib.parse.quote(urllib.parse.unquote(user),     safe='')
    encoded_password = urllib.parse.quote(urllib.parse.unquote(password), safe='')

    return f"rtsp://{encoded_user}:{encoded_password}@{ip}:554{endpoint}"


# =========================
# TEST RTSP CONNECTION
# =========================
def test_rtsp_connection(rtsp_url):

    cap = cv2.VideoCapture(rtsp_url)

    # =========================
    # VALIDAR APERTURA
    # =========================
    if not cap.isOpened():

        cap.release()
        return False

    # =========================
    # VALIDAR FRAME REAL
    # =========================
    ret, frame = cap.read()

    cap.release()

    return ret

# =========================
# RTSP DIAGNOSTICO
# =========================
def open_rtsp_diagnostico(rtsp_url):
    """Ventana de diagnóstico RTSP: prueba OpenCV, FFmpeg TCP y FFmpeg UDP."""
    import re as _re

    win = tk.Toplevel()
    win.title("Diagnóstico RTSP")
    win.configure(bg="#020617")
    win.geometry("860x660")
    win.resizable(True, True)

    _running = [True]

    def _on_close():
        _running[0] = False
        win.destroy()
    win.protocol("WM_DELETE_WINDOW", _on_close)

    # ── Header ──────────────────────────────────────────────────────────────
    tk.Label(win, text="Diagnóstico RTSP",
             fg="#3b82f6", bg="#020617",
             font=("Segoe UI", 14, "bold")).pack(pady=(16, 2))

    url_safe = _re.sub(r'(rtsp://)([^:]+):([^@]+)@', r'\1***:***@', rtsp_url)
    tk.Label(win, text=url_safe, fg="#64748b", bg="#020617",
             font=("Consolas", 9)).pack(pady=(0, 8))

    # ── Status grid ─────────────────────────────────────────────────────────
    STATUS_DEF = [
        ("opencv",     "OpenCV VideoCapture"),
        ("ffmpeg_tcp", "FFmpeg  TCP"),
        ("ffmpeg_udp", "FFmpeg  UDP"),
        ("frame",      "Primer frame recibido"),
        ("diag",       "Diagnóstico final"),
    ]

    status_labels = {}
    time_labels   = {}

    grid_f = tk.Frame(win, bg="#0b1220", padx=24, pady=10)
    grid_f.pack(fill="x", padx=20)

    for key, label_text in STATUS_DEF:
        row_f = tk.Frame(grid_f, bg="#0b1220")
        row_f.pack(fill="x", pady=2)
        tk.Label(row_f, text=label_text, fg="#94a3b8", bg="#0b1220",
                 width=26, anchor="w",
                 font=("Segoe UI", 10)).pack(side="left")
        lbl_s = tk.Label(row_f, text="Pendiente", fg="#475569", bg="#0b1220",
                         font=("Segoe UI", 10, "bold"), width=32, anchor="w")
        lbl_s.pack(side="left")
        lbl_t = tk.Label(row_f, text="", fg="#475569", bg="#0b1220",
                         font=("Consolas", 9))
        lbl_t.pack(side="left", padx=(4, 0))
        status_labels[key] = lbl_s
        time_labels[key]   = lbl_t

    # ── Log area ────────────────────────────────────────────────────────────
    tk.Label(win, text="Log detallado:", fg="#3b82f6", bg="#020617",
             font=("Segoe UI", 10, "bold"),
             anchor="w").pack(fill="x", padx=24, pady=(10, 2))

    log_outer = tk.Frame(win, bg="#020617")
    log_outer.pack(fill="both", expand=True, padx=20, pady=(0, 4))

    scrollbar = tk.Scrollbar(log_outer)
    scrollbar.pack(side="right", fill="y")
    hscrollbar = tk.Scrollbar(log_outer, orient="horizontal")
    hscrollbar.pack(side="bottom", fill="x")

    log_text = tk.Text(
        log_outer, bg="#020617", fg="#e2e8f0",
        font=("Consolas", 9), relief="flat",
        yscrollcommand=scrollbar.set,
        xscrollcommand=hscrollbar.set,
        state="disabled", wrap="none", height=16
    )
    log_text.pack(fill="both", expand=True)
    scrollbar.config(command=log_text.yview)
    hscrollbar.config(command=log_text.xview)

    log_text.tag_configure("ok",    foreground="#22c55e")
    log_text.tag_configure("error", foreground="#ef4444")
    log_text.tag_configure("warn",  foreground="#f59e0b")
    log_text.tag_configure("info",  foreground="#94a3b8")
    log_text.tag_configure("cmd",   foreground="#60a5fa")
    log_text.tag_configure("head",  foreground="#3b82f6",
                            font=("Consolas", 9, "bold"))

    def _log(msg, tag="info"):
        def _do():
            if not win.winfo_exists():
                return
            log_text.configure(state="normal")
            log_text.insert("end", msg + "\n", tag)
            log_text.configure(state="disabled")
            log_text.see("end")
        win.after(0, _do)

    def _set_status(key, text, state):
        colors = {"ok": "#22c55e", "error": "#ef4444",
                  "warn": "#f59e0b", "pending": "#475569"}
        color = colors.get(state, "#475569")
        def _do():
            if not win.winfo_exists():
                return
            status_labels[key].configure(text=text, fg=color)
        win.after(0, _do)

    def _set_time(key, text):
        def _do():
            if not win.winfo_exists():
                return
            time_labels[key].configure(text=text)
        win.after(0, _do)

    # ── Botón cerrar ─────────────────────────────────────────────────────────
    tk.Button(win, text="Cerrar", bg="#1e293b", fg="#e2e8f0",
              relief="flat", padx=18, pady=6,
              command=_on_close).pack(pady=(4, 14))

    # ── Hilo de diagnóstico ───────────────────────────────────────────────────
    def _run_diag():
        has_frame  = False
        diag_notes = []

        # ── A) OpenCV ────────────────────────────────────────────────────────
        _log("━" * 72, "head")
        _log("  TEST A — OpenCV  cv2.VideoCapture", "head")
        _log("━" * 72, "head")
        _set_status("opencv", "Probando...", "pending")

        try:
            t0  = time.time()
            cap = cv2.VideoCapture(rtsp_url)
            t1  = time.time()
            opened = cap.isOpened()
            _log(f"  isOpened()      : {opened}   ({t1-t0:.2f}s)")

            if opened:
                t2 = time.time()
                ret, frame = cap.read()
                t3 = time.time()
                _log(f"  cap.read() ok   : {ret}   ({t3-t2:.2f}s)")
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    _log(f"  Resolución      : {w}x{h}", "ok")
                    _set_status("opencv", f"OK  {w}x{h}", "ok")
                    _set_time("opencv", f"{t3-t0:.2f}s")
                    has_frame = True
                    _set_status("frame", f"SI  (OpenCV  {t3-t0:.1f}s)", "ok")
                else:
                    _log("  Sin frame — stream abierto pero no entrega imágenes", "warn")
                    _set_status("opencv", "Abierto / sin frame", "warn")
                    _set_time("opencv", f"{t3-t0:.2f}s")
            else:
                _log("  No se pudo abrir el stream", "error")
                _set_status("opencv", "FALLO isOpened", "error")
                _set_time("opencv", f"{t1-t0:.2f}s")
            cap.release()
        except Exception as exc:
            _log(f"  EXCEPCION: {exc}", "error")
            _set_status("opencv", "EXCEPCION", "error")

        # ── B) FFmpeg TCP + UDP ──────────────────────────────────────────────
        for tr in ('tcp', 'udp'):
            if not _running[0]:
                break

            _log("", "info")
            _log("━" * 72, "head")
            _log(f"  TEST B — FFmpeg  transport={tr.upper()}", "head")
            _log("━" * 72, "head")
            key_ff = f"ffmpeg_{tr}"
            _set_status(key_ff, "Probando...", "pending")

            ffcmd = [
                FFMPEG_BIN,
                '-rtsp_transport', tr,
                '-fflags',          'nobuffer',
                '-flags',           'low_delay',
                '-probesize',       '500000',
                '-analyzeduration', '500000',
                '-i',               rtsp_url,
                '-an', '-sn', '-dn',
                '-vf', 'scale=320:180',
                '-f',       'image2pipe',
                '-pix_fmt', 'bgr24',
                '-vcodec',  'rawvideo',
                '-r',       '5',
                '-t',       '8',
                '-'
            ]

            _log("  COMANDO COMPLETO:", "cmd")
            _log("  " + " ".join(ffcmd), "cmd")
            _log("", "info")

            t0   = time.time()
            proc = None
            try:
                proc = subprocess.Popen(
                    ffcmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10**7
                )

                frame_size = 320 * 180 * 3
                got_frame  = [False]
                frame_time = [None]

                def _drain_stdout(p=proc, gf=got_frame, ft=frame_time):
                    try:
                        raw = p.stdout.read(frame_size)
                        if len(raw) == frame_size:
                            gf[0] = True
                            ft[0] = time.time()
                        p.stdout.read()
                    except Exception:
                        pass

                t_out = threading.Thread(target=_drain_stdout, daemon=True)
                t_out.start()

                try:
                    proc.wait(timeout=14)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    _log(f"  TIMEOUT (>14s) — ffmpeg no respondió en red", "error")
                    _set_status(key_ff, "TIMEOUT >14s", "error")
                    _set_time(key_ff, ">14s")
                    diag_notes.append(("error",
                        f"FFmpeg {tr.upper()} no respondió (timeout) — cámara no alcanzable"))
                    t_out.join(timeout=2)
                    continue
                t_out.join(timeout=2)

            except Exception as exc:
                _log(f"  EXCEPCION al lanzar ffmpeg: {exc}", "error")
                _set_status(key_ff, "EXCEPCION", "error")
                continue

            elapsed     = time.time() - t0
            rc          = proc.returncode
            stderr_text = proc.stderr.read().decode('utf-8', errors='replace')

            _log(f"  Return code     : {rc}   elapsed: {elapsed:.2f}s")
            _log("", "info")
            _log("  ── stderr de FFmpeg ─────────────────────────────────────────", "head")
            for line in stderr_text.splitlines():
                ll = line.lower()
                if any(k in ll for k in ('error', 'failed', 'invalid', '401', '403',
                                          'refused', 'timed out', 'no such', 'not found')):
                    _log("  " + line, "error")
                elif any(k in ll for k in ('warning', 'warn')):
                    _log("  " + line, "warn")
                else:
                    _log("  " + line, "info")
            _log("  ─────────────────────────────────────────────────────────────", "head")

            lower = stderr_text.lower()

            if got_frame[0]:
                conn_t = (frame_time[0] - t0) if frame_time[0] else elapsed
                _log(f"  Primer frame recibido en {conn_t:.2f}s", "ok")
                _set_status(key_ff, f"OK  frame {conn_t:.1f}s", "ok")
                _set_time(key_ff, f"{elapsed:.2f}s")
                if not has_frame:
                    has_frame = True
                    _set_status("frame",
                                f"SI  (FFmpeg {tr.upper()}  {conn_t:.1f}s)", "ok")
            else:
                _set_time(key_ff, f"{elapsed:.2f}s")
                if rc != 0:
                    _set_status(key_ff, f"ERROR  rc={rc}", "error")
                    _log(f"  ffmpeg terminó con error rc={rc}", "error")
                else:
                    _set_status(key_ff, "Sin frames (rc=0)", "warn")
                    _log("  ffmpeg terminó OK pero sin frames", "warn")

            # Notas de diagnóstico automático
            if '401' in lower or 'unauthorized' in lower or 'authentication failed' in lower:
                diag_notes.append(("error",
                    "Autenticación fallida (401) — verifica usuario y contraseña"))
                _set_status("diag", "Error 401 Autenticación", "error")
            if '403' in lower or 'forbidden' in lower:
                diag_notes.append(("error",
                    "Acceso denegado (403) — sin permiso al stream"))
            if 'connection refused' in lower:
                diag_notes.append(("error",
                    "Conexión rechazada — puerto 554 cerrado o cámara apagada"))
            if 'timed out' in lower or 'connection timed out' in lower:
                diag_notes.append(("error",
                    "Timeout de conexión — cámara no responde (IP incorrecta o sin red)"))
            if 'no route to host' in lower:
                diag_notes.append(("error",
                    "Sin ruta al host — IP incorrecta o fuera de la red local"))
            if ('not found' in lower or '404' in lower) and ('onvif' in lower or 'stream' in lower):
                diag_notes.append(("warn",
                    "Endpoint RTSP no encontrado — prueba /stream, /ch0, /h264 u otro path"))
            if 'codec' in lower and ('not found' in lower or 'unsupported' in lower):
                diag_notes.append(("warn",
                    "Codec incompatible — cámara puede usar H.265/HEVC (no soportado sin extra build)"))
            if 'invalid data' in lower:
                diag_notes.append(("warn",
                    "Datos inválidos — posible problema de codec o stream corrupto"))

        # ── C) Diagnóstico final ──────────────────────────────────────────────
        _log("", "info")
        _log("━" * 72, "head")
        _log("  DIAGNOSTICO FINAL", "head")
        _log("━" * 72, "head")

        if not diag_notes:
            if has_frame:
                diag_notes.append(("ok",
                    "Stream alcanzable y con frames — conexión establecida correctamente"))
            else:
                diag_notes.append(("warn",
                    "No se recibieron frames — revisar logs completos arriba"))

        for state, note in diag_notes:
            prefix = {"ok": "[OK]   ", "error": "[ERROR]", "warn": "[WARN] "}.get(state, "       ")
            _log(f"  {prefix}  {note}", state)

        if has_frame:
            _set_status("diag", "Stream funcional con frames", "ok")
        elif any(s == "error" for s, _ in diag_notes):
            first_err = next(n for s, n in diag_notes if s == "error")
            _set_status("diag", (first_err[:50] + "…") if len(first_err) > 50 else first_err,
                        "error")
        else:
            _set_status("diag", "Sin frames — revisar log", "warn")

        _log("", "info")
        _log("  Diagnóstico completado.", "ok")

    threading.Thread(target=_run_diag, daemon=True).start()


# =========================
# VALIDATE IP
# =========================
def validate_ip(ip):

    parts = ip.split(".")

    # =========================
    # 4 BLOQUES
    # =========================
    if len(parts) != 4:
        return False

    # =========================
    # VALIDAR CADA BLOQUE
    # =========================
    for part in parts:

        if not part.isdigit():
            return False

        value = int(part)

        if value < 0 or value > 255:
            return False

    return True

# =========================
# HANDLE RTSP PREVIEW
# =========================
def handle_rtsp_preview(
    ip_entry,
    user_entry,
    password_entry,
    preview_label,
    transport,
    save_button
):

    ip = ip_entry.get().strip()
    user = user_entry.get().strip()
    password = password_entry.get().strip()

    # =========================
    # VALIDAR CAMPOS
    # =========================
    if not ip or not user or not password:

        show_notification(
            "CAMPOS INCOMPLETOS",
            "Completa IP, usuario y contraseña.",
            "#f59e0b"
        )

        return

    # =========================
    # VALIDAR IP
    # =========================
    if not validate_ip(ip):

        show_notification(
            "IP INVÁLIDA",
            "Formato esperado: 192.168.1.10",
            "#f59e0b"
        )

        return

    # =========================
    # CONSTRUIR URL RTSP
    # =========================
    rtsp_url = build_rtsp_url(
        ip,
        user,
        password
    )

    show_notification(
        "CONECTANDO",
        "Intentando abrir stream RTSP...",
        "#22d3ee"
    )

    # DEBUG
    print(f"rtsp://{user}:******@{ip}:554")

    # =========================
    # WORKER THREAD
    # =========================
    def worker():

        try:

            root.after(
                0,
                lambda: start_rtsp_preview(
                    rtsp_url,
                    preview_label,
                    transport,
                    save_button
                )
            )

        except Exception as e:

            root.after(
                0,
                lambda: show_notification(
                    "ERROR RTSP",
                    str(e),
                    "#ef4444"
                )
            )

    # =========================
    # START THREAD
    # =========================
    threading.Thread(
        target=worker,
        daemon=True
    ).start()

# =========================
# RTSP PREVIEW (FFMPEG)
# =========================
def start_rtsp_preview(
    rtsp_url, label, transport, save_button,
    retry=False, real_cam_id=None, enable_ai=False
):
    global ffmpeg_process, preview_after_id, current_ffmpeg, preview_running

    _frame_counter = [0]
    stop_preview()

    label.update_idletasks()
    width  = label.winfo_width()
    height = label.winfo_height()
    if width  < 100: width  = label.master.winfo_width()  or 480
    if width  < 100: width  = 480
    if height < 100: height = label.master.winfo_height() or 320
    if height < 100: height = 320

    cmd = [
        FFMPEG_BIN,
        '-rtsp_transport', transport,
        '-fflags', 'nobuffer',
        '-flags', 'low_delay',
        '-probesize', '500000',
        '-analyzeduration', '500000',
        '-i', rtsp_url,
        '-an', '-sn', '-dn',
        '-vf', f'scale={width}:{height}',
        '-f', 'image2pipe',
        '-pix_fmt', 'bgr24',
        '-vcodec', 'rawvideo',
        '-r', '15',
        '-'
    ]

    print(f"[RTSP-PREVIEW] URL       : {rtsp_url}")
    print(f"[RTSP-PREVIEW] TRANSPORTE: {transport}")
    print(f"[RTSP-PREVIEW] COMANDO   : {' '.join(cmd)}")

    _t_launch = time.time()
    try:
        ffmpeg_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**7
        )
        current_ffmpeg = ffmpeg_process
        preview_running = True

        def _print_stderr_preview(proc=ffmpeg_process):
            try:
                for line in proc.stderr:
                    decoded = line.decode('utf-8', errors='replace').rstrip()
                    if decoded:
                        print(f"[RTSP-STDERR] {decoded}")
            except Exception:
                pass
            elapsed = time.time() - _t_launch
            rc = proc.returncode
            print(f"[RTSP-PREVIEW] ffmpeg terminó: rc={rc}  elapsed={elapsed:.2f}s")

        threading.Thread(target=_print_stderr_preview, daemon=True).start()

    except Exception as e:
        print(f"[RTSP-PREVIEW] ERROR al lanzar ffmpeg: {e}")
        show_notification('FFMPEG ERROR', str(e), '#ef4444')
        raise Exception('RTSP ERROR')

    save_button.configure(state='normal', bg='#16a34a', fg='white')

    _frame_queue  = collections.deque(maxlen=2)
    _reader_alive = [True]

    frame_size = width * height * 3

    def _ffmpeg_reader():
        while _reader_alive[0] and preview_running:
            try:
                raw = ffmpeg_process.stdout.read(frame_size)
                if len(raw) != frame_size:
                    _reader_alive[0] = False
                    break
                frame = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
                _frame_queue.append(frame.copy())
            except Exception:
                _reader_alive[0] = False
                break

    threading.Thread(target=_ffmpeg_reader, daemon=True).start()

    def update_frame():
        global preview_after_id, preview_running

        if not preview_running:
            return
        if not label.winfo_exists():
            preview_running = False
            return

        if not _reader_alive[0] or (ffmpeg_process and ffmpeg_process.poll() is not None):
            preview_running = False
            # intentar fallback de transporte antes de reconectar
            fallback = 'tcp' if transport == 'udp' else 'udp'
            if not retry:
                show_notification('RTSP FALLBACK',
                    f'Reintentando con {fallback.upper()}...', '#f59e0b')
                def _do_fallback():
                    if not label.winfo_exists():
                        return
                    try:
                        start_rtsp_preview(rtsp_url, label, fallback,
                                           save_button, retry=True,
                                           real_cam_id=real_cam_id,
                                           enable_ai=enable_ai)
                    except Exception:
                        pass
                label.after(1000, _do_fallback)
                return
            show_notification('RED INTERRUMPIDA', 'Reconectando en 5s...', '#f59e0b')

            def _retry():
                if not label.winfo_exists():
                    return
                try:
                    start_rtsp_preview(rtsp_url, label, transport,
                                       save_button, retry=False,
                                       real_cam_id=real_cam_id,
                                       enable_ai=enable_ai)
                except Exception:
                    label.after(5000, _retry)
            label.after(5000, _retry)
            return

        if _frame_queue:
            frame = _frame_queue.pop()

            actual_id    = str(real_cam_id) if real_cam_id is not None else rtsp_url
            config_check = ai_config.get(actual_id, {})

            if enable_ai and config_check:
                if actual_id not in _ai_lock:
                    _ai_lock[actual_id]          = threading.Lock()
                    _ai_thread_active[actual_id] = True
                    _ai_raw_buffer[actual_id]    = None
                    _ai_frame_buffer[actual_id]  = None
                    threading.Thread(
                        target=_ai_worker, args=(actual_id,), daemon=True).start()

                _frame_counter[0] += 1
                if _ai_raw_buffer.get(actual_id) is None and _frame_counter[0] % 3 == 0:
                    _ai_raw_buffer[actual_id] = frame.copy()

                lock = _ai_lock.get(actual_id)
                if lock:
                    with lock:
                        buf = _ai_frame_buffer.get(actual_id)
                    if buf is not None:
                        try:
                            if buf.shape == frame.shape:
                                frame = buf
                        except Exception:
                            pass

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pw = label.winfo_width()  or 480
                ph = label.winfo_height() or 320
                if pw < 50: pw = 480
                if ph < 50: ph = 320
                frame_rgb = cv2.resize(frame_rgb, (pw, ph))
                img   = Image.fromarray(frame_rgb)
                imgtk = ImageTk.PhotoImage(image=img)
                label.imgtk = imgtk
                label.image = imgtk
                label.configure(image=imgtk)
            except Exception:
                pass

        preview_after_id = label.after(66, update_frame)

    update_frame()

# =========================
# WATCHDOGS OVERLAYS
# =========================
def glow_rectangle(frame, x1, y1, x2, y2, color):

    overlay = frame.copy()

    for i in range(1, 3):

        cv2.rectangle(
            overlay,
            (x1-i, y1-i),
            (x2+i, y2+i),
            color,
            1
        )

    cv2.addWeighted(
        overlay,
        0.3,
        frame,
        0.7,
        0,
        frame
    )


def caja_watchdogs(frame, x1, y1, x2, y2):

    g = 1
    e = 20

    color = (255,255,255)

    cv2.line(frame, (x1, y1), (x1+e, y1), color, g)
    cv2.line(frame, (x1, y1), (x1, y1+e), color, g)

    cv2.line(frame, (x2, y1), (x2-e, y1), color, g)
    cv2.line(frame, (x2, y1), (x2, y1+e), color, g)

    cv2.line(frame, (x1, y2), (x1+e, y2), color, g)
    cv2.line(frame, (x1, y2), (x1, y2-e), color, g)

    cv2.line(frame, (x2, y2), (x2-e, y2), color, g)
    cv2.line(frame, (x2, y2), (x2, y2-e), color, g)


def crosshair(frame, x1, y1, x2, y2):

    cx = (x1+x2)//2
    cy = (y1+y2)//2

    cv2.line(
        frame,
        (cx-6, cy),
        (cx+6, cy),
        (255,255,255),
        1
    )

    cv2.line(
        frame,
        (cx, cy-6),
        (cx, cy+6),
        (255,255,255),
        1
    )

# =========================
# AI TRACKING INFRASTRUCTURE
# =========================
_tracker_history     = collections.defaultdict(lambda: collections.deque(maxlen=30))
_event_registry      = {}
_zone_entry_times    = {}
_zone_persons_inside = {}   # {(cam_id, zone_key): set of track_ids currently inside}
_zone_person_ids     = {}   # {(cam_id, zone_key): {track_id: zone_id_str}}
_zone_id_counters    = {}   # {(cam_id, zone_key): int}
_ZONE_PREFIXES       = {"verde": "V", "amarilla": "A", "roja": "R"}
_fall_suspects       = {}   # {(cam_id, track_id): {"since": float, "confirmed": bool, "cx": int, "cy": int}}

# =========================
# OCCUPANCY / PERSON COUNTER
# =========================
# Independent of ByteTrack IDs — uses centroid proximity to survive ID reassignments.

_OCCUPY_MATCH_RADIUS  = 120   # px: max centroid distance to consider same logical person
_OCCUPY_LOST_TIMEOUT  = 10.0  # s:  grace window before counting person as departed

class _PersonOccupancy:
    """Spatial proximity-based person counter.

    Survives ByteTrack ID changes for stationary/slow-moving people:
    if a detection appears within _OCCUPY_MATCH_RADIUS of a known logical person
    (or a recently-lost one) it is re-associated instead of creating a new entry.
    """
    __slots__ = ("active", "lost", "max_count", "entries", "exits", "_nxt")

    def __init__(self):
        self.active    = {}  # {lid: {"cx":, "cy":, "ts":, "tid":}}
        self.lost      = {}  # {lid: {"cx":, "cy":, "ts_lost":}}
        self.max_count = 0
        self.entries   = 0
        self.exits     = 0
        self._nxt      = 0

    @property
    def current(self):
        return len(self.active)

    def _nearest(self, cx, cy, pool, radius):
        best_lid, best_d = None, float("inf")
        for lid, p in pool.items():
            d = ((p["cx"] - cx) ** 2 + (p["cy"] - cy) ** 2) ** 0.5
            if d < radius and d < best_d:
                best_lid, best_d = lid, d
        return best_lid

    def update(self, now, detections):
        """detections = [(track_id, cx, cy), ...]  — one frame.
        Returns current logical person count."""
        seen_lids = set()

        for tid, cx, cy in detections:
            # 1. Match against currently active logical persons
            lid = self._nearest(cx, cy, self.active, _OCCUPY_MATCH_RADIUS)
            if lid is not None:
                self.active[lid].update({"cx": cx, "cy": cy, "ts": now, "tid": tid})
                seen_lids.add(lid)
                continue
            # 2. Match against recently-lost pool (ID change / brief occlusion)
            lid = self._nearest(cx, cy, self.lost, _OCCUPY_MATCH_RADIUS * 1.5)
            if lid is not None:
                self.active[lid] = {"cx": cx, "cy": cy, "ts": now, "tid": tid}
                del self.lost[lid]
                seen_lids.add(lid)
                continue
            # 3. Genuinely new person
            lid = self._nxt
            self._nxt += 1
            self.active[lid] = {"cx": cx, "cy": cy, "ts": now, "tid": tid}
            seen_lids.add(lid)
            self.entries += 1
            if self.current > self.max_count:
                self.max_count = self.current

        # 4. Active persons not seen this frame → move to lost pool
        for lid in [l for l in self.active if l not in seen_lids]:
            p = self.active.pop(lid)
            self.lost[lid] = {"cx": p["cx"], "cy": p["cy"], "ts_lost": now}

        # 5. Expire lost persons that exceeded grace window → confirmed exit
        for lid in [l for l, p in self.lost.items()
                    if now - p["ts_lost"] > _OCCUPY_LOST_TIMEOUT]:
            del self.lost[lid]
            self.exits += 1

        return self.current

_person_occupancy = {}  # {cam_id: _PersonOccupancy}

def _get_occupancy(cam_id):
    if cam_id not in _person_occupancy:
        _person_occupancy[cam_id] = _PersonOccupancy()
    return _person_occupancy[cam_id]

# =========================
# SMART OBJECT TRACKING
# =========================
# Sistema 1 — Objeto Abandonado (comparación con fondo de escena)
_background_frame = {}   # {cam_id: frame capturado como fondo}
_background_time  = {}   # {cam_id: timestamp de captura}
_foreign_objects    = {}   # {cam_id: lista de dicts con objetos extraños confirmados}
_foreign_candidates = {}   # {cam_id: [{x1,y1,x2,y2,first_seen}]} regiones aún no confirmadas
_foreign_yolo_cache = {}   # {cam_id: list[region]} — última pasada YOLO-snapshot (Rama A)

# Sistema 2 — Objeto Movido (vigilancia activa de objetos registrados)
_watched_objects   = {}  # {cam_id: {wid: {class, orig_cx, orig_cy, x1,y1,x2,y2, time}}}
_moved_alerts      = {}  # {cam_id: {wid: {move_time, dist_px, class, cur_x1..}}}
_missing_objects   = {}  # {cam_id: {wid: {miss_time, class, last_x1,y1,x2,y2, source}}}
_object_trail_hist = collections.defaultdict(lambda: collections.deque(maxlen=50))

# ── Parámetros configurables de detección ────────────────────────────────────
# Fracción mínima del cuerpo del operador que debe estar dentro de la zona
# para considerarlo PRESENTE. Editable sin tocar la lógica.
OPERATOR_ZONE_OVERLAP_THRESHOLD = 0.20

# Segundos que un objeto nuevo debe permanecer inmóvil para confirmar abandono.
ABANDONED_CONFIRM_SECONDS = 10.0

# Segundos que un objeto vigilado debe estar ausente para confirmar que fue movido.
MOVED_MISSING_CONFIRM_SECONDS = 10.0

# Sistema de grabación
_recordings_dir    = {}   # {cam_id: str}
_recording_enabled = {}   # {cam_id: bool}
_record_mode       = "continuous"   # "continuous" | "event"  — modo global (UI settings)
_cam_record_mode   = {}             # {cam_id: str} — modo por cámara (sobrescribe global)
_monitoring_mode   = [False]        # True cuando está activo el modo monitoreo profesional
# Pre-event buffer: almacena bytes JPEG (no numpy) — ~15-20x menos RAM que frames raw.
# maxlen por defecto = 450 (30s × 15fps); se reemplaza con el valor correcto en _init_ai_for_cam.
_pre_event_buffer  = collections.defaultdict(lambda: collections.deque(maxlen=450))
_post_event_frames = {}   # {cam_id: int} frames restantes post-evento
_event_active      = {}   # {cam_id: bool}
_ai_writers        = {}   # {cam_id: cv2.VideoWriter} — analítica IA (analitica/)
_rec_pre_s         = {}   # {cam_id: int} — segundos de pre-buffer configurados por el usuario
_rec_post_s        = {}   # {cam_id: int} — segundos de post-buffer configurados por el usuario
_snap_cooldown_ts  = {}   # {(cam_id, event_type): float} — último snapshot por evento/cámara
_SNAP_COOLDOWN_S   = 30   # segundos mínimos entre snapshots del mismo evento en la misma cámara
_display_mode      = ["ia"]  # "normal" | "ia" — selector de visualización en vivo

_GENERAL_KEYS  = ["Persona", "Mochila", "Caja", "No Celular"]
_SECURITY_KEYS = [
    "Arma Blanca", "Arma Corta", "Arma Larga",
    "Casco Seguridad", "Bata Industrial", "Botas Seguridad",
    "Cubrebocas Seguridad", "Lentes Seguridad",
    "Tapones Auditivos", "Audífono Inalámbrico",
    "Extintor", "Carro de Carga"
]
_TRACKING_KEYS = ["Persona Corriendo", "Persona Inmóvil", "Objeto Abandonado", "Objeto Movido"]

def _has_smart_zones(config):
    return any(config.get(f"smart_{z}_zone") for z in ("verde", "amarilla", "roja"))

def _need_general(config):
    return (any(config.get(k) for k in _GENERAL_KEYS)
            or _has_smart_zones(config)
            or bool(config.get("operators")))

def _need_security(config):
    return any(config.get(k) for k in _SECURITY_KEYS)

def _need_tracking(config):
    return (any(config.get(k) for k in _TRACKING_KEYS)
            or _has_smart_zones(config)
            or bool(config.get("operators")))

def _general_class_ok(config, class_name):
    cl = class_name.lower()
    if cl == "person":
        return bool(config.get("Persona"))
    if cl in ("backpack", "handbag", "bag"):
        return bool(config.get("Mochila"))
    if cl in ("suitcase", "box"):
        return bool(config.get("Caja"))
    if cl == "cell phone":
        return bool(config.get("No Celular"))
    return False

def _security_class_ok(config, class_name):
    cl = class_name.lower().strip()

    # ARMAS
    if any(x in cl for x in ("arma blanca", "cuchillo", "navaja", "blade")):
        return bool(config.get("Arma Blanca"))
    if any(x in cl for x in ("arma corta", "pistola", "gun", "handgun")):
        return bool(config.get("Arma Corta"))
    if any(x in cl for x in ("arma larga", "rifle", "escopeta", "longgun")):
        return bool(config.get("Arma Larga"))

    # EPP
    if any(x in cl for x in ("casco", "helmet", "hard hat")):
        return bool(config.get("Casco Seguridad"))
    if any(x in cl for x in ("bata", "bata industrial", "lab coat")):
        return bool(config.get("Bata Industrial"))
    if any(x in cl for x in ("bota", "boot", "botas de seguridad")):
        return bool(config.get("Botas Seguridad"))
    if any(x in cl for x in ("cubrebocas", "mascarilla", "mask", "cubrebocas_seguridad")):
        return bool(config.get("Cubrebocas Seguridad"))
    if any(x in cl for x in ("lente", "gafas", "lentes de seguridad", "goggle")):
        return bool(config.get("Lentes Seguridad"))
    if any(x in cl for x in ("tapon", "tapones", "tapones auditivos", "earplug")):
        return bool(config.get("Tapones Auditivos"))
    if any(x in cl for x in ("aud_wireless", "audifonos", "audifono", "wireless", "auricular")):
        return bool(config.get("Audífono Inalámbrico"))

    # OBJETOS
    if any(x in cl for x in ("extintor", "extinguisher")):
        return bool(config.get("Extintor"))
    if any(x in cl for x in ("carro", "carros de carga", "cart", "trolley")):
        return bool(config.get("Carro de Carga"))
    if any(x in cl for x in ("mochila", "backpack", "handbag", "bag")):
        return bool(config.get("Mochila"))
    if any(x in cl for x in ("caja", "box", "suitcase")):
        return bool(config.get("Caja"))

    return False

def _update_history(cam_id, track_id, cx, cy):
    _tracker_history[(cam_id, track_id)].append((time.time(), cx, cy))

def _is_running(cam_id, track_id, threshold=60.0):
    hist = _tracker_history.get((cam_id, track_id))
    if not hist or len(hist) < 5:
        return False
    pts = list(hist)[-5:]
    dt = pts[-1][0] - pts[0][0]
    if dt <= 0:
        return False
    dx = pts[-1][1] - pts[0][1]
    dy = pts[-1][2] - pts[0][2]
    return ((dx**2 + dy**2) ** 0.5 / dt) > threshold

def _is_immobile(cam_id, track_id, min_pts=60, spread=15):
    hist = _tracker_history.get((cam_id, track_id))
    if not hist or len(hist) < min_pts:
        return False
    pts = list(hist)
    xs = [p[1] for p in pts]; ys = [p[2] for p in pts]
    return ((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2) ** 0.5 < spread

_FALL_RATIO_THRESHOLDS = {"baja": 0.50, "media": 0.65, "alta": 0.80}

def _get_fall_ratio_threshold(config):
    sens = str(config.get("fall_sensitivity") or "media").lower()
    return _FALL_RATIO_THRESHOLDS.get(sens, 0.65)

def _is_fallen_posture(x1, y1, x2, y2, threshold):
    """Returns True when the bounding box is more horizontal than the threshold allows."""
    bh = max(1, y2 - y1)
    bw = max(1, x2 - x1)
    return (bh / bw) < threshold

def _check_fall(cam_id, config, track_id, x1, y1, x2, y2, cx, cy, frame):
    """
    Core fall-detection state machine called once per person per frame.
    Returns True if a confirmed fall alert was just fired this frame.
    """
    if not config.get("Persona Caída") and not config.get("Caídas"):
        # Clean up any stale state if rule is toggled off
        _fall_suspects.pop((cam_id, track_id), None)
        return False

    threshold    = _get_fall_ratio_threshold(config)
    confirm_secs = float(config.get("fall_confirm_secs") or 3)
    key          = (cam_id, track_id)
    now          = time.time()
    horizontal   = _is_fallen_posture(x1, y1, x2, y2, threshold)

    if not horizontal:
        # Person stood up — clear suspect entry and log recovery if was confirmed
        suspect = _fall_suspects.pop(key, None)
        if suspect and suspect.get("confirmed"):
            duration = now - suspect["since"]
            if _throttle(cam_id, f"fall_recovery_{track_id}", 20):
                log_event(cam_id, "Persona Recuperada", "INFO",
                          f"ID:{track_id} recuperó postura vertical — duración en suelo: {duration:.0f}s",
                          track_id=track_id)
        return False

    if key not in _fall_suspects:
        _fall_suspects[key] = {"since": now, "confirmed": False, "cx": cx, "cy": cy}
        return False

    suspect = _fall_suspects[key]

    # Verify the centroid hasn't moved significantly (distinguishes fall from crouching/walking)
    dist = ((cx - suspect["cx"])**2 + (cy - suspect["cy"])**2) ** 0.5
    if dist > 50:
        # Significant movement → reset timer (person is still active/moving)
        suspect["since"] = now
        suspect["cx"]    = cx
        suspect["cy"]    = cy
        suspect["confirmed"] = False
        return False

    elapsed = now - suspect["since"]

    # Draw warning overlay while in suspicion period (before confirmation)
    if not suspect["confirmed"] and elapsed >= 0.5:
        bh = max(1, y2 - y1)
        bw = max(1, x2 - x1)
        ratio = bh / bw
        pct = min(1.0, elapsed / confirm_secs)
        bar_w = int((x2 - x1) * pct)
        cv2.rectangle(frame, (x1, y2 + 3), (x1 + bar_w, y2 + 7), (0, 165, 255), -1)

    if elapsed >= confirm_secs and not suspect["confirmed"]:
        suspect["confirmed"] = True
        duration = elapsed

        # Notify and log
        if _throttle(cam_id, f"fall_{track_id}", 60):
            show_notification("ALERTA", f"⚠ Persona caída ID:{track_id}", "#ef4444")
            log_event(cam_id, "Persona Caída", "ALERTA",
                      f"ID:{track_id} caído — {duration:.0f}s en suelo — cámara {cam_id}",
                      track_id=track_id, frame=frame)
            _update_track_timeline(cam_id, track_id, behavior="caída")
        return True

    return False

def _draw_fall_overlay(frame, track_id, x1, y1, x2, y2):
    """Draws the confirmed-fall bounding box and warning label."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    lines  = ["⚠ PERSONA CAIDA", f"ID: {track_id}"]
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 0.5
    thick  = 1
    widths = [cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines]
    bg_w   = max(widths) + 14
    bg_h   = len(lines) * 18 + 8
    y_top  = max(y1 - bg_h - 4, 2)
    ov     = frame.copy()
    cv2.rectangle(ov, (x1, y_top), (x1 + bg_w, y_top + bg_h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.70, frame, 0.30, 0, frame)
    for i, line in enumerate(lines):
        lc = (0, 0, 255) if i == 0 else (220, 220, 220)
        cv2.putText(frame, line, (x1 + 7, y_top + 6 + (i + 1) * 16),
                    font, scale, lc, thick, cv2.LINE_AA)

def _throttle(cam_id, key, cooldown=15.0):
    reg_key = (cam_id, key)
    now = time.time()
    if now - _event_registry.get(reg_key, 0) >= cooldown:
        _event_registry[reg_key] = now
        return True
    return False

def _point_in_zone(cx, cy, zx1, zy1, zx2, zy2):
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

def _parse_shift_minutes(time_str):
    if not time_str:
        return None
    try:
        parts = time_str.strip().split()
        h, m = map(int, parts[0].split(":"))
        if parts[1] == "PM" and h != 12:
            h += 12
        elif parts[1] == "AM" and h == 12:
            h = 0
        return h * 60 + m
    except Exception:
        return None


_ZONE_COLORS_BGR = {
    "verde":    (80, 200, 0),
    "amarilla": (0, 200, 240),
    "roja":     (0, 0, 255),
}
_OP_COLORS_BGR = [
    (0, 180, 255), (0, 255, 140), (200, 0, 255),
    (255, 180, 0), (255, 80, 0),  (80, 255, 200),
]

def _render_smart_zones(frame, config, h, w):
    for zone_key, color in _ZONE_COLORS_BGR.items():
        z = config.get(f"smart_{zone_key}_zone")
        if not z:
            continue
        zx1 = int(z["x1"]*w); zy1 = int(z["y1"]*h)
        zx2 = int(z["x2"]*w); zy2 = int(z["y2"]*h)
        overlay = frame.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), color, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 1)
        cv2.putText(frame, zone_key.upper(), (zx1+4, zy1+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

def _render_operator_zones(frame, frame_for_detection, config, cam_id, h, w, persons,
                           person_boxes=None):
    """Dibuja zonas de operadores y evalúa presencia por overlap de área (≥ OPERATOR_ZONE_OVERLAP_THRESHOLD)."""
    ops = config.get("operators", [])
    if not isinstance(ops, list):
        return
    now_min = datetime.now().hour * 60 + datetime.now().minute

    # effective_boxes: lista de (track_id, x1, y1, x2, y2) con coordenadas reales del bounding box.
    # Primero usamos los boxes que llegaron del pipeline principal.
    effective_boxes = list(person_boxes) if person_boxes else []

    # Fallback autónomo: si el pipeline no corrió detección general pero hay operadores configurados,
    # ejecutar un predict rápido a baja resolución para obtener boxes completos.
    # Throttleado a _OPERATORS_INTERVAL para no lanzar YOLO extra en cada frame de IA.
    # Cuando el throttle bloquea, se usan los boxes cacheados del último YOLO para evitar
    # que el operador aparezca "ausente" entre llamadas y dispare falsas alertas.
    _now_op = time.time()
    if not effective_boxes and config.get('operators') and not _need_general(config):
        if _now_op - _operators_last.get(cam_id, 0) >= _OPERATORS_INTERVAL:
            _operators_last[cam_id] = _now_op
            try:
                model = get_model_general()
                h_f, w_f = frame_for_detection.shape[:2]
                scale_f = 320 / max(w_f, h_f)
                mini = cv2.resize(frame_for_detection,
                                  (int(w_f * scale_f), int(h_f * scale_f)))
                results = model.predict(source=mini, verbose=False, classes=[0], conf=0.20, device=0)
                if results and results[0].boxes is not None:
                    for box in results[0].boxes:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        bx1 = int(bx1 / scale_f); by1 = int(by1 / scale_f)
                        bx2 = int(bx2 / scale_f); by2 = int(by2 / scale_f)
                        effective_boxes.append((-1, bx1, by1, bx2, by2))
            except Exception:
                pass
            _operators_boxes_cache[cam_id] = list(effective_boxes)
        else:
            effective_boxes = _operators_boxes_cache.get(cam_id, [])

    shift_auto = config.get("shift_auto", False)

    for idx in range(len(ops)):
        z = config.get(f"op_{idx}_zone")
        if not z:
            continue
        # Saltar si la gestión automática de turno está activa y el turno no coincide
        if shift_auto:
            op_turno = config.get(f"op_{idx}_turno", "")
            if op_turno and op_turno != _active_shift:
                continue
        llegada = _parse_shift_minutes(config.get(f"op_{idx}_llegada", ""))
        salida  = _parse_shift_minutes(config.get(f"op_{idx}_salida",  ""))
        if llegada is not None and salida is not None:
            in_shift = (llegada <= now_min <= salida) if llegada <= salida else (now_min >= llegada or now_min <= salida)
            if not in_shift:
                continue

        color       = _OP_COLORS_BGR[idx % len(_OP_COLORS_BGR)]
        zx1         = int(z["x1"]*w); zy1 = int(z["y1"]*h)
        zx2         = int(z["x2"]*w); zy2 = int(z["y2"]*h)
        overlay     = frame.copy()
        cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), color, -1)
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 2)
        op_name     = config.get(f"op_{idx}_nombre", "") or f"Op.{idx+1}"
        absence_min = float(config.get(f"op_{idx}_ausencia") or 5)
        llegada_str = config.get(f"op_{idx}_llegada", "--:--") or "--:--"
        salida_str  = config.get(f"op_{idx}_salida",  "--:--") or "--:--"

        # ── Evaluación de presencia por overlap de área ──────────────────────
        # Se calcula la fracción del bounding box de la persona que cae dentro
        # de la zona. Si ≥ OPERATOR_ZONE_OVERLAP_THRESHOLD → PRESENTE.
        present      = False
        best_tid     = -1
        best_overlap = 0.0
        for tid, px1, py1, px2, py2 in effective_boxes:
            p_area = max(1, (px2 - px1) * (py2 - py1))
            ix1 = max(px1, zx1); iy1 = max(py1, zy1)
            ix2 = min(px2, zx2); iy2 = min(py2, zy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            overlap = inter / p_area
            if overlap > best_overlap:
                best_overlap = overlap
                best_tid     = tid
            if overlap >= OPERATOR_ZONE_OVERLAP_THRESHOLD:
                present = True
                break

        # Log de presencia (throttled para no saturar)
        if _throttle(cam_id, f"op_log_{idx}", 5):
            estado_txt = "PRESENTE" if present else "FUERA"
            print(f"[OPERADOR] #{idx+1} {op_name} | cam={cam_id} | estado={estado_txt} | "
                  f"overlap={best_overlap*100:.0f}% | bt_id={best_tid} | personas={len(effective_boxes)}")
            log_event(cam_id, f"Operador #{idx+1} — {op_name}", "INFO",
                      f"ID ByteTrack:{best_tid}  Overlap:{best_overlap*100:.0f}%  Estado:{estado_txt}")

        absent_key = (cam_id, "op_absent_start", idx)
        if present:
            _event_registry.pop(absent_key, None)
            lines      = [
                f"{op_name.upper()} - DENTRO",
                f"Overlap:{best_overlap*100:.0f}%  Entrada:{llegada_str}",
            ]
            text_color = color
        else:
            if absent_key not in _event_registry:
                _event_registry[absent_key] = time.time()
            elapsed_sec = time.time() - _event_registry[absent_key]
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_sec))
            max_str     = time.strftime("%H:%M:%S", time.gmtime(absence_min * 60))
            if elapsed_sec > absence_min * 60:
                lines      = [
                    f"{op_name.upper()} - FUERA DE ESTACION",
                    f"ALERTA AUSENCIA: {elapsed_str}",
                    f"Limite: {max_str}",
                ]
                text_color = (0, 0, 255)
                if _throttle(cam_id, f"op_alert_{cam_id}_{idx}", absence_min * 60):
                    show_notification("OPERADOR AUSENTE", f"{op_name} fuera de zona", "#ef4444")
                    _trigger_event_recording(cam_id, 'Operador Ausente', frame)
            else:
                lines      = [
                    f"{op_name.upper()} - FUERA",
                    f"Ausencia: {elapsed_str} / Max: {max_str}",
                    f"Entrada: {llegada_str}  Salida: {salida_str}",
                ]
                text_color = (0, 200, 255)
        for line_idx, line in enumerate(lines):
            cv2.putText(frame, line, (zx1+6, zy1+20+(line_idx*18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, text_color, 1)

def _fmt_elapsed(seconds):
    s = int(seconds)
    m, sec = divmod(s, 60)
    h, m   = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"

def _get_zone_id(cam_id, zone_key, track_id):
    """Returns (or creates) the zone-specific display ID for a track, e.g. 'A-3'."""
    key = (str(cam_id), zone_key)
    if key not in _zone_person_ids:
        _zone_person_ids[key] = {}
    if key not in _zone_id_counters:
        _zone_id_counters[key] = 0
    if track_id not in _zone_person_ids[key]:
        _zone_id_counters[key] += 1
        prefix = _ZONE_PREFIXES.get(zone_key, "Z")
        _zone_person_ids[key][track_id] = f"{prefix}-{_zone_id_counters[key]}"
    return _zone_person_ids[key][track_id]

def _release_zone_id(cam_id, zone_key, track_id):
    """Removes the zone-specific ID when a track leaves the zone."""
    key = (str(cam_id), zone_key)
    if key in _zone_person_ids:
        _zone_person_ids[key].pop(track_id, None)

def _draw_zone_person_label(frame, cam_id, zone_key, tid, cx, cy, entry_ts):
    """Draws zone ID and elapsed timer above the person's centroid."""
    zone_id = _get_zone_id(cam_id, zone_key, tid)
    color   = _ZONE_COLORS_BGR.get(zone_key, (200, 200, 200))
    lines   = [zone_id]
    if zone_key == "amarilla" and entry_ts is not None:
        lines.append(_fmt_elapsed(time.time() - entry_ts))
    elif zone_key == "roja":
        lines.append("RESTRINGIDO")
    font   = cv2.FONT_HERSHEY_SIMPLEX
    scale  = 0.42
    thick  = 1
    widths = [cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines]
    bg_w   = max(widths) + 10
    total_h = len(lines) * 16 + 6
    y_top   = max(cy - 16 - total_h, 2)
    ov = frame.copy()
    cv2.rectangle(ov, (cx - bg_w // 2, y_top),
                  (cx + bg_w // 2, y_top + total_h), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    for i, line in enumerate(lines):
        tw = widths[i]
        lc = color if i == 0 else (220, 220, 220)
        cv2.putText(frame, line, (cx - tw // 2, y_top + 4 + (i + 1) * 14),
                    font, scale, lc, thick, cv2.LINE_AA)

def _check_zone_intrusion(config, cam_id, persons, h, w, frame=None):
    try:
        amarilla_max_s = float(config.get("smart_amarilla_max_time") or 15) * 60
    except Exception:
        amarilla_max_s = 900.0

    for zone_key in ("verde", "amarilla", "roja"):
        z = config.get(f"smart_{zone_key}_zone")
        if not z:
            continue

        zx1 = int(z["x1"]*w); zy1 = int(z["y1"]*h)
        zx2 = int(z["x2"]*w); zy2 = int(z["y2"]*h)

        inside_key = (cam_id, zone_key)
        if inside_key not in _zone_persons_inside:
            _zone_persons_inside[inside_key] = set()

        previously_inside = _zone_persons_inside[inside_key]
        currently_inside  = set()

        for tid, cx, cy in persons:
            if not _point_in_zone(cx, cy, zx1, zy1, zx2, zy2):
                _zone_entry_times.pop((cam_id, zone_key, tid), None)
                continue

            currently_inside.add(tid)
            ek = (cam_id, zone_key, tid)

            # first frame this person is in the zone → entry event + assign zone ID
            if tid not in previously_inside:
                _zone_entry_times[ek] = time.time()
                _update_track_timeline(cam_id, tid, zone=zone_key)
                zone_id = _get_zone_id(cam_id, zone_key, tid)
                if zone_key == "roja":
                    show_notification("INTRUSIÓN ZONA ROJA",
                                      f"{zone_id} ingresó a zona restringida", "#ef4444")
                    _trigger_event_recording(cam_id, 'Intrusión Zona', frame)
                    register_event(current_user, "INTRUSIÓN_ZONA_ROJA", "OK",
                                   f"{zone_id} (BT:{tid}) en zona roja — Cam:{cam_id}")
                    log_event(cam_id, "Intrusión Zona Roja", "CRÍTICO",
                              f"{zone_id} (BT:{tid}) ingresó a zona restringida",
                              track_id=tid, frame=frame)
                elif zone_key == "amarilla":
                    show_notification("ZONA AMARILLA",
                                      f"{zone_id} ingresó a zona de advertencia", "#f59e0b")
                    log_event(cam_id, "Intrusión Zona Amarilla", "WARNING",
                              f"{zone_id} (BT:{tid}) ingresó a zona de advertencia",
                              track_id=tid)
                elif zone_key == "verde":
                    show_notification("ZONA VERDE",
                                      f"{zone_id} detectado en zona permitida", "#22c55e")
                    log_event(cam_id, "Zona Verde", "INFO",
                              f"{zone_id} (BT:{tid}) detectado en zona permitida",
                              track_id=tid)

            # continuous duration check for amarilla zone → Merodeador Detectado
            if zone_key == "amarilla" and ek in _zone_entry_times:
                elapsed   = time.time() - _zone_entry_times[ek]
                zone_id   = _get_zone_id(cam_id, zone_key, tid)
                elapsed_s = _fmt_elapsed(elapsed)
                if elapsed > amarilla_max_s and _throttle(cam_id, f"merodeo_{tid}", 30):
                    show_notification("MERODEADOR DETECTADO",
                                      f"{zone_id} lleva {elapsed_s} en zona amarilla",
                                      "#f97316")
                    log_event(cam_id, "Merodeador Detectado", "ALERTA",
                              f"{zone_id} (BT:{tid}) {elapsed_s} en zona amarilla",
                              track_id=tid, frame=frame)
                    _trigger_event_recording(cam_id, 'Merodeador Zona Amarilla', frame)

            # draw zone-specific label (ID + timer) on frame
            if frame is not None:
                _draw_zone_person_label(frame, cam_id, zone_key, tid, cx, cy,
                                        _zone_entry_times.get(ek))

        # clean up for persons who left the zone
        for tid in previously_inside - currently_inside:
            entry_ts = _zone_entry_times.pop((cam_id, zone_key, tid), None)
            if entry_ts and zone_key in ("roja", "amarilla"):
                elapsed = int(time.time() - entry_ts)
                sev     = "ALERTA" if zone_key == "roja" else "INFO"
                zone_id = _get_zone_id(cam_id, zone_key, tid)
                log_event(cam_id, f"Salida Zona {zone_key.capitalize()}", sev,
                          f"{zone_id} (BT:{tid}) salió tras {elapsed}s", track_id=tid)
            _release_zone_id(cam_id, zone_key, tid)

        _zone_persons_inside[inside_key] = currently_inside

# ── SISTEMA DE GRABACIÓN ─────────────────────────────────────────────────────

RECORDINGS_DIR_FILE = os.path.join(BASE_DIR, "recordings_dir.json")

def _ensure_recordings_dir(cam_id, subtype="analitica"):
    """Returns (and creates) the recordings/analitica subdir for cam_id."""
    global _recordings_dir
    if cam_id not in _recordings_dir:
        try:
            with open(RECORDINGS_DIR_FILE) as f:
                _recordings_dir[cam_id] = json.load(f).get("path", "recordings")
        except Exception:
            _recordings_dir[cam_id] = "recordings"
    path = os.path.join(_recordings_dir[cam_id], subtype)
    os.makedirs(path, exist_ok=True)
    return path

def _start_ai_writer(cam_id, frame):
    """Starts the AI-annotated VideoWriter → recordings/analitica/ at 15 FPS."""
    cam_id = str(cam_id)
    if _ai_writers.get(cam_id) is not None:
        return
    rec_dir = _ensure_recordings_dir(cam_id, "analitica")
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname   = os.path.join(rec_dir, f"cam{cam_id}_{ts}_ia.avi")
    h, w    = frame.shape[:2]
    fourcc  = cv2.VideoWriter_fourcc(*"XVID")
    writer  = cv2.VideoWriter(fname, fourcc, 15.0, (w, h))
    if writer.isOpened():
        _ai_writers[cam_id] = writer
        mode = _cam_record_mode.get(cam_id, _record_mode)
        print(f"[GRAB] Writer IA creado  | cam={cam_id} | modo={mode} | archivo={os.path.basename(fname)}")
        log_event(cam_id, "Grabación Iniciada", "INFO",
                  f"IA: {os.path.basename(fname)} ({mode})")
    else:
        print(f"[GRAB] ERROR VideoWriter | cam={cam_id} | NO pudo abrirse | archivo={fname}")

def _stop_writer(cam_id):
    cam_id = str(cam_id)
    ai_w = _ai_writers.pop(cam_id, None)
    if ai_w:
        ai_w.release()
        log_event(cam_id, "Grabación Finalizada", "INFO", "Clip cerrado")

def _stop_all_writers():
    for cid in list(_ai_writers.keys()):
        w = _ai_writers.pop(cid, None)
        if w:
            w.release()

def _write_frame(cam_id, frame):
    """Pre-event circular buffer only (JPEG bytes, ~15x menos RAM que numpy)."""
    cam_id = str(cam_id)
    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    _pre_event_buffer[cam_id].append(jpg.tobytes())

def _write_ai_frame(cam_id, frame):
    """Writes an AI-annotated frame to the analítica writer; manages post-event counter."""
    cam_id = str(cam_id)
    writer = _ai_writers.get(cam_id)
    if writer and writer.isOpened():
        writer.write(frame)
    remaining = _post_event_frames.get(cam_id, 0)
    if remaining > 0:
        _post_event_frames[cam_id] = remaining - 1
        if remaining - 1 == 0:
            _stop_writer(cam_id)
            _recording_enabled[cam_id] = False

def _trigger_event_recording(cam_id, event_name, frame):
    if not _recording_enabled.get(cam_id):
        return
    if not _ai_started.get(str(cam_id)):
        return
    if _ai_writers.get(cam_id) is None:
        _start_ai_writer(cam_id, frame)
    mode = _cam_record_mode.get(cam_id, _record_mode)
    if mode != "continuous":
        # event / smart / hybrid: vaciar pre-buffer al writer IA + clip post-evento
        writer = _ai_writers.get(cam_id)
        if writer and writer.isOpened():
            for jpg_bytes in list(_pre_event_buffer[cam_id]):
                f = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                if f is not None:
                    writer.write(f)
        _pre_event_buffer[cam_id].clear()
        post_frames = _rec_post_s.get(cam_id, 30) * 15
        _post_event_frames[cam_id] = post_frames
        _event_active[cam_id] = True
    _write_ai_frame(cam_id, frame)

# ── SISTEMA 1: Objeto Abandonado ─────────────────────────────────────────────

def _capture_background(cam_id, frame):
    _background_frame[cam_id] = frame.copy().astype(np.float32)
    _background_time[cam_id]  = time.time()
    _foreign_objects[cam_id]    = []
    _foreign_candidates[cam_id] = []
    _foreign_yolo_cache.pop(cam_id, None)
    h_bg, w_bg = frame.shape[:2]
    print(f"[OBJ_MOV] Fondo capturado | cam={cam_id} | Resolución: {w_bg}x{h_bg} | candidatos reiniciados")
    show_notification("FONDO CAPTURADO",
                      "Escena base guardada — mueve o agrega objetos para iniciar monitoreo.", "#22c55e")

def _build_person_mask(shape, person_boxes, expand=30):
    """Devuelve una máscara binaria (uint8) con ceros en las regiones ocupadas por personas.
    expand: píxeles extra alrededor del bounding box para absorber bordes y sombras."""
    mask = np.ones(shape[:2], dtype=np.uint8) * 255
    for _, px1, py1, px2, py2 in person_boxes:
        ex1 = max(0, px1 - expand)
        ey1 = max(0, py1 - expand)
        ex2 = min(shape[1] - 1, px2 + expand)
        ey2 = min(shape[0] - 1, py2 + expand)
        mask[ey1:ey2, ex1:ex2] = 0
    return mask


# Clases que nunca se vigilan (personas y animales)
_SKIP_CLASSES_WATCH = {"person", "cat", "dog", "horse", "sheep", "cow",
                        "elephant", "bear", "zebra", "giraffe", "bird"}


def _yolo_confirms_object_in_region(frame, x1, y1, x2, y2):
    """Returns True if YOLO detects a real (non-person) object with centroid inside the region.
    Used to cross-validate diff-based abandoned-object candidates and reject illumination noise."""
    try:
        model = get_model_general()
        results = model.predict(source=frame, verbose=False, conf=0.20, device=0)
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls        = int(box.cls[0])
                class_name = str(model.names[cls]).lower()
                if class_name in _SKIP_CLASSES_WATCH:
                    continue
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                bcx = (bx1 + bx2) // 2
                bcy = (by1 + by2) // 2
                if x1 <= bcx <= x2 and y1 <= bcy <= y2:
                    return True
    except Exception:
        pass
    return False


def _detect_foreign_objects(frame, cam_id, person_boxes=None):
    """Detecta objetos abandonados — arquitectura híbrida.

    Rama A — YOLO-snapshot (activa cuando existen watched_objects registrados):
      YOLO throttleado a 1 s; candidatos = detecciones nuevas no presentes en la
      referencia de objetos vigilados. Robusto para vistas cenitales y resistente
      a cambios de iluminación.

    Rama B — pixel-diff (activa cuando solo hay _background_frame, sin escaneo previo):
      Algoritmo original intacto para compatibilidad con el flujo sin escaneo.
    """
    now      = time.time()
    watched  = _watched_objects.get(cam_id, {})
    has_bg   = cam_id in _background_frame

    if not watched and not has_bg:
        return

    config_cam = ai_config.get(cam_id, {})
    confirm_s  = float(config_cam.get("abandoned_confirm_secs", ABANDONED_CONFIRM_SECONDS))

    if _throttle(cam_id, "abnd_rama_log", 8):
        print(f"[OBJ_ABN] Config | cam={cam_id} | Rama={'A-YOLO' if watched else 'B-diff'} "
              f"watched={len(watched)} bg={'sí' if has_bg else 'no'} "
              f"confirm={confirm_s:.0f}s candidates={len(_foreign_candidates.get(cam_id, []))}")

    if cam_id not in _foreign_candidates:
        _foreign_candidates[cam_id] = []
    if cam_id not in _foreign_objects:
        _foreign_objects[cam_id] = []

    detected_regions = []
    # ── RAMA A: YOLO-snapshot ─────────────────────────────────────────────────
    if watched:
        if _throttle(cam_id, "abnd_yolo_scan", 1.0):
            new_regions = []
            for model, _lbl in [(get_model_general(), "yolo"), (get_model_seguridad(), "sec")]:
                try:
                    results = model.predict(source=frame, verbose=False,
                                            conf=WATCH_CHECK_CONF, device=0)
                    if not (results and results[0].boxes is not None):
                        continue
                    for box in results[0].boxes:
                        cls        = int(box.cls[0])
                        class_name = str(model.names[cls]).lower()
                        if class_name in _SKIP_CLASSES_WATCH:
                            continue
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        cx = (bx1 + bx2) // 2
                        cy = (by1 + by2) // 2
                        # Excluir zonas de personas
                        if person_boxes:
                            _skip = False
                            for _, px1, py1, px2, py2 in person_boxes:
                                if px1 - 40 <= cx <= px2 + 40 and py1 - 40 <= cy <= py2 + 40:
                                    _skip = True
                                    break
                            if _skip:
                                continue
                        # Verificar si coincide con algún objeto vigilado conocido
                        fam       = _get_class_family(class_name)
                        is_known  = False
                        for wobj in watched.values():
                            if wobj.get("ignored"):
                                continue
                            if (_get_class_family(wobj["class"]) != fam
                                    and wobj["class"] != class_name):
                                continue
                            d_orig = ((cx - wobj["orig_cx"])**2 +
                                      (cy - wobj["orig_cy"])**2) ** 0.5
                            d_cur  = ((cx - wobj.get("cur_cx", wobj["orig_cx"]))**2 +
                                      (cy - wobj.get("cur_cy", wobj["orig_cy"]))**2) ** 0.5
                            if min(d_orig, d_cur) < 100:
                                is_known = True
                                break
                        if not is_known:
                            new_regions.append({"x1": bx1, "y1": by1,
                                                "x2": bx2, "y2": by2})
                except Exception:
                    pass
            _foreign_yolo_cache[cam_id] = new_regions
            print(f"[OBJ_ABN] Rama A scan | cam={cam_id} | desconocidos={len(new_regions)}")

        detected_regions = _foreign_yolo_cache.get(cam_id, [])

    # ── RAMA B: pixel-diff — fallback cuando Rama A sin resultados o sin watched ──
    if not detected_regions and has_bg:
        bg   = _background_frame[cam_id]
        curr = frame.astype(np.float32)
        if bg.shape != curr.shape:
            if _throttle(cam_id, "obj_abndnd_shape_warn", 30):
                print(f"[OBJ_MOV] ADVERTENCIA mismatch resolución | cam={cam_id} | bg:{bg.shape[:2]} curr:{curr.shape[:2]}")
            return
        if _throttle(cam_id, "obj_abndnd_diag", 10):
            print(f"[OBJ_MOV] detect_foreign_objects() | cam={cam_id} | {curr.shape[1]}x{curr.shape[0]} | candidates={len(_foreign_candidates.get(cam_id,[]))}")
        diff  = cv2.absdiff(bg, curr)
        gray  = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        kernel_close = np.ones((20, 20), np.uint8)
        kernel_open  = np.ones((10, 10), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel_open)
        if person_boxes:
            person_mask = _build_person_mask(frame.shape, person_boxes, expand=40)
            thresh      = cv2.bitwise_and(thresh, person_mask)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_regions = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 4000:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            detected_regions.append({"x1": x, "y1": y, "x2": x + cw, "y2": y + ch})
        if detected_regions and _throttle(cam_id, "obj_region_log", 5):
            print(f"[OBJ_MOV] Región detectada | cam={cam_id} | {len(detected_regions)} región(es) diff activas")

    # ── Actualizar candidatos (grace period 4 s, igual en ambas ramas) ────────
    grace_s            = 4.0
    current_candidates = []
    matched_old_ids    = set()

    for region in detected_regions:
        rcx = (region["x1"] + region["x2"]) // 2
        rcy = (region["y1"] + region["y2"]) // 2
        matched = None
        for i, cand in enumerate(_foreign_candidates[cam_id]):
            if i in matched_old_ids:
                continue
            ccx = (cand["x1"] + cand["x2"]) // 2
            ccy = (cand["y1"] + cand["y2"]) // 2
            if ((rcx - ccx) ** 2 + (rcy - ccy) ** 2) ** 0.5 < 80:
                matched = cand
                matched_old_ids.add(i)
                break
        current_candidates.append({
            "x1": region["x1"], "y1": region["y1"],
            "x2": region["x2"], "y2": region["y2"],
            "first_seen": matched["first_seen"] if matched else now,
            "last_seen":  now,
        })

    for i, cand in enumerate(_foreign_candidates[cam_id]):
        if i in matched_old_ids:
            continue
        _cand_age = now - cand.get("last_seen", cand["first_seen"])
        if _cand_age < grace_s:
            current_candidates.append({**cand, "last_seen": cand.get("last_seen", now)})
        else:
            if _throttle(cam_id, f"abnd_grexp_{cand['x1']}", 15):
                print(f"[OBJ_ABN] Candidato expiró (grace {grace_s:.0f}s) | cam={cam_id} | "
                      f"pos=({(cand['x1']+cand['x2'])//2},{(cand['y1']+cand['y2'])//2}) "
                      f"sin detección {_cand_age:.1f}s")

    _foreign_candidates[cam_id] = current_candidates

    for cand in current_candidates:
        elapsed = now - cand["first_seen"]
        if elapsed > 1.0 and _throttle(cam_id, f"abnd_cand_{cand['x1']}_{cand['y1']}", 3):
            print(f"[OBJ_ABN] Candidato activo | cam={cam_id} | "
                  f"elapsed={elapsed:.1f}s/{confirm_s:.0f}s req "
                  f"pos=({(cand['x1']+cand['x2'])//2},{(cand['y1']+cand['y2'])//2})")

    # ── Promover candidatos que llevan ≥ confirm_s inmóviles ─────────────────
    confirmed = []
    for cand in current_candidates:
        if now - cand["first_seen"] < confirm_s:
            continue
        already = any(
            abs((cand["x1"] + cand["x2"]) // 2 - (o["x1"] + o["x2"]) // 2) < 80 and
            abs((cand["y1"] + cand["y2"]) // 2 - (o["y1"] + o["y2"]) // 2) < 80
            for o in _foreign_objects[cam_id]
        )
        if already:
            continue
        # Rama B cross-valida con YOLO; Rama A ya validó semánticamente en el scan
        if not watched:
            if not _yolo_confirms_object_in_region(frame, cand["x1"], cand["y1"],
                                                   cand["x2"], cand["y2"]):
                if _throttle(cam_id, f"abnd_skip_{cand['x1']}", 15):
                    print(f"[OBJ_MOV] Candidato RECHAZADO (YOLO no confirma) | cam={cam_id}")
                continue
        persist_s = now - cand["first_seen"]
        confirmed.append({
            "x1": cand["x1"], "y1": cand["y1"],
            "x2": cand["x2"], "y2": cand["y2"],
            "time":    datetime.now().strftime("%H:%M:%S"),
            "start":   cand["first_seen"],
            "persist": persist_s,
        })
        if _throttle(cam_id, f"foreign_{cand['x1']}_{cand['y1']}", 60):
            print(f"[OBJ_MOV] Evento Objeto Abandonado | cam={cam_id} | "
                  f"{persist_s:.0f}s inmóvil ({cand['x1']},{cand['y1']})")
            show_notification("OBJETO ABANDONADO",
                              f"Objeto nuevo en escena — {persist_s:.0f}s inmóvil.", "#ef4444")
            _trigger_event_recording(cam_id, 'Objeto Abandonado', frame)
            register_event(current_user, "OBJETO_ABANDONADO", "OK",
                           f"Objeto nuevo detectado - Cam:{cam_id}")
            log_event(cam_id, "Objeto Abandonado", "ALERTA",
                      f"Objeto nuevo confirmado {persist_s:.0f}s (x:{cand['x1']},y:{cand['y1']})",
                      coords=[cand["x1"], cand["y1"], cand["x2"], cand["y2"]],
                      frame=frame)
    _foreign_objects[cam_id].extend(confirmed)

    # Retirar objetos confirmados cuya región ya no está presente
    _foreign_objects[cam_id] = [
        obj for obj in _foreign_objects[cam_id]
        if any(
            (((obj["x1"] + obj["x2"]) // 2 - (r["x1"] + r["x2"]) // 2) ** 2 +
             ((obj["y1"] + obj["y2"]) // 2 - (r["y1"] + r["y2"]) // 2) ** 2) ** 0.5 < 80
            for r in detected_regions
        )
    ]

def _render_foreign_objects(frame, cam_id):
    now = time.time()
    for obj in _foreign_objects.get(cam_id, []):
        x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
        # Color amarillo para objeto abandonado (diferencia visual con rojo de armas)
        color = (0, 200, 255)
        alpha = 0.3 + 0.15 * abs((now % 1.0) - 0.5)
        ov = frame.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha * 0.25, frame, 1 - alpha * 0.25, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        persist_s = now - obj.get("start", now)
        lines = ["OBJETO ABANDONADO", f"Hora: {obj['time']}", f"Inmovil: {persist_s:.0f}s"]
        bg_h = len(lines) * 16 + 8
        ov2 = frame.copy()
        cv2.rectangle(ov2, (x1, y1 - bg_h - 4), (x1 + 190, y1), (0, 0, 0), -1)
        cv2.addWeighted(ov2, 0.65, frame, 0.35, 0, frame)
        for li, line in enumerate(lines):
            col = color if li == 0 else (255, 255, 255)
            cv2.putText(frame, line, (x1 + 4, y1 - bg_h + 4 + li * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)


# ── SISTEMA 2: Objeto Movido ──────────────────────────────────────────────────

# ── Familias de clases para matching semántico tolerante ─────────────────────
# backpack / bag / mochila → misma familia → no se genera falso "faltante"
_OBJECT_ONLY_CLASSES = {
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "tv", "microwave", "oven", "toaster", "sink", "refrigerator",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "extintor", "mochila", "caja",
    "casco de seguridad", "bata industrial", "botas de seguridad",
}

_CLASS_FAMILIES_DEF = {
    "bag":         {"backpack", "handbag", "bag", "purse", "mochila", "bolsa"},
    "luggage":     {"suitcase", "luggage"},
    "box":         {"box", "caja"},
    "drinkware":   {"bottle", "wine glass", "cup"},
    "laptop":      {"laptop"},
    "phone":       {"cell phone"},
    "chair":       {"chair"},
    "table":       {"dining table"},
    "tv":          {"tv"},
    "clock":       {"clock"},
    "umbrella":    {"umbrella"},
    "book":        {"book"},
    "plant":       {"potted plant"},
    "extintor":    {"extintor"},
    "safety_wear": {"casco de seguridad", "bata industrial", "botas de seguridad"},
    "food":        {"banana", "apple", "sandwich", "orange", "broccoli",
                    "carrot", "hot dog", "pizza", "donut", "cake", "bowl"},
}
_CLASS_FAMILY_MAP = {}
for _fam_k, _fam_set in _CLASS_FAMILIES_DEF.items():
    for _cl in _fam_set:
        _CLASS_FAMILY_MAP[_cl] = _fam_k


def _get_class_family(class_name):
    cl = class_name.lower().strip()
    if cl in _CLASS_FAMILY_MAP:
        return _CLASS_FAMILY_MAP[cl]
    for fam, classes in _CLASS_FAMILIES_DEF.items():
        if any(c in cl for c in classes) or any(cl in c for c in classes):
            return fam
    return cl


# ── Parámetros de detección v3 ────────────────────────────────────────────────
WATCH_REGISTER_CONF   = 0.30   # conf mínima para registrar un objeto vigilado
WATCH_CHECK_CONF      = 0.15   # conf mínima durante verificación frame a frame
MISS_STREAK_THRESHOLD = 8      # frames consecutivos sin detectar → iniciar timer ausencia
MOVE_DISTANCE_PX      = 120    # distancia en px para considerar desplazamiento real
MOVE_CONFIRM_FRAMES   = 5      # frames en nueva posición para confirmar movimiento
WATCH_SEARCH_RADIUS   = 350    # radio px de búsqueda alrededor del origen/posición actual
OPENCV_MIN_AREA_PX    = 800    # área mínima (px²) para registrar objeto desconocido vía OpenCV
OBJ_EVIDENCE_DIR      = os.path.join(SNAPSHOTS_DIR, "obj_evidence")


def _save_obj_evidence_crop(frame, wid, cam_id, wobj, current_box, suffix):
    """Guarda un recorte de evidencia para auditoría (before/after)."""
    try:
        os.makedirs(OBJ_EVIDENCE_DIR, exist_ok=True)
        pad  = 25
        h_f, w_f = frame.shape[:2]
        if current_box:
            x1 = max(0, current_box["x1"] - pad); y1 = max(0, current_box["y1"] - pad)
            x2 = min(w_f, current_box["x2"] + pad); y2 = min(h_f, current_box["y2"] + pad)
        else:
            x1 = max(0, wobj["x1"] - pad); y1 = max(0, wobj["y1"] - pad)
            x2 = min(w_f, wobj["x2"] + pad); y2 = min(h_f, wobj["y2"] + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            path = os.path.join(OBJ_EVIDENCE_DIR, f"obj{wid}_c{cam_id}_{suffix}.jpg")
            cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return path
    except Exception:
        pass
    return None


def _scan_opencv_objects(frame, existing_regions):
    """Detecta regiones significativas con OpenCV para objetos que YOLO no reconoce.
    existing_regions: lista de (x1,y1,x2,y2) de detecciones YOLO — se excluyen para evitar duplicados."""
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for c in cnts:
        if cv2.contourArea(c) < OPENCV_MIN_AREA_PX:
            continue
        x1, y1, bw, bh = cv2.boundingRect(c)
        x2, y2          = x1 + bw, y1 + bh
        cx, cy          = x1 + bw // 2, y1 + bh // 2
        # Ignorar si el centroide cae dentro de una detección YOLO ya registrada
        if any(ex1 <= cx <= ex2 and ey1 <= cy <= ey2
               for ex1, ey1, ex2, ey2 in existing_regions):
            continue
        found.append({"class": "unknown", "source": "opencv",
                      "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "cx": cx,  "cy": cy,  "conf": 1.0})
    return found


def _scan_all_objects(frame, cam_id):
    """Escanea objetos usando YOLO únicamente (conf ≥ WATCH_REGISTER_CONF).
    Sin diff-based fallback — elimina registros falsos por ruido de iluminación."""
    found        = []
    seen_regions = []

    def _no_overlap(x1, y1, x2, y2):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        for rx1, ry1, rx2, ry2 in seen_regions:
            if ((cx - (rx1+rx2)//2)**2 + (cy - (ry1+ry2)//2)**2)**0.5 < 60:
                return False
        return True

    for model, label in [(get_model_general(), "yolo"), (get_model_seguridad(), "security")]:
        try:
            results = model.predict(source=frame, verbose=False, conf=WATCH_REGISTER_CONF, device=0)
            if not results or results[0].boxes is None:
                continue
            for box in results[0].boxes:
                cls        = int(box.cls[0])
                class_name = str(model.names[cls]).lower()
                if class_name in _SKIP_CLASSES_WATCH:
                    continue
                if any(x in class_name for x in ("carro", "cart", "trolley")):
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if _no_overlap(x1, y1, x2, y2):
                    seen_regions.append((x1, y1, x2, y2))
                    found.append({"class": class_name, "source": label,
                                  "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                  "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
                                  "conf": conf})
        except Exception:
            pass

    # ── Capa OpenCV: objetos desconocidos no cubiertos por YOLO ──────────────
    yolo_regions = [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in found]
    for ob in _scan_opencv_objects(frame, yolo_regions):
        if _no_overlap(ob["x1"], ob["y1"], ob["x2"], ob["y2"]):
            seen_regions.append((ob["x1"], ob["y1"], ob["x2"], ob["y2"]))
            found.append(ob)

    return found


def _register_watched_objects(frame, cam_id):
    """Registra objetos en el frame de referencia con estado completo para la máquina de estados."""
    if cam_id not in _background_frame:
        _background_frame[cam_id] = frame.astype(np.float32)

    h_rw, w_rw = frame.shape[:2]
    print(f"[OBJ_MOV] Registro iniciado | cam={cam_id} | {w_rw}x{h_rw}")
    boxes = _scan_all_objects(frame, cam_id)

    _watched_objects[cam_id]  = {}
    _moved_alerts[cam_id]     = {}
    _missing_objects[cam_id]  = {}
    for k in list(_object_trail_hist.keys()):
        if k[0] == cam_id:
            del _object_trail_hist[k]

    os.makedirs(OBJ_EVIDENCE_DIR, exist_ok=True)
    now_str = datetime.now().strftime("%H:%M:%S")

    for i, b in enumerate(boxes):
        before_path = _save_obj_evidence_crop(frame, i, cam_id, b, b, "before")
        # Shape descriptors (filtrado de matches y base para OBB futuro)
        _bx1, _by1, _bx2, _by2 = b["x1"], b["y1"], b["x2"], b["y2"]
        _bw = max(1, _bx2 - _bx1)
        _bh = max(1, _by2 - _by1)
        _ref_area   = _bw * _bh
        _ref_aspect = _bw / _bh
        _ref_cr     = 0.5  # contour ratio: área hull / área bbox
        try:
            _roi_g = cv2.cvtColor(
                frame[max(0, _by1):min(frame.shape[0], _by2),
                      max(0, _bx1):min(frame.shape[1], _bx2)],
                cv2.COLOR_BGR2GRAY)
            _, _roi_t = cv2.threshold(_roi_g, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _cnts_r, _ = cv2.findContours(_roi_t, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if _cnts_r:
                _hull_r = cv2.convexHull(max(_cnts_r, key=cv2.contourArea))
                _ref_cr = cv2.contourArea(_hull_r) / _ref_area
        except Exception:
            pass

        # Plantilla de píxeles para matching de objetos desconocidos (source=opencv)
        _tmpl_crop = None
        if b.get("source") == "opencv":
            _ty1 = max(0, b["y1"]); _ty2 = min(frame.shape[0], b["y2"])
            _tx1 = max(0, b["x1"]); _tx2 = min(frame.shape[1], b["x2"])
            if _ty2 > _ty1 and _tx2 > _tx1:
                _tmpl_crop = frame[_ty1:_ty2, _tx1:_tx2].copy()

        _watched_objects[cam_id][i] = {
            "class":   b["class"],
            "family":  _get_class_family(b["class"]),
            "source":  b["source"],
            "conf":    b["conf"],
            "orig_cx": b["cx"],   "orig_cy": b["cy"],
            "x1": b["x1"],        "y1": b["y1"],
            "x2": b["x2"],        "y2": b["y2"],
            "time":    now_str,
            "ignored": False,
            "miss_streak":    0,
            "present_streak": 0,
            "missing_since":  None,
            "miss_alerted":   False,
            "move_start":     None,
            "move_alerted":   False,
            "move_streak":    0,
            "cur_cx": b["cx"],    "cur_cy": b["cy"],
            "cur_x1": b["x1"],    "cur_y1": b["y1"],
            "cur_x2": b["x2"],    "cur_y2": b["y2"],
            "before_path": before_path,
            "after_path":  None,
            # Descriptores de forma para filtrado de matches
            "ref_area":          _ref_area,
            "ref_aspect":        _ref_aspect,
            "ref_contour_ratio": _ref_cr,
            # Placeholders OBB (orientación y ángulo — sin implementar aún)
            "orientation":       0.0,
            "angle_obb":         None,
            # Template para objetos sin clase reconocida por YOLO
            "_template":         _tmpl_crop,
        }

    yolo_count  = sum(1 for b in boxes if b["source"] in ("yolo", "security"))
    opencv_count = sum(1 for b in boxes if b["source"] == "opencv")
    print(f"[OBJ_MOV] Objetos registrados | cam={cam_id} | total={len(boxes)} YOLO:{yolo_count} OpenCV:{opencv_count}")
    show_notification("OBJETOS REGISTRADOS",
                      f"{len(boxes)} objetos vigilados | YOLO:{yolo_count} + OpenCV:{opencv_count}",
                      "#3b82f6")
    return boxes


def _match_opencv_object(frame, wobj):
    """Template matching para objetos sin clase (source='opencv').
    Busca el template guardado en un radio WATCH_SEARCH_RADIUS alrededor de cur_cx/cur_cy.
    Retorna dict con cx/cy/x1/y1/x2/y2 si score ≥ 0.55, None si no se encuentra."""
    tmpl = wobj.get("_template")
    if tmpl is None or tmpl.size == 0:
        return None
    th, tw = tmpl.shape[:2]
    cx_s   = wobj["cur_cx"]
    cy_s   = wobj["cur_cy"]
    sx1    = max(0, cx_s - WATCH_SEARCH_RADIUS)
    sy1    = max(0, cy_s - WATCH_SEARCH_RADIUS)
    sx2    = min(frame.shape[1], cx_s + WATCH_SEARCH_RADIUS)
    sy2    = min(frame.shape[0], cy_s + WATCH_SEARCH_RADIUS)
    roi    = frame[sy1:sy2, sx1:sx2]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return None
    try:
        res              = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, maxval, _, ml = cv2.minMaxLoc(res)
        if maxval < 0.55:
            return None
        ax1 = sx1 + ml[0]
        ay1 = sy1 + ml[1]
        return {"class": "unknown", "family": "unknown",
                "cx": ax1 + tw // 2, "cy": ay1 + th // 2,
                "x1": ax1, "y1": ay1, "x2": ax1 + tw, "y2": ay1 + th}
    except Exception:
        return None


def _check_moved_objects(frame, cam_id, person_boxes=None):
    """Máquina de estados por objeto vigilado.
    YOLO: matching por familia de clase + radio.
    OpenCV: template matching TM_CCOEFF_NORMED para objetos desconocidos.
    PRESENTE → miss_streak ≥ MISS_STREAK_THRESHOLD → timer ausencia → FALTANTE
    PRESENTE → dist > MOVE_DISTANCE_PX durante MOVE_CONFIRM_FRAMES → MOVIDO
    """
    if cam_id not in _watched_objects:
        if _throttle(cam_id, "obj_check_no_watched", 15):
            print(f"[OBJ_MOV] Sin objetos registrados | cam={cam_id}")
        return

    watched = _watched_objects[cam_id]
    if not watched:
        return

    if _throttle(cam_id, "obj_check_log", 10):
        active = sum(1 for w in watched.values() if not w.get("ignored"))
        print(f"[OBJ_MOV] check_moved | cam={cam_id} | activos={active}")

    # ── Pasada YOLO conjunta (general + seguridad) ────────────────────────────
    current_boxes = []
    for model, _lbl in [(_get_track_gen(cam_id), "yolo"), (_get_track_sec(cam_id), "sec")]:
        try:
            results = model.predict(source=frame, verbose=False, conf=WATCH_CHECK_CONF, device=0)
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls        = int(box.cls[0])
                    class_name = str(model.names[cls]).lower()
                    if class_name in _SKIP_CLASSES_WATCH:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    current_boxes.append({
                        "class":  class_name,
                        "family": _get_class_family(class_name),
                        "cx": (x1+x2)//2, "cy": (y1+y2)//2,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })
        except Exception:
            pass

    if _throttle(cam_id, "obj_scan_sum", 5):
        print(f"[OBJ_MOV] Escaneado | cam={cam_id} | "
              f"vigilados={sum(1 for w in watched.values() if not w.get('ignored'))} "
              f"detectados={len(current_boxes)}")

    def _in_person_zone(cx, cy):
        if not person_boxes:
            return False
        for _, px1, py1, px2, py2 in person_boxes:
            if px1 - 40 <= cx <= px2 + 40 and py1 - 40 <= cy <= py2 + 40:
                return True
        return False

    now    = time.time()
    conf_s = float(MOVED_MISSING_CONFIRM_SECONDS)

    for wid, wobj in watched.items():
        if wobj.get("ignored"):
            continue

        ox, oy = wobj["orig_cx"], wobj["orig_cy"]
        wfam   = wobj["family"]
        wclass = wobj["class"]

        # ── Buscar mejor match: misma familia + radio ────────────────────────────
        best   = None
        best_d = float("inf")
        for cb in current_boxes:
            if cb["family"] != wfam and cb["class"] != wclass:
                continue
            if _in_person_zone(cb["cx"], cb["cy"]):
                continue
            d_orig = ((cb["cx"] - ox)**2             + (cb["cy"] - oy)**2)             ** 0.5
            d_cur  = ((cb["cx"] - wobj["cur_cx"])**2 + (cb["cy"] - wobj["cur_cy"])**2) ** 0.5
            d_eff  = min(d_orig, d_cur)
            if d_eff < best_d and d_eff < WATCH_SEARCH_RADIUS:
                best_d = d_eff
                best   = cb

        # ── Fallback OpenCV: template matching para objetos desconocidos ──────
        if best is None and wobj.get("source") == "opencv":
            best = _match_opencv_object(frame, wobj)

        # ── OBJETO ENCONTRADO ─────────────────────────────────────────────────
        if best is not None:
            dist_from_origin = ((best["cx"] - ox)**2 + (best["cy"] - oy)**2) ** 0.5

            wobj["miss_streak"]   = 0
            wobj["missing_since"] = None

            wobj["cur_cx"] = best["cx"]; wobj["cur_cy"] = best["cy"]
            wobj["cur_x1"] = best["x1"]; wobj["cur_y1"] = best["y1"]
            wobj["cur_x2"] = best["x2"]; wobj["cur_y2"] = best["y2"]

            if dist_from_origin > MOVE_DISTANCE_PX:
                key = (cam_id, wid)
                _object_trail_hist[key].append((now, best["cx"], best["cy"]))
                wobj["move_streak"] += 1
                if wobj["move_start"] is None:
                    wobj["move_start"] = now

                elapsed_move = now - wobj["move_start"]

                if wobj["move_streak"] >= MOVE_CONFIRM_FRAMES and not wobj.get("move_alerted"):
                    wobj["move_alerted"] = True
                    if _throttle(cam_id, f"moved_w_{wid}", 30):
                        dist_m = round(dist_from_origin / 100, 1)
                        print(f"[OBJ_MOV] EVENTO Movido | cam={cam_id} | "
                              f"{wobj['class']} #{wid} dist={dist_m}m en {elapsed_move:.0f}s")
                        after_p = _save_obj_evidence_crop(frame, wid, cam_id, wobj, best, "after")
                        wobj["after_path"] = after_p
                        show_notification("OBJETO MOVIDO",
                                          f"{wobj['class'].title()} #{wid} desplazado {dist_m}m",
                                          "#f97316")
                        _trigger_event_recording(cam_id, 'Objeto Movido', frame)
                        register_event(current_user, "OBJETO_MOVIDO", "OK",
                                       f"#{wid} {wobj['class']} movido {dist_m}m Cam:{cam_id} "
                                       f"Origen:({ox},{oy}) Final:({best['cx']},{best['cy']}) "
                                       f"Tiempo:{elapsed_move:.0f}s")
                        log_event(cam_id, "Objeto Movido", "ALERTA",
                                  f"#{wid} {wobj['class'].title()} desplazado {dist_m}m "
                                  f"en {elapsed_move:.0f}s — "
                                  f"Origen:({ox},{oy}) → Final:({best['cx']},{best['cy']})",
                                  coords=[best["x1"], best["y1"], best["x2"], best["y2"]],
                                  frame=frame)
            else:
                wobj["move_streak"]  = 0
                wobj["move_start"]   = None
                wobj["move_alerted"] = False
            continue

        # ── OBJETO NO ENCONTRADO ──────────────────────────────────────────────
        wobj["move_streak"] = 0
        wobj["miss_streak"] += 1
        if _throttle(cam_id, f"obj_miss_{wid}", 8):
            print(f"[OBJ_MOV] #{wid} {wobj['class']} sin match | cam={cam_id} | "
                  f"miss_streak={wobj['miss_streak']}/{MISS_STREAK_THRESHOLD}")

        if wobj["miss_streak"] < MISS_STREAK_THRESHOLD:
            continue

        if wobj["missing_since"] is None:
            wobj["missing_since"] = now

        elapsed_miss = now - wobj["missing_since"]

        if elapsed_miss >= conf_s and not wobj.get("miss_alerted"):
            wobj["miss_alerted"] = True
            if _throttle(cam_id, f"missing_{wid}", 30):
                print(f"[OBJ_MOV] EVENTO Faltante | cam={cam_id} | "
                      f"{wobj['class']} #{wid} ausente {elapsed_miss:.0f}s")
                after_p = _save_obj_evidence_crop(frame, wid, cam_id, wobj, None, "after")
                wobj["after_path"] = after_p
                show_notification("OBJETO MOVIDO",
                                  f"{wobj['class'].title()} #{wid} desapareció ({elapsed_miss:.0f}s)",
                                  "#f97316")
                _trigger_event_recording(cam_id, 'Objeto Movido', frame)
                register_event(current_user, "OBJETO_MOVIDO", "OK",
                               f"#{wid} {wobj['class']} desapareció Cam:{cam_id} "
                               f"Posición:({ox},{oy}) Ausente:{elapsed_miss:.0f}s")
                log_event(cam_id, "Objeto Movido — Faltante", "ALERTA",
                          f"#{wid} {wobj['class'].title()} ausente {elapsed_miss:.0f}s — "
                          f"Última posición:({ox},{oy})",
                          coords=[wobj["x1"], wobj["y1"], wobj["x2"], wobj["y2"]],
                          frame=frame)


def _draw_overlay_text(frame, lines, x1, y1, title_color):
    bg_h  = len(lines) * 16 + 8
    y_top = max(y1 - bg_h - 4, 0)
    ov    = frame.copy()
    cv2.rectangle(ov, (x1, y_top), (x1 + 230, y1), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
    for li, line in enumerate(lines):
        col = title_color if li == 0 else (220, 220, 220)
        cv2.putText(frame, line, (x1 + 4, y_top + 4 + li * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)


def _render_watched_and_moved(frame, cam_id):
    """Overlay: verde (presente), naranja+flecha (movido), rojo pulsante (faltante)."""
    watched = _watched_objects.get(cam_id, {})
    now     = time.time()
    conf_s  = float(MOVED_MISSING_CONFIRM_SECONDS)

    for wid, wobj in watched.items():
        if wobj.get("ignored"):
            continue

        ox, oy        = wobj["orig_cx"], wobj["orig_cy"]
        miss_streak   = wobj.get("miss_streak", 0)
        missing_since = wobj.get("missing_since")
        move_start    = wobj.get("move_start")
        move_alerted  = wobj.get("move_alerted", False)
        miss_alerted  = wobj.get("miss_alerted", False)

        # ── Presente en posición original (verde tenue) ───────────────────────
        if miss_streak < MISS_STREAK_THRESHOLD and move_start is None and not move_alerted:
            x1, y1, x2, y2 = wobj["x1"], wobj["y1"], wobj["x2"], wobj["y2"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 100), 1)
            cv2.putText(frame, f"{wobj['class']} [V]",
                        (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 100), 1)
            continue

        # ── Desplazado (naranja + flecha desde origen) ────────────────────────
        if move_start is not None or move_alerted:
            cx1, cy1 = wobj["cur_x1"], wobj["cur_y1"]
            cx2, cy2 = wobj["cur_x2"], wobj["cur_y2"]
            ccx, ccy = wobj["cur_cx"], wobj["cur_cy"]

            key  = (cam_id, wid)
            hist = list(_object_trail_hist.get(key, []))
            for i in range(1, len(hist)):
                at = i / max(len(hist), 1)
                p1 = (int(hist[i-1][1]), int(hist[i-1][2]))
                p2 = (int(hist[i][1]),   int(hist[i][2]))
                cv2.line(frame, p1, p2, (0, int(180 * at), 255), max(1, int(2 * at)))

            cv2.arrowedLine(frame, (ox, oy), (ccx, ccy), (0, 165, 255), 2, tipLength=0.12)
            cv2.circle(frame, (ox, oy), 7, (0, 255, 255), -1)
            cv2.putText(frame, "ORIGEN",
                        (ox + 9, oy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 255), 1)

            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 140, 255), 2)
            dist_px = ((ccx - ox)**2 + (ccy - oy)**2) ** 0.5
            dist_m  = round(dist_px / 100, 1)
            elapsed = round(now - move_start, 0) if move_start else 0
            _draw_overlay_text(frame,
                               ["OBJETO MOVIDO",
                                f"ID:{wid} {wobj['class'].upper()}",
                                f"Dist:{dist_m}m | {elapsed:.0f}s",
                                f"Origen:({ox},{oy})",
                                f"Final:({ccx},{ccy})"],
                               cx1, cy1, (0, 165, 255))
            continue

        # ── Faltante (caja roja pulsante en última posición conocida) ─────────
        if miss_alerted or (missing_since and now - missing_since >= conf_s):
            x1, y1, x2, y2 = wobj["x1"], wobj["y1"], wobj["x2"], wobj["y2"]
            pulse = 0.3 + 0.25 * abs((now % 1.0) - 0.5)
            ov    = frame.copy()
            cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.addWeighted(ov, pulse * 0.4, frame, 1 - pulse * 0.4, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            elapsed = round(now - missing_since, 0) if missing_since else 0
            _draw_overlay_text(frame,
                               ["OBJETO FALTANTE",
                                f"ID:{wid} {wobj['class'].upper()}",
                                f"Ausente: {elapsed:.0f}s",
                                f"Posicion:({ox},{oy})"],
                               x1, y1, (0, 60, 255))



# =========================
# FRAME PROCESSING PIPELINE
# =========================
def process_frame(frame, camera_id):
    cam_id = str(camera_id)
    config = ai_config.get(cam_id, {})
    if not config:
        return frame

    if frame is None or frame.size == 0:
        return frame
    if frame.shape[0] < 10 or frame.shape[1] < 10:
        return frame

    h, w = frame.shape[:2]
    persons      = []   # (track_id, cx, cy) — usado por zonas inteligentes
    person_boxes = []   # (track_id, x1, y1, x2, y2) — usado por operadores y máscaras

    # Frame limpio para comparación diff — antes de que se dibujen overlays YOLO
    _clean_detection_frame = frame.copy()

    # CAPA 1: zonas inteligentes fijas (debajo de todo)
    _render_smart_zones(frame, config, h, w)

    need_gen   = _need_general(config)
    need_sec   = _need_security(config)
    need_track = _need_tracking(config)

    # Actualizar FPS de IA
    now_ts = time.time()
    _ai_fps_ts[cam_id].append(now_ts)

    # CAPA 2: detección YOLO + bounding boxes
    _t_total_start = time.time()
    _t_gen_start   = time.time()
    if need_gen:
        model = _get_track_gen(cam_id)
        try:
            results = model.track(source=frame, persist=True,
                                  tracker=TRACKER_CONFIG, verbose=False, device=0) if need_track \
                      else model.predict(source=frame, verbose=False, device=0)
        except Exception as exc:
            log_event(cam_id, "Error YOLO General", "WARNING",
                      f"Fallo en inferencia: {exc}")
            results = []
        if results:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls        = int(box.cls[0])
                    class_name = str(model.names[cls]).lower()
                    is_person  = (class_name == "person")

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy   = (x1+x2)//2, (y1+y2)//2
                    conf_val = float(box.conf[0])
                    track_id = int(box.id.item()) if box.id is not None else -1

                    # Always register persons for downstream logic regardless of display config.
                    # History/timeline require a valid track_id; box registration does not.
                    if is_person:
                        if track_id >= 0:
                            _update_history(cam_id, track_id, cx, cy)
                            _update_track_timeline(cam_id, track_id)
                        persons.append((track_id, cx, cy))
                        person_boxes.append((track_id, x1, y1, x2, y2))

                    # Skip display/logging for classes not enabled in config
                    if not _general_class_ok(config, class_name):
                        continue

                    color  = (255, 200, 0)
                    alerts = []
                    _show_fall_box = False
                    if is_person:
                        if track_id >= 0:
                            # Log first appearance of this track ID (only when Persona display is on)
                            if _confirm_detection(cam_id, f"person_first_{track_id}", 2, 3.0):
                                log_event(cam_id, "Persona Detectada", "INFO",
                                          f"ID:{track_id} — confianza {conf_val:.2f}",
                                          track_id=track_id, conf=conf_val,
                                          coords=[x1,y1,x2,y2])
                        if config.get("Persona Corriendo") and track_id >= 0 and _is_running(cam_id, track_id):
                            color = (0, 0, 255)
                            alerts.append("CORRE")
                            if _throttle(cam_id, f"running_{track_id}", 10):
                                show_notification("ALERTA", f"Persona corriendo ID:{track_id}", "#ef4444")
                            if _confirm_detection(cam_id, f"running_{track_id}", 3, 4.0):
                                log_event(cam_id, "Persona Corriendo", "ALERTA",
                                          f"ID:{track_id} corriendo — comportamiento anómalo",
                                          track_id=track_id, conf=conf_val, frame=frame)
                                _update_track_timeline(cam_id, track_id, behavior="corriendo")
                        if config.get("Persona Inmóvil") and track_id >= 0 and _is_immobile(cam_id, track_id):
                            color = (0, 140, 255)
                            alerts.append("INMÓVIL")
                            if _throttle(cam_id, f"immobile_{track_id}", 30):
                                show_notification("ALERTA", f"Persona inmóvil ID:{track_id}", "#f59e0b")
                            if _confirm_detection(cam_id, f"immobile_{track_id}", 5, 8.0):
                                log_event(cam_id, "Persona Inmóvil", "WARNING",
                                          f"ID:{track_id} sin movimiento prolongado",
                                          track_id=track_id, frame=frame)
                                _update_track_timeline(cam_id, track_id, behavior="inmóvil")
                        if track_id >= 0:
                            _check_fall(cam_id, config, track_id,
                                        x1, y1, x2, y2, cx, cy, frame)
                            if _fall_suspects.get((cam_id, track_id), {}).get("confirmed"):
                                color = (0, 0, 255)
                                alerts.append("CAÍDA")
                                _show_fall_box = True
                    else:
                        if class_name == "cell phone" and config.get("No Celular"):
                            if _confirm_detection(cam_id, f"celular_{x1//50}_{y1//50}", 2, 3.0):
                                if _throttle(cam_id, f"celular_{x1//50}_{y1//50}", 30):
                                    show_notification("EPP", "Uso de celular detectado", "#f59e0b")
                                    log_event(cam_id, "Uso de Celular", "ALERTA",
                                              f"Confianza {conf_val:.2f}",
                                              conf=conf_val, coords=[x1,y1,x2,y2], frame=frame)
                        # Otros objetos generales (mochila, caja…)
                        elif _confirm_detection(cam_id, f"obj_{class_name}_{x1//50}_{y1//50}", 2, 3.0):
                            log_event(cam_id, f"Objeto Detectado: {class_name}", "INFO",
                                      f"Confianza {conf_val:.2f}", conf=conf_val,
                                      coords=[x1,y1,x2,y2])
                    if _show_fall_box:
                        _draw_fall_overlay(frame, track_id, x1, y1, x2, y2)
                        caja_watchdogs(frame, x1, y1, x2, y2)
                        crosshair(frame, x1, y1, x2, y2)
                        lbl = (f"ID:{track_id} " if track_id >= 0 else "") + " ".join(alerts)
                        cv2.putText(frame, lbl.strip() or class_name, (x1, y1-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        glow_rectangle(frame, x1, y1, x2, y2, color)
                        caja_watchdogs(frame, x1, y1, x2, y2)
                        crosshair(frame, x1, y1, x2, y2)
                        lbl = (f"ID:{track_id} " if track_id >= 0 else "") + " ".join(alerts)
                        cv2.putText(frame, lbl.strip() or class_name, (x1, y1-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    _perf_t_gen[cam_id].append(time.time() - _t_gen_start)
    _t_sec_start = time.time()
    if need_sec:
        model = _get_track_sec(cam_id)
        try:
            results = model.track(source=frame, persist=True,
                                  tracker=TRACKER_CONFIG, verbose=False,
                                  conf=0.10, device=0) if need_track \
                      else model.predict(source=frame, verbose=False, conf=0.10, device=0)
        except Exception as exc:
            log_event(cam_id, "Error YOLO Seguridad", "WARNING",
                      f"Fallo en inferencia: {exc}")
            results = []
        if results:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    cls        = int(box.cls[0])
                    class_name = str(model.names[cls]).lower()
                    if not _security_class_ok(config, class_name):
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf  = float(box.conf[0])
                    color = (0, 255, 255)
                    is_weapon = "arma" in class_name
                    if is_weapon:           color = (0, 0, 255)
                    elif "casco"   in class_name: color = (0, 255, 0)
                    elif "extintor" in class_name: color = (255, 120, 0)
                    # Confirmación multi-frame: armas requieren 3 frames, EPP 2
                    req = 3 if is_weapon else 2
                    sev = "CRÍTICO" if is_weapon else "ALERTA"
                    if _confirm_detection(cam_id, f"sec_{class_name}_{x1//50}_{y1//50}", req, 3.0):
                        show_notification("DETECCIÓN SEGURIDAD",
                                          f"{class_name} — conf: {conf:.2f}", "#ef4444" if is_weapon else "#f59e0b")
                        log_event(cam_id, f"Detección Seguridad: {class_name}", sev,
                                  f"Confianza: {conf:.2f}", conf=conf,
                                  coords=[x1,y1,x2,y2], frame=frame)
                        if is_weapon:
                            _trigger_event_recording(cam_id, class_name, frame)
                    glow_rectangle(frame, x1, y1, x2, y2, color)
                    caja_watchdogs(frame, x1, y1, x2, y2)
                    crosshair(frame, x1, y1, x2, y2)
                    cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    _perf_t_sec[cam_id].append(time.time() - _t_sec_start)

    # Smart recording: start a clip when any configured class was detected
    if _cam_record_mode.get(cam_id, _record_mode) == "smart" and _recording_enabled.get(cam_id) and persons:
        if _throttle(cam_id, "smart_rec", 5):
            _trigger_event_recording(cam_id, 'Smart Detection', frame)

    # CAPA 3: zonas operadores encima de YOLO, con estado de presencia
    _t_ops_start = time.time()
    _render_operator_zones(frame, _clean_detection_frame, config, cam_id, h, w, persons, person_boxes)
    _perf_t_ops[cam_id].append(time.time() - _t_ops_start)

    # CAPA 4: intrusiones en zonas inteligentes (alertas)
    # Always call even when persons is empty so exit events fire for anyone who left the zone
    if persons or _has_smart_zones(config):
        _check_zone_intrusion(config, cam_id, persons, h, w, frame)

    # CAPA 5: objetos inteligentes — solo si están habilitados explícitamente en config
    # Se pasa _clean_detection_frame (sin overlays YOLO) para que el diff contra el fondo
    # no detecte las propias cajas de análisis como "objetos nuevos"
    _t_abnd_start = time.time()
    if config.get("Objeto Abandonado"):
        if _throttle(cam_id, "obj_pf_log_abandono", 10):
            print(f"[OBJ_MOV] process_frame() CAPA5 | cam={cam_id} | Abandonado=True | bg_ok={cam_id in _background_frame} | candidates={len(_foreign_candidates.get(cam_id,[]))}")
        _detect_foreign_objects(_clean_detection_frame, cam_id, person_boxes)
        _render_foreign_objects(frame, cam_id)
    _perf_t_abnd[cam_id].append(time.time() - _t_abnd_start)
    _t_moved_start = time.time()
    if config.get("Objeto Movido"):
        if _throttle(cam_id, "obj_pf_log_movido", 10):
            print(f"[OBJ_MOV] process_frame() CAPA5 | cam={cam_id} | Movido=True | watched={len(_watched_objects.get(cam_id,{}))}")
        _now_m = time.time()
        if _now_m - _moved_obj_last.get(cam_id, 0) >= _MOVED_OBJ_INTERVAL:
            _moved_obj_last[cam_id] = _now_m
            _check_moved_objects(_clean_detection_frame, cam_id, person_boxes)
        _render_watched_and_moved(frame, cam_id)
    _perf_t_moved[cam_id].append(time.time() - _t_moved_start)

    # CAPA 6: contador de ocupación independiente de ByteTrack IDs
    # Solo activa cuando hay detección de Persona habilitada
    if need_gen and config.get("Persona"):
        _get_occupancy(cam_id).update(time.time(), persons)

    _perf_t_total[cam_id].append(time.time() - _t_total_start)
    return frame

# =========================
# PREVIEW EN VIVO
# =========================
def start_preview(camera_id, label, enable_ai=False):

    global active_preview_id
    global preview_camera

    # =========================
    # CERRAR PREVIEW ANTERIOR
    # =========================
    if preview_camera is not None:

        stop_preview()

    preview_camera = cv2.VideoCapture(camera_id)

    # =========================
    # VALIDAR APERTURA REAL
    # =========================
    if not preview_camera.isOpened():

        show_notification(
            "CÁMARA ERROR",
            "No se pudo abrir la cámara.",
            "#ef4444"
        )

        preview_camera.release()

        preview_camera = None

        return

    active_preview_id = camera_id
    _frame_counter_usb = [0]

    def update_frame():

        global preview_camera

        # =========================
        # VALIDAR WIDGET
        # =========================
        if not label.winfo_exists():
            return

        # =========================
        # VALIDAR CÁMARA
        # =========================
        if preview_camera is None:

            label.configure(image="")
            label.imgtk = None

            return

        ret, frame = preview_camera.read()

        # =========================
        # VALIDAR FRAME REAL
        # =========================
        if not ret or frame is None:
            show_notification(
                "SIN SEÑAL",
                "La cámara dejó de responder.",
                "#f59e0b"
            )

            stop_preview()

            return

        else:

            cam_id_str   = str(camera_id)
            config_check = ai_config.get(cam_id_str, {})

            if enable_ai and config_check:
                if cam_id_str not in _ai_lock:
                    _ai_lock[cam_id_str]          = threading.Lock()
                    _ai_thread_active[cam_id_str] = True
                    _ai_raw_buffer[cam_id_str]    = None
                    _ai_frame_buffer[cam_id_str]  = None
                    threading.Thread(
                        target=_ai_worker,
                        args=(cam_id_str,),
                        daemon=True
                    ).start()

                _frame_counter_usb[0] += 1
                if _ai_raw_buffer.get(cam_id_str) is None and _frame_counter_usb[0] % 2 == 0:
                    _ai_raw_buffer[cam_id_str] = frame.copy()

                lock = _ai_lock.get(cam_id_str)
                if lock:
                    with lock:
                        buf = _ai_frame_buffer.get(cam_id_str)
                    if buf is not None:
                        try:
                            if buf.shape == frame.shape:
                                frame = buf
                        except Exception:
                            pass
            else:
                pass  # sin config, mostrar frame crudo

            # =========================
            # OpenCV BGR -> RGB
            # =========================
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # =========================
            # TAMAÑO REAL DEL PANEL
            # =========================
            panel_width = label.winfo_width()
            panel_height = label.winfo_height()

            # fallback inicial
            if panel_width < 50:
                panel_width = label.master.winfo_width() or 700

            if panel_height < 50:
                panel_height = label.master.winfo_height() or 500

            img = Image.fromarray(frame)
            if panel_width > 50 and panel_height > 50:
                img = img.resize((panel_width, panel_height), Image.BILINEAR)
            imgtk = ImageTk.PhotoImage(image=img)

            label.imgtk = imgtk
            label.image = imgtk
            label.configure(image=imgtk)

        global preview_after_id
        preview_after_id = label.after(40, update_frame)
        
    update_frame()

# =========================
# CERRAR PREVIEW
# =========================
def _perf_avg_ms(deq):
    d = list(deq)
    return sum(d) / len(d) * 1000 if d else 0.0

def _load_models_for_cam(cid_str, config, models_ready):
    """Carga los modelos YOLO en un hilo dedicado y señaliza cuando están listos."""
    try:
        if _need_general(config) or config.get("Objeto Abandonado") or config.get("Objeto Movido"):
            _get_track_gen(cid_str)
        if _need_security(config) or config.get("Objeto Movido"):
            _get_track_sec(cid_str)
    finally:
        models_ready.set()

def _ai_worker(cam_id):
    models_ready = _ai_models_ready.get(cam_id)
    if models_ready:
        models_ready.wait()
    print(f"[OBJ_MOV] IA iniciada | cam={cam_id} | worker arrancado")
    while _ai_thread_active.get(cam_id):
        try:
            raw = _ai_raw_buffer.get(cam_id)
            if raw is None:
                time.sleep(0.05)
                continue

            now  = time.time()
            last = _ai_last_process.get(cam_id, 0)
            if now - last < _AI_PROCESS_INTERVAL:
                time.sleep(0.05)
                continue

            _ai_raw_buffer[cam_id]    = None
            _ai_last_process[cam_id] = now

            # escalar a máx 640 px antes de procesar con IA
            h, w = raw.shape[:2]
            scale = 640 / max(w, h)
            if scale < 1.0:
                small = cv2.resize(raw, (int(w * scale), int(h * scale)))
            else:
                small = raw

            # Capturar fondo y registrar objetos en la MISMA resolución que process_frame.
            # De esta forma bg.shape == curr.shape en todas las comparaciones absdiff
            # y las coordenadas de objetos registrados son coherentes con las re-detecciones.
            bg_ts = _ai_pending_bg.get(cam_id)
            if bg_ts and time.time() >= bg_ts:
                del _ai_pending_bg[cam_id]
                print(f"[OBJ_MOV] Fondo capturado | cam={cam_id} | {small.shape[1]}x{small.shape[0]} — auto desde worker")
                _capture_background(cam_id, small)

            obj_ts = _ai_pending_obj.get(cam_id)
            if obj_ts and time.time() >= obj_ts:
                del _ai_pending_obj[cam_id]
                print(f"[OBJ_MOV] Fondo cargado (registro objetos) | cam={cam_id} | bg_ok={cam_id in _background_frame}")
                _register_watched_objects(small.copy(), cam_id)

            # process_frame recibe small.copy() para que los overlays que dibuja
            # no contaminen el buffer 'small' usado para capturar el fondo arriba
            _t_proc = time.time()
            try:
                processed_small = process_frame(small.copy(), cam_id)
                if scale < 1.0:
                    processed = cv2.resize(processed_small, (w, h))
                else:
                    processed = processed_small
            except Exception as _pf_exc:
                print(f"[OBJ_MOV] ERROR en process_frame | cam={cam_id} | {_pf_exc}")
                processed = raw
            _lat = time.time() - _t_proc
            _diag_latency_all.append(_lat)
            _diag_cam_lat[cam_id].append(_lat)

            # ── Monitoreo de rendimiento ──────────────────────────────────
            _t_done = time.time()
            _perf_ai_ts[cam_id].append(_t_done)
            _perf_raw_shape[cam_id] = (h, w)
            _now_perf = _t_done
            if _now_perf - _perf_last_log.get(cam_id, 0) >= 10.0:
                _perf_last_log[cam_id] = _now_perf
                _ai_deq  = list(_perf_ai_ts[cam_id])
                _cap_deq = list(_perf_cap_ts[cam_id])
                _ai_fps  = (len(_ai_deq)  - 1) / (_ai_deq[-1]  - _ai_deq[0])  if len(_ai_deq)  >= 2 and _ai_deq[-1]  > _ai_deq[0]  else 0.0
                _cap_fps = (len(_cap_deq) - 1) / (_cap_deq[-1] - _cap_deq[0]) if len(_cap_deq) >= 2 and _cap_deq[-1] > _cap_deq[0] else 0.0
                _drops   = _perf_drops.get(cam_id, 0)
                # Tiempos promedio de cada etapa (ms)
                _gen_ms   = _perf_avg_ms(_perf_t_gen[cam_id])
                _sec_ms   = _perf_avg_ms(_perf_t_sec[cam_id])
                _ops_ms   = _perf_avg_ms(_perf_t_ops[cam_id])
                _abnd_ms  = _perf_avg_ms(_perf_t_abnd[cam_id])
                _mov_ms   = _perf_avg_ms(_perf_t_moved[cam_id])
                _rec_ms   = _perf_avg_ms(_perf_t_record[cam_id])
                _ren_ms   = _perf_avg_ms(_perf_t_render[cam_id])
                _tot_ms   = _perf_avg_ms(_perf_t_total[cam_id])
                # Captura: intervalo promedio entre frames (ms/frame)
                _cap_interval_ms = (1000.0 / _cap_fps) if _cap_fps > 0 else 0.0
                # Longitudes de colas
                _q_notif  = len(_notification_queue)
                _q_events = len(_events_list)
                _q_prebuf = sum(len(v) for v in _pre_event_buffer.values())
                _q_rawbuf = sum(1 for v in _ai_raw_buffer.values() if v is not None)
                _q_aibuf  = sum(1 for v in _ai_frame_buffer.values() if v is not None)
                print(
                    f"\n[PERF] ═══════════════════════════════════\n"
                    f"[PERF] CAM {cam_id}   src={w}x{h}   drops={_drops}\n"
                    f"[PERF]   CAPTURE    = {_cap_interval_ms:7.1f} ms/frame  ({_cap_fps:.1f} fps)\n"
                    f"[PERF]   YOLO_GEN   = {_gen_ms:7.1f} ms   (general + ByteTrack)\n"
                    f"[PERF]   YOLO_SEC   = {_sec_ms:7.1f} ms   (seguridad)\n"
                    f"[PERF]   OPERATORS  = {_ops_ms:7.1f} ms\n"
                    f"[PERF]   ABANDONED  = {_abnd_ms:7.1f} ms\n"
                    f"[PERF]   MOVED      = {_mov_ms:7.1f} ms\n"
                    f"[PERF]   RECORD     = {_rec_ms:7.1f} ms\n"
                    f"[PERF]   RENDER_UI  = {_ren_ms:7.1f} ms\n"
                    f"[PERF]   TOTAL_AI   = {_tot_ms:7.1f} ms   lat={_lat*1000:.0f}ms  ai={_ai_fps:.1f}fps\n"
                    f"[PERF]   QUEUES: notif={_q_notif} events={_q_events} "
                    f"pre_buf={_q_prebuf} raw_buf={_q_rawbuf} ai_buf={_q_aibuf}\n"
                    f"[PERF] ═══════════════════════════════════"
                )

            lock = _ai_lock.get(cam_id)
            if lock:
                with lock:
                    _ai_frame_buffer[cam_id] = processed
                    _ai_frame_buffer_ts[cam_id] = time.time()  # [H1-FIX] timestamp atómico con el frame

        except Exception:
            pass
        time.sleep(0.02)

def _init_ai_for_cam(cam_id):
    """Fuerza la inicialización completa de todos los sistemas IA para una cámara."""
    cid_str = str(cam_id)
    config  = ai_config.get(cid_str, {})
    if not config:
        print(f"[OBJ_MOV] Configuración cargada | cam={cid_str} | VACÍA — IA no iniciará")
        return False
    print(f"[OBJ_MOV] Configuración cargada | cam={cid_str} | Abandonado={config.get('Objeto Abandonado')} Movido={config.get('Objeto Movido')}")

    # Detener worker anterior si existe y esperar a que salga de su bucle
    if _ai_thread_active.get(cid_str):
        _ai_thread_active[cid_str] = False
        time.sleep(0.15)  # el worker duerme 0.02-0.05 s por iteración; 0.15 s es suficiente

    # Reinicializar todos los buffers y locks
    _ai_lock[cid_str]           = threading.Lock()
    _ai_raw_buffer[cid_str]     = None
    _ai_frame_buffer[cid_str]   = None
    _ai_frame_buffer_ts[cid_str] = time.time()
    _ai_last_process[cid_str]   = 0

    # Activar flag de IA iniciada
    _ai_started[cid_str]      = True
    _ai_thread_active[cid_str] = True
    if _diag_ai_start_time[0] is None:
        _diag_ai_start_time[0] = time.time()

    # Crear evento de sincronización: worker espera hasta que modelos estén listos
    models_ready = threading.Event()
    _ai_models_ready[cid_str] = models_ready

    # Cargar modelos en hilo dedicado — no bloquea UI ni el hilo que llama a esta función
    threading.Thread(
        target=_load_models_for_cam, args=(cid_str, config, models_ready), daemon=True).start()

    # Iniciar worker: bloqueará en models_ready.wait() hasta que los modelos estén disponibles
    threading.Thread(target=_ai_worker, args=(cid_str,), daemon=True).start()

    # Aplicar configuración de grabación desde ai_config
    _rec_mode_val = config.get("recording_mode")
    if _rec_mode_val is None:
        # Compatibilidad con configuraciones guardadas antes del nuevo sistema
        if config.get("Grabación Total"):
            _rec_mode_val = "continuous"
        elif config.get("Grabación por Evento"):
            _rec_mode_val = "hybrid"
        elif config.get("Grabación Inteligente"):
            _rec_mode_val = "smart"
        else:
            _rec_mode_val = "none"
    _rec_pre_s_val  = int(config.get("rec_pre_s",  30))
    _rec_post_s_val = int(config.get("rec_post_s", 30))
    if _rec_mode_val == "continuous":
        _recording_enabled[cid_str] = True
        _cam_record_mode[cid_str]   = "continuous"
    elif _rec_mode_val in ("hybrid", "event"):
        _recording_enabled[cid_str] = True
        _cam_record_mode[cid_str]   = "hybrid"
        _rec_pre_s[cid_str]          = _rec_pre_s_val
        _rec_post_s[cid_str]         = _rec_post_s_val
        _pre_event_buffer[cid_str]   = collections.deque(maxlen=_rec_pre_s_val * 15)
    elif _rec_mode_val == "smart":
        _recording_enabled[cid_str] = True
        _cam_record_mode[cid_str]   = "smart"
        _rec_pre_s[cid_str]          = _rec_pre_s_val
        _rec_post_s[cid_str]         = _rec_post_s_val
        _pre_event_buffer[cid_str]   = collections.deque(maxlen=_rec_pre_s_val * 15)
    else:
        _recording_enabled[cid_str] = False

    print(f"[GRAB] Config cargada    | cam={cid_str} | modo={_rec_mode_val} | pre={_rec_pre_s_val}s | post={_rec_post_s_val}s | enabled={_recording_enabled.get(cid_str, False)}")

    # Activar detección de objetos solo si está habilitado en config para esta cámara
    _background_frame.pop(cid_str, None)
    _foreign_objects.pop(cid_str, None)
    _foreign_candidates.pop(cid_str, None)
    _foreign_yolo_cache.pop(cid_str, None)
    if config.get("Objeto Abandonado"):
        _ai_pending_bg[cid_str] = time.time() + 2.5
        print(f"[OBJ_MOV] IA iniciada | cam={cid_str} | Abandonado activo — fondo se capturará en 2.5s")

    _watched_objects.pop(cid_str, None)
    _moved_alerts.pop(cid_str, None)
    _missing_objects.pop(cid_str, None)
    if config.get("Objeto Movido"):
        # +3.5s en lugar de +2.5s: asegura que el fondo ya esté capturado cuando
        # se registren los objetos, evitando que el diff devuelva siempre cero
        _ai_pending_obj[cid_str] = time.time() + 3.5
        print(f"[OBJ_MOV] IA iniciada | cam={cid_str} | Movido activo — objetos se registrarán en 3.5s")

    # Reinicializar historial de tracking para esta cámara
    keys_to_clear = [k for k in _tracker_history if k[0] == cid_str]
    for k in keys_to_clear:
        _tracker_history[k].clear()

    # Registrar inicio de IA
    rules_active = [k for k, v in config.items() if v is True]
    log_event(cid_str, "IA Iniciada", "INFO",
              f"YOLO+ByteTrack activos. Reglas: {', '.join(rules_active) or 'ninguna'}")

    return True


# ── WATCHDOG ──────────────────────────────────────────────────────────────────

def _watchdog_loop():
    while _watchdog_active[0]:
        try:
            now = time.time()
            for cid_str in list(_ai_started.keys()):
                if not _ai_started.get(cid_str):
                    continue
                # Check for camera freeze (no frame for 8 seconds)
                last = _cam_last_frame.get(cid_str)
                if last and (now - last) > 8:
                    log_event(cid_str, "Cámara Congelada", "ALERTA",
                              f"Sin frames por {int(now - last)}s — revisar cámara")
                    _cam_last_frame[cid_str] = now   # reset to avoid repeated alerts
                # Auto-restart dead AI worker
                if not _ai_thread_active.get(cid_str):
                    log_event(cid_str, "Worker IA Reiniciado", "WARNING",
                              "Thread IA murió — reinicio automático")
                    _ai_thread_active[cid_str] = True
                    threading.Thread(target=_ai_worker, args=(cid_str,), daemon=True).start()
        except Exception:
            pass
        time.sleep(5)

def start_watchdog():
    if not _watchdog_active[0]:
        _watchdog_active[0] = True
        threading.Thread(target=_watchdog_loop, daemon=True).start()

def stop_watchdog():
    _watchdog_active[0] = False


def stop_preview():

    global ffmpeg_process
    global active_preview_id
    global preview_camera
    global preview_label
    global preview_after_id
    global current_ffmpeg
    global preview_running

    for cid in list(_ai_thread_active.keys()):
        _ai_thread_active[cid] = False
    _ai_raw_buffer.clear()
    _ai_frame_buffer.clear()
    _ai_models_ready.clear()

    if preview_after_id is not None and preview_label is not None:

        try:
            preview_label.after_cancel(preview_after_id)
        except:
            pass

    preview_after_id = None

    if preview_camera is not None:

        preview_camera.release()
        preview_camera = None

    if ffmpeg_process is not None:

        try:
            ffmpeg_process.terminate()
            ffmpeg_process.wait(timeout=2)
        except:
            pass

        ffmpeg_process = None

    current_ffmpeg = None
    preview_running = False

    if preview_label is not None:

        try:

            if preview_label.winfo_exists():

                preview_label.configure(image="")
                preview_label.imgtk = None
                preview_label.configure(
                    image="",
                    bg="black"
                )

        except:
            pass

    active_preview_id = None

# =========================
# MULTI-CAMERA LIVE STOP
# =========================
def stop_all_live():
    global _live_cameras, _live_after_ids
    for cid in list(_ai_thread_active.keys()):
        _ai_thread_active[cid] = False
    _ai_raw_buffer.clear()
    _ai_frame_buffer.clear()
    _ai_models_ready.clear()
    _ai_started.clear()
    _ai_pending_bg.clear()
    _ai_pending_obj.clear()
    for running, cap in _live_after_ids:
        try:
            running[0] = False
        except Exception:
            pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    _live_after_ids.clear()
    _live_cameras.clear()

# =========================
# 🔐 ADMIN DINÁMICO
# =========================
def get_admin_password():
    for u in users_data:
        if u["user"] == "admin":
            return u["password"]
    return None
# =========================
# DETECTAR USB
# =========================
def get_usb_drives():

    drives = []

    bitmask = ctypes.windll.kernel32.GetLogicalDrives()

    for letter in string.ascii_uppercase:

        if bitmask & 1:

            drive = f"{letter}:\\"

            drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive)

            # 2 = USB removable
            if drive_type == 2:
                drives.append(drive)

        bitmask >>= 1

    return drives


# =========================
# CREAR ROOT USB KEY
# =========================
def create_root_key(admin_password):

    # =========================
    # VALIDAR ADMIN
    # =========================
    if hash_password(admin_password) != get_admin_password():

        show_notification(
            "ADMIN ERROR",
            "Contraseña admin incorrecta",
            "#ef4444"
        )

        return

    # =========================
    # DETECTAR USB
    # =========================
    drives = get_usb_drives()

    if not drives:

        show_notification(
            "USB ERROR",
            "No se detectó ninguna USB",
            "#ef4444"
        )

        return

    usb = drives[0]

    # =========================
    # TOKEN ROOT
    # =========================
    token = generate_root_token()

    data = {
        "system": "VIGILANT_PRO",
        "token": token
    }

    # =========================
    # GUARDAR TOKEN LOCAL
    # =========================
    with open(ROOT_KEY_FILE, "w") as f:
        json.dump(data, f, indent=4)

    # =========================
    # GUARDAR EN USB
    # =========================
    usb_file = os.path.join(
        usb,
        ".vpro_root.key"
    )

    with open(usb_file, "w") as f:
        json.dump(data, f, indent=4)

    register_event(
        current_user,
        "CREATE_ROOT_USB",
        "OK",
        f"USB autorizada: {usb}"
    )

    show_notification(
        "ROOT KEY CREADA",
        f"USB autorizada: {usb}",
        "#22c55e"
    )

# =========================
# 🔐 AUTH
# =========================
def authenticate(user, password):
    hashed = hash_password(password)
    for u in users_data:
        if u["user"] == user and u["password"] == hashed:
            return True
    return False 

# =========================
# LOGIN ROOT USB
# =========================
def login_with_root_usb():

    global current_user
    global session_start_time

    # =========================
    # EXISTE ROOT LOCAL
    # =========================
    if not os.path.exists(ROOT_KEY_FILE):

        show_notification(
            "ROOT USB",
            "No existe una llave root registrada.",
            "#ef4444"
        )

        return

    # =========================
    # LEER ROOT LOCAL
    # =========================
    with open(ROOT_KEY_FILE, "r") as f:

        local_root = json.load(f)

    # =========================
    # BUSCAR USB
    # =========================
    drives = get_usb_drives()

    if not drives:

        show_notification(
            "USB ERROR",
            "No se detectó ninguna USB.",
            "#ef4444"
        )

        return

    # =========================
    # RECORRER USBs
    # =========================
    for usb in drives:

        usb_file = os.path.join(
            usb,
            ".vpro_root.key"
        )

        # =========================
        # EXISTE KEY
        # =========================
        if os.path.exists(usb_file):

            try:

                with open(usb_file, "r") as f:

                    usb_root = json.load(f)

                # =========================
                # VALIDAR TOKEN
                # =========================
                if usb_root["token"] == local_root["token"]:

                    current_user = "ROOT_USB"

                    session_start_time = datetime.now()

                    build_sidebar()

                    register_event(
                        "ROOT_USB",
                        "ROOT_LOGIN",
                        "OK",
                        f"Acceso ROOT desde USB: {usb}"
                    )

                    lock_screen.place_forget()

                    show_notification(
                        "ROOT ACCESS",
                        "Acceso ROOT concedido.",
                        "#22c55e"
                    )

                    show_inicio()
                    return

            except Exception as e:

                show_notification(
                    "USB ERROR",
                    str(e),
                    "#ef4444"
                )

                return

    # =========================
    # NO COINCIDE
    # =========================
    register_event(
        "UNKNOWN",
        "ROOT_LOGIN",
        "FAILED",
        "USB root inválida"
    )

    show_notification(
        "ROOT DENIED",
        "USB no autorizada.",
        "#ef4444"
    )

# =========================
# SHIFT MONITOR
# =========================
_shift_monitor_id = [None]

def _update_shift_display():
    global _active_shift
    prev = _active_shift
    _active_shift = get_active_shift()
    # Update sidebar label
    if _shift_lbl_ref[0]:
        try:
            if _shift_lbl_ref[0].winfo_exists():
                sh = shifts_config.get(_active_shift, {})
                _shift_lbl_ref[0].config(
                    text=f"{sh.get('icono','☀')} Turno {sh.get('nombre',_active_shift.title())}")
        except Exception:
            pass
    # Notify on transition
    if _active_shift != prev:
        sh = shifts_config.get(_active_shift, {})
        show_notification(
            "CAMBIO DE TURNO",
            f"Turno activo: {sh.get('icono','')} {sh.get('nombre',_active_shift.title())}",
            "#f59e0b")
        register_event(
            current_user or "sistema", "CAMBIO_TURNO", "OK",
            f"Turno anterior: {prev} → nuevo: {_active_shift}")
    _shift_monitor_id[0] = root.after(60_000, _update_shift_display)

# =========================
# LOGOUT
# =========================
def logout():

    global current_user
    global session_start_time
    global selected_cameras


    global preview_camera
    global preview_label
    global active_preview_id
    global preview_after_id
    global ffmpeg_process
    global current_ffmpeg

    duration = "Desconocida"

    if session_start_time is not None:

        delta = datetime.now() - session_start_time

        minutes = int(delta.total_seconds() // 60)

        duration = f"{minutes} minutos"

    logs = load_audit_log()

    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": current_user,
        "action": "LOGOUT",
        "status": "OK",
        "duration": duration,
        "details": "Cierre de sesión"
    }

    logs.append(event)

    save_audit_log(logs)

    _stop_all_writers()
    stop_preview()

    current_user = None
    session_start_time = None
    save_selected_cameras()
    selected_cameras = []

    preview_camera = None
    preview_label = None

    active_preview_id = None
    preview_after_id = None

    ffmpeg_process = None
    current_ffmpeg = None

    build_sidebar()

    lock_screen.place(
        relwidth=1,
        relheight=1
    )

    show_notification(
        "SESIÓN CERRADA",
        "La sesión fue finalizada correctamente.",
        "#22d3ee"
    )
# =========================
# DPI FIX WINDOWS
# =========================
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except:
    pass

root = tk.Tk()
root.title("Vigilant Pro")
root.state("zoomed")
root.configure(bg="#020617")

# =========================
# LOAD LOGOS
# =========================
def load_logo(path, size):
    img = Image.open(path).convert("RGBA")
    img = img.resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(img)

logo_img = load_logo(LOGO_PATH, (32, 32))
big_logo = load_logo(BIG_LOGO_PATH, (120, 120))

# =========================
# SIDEBAR
# =========================
sidebar = tk.Frame(root, bg="#020617", width=240)
sidebar.pack(side="left", fill="y")

header = tk.Frame(sidebar, bg="#020617")
header.pack(fill="x", pady=20, padx=16)

row = tk.Frame(header, bg="#020617")
row.pack(fill="x")

tk.Label(row, image=logo_img, bg="#020617").pack(side="left")

text_container = tk.Frame(row, bg="#020617")
text_container.pack(side="left", padx=10)

tk.Label(text_container, text="VIGILANT PRO",
         fg="#e5e7eb", bg="#020617",
         font=("Segoe UI", 13, "bold")).pack(anchor="w")

tk.Label(text_container, text="Sistema de vigilancia",
         fg="#64748b", bg="#020617",
         font=("Segoe UI", 9)).pack(anchor="w")

# Shift indicator (updated by _update_shift_display)
_sh = shifts_config.get(_active_shift, {})
_shift_lbl_ref[0] = tk.Label(
    text_container,
    text=f"{_sh.get('icono','☀')} Turno {_sh.get('nombre',_active_shift.title())}",
    fg="#f59e0b", bg="#020617",
    font=("Segoe UI", 8, "bold"))
_shift_lbl_ref[0].pack(anchor="w")

# =========================
# MAIN
# =========================
main = tk.Frame(root, bg="#020617")
main.pack(side="left", fill="both", expand=True)

def clear_main():
    _stop_all_writers()
    stop_preview()
    stop_all_live()
    try:
        root.after_cancel("all")
    except Exception:
        pass
    for widget in main.winfo_children():
        try:
            widget.place_forget()
        except Exception:
            pass
        try:
            widget.destroy()
        except Exception:
            pass
    main.update_idletasks()

def show_users():
    clear_main()

    is_admin = current_user == "admin"

    if not is_admin:
        register_event(current_user, "ACCESS_USERS_DENIED", "FAILED",
                       "Intento de acceso a gestión de usuarios")
        show_notification("ACCESO DENEGADO",
                          "Solo el administrador puede gestionar usuarios.", "#ef4444")
        return

    BG     = "#020B25"
    PANEL  = "#07142F"
    BORDER = "#132B57"

    ROLE_BADGE = {
        "Administrador": ("#8B0000", "#FFFFFF"),
        "Operador":      ("#2D7FF9", "#FFFFFF"),
        "Visualizador":  ("#6C757D", "#FFFFFF"),
    }

    container = tk.Frame(main, bg=BG)
    container.pack(fill="both", expand=True, padx=24, pady=16)

    # ── Page header ───────────────────────────────────────────────────────────
    hdr = tk.Frame(container, bg=BG)
    hdr.pack(fill="x", pady=(0, 12))
    tk.Label(hdr, text="Usuarios", fg="#FFFFFF", bg=BG,
             font=("Segoe UI", 20, "bold")).pack(side="left", anchor="w")
    tk.Label(hdr, text="  Gestión de acceso al sistema",
             fg="#9DB2D4", bg=BG, font=("Segoe UI", 10)).pack(side="left", pady=(6, 0))

    # ── Top row: form card + USB card ─────────────────────────────────────────
    top_row = tk.Frame(container, bg=BG)
    top_row.pack(fill="x", pady=(0, 10))

    # ── Form card ─────────────────────────────────────────────────────────────
    form_card = tk.Frame(top_row, bg=PANEL, height=120,
                         highlightbackground=BORDER, highlightthickness=1)
    form_card.pack(side="left", fill="x", expand=True, padx=(0, 8))
    form_card.pack_propagate(False)

    tk.Label(form_card, text="Crear / Editar Usuario",
             fg="#2D7FF9", bg=PANEL,
             font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=16, pady=(10, 6))

    fields_row = tk.Frame(form_card, bg=PANEL)
    fields_row.pack(fill="x", padx=16, pady=(0, 10))

    def _field(parent, label, is_pw=False):
        cell = tk.Frame(parent, bg=PANEL)
        cell.pack(side="left", padx=(0, 10), fill="x", expand=True)
        tk.Label(cell, text=label, fg="#9DB2D4", bg=PANEL,
                 font=("Segoe UI", 8)).pack(anchor="w")
        e = tk.Entry(cell, bg="#0A1828", fg="white",
                     insertbackground="white", relief="flat",
                     show="*" if is_pw else "",
                     font=("Segoe UI", 9),
                     highlightbackground=BORDER, highlightthickness=1,
                     highlightcolor="#2D7FF9")
        e.pack(fill="x", ipady=5, pady=(2, 0))
        return e

    entry_user  = _field(fields_row, "Usuario")
    entry_pass  = _field(fields_row, "Contraseña", is_pw=True)

    role_cell = tk.Frame(fields_row, bg=PANEL)
    role_cell.pack(side="left", padx=(0, 10))
    tk.Label(role_cell, text="Rol", fg="#9DB2D4", bg=PANEL,
             font=("Segoe UI", 8)).pack(anchor="w")
    role_var = tk.StringVar(value="Operador")
    role_menu = tk.OptionMenu(role_cell, role_var,
                               "Administrador", "Operador", "Visualizador")
    role_menu.config(bg="#0A1828", fg="white", relief="flat",
                     font=("Segoe UI", 9), highlightthickness=0,
                     activebackground="#1e3a5f", activeforeground="white")
    role_menu["menu"].config(bg="#0A1828", fg="white")
    role_menu.pack(fill="x", pady=(2, 0))

    entry_pin   = _field(fields_row, "PIN Monitoreo", is_pw=True)
    entry_admin = _field(fields_row, "Clave Admin",   is_pw=True)

    selected = {"i": None}

    def save_user():
        user     = entry_user.get().strip()
        password = entry_pass.get()
        pin_raw  = entry_pin.get().strip()
        role     = role_var.get() or "Operador"
        if not user:
            show_notification("CAMPO VACÍO", "Ingresa un nombre de usuario.", "#FF6D00")
            return
        if hash_password(entry_admin.get()) != get_admin_password():
            show_notification("CLAVE INVÁLIDA", "Contraseña de admin incorrecta.", "#FF3D57")
            return
        if selected["i"] is None:
            users_data.append({
                "user": user, "name": user,
                "password": hash_password(password or "1234"),
                "pin":      hash_password(pin_raw or "1234"),
                "role": role, "status": "Activo",
            })
            register_event(current_user, "CREATE_USER", "OK", f"Usuario creado: {user}")
        else:
            u = users_data[selected["i"]]
            if password:
                u["password"] = hash_password(password)
            if pin_raw:
                u["pin"] = hash_password(pin_raw)
            elif "pin" not in u:
                u["pin"] = hash_password("1234")
            u["role"] = role
            register_event(current_user, "EDIT_USER", "OK", f"Usuario modificado: {user}")
        save_users()
        selected["i"] = None
        show_users()

    def select_user(i):
        u = users_data[i]
        entry_user.delete(0, "end"); entry_user.insert(0, u["user"])
        entry_pass.delete(0, "end")
        entry_pin.delete(0, "end")
        role_var.set(u.get("role", "Operador"))
        selected["i"] = i

    def delete_user(i):
        if users_data[i]["user"] == "admin":
            return
        if not messagebox.askyesno("Confirmar",
                                    f"¿Eliminar usuario '{users_data[i]['user']}'?"):
            return
        deleted = users_data[i]["user"]
        register_event(current_user, "DELETE_USER", "OK", f"Usuario eliminado: {deleted}")
        del users_data[i]
        save_users()
        show_users()

    tk.Button(fields_row, text="Guardar Usuario",
              bg="#2D7FF9", fg="white", relief="flat",
              font=("Segoe UI", 9, "bold"), padx=14, pady=5,
              cursor="hand2", command=save_user
              ).pack(side="left", anchor="s", padx=(4, 0), pady=(2, 0))

    # ── USB Recovery card (compact) ───────────────────────────────────────────
    usb_card = tk.Frame(top_row, bg=PANEL, width=224, height=120,
                        highlightbackground=BORDER, highlightthickness=1)
    usb_card.pack(side="left", fill="y")
    usb_card.pack_propagate(False)

    tk.Label(usb_card, text="USB Root Recovery",
             fg="#FF3D57", bg=PANEL,
             font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
    tk.Label(usb_card,
             text="Acceso de emergencia mediante\nllave USB autorizada.",
             fg="#9DB2D4", bg=PANEL,
             font=("Segoe UI", 8), justify="left").pack(anchor="w", padx=14)
    tk.Button(usb_card, text="Crear USB Root Key",
              bg="#7f1d1d", fg="white", relief="flat",
              font=("Segoe UI", 8, "bold"), padx=12, pady=4,
              cursor="hand2",
              command=lambda: create_root_key(entry_admin.get())
              ).pack(anchor="w", padx=14, pady=(8, 10))

    # ── User table ────────────────────────────────────────────────────────────
    tbl_wrap = tk.Frame(container, bg=PANEL,
                        highlightbackground=BORDER, highlightthickness=1)
    tbl_wrap.pack(fill="both", expand=True)

    HDR_COLS = ["Usuario", "Nombre completo", "Rol", "Estado", "Acciones"]
    HDR_W    = [16, 24, 16, 12, 1]

    hdr_line = tk.Frame(tbl_wrap, bg="#0A1828")
    hdr_line.pack(fill="x")
    for txt, w_ in zip(HDR_COLS, HDR_W):
        tk.Label(hdr_line, text=txt, fg="#9DB2D4", bg="#0A1828",
                 font=("Segoe UI", 9, "bold"),
                 width=w_ if w_ > 1 else None, anchor="w"
                 ).pack(side="left", padx=(14, 0), pady=10)

    scroll_outer = tk.Frame(tbl_wrap, bg=PANEL)
    scroll_outer.pack(fill="both", expand=True)
    canvas = tk.Canvas(scroll_outer, bg=PANEL, highlightthickness=0)
    vsb    = tk.Scrollbar(scroll_outer, orient="vertical", command=canvas.yview)
    inner  = tk.Frame(canvas, bg=PANEL)
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
    canvas.configure(yscrollcommand=vsb.set)
    canvas.bind("<MouseWheel>",
                lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    def _make_tooltip(btn, text):
        tip = [None]
        def _show(e):
            tip[0] = tk.Label(root, text=text, bg="#0A1828", fg="white",
                               font=("Segoe UI", 8), relief="flat",
                               highlightbackground=BORDER, highlightthickness=1)
            tip[0].place(x=e.x_root - root.winfo_rootx() + 14,
                         y=e.y_root - root.winfo_rooty() - 26)
        def _hide(e):
            if tip[0]:
                tip[0].destroy()
                tip[0] = None
        btn.bind("<Enter>", _show)
        btn.bind("<Leave>", _hide)

    for i, user in enumerate(users_data):
        row_bg = "#0A1828" if i % 2 == 0 else PANEL

        row = tk.Frame(inner, bg=row_bg, height=40)
        row.pack(fill="x")
        row.pack_propagate(False)
        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x")

        # Username
        tk.Label(row, text=user["user"], fg="#FFFFFF", bg=row_bg,
                 font=("Segoe UI", 9, "bold"), width=HDR_W[0],
                 anchor="w").pack(side="left", padx=(14, 0))

        # Full name
        tk.Label(row, text=user.get("name", user["user"]), fg="#9DB2D4", bg=row_bg,
                 font=("Segoe UI", 9), width=HDR_W[1],
                 anchor="w").pack(side="left", padx=(14, 0))

        # Role badge
        role      = user.get("role", "Operador")
        bdg_bg, bdg_fg = ROLE_BADGE.get(role, ("#6C757D", "#FFFFFF"))
        badge_holder = tk.Frame(row, bg=row_bg)
        badge_holder.pack(side="left", padx=(14, 0))
        tk.Label(badge_holder, text=role, fg=bdg_fg, bg=bdg_bg,
                 font=("Segoe UI", 8, "bold"), padx=10, pady=2
                 ).pack(anchor="w", pady=10)

        # Status
        status      = user.get("status", "Activo")
        st_icon     = "🟢" if status == "Activo" else "🔴"
        st_color    = "#00C853" if status == "Activo" else "#FF3D57"
        tk.Label(row, text=f"{st_icon}  {status}", fg=st_color, bg=row_bg,
                 font=("Segoe UI", 9), width=HDR_W[3],
                 anchor="w").pack(side="left", padx=(14, 0))

        # Action buttons
        act = tk.Frame(row, bg=row_bg)
        act.pack(side="left", padx=(14, 0))

        if is_admin:
            edit_btn = tk.Button(act, text="✏",
                                  bg="#2D7FF9", fg="white", relief="flat",
                                  width=3, height=1, font=("Segoe UI", 10),
                                  cursor="hand2",
                                  command=lambda idx=i: select_user(idx))
            edit_btn.pack(side="left", padx=(0, 5))
            _make_tooltip(edit_btn, "Editar usuario")

        if is_admin and user["user"] != "admin":
            del_btn = tk.Button(act, text="🗑",
                                 bg="#C62828", fg="white", relief="flat",
                                 width=3, height=1, font=("Segoe UI", 10),
                                 cursor="hand2",
                                 command=lambda idx=i: delete_user(idx))
            del_btn.pack(side="left")
            _make_tooltip(del_btn, "Eliminar usuario")

        # Hover effect (skip badge_holder children to preserve badge color)
        def _enter(e, r=row, bg=row_bg):
            r.config(bg="#0D1F47")
            for w in r.winfo_children():
                try:
                    if w.cget("bg") == bg:
                        w.config(bg="#0D1F47")
                    for ww in w.winfo_children():
                        if ww.cget("bg") == bg:
                            ww.config(bg="#0D1F47")
                except Exception:
                    pass

        def _leave(e, r=row, bg=row_bg):
            r.config(bg=bg)
            for w in r.winfo_children():
                try:
                    if w.cget("bg") in ("#0D1F47", bg):
                        w.config(bg=bg)
                    for ww in w.winfo_children():
                        if ww.cget("bg") in ("#0D1F47", bg):
                            ww.config(bg=bg)
                except Exception:
                    pass

        row.bind("<Enter>", _enter)
        row.bind("<Leave>", _leave)
        for child in row.winfo_children():
            child.bind("<Enter>", _enter)
            child.bind("<Leave>", _leave)


# =========================
# (camaras)
# =========================
next_button = None
def show_cameras():
    global preview_label
    global selected_cameras
    global next_button
    clear_main()
    global camera_checkboxes
    camera_checkboxes.clear()

    # Sanear selected_cameras: conservar IDs USB presentes e IDs RTSP registrados
    usb_scan         = scan_usb_cameras()
    rtsp_ids_present = {c["id"] for c in rtsp_cameras}
    if usb_scan:
        usb_ids_present = {cam["id"] for cam in usb_scan}
        valid_selected  = [c for c in selected_cameras
                           if isinstance(c, int) and
                           (c in usb_ids_present or c in rtsp_ids_present)]
    else:
        # scan returned nothing — keep valid-range USB IDs without evicting them
        valid_selected = [c for c in selected_cameras
                          if isinstance(c, int) and
                          ((0 <= c < 20) or c in rtsp_ids_present)]
    if len(valid_selected) != len(selected_cameras):
        selected_cameras[:] = valid_selected
        save_selected_cameras()

    container = tk.Frame(main, bg="#020617")
    container.pack(fill="both", expand=True, padx=30, pady=25)

    # ===== TÍTULO =====
    tk.Label(container, text="Cámaras",
             fg="#e5e7eb", bg="#020617",
             font=("Segoe UI", 20, "bold")).pack(anchor="w")

    tk.Label(container, text="Gestiona y configura las cámaras del sistema.",
             fg="#64748b", bg="#020617",
             font=("Segoe UI", 11)).pack(anchor="w", pady=(0, 20))

    # =========================
    # 🟦 CARD FORM  (RTSP — deshabilitado, solo webcams locales)
    # =========================
    card_form = tk.Frame(container, bg="#0b1220")
    card_form.pack(fill="x", pady=(0, 10))

    form = tk.Frame(card_form, bg="#0b1220", padx=30, pady=25)
    form.pack(fill="x")

    tk.Label(form, text="Agregar nueva cámara",
             fg="#3b82f6", bg="#0b1220",
             font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=7, sticky="w", pady=(0, 15))

    labels = [
        "Ubicación",
        "IP Cámara",
        "Usuario",
        "Contraseña",
        "Transporte"
    ]
    for i, text in enumerate(labels):
        tk.Label(form, text=text, bg="#0b1220", fg="#94a3b8")\
            .grid(row=1, column=i, sticky="w", padx=10)

    def create_entry(parent, width=28, show=None):
        frame = tk.Frame(parent, bg="#020617",
                         highlightbackground="#1f2937",
                         highlightthickness=1)
        entry = tk.Entry(frame,
                         bg="#020617",
                         fg="white",
                         insertbackground="white",
                         relief="flat",
                         width=width,
                         show=show)
        entry.pack(padx=8, pady=6)
        return frame, entry

    # =========================
    # 🔧 LAYOUT CORRECTO (FUERA DE create_entry)
    # =========================

    for i in range(7):
        form.grid_columnconfigure(i, weight=1)

    # INPUTS
    e1_frame, e1 = create_entry(form, 18)
    e2_frame, e2 = create_entry(form, 22)
    e3_frame, e3 = create_entry(form, 18)
    e4_frame, e4 = create_entry(form, 18, show="*")

    e1_frame.grid(row=2, column=0, padx=5, pady=10, sticky="ew")
    e2_frame.grid(row=2, column=1, padx=5, pady=10, sticky="ew")
    e3_frame.grid(row=2, column=2, padx=5, pady=10, sticky="ew")
    e4_frame.grid(row=2, column=3, padx=5, pady=10, sticky="ew")

    # =========================
    # TRANSPORTE RTSP
    # =========================
    transport_var = tk.StringVar(value="udp")

    transport_frame = tk.Frame(
        form,
        bg="#0b1220"
    )

    transport_frame.grid(
        row=2,
        column=4,
        padx=5,
        pady=10,
        sticky="w"
    )

    tk.Radiobutton(
        transport_frame,
        text="UDP",
        variable=transport_var,
        value="udp",
        bg="#0b1220",
        fg="white",
        selectcolor="#020617",
        activebackground="#0b1220",
        activeforeground="white"
    ).pack(side="left")

    tk.Radiobutton(
        transport_frame,
        text="TCP",
        variable=transport_var,
        value="tcp",
        bg="#0b1220",
        fg="white",
        selectcolor="#020617",
        activebackground="#0b1220",
        activeforeground="white"
    ).pack(side="left", padx=(10, 0))

    connect_button = tk.Button(
        form,
        text="Conectar",
        bg="#2563eb",
        fg="white",
        relief="flat",
        command=lambda: handle_rtsp_preview(
            e2,
            e3,
            e4,
            preview_label,
            transport_var.get(),
            save_button
        )
    )

    connect_button.grid(
        row=2,
        column=5,
        padx=5,
        sticky="ew"
    )

    # BOTONES
    def save_rtsp_camera():

        global rtsp_cameras

        location = e1.get().strip()
        ip = e2.get().strip()
        user = e3.get().strip()
        password = e4.get().strip()

        rtsp_url = build_rtsp_url(
            ip,
            user,
            password
        )

        camera_data = {
            "id": len(rtsp_cameras) + 100,
            "name": "RTSP CAMERA",
            "alias": location,
            "rtsp": rtsp_url,
            "transport": transport_var.get(),
            "status": "Funcional"
        }

        rtsp_cameras.append(camera_data)

        save_rtsp_cameras(rtsp_cameras)

        show_notification(
            "CÁMARA GUARDADA",
            "La cámara WiFi fue registrada.",
            "#22c55e"
        )

        show_cameras()

    save_button = tk.Button(
        form,
        text="Guardar",
        bg="#1e293b",
        fg="#64748b",
        relief="flat",
        state="disabled",
        command=save_rtsp_camera
    )

    save_button.grid(
        row=2,
        column=6,
        padx=7,
        sticky="ew"
    )

    def _abrir_diagnostico():
        ip       = e2.get().strip()
        user     = e3.get().strip()
        password = e4.get().strip()
        if not ip or not user or not password:
            show_notification("CAMPOS INCOMPLETOS",
                              "Completa IP, usuario y contraseña para diagnosticar.",
                              "#f59e0b")
            return
        rtsp_url = build_rtsp_url(ip, user, password)
        open_rtsp_diagnostico(rtsp_url)

    tk.Button(
        form,
        text="Diagnóstico",
        bg="#7c3aed",
        fg="white",
        relief="flat",
        command=_abrir_diagnostico
    ).grid(row=3, column=5, columnspan=2, padx=5, pady=(4, 0), sticky="ew")

    # =========================
    # 🟦 CARD TABLA
    # =========================
    card_table = tk.Frame(container, bg="#0b1220")
    card_table.pack(fill="both", expand=True, pady=15)

    content = tk.Frame(card_table, bg="#0b1220")
    content.grid_columnconfigure(
        0,
        weight=1,
        minsize=850
    )

    content.grid_columnconfigure(
        1,
        weight=0,
        minsize=430
    )
    content.grid_rowconfigure(0, weight=1)
    content.pack(fill="both", expand=True)

    table = tk.Frame(
        content,
        bg="#0b1220",
        padx=30,
        pady=25
    )

    # =========================
    # PREVIEW PANEL
    # =========================
    preview_panel = tk.Frame(
        content,
        bg="#020617"
    )

    preview_panel.grid(
        row=0,
        column=1,
        sticky="nsew",
        padx=20,
        pady=20
    )

    tk.Label(
        preview_panel,
        text="Preview cámara",
        fg="#3b82f6",
        bg="#020617",
        font=("Segoe UI", 12, "bold")
    ).pack(pady=(10, 5))


    preview_panel.pack_propagate(False)

    preview_label = tk.Label(
        preview_panel,
        bg="black",
        width=380,
        height=240
    )
    preview_label.pack(padx=10, pady=10)
    preview_label.pack_propagate(False)

    table.grid(
        row=0,
        column=0,
        sticky="nsew"
    )
    # =========================
    # EXPANSIÓN TABLA
    # =========================
    for col in range(6):

        table.grid_columnconfigure(
            col,
            weight=1
        )

    tk.Label(table, text="Lista de cámaras",
             fg="#3b82f6", bg="#0b1220",
             font=("Segoe UI", 13, "bold")).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0,10))

    headers = [
    "",
    "Cámara",
    "Ubicacion",
    "RTSP",
    "Estado",
    "Acciones"
]

    for col, h in enumerate(headers):
        tk.Label(table, text=h,
                 fg="#94a3b8", bg="#0b1220")\
            .grid(row=1, column=col, padx=25, pady=10)

    data = scan_usb_cameras() + rtsp_cameras
    for i, cam in enumerate(data, start=2):

        bg = "#0b1220" if i % 2 == 0 else "#020617"

        # =========================
        # SELECTOR
        # =========================
        selected_var = tk.BooleanVar(value=cam["id"] in selected_cameras)
        camera_checkboxes.append(selected_var)

        def toggle_camera(
            cam_id=cam["id"],
            var=selected_var
        ):

            if var.get():

                if cam_id not in selected_cameras:

                    # máximo 4
                    if len(selected_cameras) >= 4:

                        var.set(False)

                        show_notification(
                            "LÍMITE",
                            "Máximo 4 cámaras.",
                            "#f59e0b"
                        )

                        return

                    selected_cameras.append(cam_id)
                    save_selected_cameras()
                    print(selected_cameras)

                    next_button.configure(
                        state="normal",
                        bg="#2563eb",
                        fg="white"
                    )



            else:

                if cam_id in selected_cameras:

                    selected_cameras.remove(cam_id)
                    save_selected_cameras()

                if len(selected_cameras) == 0:

                    next_button.configure(
                        state="disabled",
                        bg="#1e293b",
                        fg="#64748b"
                    )

        check = tk.Checkbutton(
            table,
            variable=selected_var,
            command=toggle_camera,
            bg=bg,
            activebackground=bg,
            selectcolor="#2563eb",
            fg="white",
            activeforeground="white",
            highlightthickness=0,
            bd=0,
            relief="flat"
        )

        check.grid(
            row=i,
            column=0,
            padx=10
        )

        # =========================
        # NOMBRE
        # =========================
        tk.Label(
            table,
            text=cam["name"],
            fg="white",
            bg=bg
        ).grid(row=i, column=1, padx=25, pady=10)
        # =========================
        # ALIAS EDITABLE
        # =========================
        alias_label = tk.Label(
            table,
            text=cam["alias"] if cam["alias"] else "Asignar ubicacion",
            fg="#cbd5f5",
            bg=bg,
            cursor="hand2"
        )

        alias_label.grid(row=i, column=2, padx=25)

        # =========================

        def start_rename(
            event,
            cam_id=cam["id"],
            current_alias=cam["alias"],
            row=i,
            current_label=alias_label
        ):

            # Ocultar label actual
            current_label.grid_remove()

            # Entry inline
            entry = tk.Entry(
                table,
                bg="#020617",
                fg="white",
                insertbackground="white",
                relief="flat",
                width=18
            )

            entry.grid(row=row, column=2, padx=25)

            # Texto actual
            entry.insert(0, current_alias)

            entry.focus()

            # =========================
            # GUARDAR
            # =========================
            def save_alias(event=None):

                camera_names[str(cam_id)] = entry.get()

                save_camera_names()

                show_cameras()

            # ENTER = guardar
            entry.bind("<Return>", save_alias)

            # ESC = cancelar
            def cancel(event=None):

                entry.destroy()

                current_label.grid()

            entry.bind("<Escape>", cancel)

        # DOBLE CLICK
        alias_label.bind("<Double-Button-1>", start_rename)
        

        # =========================
        # ÍNDICE / URL RTSP
        # =========================
        if "rtsp" in cam:
            rtsp_short = cam["rtsp"]
            # ocultar credenciales: rtsp://user:pass@ip → rtsp://***@ip
            import re as _re
            rtsp_display = _re.sub(r'(rtsp://)([^@]+@)', r'\1***@', rtsp_short)
            col3_text = rtsp_display[:40] + ("…" if len(rtsp_display) > 40 else "")
        else:
            col3_text = f"Índice {cam['id']}"
        tk.Label(
            table,
            text=col3_text,
            fg="#cbd5f5",
            bg=bg
        ).grid(row=i, column=3, padx=25)

        # =========================
        # COLOR ESTADO
        # =========================
        if cam["status"] == "Funcional":
            color = "#22c55e"

        elif cam["status"] == "Sin señal":
            color = "#f59e0b"

        else:
            color = "#ef4444"

        # =========================
        # ESTADO
        # =========================
        tk.Label(
            table,
            text="● " + cam["status"],
            fg=color,
            bg=bg
        ).grid(row=i, column=4)

        # =========================
        # ACCIONES
        # =========================
        actions = tk.Frame(table, bg=bg)
        actions.grid(row=i, column=5)

        def open_camera_preview(cam_data=cam):

            # =========================
            # RTSP CAMERA
            # =========================
            if "rtsp" in cam_data:

                transport = cam_data.get("transport", "tcp")

                try:

                    start_rtsp_preview(
                        cam_data["rtsp"],
                        preview_label,
                        transport,
                        save_button,
                        real_cam_id=cam_data["id"]
                    )

                except:

                    fallback = "tcp" if transport == "udp" else "udp"

                    start_rtsp_preview(
                        cam_data["rtsp"],
                        preview_label,
                        fallback,
                        save_button,
                        real_cam_id=cam_data["id"]
                    )

            # =========================
            # USB CAMERA
            # =========================
            else:

                start_preview(
                    cam_data["id"],
                    preview_label
                )


        tk.Button(
            actions,
            text="👁",
            bg="#1f2937",
            fg="white",
            command=open_camera_preview
        ).pack(side="left", padx=3)

        tk.Button(
            actions,
            text="↻",
            bg="#1f2937",
            fg="white"
        ).pack(side="left", padx=3)


    # =========================
    # 🟦 STATS (CÍRCULO ABAJO)
    # =========================
    stats = tk.Frame(container, bg="#0b1220", padx=20, pady=20)
    stats.pack(fill="x", pady=10)

    def stat(text, value, color):
        f = tk.Frame(stats, bg="#0b1220", height=90)
        f.pack(side="left", expand=True, fill="both")
        f.pack_propagate(False)

        # TEXTO
        tk.Label(f, text=text,
                fg="#94a3b8", bg="#0b1220",
                font=("Segoe UI", 10)).pack(pady=(5,0))

        # CONTENEDOR DEL CÍRCULO
        circle = tk.Canvas(f,
                        width=50,
                        height=50,
                        bg="#0b1220",
                        highlightthickness=0)
        circle.pack(pady=(3, 3))

        # CÍRCULO
        circle.create_oval(5, 5, 45, 45,
                        fill=color,
                        outline="")

        # TEXTO (NÚMERO)
        circle.create_text(25, 25,
                        text=str(value),
                        fill="white",
                        font=("Segoe UI", 16, "bold"))
        

# =========================
# ESTADÍSTICAS REALES
# =========================
    total_cams = len(data)

    online_cams = sum(
        1 for cam in data
        if cam["status"] == "Funcional"
    )

    offline_cams = sum(
        1 for cam in data
        if cam["status"] != "Funcional"
    )
    
    stat("Total de cámaras", total_cams, "#3b82f6")
    stat("En línea", online_cams, "#22c55e")
    stat("Desconectadas", offline_cams, "#ef4444")
    # =========================
    # BOTÓN CONFIGURAR IA
    # =========================
    next_button = tk.Button(
        stats,
        text="Configurar IA →",
        bg="#2563eb" if selected_cameras else "#1e293b",
        state="normal" if selected_cameras else "disabled",
        fg="white",
        font=("Segoe UI", 11, "bold"),
        relief="flat",
        padx=20,
        pady=10,
        cursor="hand2",
        command=open_ai_selection
    )

    next_button.pack(
        side="right",
        padx=20
    )

    
# =========================
# DASHBOARD — INICIO
# =========================
def show_inicio():
    import shutil as _shutil
    clear_main()

    BG     = "#020B25"
    PANEL  = "#07142F"
    BORDER = "#132B57"
    ACCENT = "#2D7FF9"

    container = tk.Frame(main, bg=BG)
    container.pack(fill="both", expand=True)

    padded = tk.Frame(container, bg=BG)
    padded.pack(fill="both", expand=True, padx=28, pady=20)

    # ── SECCIÓN 1: BIENVENIDA ─────────────────────────────────────────────────
    welcome = tk.Frame(padded, bg=BG)
    welcome.pack(fill="x", pady=(0, 16))

    brand_col = tk.Frame(welcome, bg=BG)
    brand_col.pack(side="left", fill="y")

    tk.Label(brand_col, text="VIGILANT PRO",
             fg=ACCENT, bg=BG,
             font=("Segoe UI", 30, "bold")).pack(anchor="w")
    tk.Label(brand_col,
             text="Sistema Inteligente de Vigilancia Perimetral",
             fg="#475569", bg=BG,
             font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 10))

    user_role = next(
        (u.get("role", "") for u in users_data if u["user"] == current_user), ""
    )
    greet_name = current_user or "Usuario"
    tk.Label(brand_col, text=f"Bienvenido, {greet_name}",
             fg="#e2e8f0", bg=BG,
             font=("Segoe UI", 14)).pack(anchor="w")
    if user_role:
        tk.Label(brand_col, text=user_role,
                 fg="#64748b", bg=BG,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(2, 0))

    clock_col = tk.Frame(welcome, bg=BG)
    clock_col.pack(side="right", anchor="ne")

    time_lbl = tk.Label(clock_col, text="", fg="#e2e8f0", bg=BG,
                        font=("Segoe UI", 22, "bold"))
    time_lbl.pack(anchor="e")
    date_lbl = tk.Label(clock_col, text="", fg="#64748b", bg=BG,
                        font=("Segoe UI", 9))
    date_lbl.pack(anchor="e", pady=(2, 0))

    def _tick():
        if time_lbl.winfo_exists():
            _now = datetime.now()
            time_lbl.configure(text=_now.strftime("%H:%M:%S"))
            date_lbl.configure(
                text=_now.strftime("%A %d de %B, %Y").capitalize()
            )
            time_lbl.after(1000, _tick)
    _tick()

    tk.Frame(padded, bg=BORDER, height=1).pack(fill="x", pady=(0, 16))

    # ── SECCIÓN 2: ESTADO GENERAL ─────────────────────────────────────────────
    tk.Label(padded, text="ESTADO GENERAL",
             fg="#475569", bg=BG,
             font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(0, 8))

    status_row = tk.Frame(padded, bg=BG)
    status_row.pack(fill="x", pady=(0, 20))

    def _status_pill(label, ok):
        dot   = "●"
        color = "#22c55e" if ok else "#ef4444"
        bg_p  = "#0a1f0a" if ok else "#1a0a0a"
        bdr   = "#166534" if ok else "#7f1d1d"
        pill  = tk.Frame(status_row, bg=bg_p,
                         highlightbackground=bdr, highlightthickness=1)
        pill.pack(side="left", padx=(0, 10), pady=2)
        inner = tk.Frame(pill, bg=bg_p)
        inner.pack(padx=14, pady=7)
        tk.Label(inner, text=dot, fg=color, bg=bg_p,
                 font=("Segoe UI", 9)).pack(side="left")
        tk.Label(inner, text=f"  {label}", fg="#cbd5e1", bg=bg_p,
                 font=("Segoe UI", 9)).pack(side="left")

    db_ok = os.path.exists(REGISTRY_LOG_FILE)
    try:
        _du = _shutil.disk_usage(".")
        storage_ok = (_du.free / _du.total) > 0.05
    except Exception:
        storage_ok = True

    _status_pill("Sistema operativo",        True)
    _status_pill("Monitoreo disponible",      bool(selected_cameras))
    _status_pill("Base de datos disponible",  db_ok)
    _status_pill("Almacenamiento disponible", storage_ok)

    # ── BODY: izquierda + derecha ─────────────────────────────────────────────
    body = tk.Frame(padded, bg=BG)
    body.pack(fill="both", expand=True)
    body.columnconfigure(0, weight=3)
    body.columnconfigure(1, weight=2)
    body.rowconfigure(0, weight=1)

    # ── COLUMNA IZQUIERDA ─────────────────────────────────────────────────────
    left = tk.Frame(body, bg=BG)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
    left.rowconfigure(1, weight=1)
    left.columnconfigure(0, weight=1)

    # --- RESUMEN DE ACTIVIDAD -------------------------------------------------
    sum_card = tk.Frame(left, bg=PANEL,
                        highlightbackground=BORDER, highlightthickness=1)
    sum_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))

    tk.Label(sum_card, text="Resumen de Actividad",
             fg="#FFFFFF", bg=PANEL,
             font=("Segoe UI", 11, "bold"), anchor="w").pack(
        fill="x", padx=16, pady=(14, 4))
    tk.Frame(sum_card, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(0, 12))

    metrics_grid = tk.Frame(sum_card, bg=PANEL)
    metrics_grid.pack(fill="x", padx=16, pady=(0, 16))
    metrics_grid.columnconfigure(0, weight=1)
    metrics_grid.columnconfigure(1, weight=1)

    total_events    = len(_events_list)
    critical_events = sum(1 for e in _events_list if e.get("severity") == "CRÍTICO")

    _rec_path = "recordings"
    try:
        if os.path.exists(RECORDINGS_DIR_FILE):
            with open(RECORDINGS_DIR_FILE, "r", encoding="utf-8") as _rf:
                _rec_path = json.load(_rf).get("path", "recordings")
    except Exception:
        pass
    _vid_exts = {".avi", ".mp4", ".mkv"}
    try:
        rec_count = sum(
            1 for f in os.listdir(_rec_path)
            if os.path.splitext(f)[1].lower() in _vid_exts
        ) if os.path.isdir(_rec_path) else 0
    except Exception:
        rec_count = 0

    active_users_count = len(users_data)

    def _metric_cell(row, col, icon, value, label, accent):
        cell = tk.Frame(metrics_grid, bg="#0d1f3a",
                        highlightbackground="#1e3a5f", highlightthickness=1)
        cell.grid(row=row, column=col, sticky="ew", padx=4, pady=4)
        inner = tk.Frame(cell, bg="#0d1f3a")
        inner.pack(padx=14, pady=10, fill="x")
        top = tk.Frame(inner, bg="#0d1f3a")
        top.pack(fill="x")
        tk.Label(top, text=icon, bg="#0d1f3a",
                 font=("Segoe UI", 15)).pack(side="left")
        tk.Label(top, text=str(value), fg=accent, bg="#0d1f3a",
                 font=("Segoe UI", 22, "bold")).pack(side="right")
        tk.Label(inner, text=label, fg="#64748b", bg="#0d1f3a",
                 font=("Segoe UI", 8), anchor="w").pack(fill="x", pady=(3, 0))

    _metric_cell(0, 0, "📋", total_events,        "Eventos registrados",    "#2D7FF9")
    _metric_cell(0, 1, "🚨", critical_events,      "Incidentes críticos",    "#ef4444")
    _metric_cell(1, 0, "🎬", rec_count,            "Grabaciones almacenadas","#22c55e")
    _metric_cell(1, 1, "👤", active_users_count,   "Usuarios activos",       "#a855f7")

    # --- ACTIVIDAD RECIENTE ---------------------------------------------------
    act_card = tk.Frame(left, bg=PANEL,
                        highlightbackground=BORDER, highlightthickness=1)
    act_card.grid(row=1, column=0, sticky="nsew")

    tk.Label(act_card, text="Actividad Reciente",
             fg="#FFFFFF", bg=PANEL,
             font=("Segoe UI", 11, "bold"), anchor="w").pack(
        fill="x", padx=16, pady=(14, 4))
    tk.Frame(act_card, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(0, 4))

    act_outer = tk.Frame(act_card, bg=PANEL)
    act_outer.pack(fill="both", expand=True)
    act_cvs = tk.Canvas(act_outer, bg=PANEL, highlightthickness=0)
    act_vsb = tk.Scrollbar(act_outer, orient="vertical", command=act_cvs.yview)
    act_inner = tk.Frame(act_cvs, bg=PANEL)
    act_inner.bind("<Configure>",
                   lambda e: act_cvs.configure(scrollregion=act_cvs.bbox("all")))
    _act_win = act_cvs.create_window((0, 0), window=act_inner, anchor="nw")
    act_cvs.bind("<Configure>",
                 lambda e: act_cvs.itemconfig(_act_win, width=e.width))
    act_cvs.configure(yscrollcommand=act_vsb.set)
    act_cvs.bind("<MouseWheel>",
                 lambda e: act_cvs.yview_scroll(int(-1*(e.delta/120)), "units"))
    act_cvs.pack(side="left", fill="both", expand=True)
    act_vsb.pack(side="right", fill="y")

    _SKIP_TYPES = {"FRAME_DROP", "RECONEXION_INTENTO", "CAMARA_FRAME", "HEARTBEAT"}
    important_events = [
        e for e in _events_list
        if e.get("type", "") not in _SKIP_TYPES
    ][:12]

    _dot_map = {"CRÍTICO": "🔴", "ALERTA": "🟠", "WARNING": "🟡", "INFO": "🔵"}
    if not important_events:
        tk.Label(act_inner, text="Sin actividad reciente", fg="#64748b",
                 bg=PANEL, font=("Segoe UI", 10)).pack(pady=24)
    else:
        for ev in important_events:
            sev     = ev.get("severity", "INFO")
            dot     = _dot_map.get(sev, "🔵")
            col_fg  = SEVERITY_COLORS.get(sev, "#9DB2D4")
            ev_row  = tk.Frame(act_inner, bg=PANEL, cursor="hand2")
            ev_row.pack(fill="x", padx=6, pady=1)
            tk.Label(ev_row, text=ev.get("ts_display", "??:??"),
                     fg="#475569", bg=PANEL,
                     font=("Segoe UI", 8), width=6).pack(
                side="left", padx=(8, 2), pady=6)
            tk.Label(ev_row, text=dot, bg=PANEL,
                     font=("Segoe UI", 9)).pack(side="left", padx=2)
            _desc = ev.get("type", "")
            if ev.get("cam") and str(ev.get("cam")) != "?":
                _desc = f"Cam {ev['cam']}  —  {_desc}"
            tk.Label(ev_row, text=_desc, fg=col_fg, bg=PANEL,
                     font=("Segoe UI", 9), anchor="w").pack(
                side="left", padx=6, fill="x", expand=True)
            tk.Frame(act_inner, bg=BORDER, height=1).pack(fill="x", padx=10)

            def _open_act(e, _ev=ev):
                _show_evidence_popup(_ev)
            ev_row.bind("<Button-1>", _open_act)
            for _ch in ev_row.winfo_children():
                _ch.bind("<Button-1>", _open_act)

    # ── COLUMNA DERECHA ───────────────────────────────────────────────────────
    right = tk.Frame(body, bg=BG)
    right.grid(row=0, column=1, sticky="nsew")
    right.rowconfigure(0, weight=0)
    right.rowconfigure(1, weight=1)
    right.rowconfigure(2, weight=0)
    right.columnconfigure(0, weight=1)

    # --- ACCESOS RÁPIDOS ------------------------------------------------------
    quick_card = tk.Frame(right, bg=PANEL,
                          highlightbackground=BORDER, highlightthickness=1)
    quick_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))

    tk.Label(quick_card, text="Accesos Rápidos",
             fg="#FFFFFF", bg=PANEL,
             font=("Segoe UI", 11, "bold"), anchor="w").pack(
        fill="x", padx=16, pady=(14, 8))
    tk.Frame(quick_card, bg=BORDER, height=1).pack(fill="x", padx=16, pady=(0, 10))

    qbtns = tk.Frame(quick_card, bg=PANEL)
    qbtns.pack(fill="x", padx=14, pady=(0, 14))
    qbtns.columnconfigure(0, weight=1)
    qbtns.columnconfigure(1, weight=1)

    _QS = [
        (0, 0, "Vista en Vivo",  "#1d4ed8", show_live_view),
        (0, 1, "Eventos",        "#c2410c", show_events),
        (1, 0, "Reportes",       "#6d28d9", show_reportes),
        (1, 1, "Configuración",  "#0f766e", show_settings),
    ]
    for _r, _c, _txt, _col, _cmd in _QS:
        tk.Button(qbtns, text=_txt, bg=_col, fg="white",
                  activebackground=_col, activeforeground="white",
                  relief="flat", font=("Segoe UI", 10, "bold"),
                  pady=11, cursor="hand2",
                  command=_cmd).grid(row=_r, column=_c,
                                     sticky="ew", padx=4, pady=4)

    # --- ESPACIADOR CENTRAL ---------------------------------------------------
    tk.Frame(right, bg=BG).grid(row=1, column=0, sticky="nsew")

    # --- FRASE INSTITUCIONAL --------------------------------------------------
    phrase_card = tk.Frame(right, bg=PANEL,
                           highlightbackground="#1e3a5f", highlightthickness=1)
    phrase_card.grid(row=2, column=0, sticky="ew")

    phrase_inner = tk.Frame(phrase_card, bg=PANEL)
    phrase_inner.pack(fill="x", padx=20, pady=20)

    tk.Label(phrase_inner, text="❝",
             fg="#1e3a5f", bg=PANEL,
             font=("Segoe UI", 28, "bold")).pack(anchor="w")
    tk.Label(phrase_inner,
             text="Transformando video en\ninformación accionable.",
             fg="#94a3b8", bg=PANEL,
             font=("Segoe UI", 12, "italic"),
             justify="left").pack(anchor="w", pady=(2, 6))
    tk.Label(phrase_inner, text="— Vigilant Pro",
             fg="#475569", bg=PANEL,
             font=("Segoe UI", 8)).pack(anchor="e")

    # ── Auto-refresh cada 60 s ────────────────────────────────────────────────
    container.after(
        60000,
        lambda: show_inicio() if container.winfo_exists() else None
    )


def _show_evidence_popup(ev):
    """Abre ventana de evidencia para un evento."""
    win = tk.Toplevel()
    win.title("Evidencia")
    win.configure(bg="#020B25")
    win.geometry("700x500")
    win.resizable(True, True)

    tk.Label(win, text="Evidencia del Evento", fg="#FFFFFF", bg="#020B25",
             font=("Segoe UI", 13, "bold")).pack(pady=(16, 4))

    meta = tk.Frame(win, bg="#07142F")
    meta.pack(fill="x", padx=20, pady=(0, 10))
    sev   = ev.get("severity", "INFO")
    col   = SEVERITY_COLORS.get(sev, "#9DB2D4")
    items = [
        ("Tipo",     ev.get("type", "—"),          col),
        ("Cámara",   f"Cam {ev.get('cam','?')}",  "#2D7FF9"),
        ("Hora",     ev.get("ts", "—"),             "#9DB2D4"),
        ("Prioridad",sev,                           col),
        ("Detalles", ev.get("details", "—"),        "#9DB2D4"),
    ]
    for label, val, fg in items:
        row = tk.Frame(meta, bg="#07142F")
        row.pack(fill="x", padx=14, pady=2)
        tk.Label(row, text=f"{label}:", fg="#9DB2D4", bg="#07142F",
                 width=10, anchor="w", font=("Segoe UI", 9)).pack(side="left")
        tk.Label(row, text=val, fg=fg, bg="#07142F",
                 font=("Segoe UI", 9, "bold"), anchor="w").pack(side="left")

    snap = ev.get("snapshot")
    if snap and os.path.exists(snap):
        try:
            img_cv = cv2.imread(snap)
            if img_cv is not None:
                img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_pil.thumbnail((640, 360))
                img_tk  = ImageTk.PhotoImage(img_pil)
                lbl_img = tk.Label(win, image=img_tk, bg="#020B25")
                lbl_img.imgtk = img_tk
                lbl_img.pack(pady=8)
        except Exception:
            tk.Label(win, text="Error al cargar la imagen.",
                     fg="#FF3D57", bg="#020B25").pack(pady=8)
    else:
        tk.Label(win, text="Sin fotografía disponible para este evento.",
                 fg="#9DB2D4", bg="#020B25", font=("Segoe UI", 10)).pack(pady=30)

    tk.Button(win, text="Cerrar", bg="#2D7FF9", fg="white", relief="flat",
              padx=20, pady=6, font=("Segoe UI", 9, "bold"),
              cursor="hand2", command=win.destroy).pack(pady=(4, 16))


# =========================
# EVENTOS — TIMELINE OPERATIVO
# =========================
def show_events():
    clear_main()

    BG = "#020B25"
    PANEL = "#07142F"
    BORDER = "#132B57"

    container = tk.Frame(main, bg=BG)
    container.pack(fill="both", expand=True, padx=20, pady=16)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = tk.Frame(container, bg=BG)
    hdr.pack(fill="x", pady=(0, 10))

    tk.Label(hdr, text="Centro de Incidentes", fg="#FFFFFF", bg=BG,
             font=("Segoe UI", 20, "bold")).pack(side="left", anchor="w")
    tk.Label(hdr, text="  Eventos de seguridad y analítica",
             fg="#9DB2D4", bg=BG, font=("Segoe UI", 10)).pack(side="left", pady=(6, 0))

    def _export_pdf():
        incidents = [e for e in _events_list
                     if any(kw in e.get("type", "") for kw in INCIDENT_KEYWORDS)]
        if not incidents:
            show_notification("EXPORTAR", "Sin incidentes para exportar.", "#FF6D00")
            return
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname  = f"incidentes_{ts_str}.txt"
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write("=" * 70 + "\n")
                f.write("  VIGILANT PRO — REPORTE DE INCIDENTES\n")
                f.write(f"  Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        f"   Usuario: {current_user or '—'}\n")
                f.write("=" * 70 + "\n\n")
                for ev in incidents:
                    f.write(f"[{ev.get('ts','')}] "
                            f"Cam {ev.get('cam','?')}  "
                            f"{ev.get('severity','INFO'):<8}  "
                            f"{ev.get('type','')}  — {ev.get('details','')}\n")
            show_notification("EXPORTADO", f"Guardado: {fname}", "#00C853")
        except Exception as exc:
            show_notification("ERROR", str(exc), "#FF3D57")

    tk.Button(hdr, text="Exportar PDF", bg="#0f766e", fg="white", relief="flat",
              font=("Segoe UI", 9, "bold"), padx=12, pady=4,
              cursor="hand2", command=_export_pdf).pack(side="right", padx=4)
    tk.Button(hdr, text="↻ Actualizar", bg="#1e3a5f", fg="white", relief="flat",
              font=("Segoe UI", 9, "bold"), padx=10, pady=4,
              cursor="hand2", command=lambda: _render_table()).pack(side="right", padx=4)

    # ── Filter row ────────────────────────────────────────────────────────────
    frow = tk.Frame(container, bg=BG)
    frow.pack(fill="x", pady=(0, 8))

    filter_sev = tk.StringVar(value="TODOS")
    tk.Label(frow, text="Prioridad:", fg="#9DB2D4", bg=BG,
             font=("Segoe UI", 9)).pack(side="left", padx=(0, 6))
    for sev in ["TODOS", "CRÍTICO", "ALERTA", "WARNING", "INFO"]:
        col = SEVERITY_COLORS.get(sev, "#9DB2D4")
        tk.Radiobutton(frow, text=sev, variable=filter_sev, value=sev,
                       fg=col, bg=BG, selectcolor=PANEL,
                       activebackground=BG, font=("Segoe UI", 8, "bold"),
                       command=lambda: _render_table()).pack(side="left", padx=5)

    # ── Column headers ────────────────────────────────────────────────────────
    COL_WIDTHS = [12, 8, 22, 9, 30, 9]
    COL_LABELS = ["Hora", "Cámara", "Tipo", "Prioridad", "Descripción", "Estado"]
    COL_FG     = ["#9DB2D4"] * 6

    hdr_row = tk.Frame(container, bg=PANEL,
                       highlightbackground=BORDER, highlightthickness=1)
    hdr_row.pack(fill="x", pady=(0, 2))
    for txt, w in zip(COL_LABELS, COL_WIDTHS):
        tk.Label(hdr_row, text=txt, fg="#9DB2D4", bg=PANEL,
                 font=("Segoe UI", 9, "bold"), width=w,
                 anchor="w").pack(side="left", padx=(10, 0), pady=6)

    # ── Scrollable table ──────────────────────────────────────────────────────
    scroll_outer = tk.Frame(container, bg=PANEL,
                            highlightbackground=BORDER, highlightthickness=1)
    scroll_outer.pack(fill="both", expand=True)

    canvas = tk.Canvas(scroll_outer, bg=PANEL, highlightthickness=0)
    vsb    = tk.Scrollbar(scroll_outer, orient="vertical", command=canvas.yview)
    inner  = tk.Frame(canvas, bg=PANEL)

    inner.bind("<Configure>",
               lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
    win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
    canvas.configure(yscrollcommand=vsb.set)
    canvas.bind("<MouseWheel>",
                lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    def _render_table():
        for w in inner.winfo_children():
            w.destroy()

        sev_f = filter_sev.get()
        incidents = [e for e in _events_list
                     if any(kw in e.get("type", "") for kw in INCIDENT_KEYWORDS)]
        if sev_f != "TODOS":
            incidents = [e for e in incidents if e.get("severity") == sev_f]

        if not incidents:
            tk.Label(inner, text="Sin incidentes para el filtro seleccionado.",
                     fg="#9DB2D4", bg=PANEL,
                     font=("Segoe UI", 11)).pack(pady=40)
            canvas.yview_moveto(0)
            return

        for i, ev in enumerate(incidents):
            sev = ev.get("severity", "INFO")
            col = SEVERITY_COLORS.get(sev, "#9DB2D4")
            row_bg = "#0A1828" if i % 2 == 0 else PANEL

            row = tk.Frame(inner, bg=row_bg, cursor="hand2")
            row.pack(fill="x", pady=0)
            tk.Frame(inner, bg=BORDER, height=1).pack(fill="x")

            ev_key = f"{ev.get('ts','')}|{ev.get('cam','')}|{ev.get('type','')}"
            estado = "Revisado" if ev_key in _events_acked else "Nuevo"
            estado_col = "#9DB2D4" if estado == "Revisado" else "#00C853"

            det = ev.get("details", "") or ""
            if len(det) > 35:
                det = det[:35] + "…"

            vals = [
                (ev.get("ts_display", "??:??"),  "#FFFFFF"),
                (f"Cam {ev.get('cam','?')}",      "#2D7FF9"),
                (ev.get("type", ""),              col),
                (sev,                             col),
                (det,                             "#9DB2D4"),
                (estado,                          estado_col),
            ]
            for (txt, fg), w in zip(vals, COL_WIDTHS):
                tk.Label(row, text=txt, fg=fg, bg=row_bg,
                         font=("Segoe UI", 9), width=w,
                         anchor="w").pack(side="left", padx=(10, 0), pady=6)

            # Snapshot indicator
            if ev.get("snapshot"):
                tk.Label(row, text="📷", bg=row_bg,
                         font=("Segoe UI", 9)).pack(side="right", padx=8)

            def _on_dblclick(e, ev=ev, ev_key=ev_key):
                _events_acked.add(ev_key)
                _show_evidence_popup(ev)
                _render_table()

            row.bind("<Double-Button-1>", _on_dblclick)
            for child in row.winfo_children():
                child.bind("<Double-Button-1>", _on_dblclick)

        canvas.yview_moveto(0)

    _render_table()

    def _auto_refresh():
        if container.winfo_exists():
            _render_table()
            container.after(4000, _auto_refresh)
    container.after(4000, _auto_refresh)


# =========================
# REGISTRO TÉCNICO — AUDITORÍA COMPLETA
# =========================
def show_registro():
    clear_main()

    import csv as _csv

    BG = "#020B25"
    PANEL = "#07142F"
    BORDER = "#132B57"

    container = tk.Frame(main, bg=BG)
    container.pack(fill="both", expand=True, padx=20, pady=16)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = tk.Frame(container, bg=BG)
    hdr.pack(fill="x", pady=(0, 10))

    tk.Label(hdr, text="Registro Técnico", fg="#FFFFFF", bg=BG,
             font=("Segoe UI", 20, "bold")).pack(side="left")
    tk.Label(hdr, text="  Caja negra del sistema",
             fg="#9DB2D4", bg=BG, font=("Segoe UI", 10)).pack(side="left", pady=(6, 0))

    def _export_csv():
        entries = _get_filtered()
        if not entries:
            show_notification("EXPORTAR", "Sin registros para exportar.", "#FF6D00")
            return
        fname = f"registro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(fname, "w", newline="", encoding="utf-8-sig") as f:
                w = _csv.writer(f)
                w.writerow(["Timestamp", "Cámara", "Tipo", "Severidad",
                             "Track ID", "Conf", "Usuario", "Detalles"])
                for e in entries:
                    w.writerow([
                        e.get("ts", ""), e.get("cam", ""), e.get("type", ""),
                        e.get("severity", ""), e.get("track_id", ""),
                        f"{e['conf']:.2f}" if e.get("conf") is not None else "",
                        e.get("user", ""), e.get("details", ""),
                    ])
            show_notification("EXPORTADO", f"Excel: {fname}", "#00C853")
        except Exception as exc:
            show_notification("ERROR", str(exc), "#FF3D57")

    tk.Button(hdr, text="Exportar Excel", bg="#0f766e", fg="white", relief="flat",
              font=("Segoe UI", 9, "bold"), padx=12, pady=4,
              cursor="hand2", command=_export_csv).pack(side="right", padx=4)
    tk.Button(hdr, text="↻ Recargar", bg="#1e3a5f", fg="white", relief="flat",
              font=("Segoe UI", 9, "bold"), padx=10, pady=4,
              cursor="hand2", command=lambda: _load_and_render()).pack(side="right", padx=4)

    # ── Filters ───────────────────────────────────────────────────────────────
    fbox = tk.Frame(container, bg=PANEL,
                    highlightbackground=BORDER, highlightthickness=1)
    fbox.pack(fill="x", pady=(0, 10))

    frow1 = tk.Frame(fbox, bg=PANEL)
    frow1.pack(fill="x", padx=12, pady=(8, 4))
    frow2 = tk.Frame(fbox, bg=PANEL)
    frow2.pack(fill="x", padx=12, pady=(0, 8))

    filter_sev  = tk.StringVar(value="TODOS")
    filter_cam  = tk.StringVar(value="TODAS")
    filter_text = tk.StringVar()
    filter_date_ini = tk.StringVar()
    filter_date_fin = tk.StringVar()

    # Row 1: severity + camera
    tk.Label(frow1, text="Severidad:", fg="#9DB2D4", bg=PANEL,
             font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))
    for sev in ["TODOS", "CRÍTICO", "ALERTA", "WARNING", "INFO"]:
        col = SEVERITY_COLORS.get(sev, "#9DB2D4")
        tk.Radiobutton(frow1, text=sev, variable=filter_sev, value=sev,
                       fg=col, bg=PANEL, selectcolor="#0A1828",
                       activebackground=PANEL, font=("Segoe UI", 8, "bold"),
                       command=lambda: _load_and_render()).pack(side="left", padx=4)

    tk.Label(frow1, text="  Cámara:", fg="#9DB2D4", bg=PANEL,
             font=("Segoe UI", 9)).pack(side="left", padx=(10, 4))
    cam_vals = ["TODAS"] + [str(c) for c in selected_cameras]
    cam_menu = tk.OptionMenu(frow1, filter_cam, *cam_vals,
                             command=lambda _: _load_and_render())
    cam_menu.config(bg="#1e293b", fg="white", relief="flat",
                    font=("Segoe UI", 8), highlightthickness=0)
    cam_menu["menu"].config(bg="#1e293b", fg="white")
    cam_menu.pack(side="left")

    # Row 2: date range + text search
    def _lbl(parent, txt):
        tk.Label(parent, text=txt, fg="#9DB2D4", bg=PANEL,
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))

    _lbl(frow2, "Fecha inicio (YYYY-MM-DD):")
    e_ini = tk.Entry(frow2, textvariable=filter_date_ini, bg="#0A1828", fg="white",
                     insertbackground="white", relief="flat", width=12,
                     font=("Segoe UI", 9))
    e_ini.pack(side="left", padx=(0, 10))

    _lbl(frow2, "Fecha fin:")
    e_fin = tk.Entry(frow2, textvariable=filter_date_fin, bg="#0A1828", fg="white",
                     insertbackground="white", relief="flat", width=12,
                     font=("Segoe UI", 9))
    e_fin.pack(side="left", padx=(0, 10))

    _lbl(frow2, "Buscar:")
    e_txt = tk.Entry(frow2, textvariable=filter_text, bg="#0A1828", fg="white",
                     insertbackground="white", relief="flat", width=20,
                     font=("Segoe UI", 9))
    e_txt.pack(side="left")
    tk.Button(frow2, text="Buscar", bg="#2D7FF9", fg="white", relief="flat",
              font=("Segoe UI", 8, "bold"), padx=8, pady=2,
              cursor="hand2", command=lambda: _load_and_render()).pack(side="left", padx=(6, 0))

    # ── Column headers ────────────────────────────────────────────────────────
    HDR_COLS  = ["Timestamp",  "Cam", "Tipo",  "Severidad", "TrackID", "Conf", "Usuario", "Detalles"]
    HDR_W     = [17,            6,     26,       9,           7,         5,      9,          1]

    hdr_row = tk.Frame(container, bg=PANEL,
                       highlightbackground=BORDER, highlightthickness=1)
    hdr_row.pack(fill="x", pady=(0, 2))
    for h, w_ in zip(HDR_COLS, HDR_W):
        tk.Label(hdr_row, text=h, fg="#9DB2D4", bg=PANEL,
                 font=("Segoe UI", 9, "bold"),
                 width=w_ if w_ > 1 else None,
                 anchor="w").pack(side="left", padx=(8, 0), pady=6)

    # ── Scrollable table ──────────────────────────────────────────────────────
    scroll_outer = tk.Frame(container, bg=PANEL,
                            highlightbackground=BORDER, highlightthickness=1)
    scroll_outer.pack(fill="both", expand=True)

    canvas = tk.Canvas(scroll_outer, bg=PANEL, highlightthickness=0)
    vsb    = tk.Scrollbar(scroll_outer, orient="vertical", command=canvas.yview)
    inner  = tk.Frame(canvas, bg=PANEL)

    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
    canvas.configure(yscrollcommand=vsb.set)
    canvas.bind("<MouseWheel>",
                lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    def _get_filtered():
        entries  = load_registry_log(limit=2000)
        sev_f    = filter_sev.get()
        cam_f    = filter_cam.get()
        txt_f    = filter_text.get().strip().lower()
        d_ini    = filter_date_ini.get().strip()
        d_fin    = filter_date_fin.get().strip()
        result   = []
        for e in entries:
            if sev_f != "TODOS" and e.get("severity") != sev_f:
                continue
            if cam_f != "TODAS" and e.get("cam") != cam_f:
                continue
            ts = e.get("ts", "")
            if d_ini and ts < d_ini:
                continue
            if d_fin and ts > d_fin + " 99":
                continue
            if txt_f:
                blob = (e.get("type", "") + e.get("details", "") + e.get("user", "")).lower()
                if txt_f not in blob:
                    continue
            result.append(e)
        return result

    _REG_MAX = 300   # filas máximas renderizadas; más haría la UI impráctica

    def _load_and_render():
        for w in inner.winfo_children():
            w.destroy()

        shown = _get_filtered()

        if not shown:
            tk.Label(inner, text="Sin registros para el filtro seleccionado.",
                     fg="#9DB2D4", bg=PANEL,
                     font=("Segoe UI", 10)).pack(pady=30, padx=20, anchor="w")
            return

        total = len(shown)
        shown = shown[:_REG_MAX]

        if total > _REG_MAX:
            tk.Label(inner,
                     text=f"Mostrando {_REG_MAX} registros de {total}. Aplica filtros para acotar.",
                     fg="#f59e0b", bg=PANEL,
                     font=("Segoe UI", 8)).pack(pady=(6, 2), padx=20, anchor="w")

        FG_COLS = ["white", "#2D7FF9", None, None, "#8E5BFF", "#FFB300", "#cbd5e1", "#9DB2D4"]

        _batch_idx = [0]

        def _render_batch():
            start = _batch_idx[0]
            end   = min(start + 50, len(shown))
            for i in range(start, end):
                e   = shown[i]
                sev = e.get("severity", "INFO")
                col = SEVERITY_COLORS.get(sev, "#9DB2D4")
                bg  = "#0A1828" if i % 2 == 0 else PANEL
                row = tk.Frame(inner, bg=bg)
                row.pack(fill="x")
                tk.Frame(inner, bg=BORDER, height=1).pack(fill="x")
                vals = [
                    e.get("ts", ""), e.get("cam", ""), e.get("type", ""), sev,
                    str(e.get("track_id", "")) if e.get("track_id") is not None else "",
                    f"{e['conf']:.2f}" if e.get("conf") is not None else "",
                    e.get("user", ""), e.get("details", ""),
                ]
                for idx, (val, w_) in enumerate(zip(vals, HDR_W)):
                    fg = FG_COLS[idx] if FG_COLS[idx] else col
                    tk.Label(row, text=val, fg=fg, bg=bg,
                             font=("Segoe UI", 8),
                             width=w_ if w_ > 1 else None,
                             anchor="w").pack(side="left", padx=(8, 0), pady=4)
            _batch_idx[0] = end
            if end < len(shown) and inner.winfo_exists():
                inner.after(0, _render_batch)

        _render_batch()

    _load_and_render()


# =========================
# REPORTES
# =========================
def show_reportes():
    clear_main()

    from collections import defaultdict as _dd, Counter as _Ctr

    BG    = "#020B25"
    PANEL = "#07142F"
    BORDER= "#132B57"

    INCIDENT_DISPLAY = [
        ("Intrusión",         "Intrusiones"),
        ("Objeto Abandonado", "Objetos abandonados"),
        ("Objeto Movido",     "Objetos movidos"),
        ("Operador Ausente",  "Operadores ausentes"),
        ("Persona Corriendo", "Personas corriendo"),
        ("Persona Inmóvil",   "Personas inmóviles"),
        ("Arma",              "Armas detectadas"),
        ("Zona Roja",         "Violaciones zona roja"),
        ("Zona Amarilla",     "Violaciones zona amarilla"),
        ("Permanencia",       "Permanencia excedida"),
    ]

    container = tk.Frame(main, bg=BG)
    container.pack(fill="both", expand=True, padx=20, pady=16)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = tk.Frame(container, bg=BG)
    hdr.pack(fill="x", pady=(0, 10))

    tk.Label(hdr, text="Reportes", fg="#FFFFFF", bg=BG,
             font=("Segoe UI", 20, "bold")).pack(side="left", anchor="w")
    tk.Label(hdr, text="  Resumen ejecutivo del sistema de vigilancia",
             fg="#9DB2D4", bg=BG, font=("Segoe UI", 10)).pack(side="left", pady=(6, 0))

    # ── Summary bar (compact, max ~90px) ─────────────────────────────────────
    summary_bar = tk.Frame(container, bg=PANEL,
                           highlightbackground=BORDER, highlightthickness=1,
                           height=88)
    summary_bar.pack(fill="x", pady=(0, 10))
    summary_bar.pack_propagate(False)

    summary_inner = tk.Frame(summary_bar, bg=PANEL)
    summary_inner.place(relx=0.5, rely=0.5, anchor="center")

    _summary_labels = {}

    def _stat_widget(parent, key, label, col):
        cell = tk.Frame(parent, bg=PANEL)
        cell.pack(side="left", padx=28)
        val_lbl = tk.Label(cell, text="—", fg=col, bg=PANEL,
                           font=("Segoe UI", 22, "bold"))
        val_lbl.pack()
        tk.Label(cell, text=label, fg="#9DB2D4", bg=PANEL,
                 font=("Segoe UI", 8)).pack()
        _summary_labels[key] = val_lbl
        # separator
        tk.Frame(parent, bg=BORDER, width=1).pack(side="left", fill="y",
                                                   padx=(28, 0), pady=12)

    _stat_widget(summary_inner, "periodo",    "Periodo",          "#9DB2D4")
    _stat_widget(summary_inner, "total",      "Total Eventos",    "#2D7FF9")
    _stat_widget(summary_inner, "incidentes", "Incidentes IA",    "#FF6D00")
    _stat_widget(summary_inner, "criticos",   "Alertas Críticas", "#FF3D57")
    _stat_widget(summary_inner, "alertas",    "Alertas Generales","#FFB300")

    # ── Buttons (placed right of header after summary) ────────────────────────
    def _export_pdf():
        content = _report_lines[0]
        if not content:
            show_notification("REPORTE", "Genera el reporte primero.", "#FF6D00")
            return
        fname = f"reporte_vigilant_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(fname, "w", encoding="utf-8") as f:
                f.write(content)
            show_notification("EXPORTADO", f"Archivo: {fname}", "#00C853")
        except Exception as exc:
            show_notification("ERROR", str(exc), "#FF3D57")

    tk.Button(hdr, text="Exportar PDF", bg="#0f766e", fg="white", relief="flat",
              font=("Segoe UI", 9, "bold"), padx=12, pady=4,
              cursor="hand2", command=_export_pdf).pack(side="right", padx=4)
    tk.Button(hdr, text="↻ Actualizar", bg="#1e3a5f", fg="white", relief="flat",
              font=("Segoe UI", 9, "bold"), padx=10, pady=4,
              cursor="hand2", command=lambda: _build_report()).pack(side="right", padx=4)

    # ── Report text area ──────────────────────────────────────────────────────
    txt_frame = tk.Frame(container, bg=PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
    txt_frame.pack(fill="both", expand=True)

    txt = tk.Text(
        txt_frame, bg="#020B25", fg="#9DB2D4",
        font=("Consolas", 10), relief="flat",
        wrap="word", state="disabled",
        selectbackground="#1e3a5f",
        padx=24, pady=16,
        cursor="arrow",
    )
    vsb = tk.Scrollbar(txt_frame, orient="vertical", command=txt.yview)
    txt.configure(yscrollcommand=vsb.set)
    txt.bind("<MouseWheel>", lambda e: txt.yview_scroll(int(-1*(e.delta/120)), "units"))
    vsb.pack(side="right", fill="y")
    txt.pack(side="left", fill="both", expand=True)

    # Tag palette
    txt.tag_configure("banner",   foreground="#2D7FF9",  font=("Consolas", 10, "bold"))
    txt.tag_configure("title",    foreground="#FFFFFF",  font=("Consolas", 12, "bold"))
    txt.tag_configure("meta",     foreground="#9DB2D4",  font=("Consolas", 10))
    txt.tag_configure("sep",      foreground="#132B57",  font=("Consolas", 10))
    txt.tag_configure("section",  foreground="#2D7FF9",  font=("Consolas", 10, "bold"))
    txt.tag_configure("cam",      foreground="#FFFFFF",  font=("Consolas", 10, "bold"))
    txt.tag_configure("label",    foreground="#9DB2D4",  font=("Consolas", 10))
    txt.tag_configure("val_ok",   foreground="#9DB2D4",  font=("Consolas", 10))
    txt.tag_configure("val_warn", foreground="#FFB300",  font=("Consolas", 10, "bold"))
    txt.tag_configure("val_crit", foreground="#FF3D57",  font=("Consolas", 10, "bold"))
    txt.tag_configure("val_alrt", foreground="#FF6D00",  font=("Consolas", 10, "bold"))
    txt.tag_configure("val_info", foreground="#2D7FF9",  font=("Consolas", 10, "bold"))
    txt.tag_configure("total",    foreground="#00C853",  font=("Consolas", 10, "bold"))

    _report_lines = [""]  # mutable holder for export

    SEP72  = "━" * 72
    LINE72 = "─" * 72

    def _w(line="", tag="meta"):
        txt.configure(state="normal")
        txt.insert("end", line + "\n", tag)
        txt.configure(state="disabled")

    def _build_report():
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        txt.configure(state="disabled")

        entries   = load_registry_log(limit=2000)
        now       = datetime.now()
        incidents = [e for e in entries
                     if any(kw in (e.get("type") or "") for kw in INCIDENT_KEYWORDS)]
        criticos  = sum(1 for e in incidents if e.get("severity") == "CRÍTICO")
        alertas   = sum(1 for e in incidents if e.get("severity") == "ALERTA")
        warnings  = sum(1 for e in incidents if e.get("severity") == "WARNING")

        ts_list = [e.get("ts", "") for e in entries if e.get("ts")]
        periodo = (f"{min(ts_list)[:10]}  →  {max(ts_list)[:10]}"
                   if ts_list else "Sin datos")

        # ── Update summary bar ────────────────────────────────────────────────
        _summary_labels["periodo"].config(
            text=periodo if ts_list else "—",
            font=("Segoe UI", 9, "bold"))
        _summary_labels["total"].config(text=str(len(entries)))
        _summary_labels["incidentes"].config(text=str(len(incidents)))
        _summary_labels["criticos"].config(text=str(criticos))
        _summary_labels["alertas"].config(text=str(alertas))

        lines = []
        def _wl(line="", tag="meta"):
            _w(line, tag)
            lines.append(line)

        # ── Document header ───────────────────────────────────────────────────
        _wl(SEP72, "sep")
        _wl("", "meta")
        _wl("  VIGILANT PRO  —  REPORTE EJECUTIVO DE SEGURIDAD", "title")
        _wl("", "meta")
        _wl(f"  Generado    : {now.strftime('%Y-%m-%d  %H:%M:%S')}", "meta")
        _wl(f"  Usuario     : {current_user or '—'}", "meta")
        _wl(f"  Periodo     : {periodo}", "meta")
        _wl("", "meta")
        _wl(SEP72, "sep")

        if not entries:
            _wl("", "meta")
            _wl("  Sin registros disponibles.", "meta")
            _report_lines[0] = "\n".join(lines)
            return

        # ── Resumen general ───────────────────────────────────────────────────
        _wl("", "meta")
        _wl("  RESUMEN GENERAL", "section")
        _wl("  " + LINE72, "sep")
        _wl("", "meta")
        _wl(f"  {'Eventos Totales':<38} {len(entries):>6}", "meta")

        _wl(f"  {'Total Incidentes IA':<38} {len(incidents):>6}",
            "val_alrt" if incidents else "meta")
        _wl(f"  {'Alertas Críticas (CRÍTICO)':<38} {criticos:>6}",
            "val_crit" if criticos else "meta")
        _wl(f"  {'Alertas Generales (ALERTA)':<38} {alertas:>6}",
            "val_alrt" if alertas else "meta")
        _wl(f"  {'Advertencias (WARNING)':<38} {warnings:>6}",
            "val_warn" if warnings else "meta")
        _wl("", "meta")

        # ── Análisis por cámara ───────────────────────────────────────────────
        _wl(SEP72, "sep")
        _wl("", "meta")
        _wl("  ANÁLISIS POR CÁMARA", "section")
        _wl("", "meta")

        cam_inc = _dd(list)
        for e in incidents:
            cam_inc[e.get("cam", "?")].append(e)

        # Include cameras from selected_cameras that have no incidents (zero report)
        all_cam_ids = sorted(
            set(cam_inc.keys()) |
            {str(c) for c in selected_cameras}
        )

        for cam_s in all_cam_ids:
            inc_evs  = cam_inc.get(cam_s, [])
            cam_total= len(inc_evs)

            _wl(f"  {LINE72}", "sep")
            _wl(f"  ▌ CÁMARA {cam_s}", "cam")
            _wl(f"  {LINE72}", "sep")

            for kw, label in INCIDENT_DISPLAY:
                cnt = sum(1 for e in inc_evs if kw in (e.get("type") or ""))
                val_tag = "val_ok" if cnt == 0 else (
                    "val_crit" if kw in ("Intrusión", "Arma", "Zona Roja") else
                    "val_alrt" if kw in ("Objeto Abandonado", "Objeto Movido",
                                         "Zona Amarilla", "Permanencia") else
                    "val_warn"
                )
                _wl(f"  {label:<38} {cnt:>6}", val_tag if cnt > 0 else "label")

            _wl("", "meta")
            _wl(f"  {'Total incidentes':<38} {cam_total:>6}",
                "total" if cam_total > 0 else "label")
            _wl("", "meta")

        # ── Footer ────────────────────────────────────────────────────────────
        _wl(SEP72, "sep")
        _wl("", "meta")
        _wl(f"  Reporte generado por VIGILANT PRO  —  {now.strftime('%Y-%m-%d %H:%M:%S')}",
            "meta")
        _wl(SEP72, "sep")

        _report_lines[0] = "\n".join(lines)
        txt.see("1.0")

    _build_report()


# =========================
# AUDITORÍA
# =========================
def show_audit():
    clear_main()

    import csv as _csv

    BG    = "#020B25"
    PANEL = "#07142F"
    BORDER= "#132B57"

    # Only these actions are considered human audit events
    HUMAN_ACTIONS = {
        "LOGIN", "LOGOUT", "CONFIG_CHANGE", "USER_ADDED", "USER_REMOVED",
        "PERM_CHANGE", "EXPORT", "LOG_DELETE", "PASSWORD_CHANGE",
        "CAMBIO_CONFIG", "ALTA_USUARIO", "BAJA_USUARIO", "CAMBIO_PERMISOS",
        "EXPORTACION", "ELIMINACION_REGISTRO",
    }

    container = tk.Frame(main, bg=BG)
    container.pack(fill="both", expand=True, padx=20, pady=16)

    # ── Header ────────────────────────────────────────────────────────────────
    hdr = tk.Frame(container, bg=BG)
    hdr.pack(fill="x", pady=(0, 10))

    tk.Label(hdr, text="Auditoría", fg="#FFFFFF", bg=BG,
             font=("Segoe UI", 20, "bold")).pack(side="left")
    tk.Label(hdr, text="  Acciones de usuarios y operadores",
             fg="#9DB2D4", bg=BG, font=("Segoe UI", 10)).pack(side="left", pady=(6, 0))

    def _export_csv():
        shown = _get_filtered()
        if not shown:
            show_notification("EXPORTAR", "Sin registros para exportar.", "#FF6D00")
            return
        fname = f"auditoria_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(fname, "w", newline="", encoding="utf-8-sig") as f:
                w = _csv.writer(f)
                w.writerow(["Fecha", "Usuario", "Acción", "Estado", "Detalles"])
                for log in shown:
                    w.writerow([log.get("timestamp",""), log.get("user",""),
                                 log.get("action",""), log.get("status",""),
                                 log.get("details","")])
            show_notification("EXPORTADO", f"Excel: {fname}", "#00C853")
        except Exception as exc:
            show_notification("ERROR", str(exc), "#FF3D57")

    tk.Button(hdr, text="Exportar Excel", bg="#0f766e", fg="white", relief="flat",
              font=("Segoe UI", 9, "bold"), padx=12, pady=4,
              cursor="hand2", command=_export_csv).pack(side="right", padx=4)
    tk.Button(hdr, text="↻ Recargar", bg="#1e3a5f", fg="white", relief="flat",
              font=("Segoe UI", 9, "bold"), padx=10, pady=4,
              cursor="hand2", command=lambda: _render()).pack(side="right", padx=4)

    # ── Filters ───────────────────────────────────────────────────────────────
    fbox = tk.Frame(container, bg=PANEL,
                    highlightbackground=BORDER, highlightthickness=1)
    fbox.pack(fill="x", pady=(0, 10))

    frow = tk.Frame(fbox, bg=PANEL)
    frow.pack(fill="x", padx=12, pady=8)

    filter_user   = tk.StringVar()
    filter_action = tk.StringVar(value="TODAS")
    filter_status = tk.StringVar(value="TODOS")
    filter_date   = tk.StringVar()

    def _lbl(t):
        tk.Label(frow, text=t, fg="#9DB2D4", bg=PANEL,
                 font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))

    _lbl("Usuario:")
    tk.Entry(frow, textvariable=filter_user, bg="#0A1828", fg="white",
             insertbackground="white", relief="flat", width=12,
             font=("Segoe UI", 9)).pack(side="left", padx=(0, 12))

    _lbl("Acción:")
    action_choices = ["TODAS", "LOGIN", "LOGOUT", "CONFIG_CHANGE",
                      "USER_ADDED", "USER_REMOVED", "EXPORT"]
    am = tk.OptionMenu(frow, filter_action, *action_choices,
                       command=lambda _: _render())
    am.config(bg="#1e293b", fg="white", relief="flat",
              font=("Segoe UI", 8), highlightthickness=0)
    am["menu"].config(bg="#1e293b", fg="white")
    am.pack(side="left", padx=(0, 12))

    _lbl("Estado:")
    for s in ["TODOS", "OK", "FAILED"]:
        col = "#00C853" if s == "OK" else "#FF3D57" if s == "FAILED" else "#9DB2D4"
        tk.Radiobutton(frow, text=s, variable=filter_status, value=s,
                       fg=col, bg=PANEL, selectcolor="#0A1828",
                       activebackground=PANEL, font=("Segoe UI", 8, "bold"),
                       command=lambda: _render()).pack(side="left", padx=3)

    _lbl("  Fecha (YYYY-MM-DD):")
    tk.Entry(frow, textvariable=filter_date, bg="#0A1828", fg="white",
             insertbackground="white", relief="flat", width=12,
             font=("Segoe UI", 9)).pack(side="left", padx=(0, 8))
    tk.Button(frow, text="Buscar", bg="#2D7FF9", fg="white", relief="flat",
              font=("Segoe UI", 8, "bold"), padx=8, pady=2,
              cursor="hand2", command=lambda: _render()).pack(side="left")

    # ── Column headers ────────────────────────────────────────────────────────
    COL_H = ["Fecha",  "Usuario", "Acción",   "Estado", "Detalles"]
    COL_W = [18,        12,        16,          7,        1]

    hdr_row = tk.Frame(container, bg=PANEL,
                       highlightbackground=BORDER, highlightthickness=1)
    hdr_row.pack(fill="x", pady=(0, 2))
    for h, w_ in zip(COL_H, COL_W):
        tk.Label(hdr_row, text=h, fg="#9DB2D4", bg=PANEL,
                 font=("Segoe UI", 9, "bold"),
                 width=w_ if w_ > 1 else None,
                 anchor="w").pack(side="left", padx=(10, 0), pady=6)

    # ── Scrollable table ──────────────────────────────────────────────────────
    scroll_outer = tk.Frame(container, bg=PANEL,
                            highlightbackground=BORDER, highlightthickness=1)
    scroll_outer.pack(fill="both", expand=True)

    canvas = tk.Canvas(scroll_outer, bg=PANEL, highlightthickness=0)
    vsb    = tk.Scrollbar(scroll_outer, orient="vertical", command=canvas.yview)
    inner  = tk.Frame(canvas, bg=PANEL)

    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
    canvas.configure(yscrollcommand=vsb.set)
    canvas.bind("<MouseWheel>",
                lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    canvas.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")

    def _get_filtered():
        all_logs = load_audit_log()
        all_logs.reverse()
        usr_f  = filter_user.get().strip().lower()
        act_f  = filter_action.get()
        sta_f  = filter_status.get()
        dat_f  = filter_date.get().strip()
        result = []
        for log in all_logs:
            action = log.get("action", "")
            # include only if it's a human action OR no match found (keep LOGIN/LOGOUT always)
            if HUMAN_ACTIONS and action.upper() not in {a.upper() for a in HUMAN_ACTIONS}:
                # allow through if it doesn't look like a system event
                if any(kw in action for kw in ("IA", "YOLO", "RTSP", "USB", "Grabación",
                                                "Cámara", "Worker", "FFMPEG")):
                    continue
            if act_f != "TODAS" and action != act_f:
                continue
            if sta_f != "TODOS" and log.get("status") != sta_f:
                continue
            if usr_f and usr_f not in log.get("user", "").lower():
                continue
            ts = log.get("timestamp", "")
            if dat_f and not ts.startswith(dat_f):
                continue
            result.append(log)
        return result[:500]

    def _render():
        for w in inner.winfo_children():
            w.destroy()

        shown = _get_filtered()
        if not shown:
            tk.Label(inner, text="Sin registros para el filtro seleccionado.",
                     fg="#9DB2D4", bg=PANEL,
                     font=("Segoe UI", 10)).pack(pady=30, padx=20, anchor="w")
            return

        for i, log in enumerate(shown):
            status = log.get("status", "OK")
            sta_col = "#00C853" if status == "OK" else "#FF3D57"
            bg = "#0A1828" if i % 2 == 0 else PANEL

            details = log.get("details", "")
            action  = log.get("action", "")
            if action == "LOGIN":
                details = details or "Acceso concedido"
            elif action == "LOGOUT":
                details = details or "Sesión finalizada"
            elif status == "FAILED":
                details = details or "Intento fallido"

            row = tk.Frame(inner, bg=bg)
            row.pack(fill="x")
            tk.Frame(inner, bg=BORDER, height=1).pack(fill="x")

            vals = [
                (log.get("timestamp", ""), "white"),
                (log.get("user", ""),       "#2D7FF9"),
                (action,                    "#9DB2D4"),
                (status,                    sta_col),
                (details,                   "#9DB2D4"),
            ]
            for (txt, fg), w_ in zip(vals, COL_W):
                tk.Label(row, text=txt, fg=fg, bg=bg,
                         font=("Segoe UI", 9),
                         width=w_ if w_ > 1 else None,
                         anchor="w").pack(side="left", padx=(10, 0), pady=6)

    _render()
# =========================
# TURNOS
# =========================
def show_turnos():
    clear_main()

    # Only admins
    is_admin = False
    for u in users_data:
        if u["user"] == current_user and u.get("role") == "Administrador":
            is_admin = True
            break
    if current_user in ("admin", "ROOT_USB"):
        is_admin = True

    BG     = "#020B25"
    PANEL  = "#07142F"
    CARD   = "#0d1f3c"
    BORDER = "#132B57"
    ACCENT = "#2D7FF9"

    SHIFT_ICONS  = {"mañana": "☀", "tarde": "🌤", "noche": "🌙"}
    SHIFT_COLORS = {"mañana": "#f59e0b", "tarde": "#3b82f6", "noche": "#8b5cf6"}

    outer = tk.Frame(main, bg=BG)
    outer.pack(fill="both", expand=True)

    # ── Page header ──────────────────────────────────────────────────────────
    hdr = tk.Frame(outer, bg=PANEL, pady=14)
    hdr.pack(fill="x", padx=0, pady=0)
    tk.Label(hdr, text="  Perfiles Operativos por Turno",
             fg="#e5e7eb", bg=PANEL,
             font=("Segoe UI", 16, "bold")).pack(side="left", padx=20)

    # Active shift chip
    sh_now = shifts_config.get(_active_shift, {})
    chip_col = SHIFT_COLORS.get(_active_shift, ACCENT)
    chip_frm = tk.Frame(hdr, bg=chip_col, padx=10, pady=4)
    chip_frm.pack(side="right", padx=20)
    tk.Label(chip_frm, text=f"Turno Activo: {sh_now.get('icono','')} {sh_now.get('nombre','')}",
             fg="white", bg=chip_col,
             font=("Segoe UI", 11, "bold")).pack()

    # ── Scrollable body ───────────────────────────────────────────────────────
    body_wrap = tk.Frame(outer, bg=BG)
    body_wrap.pack(fill="both", expand=True, padx=20, pady=16)

    canvas_scroll = tk.Canvas(body_wrap, bg=BG, highlightthickness=0)
    vbar = tk.Scrollbar(body_wrap, orient="vertical", command=canvas_scroll.yview)
    canvas_scroll.configure(yscrollcommand=vbar.set)
    vbar.pack(side="right", fill="y")
    canvas_scroll.pack(side="left", fill="both", expand=True)

    scroll_frame = tk.Frame(canvas_scroll, bg=BG)
    _sw = canvas_scroll.create_window((0, 0), window=scroll_frame, anchor="nw")

    def _on_cfg(e):
        canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all"))
        canvas_scroll.itemconfig(_sw, width=canvas_scroll.winfo_width())
    scroll_frame.bind("<Configure>", _on_cfg)
    canvas_scroll.bind("<Configure>",
                       lambda e: canvas_scroll.itemconfig(_sw, width=e.width))
    canvas_scroll.bind("<MouseWheel>",
                       lambda e: canvas_scroll.yview_scroll(-1*(e.delta//120), "units"))

    # ── Shift schedule cards ──────────────────────────────────────────────────
    schedule_lbl = tk.Label(scroll_frame, text="Horarios de Turno",
                            fg="#93c5fd", bg=BG,
                            font=("Segoe UI", 12, "bold"))
    schedule_lbl.pack(anchor="w", pady=(0, 8))

    cards_row = tk.Frame(scroll_frame, bg=BG)
    cards_row.pack(fill="x", pady=(0, 20))

    def _rebuild_shift_cards():
        for w in cards_row.winfo_children():
            w.destroy()
        for key in ("mañana", "tarde", "noche"):
            sh = shifts_config.get(key, _SHIFTS_DEFAULT[key])
            col = SHIFT_COLORS[key]
            is_active = (key == _active_shift)
            border_col = col if is_active else BORDER

            card = tk.Frame(cards_row, bg=CARD,
                            highlightthickness=2,
                            highlightbackground=border_col)
            card.pack(side="left", fill="x", expand=True,
                      padx=(0, 12) if key != "noche" else 0)

            # Title row
            title_row = tk.Frame(card, bg=col)
            title_row.pack(fill="x")
            tk.Label(title_row,
                     text=f"  {sh.get('icono','')}  {sh.get('nombre',key.title())}",
                     fg="white", bg=col,
                     font=("Segoe UI", 13, "bold")).pack(side="left", pady=10, padx=6)
            if is_active:
                tk.Label(title_row, text="● ACTIVO",
                         fg="white", bg=col,
                         font=("Segoe UI", 8, "bold")).pack(side="right", padx=10)

            body_c = tk.Frame(card, bg=CARD)
            body_c.pack(fill="x", padx=14, pady=12)

            if not is_admin:
                # Read-only display
                tk.Label(body_c,
                         text=f"Inicio: {sh.get('inicio','--:--')}   Fin: {sh.get('fin','--:--')}",
                         fg="#e5e7eb", bg=CARD,
                         font=("Segoe UI", 11)).pack(anchor="w")
                continue

            # Editable hour inputs
            for lbl_text, field in (("Inicio:", "inicio"), ("Fin:", "fin")):
                row_f = tk.Frame(body_c, bg=CARD)
                row_f.pack(fill="x", pady=3)
                tk.Label(row_f, text=lbl_text, fg="#94a3b8", bg=CARD,
                         font=("Segoe UI", 10), width=7, anchor="w").pack(side="left")
                ent = tk.Entry(row_f, bg="#111827", fg="white",
                               insertbackground="white", relief="flat",
                               font=("Segoe UI", 11), width=7)
                ent.insert(0, sh.get(field, "00:00"))
                ent.pack(side="left", padx=6)

                def _save_time(event=None, k=key, f=field, e=ent):
                    val = e.get().strip()
                    # Validate HH:MM
                    import re as _re
                    if not _re.match(r'^\d{1,2}:\d{2}$', val):
                        show_notification("FORMATO INVÁLIDO", "Use HH:MM (ej: 06:00)", "#ef4444")
                        return
                    try:
                        hh, mm = map(int, val.split(":"))
                        if not (0 <= hh <= 23 and 0 <= mm <= 59):
                            raise ValueError
                    except ValueError:
                        show_notification("HORA INVÁLIDA", "Hora 0-23, minutos 0-59", "#ef4444")
                        return
                    shifts_config[k][f] = val
                    save_shifts_config()
                    show_notification("TURNO GUARDADO",
                                      f"{shifts_config[k].get('nombre',k)} actualizado.", "#22c55e")
                    register_event(current_user, "TURNO_MODIFICADO", "OK",
                                   f"Turno {k}: {f}={val}")

                ent.bind("<FocusOut>", _save_time)
                ent.bind("<Return>", _save_time)

        cards_row.update_idletasks()

    _rebuild_shift_cards()

    # ── Per-camera shift status ───────────────────────────────────────────────
    tk.Label(scroll_frame, text="Estado por Cámara",
             fg="#93c5fd", bg=BG,
             font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 8))

    cam_table = tk.Frame(scroll_frame, bg=PANEL,
                         highlightthickness=1,
                         highlightbackground=BORDER)
    cam_table.pack(fill="x", pady=(0, 20))

    # Header row
    hdr_row = tk.Frame(cam_table, bg="#0d1f3c")
    hdr_row.pack(fill="x")
    for txt, w in (("Cámara", 14), ("Estado", 14), ("Gestión por turno", 18),
                   ("Operadores configurados", 26), ("Turno activo", 14)):
        tk.Label(hdr_row, text=txt, fg="#93c5fd", bg="#0d1f3c",
                 font=("Segoe UI", 9, "bold"), width=w, anchor="w").pack(
                     side="left", padx=8, pady=6)

    # Build set of cameras that actually exist in the system right now
    _usb_scan_now  = scan_usb_cameras()
    _usb_ids_now   = {str(cam["id"]) for cam in _usb_scan_now}
    _rtsp_ids_now  = {str(c["id"]) for c in rtsp_cameras}
    _existing_ids  = _usb_ids_now | _rtsp_ids_now

    # Only show cameras that are in selected_cameras AND still registered
    # (silently skip orphaned ai_config entries whose camera was deleted)
    all_cam_ids = {str(c) for c in selected_cameras if str(c) in _existing_ids}
    # Also include cameras with operator configs that are still registered
    for k in ai_config:
        if ai_config[k].get("operators") and str(k) in _existing_ids:
            all_cam_ids.add(str(k))

    if not all_cam_ids:
        tk.Label(cam_table, text="  No hay cámaras configuradas.",
                 fg="#64748b", bg=PANEL,
                 font=("Segoe UI", 10)).pack(anchor="w", padx=14, pady=10)
    else:
        for idx, cid in enumerate(sorted(all_cam_ids,
                                         key=lambda x: int(x) if x.isdigit() else 9999)):
            cfg = ai_config.get(cid, {})
            ops = cfg.get("operators", [])
            sa  = cfg.get("shift_auto", False)

            # Determine connection status
            _in_usb   = cid in _usb_ids_now
            _in_rtsp  = cid in _rtsp_ids_now
            _last_frm = _cam_last_frame.get(cid, 0)
            if _in_rtsp:
                _connected = (time.time() - _last_frm) < 15
            else:
                _connected = _in_usb

            # Active-shift operator names
            active_ops = []
            for i in range(len(ops)):
                t = cfg.get(f"op_{i}_turno", "")
                n = cfg.get(f"op_{i}_nombre", f"Op.{i+1}")
                if not sa or not t or t == _active_shift:
                    active_ops.append(n)

            row_bg = BG if idx % 2 == 0 else PANEL
            r = tk.Frame(cam_table, bg=row_bg)
            r.pack(fill="x")

            cam_name = camera_names.get(str(cid), f"Cámara {cid}")
            tk.Label(r, text=cam_name, fg="#e5e7eb", bg=row_bg,
                     font=("Segoe UI", 9), width=14, anchor="w").pack(
                         side="left", padx=8, pady=5)

            if _connected:
                st_txt, st_col = "● En línea", "#22c55e"
            else:
                st_txt, st_col = "⚫ Desconectada", "#6b7280"
            tk.Label(r, text=st_txt, fg=st_col, bg=row_bg,
                     font=("Segoe UI", 9, "bold"), width=14, anchor="w").pack(
                         side="left", padx=8)

            sa_col = "#22c55e" if sa else "#6b7280"
            sa_txt = "Activada" if sa else "Desactivada"
            tk.Label(r, text=sa_txt, fg=sa_col, bg=row_bg,
                     font=("Segoe UI", 9, "bold"), width=18, anchor="w").pack(
                         side="left", padx=8)

            ops_summary = ", ".join(active_ops[:3]) if active_ops else "—"
            if len(active_ops) > 3:
                ops_summary += f" (+{len(active_ops)-3})"
            tk.Label(r, text=ops_summary, fg="#94a3b8", bg=row_bg,
                     font=("Segoe UI", 9), width=26, anchor="w").pack(
                         side="left", padx=8)

            active_sh = shifts_config.get(_active_shift, {})
            sh_txt = f"{active_sh.get('icono','')} {active_sh.get('nombre','')}"
            tk.Label(r, text=sh_txt, fg=SHIFT_COLORS.get(_active_shift, ACCENT), bg=row_bg,
                     font=("Segoe UI", 9, "bold"), width=14, anchor="w").pack(
                         side="left", padx=8)

    # ── Future rules placeholder ──────────────────────────────────────────────
    tk.Label(scroll_frame, text="Reglas Futuras",
             fg="#93c5fd", bg=BG,
             font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 8))
    rules_frame = tk.Frame(scroll_frame, bg=PANEL,
                           highlightthickness=1,
                           highlightbackground=BORDER)
    rules_frame.pack(fill="x", pady=(0, 20))
    for rule_txt in (
        "Operador fuera de turno — próximamente",
        "Operador incorrecto en puesto — próximamente",
        "Puesto sin operador asignado — próximamente",
    ):
        rf = tk.Frame(rules_frame, bg=PANEL)
        rf.pack(fill="x", padx=14, pady=4)
        tk.Label(rf, text="⚙", fg="#475569", bg=PANEL,
                 font=("Segoe UI", 11)).pack(side="left", padx=(0, 8))
        tk.Label(rf, text=rule_txt, fg="#475569", bg=PANEL,
                 font=("Segoe UI", 10, "italic")).pack(side="left")

# =========================
# REPRODUCCIÓN / BIBLIOTECA
# =========================
def show_biblioteca():
    clear_main()
    import re as _re
    import threading as _thr
    from datetime import timedelta

    BG     = "#020B25"
    PANEL  = "#07142F"
    CARD   = "#0d1f3c"
    BORDER = "#132B57"
    ACCENT = "#2D7FF9"
    CARD_W = 168
    CARD_H = 156
    THW    = 168
    THH    = 92

    # ── helpers ───────────────────────────────────────────────────────────────
    def _rb():
        try:
            with open(RECORDINGS_DIR_FILE) as _f:
                return json.load(_f).get("path", "recordings")
        except Exception:
            return "recordings"

    def _parse_item(fpath, subtype):
        fname = os.path.basename(fpath)
        ext   = os.path.splitext(fname)[1].lower()
        media = "video" if ext in (".avi", ".mp4", ".mkv") else "image"
        cam_id, dt_obj, event = "?", None, "—"

        # cam{id}_{YYYYMMDD}_{HHMMSS}[_ia].avi
        m = _re.match(r'^cam(\d+)_(\d{8})_(\d{6})(?:_ia)?\.avi$', fname)
        if m:
            cam_id, event = m.group(1), "Grabación"
            try:
                dt_obj = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S")
            except Exception:
                pass

        # snap_c{id}_manual_{YYYYMMDD}_{HHMMSS}[_ia].jpg
        if cam_id == "?":
            m = _re.match(r'^snap_c(\d+)_(manual)_(\d{8})_(\d{6})(?:_ia)?\.jpe?g$',
                          fname, _re.IGNORECASE)
            if m:
                cam_id, event = m.group(1), "Manual"
                try:
                    dt_obj = datetime.strptime(m.group(3) + m.group(4), "%Y%m%d%H%M%S")
                except Exception:
                    pass

        # snap_c{id}_{event}_{YYYY-MM-DD_HH-MM-SS}[_ia].jpg
        if cam_id == "?":
            m = _re.match(
                r'^snap_c(\d+)_(.+?)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})(?:_ia)?\.jpe?g$',
                fname, _re.IGNORECASE)
            if m:
                cam_id = m.group(1)
                event  = m.group(2).replace("_", " ").title()
                try:
                    dt_obj = datetime.strptime(m.group(3), "%Y-%m-%d_%H-%M-%S")
                except Exception:
                    pass

        try:
            st = os.stat(fpath)
            sz, mt = st.st_size, st.st_mtime
        except Exception:
            sz, mt = 0, 0

        cn = _cam_name_for(cam_id) if cam_id != "?" else fname[:22]
        return {
            "path":     fpath,
            "fname":    fname,
            "media":    media,
            "subtype":  subtype,
            "cam_id":   cam_id,
            "cam_name": cn,
            "dt_obj":   dt_obj,
            "dt_str":   dt_obj.strftime("%Y-%m-%d %H:%M:%S") if dt_obj else "",
            "event":    event,
            "size":     sz,
            "mtime":    mt,
        }

    def _scan_all():
        rb  = os.path.abspath(_rb())
        ssd = os.path.abspath(SNAPSHOTS_DIR)
        print(f"[REPRO] Escaneo inicio   | rec_base={rb} | snap_base={ssd}")
        out = []
        for base in (rb, ssd):
            d = os.path.join(base, "analitica")
            existe = os.path.isdir(d)
            print(f"[REPRO] Directorio       | path={d} | existe={existe}")
            if existe:
                for rt, _, files in os.walk(d):
                    for f in sorted(files):
                        if f.lower().endswith((".avi", ".mp4", ".jpg", ".png")):
                            fpath = os.path.abspath(os.path.join(rt, f))
                            print(f"[REPRO] Archivo encontrado| {fpath}")
                            out.append(_parse_item(fpath, "analitica"))
        out.sort(key=lambda x: x["mtime"], reverse=True)
        print(f"[REPRO] Escaneo completo | total={len(out)} archivos")
        return out

    def _pil_thumb(item):
        try:
            if item["media"] == "video":
                cap = cv2.VideoCapture(item["path"])
                ok, fr = cap.read()
                if not ok or fr is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
                    ok, fr = cap.read()
                cap.release()
                if not ok or fr is None:
                    print(f"[REPRO] Miniatura falló  | {item['fname']} (sin frames legibles)")
                    return None
                print(f"[REPRO] Miniatura generada| {item['fname']}")
                return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            img = Image.open(item["path"]).convert("RGB")
            print(f"[REPRO] Miniatura generada| {item['fname']}")
            return img
        except Exception as _e:
            print(f"[REPRO] Miniatura falló  | {item['fname']} — {_e}")
            return None

    def _get_dur(fpath):
        try:
            cap = cv2.VideoCapture(fpath)
            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            tot = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            s = int(tot / fps)
            if s < 3600:
                return f"{s//60:02d}:{s%60:02d}"
            return f"{s//3600}:{(s%3600)//60:02d}:{s%60:02d}"
        except Exception:
            return ""

    def _fsz(b):
        if b >= 1_073_741_824: return f"{b/1_073_741_824:.1f} GB"
        if b >= 1_048_576:     return f"{b/1_048_576:.1f} MB"
        if b >= 1024:          return f"{b/1024:.0f} KB"
        return f"{b} B"

    # ── mutable state ─────────────────────────────────────────────────────────
    _all      = []
    _disp     = []
    _sel      = [None]
    _cache    = {}   # path → PIL.Image | None
    _photos   = []   # keep PhotoImage refs alive
    _sel_card = [None]
    _filt     = {"q": "", "cam": "Todas", "date": "Todas",
                 "media": "Todos", "sub": "Todas"}

    # ── outer container ────────────────────────────────────────────────────────
    outer = tk.Frame(main, bg=BG)
    outer.pack(fill="both", expand=True)

    # status bar (packed first → stays at bottom)
    bot = tk.Frame(outer, bg="#040e1f", height=26)
    bot.pack(fill="x", side="bottom")
    bot.pack_propagate(False)
    _st  = tk.Label(bot, text="Escaneando…", fg="#475569", bg="#040e1f",
                    font=("Segoe UI", 8))
    _st.pack(side="left", padx=12)
    _cnt = tk.Label(bot, text="", fg="#475569", bg="#040e1f", font=("Segoe UI", 8))
    _cnt.pack(side="right", padx=12)

    # header
    hdr = tk.Frame(outer, bg=BG)
    hdr.pack(fill="x", padx=20, pady=(14, 6))
    tk.Label(hdr, text="Reproducción", fg="#FFFFFF", bg=BG,
             font=("Segoe UI", 20, "bold")).pack(side="left")
    tk.Label(hdr, text="  Biblioteca de evidencias y grabaciones",
             fg="#9DB2D4", bg=BG, font=("Segoe UI", 10)).pack(side="left", pady=(8, 0))

    def _open_folder():
        p = _rb()
        try:
            os.startfile(p if os.path.isdir(p) else SNAPSHOTS_DIR)
        except Exception:
            pass

    tk.Button(hdr, text="↻ Actualizar", bg=PANEL, fg="white", relief="flat",
              font=("Segoe UI", 9, "bold"), padx=12, pady=5, cursor="hand2",
              command=lambda: _load()).pack(side="right")
    tk.Button(hdr, text="📂 Carpetas", bg=PANEL, fg="#9DB2D4", relief="flat",
              font=("Segoe UI", 9), padx=12, pady=5, cursor="hand2",
              command=_open_folder).pack(side="right", padx=(0, 6))

    # stats chips
    sbar = tk.Frame(outer, bg=PANEL, height=52,
                    highlightbackground=BORDER, highlightthickness=1)
    sbar.pack(fill="x", padx=20, pady=(0, 8))
    sbar.pack_propagate(False)
    sin = tk.Frame(sbar, bg=PANEL)
    sin.place(relx=0, rely=0.5, anchor="w", x=16)

    _sv = {}
    def _chip(key, lbl, col):
        c = tk.Frame(sin, bg=PANEL)
        c.pack(side="left", padx=14)
        v = tk.Label(c, text="—", fg=col, bg=PANEL, font=("Segoe UI", 15, "bold"))
        v.pack()
        tk.Label(c, text=lbl, fg="#475569", bg=PANEL, font=("Segoe UI", 8)).pack()
        _sv[key] = v
        tk.Frame(sin, bg=BORDER, width=1).pack(side="left", fill="y", pady=8)

    _chip("videos", "Videos",          "#2D7FF9")
    _chip("images", "Imágenes",        "#8b5cf6")
    _chip("today",  "Hoy",             "#10b981")
    _chip("week",   "Esta semana",     "#f59e0b")
    _chip("space",  "Almacenamiento",  "#94a3b8")

    # filter bar
    fb = tk.Frame(outer, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
    fb.pack(fill="x", padx=20, pady=(0, 8))
    fr = tk.Frame(fb, bg=PANEL)
    fr.pack(fill="x", padx=10, pady=8)

    def _vsep():
        tk.Frame(fr, bg=BORDER, width=1).pack(side="left", fill="y", pady=2, padx=6)

    # search box
    search_var = tk.StringVar()
    tk.Label(fr, text="🔍", fg="#475569", bg=PANEL,
             font=("Segoe UI", 10)).pack(side="left")
    se = tk.Entry(fr, textvariable=search_var, bg="#0A1828", fg="white",
                  insertbackground="white", relief="flat", font=("Segoe UI", 9),
                  width=22, highlightbackground=BORDER, highlightthickness=1)
    se.pack(side="left", ipady=4, padx=(3, 0))
    search_var.trace_add("write", lambda *_a: _apply_filters())

    _vsep()

    # camera OptionMenu (populated after scan)
    tk.Label(fr, text="Cámara:", fg="#64748b", bg=PANEL,
             font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 3))
    cam_var    = tk.StringVar(value="Todas")
    _cam_frame = tk.Frame(fr, bg=PANEL)
    _cam_frame.pack(side="left")

    _vsep()

    # date pills
    tk.Label(fr, text="Fecha:", fg="#64748b", bg=PANEL,
             font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 3))
    _dbtns = {}
    for _dv in ["Todas", "Hoy", "Ayer", "Semana", "Mes"]:
        def _dc(v=_dv):
            _filt["date"] = v
            _pills()
            _apply_filters()
        b = tk.Button(fr, text=_dv, relief="flat", cursor="hand2",
                      font=("Segoe UI", 8), padx=7, pady=2, command=_dc)
        b.pack(side="left", padx=1)
        _dbtns[_dv] = b

    _vsep()

    # media type pills
    tk.Label(fr, text="Tipo:", fg="#64748b", bg=PANEL,
             font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 3))
    _mbtns = {}
    for _mv in ["Todos", "Videos", "Imágenes"]:
        def _mc(v=_mv):
            _filt["media"] = v
            _pills()
            _apply_filters()
        b = tk.Button(fr, text=_mv, relief="flat", cursor="hand2",
                      font=("Segoe UI", 8), padx=7, pady=2, command=_mc)
        b.pack(side="left", padx=1)
        _mbtns[_mv] = b

    _vsep()

    # subtype pills
    tk.Label(fr, text="Evidencia:", fg="#64748b", bg=PANEL,
             font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 3))
    _sbtns = {}
    for _sv2 in ["Todas", "Analítica"]:
        def _sc(v=_sv2):
            _filt["sub"] = v
            _pills()
            _apply_filters()
        b = tk.Button(fr, text=_sv2, relief="flat", cursor="hand2",
                      font=("Segoe UI", 8), padx=7, pady=2, command=_sc)
        b.pack(side="left", padx=1)
        _sbtns[_sv2] = b

    def _pills():
        for k, b in _dbtns.items():
            b.config(bg=ACCENT if _filt["date"] == k else "#1e293b",
                     fg="white"  if _filt["date"] == k else "#64748b")
        for k, b in _mbtns.items():
            b.config(bg=ACCENT if _filt["media"] == k else "#1e293b",
                     fg="white"  if _filt["media"] == k else "#64748b")
        for k, b in _sbtns.items():
            b.config(bg=ACCENT if _filt["sub"] == k else "#1e293b",
                     fg="white"  if _filt["sub"] == k else "#64748b")

    _pills()

    # body: gallery (left) + detail panel (right)
    body = tk.Frame(outer, bg=BG)
    body.pack(fill="both", expand=True, padx=20, pady=(0, 4))

    # gallery
    gf  = tk.Frame(body, bg=BG)
    gf.pack(side="left", fill="both", expand=True)

    gcv = tk.Canvas(gf, bg=BG, highlightthickness=0)
    gsb = tk.Scrollbar(gf, orient="vertical", command=gcv.yview)
    gcv.configure(yscrollcommand=gsb.set)
    gcv.bind("<MouseWheel>",
             lambda e: gcv.yview_scroll(int(-1 * (e.delta / 120)), "units"))
    gsb.pack(side="right", fill="y")
    gcv.pack(side="left", fill="both", expand=True)

    gin  = tk.Frame(gcv, bg=BG)
    gwin = gcv.create_window((0, 0), window=gin, anchor="nw")
    gin.bind("<Configure>",
             lambda e: gcv.configure(scrollregion=gcv.bbox("all")))
    gcv.bind("<Configure>",
             lambda e: gcv.itemconfig(gwin, width=e.width))

    # detail panel (fixed 290px on right)
    det = tk.Frame(body, bg=PANEL, width=290,
                   highlightbackground=BORDER, highlightthickness=1)
    det.pack(side="right", fill="y", padx=(10, 0))
    det.pack_propagate(False)
    tk.Label(det, text="Selecciona\nuna evidencia",
             fg="#334155", bg=PANEL, font=("Segoe UI", 11),
             justify="center").place(relx=0.5, rely=0.5, anchor="center")

    # ── detail panel builder ──────────────────────────────────────────────────
    def _show_detail(it):
        _sel[0] = it
        for w in det.winfo_children():
            w.destroy()

        # top bar
        th = tk.Frame(det, bg="#0a1828")
        th.pack(fill="x")
        tk.Label(th, text="ANALÍTICA",
                 fg="#8b5cf6", bg="#0a1828", font=("Segoe UI", 8, "bold")
                 ).pack(side="left", padx=10, pady=5)
        tk.Label(th, text="🎬" if it["media"] == "video" else "📷",
                 fg="#64748b", bg="#0a1828",
                 font=("Segoe UI", 11)).pack(side="right", padx=10)

        # preview (162px tall)
        pf = tk.Frame(det, bg="#000000", height=162)
        pf.pack(fill="x")
        pf.pack_propagate(False)

        pil = _cache.get(it["path"])
        if pil:
            pc = pil.copy()
            pc.thumbnail((290, 162), Image.LANCZOS)
            bg_img = Image.new("RGB", (290, 162), (0, 0, 0))
            bg_img.paste(pc, ((290 - pc.width) // 2, (162 - pc.height) // 2))
            ph = ImageTk.PhotoImage(bg_img)
            _photos.append(ph)
            pl = tk.Label(pf, image=ph, bg="#000000", cursor="hand2")
            pl.pack(fill="both", expand=True)
            if it["media"] == "video":
                play = tk.Label(pf, text="▶", fg="white", bg="#000000",
                                font=("Segoe UI", 28), cursor="hand2")
                play.place(relx=0.5, rely=0.5, anchor="center")
                play.bind("<Button-1>", lambda e: _open_item(it))
                pl.bind("<Button-1>",   lambda e: _open_item(it))
        else:
            tk.Label(pf, text="📷" if it["media"] == "image" else "🎬",
                     fg="#334155", bg="#000000",
                     font=("Segoe UI", 36)).pack(expand=True)

        # metadata rows
        mf = tk.Frame(det, bg=PANEL)
        mf.pack(fill="x", padx=10, pady=6)

        def _row(lbl, val, vc="#e2e8f0"):
            r = tk.Frame(mf, bg=PANEL)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=lbl, fg="#475569", bg=PANEL, font=("Segoe UI", 8),
                     width=10, anchor="w").pack(side="left")
            tk.Label(r, text=str(val) if val else "—", fg=vc, bg=PANEL,
                     font=("Segoe UI", 8, "bold"), anchor="w",
                     wraplength=155).pack(side="left")

        _row("Cámara:",    it["cam_name"],           "#2D7FF9")
        _row("Fecha:",     it["dt_str"][:10] if it["dt_str"] else "")
        _row("Hora:",      it["dt_str"][11:] if len(it["dt_str"]) > 10 else "")
        _row("Evento:",    it["event"])
        _row("Tipo:",      "Video" if it["media"] == "video" else "Imagen")
        _row("Evidencia:", it["subtype"].title())
        _row("Tamaño:",    _fsz(it["size"]))
        if it["media"] == "video":
            dur = _get_dur(it["path"])
            if dur:
                _row("Duración:", dur, "#f59e0b")


        # path display
        short = ("…" + it["path"][-40:]) if len(it["path"]) > 43 else it["path"]
        tk.Label(det, text=short, fg="#334155", bg="#050e1f",
                 font=("Segoe UI", 7), wraplength=266, justify="left",
                 highlightbackground=BORDER, highlightthickness=1).pack(
            fill="x", padx=10, pady=(0, 6))

        # action buttons
        af = tk.Frame(det, bg=PANEL)
        af.pack(fill="x", padx=10, pady=(0, 8))
        tk.Button(af, text="▶ Abrir", bg=ACCENT, fg="white", relief="flat",
                  font=("Segoe UI", 9, "bold"), pady=5, cursor="hand2",
                  command=lambda: _open_item(it)).pack(
            side="left", fill="x", expand=True, padx=(0, 4))

        def _copy_path():
            root.clipboard_clear()
            root.clipboard_append(it["path"])
            show_notification("COPIADO", "Ruta copiada al portapapeles.", "#22c55e")

        tk.Button(af, text="📋", bg="#1e3a5f", fg="white", relief="flat",
                  font=("Segoe UI", 9), padx=8, pady=5, cursor="hand2",
                  command=_copy_path).pack(side="left", padx=(0, 4))

        is_adm = current_user in ("admin", "ROOT_USB") or any(
            u["user"] == current_user and u.get("role") == "Administrador"
            for u in users_data)
        if is_adm:
            tk.Button(af, text="🗑", bg="#7f1d1d", fg="white", relief="flat",
                      font=("Segoe UI", 9), padx=8, pady=5, cursor="hand2",
                      command=lambda i=it: _delete_item(i)).pack(side="left")

    # ── open / delete ─────────────────────────────────────────────────────────
    def _open_item(it):
        try:
            print(f"[REPRO] Ruta final       | {it['path']}")
            os.startfile(it["path"])
        except Exception as ex:
            show_notification("ERROR", str(ex), "#ef4444")

    def _delete_item(it):
        is_adm = current_user in ("admin", "ROOT_USB") or any(
            u["user"] == current_user and u.get("role") == "Administrador"
            for u in users_data)
        if not is_adm:
            show_notification("ACCESO DENEGADO", "Solo administradores.", "#ef4444")
            return
        try:
            os.remove(it["path"])
            show_notification("ELIMINADO", os.path.basename(it["path"]), "#22c55e")
            register_event(current_user, "EVIDENCIA_ELIMINADA", "WARNING", it["path"])
            _load()
        except Exception as ex:
            show_notification("ERROR", str(ex), "#ef4444")

    # ── gallery render ─────────────────────────────────────────────────────────
    def _render_gallery():
        print(f"[REPRO] Renderizando     | {len(_disp)} elementos en galería")
        for w in gin.winfo_children():
            w.destroy()
        _photos.clear()

        if not _disp:
            tk.Label(gin, text="Sin evidencias que mostrar",
                     fg="#334155", bg=BG,
                     font=("Segoe UI", 13)).pack(pady=60)
            _cnt.config(text="0 registros")
            return

        _cnt.config(text=f"{len(_disp)} registros")

        cw = gcv.winfo_width()
        if cw < 100:
            cw = 800
        cols = max(2, min(6, cw // (CARD_W + 14)))
        for ci in range(cols):
            gin.columnconfigure(ci, weight=1)

        CARD_PALETTE = {
            ("video", "analitica"): "#1a0d3c",
            ("image", "analitica"): "#1f0d2a",
        }
        BADGE_COL = {"analitica": "#7c3aed"}

        for idx, it in enumerate(_disp):
            ri = idx // cols
            ci = idx % cols

            cbg = CARD_PALETTE.get((it["media"], it["subtype"]), CARD)
            card = tk.Frame(gin, bg=cbg, width=CARD_W, height=CARD_H,
                            highlightbackground=BORDER, highlightthickness=1,
                            cursor="hand2")
            card.grid(row=ri, column=ci, padx=6, pady=6, sticky="nw")
            card.grid_propagate(False)

            # thumbnail zone
            tc = tk.Frame(card, bg="#000000", height=THH)
            tc.place(x=0, y=0, width=CARD_W, height=THH)

            pil = _cache.get(it["path"])
            if pil:
                pc = pil.copy()
                pc.thumbnail((CARD_W, THH), Image.LANCZOS)
                bg_img = Image.new("RGB", (CARD_W, THH), (0, 0, 0))
                bg_img.paste(pc, ((CARD_W - pc.width) // 2,
                                  (THH - pc.height) // 2))
                ph = ImageTk.PhotoImage(bg_img)
                _photos.append(ph)
                tk.Label(tc, image=ph, bg="#000000").place(
                    x=0, y=0, width=CARD_W, height=THH)
            else:
                ico = "🎬" if it["media"] == "video" else "📷"
                tk.Label(tc, text=ico, fg="#1e3a5f", bg="#000000",
                         font=("Segoe UI", 26)).place(
                    relx=0.5, rely=0.5, anchor="center")

            # subtype badge
            bco = BADGE_COL.get(it["subtype"], ACCENT)
            tk.Label(tc,
                     text="IA",
                     fg="white", bg=bco,
                     font=("Segoe UI", 6, "bold"),
                     padx=4, pady=1).place(x=4, y=4)

            # play icon for videos
            if it["media"] == "video":
                tk.Label(tc, text="▶", fg="white", bg="#000000",
                         font=("Segoe UI", 14)).place(
                    relx=0.5, rely=0.5, anchor="center")

            # info labels below thumbnail
            iy = THH + 4
            cam_s = (it["cam_name"][:18] + "…") if len(it["cam_name"]) > 19 else it["cam_name"]
            tk.Label(card, text=cam_s, fg="#e2e8f0", bg=cbg,
                     font=("Segoe UI", 8, "bold"), anchor="w").place(
                x=5, y=iy, width=CARD_W - 10)
            tk.Label(card, text=it["dt_str"][:16] if it["dt_str"] else "",
                     fg="#64748b", bg=cbg,
                     font=("Segoe UI", 7), anchor="w").place(
                x=5, y=iy + 16, width=CARD_W - 10)
            ev_s = (it["event"][:21] + "…") if len(it["event"]) > 22 else it["event"]
            tk.Label(card, text=ev_s, fg="#475569", bg=cbg,
                     font=("Segoe UI", 7), anchor="w").place(
                x=5, y=iy + 30, width=CARD_W - 10)

            # bind click/double-click recursively
            def _bind_card(c, i):
                def _on_click(e):
                    if _sel_card[0] and _sel_card[0].winfo_exists():
                        _sel_card[0].config(highlightbackground=BORDER)
                    c.config(highlightbackground=ACCENT)
                    _sel_card[0] = c
                    _show_detail(i)

                def _on_dbl(e):
                    _open_item(i)

                def _bind_w(w):
                    w.bind("<Button-1>", _on_click)
                    w.bind("<Double-Button-1>", _on_dbl)
                    for ch in w.winfo_children():
                        _bind_w(ch)

                _bind_w(c)

            _bind_card(card, it)

        gin.update_idletasks()
        gcv.configure(scrollregion=gcv.bbox("all"))

    # ── filtering ─────────────────────────────────────────────────────────────
    def _apply_filters():
        q  = search_var.get().lower().strip()
        cf = cam_var.get()
        df = _filt["date"]
        mf = _filt["media"]
        sf = _filt["sub"]

        print(f"[REPRO] Filtros          | total_all={len(_all)} | q={repr(q)} | cam={cf!r} | fecha={df} | tipo={mf} | sub={sf}")

        today = datetime.now().date()
        yday  = (datetime.now() - timedelta(days=1)).date()
        week  = (datetime.now() - timedelta(days=7)).date()
        month = (datetime.now() - timedelta(days=30)).date()

        out = []
        for it in _all:
            if q and not any(q in str(v).lower()
                             for v in [it["cam_name"], it["event"],
                                       it["fname"], it["dt_str"]]):
                continue
            if cf != "Todas" and it["cam_name"] != cf and it["cam_id"] != cf:
                continue
            if df != "Todas" and it["dt_obj"]:
                d = it["dt_obj"].date()
                if df == "Hoy"    and d != today: continue
                if df == "Ayer"   and d != yday:  continue
                if df == "Semana" and d < week:   continue
                if df == "Mes"    and d < month:  continue
            if mf == "Videos"   and it["media"] != "video": continue
            if mf == "Imágenes" and it["media"] != "image": continue
            if sf == "Analítica" and it["subtype"] != "analitica": continue
            print(f"[REPRO] Video agregado a galería | {it['fname']} | subtype={it['subtype']} | media={it['media']} | dt={it['dt_str']}")
            out.append(it)

        print(f"[REPRO] Filtros resultado | {len(out)} de {len(_all)} pasan")
        _disp.clear()
        _disp.extend(out)
        _render_gallery()

    # ── stats update ──────────────────────────────────────────────────────────
    def _update_stats():
        today = datetime.now().date()
        week  = (datetime.now() - timedelta(days=7)).date()
        nv = sum(1 for i in _all if i["media"] == "video")
        ni = sum(1 for i in _all if i["media"] == "image")
        nt = sum(1 for i in _all if i["dt_obj"] and i["dt_obj"].date() == today)
        nw = sum(1 for i in _all if i["dt_obj"] and i["dt_obj"].date() >= week)
        ts = sum(i["size"] for i in _all)
        _sv["videos"].config(text=str(nv))
        _sv["images"].config(text=str(ni))
        _sv["today"].config(text=str(nt))
        _sv["week"].config(text=str(nw))
        _sv["space"].config(text=_fsz(ts))

    # ── camera menu ───────────────────────────────────────────────────────────
    def _rebuild_cam():
        for w in _cam_frame.winfo_children():
            w.destroy()
        names = sorted({i["cam_name"] for i in _all})
        opts  = ["Todas"] + names
        if cam_var.get() not in opts:
            cam_var.set("Todas")
        om = tk.OptionMenu(_cam_frame, cam_var, *opts,
                           command=lambda v: _apply_filters())
        om.config(bg=PANEL, fg="white", relief="flat", font=("Segoe UI", 8),
                  highlightthickness=0, activebackground="#1e3a5f",
                  activeforeground="white", padx=4, pady=2)
        om["menu"].config(bg=PANEL, fg="white", font=("Segoe UI", 8),
                          activebackground=ACCENT, activeforeground="white")
        om.pack()

    # ── load (scan + background thumbnails) ───────────────────────────────────
    def _load():
        if outer.winfo_exists():
            _st.config(text="Escaneando archivos…")

        def _worker():
            items = _scan_all()
            if not outer.winfo_exists():
                return

            def _done():
                if not outer.winfo_exists():
                    return
                _all.clear()
                _all.extend(items)
                _update_stats()
                _rebuild_cam()
                _apply_filters()
                _st.config(text=f"Listo — {len(_all)} archivo(s)")
                _gen_thumbs()

            root.after(0, _done)

        _thr.Thread(target=_worker, daemon=True).start()

    def _gen_thumbs():
        todo = [i for i in _all if i["path"] not in _cache]

        def _worker():
            n = 0
            for it in todo:
                if not outer.winfo_exists():
                    return
                _cache[it["path"]] = _pil_thumb(it)
                n += 1
                if n % 6 == 0 and outer.winfo_exists():
                    root.after(0, _render_gallery)
            if outer.winfo_exists():
                root.after(0, _render_gallery)

        _thr.Thread(target=_worker, daemon=True).start()

    _load()

# =========================
# CONFIGURACIÓN
# =========================
def show_settings():

    import shutil as _shutil
    clear_main()

    if current_user != "admin":
        show_notification(
            "ACCESO DENEGADO",
            "Solo el administrador puede entrar.",
            "#ef4444"
        )
        return

    # ── scroll wrapper ─────────────────────────────────────────────
    outer = tk.Frame(main, bg="#020617")
    outer.pack(fill="both", expand=True)

    canvas = tk.Canvas(outer, bg="#020617", highlightthickness=0)
    vsb = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    container = tk.Frame(canvas, bg="#020617")
    cwin = canvas.create_window((0, 0), window=container, anchor="nw")
    container.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.bind(
        "<Configure>",
        lambda e: canvas.itemconfig(cwin, width=e.width)
    )

    def _on_wheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_wheel)

    padded = tk.Frame(container, bg="#020617")
    padded.pack(fill="both", expand=True, padx=30, pady=25)

    # ── TÍTULO ─────────────────────────────────────────────────────
    tk.Label(
        padded,
        text="Configuración del Sistema",
        fg="#e5e7eb",
        bg="#020617",
        font=("Segoe UI", 22, "bold")
    ).pack(anchor="w")

    tk.Label(
        padded,
        text="Administración avanzada del sistema.",
        fg="#64748b",
        bg="#020617",
        font=("Segoe UI", 11)
    ).pack(anchor="w", pady=(0, 20))

    # ── section header helper ──────────────────────────────────────
    def _section_hdr(text, color="#94a3b8"):
        frm = tk.Frame(padded, bg="#020617")
        frm.pack(fill="x", pady=(18, 6))
        tk.Label(
            frm, text=text,
            fg=color, bg="#020617",
            font=("Segoe UI", 13, "bold")
        ).pack(anchor="w")
        tk.Frame(frm, bg=color, height=2).pack(fill="x", pady=(4, 0))

    # ── storage helpers ────────────────────────────────────────────
    def _get_recordings_path():
        try:
            with open(RECORDINGS_DIR_FILE) as f:
                return json.load(f).get("path", "recordings")
        except Exception:
            return "recordings"

    def _count_files(path, exts):
        if not os.path.isdir(path):
            return 0
        count = 0
        for root, _, files in os.walk(path):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in exts:
                    count += 1
        return count

    def _dir_size(path):
        total = 0
        if not os.path.isdir(path):
            return 0
        try:
            for root, _, files in os.walk(path):
                for fn in files:
                    try:
                        total += os.path.getsize(os.path.join(root, fn))
                    except Exception:
                        pass
        except Exception:
            pass
        return total

    def _fmt_size(n):
        if n < 1024:
            return f"{n} B"
        if n < 1024 ** 2:
            return f"{n / 1024:.1f} KB"
        if n < 1024 ** 3:
            return f"{n / (1024 ** 2):.1f} MB"
        return f"{n / (1024 ** 3):.2f} GB"

    # ── card helper ────────────────────────────────────────────────
    def create_setting_card(
        parent, title, description, color,
        button_text, command, warning=False
    ):
        card = tk.Frame(
            parent, bg="#0b1220",
            width=380, height=240,
            highlightbackground="#1e293b", highlightthickness=1
        )
        card.pack_propagate(False)
        card.grid_propagate(False)

        tk.Label(
            card, text=title,
            fg=color, bg="#0b1220",
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w", padx=18, pady=(18, 8))

        tk.Label(
            card, text=description,
            fg="#94a3b8", bg="#0b1220",
            justify="left", wraplength=320,
            font=("Segoe UI", 10)
        ).pack(anchor="w", padx=18)

        if warning:
            tk.Label(
                card, text="⚠ Operación crítica",
                fg="#ef4444", bg="#0b1220",
                font=("Segoe UI", 9, "bold")
            ).pack(anchor="w", padx=18, pady=(10, 0))

        tk.Button(
            card, text=button_text,
            bg=color, fg="white", relief="flat",
            activebackground=color, activeforeground="white",
            cursor="hand2", padx=18, pady=8,
            command=command
        ).pack(anchor="w", padx=18, pady=18)

        return card

    # ── admin credential modal ─────────────────────────────────────
    def _admin_cred_dialog(action_name, on_ok):
        dlg = tk.Toplevel(root)
        dlg.title("Verificación de Administrador")
        dlg.configure(bg="#0b1220")
        dlg.geometry("420x300")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.transient(root)
        dlg.focus_set()

        tk.Label(
            dlg, text="Verificación Requerida",
            fg="#ef4444", bg="#0b1220",
            font=("Segoe UI", 15, "bold")
        ).pack(pady=(22, 4))

        tk.Label(
            dlg, text="Ingrese la contraseña o PIN del Administrador.",
            fg="#94a3b8", bg="#0b1220",
            font=("Segoe UI", 10)
        ).pack(pady=(0, 16))

        ef = tk.Frame(
            dlg, bg="#1e293b",
            highlightbackground="#334155", highlightthickness=1
        )
        ef.pack()
        entry = tk.Entry(
            ef, show="*",
            bg="#0b1220", fg="white",
            insertbackground="white", relief="flat",
            font=("Segoe UI", 12), width=28
        )
        entry.pack(padx=12, pady=10)
        entry.focus_set()

        err = tk.Label(
            dlg, text="",
            fg="#ef4444", bg="#0b1220",
            font=("Segoe UI", 9)
        )
        err.pack(pady=(6, 0))

        def _verify():
            h = hash_password(entry.get())
            if h == get_admin_password():
                dlg.destroy()
                on_ok()
                return
            for u in users_data:
                if u.get("role") == "Administrador":
                    if h == u.get("pin", hash_password("1234")):
                        dlg.destroy()
                        on_ok()
                        return
            register_event(
                current_user, action_name, "FAILED",
                "Credencial admin incorrecta en configuración"
            )
            err.config(text="Credencial incorrecta.")
            entry.delete(0, "end")

        entry.bind("<Return>", lambda e: _verify())

        br = tk.Frame(dlg, bg="#0b1220")
        br.pack(pady=16)
        tk.Button(
            br, text="Confirmar",
            bg="#2563eb", fg="white", relief="flat",
            padx=20, pady=8, cursor="hand2",
            command=_verify
        ).pack(side="left", padx=8)
        tk.Button(
            br, text="Cancelar",
            bg="#1e293b", fg="#94a3b8", relief="flat",
            padx=20, pady=8, cursor="hand2",
            command=dlg.destroy
        ).pack(side="left", padx=8)

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 1 — MANTENIMIENTO GENERAL
    # ══════════════════════════════════════════════════════════════
    _section_hdr("Mantenimiento General")

    def clear_audit_logs():
        logs = [{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": current_user,
            "action": "CLEAR_AUDIT",
            "status": "CRITICAL",
            "details": "Auditoría eliminada"
        }]
        save_audit_log(logs)
        show_notification("AUDITORÍA LIMPIADA", "La operación fue registrada.", "#22c55e")

    def clear_cameras():
        global rtsp_cameras
        global camera_names
        rtsp_cameras = []
        camera_names = {}
        save_rtsp_cameras(rtsp_cameras)
        save_camera_names()
        register_event(current_user, "CLEAR_CAMERAS", "CRITICAL", "Cámaras eliminadas")
        show_notification("CÁMARAS ELIMINADAS", "Todas las cámaras fueron eliminadas.", "#22c55e")

    row1 = tk.Frame(padded, bg="#020617")
    row1.pack(fill="x", pady=10)

    create_setting_card(
        row1,
        "🧹 Limpiar Auditoría",
        "Elimina todos los registros de auditoría excepto el evento actual.",
        "#dc2626", "Ejecutar",
        lambda: _admin_cred_dialog("CLEAR_AUDIT_AUTH", clear_audit_logs),
        warning=True
    ).pack(side="left", padx=10)

    create_setting_card(
        row1,
        "📷 Reiniciar Cámaras",
        "Borra cámaras RTSP registradas y aliases guardados.",
        "#ea580c", "Ejecutar",
        lambda: _admin_cred_dialog("CLEAR_CAMERAS_AUTH", clear_cameras),
        warning=True
    ).pack(side="left", padx=10)

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 2 — ALMACENAMIENTO Y EVIDENCIAS
    # ══════════════════════════════════════════════════════════════
    _section_hdr("Almacenamiento y Evidencias")

    # ── stats panel ────────────────────────────────────────────────
    stats_outer = tk.Frame(
        padded, bg="#0b1220",
        highlightbackground="#1e293b", highlightthickness=1
    )
    stats_outer.pack(fill="x", pady=(0, 14))

    stats_hdr_row = tk.Frame(stats_outer, bg="#0b1220")
    stats_hdr_row.pack(fill="x", padx=16, pady=(12, 8))

    tk.Label(
        stats_hdr_row,
        text="Estado del Almacenamiento",
        fg="#e5e7eb", bg="#0b1220",
        font=("Segoe UI", 12, "bold")
    ).pack(side="left")

    stats_row = tk.Frame(stats_outer, bg="#0b1220")
    stats_row.pack(fill="x", padx=16, pady=(0, 16))

    def _stat_box(parent, label, var):
        box = tk.Frame(
            parent, bg="#020617",
            highlightbackground="#1e293b", highlightthickness=1,
            width=190
        )
        box.pack_propagate(False)
        box.pack(side="left", padx=8, pady=4)
        tk.Label(
            box, textvariable=var,
            fg="#60a5fa", bg="#020617",
            font=("Segoe UI", 18, "bold")
        ).pack(pady=(12, 2))
        tk.Label(
            box, text=label,
            fg="#64748b", bg="#020617",
            font=("Segoe UI", 9)
        ).pack(pady=(0, 10))

    v_recs = tk.StringVar(value="…")
    v_imgs = tk.StringVar(value="…")
    v_used = tk.StringVar(value="…")
    v_free = tk.StringVar(value="…")

    _stat_box(stats_row, "Grabaciones", v_recs)
    _stat_box(stats_row, "Imágenes / Snapshots", v_imgs)
    _stat_box(stats_row, "Espacio Utilizado", v_used)
    _stat_box(stats_row, "Espacio Libre en Disco", v_free)

    def _refresh_stats():
        rec_path = _get_recordings_path()
        n_recs = _count_files(rec_path, {".avi", ".mp4", ".mkv"})
        n_imgs = _count_files(SNAPSHOTS_DIR, {".jpg", ".jpeg", ".png"})
        used   = _dir_size(rec_path) + _dir_size(SNAPSHOTS_DIR)
        try:
            free = _shutil.disk_usage(".").free
        except Exception:
            free = 0
        v_recs.set(str(n_recs))
        v_imgs.set(str(n_imgs))
        v_used.set(_fmt_size(used))
        v_free.set(_fmt_size(free))

    _refresh_stats()

    tk.Button(
        stats_hdr_row,
        text="↻ Actualizar",
        bg="#1e3a5f", fg="#60a5fa",
        relief="flat", cursor="hand2",
        padx=10, pady=4,
        command=_refresh_stats
    ).pack(side="right")

    # ── evidence deletion functions ────────────────────────────────
    def _delete_recordings():
        rec_path = _get_recordings_path()
        if not os.path.isdir(rec_path):
            show_notification("SIN GRABACIONES", "No hay directorio de grabaciones.", "#f59e0b")
            return
        deleted = 0
        for root, _, files in os.walk(rec_path):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in {".avi", ".mp4", ".mkv"}:
                    try:
                        os.remove(os.path.join(root, fn))
                        deleted += 1
                    except Exception:
                        pass
        register_event(
            current_user, "DELETE_RECORDINGS", "CRITICAL",
            f"{deleted} grabaciones eliminadas"
        )
        _refresh_stats()
        show_notification("GRABACIONES ELIMINADAS", f"{deleted} archivo(s) eliminados.", "#22c55e")

    def _delete_images():
        if not os.path.isdir(SNAPSHOTS_DIR):
            show_notification("SIN IMÁGENES", "No hay directorio de snapshots.", "#f59e0b")
            return
        deleted = 0
        for root, _, files in os.walk(SNAPSHOTS_DIR):
            for fn in files:
                if os.path.splitext(fn)[1].lower() in {".jpg", ".jpeg", ".png"}:
                    try:
                        os.remove(os.path.join(root, fn))
                        deleted += 1
                    except Exception:
                        pass
        register_event(
            current_user, "DELETE_IMAGES", "CRITICAL",
            f"{deleted} imágenes eliminadas"
        )
        _refresh_stats()
        show_notification("IMÁGENES ELIMINADAS", f"{deleted} archivo(s) eliminados.", "#22c55e")

    def _delete_evidences():
        _delete_recordings()
        _delete_images()

    def _confirm_then_auth(confirm_msg, action_name, action_fn):
        if messagebox.askyesno("Confirmar eliminación", confirm_msg, icon="warning"):
            _admin_cred_dialog(action_name, action_fn)

    row2 = tk.Frame(padded, bg="#020617")
    row2.pack(fill="x", pady=10)

    create_setting_card(
        row2,
        "🎬 Eliminar Grabaciones",
        "Elimina todos los videos almacenados por el sistema.\nSnapshots e imágenes se conservan.",
        "#b45309", "Eliminar",
        lambda: _confirm_then_auth(
            "¿Desea eliminar todas las grabaciones almacenadas?",
            "DELETE_RECORDINGS_AUTH",
            _delete_recordings
        ),
        warning=True
    ).pack(side="left", padx=10)

    create_setting_card(
        row2,
        "🖼 Eliminar Imágenes",
        "Elimina snapshots, capturas de eventos\ny evidencias fotográficas.",
        "#b45309", "Eliminar",
        lambda: _confirm_then_auth(
            "¿Desea eliminar todas las imágenes almacenadas?",
            "DELETE_IMAGES_AUTH",
            _delete_images
        ),
        warning=True
    ).pack(side="left", padx=10)

    create_setting_card(
        row2,
        "🗑 Eliminar Evidencias",
        "Elimina grabaciones e imágenes.\nConfiguraciones y auditoría se conservan.",
        "#dc2626", "Eliminar Todo",
        lambda: _confirm_then_auth(
            "¿Desea eliminar todas las evidencias\n(grabaciones e imágenes)?",
            "DELETE_EVIDENCES_AUTH",
            _delete_evidences
        ),
        warning=True
    ).pack(side="left", padx=10)

    # ══════════════════════════════════════════════════════════════
    # SECCIÓN 3 — RESTAURAR SISTEMA
    # ══════════════════════════════════════════════════════════════
    _section_hdr("Restaurar Sistema", color="#ef4444")

    reset_card = tk.Frame(
        padded, bg="#1a0a0a",
        highlightbackground="#7f1d1d", highlightthickness=2
    )
    reset_card.pack(fill="x", pady=(6, 30))

    rc_inner = tk.Frame(reset_card, bg="#1a0a0a")
    rc_inner.pack(fill="x", padx=24, pady=20)

    tk.Label(
        rc_inner,
        text="⚠ Restaurar Configuración de Fábrica",
        fg="#ef4444", bg="#1a0a0a",
        font=("Segoe UI", 16, "bold")
    ).pack(anchor="w")

    tk.Label(
        rc_inner,
        text=(
            "Esta acción eliminará permanentemente:\n"
            "  • Todas las cámaras configuradas (USB, RTSP, alias y selección)\n"
            "  • Toda la configuración de IA (reglas, zonas, operadores)\n"
            "  • Todas las grabaciones y evidencias fotográficas\n"
            "  • Todo el historial de eventos, registros y auditoría\n\n"
            "Se conservan: usuarios, código fuente, modelos YOLO y el programa."
        ),
        fg="#fca5a5", bg="#1a0a0a",
        justify="left",
        font=("Segoe UI", 10)
    ).pack(anchor="w", pady=(10, 16))

    def _do_factory_reset():
        global rtsp_cameras, camera_names, ai_config
        errors = []

        # 1. Camera list and aliases
        try:
            rtsp_cameras = []
            camera_names = {}
            save_rtsp_cameras(rtsp_cameras)
            save_camera_names()
        except Exception as e:
            errors.append(f"cameras: {e}")

        # 2. Selected cameras
        try:
            with open(SELECTED_CAMERAS_FILE, "w") as _f:
                json.dump([], _f)
        except Exception as e:
            errors.append(f"selected_cameras: {e}")

        # 3. AI config (detections, zones, operators, rules)
        try:
            ai_config = {}
            save_ai_config()
        except Exception as e:
            errors.append(f"ai_config: {e}")

        # 4. Recordings
        try:
            _rp = _get_recordings_path()
            if os.path.isdir(_rp):
                for _fn in list(os.listdir(_rp)):
                    if os.path.splitext(_fn)[1].lower() in {".avi", ".mp4", ".mkv"}:
                        try:
                            os.remove(os.path.join(_rp, _fn))
                        except Exception:
                            pass
        except Exception as e:
            errors.append(f"recordings: {e}")

        # 5. Snapshots / images
        try:
            if os.path.isdir(SNAPSHOTS_DIR):
                for _fn in list(os.listdir(SNAPSHOTS_DIR)):
                    if os.path.splitext(_fn)[1].lower() in {".jpg", ".jpeg", ".png"}:
                        try:
                            os.remove(os.path.join(SNAPSHOTS_DIR, _fn))
                        except Exception:
                            pass
        except Exception as e:
            errors.append(f"snapshots: {e}")

        # 6. Events log
        try:
            with open(EVENTS_LOG_FILE, "w") as _f:
                json.dump([], _f)
            _events_list.clear()
        except Exception as e:
            errors.append(f"events: {e}")

        # 7. Technical registry log
        try:
            with open(REGISTRY_LOG_FILE, "w", encoding="utf-8") as _f:
                _f.write("")
        except Exception as e:
            errors.append(f"registry: {e}")

        # 8. Audit — keep only this reset event
        _reset_evt = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": current_user,
            "action": "FACTORY_RESET",
            "status": "CRITICAL",
            "details": f"Restauración total de fábrica. Errores: {len(errors)}"
        }
        try:
            save_audit_log([_reset_evt])
        except Exception as e:
            errors.append(f"audit: {e}")

        if errors:
            show_notification(
                "RESET CON ERRORES",
                f"Reset ejecutado con {len(errors)} error(es).",
                "#f59e0b"
            )
        else:
            show_notification(
                "RESET COMPLETADO",
                "Sistema restaurado a configuración de fábrica.",
                "#22c55e"
            )
        show_settings()

    def _reset_step3():
        if messagebox.askokcancel(
            "Confirmación Final",
            "¿Está completamente seguro?\n\n"
            "Esta es su última oportunidad de cancelar.\n"
            "Todos los datos seleccionados serán eliminados definitivamente.",
            icon="warning"
        ):
            _do_factory_reset()

    def _reset_step2():
        _admin_cred_dialog("FACTORY_RESET_AUTH", _reset_step3)

    def _reset_step1():
        if messagebox.askokcancel(
            "⚠ Restaurar Configuración de Fábrica",
            "Esta acción eliminará permanentemente todas las configuraciones,\n"
            "grabaciones, imágenes y registros del sistema.\n\n"
            "Esta operación NO puede deshacerse.\n\n"
            "¿Desea continuar?",
            icon="warning"
        ):
            _reset_step2()

    tk.Button(
        rc_inner,
        text="⚠ Restaurar Configuración de Fábrica",
        bg="#7f1d1d", fg="white",
        relief="flat",
        activebackground="#991b1b", activeforeground="white",
        cursor="hand2", padx=20, pady=10,
        font=("Segoe UI", 11, "bold"),
        command=_reset_step1
    ).pack(anchor="w")
# =========================
# CONFIGURACIÓN IA
# =========================
AI_SECTIONS = {
    "Detecciones": {
        "HUMANOS": ["Persona"],
        "ARMAS": [
            "Arma Blanca",
            "Arma Corta",
            "Arma Larga"
        ],
        "EPP": [
            "Casco Seguridad",
            "Bata Industrial",
            "Botas Seguridad",
            "Cubrebocas Seguridad",
            "Lentes Seguridad",
            "Tapones Auditivos",
            "Audífono Inalámbrico",
            "No Celular",
            "Caídas"
        ],
        "OBJETOS": [
            "Extintor",
            "Mochila",
            "Caja",
            "Carro de Carga"
        ]
    },
    "Reglas Inteligentes": {
        "CONDUCTA": ["Persona Corriendo", "Persona Inmóvil", "Persona Caída"],
        "OBJETOS INTELIGENTES": ["Objeto Abandonado", "Objeto Movido"]
    },
    "Gestión Operadores": "__operadores__",
    "Zonas Inteligentes": "__zonas_inteligentes__",
}


def _show_recording_modal(camera_id, on_confirm):
    """Modal de configuración de grabación.
    on_confirm(mode: str, pre_s: int, post_s: int) se llama al guardar.
    mode: "none" | "continuous" | "smart" | "hybrid"
    """
    is_admin = current_user in ("admin", "ROOT_USB") or any(
        u.get("role", "") == "Administrador" for u in users_data if u["user"] == current_user)

    # Leer modo actual guardado en ai_config
    cam_cfg       = ai_config.get(camera_id, {})
    saved_mode    = cam_cfg.get("recording_mode", "hybrid")
    saved_pre_s   = int(cam_cfg.get("rec_pre_s",  30))
    saved_post_s  = int(cam_cfg.get("rec_post_s", 30))
    # Compatibilidad con claves antiguas
    if cam_cfg.get("Grabación Total"):
        saved_mode = "continuous"
    elif cam_cfg.get("Grabación por Evento"):
        saved_mode = "hybrid"
    elif cam_cfg.get("Grabación Inteligente"):
        saved_mode = "smart"

    modal = tk.Toplevel(root)
    modal.title("Configuración de Grabación")
    modal.configure(bg="#0f172a")
    modal.resizable(False, False)
    modal.grab_set()

    # Centrar en pantalla
    modal.update_idletasks()
    mw, mh = 460, 480 if is_admin else 390
    sx = root.winfo_x() + (root.winfo_width()  - mw) // 2
    sy = root.winfo_y() + (root.winfo_height() - mh) // 2
    modal.geometry(f"{mw}x{mh}+{sx}+{sy}")

    # ── Header ────────────────────────────────────────────────────────────────
    tk.Label(modal, text="CONFIGURACIÓN DE GRABACIÓN",
             fg="#e5e7eb", bg="#0f172a",
             font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=20, pady=(18, 2))
    tk.Label(modal, text="¿Cómo desea grabar esta cámara?",
             fg="#64748b", bg="#0f172a",
             font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=(0, 12))
    tk.Frame(modal, bg="#1e293b", height=1).pack(fill="x", padx=20, pady=(0, 10))

    # ── Radio buttons ─────────────────────────────────────────────────────────
    selected_mode = tk.StringVar(value=saved_mode)
    MODES = [
        ("none",       "Sin grabación",              "Solo eventos y auditoría. No guarda video."),
        ("continuous", "Grabación continua",          "Graba 24/7 sin interrupciones."),
        ("smart",      "Grabación inteligente",       "Graba solo cuando hay eventos IA activos."),
        ("hybrid",     "Grabación híbrida  ★ Recomendada", "Buffer circular: pre y post-evento guardados."),
    ]
    radio_frame = tk.Frame(modal, bg="#0f172a")
    radio_frame.pack(fill="x", padx=20)
    for val, label, desc in MODES:
        row = tk.Frame(radio_frame, bg="#0f172a")
        row.pack(fill="x", pady=3)
        rb = tk.Radiobutton(row, variable=selected_mode, value=val,
                            text=label,
                            fg="#e5e7eb" if val != "hybrid" else "#60a5fa",
                            bg="#0f172a", activebackground="#0f172a",
                            selectcolor="#1e3a5f",
                            font=("Segoe UI", 10, "bold"), anchor="w", cursor="hand2")
        rb.pack(anchor="w")
        tk.Label(row, text=f"   {desc}", fg="#64748b", bg="#0f172a",
                 font=("Segoe UI", 9)).pack(anchor="w")

    tk.Frame(modal, bg="#1e293b", height=1).pack(fill="x", padx=20, pady=(10, 8))

    # ── Configuración Avanzada (solo administradores) ─────────────────────────
    if is_admin:
        adv_label = tk.Label(modal, text="⚙  Configuración Avanzada  (Administrador)",
                             fg="#94a3b8", bg="#0f172a",
                             font=("Segoe UI", 9, "bold"), cursor="hand2")
        adv_label.pack(anchor="w", padx=20, pady=(0, 4))
        adv_frame = tk.Frame(modal, bg="#111827", bd=0)

        PRE_OPTS  = [15, 30, 60, 120]
        POST_OPTS = [15, 30, 60, 120]
        pre_var  = tk.IntVar(value=saved_pre_s  if saved_pre_s  in PRE_OPTS  else 30)
        post_var = tk.IntVar(value=saved_post_s if saved_post_s in POST_OPTS else 30)

        def _toggle_adv():
            if adv_frame.winfo_ismapped():
                adv_frame.pack_forget()
            else:
                adv_frame.pack(fill="x", padx=20, pady=(0, 6))
        adv_label.config(command=None)
        adv_label.bind("<Button-1>", lambda _e: _toggle_adv())

        row_pre = tk.Frame(adv_frame, bg="#111827")
        row_pre.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(row_pre, text="Pre-evento:", fg="#e5e7eb", bg="#111827",
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        pre_menu = tk.OptionMenu(row_pre, pre_var, *PRE_OPTS)
        pre_menu.config(bg="#1e293b", fg="#e5e7eb", relief="flat",
                        font=("Segoe UI", 9), activebackground="#374151")
        pre_menu["menu"].config(bg="#1e293b", fg="#e5e7eb")
        pre_menu.pack(side="left", padx=6)
        tk.Label(row_pre, text="segundos", fg="#64748b", bg="#111827",
                 font=("Segoe UI", 9)).pack(side="left")

        row_post = tk.Frame(adv_frame, bg="#111827")
        row_post.pack(fill="x", padx=10, pady=(0, 8))
        tk.Label(row_post, text="Post-evento:", fg="#e5e7eb", bg="#111827",
                 font=("Segoe UI", 9), width=14, anchor="w").pack(side="left")
        post_menu = tk.OptionMenu(row_post, post_var, *POST_OPTS)
        post_menu.config(bg="#1e293b", fg="#e5e7eb", relief="flat",
                         font=("Segoe UI", 9), activebackground="#374151")
        post_menu["menu"].config(bg="#1e293b", fg="#e5e7eb")
        post_menu.pack(side="left", padx=6)
        tk.Label(row_post, text="segundos", fg="#64748b", bg="#111827",
                 font=("Segoe UI", 9)).pack(side="left")
    else:
        pre_var  = tk.IntVar(value=saved_pre_s)
        post_var = tk.IntVar(value=saved_post_s)

    # ── Botones ───────────────────────────────────────────────────────────────
    btn_row = tk.Frame(modal, bg="#0f172a")
    btn_row.pack(side="bottom", fill="x", padx=20, pady=16)

    def _do_cancel():
        modal.destroy()

    def _do_save():
        mode  = selected_mode.get()
        pre_s = pre_var.get()
        post_s = post_var.get()
        modal.destroy()
        on_confirm(mode, pre_s, post_s)

    tk.Button(btn_row, text="Cancelar", bg="#374151", fg="white",
              relief="flat", font=("Segoe UI", 10), padx=18, pady=7,
              cursor="hand2", command=_do_cancel).pack(side="left")
    tk.Button(btn_row, text="Guardar", bg="#16a34a", fg="white",
              relief="flat", font=("Segoe UI", 10, "bold"), padx=22, pady=7,
              cursor="hand2", command=_do_save).pack(side="right")


def open_ai_selection():

    clear_main()

    container = tk.Frame(
        main,
        bg="#020617"
    )

    container.pack(
        fill="both",
        expand=True,
        padx=20,
        pady=20
    )
    

    # =========================
    # HEADER
    # =========================
    tk.Label(
        container,
        text="Configuración Inteligencia Artificial",
        fg="#e5e7eb",
        bg="#020617",
        font=("Segoe UI", 22, "bold")
    ).pack(anchor="w")

    tk.Label(
        container,
        text="Configure reglas, zonas y comportamiento inteligente.",
        fg="#64748b",
        bg="#020617",
        font=("Segoe UI", 11)
    ).pack(anchor="w", pady=(0, 20))

    # =========================
    # SELECTOR CÁMARA IA
    # =========================
    camera_selector_frame = tk.Frame(
        container,
        bg="#020617"
    )

    camera_selector_frame.pack(
        fill="x",
        pady=(0, 15)
    )

    tk.Label(
        camera_selector_frame,
        text="Cámara activa:",
        fg="white",
        bg="#020617",
        font=("Segoe UI", 11, "bold")
    ).pack(side="left")

    # =========================
    # MAPEAR CÁMARAS
    # =========================
    all_cameras = scan_usb_cameras() + rtsp_cameras

    selected_camera_names = []
    camera_name_to_id = {}

    for cam in all_cameras:

        if cam["id"] not in selected_cameras:
            continue

        display_name = cam["name"]

        if cam.get("alias"):
            display_name += f" ({cam['alias']})"

        selected_camera_names.append(display_name)

        camera_name_to_id[display_name] = cam["id"]

    # =========================
    # DROPDOWN
    # =========================
    selected_ai_camera = tk.StringVar()

    if selected_camera_names:
        selected_ai_camera.set(selected_camera_names[0])

    camera_dropdown = tk.OptionMenu(
        camera_selector_frame,
        selected_ai_camera,
        *selected_camera_names
    )

    camera_dropdown.config(
        bg="#0f172a",
        fg="white",
        activebackground="#2563eb",
        activeforeground="white",
        relief="flat",
        font=("Segoe UI", 10),
        highlightthickness=0
    )

    camera_dropdown["menu"].config(
        bg="#0f172a",
        fg="white"
    )

    camera_dropdown.pack(
        side="left",
        padx=(10, 0)
    )

    # =========================
    # TOP ACTION BAR
    # =========================
    top_bar = tk.Frame(
        container,
        bg="#020617"
    )

    top_bar.pack(
        fill="x",
        pady=(0, 10)
    )

    # =========================
    # BOTÓN GUARDAR
    # =========================
    def _on_save_with_recording():
        cam_id = str(camera_name_to_id[selected_ai_camera.get()])

        def _apply_and_save(mode, pre_s, post_s):
            if cam_id not in ai_config:
                ai_config[cam_id] = {}
            ai_config[cam_id]["recording_mode"] = mode
            ai_config[cam_id]["rec_pre_s"]      = pre_s
            ai_config[cam_id]["rec_post_s"]     = post_s
            # Limpiar claves antiguas para evitar conflictos en _init_ai_for_cam
            for _old in ("Grabación Total", "Grabación por Evento", "Grabación Inteligente",
                         "Segundos Antes Evento", "Segundos Después Evento"):
                ai_config[cam_id].pop(_old, None)
            save_ai_config()
            # Aplicar inmediatamente si la IA ya está activa en esta cámara
            if _ai_started.get(cam_id):
                if mode == "continuous":
                    _recording_enabled[cam_id] = True
                    _cam_record_mode[cam_id]   = "continuous"
                elif mode in ("hybrid", "event"):
                    _recording_enabled[cam_id] = True
                    _cam_record_mode[cam_id]   = "hybrid"
                    _rec_pre_s[cam_id]          = pre_s
                    _rec_post_s[cam_id]         = post_s
                    _pre_event_buffer[cam_id]   = collections.deque(maxlen=pre_s * 20)
                elif mode == "smart":
                    _recording_enabled[cam_id] = True
                    _cam_record_mode[cam_id]   = "smart"
                    _rec_pre_s[cam_id]          = pre_s
                    _rec_post_s[cam_id]         = post_s
                    _pre_event_buffer[cam_id]   = collections.deque(maxlen=pre_s * 20)
                else:
                    _recording_enabled[cam_id] = False
                    _stop_writer(cam_id)
            mode_labels = {"none": "Sin grabación", "continuous": "Continua",
                           "smart": "Inteligente", "hybrid": "Híbrida"}
            show_notification("CONFIGURACIÓN GUARDADA",
                              f"Grabación: {mode_labels.get(mode, mode)} | "
                              f"Pre:{pre_s}s Post:{post_s}s",
                              "#16a34a")

        _show_recording_modal(cam_id, _apply_and_save)

    save_ai_button = tk.Button(
        top_bar,
        text="Guardar configuración",
        bg="#16a34a",
        fg="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=18,
        pady=8,
        cursor="hand2",
        command=_on_save_with_recording
    )

    save_ai_button.pack(
        side="right",
        padx=(10,0)
    )

    # =========================
    # RESTABLECER
    # =========================
    def reset_current_camera():

        camera_id = str(
            camera_name_to_id[
                selected_ai_camera.get()
            ]
        )

        ai_config[camera_id] = {}

        save_ai_config()

        show_notification(
            "CONFIGURACIÓN LIMPIADA",
            "La cámara fue restablecida.",
            "#f59e0b"
        )

        open_ai_selection()

    reset_button = tk.Button(
        top_bar,
        text="Restablecer cámara",
        bg="#7f1d1d",
        fg="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=18,
        pady=8,
        cursor="hand2",
        command=reset_current_camera
    )

    reset_button.pack(
        side="right"
    )

    def clear_all_zones():
        cam_id = str(camera_name_to_id[selected_ai_camera.get()])
        if cam_id in ai_config:
            keys_to_delete = [k for k in ai_config[cam_id] if "_zone" in k]
            for k in keys_to_delete:
                ai_config[cam_id].pop(k, None)
        save_ai_config()
        draw_all_zones()
        show_notification("ZONAS LIMPIADAS", "Todas las zonas fueron eliminadas.", "#22c55e")

    tk.Button(
        top_bar,
        text="Limpiar zonas",
        bg="#374151",
        fg="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=18,
        pady=8,
        cursor="hand2",
        command=clear_all_zones
    ).pack(side="right", padx=(10, 0))

    # =========================
    # BODY
    # =========================
    body = tk.Frame(
        container,
        bg="#020617"
    )

    body.pack(
        fill="both",
        expand=True
    )

    # =========================
    # VIDEO PANEL
    # =========================
    video_panel = tk.Frame(
        body,
        bg="#000000",
        highlightbackground="#1e293b",
        highlightthickness=1
    )

    video_panel.place(
        relx=0,
        rely=0,
        relwidth=0.76,
        relheight=0.82
    )

    # VIDEO TITLE
    tk.Label(
        video_panel,
        text="Cámara Seleccionada",
        fg="#3b82f6",
        bg="#000000",
        font=("Segoe UI", 12, "bold")
    ).pack(anchor="w", padx=15, pady=10)

    # VIDEO FAKE
    fake_video = tk.Frame(
        video_panel,
        bg="#050816"
    )

    fake_video.pack(
        fill="both",
        expand=True,
        padx=15,
        pady=(0,15)
    )
    # =========================
    # CONTROLES PREVIEW IA
    # =========================
    controls_frame = tk.Frame(
        video_panel,
        bg="#000000"
    )

    controls_frame.pack(
        fill="x",
        padx=15,
        pady=(0, 10)
    )

    # =========================
    # CANVAS VIDEO + ZONA DRAWING
    # =========================
    ai_canvas = tk.Canvas(fake_video, bg="black", highlightthickness=0)
    ai_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

    _vid_img_id = [None]
    _zone_draw_fn = [None]

    def _ai_configure(**kw):
        img = kw.pop("image", ...)
        if img is not ...:
            ai_canvas.imgtk = img
            ai_canvas.image = img
            if img == "" or img is None:
                if _vid_img_id[0]:
                    ai_canvas.delete(_vid_img_id[0])
                    _vid_img_id[0] = None
            else:
                if _vid_img_id[0] is None:
                    _vid_img_id[0] = ai_canvas.create_image(0, 0, anchor="nw", image=img)
                else:
                    ai_canvas.itemconfig(_vid_img_id[0], image=img)
                if _zone_draw_fn[0]:
                    _zone_draw_fn[0]()
        if kw:
            tk.Canvas.configure(ai_canvas, **kw)

    ai_canvas.configure = _ai_configure
    ai_canvas.imgtk = None
    ai_canvas.image = None
    ai_preview_label = ai_canvas

    # =========================
    # INICIAR CÁMARA IA
    # =========================
    def start_ai_camera():

        selected_name = selected_ai_camera.get()

        camera_id = camera_name_to_id[selected_name]

        # =========================
        # BUSCAR CÁMARA
        # =========================
        selected_cam = None

        for cam in all_cameras:

            if cam["id"] == camera_id:

                selected_cam = cam
                break

        if selected_cam is None:
            return

        # =========================
        # RTSP
        # =========================
        if "rtsp" in selected_cam:

            start_rtsp_preview(
                selected_cam["rtsp"],
                ai_preview_label,
                selected_cam.get("transport", "tcp"),
                save_ai_button,
                real_cam_id=selected_cam["id"],
                enable_ai=True
            )

        # =========================
        # USB
        # =========================
        else:

            start_preview(
                selected_cam["id"],
                ai_preview_label,
                enable_ai=True
            )

    # =========================
    # DETENER CÁMARA IA
    # =========================
    def stop_ai_camera():

        stop_preview()

    # =========================
    # BOTÓN INICIAR
    # =========================
    start_ai_button = tk.Button(
        controls_frame,
        text="▶ Iniciar cámara",
        bg="#2563eb",
        fg="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=15,
        pady=8,
        cursor="hand2",
        command=start_ai_camera
    )

    start_ai_button.pack(
        side="left"
    )

    # =========================
    # BOTÓN DETENER
    # =========================
    stop_ai_button = tk.Button(
        controls_frame,
        text="■ Detener",
        bg="#7f1d1d",
        fg="white",
        relief="flat",
        font=("Segoe UI", 10, "bold"),
        padx=15,
        pady=8,
        cursor="hand2",
        command=stop_ai_camera
    )

    stop_ai_button.pack(
        side="left",
        padx=(10,0)
    )


    # =========================
    # RIGHT CONFIG PANEL
    # =========================
    config_panel = tk.Frame(
        body,
        bg="#0b1220",
        highlightbackground="#1e293b",
        highlightthickness=1
    )

    config_panel.place(
        relx=0.77,
        rely=0,
        relwidth=0.22,
        relheight=0.92
    )

    # =========================
    # PANEL TITLE
    # =========================
    tk.Label(
        config_panel,
        text="Configuración IA",
        fg="#e5e7eb",
        bg="#0b1220",
        font=("Segoe UI", 16, "bold")
    ).pack(anchor="w", padx=15, pady=15)

    # =========================
    # SCROLLABLE
    # =========================
    canvas = tk.Canvas(
        config_panel,
        bg="#0b1220",
        highlightthickness=0
    )

    scrollbar = tk.Scrollbar(
        config_panel,
        orient="vertical",
        command=canvas.yview,
        bg="#111827",
        troughcolor="#020617",
        activebackground="#2563eb",
        width=12
    )

    scroll_frame = tk.Frame(
        canvas,
        bg="#0b1220"
    )

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        )
    )

    canvas_window = canvas.create_window(
        (0,0),
        window=scroll_frame,
        anchor="nw"
    )
    def resize_scroll(event):

        canvas.itemconfig(
            canvas_window,
            width=event.width
        )

    canvas.bind(
        "<Configure>",
        resize_scroll
    )

    canvas.configure(
        yscrollcommand=scrollbar.set
    )

    canvas.pack(
        side="left",
        fill="both",
        expand=True
    )

    scrollbar.pack(
        side="right",
        fill="y",
        padx=(0, 4),
        pady=10
    )
    # =========================
    # SCROLL CON MOUSE
    # =========================
    def on_mousewheel(event):

        canvas.yview_scroll(
            -1 if event.delta > 0 else 1,
            "units"
        )

    canvas.bind_all(
        "<MouseWheel>",
        on_mousewheel
    )

    # =========================
    # TOGGLE CREATOR
    # =========================
    def create_toggle(parent, text, key):

        global ai_config

        # =========================
        # CÁMARA ACTIVA REAL
        # =========================
        camera_id = str(
            camera_name_to_id[
                selected_ai_camera.get()
            ]
        )

        # =========================
        # CREAR CONFIG SI NO EXISTE
        # =========================
        if camera_id not in ai_config:
            ai_config[camera_id] = {}

        # =========================
        # VALOR GUARDADO
        # =========================
        saved_value = ai_config[camera_id].get(key, False)

        row = tk.Frame(
            parent,
            bg="#111827"
        )

        row.pack(
            fill="x",
            padx=10,
            pady=5
        )

        enabled = tk.BooleanVar(value=saved_value)

        label = tk.Label(
            row,
            text=text,
            fg="#e5e7eb",
            bg="#111827",
            font=("Segoe UI", 10)
        )

        label.pack(
            side="left",
            padx=10,
            pady=10
        )

        def on_toggle():
            ai_config[camera_id][key] = enabled.get()
            # No auto-save: user must press "Guardar configuración"

        toggle = tk.Checkbutton(
            row,
            variable=enabled,
            command=on_toggle,
            bg="#111827",
            activebackground="#111827",
            selectcolor="#2563eb"
        )

        toggle.pack(
            side="right",
            padx=10
        )


    SECTION_COLORS = {
        "Detecciones": "#3b82f6",
        "Reglas Inteligentes": "#ef4444",
        "Gestión Operadores": "#f59e0b",
        "Zonas Inteligentes": "#f59e0b",
        "Sistema de Grabación": "#8b5cf6"
    }

    def create_input_field(parent, label_text, key, camera_id):
        row = tk.Frame(parent, bg="#111827")
        row.pack(fill="x", padx=10, pady=4)
        tk.Label(row, text=label_text, fg="#e5e7eb", bg="#111827",
                 font=("Segoe UI", 10)).pack(side="left", padx=10, pady=8)
        saved_val = ai_config.get(camera_id, {}).get(key, "")
        entry = tk.Entry(row, bg="#1f2937", fg="white", insertbackground="white",
                         relief="flat", width=10, font=("Segoe UI", 9))
        entry.insert(0, saved_val)
        entry.pack(side="right", padx=10, pady=6)
        def on_change(event=None):
            if camera_id not in ai_config:
                ai_config[camera_id] = {}
            ai_config[camera_id][key] = entry.get()
            save_ai_config()
        entry.bind("<FocusOut>", on_change)
        entry.bind("<Return>", on_change)

    def create_time_ampm_field(parent, label_text, key, camera_id):
        row = tk.Frame(parent, bg="#111827")
        row.pack(fill="x", padx=10, pady=4)
        tk.Label(row, text=label_text, fg="#e5e7eb", bg="#111827",
                 font=("Segoe UI", 10)).pack(side="left", padx=10, pady=8)

        saved = ai_config.get(camera_id, {}).get(key, "")
        h_val, m_val, ap_val = "08", "00", "AM"
        if saved:
            try:
                parts = saved.strip().split()
                if len(parts) == 2:
                    hm = parts[0].split(":")
                    h_val = hm[0].zfill(2)
                    m_val = hm[1].zfill(2)
                    ap_val = parts[1] if parts[1] in ("AM", "PM") else "AM"
            except:
                pass

        time_frame = tk.Frame(row, bg="#111827")
        time_frame.pack(side="right", padx=10)

        hour_var = tk.StringVar(value=h_val)
        min_var = tk.StringVar(value=m_val)
        ampm_var = tk.StringVar(value=ap_val)
        ampm_btn_ref = [None]

        def save_time(*args):
            h = hour_var.get().strip().zfill(2)
            m = min_var.get().strip().zfill(2)
            ap = ampm_var.get()
            if camera_id not in ai_config:
                ai_config[camera_id] = {}
            ai_config[camera_id][key] = f"{h}:{m} {ap}"
            save_ai_config()

        hour_entry = tk.Entry(time_frame, textvariable=hour_var,
                              bg="#1f2937", fg="white", insertbackground="white",
                              relief="flat", width=3, font=("Segoe UI", 10),
                              justify="center")
        hour_entry.pack(side="left")
        hour_entry.bind("<FocusOut>", save_time)
        hour_entry.bind("<Return>", save_time)

        tk.Label(time_frame, text=":", fg="white", bg="#111827",
                 font=("Segoe UI", 10, "bold")).pack(side="left")

        min_entry = tk.Entry(time_frame, textvariable=min_var,
                             bg="#1f2937", fg="white", insertbackground="white",
                             relief="flat", width=3, font=("Segoe UI", 10),
                             justify="center")
        min_entry.pack(side="left")
        min_entry.bind("<FocusOut>", save_time)
        min_entry.bind("<Return>", save_time)

        def toggle_ampm():
            new_ap = "PM" if ampm_var.get() == "AM" else "AM"
            ampm_var.set(new_ap)
            ampm_btn_ref[0].configure(
                text=new_ap,
                bg="#2563eb" if new_ap == "AM" else "#7c3aed"
            )
            save_time()

        btn = tk.Button(time_frame, text=ap_val,
                        bg="#2563eb" if ap_val == "AM" else "#7c3aed",
                        fg="white", relief="flat",
                        font=("Segoe UI", 9, "bold"),
                        width=3, padx=4, pady=2,
                        cursor="hand2", command=toggle_ampm)
        btn.pack(side="left", padx=(6, 0))
        ampm_btn_ref[0] = btn

    OPERATOR_ZONE_COLORS = ["#3b82f6", "#22c55e", "#f97316", "#a855f7", "#ef4444"]
    _zone_state = {"active": False, "op_idx": None, "sx": None, "sy": None, "rect_id": None}
    _zone_buttons = {}

    def draw_all_zones():
        ai_canvas.delete("zone")
        if _zone_state["rect_id"]:
            ai_canvas.delete(_zone_state["rect_id"])
            _zone_state["rect_id"] = None
        cam_id = str(camera_name_to_id[selected_ai_camera.get()])
        ops = ai_config.get(cam_id, {}).get("operators", [])
        w = max(ai_canvas.winfo_width(), 1)
        h = max(ai_canvas.winfo_height(), 1)
        for idx in range(len(ops)):
            z = ai_config.get(cam_id, {}).get(f"op_{idx}_zone")
            if z:
                color = OPERATOR_ZONE_COLORS[idx % len(OPERATOR_ZONE_COLORS)]
                x1 = int(z["x1"] * w)
                y1 = int(z["y1"] * h)
                x2 = int(z["x2"] * w)
                y2 = int(z["y2"] * h)
                ai_canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="zone")
                op_name = ai_config.get(cam_id, {}).get(f"op_{idx}_nombre", "") or f"Op.{idx+1}"
                ai_canvas.create_text(x1 + 6, y1 + 6, text=op_name,
                                      fill=color, anchor="nw",
                                      font=("Segoe UI", 9, "bold"), tags="zone")
        for key, color, label in [
            ("verde", "#22c55e", "Zona Verde"),
            ("amarilla", "#f59e0b", "Zona Amarilla"),
            ("roja", "#ef4444", "Zona Roja")
        ]:
            z = ai_config.get(cam_id, {}).get(f"smart_{key}_zone")
            if z:
                x1 = int(z["x1"] * w)
                y1 = int(z["y1"] * h)
                x2 = int(z["x2"] * w)
                y2 = int(z["y2"] * h)
                ai_canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2, tags="zone")
                ai_canvas.create_text(x1 + 6, y1 + 6, text=label, fill=color, anchor="nw",
                                      font=("Segoe UI", 9, "bold"), tags="zone")

    _zone_draw_fn[0] = draw_all_zones
    ai_canvas.after(200, draw_all_zones)

    def _on_zone_press(event):
        if not _zone_state["active"]:
            return
        _zone_state["sx"] = event.x
        _zone_state["sy"] = event.y
        if _zone_state["rect_id"]:
            ai_canvas.delete(_zone_state["rect_id"])
            _zone_state["rect_id"] = None

    def _on_zone_drag(event):
        if not _zone_state["active"] or _zone_state["sx"] is None:
            return
        if _zone_state["rect_id"]:
            ai_canvas.delete(_zone_state["rect_id"])
        idx = _zone_state["op_idx"]
        if isinstance(idx, int):
            color = OPERATOR_ZONE_COLORS[idx % len(OPERATOR_ZONE_COLORS)]
        else:
            color = {"verde": "#22c55e", "amarilla": "#f59e0b", "roja": "#ef4444"}.get(idx, "#3b82f6")
        _zone_state["rect_id"] = ai_canvas.create_rectangle(
            _zone_state["sx"], _zone_state["sy"], event.x, event.y,
            outline=color, width=2, dash=(6, 3))

    def _on_zone_release(event):
        if not _zone_state["active"] or _zone_state["sx"] is None:
            return
        idx = _zone_state["op_idx"]
        cam_id = str(camera_name_to_id[selected_ai_camera.get()])
        w = max(ai_canvas.winfo_width(), 1)
        h = max(ai_canvas.winfo_height(), 1)
        x1 = min(_zone_state["sx"], event.x) / w
        y1 = min(_zone_state["sy"], event.y) / h
        x2 = max(_zone_state["sx"], event.x) / w
        y2 = max(_zone_state["sy"], event.y) / h
        if cam_id not in ai_config:
            ai_config[cam_id] = {}
        config_key = f"op_{idx}_zone" if isinstance(idx, int) else f"smart_{idx}_zone"
        ai_config[cam_id][config_key] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        save_ai_config()
        if _zone_state["rect_id"]:
            ai_canvas.delete(_zone_state["rect_id"])
        _zone_state.update({"active": False, "sx": None, "sy": None, "rect_id": None})
        draw_all_zones()
        if idx in _zone_buttons:
            _zone_buttons[idx].config(text="✏ Redibujar zona")

    ai_canvas.bind("<ButtonPress-1>", _on_zone_press)
    ai_canvas.bind("<B1-Motion>", _on_zone_drag)
    ai_canvas.bind("<ButtonRelease-1>", _on_zone_release)

    def create_zone_button(parent, op_idx, camera_id):
        row = tk.Frame(parent, bg="#111827")
        row.pack(fill="x", padx=10, pady=6)
        z = ai_config.get(camera_id, {}).get(f"op_{op_idx}_zone")
        has_zone = z is not None

        def activate_draw():
            _zone_state["active"] = True
            _zone_state["op_idx"] = op_idx
            _zone_state["sx"] = None
            _zone_state["sy"] = None
            color = OPERATOR_ZONE_COLORS[op_idx % len(OPERATOR_ZONE_COLORS)]
            show_notification("MODO DIBUJO",
                f"Dibuja la zona del Operador {op_idx+1} sobre el video.", color)

        btn_text = "✏ Redibujar zona" if has_zone else "Seleccionar zona"
        btn = tk.Button(row, text=btn_text,
                        bg="#2563eb",
                        fg="white", relief="flat",
                        font=("Segoe UI", 9, "bold"), padx=10, pady=6,
                        cursor="hand2", command=activate_draw)
        btn.pack(side="left")

        _zone_buttons[op_idx] = btn

    def make_collapsible(parent, title, color="#3b82f6", indent=0, start_open=False):
        card = tk.Frame(parent, bg="#111827",
                        highlightbackground="#1e293b", highlightthickness=1)
        card.pack(fill="x", padx=10 + indent, pady=4)
        header = tk.Frame(card, bg="#111827", cursor="hand2")
        header.pack(fill="x")
        expanded = tk.BooleanVar(value=start_open)
        icon = tk.Label(header, text="▼" if start_open else "▶",
                        fg=color, bg="#111827", font=("Segoe UI", 10, "bold"))
        icon.pack(side="left", padx=(12, 5), pady=10)
        tk.Label(header, text=title, fg=color, bg="#111827",
                 font=("Segoe UI", 11, "bold")).pack(side="left", pady=10)
        content = tk.Frame(card, bg="#111827")
        if start_open:
            content.pack(fill="x", pady=(0, 8))
        def toggle(e=None):
            if expanded.get():
                content.pack_forget()
                icon.config(text="▶")
                expanded.set(False)
            else:
                content.pack(fill="x", pady=(0, 8))
                icon.config(text="▼")
                expanded.set(True)
        header.bind("<Button-1>", toggle)
        for w in header.winfo_children():
            w.bind("<Button-1>", toggle)
        return content

    def build_operadores_section(parent, camera_id):
        ops_container = tk.Frame(parent, bg="#111827")
        ops_container.pack(fill="x", padx=5, pady=4)

        # ── Gestión automática por turnos ─────────────────────────────────────
        shift_hdr = tk.Frame(ops_container, bg="#1a2744", highlightthickness=1,
                             highlightbackground="#2d4a7a")
        shift_hdr.pack(fill="x", padx=5, pady=(4, 6))
        tk.Label(shift_hdr, text="Gestión automática por turnos",
                 fg="#93c5fd", bg="#1a2744",
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=10, pady=6)
        _sa_val = ai_config.get(camera_id, {}).get("shift_auto", False)
        _sa_var = tk.BooleanVar(value=_sa_val)
        def _sa_changed():
            if camera_id not in ai_config:
                ai_config[camera_id] = {}
            ai_config[camera_id]["shift_auto"] = _sa_var.get()
            save_ai_config()
            _lbl_sa.config(text=("Activada" if _sa_var.get() else "Desactivada"),
                           fg=("#22c55e" if _sa_var.get() else "#6b7280"))
        _lbl_sa = tk.Label(shift_hdr,
                           text=("Activada" if _sa_val else "Desactivada"),
                           fg=("#22c55e" if _sa_val else "#6b7280"),
                           bg="#1a2744", font=("Segoe UI", 9))
        _lbl_sa.pack(side="right", padx=(0, 6))
        tk.Checkbutton(shift_hdr, variable=_sa_var, command=_sa_changed,
                       bg="#1a2744", activebackground="#1a2744",
                       fg="#93c5fd", selectcolor="#1e3a5f").pack(side="right", padx=4)

        def refresh_operators():
            for w in ops_container.winfo_children():
                if w is not shift_hdr:
                    w.destroy()
            render_operators()

        def render_operators():
            ops = ai_config.get(camera_id, {}).get("operators", [])
            _TURNO_OPTS = [
                ("", "Sin turno fijo"),
                ("mañana", "☀ Mañana (06-14)"),
                ("tarde",  "🌤 Tarde (14-22)"),
                ("noche",  "🌙 Noche (22-06)"),
            ]
            for idx, op in enumerate(ops):
                op_frame = make_collapsible(ops_container,
                    f"Operador {idx + 1}", color="#f59e0b")
                create_input_field(op_frame, "Nombre operador", f"op_{idx}_nombre", camera_id)
                create_zone_button(op_frame, idx, camera_id)
                create_time_ampm_field(op_frame, "Hora llegada", f"op_{idx}_llegada", camera_id)
                create_time_ampm_field(op_frame, "Hora salida", f"op_{idx}_salida", camera_id)
                create_input_field(op_frame, "Tiempo máx. ausencia (min)", f"op_{idx}_ausencia", camera_id)

                # ── Turno assignment ──────────────────────────────────────────
                turno_row = tk.Frame(op_frame, bg="#111827")
                turno_row.pack(fill="x", padx=10, pady=4)
                tk.Label(turno_row, text="Turno", fg="#e5e7eb", bg="#111827",
                         font=("Segoe UI", 10)).pack(side="left", padx=10, pady=6)
                saved_turno = ai_config.get(camera_id, {}).get(f"op_{idx}_turno", "")
                turno_labels = [lbl for _, lbl in _TURNO_OPTS]
                turno_values = [val for val, _ in _TURNO_OPTS]
                turno_init = turno_labels[turno_values.index(saved_turno)] if saved_turno in turno_values else turno_labels[0]
                _tv = tk.StringVar(value=turno_init)
                def _turno_changed(event=None, i=idx, tv=_tv, vals=turno_values, lbls=turno_labels):
                    chosen = tv.get()
                    val = vals[lbls.index(chosen)] if chosen in lbls else ""
                    if camera_id not in ai_config:
                        ai_config[camera_id] = {}
                    ai_config[camera_id][f"op_{i}_turno"] = val
                    save_ai_config()
                om = tk.OptionMenu(turno_row, _tv, *turno_labels, command=_turno_changed)
                om.config(bg="#1f2937", fg="white", relief="flat",
                          font=("Segoe UI", 9), activebackground="#2d4a7a",
                          highlightthickness=0)
                om["menu"].config(bg="#1f2937", fg="white", activebackground="#2563eb")
                om.pack(side="right", padx=10)

                del_row = tk.Frame(op_frame, bg="#111827")
                del_row.pack(fill="x", padx=10, pady=(0, 10))
                def save_op(i=idx):
                    if camera_id not in ai_config:
                        ai_config[camera_id] = {}
                    save_ai_config()
                    show_notification(
                        "OPERADOR GUARDADO",
                        f"Operador {i + 1} guardado correctamente.",
                        "#22c55e"
                    )
                def delete_op(i=idx):
                    ops_list = ai_config.get(camera_id, {}).get("operators", [])
                    if i < len(ops_list):
                        ops_list.pop(i)
                        if camera_id not in ai_config:
                            ai_config[camera_id] = {}
                        for suffix in ("_nombre", "_llegada", "_salida", "_ausencia", "_zone", "_turno"):
                            ai_config[camera_id].pop(f"op_{i}{suffix}", None)
                        for j in range(i, len(ops_list)):
                            for suffix in ("_nombre", "_llegada", "_salida", "_ausencia", "_zone", "_turno"):
                                old_key = f"op_{j + 1}{suffix}"
                                new_key = f"op_{j}{suffix}"
                                if old_key in ai_config[camera_id]:
                                    ai_config[camera_id][new_key] = ai_config[camera_id].pop(old_key)
                        ai_config[camera_id]["operators"] = ops_list
                        save_ai_config()
                        draw_all_zones()
                        refresh_operators()
                tk.Button(del_row, text="💾 Guardar",
                          bg="#16a34a", fg="white", relief="flat",
                          font=("Segoe UI", 9), padx=10, pady=4,
                          command=save_op).pack(side="left", padx=(10, 5))
                tk.Button(del_row, text="🗑 Eliminar operador",
                          bg="#7f1d1d", fg="white", relief="flat",
                          font=("Segoe UI", 9), padx=10, pady=4,
                          command=delete_op).pack(side="left", padx=3)
            add_row = tk.Frame(ops_container, bg="#111827")
            add_row.pack(fill="x", padx=10, pady=8)
            def add_operator():
                if camera_id not in ai_config:
                    ai_config[camera_id] = {}
                if "operators" not in ai_config[camera_id]:
                    ai_config[camera_id]["operators"] = []
                ai_config[camera_id]["operators"].append({})
                save_ai_config()
                refresh_operators()
            tk.Button(add_row, text="+ Agregar Operador",
                      bg="#1e3a5f", fg="#60a5fa", relief="flat",
                      font=("Segoe UI", 10, "bold"), padx=12, pady=7,
                      cursor="hand2", command=add_operator).pack(anchor="w")

        render_operators()

    def create_smart_zone_button(parent, zone_key, zone_color, zone_name, camera_id):
        row = tk.Frame(parent, bg="#111827")
        row.pack(fill="x", padx=10, pady=6)
        z = ai_config.get(camera_id, {}).get(f"smart_{zone_key}_zone")
        has_zone = z is not None
        draw_btn_ref = [None]

        def activate_draw():
            if camera_id in ai_config:
                ai_config[camera_id].pop(f"smart_{zone_key}_zone", None)
            _zone_state["active"] = True
            _zone_state["op_idx"] = zone_key
            _zone_state["sx"] = None
            _zone_state["sy"] = None
            show_notification("MODO DIBUJO",
                f"Dibuja la {zone_name} sobre el video.", zone_color)

        def delete_zone():
            if camera_id not in ai_config:
                ai_config[camera_id] = {}
            ai_config[camera_id].pop(f"smart_{zone_key}_zone", None)
            save_ai_config()
            draw_all_zones()
            if draw_btn_ref[0]:
                draw_btn_ref[0].config(text=f"Seleccionar {zone_name}")

        btn_text = "✏ Redibujar zona" if has_zone else f"Seleccionar {zone_name}"
        btn = tk.Button(row, text=btn_text,
                        bg=zone_color,
                        fg="white", relief="flat",
                        font=("Segoe UI", 9, "bold"), padx=10, pady=6,
                        cursor="hand2", command=activate_draw)
        btn.pack(side="left")
        draw_btn_ref[0] = btn
        _zone_buttons[zone_key] = btn

        tk.Button(row, text="🗑 Borrar zona",
                  bg="#7f1d1d", fg="white", relief="flat",
                  font=("Segoe UI", 9), padx=8, pady=6,
                  cursor="hand2", command=delete_zone).pack(side="left", padx=(8, 0))

    def build_zonas_section(parent, camera_id):
        SMART_ZONES = [
            ("verde",    "#22c55e", "Zona Verde"),
            ("amarilla", "#f59e0b", "Zona Amarilla"),
            ("roja",     "#ef4444", "Zona Roja"),
        ]
        for zone_key, zone_color, zone_name in SMART_ZONES:
            zone_content = make_collapsible(parent, zone_name, color=zone_color)
            create_smart_zone_button(zone_content, zone_key, zone_color, zone_name, camera_id)
            if zone_key == "amarilla":
                create_input_field(zone_content, "Tiempo máx. permitido (min)",
                                   "smart_amarilla_max_time", camera_id)

    # ── rebuild_camera_config ──────────────────────────────────────────────
    # Called once initially and again every time the camera dropdown changes.
    # This ensures toggles always reflect the selected camera's config.
    def rebuild_camera_config(*_):
        for widget in scroll_frame.winfo_children():
            widget.destroy()
        _zone_state.update({"active": False, "op_idx": None,
                            "sx": None, "sy": None, "rect_id": None})
        _zone_buttons.clear()

        for section_name, section_data in AI_SECTIONS.items():
            color = SECTION_COLORS.get(section_name, "#3b82f6")
            section_content = make_collapsible(scroll_frame, section_name, color=color)

            if section_data == "__operadores__":
                camera_id = str(camera_name_to_id[selected_ai_camera.get()])
                build_operadores_section(section_content, camera_id)
                continue

            if section_data == "__zonas_inteligentes__":
                camera_id = str(camera_name_to_id[selected_ai_camera.get()])
                build_zonas_section(section_content, camera_id)
                continue

            if isinstance(section_data, list):
                for item in section_data:
                    create_toggle(section_content, item, item)
                continue

            for cat_name, cat_items in section_data.items():
                cat_content = make_collapsible(section_content, cat_name,
                                               color="#94a3b8", indent=5)

                if cat_name == "OBJETOS INTELIGENTES":
                    create_toggle(cat_content, "Objeto Abandonado", "Objeto Abandonado")
                    btn_row = tk.Frame(cat_content, bg="#111827")
                    btn_row.pack(fill="x", padx=10, pady=4)

                    def _do_capture_bg():
                        cam_id = str(camera_name_to_id[selected_ai_camera.get()])
                        if hasattr(ai_canvas, "imgtk") and ai_canvas.imgtk:
                            show_notification("CAPTURANDO", "Analizando escena actual...", "#22d3ee")
                            raw_bg = cv2.cvtColor(
                                np.array(ImageTk.getimage(ai_canvas.imgtk)),
                                cv2.COLOR_RGB2BGR)
                            h_bg2, w_bg2 = raw_bg.shape[:2]
                            sc_bg = 640 / max(w_bg2, h_bg2)
                            if sc_bg < 1.0:
                                raw_bg = cv2.resize(raw_bg, (int(w_bg2 * sc_bg), int(h_bg2 * sc_bg)))
                            _capture_background(cam_id, raw_bg)
                        else:
                            show_notification("SIN CÁMARA", "Inicia la cámara primero.", "#f59e0b")

                    tk.Button(btn_row, text="📷 Capturar fondo de escena",
                              bg="#1e3a5f", fg="#60a5fa", relief="flat",
                              font=("Segoe UI", 9, "bold"), padx=10, pady=6,
                              cursor="hand2", command=_do_capture_bg).pack(anchor="w")

                    def _clear_foreign():
                        cam_id = str(camera_name_to_id[selected_ai_camera.get()])
                        _foreign_objects.pop(cam_id, None)
                        _foreign_candidates.pop(cam_id, None)
                        _background_frame.pop(cam_id, None)
                        _foreign_yolo_cache.pop(cam_id, None)
                        show_notification("LIMPIADO",
                                          "Objetos abandonados y fondo eliminados.", "#22c55e")

                    tk.Button(btn_row, text="🗑 Limpiar objetos abandonados",
                              bg="#374151", fg="white", relief="flat",
                              font=("Segoe UI", 9), padx=10, pady=6,
                              cursor="hand2", command=_clear_foreign).pack(anchor="w", pady=(4, 0))

                    # Selector de tiempo de confirmación para objeto abandonado
                    tk.Label(cat_content, text="TIEMPO DE CONFIRMACIÓN (ABANDONO)",
                             fg="#94a3b8", bg="#111827",
                             font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=14, pady=(6, 2))
                    _abnd_cam_id  = str(camera_name_to_id[selected_ai_camera.get()])
                    _abnd_cur     = int(ai_config.get(_abnd_cam_id, {}).get(
                                        "abandoned_confirm_secs", int(ABANDONED_CONFIRM_SECONDS)))
                    _abnd_row     = tk.Frame(cat_content, bg="#111827")
                    _abnd_row.pack(fill="x", padx=14, pady=(0, 8))
                    def _set_abnd_time(val, _cid=_abnd_cam_id, _fr=_abnd_row):
                        if _cid not in ai_config:
                            ai_config[_cid] = {}
                        ai_config[_cid]["abandoned_confirm_secs"] = val
                        save_ai_config()
                        for _b in _fr.winfo_children():
                            _b.configure(
                                bg="#7f1d1d" if getattr(_b, "_abnd_key", None) == val
                                else "#374151")
                    for _at in [10, 20, 30]:
                        _abtn = tk.Button(_abnd_row, text=f"{_at}s",
                                          bg="#7f1d1d" if _abnd_cur == _at else "#374151",
                                          fg="white", relief="flat",
                                          font=("Segoe UI", 9, "bold"),
                                          padx=12, pady=4, cursor="hand2",
                                          command=lambda v=_at: _set_abnd_time(v))
                        _abtn._abnd_key = _at
                        _abtn.pack(side="left", padx=2, pady=2)

                    tk.Frame(cat_content, bg="#1e293b", height=1).pack(fill="x", padx=10, pady=6)

                    create_toggle(cat_content, "Objeto Movido", "Objeto Movido")
                    btn_row2 = tk.Frame(cat_content, bg="#111827")
                    btn_row2.pack(fill="x", padx=10, pady=4)

                    watched_list_frame = tk.Frame(cat_content, bg="#111827")
                    watched_list_frame.pack(fill="x", padx=10, pady=4)

                    def _refresh_watched_list(cam_id):
                        for w in watched_list_frame.winfo_children():
                            w.destroy()
                        watched = _watched_objects.get(cam_id, {})
                        if not watched:
                            tk.Label(watched_list_frame, text="Sin objetos registrados",
                                     fg="#64748b", bg="#111827",
                                     font=("Segoe UI", 9)).pack(anchor="w", padx=6, pady=4)
                            return
                        for wid, wobj in list(watched.items()):
                            row_w = tk.Frame(watched_list_frame, bg="#1e293b")
                            row_w.pack(fill="x", pady=2)
                            color_lbl = "#64748b" if wobj.get("ignored") else "#e5e7eb"
                            status = " [ignorado]" if wobj.get("ignored") else ""
                            tk.Label(row_w, text=f"#{wid} {wobj['class']}{status}",
                                     fg=color_lbl, bg="#1e293b",
                                     font=("Segoe UI", 9)).pack(side="left", padx=6, pady=4)
                            def _ignore(wid_=wid, cam_id_=cam_id):
                                if cam_id_ in _watched_objects and wid_ in _watched_objects[cam_id_]:
                                    _watched_objects[cam_id_][wid_]["ignored"] = True
                                    _refresh_watched_list(cam_id_)
                            tk.Button(row_w, text="✕", bg="#7f1d1d", fg="white",
                                      relief="flat", font=("Segoe UI", 8), padx=6, pady=2,
                                      cursor="hand2", command=_ignore).pack(side="right", padx=4)

                    def _do_scan_objects():
                        cam_id = str(camera_name_to_id[selected_ai_camera.get()])
                        if hasattr(ai_canvas, "imgtk") and ai_canvas.imgtk:
                            frame_img = cv2.cvtColor(
                                np.array(ImageTk.getimage(ai_canvas.imgtk)),
                                cv2.COLOR_RGB2BGR)
                            h_fi, w_fi = frame_img.shape[:2]
                            sc_fi = 640 / max(w_fi, h_fi)
                            if sc_fi < 1.0:
                                frame_img = cv2.resize(frame_img, (int(w_fi * sc_fi), int(h_fi * sc_fi)))
                            _register_watched_objects(frame_img, cam_id)
                            _refresh_watched_list(cam_id)
                        else:
                            show_notification("SIN CÁMARA", "Inicia la cámara primero.", "#f59e0b")

                    tk.Button(btn_row2, text="🔍 Escanear objetos en escena",
                              bg="#1e3a5f", fg="#60a5fa", relief="flat",
                              font=("Segoe UI", 9, "bold"), padx=10, pady=6,
                              cursor="hand2", command=_do_scan_objects).pack(anchor="w")

                    btn_row3 = tk.Frame(cat_content, bg="#111827")
                    btn_row3.pack(fill="x", padx=10, pady=(0, 6))

                    def _clear_watched():
                        cam_id = str(camera_name_to_id[selected_ai_camera.get()])
                        _watched_objects.pop(cam_id, None)
                        _moved_alerts.pop(cam_id, None)
                        _refresh_watched_list(cam_id)
                        show_notification("LIMPIADO", "Lista de vigilancia limpiada.", "#22c55e")

                    tk.Button(btn_row3, text="🗑 Limpiar vigilancia",
                              bg="#374151", fg="white", relief="flat",
                              font=("Segoe UI", 9), padx=10, pady=4,
                              cursor="hand2", command=_clear_watched).pack(anchor="w")
                    continue

                if cat_name == "CONDUCTA":
                    create_toggle(cat_content, "Persona Corriendo", "Persona Corriendo")
                    create_toggle(cat_content, "Persona Inmóvil", "Persona Inmóvil")
                    tk.Frame(cat_content, bg="#1e293b", height=1).pack(fill="x", padx=10, pady=6)
                    falls_content = make_collapsible(cat_content, "Detección de Caídas",
                                                     color="#ef4444", indent=5)
                    _fc_cam_id = str(camera_name_to_id[selected_ai_camera.get()])
                    create_toggle(falls_content, "Persona Caída", "Persona Caída")
                    sens_row = tk.Frame(falls_content, bg="#111827")
                    sens_row.pack(fill="x", padx=10, pady=4)
                    tk.Label(sens_row, text="Sensibilidad:", fg="#e5e7eb", bg="#111827",
                             font=("Segoe UI", 10)).pack(side="left", padx=10, pady=8)
                    _fc_sens_frame = tk.Frame(sens_row, bg="#111827")
                    _fc_sens_frame.pack(side="right", padx=10)
                    _fc_sens_cur = ai_config.get(_fc_cam_id, {}).get("fall_sensitivity", "media")
                    def _set_sens(val, _cid=_fc_cam_id, _fr=_fc_sens_frame):
                        if _cid not in ai_config:
                            ai_config[_cid] = {}
                        ai_config[_cid]["fall_sensitivity"] = val
                        save_ai_config()
                        for _b in _fr.winfo_children():
                            _b.configure(bg="#7f1d1d" if _b.cget("text").lower() == val else "#374151")
                    for _sl in ["Baja", "Media", "Alta"]:
                        tk.Button(_fc_sens_frame, text=_sl,
                                  bg="#7f1d1d" if _fc_sens_cur == _sl.lower() else "#374151",
                                  fg="white", relief="flat",
                                  font=("Segoe UI", 9, "bold"),
                                  padx=10, pady=4, cursor="hand2",
                                  command=lambda v=_sl.lower(): _set_sens(v)
                                  ).pack(side="left", padx=2)
                    create_input_field(falls_content, "Tiempo de confirmación (seg)",
                                       "fall_confirm_secs", _fc_cam_id)
                    continue

                if cat_name == "HUMANOS":
                    create_toggle(cat_content, "Persona", "Persona")
                    continue

                if isinstance(cat_items, list):
                    for item in cat_items:
                        create_toggle(cat_content, item, item)
                elif isinstance(cat_items, dict):
                    for zona_name, zona_items in cat_items.items():
                        zona_content = make_collapsible(cat_content, zona_name,
                                                        color="#64748b", indent=10)
                        for item in zona_items:
                            create_toggle(zona_content, item, item)

        draw_all_zones()
        # Auto-start camera stream for the newly selected camera
        scroll_frame.after(80, start_ai_camera)

    # Rebuild config panel whenever the camera dropdown changes
    selected_ai_camera.trace('w', rebuild_camera_config)
    # Delay initial build so canvas is rendered before draw_all_zones()
    scroll_frame.after(50, rebuild_camera_config)

    # =========================
    # EVENTOS ABAJO
    # =========================
    events_panel = tk.Frame(
        body,
        bg="#0b1220",
        highlightbackground="#1e293b",
        highlightthickness=1
    )

    events_panel.place(
        relx=0,
        rely=0.84,
        relwidth=1,
        relheight=0.16
    )

    tk.Label(
        events_panel,
        text="Eventos en Tiempo Real",
        fg="#3b82f6",
        bg="#0b1220",
        font=("Segoe UI", 12, "bold")
    ).pack(anchor="w", padx=15, pady=10)

    event_list = tk.Frame(
        events_panel,
        bg="#0b1220"
    )

    event_list.pack(
        fill="both",
        expand=True,
        padx=15,
        pady=(0,10)
    )

    demo_events = [
        ("12:44:21", "Intrusión detectada"),
        ("12:45:03", "Objeto movido"),
        ("12:45:55", "Persona sin casco")
    ]

    for hour, text in demo_events:

        row = tk.Frame(
            event_list,
            bg="#111827"
        )

        row.pack(
            fill="x",
            pady=2
        )

        tk.Label(
            row,
            text=hour,
            fg="#22c55e",
            bg="#111827",
            width=12,
            anchor="w"
        ).pack(side="left", padx=10, pady=6)

        tk.Label(
            row,
            text=text,
            fg="#e5e7eb",
            bg="#111827",
            anchor="w"
        ).pack(side="left")

# =========================
# BENCHMARK — HELPERS
# =========================

def _bm_percentile(data, pct):
    if not data:
        return None
    s = sorted(v for v in data if v is not None)
    if not s:
        return None
    idx = max(0, min(int(len(s) * pct / 100.0), len(s) - 1))
    return s[idx]


def _bm_collect_snapshot(proc, gpu_handle):
    snap = {"ts": time.time()}
    try:
        snap["cpu"]    = proc.cpu_percent(interval=None) if proc else 0.0
        mi             = proc.memory_info() if proc else None
        snap["ram_mb"] = mi.rss / 1_048_576 if mi else 0.0
    except Exception:
        snap["cpu"] = 0.0; snap["ram_mb"] = 0.0
    if gpu_handle is not None and _pynvml is not None:
        try:
            util             = _pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            mem              = _pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            snap["gpu"]      = util.gpu
            snap["vram_mb"]  = mem.used / 1_048_576
            snap["vram_total_mb"] = mem.total / 1_048_576
        except Exception:
            snap["gpu"] = None; snap["vram_mb"] = None; snap["vram_total_mb"] = None
    else:
        snap["gpu"] = None; snap["vram_mb"] = None; snap["vram_total_mb"] = None
    fps_vals = []
    for c in selected_cameras:
        tsq = _ai_fps_ts.get(str(c))
        if tsq and len(tsq) >= 2:
            span = tsq[-1] - tsq[0]
            if span > 0:
                fps_vals.append((len(tsq) - 1) / span)
    snap["fps"]        = sum(fps_vals) / len(fps_vals) if fps_vals else 0.0
    snap["throughput"] = sum(fps_vals)
    lats = list(_diag_latency_all)
    if lats:
        lats_ms = [l * 1000 for l in lats]
        snap["lat_avg_ms"] = sum(lats_ms) / len(lats_ms)
        snap["lat_p50_ms"] = _bm_percentile(lats_ms, 50)
        snap["lat_p95_ms"] = _bm_percentile(lats_ms, 95)
        snap["lat_p99_ms"] = _bm_percentile(lats_ms, 99)
    else:
        snap["lat_avg_ms"] = snap["lat_p50_ms"] = snap["lat_p95_ms"] = snap["lat_p99_ms"] = None
    snap["cams_ai"]    = sum(1 for c in selected_cameras if _ai_started.get(str(c)))
    snap["events"]     = len(_events_list)
    snap["detections"] = sum(_diag_cam_det_count.values())
    return snap


def _bm_aggregate(snaps):
    if not snaps:
        return {"n_samples": 0}
    def _s(lst):
        clean = [v for v in lst if v is not None]
        if not clean:
            return {"avg": None, "min": None, "max": None, "p95": None}
        return {"avg": sum(clean)/len(clean), "min": min(clean),
                "max": max(clean), "p95": _bm_percentile(clean, 95)}
    r = {
        "n_samples":  len(snaps),
        "duration_s": snaps[-1]["ts"] - snaps[0]["ts"] if len(snaps) > 1 else 0,
        "cpu":        _s([s["cpu"]     for s in snaps]),
        "ram":        _s([s["ram_mb"]  for s in snaps]),
        "gpu":        _s([s.get("gpu") for s in snaps]),
        "vram":       _s([s.get("vram_mb") for s in snaps]),
        "fps":        _s([s["fps"]     for s in snaps if s["fps"] > 0]),
        "throughput": _s([s["throughput"] for s in snaps if s["throughput"] > 0]),
        "lat_avg":    _s([s.get("lat_avg_ms") for s in snaps]),
        "lat_p50":    _s([s.get("lat_p50_ms") for s in snaps]),
        "lat_p95":    _s([s.get("lat_p95_ms") for s in snaps]),
        "lat_p99":    _s([s.get("lat_p99_ms") for s in snaps]),
        "cams_ai":    snaps[-1]["cams_ai"],
        "events_gen": snaps[-1]["events"] - snaps[0]["events"],
        "det_gen":    snaps[-1]["detections"] - snaps[0]["detections"],
        "_series": {
            "cpu":  [s["cpu"]    for s in snaps],
            "ram":  [s["ram_mb"] for s in snaps],
            "gpu":  [s.get("gpu") for s in snaps if s.get("gpu") is not None],
            "fps":  [s["fps"]    for s in snaps],
        },
    }
    return r


def _bm_load_history():
    try:
        with open("benchmark_history.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _bm_save_to_history(record):
    hist = _bm_load_history()
    hist.insert(0, record)
    try:
        with open("benchmark_history.json", "w", encoding="utf-8") as f:
            json.dump(hist[:50], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _bm_capacity_estimate(results_dict):
    cams = 0; cpu_max = None; gpu_max = None; fps_avg = None
    for sc_name in ("pico", "sostenido", "base", "rapido"):
        sc = results_dict.get(sc_name)
        if sc and sc.get("cpu") and sc["cpu"].get("avg") is not None:
            cams    = sc.get("cams_ai", cams)
            cpu_max = sc["cpu"]["max"]
            gpu_max = sc["gpu"]["max"] if sc.get("gpu") and sc["gpu"].get("max") is not None else None
            fps_avg = sc["fps"]["avg"] if sc.get("fps") and sc["fps"].get("avg") is not None else None
            break
    if cams == 0 or fps_avg is None:
        return {"status": "insufficient_data"}
    cpu_per_cam = cpu_max / max(cams, 1)
    max_cpu  = int(80.0 / cpu_per_cam) if cpu_per_cam > 0 else 99
    max_gpu  = 99
    if gpu_max and gpu_max > 0:
        gpu_per_cam = gpu_max / max(cams, 1)
        max_gpu = int(80.0 / gpu_per_cam) if gpu_per_cam > 0 else 99
    max_cams  = max(1, min(max_cpu, max_gpu))
    saturated = (cpu_max or 0) > 90 or (gpu_max or 0) > 90 or (fps_avg or 99) < 5
    return {
        "status":          "ok",
        "cams_tested":     cams,
        "cpu_per_cam":     round(cpu_per_cam, 1),
        "max_cams":        max_cams,
        "fps_avg":         fps_avg,
        "cpu_max":         cpu_max,
        "gpu_max":         gpu_max,
        "utilization_pct": round(max(cpu_max or 0, gpu_max or 0), 1),
        "saturated":       saturated,
        "fps_ok":          (fps_avg or 0) >= 10,
    }


def _bm_generate_pdf(result, filename):
    try:
        from fpdf import FPDF
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile, os as _os

        tmp = tempfile.gettempdir()
        charts = []

        def _chart(title, series, ylabel, path):
            COLS = {"cpu": "#3b82f6", "ram": "#a855f7", "gpu": "#f59e0b", "fps": "#22d3ee"}
            fig, ax = plt.subplots(figsize=(7, 2.5), facecolor="white")
            for k, data in series.items():
                if data:
                    ax.plot(range(len(data)), data, color=COLS.get(k, "#333"),
                            linewidth=1.5, label=k.upper())
            ax.set_title(title, fontsize=9); ax.set_xlabel("Muestras (s)", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7); ax.tick_params(labelsize=7)
            ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
            fig.tight_layout(); fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)

        for sc_key in ("base", "sostenido", "pico", "adverso", "rapido"):
            sc = result.get("scenarios", {}).get(sc_key)
            if not sc or not sc.get("_series"):
                continue
            cpath = _os.path.join(tmp, f"bm_{sc_key}.png")
            _chart(f"Escenario {sc_key.title()}", sc["_series"], "%  /  FPS", cpath)
            charts.append((sc_key, cpath))

        class _PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 8)
                self.set_text_color(120, 120, 120)
                self.cell(0, 5, "VIGILANT PRO  —  BENCHMARK OFICIAL", align="R")
                self.ln(2)
            def footer(self):
                self.set_y(-12); self.set_font("Helvetica", "I", 8)
                self.set_text_color(150, 150, 150)
                self.cell(0, 5, f"Pág. {self.page_no()}", align="C")

        pdf = _PDF()
        pdf.set_auto_page_break(auto=True, margin=14)
        pdf.add_page()

        def _T(t, sz=13, rgb=(15,23,42)):
            pdf.set_font("Helvetica","B",sz); pdf.set_text_color(*rgb)
            pdf.cell(0, 8, t, ln=True)
            pdf.set_draw_color(180,180,180); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(3)

        def _R(lbl, val, ok=None):
            pdf.set_font("Helvetica","",9); pdf.set_text_color(80,80,80); pdf.cell(72,5.5,lbl,ln=False)
            col = (22,163,74) if ok is True else (220,38,38) if ok is False else (20,20,20)
            pdf.set_font("Helvetica","B",9); pdf.set_text_color(*col); pdf.cell(0,5.5,str(val),ln=True)

        def _M(lbl, sd, unit="", ok_fn=None):
            if not sd or sd.get("avg") is None:
                _R(lbl, "NO DISPONIBLE"); return
            v = f"{sd['avg']:.1f}{unit}  (máx: {sd['max']:.1f}{unit})"
            _R(lbl, v, ok_fn(sd["avg"]) if ok_fn else None)

        # Portada
        pdf.ln(4)
        pdf.set_font("Helvetica","B",20); pdf.set_text_color(15,23,42)
        pdf.cell(0,10,"VIGILANT PRO", ln=True, align="C")
        pdf.set_font("Helvetica","B",13); pdf.set_text_color(30,64,175)
        pdf.cell(0,7,"Reporte de Benchmark Oficial", ln=True, align="C")
        pdf.set_font("Helvetica","",9); pdf.set_text_color(100,100,100)
        pdf.cell(0,5,f"Generado: {result.get('timestamp','')}  |  Usuario: {result.get('user','')}",
                 ln=True, align="C"); pdf.ln(6)
        pdf.set_draw_color(200,200,200); pdf.line(10, pdf.get_y(), 200, pdf.get_y()); pdf.ln(4)

        # Hardware
        _T("1. Hardware del Sistema")
        hw = result.get("hardware", {})
        _R("CPU:", hw.get("cpu_name","NO DISPONIBLE"))
        _R("Núcleos (físicos / lógicos):", f"{hw.get('cpu_cores_phys','?')} / {hw.get('cpu_cores_logi','?')}")
        _R("RAM total:", f"{hw.get('ram_total_gb',0):.1f} GB")
        _R("GPU:", hw.get("gpu_name","NO DISPONIBLE"))
        if hw.get("vram_total_mb"):
            _R("VRAM:", f"{hw.get('vram_total_mb',0)/1024:.2f} GB")
        _R("Sistema operativo:", hw.get("os_short",""))
        _R("Almacenamiento libre:", f"{hw.get('disk_free_gb',0):.1f} GB  ({100-hw.get('disk_used_pct',0):.0f}% libre)")
        pdf.ln(3)

        # Metodología
        _T("2. Metodología")
        for k,v in [("CPU","psutil.Process().cpu_percent()"),("RAM","psutil.Process().memory_info().rss"),
                    ("GPU","pynvml nvmlDeviceGetUtilizationRates"),("FPS","Pipeline interno (_ai_fps_ts)"),
                    ("Throughput","Suma FPS todas las cámaras"),("Latencia IA","Timestamps internos process_frame()")]:
            _R(k+":", v)
        pdf.ln(3)

        # Config
        _T("3. Configuración Detectada")
        cfg = result.get("config", {})
        _R("Cámaras configuradas:", str(cfg.get("cam_count",0)))
        _R("Cámaras con IA activa:", str(cfg.get("cams_active_ai",0)))
        for k,lbl in [("persona","Personas"),("bytetrack","ByteTrack"),("epp","EPP"),
                       ("armas","Armas"),("zonas","Zonas"),("operadores","Operadores"),
                       ("objeto_movido","Objetos Movidos"),("caidas","Caídas")]:
            v = cfg.get(k)
            _R(lbl+":", "Configurado" if v else "No configurado", ok=bool(v))
        pdf.ln(3)

        # Scenarios
        SC_META = {
            "base":     ("4. Escenario Base",       "1 cámara — 5 min — desempeño nominal"),
            "sostenido":("5. Escenario Sostenido",   "2 cámaras — 5 min — estabilidad"),
            "pico":     ("6. Escenario Pico",        "Todas las cámaras — 10 min — carga máxima"),
            "adverso":  ("7. Escenario Adverso",     "Frames degradados — condiciones difíciles"),
            "rapido":   ("4. Benchmark Rápido",      "Monitoreo 5 minutos — configuración actual"),
        }
        for sc_key, (sc_title, sc_desc) in SC_META.items():
            sc = result.get("scenarios", {}).get(sc_key)
            if not sc:
                continue
            pdf.add_page(); _T(sc_title)
            pdf.set_font("Helvetica","I",8); pdf.set_text_color(100,100,100)
            pdf.cell(0,5,sc_desc,ln=True); pdf.ln(1)
            _R("Duración:", f"{sc.get('duration_s',0):.0f} s")
            _R("Muestras:", str(sc.get("n_samples",0)))
            _R("Cámaras IA:", str(sc.get("cams_ai",0)))
            pdf.ln(2)
            pdf.set_font("Helvetica","B",10); pdf.set_text_color(30,64,175); pdf.cell(0,6,"Recursos",ln=True)
            _M("CPU promedio:", sc.get("cpu"),    "%",   lambda v: v < 80)
            _M("RAM promedio:", sc.get("ram"),    " MB")
            if sc.get("gpu") and sc["gpu"].get("avg") is not None:
                _M("GPU promedio:", sc.get("gpu"),   "%",   lambda v: v < 85)
                _M("VRAM promedio:", sc.get("vram"), " MB")
            else:
                _R("GPU:", "NO DISPONIBLE")
            pdf.ln(2)
            pdf.set_font("Helvetica","B",10); pdf.set_text_color(30,64,175); pdf.cell(0,6,"Rendimiento IA",ln=True)
            _M("FPS promedio:",          sc.get("fps"),       " fps", lambda v: v >= 10)
            _M("Throughput:",            sc.get("throughput")," fps total")
            _M("Latencia promedio:",     sc.get("lat_avg"),   " ms",  lambda v: v < 500)
            _M("Latencia p50:",          sc.get("lat_p50"),   " ms")
            _M("Latencia p95:",          sc.get("lat_p95"),   " ms")
            _M("Latencia p99:",          sc.get("lat_p99"),   " ms")
            pdf.ln(2)
            _R("Detecciones generadas:", str(sc.get("det_gen",0)))
            _R("Eventos generados:",     str(sc.get("events_gen",0)))
            for (ck, cpath) in charts:
                if ck == sc_key and _os.path.exists(cpath):
                    pdf.ln(3); pdf.image(cpath, x=14, w=172); break

        # Capacidad
        pdf.add_page(); _T("8. Estimación de Capacidad", rgb=(22,101,52))
        cap = result.get("capacity", {})
        if cap.get("status") == "ok":
            _R("Cámaras probadas:",     str(cap.get("cams_tested",0)))
            _R("CPU por cámara (máx):", f"{cap.get('cpu_per_cam',0):.1f}%")
            _R("Cámaras recomendadas:", str(cap.get("max_cams",0)), ok=True)
            _R("Utilización máxima:",   f"{cap.get('utilization_pct',0):.0f}%")
            _R("FPS saludable:",        "Sí" if cap.get("fps_ok") else "No", ok=cap.get("fps_ok"))
            sat = cap.get("saturated",False)
            _R("Estado sistema:",       "SATURADO" if sat else "NORMAL", ok=not sat)
        else:
            _R("Estado:", "Datos insuficientes")
        pdf.ln(4)

        # Conclusiones
        _T("9. Conclusiones")
        scs = result.get("scenarios", {})
        passed = bool(scs) and all(
            (sc.get("cpu",{}).get("max") or 0) < 95 and (sc.get("fps",{}).get("avg") or 0) >= 5
            for sc in scs.values() if sc)
        _R("Resultado general:", "APROBADO" if passed else "REQUIERE REVISIÓN", ok=passed)
        if cap.get("status") == "ok":
            _R("Capacidad recomendada:", f"Hasta {cap.get('max_cams',0)} cámaras Full HD con IA completa")
            _R("CPU máxima observada:",  f"{cap.get('cpu_max',0):.0f}%")
            if cap.get("gpu_max") is not None:
                _R("GPU máxima observada:", f"{cap.get('gpu_max',0):.0f}%")
            _R("FPS promedio pico:",     f"{cap.get('fps_avg',0):.1f}")

        pdf.output(filename)
        for (_, cp) in charts:
            try: _os.remove(cp)
            except: pass
        return True, filename
    except Exception as e:
        return False, str(e)


# =========================
# DIAGNÓSTICO Y RENDIMIENTO
# =========================
def show_diagnostics():
    import psutil as _psutil
    import platform as _platform
    import threading as _threading

    # ── Access control ────────────────────────────────────────────────────────
    user_role = next(
        (u.get("role", "") for u in users_data if u["user"] == current_user), ""
    )
    is_admin = current_user in ("admin", "ROOT_USB") or user_role == "Administrador"

    clear_main()

    # Cancel previous timers
    for _af in (_diag_panel_after, _benchmark_panel_after):
        if _af[0]:
            try: main.after_cancel(_af[0])
            except Exception: pass
            _af[0] = None

    # Auto-recovery: si el flag quedó True (ej. el usuario navegó fuera a mitad del benchmark),
    # lo reseteamos para que se pueda iniciar un nuevo benchmark sin bloqueos.
    if _benchmark_running[0]:
        _benchmark_running[0] = False
        _benchmark_cancel_flag[0] = True
        _benchmark_thread[0] = None

    # ── Access denied ─────────────────────────────────────────────────────────
    if not is_admin:
        _denied = tk.Frame(main, bg="#020617")
        _denied.pack(fill="both", expand=True)
        tk.Label(_denied, text="⛔", fg="#ef4444", bg="#020617",
                 font=("Segoe UI", 48)).pack(pady=(80, 8))
        tk.Label(_denied, text="Acceso restringido",
                 fg="#ef4444", bg="#020617",
                 font=("Segoe UI", 18, "bold")).pack()
        tk.Label(_denied,
                 text="Esta herramienta está disponible\núnicamente para administradores.",
                 fg="#94a3b8", bg="#020617",
                 font=("Segoe UI", 12), justify="center").pack(pady=(8, 0))
        return

    # ── Process + GPU handle ──────────────────────────────────────────────────
    try:
        _proc = _psutil.Process(os.getpid())
        _proc.cpu_percent(interval=None)
    except Exception:
        _proc = None

    _gpu_handle = None
    _gpu_name   = "NO DISPONIBLE"
    if _PYNVML_OK and _pynvml is not None:
        try:
            _gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
            _gpu_name   = _pynvml.nvmlDeviceGetName(_gpu_handle)
            if isinstance(_gpu_name, bytes):
                _gpu_name = _gpu_name.decode()
        except Exception:
            _gpu_handle = None

    # ── Scrollable layout ─────────────────────────────────────────────────────
    outer = tk.Frame(main, bg="#020617")
    outer.pack(fill="both", expand=True)
    canvas = tk.Canvas(outer, bg="#020617", highlightthickness=0)
    vsb    = tk.Scrollbar(outer, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    container = tk.Frame(canvas, bg="#020617")
    cwin      = canvas.create_window((0, 0), window=container, anchor="nw")
    container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>",    lambda e: canvas.itemconfig(cwin, width=e.width))
    canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)),"units"))
    padded = tk.Frame(container, bg="#020617")
    padded.pack(fill="both", expand=True, padx=30, pady=25)

    def _sec(parent, text, color="#3b82f6"):
        f = tk.Frame(parent, bg="#020617"); f.pack(fill="x", pady=(18, 6))
        tk.Label(f, text=text, fg=color, bg="#020617",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")
        tk.Frame(f, bg=color, height=2).pack(fill="x", pady=(4, 0))

    def _card(parent):
        c = tk.Frame(parent, bg="#0b1220",
                     highlightbackground="#1e293b", highlightthickness=1)
        c.pack(fill="x", pady=(0, 6)); return c

    def _drow(card, label, tvar, color="#22c55e", w=30):
        r = tk.Frame(card, bg="#0b1220"); r.pack(fill="x", padx=14, pady=3)
        tk.Label(r, text=label, fg="#64748b", bg="#0b1220",
                 font=("Segoe UI", 10), width=w, anchor="w").pack(side="left")
        tk.Label(r, textvariable=tvar, fg=color, bg="#0b1220",
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(side="left")

    def _srow(card, label, value, color="#94a3b8", w=30):
        r = tk.Frame(card, bg="#0b1220"); r.pack(fill="x", padx=14, pady=3)
        tk.Label(r, text=label, fg="#64748b", bg="#0b1220",
                 font=("Segoe UI", 10), width=w, anchor="w").pack(side="left")
        tk.Label(r, text=str(value), fg=color, bg="#0b1220",
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(side="left")

    def _bm_btn(parent, text, cmd, bg="#1e293b", fg="#e2e8f0"):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=fg, relief="flat",
                         font=("Segoe UI", 10, "bold"),
                         padx=18, pady=9, cursor="hand2",
                         activebackground="#334155", activeforeground="white")

    _hw = {
        "cpu_name":       _platform.processor() or _platform.machine() or "N/D",
        "cpu_cores_phys": _psutil.cpu_count(logical=False) or 1,
        "cpu_cores_logi": _psutil.cpu_count(logical=True)  or 1,
        "os_short":       "{} {}".format(_platform.system(), _platform.release()),
        "os":             _platform.version(),
    }
    _vm = _psutil.virtual_memory()
    _hw["ram_total_gb"] = _vm.total / (1024**3)
    try:
        _dsk = _psutil.disk_usage(os.getcwd())
        _hw["disk_total_gb"] = _dsk.total / (1024**3)
        _hw["disk_free_gb"]  = _dsk.free  / (1024**3)
        _hw["disk_used_pct"] = _dsk.percent
    except Exception:
        _hw["disk_total_gb"] = _hw["disk_free_gb"] = 0; _hw["disk_used_pct"] = 0
    if _gpu_handle is not None and _pynvml:
        _hw["gpu_name"] = _gpu_name
        try:
            _ghw_mem = _pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
            _hw["vram_total_mb"] = _ghw_mem.total / 1_048_576
        except Exception:
            _hw["vram_total_mb"] = 0
    else:
        _hw["gpu_name"] = "NO DISPONIBLE"; _hw["vram_total_mb"] = 0

    def _get_cfg_summary():
        def _any(keys):
            return any(ai_config.get(str(c), {}).get(k) for c in selected_cameras for k in keys)
        _az = any(_has_smart_zones(ai_config.get(str(c), {})) for c in selected_cameras)
        _ao = any(ai_config.get(str(c), {}).get("operators") for c in selected_cameras)
        return {
            "cam_count":      len(selected_cameras),
            "cams_active_ai": sum(1 for c in selected_cameras if _ai_started.get(str(c))),
            "persona":        _any(["Persona","Mochila","Caja"]),
            "bytetrack":      _any(["Persona Corriendo","Persona Inmovil","Objeto Abandonado"]) or _az,
            "epp":            _any(["Casco Seguridad","Bata Industrial","Botas Seguridad",
                                    "Cubrebocas Seguridad","Lentes Seguridad","No Celular"]),
            "armas":          _any(["Arma Blanca","Arma Corta","Arma Larga"]),
            "zonas":          _az,
            "operadores":     _ao,
            "objeto_movido":  _any(["Objeto Movido"]),
            "caidas":         _any(["Persona Caída", "Caídas"]),
        }
    _cfg_summary = _get_cfg_summary()

    tk.Label(padded, text="CENTRO DE DIAGNOSTICO Y BENCHMARK OFICIAL",
             fg="#e5e7eb", bg="#020617",
             font=("Segoe UI", 20, "bold")).pack(anchor="w")
    tk.Label(padded, text="Vigilant Pro  -  Herramienta exclusiva para administracion avanzada",
             fg="#64748b", bg="#020617", font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 4))
    tk.Label(padded, text="Usuario: {}  |  PID: {}".format(current_user, os.getpid()),
             fg="#475569", bg="#020617", font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 18))

    _sec(padded, "RESUMEN DEL SISTEMA", "#22c55e")
    _card_sys = _card(padded)
    tk.Frame(_card_sys, bg="#0b1220", height=6).pack()
    _vv = {k: tk.StringVar(value="NO DISPONIBLE") for k in
           ("uptime","cpu","ram","gpu_use","gpu_vram","gpu_temp",
            "fps","lat_avg","lat_p95","cams_ai","events","det","snaps","vids")}
    r_st = tk.Frame(_card_sys, bg="#0b1220"); r_st.pack(fill="x", padx=14, pady=3)
    tk.Label(r_st, text="Estado:", fg="#64748b", bg="#0b1220",
             font=("Segoe UI", 10), width=30, anchor="w").pack(side="left")
    tk.Label(r_st, text="Operativo", fg="#22c55e", bg="#0b1220",
             font=("Segoe UI", 10, "bold")).pack(side="left")
    _drow(_card_sys, "Tiempo activo:", _vv["uptime"], "#22c55e")
    tk.Frame(_card_sys, bg="#132030", height=1).pack(fill="x", padx=14, pady=4)
    _srow(_card_sys, "CPU:", _hw["cpu_name"][:55], "#e5e7eb")
    _srow(_card_sys, "Nucleos (fisicos/logicos):",
          "{} / {}".format(_hw["cpu_cores_phys"], _hw["cpu_cores_logi"]), "#e5e7eb")
    _drow(_card_sys, "CPU uso (Vigilant Pro):", _vv["cpu"], "#3b82f6")
    _srow(_card_sys, "RAM total:", "{:.1f} GB".format(_hw["ram_total_gb"]), "#e5e7eb")
    _drow(_card_sys, "RAM uso (Vigilant Pro):", _vv["ram"], "#a855f7")
    _srow(_card_sys, "GPU:", _hw["gpu_name"][:55], "#f59e0b")
    _drow(_card_sys, "GPU uso:", _vv["gpu_use"], "#f59e0b")
    _drow(_card_sys, "VRAM:", _vv["gpu_vram"], "#f59e0b")
    _drow(_card_sys, "Temperatura GPU:", _vv["gpu_temp"], "#f97316")
    tk.Frame(_card_sys, bg="#132030", height=1).pack(fill="x", padx=14, pady=4)
    _srow(_card_sys, "Sistema operativo:", _hw["os_short"], "#e5e7eb")
    _srow(_card_sys, "Disco libre:",
          "{:.1f} GB libres de {:.1f} GB".format(_hw["disk_free_gb"], _hw["disk_total_gb"]),
          "#e5e7eb")
    _drow(_card_sys, "FPS IA REAL:", _vv["fps"], "#22d3ee")
    _drow(_card_sys, "Latencia IA promedio:", _vv["lat_avg"], "#3b82f6")
    _drow(_card_sys, "Latencia IA p95:", _vv["lat_p95"], "#f59e0b")
    _drow(_card_sys, "Camaras con IA activa:", _vv["cams_ai"], "#22c55e")
    _drow(_card_sys, "Eventos (sesion):", _vv["events"], "#f97316")
    _drow(_card_sys, "Detecciones (sesion):", _vv["det"], "#a855f7")
    _drow(_card_sys, "Snapshots generados:", _vv["snaps"], "#94a3b8")
    _drow(_card_sys, "Videos grabados:", _vv["vids"], "#94a3b8")
    tk.Frame(_card_sys, bg="#0b1220", height=6).pack()

    _sec(padded, "CONFIGURACION DETECTADA", "#3b82f6")
    _card_cfg = _card(padded)
    tk.Frame(_card_cfg, bg="#0b1220", height=6).pack()
    _srow(_card_cfg, "Camaras configuradas:", str(_cfg_summary["cam_count"]), "#22d3ee")
    _srow(_card_cfg, "Camaras con IA activa:", str(_cfg_summary["cams_active_ai"]), "#22c55e")
    tk.Frame(_card_cfg, bg="#132030", height=1).pack(fill="x", padx=14, pady=4)
    for _k, _lbl in [("persona","Personas/Deteccion general"),("bytetrack","ByteTrack (tracking)"),
                      ("epp","EPP/Seguridad industrial"),("armas","Armas"),
                      ("zonas","Zonas inteligentes"),("operadores","Operadores"),
                      ("objeto_movido","Objetos Movidos"),("caidas","Caidas")]:
        _v = _cfg_summary[_k]
        _srow(_card_cfg, "{}:".format(_lbl), "Configurado" if _v else "No configurado",
              "#22c55e" if _v else "#475569")
    tk.Frame(_card_cfg, bg="#0b1220", height=6).pack()

    _sec(padded, "ESCENARIOS DISPONIBLES", "#a855f7")
    _card_sc = _card(padded)
    tk.Frame(_card_sc, bg="#0b1220", height=6).pack()
    _badge_f = tk.Frame(_card_sc, bg="#0b1220"); _badge_f.pack(fill="x", padx=14, pady=(0,8))
    for _k, _lbl in [("persona","Personas"),("bytetrack","ByteTrack"),("epp","EPP"),
                      ("armas","Armas"),("zonas","Zonas"),("operadores","Operadores"),
                      ("objeto_movido","Obj.Movidos"),("caidas","Caidas")]:
        _v = _cfg_summary[_k]
        _bg2 = "#052e16" if _v else "#2d1b00"; _cl2 = "#22c55e" if _v else "#f59e0b"
        _fr = tk.Frame(_badge_f, bg=_bg2, padx=8, pady=4); _fr.pack(side="left", padx=4, pady=3)
        tk.Label(_fr, text="{} {}".format("+" if _v else "!", _lbl), fg=_cl2, bg=_bg2,
                 font=("Segoe UI", 9, "bold")).pack()
        if not _v:
            tk.Label(_fr, text="No config.", fg="#78350f", bg=_bg2,
                     font=("Segoe UI", 7)).pack()
    tk.Frame(_card_sc, bg="#0b1220", height=6).pack()

    _sec(padded, "BENCHMARK", "#f59e0b")
    _card_bm = _card(padded)
    tk.Frame(_card_bm, bg="#0b1220", height=10).pack()
    _btn_row1 = tk.Frame(_card_bm, bg="#0b1220"); _btn_row1.pack(padx=14, pady=(0,6), anchor="w")
    _btn_row2 = tk.Frame(_card_bm, bg="#0b1220"); _btn_row2.pack(padx=14, pady=(0,10), anchor="w")
    tk.Frame(_card_bm, bg="#0b1220", height=4).pack()

    _card_prog = tk.Frame(padded, bg="#0b1220",
                          highlightbackground="#f59e0b", highlightthickness=1)
    _prog_title_lbl = tk.Label(_card_prog, text="BENCHMARK EN EJECUCION",
                                fg="#f59e0b", bg="#0b1220", font=("Segoe UI", 12, "bold"))
    _prog_title_lbl.pack(anchor="w", padx=14, pady=(10, 4))
    _prog_status = tk.Label(_card_prog, text="Iniciando...", fg="#94a3b8", bg="#0b1220",
                             font=("Segoe UI", 10))
    _prog_status.pack(anchor="w", padx=14, pady=(0, 6))
    _prog_vars = {}
    for _sc_key2, _sc_lbl2 in [("base","Escenario Base"),("sostenido","Esc. Sostenido"),
                                 ("pico","Escenario Pico"),("adverso","Esc. Adverso"),
                                 ("rapido","Bm. Rapido")]:
        _fp = tk.Frame(_card_prog, bg="#0b1220"); _fp.pack(fill="x", padx=14, pady=2)
        tk.Label(_fp, text=_sc_lbl2, fg="#64748b", bg="#0b1220",
                 font=("Segoe UI", 9), width=20, anchor="w").pack(side="left")
        _pv2 = tk.StringVar(value="--")
        _prog_vars[_sc_key2] = _pv2
        _cv2 = tk.Canvas(_fp, bg="#0b1220", height=14, highlightthickness=0)
        _cv2.pack(side="left", fill="x", expand=True, padx=(4,4))
        tk.Label(_fp, textvariable=_pv2, fg="#3b82f6", bg="#0b1220",
                 font=("Segoe UI", 9, "bold"), width=8, anchor="e").pack(side="left")
        def _dpb(cv, pv, *_):
            cv.delete("all"); w2 = cv.winfo_width() or 200
            try: pct2 = float(pv.get().replace("%","").strip())
            except: pct2 = 0.0
            pct2 = max(0.0, min(100.0, pct2))
            cv.create_rectangle(0, 2, w2, 12, fill="#1e293b", outline="")
            cv.create_rectangle(0, 2, int(w2*pct2/100), 12, fill="#3b82f6", outline="")
        _cv2.bind("<Configure>", lambda e, cv=_cv2, pv=_pv2: _dpb(cv, pv))
        _pv2.trace_add("write", lambda *a, cv=_cv2, pv=_pv2: _dpb(cv, pv))
    _live_metrics = tk.Label(_card_prog, text="", fg="#22d3ee", bg="#0b1220",
                              font=("Courier New", 9), justify="left")
    _live_metrics.pack(anchor="w", padx=14, pady=(4, 10))

    _card_res    = tk.Frame(padded, bg="#0b1220",
                             highlightbackground="#22c55e", highlightthickness=1)
    _res_content = tk.Frame(_card_res, bg="#0b1220"); _res_content.pack(fill="x")

    _sec(padded, "HISTORIAL", "#22d3ee")
    _card_hist  = _card(padded)
    _hist_inner = tk.Frame(_card_hist, bg="#0b1220"); _hist_inner.pack(fill="x", padx=14, pady=8)

    def _build_history_list():
        for w in _hist_inner.winfo_children():
            try: w.destroy()
            except: pass
        hist = _bm_load_history()
        if not hist:
            tk.Label(_hist_inner, text="Sin benchmarks ejecutados.", fg="#475569",
                     bg="#0b1220", font=("Segoe UI", 10)).pack(anchor="w", pady=4)
            return
        hdr = tk.Frame(_hist_inner, bg="#132030"); hdr.pack(fill="x")
        for _t2, _w2 in [("Fecha",160),("Tipo",80),("Usuario",80),
                          ("CPU",70),("FPS",70),("Estado",90),("",40)]:
            tk.Label(hdr, text=_t2, fg="#94a3b8", bg="#132030",
                     font=("Segoe UI", 9, "bold"), width=_w2//8,
                     anchor="w").pack(side="left", padx=4, pady=3)
        for i2, rec2 in enumerate(hist[:20]):
            rbg = "#0d1b2e" if i2 % 2 == 0 else "#0b1220"
            rr = tk.Frame(_hist_inner, bg=rbg); rr.pack(fill="x")
            ts2  = rec2.get("timestamp","")[:16]; tp2 = rec2.get("type","--")
            usr2 = rec2.get("user","--"); cap2 = rec2.get("capacity",{})
            cpu2 = "{:.0f}%".format(cap2.get("cpu_max",0)) if cap2.get("cpu_max") else "--"
            fps2 = "{:.1f}".format(cap2.get("fps_avg",0)) if cap2.get("fps_avg") else "--"
            ok2  = rec2.get("passed", None)
            stxt2 = ("Aprobado" if ok2 else "Revisar") if ok2 is not None else "--"
            scol2 = "#22c55e" if ok2 else "#ef4444" if ok2 is False else "#94a3b8"
            for _vv2, _ww2, _cc2 in [(ts2,160,"#e5e7eb"),(tp2,80,"#a855f7"),
                                      (usr2,80,"#94a3b8"),(cpu2,70,"#3b82f6"),
                                      (fps2,70,"#22d3ee"),(stxt2,90,scol2)]:
                tk.Label(rr, text=str(_vv2), fg=_cc2, bg=rbg,
                         font=("Segoe UI", 9), width=_ww2//8,
                         anchor="w").pack(side="left", padx=4, pady=3)
            def _del_hist(idx2=i2):
                h3 = _bm_load_history()
                if idx2 < len(h3): del h3[idx2]
                try:
                    with open("benchmark_history.json","w",encoding="utf-8") as _hf2:
                        json.dump(h3, _hf2, ensure_ascii=False, indent=2)
                except Exception: pass
                _build_history_list()
            tk.Button(rr, text="X", command=_del_hist, bg=rbg, fg="#ef4444",
                      relief="flat", font=("Segoe UI", 8), cursor="hand2",
                      width=3).pack(side="left", padx=4)
    _build_history_list()

    _sec(padded, "QUE ES BENCHMARK OFICIAL?", "#64748b")
    _card_tut = _card(padded)
    _tut_txt = (
        "Benchmark Oficial ejecuta pruebas controladas sobre la configuracion real\n"
        "de Vigilant Pro para medir rendimiento, estabilidad y escalabilidad.\n\n"
        "No modifica camaras, zonas, operadores ni configuraciones.\n\n"
        "Mide: CPU  RAM  GPU  FPS  Throughput  Latencia  Tiempo de procesamiento IA\n\n"
        "Escenario Base:      1 camara - 5 min   - desempeno nominal\n"
        "Escenario Sostenido: 2 camaras - 5 min  - estabilidad operativa\n"
        "Escenario Pico:      Todas    - 10 min  - capacidad maxima\n"
        "Escenario Adverso:   Frames degradados  - condiciones dificiles\n\n"
        "IMPORTANTE: NO mide precision, recall ni F1.\n"
        "Esas metricas pertenecen al apartado de desempeno de IA."
    )
    tk.Label(_card_tut, text=_tut_txt, fg="#94a3b8", bg="#0b1220",
             font=("Segoe UI", 9), justify="left", anchor="w").pack(padx=14, pady=12, anchor="w")

    _adv_mode  = tk.StringVar(value="normal")
    _adv_level = tk.StringVar(value="media")
    _adv_panel = tk.Frame(padded, bg="#0f1929",
                          highlightbackground="#1e293b", highlightthickness=1)
    tk.Frame(_adv_panel, bg="#0f1929", height=8).pack()
    tk.Label(_adv_panel, text="Modo Adverso:",
             fg="#f59e0b", bg="#0f1929", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=14)
    for _av, _al in [("normal","Normal"),("baja_luz","Baja Iluminacion"),
                     ("baja_cal","Baja Calidad"),("ambos","Ambos")]:
        tk.Radiobutton(_adv_panel, text=_al, variable=_adv_mode, value=_av,
                       fg="#e2e8f0", bg="#0f1929", selectcolor="#1e293b",
                       activebackground="#0f1929",
                       font=("Segoe UI", 9)).pack(anchor="w", padx=14, pady=1)
    _lv_f = tk.Frame(_adv_panel, bg="#0f1929"); _lv_f.pack(anchor="w", padx=14, pady=(4,0))
    tk.Label(_lv_f, text="Nivel iluminacion:", fg="#64748b", bg="#0f1929",
             font=("Segoe UI", 9)).pack(side="left")
    for _lv, _ll in [("leve","Leve"),("media","Media"),("severa","Severa")]:
        tk.Radiobutton(_lv_f, text=_ll, variable=_adv_level, value=_lv,
                       fg="#e2e8f0", bg="#0f1929", selectcolor="#1e293b",
                       activebackground="#0f1929",
                       font=("Segoe UI", 9)).pack(side="left", padx=6)
    tk.Frame(_adv_panel, bg="#0f1929", height=8).pack()

    _current_snaps   = {}
    _bm_run_results  = [None]
    _bm_cam_handles  = [{}]   # {cam_id: running_flag} para capturas headless del benchmark

    def _hide_extras():
        _card_prog.pack_forget(); _adv_panel.pack_forget(); _card_res.pack_forget()

    def _show_prog_panel():
        _card_prog.pack(fill="x", pady=(0, 6), before=_card_hist)

    def _upd_prog(sc_key, pct):
        if main.winfo_exists():
            _prog_vars[sc_key].set("{:d}%".format(int(pct)))

    def _live_line(snap):
        cpu3 = "{:.1f}%".format(snap["cpu"])
        ram3 = "{:.0f}MB".format(snap["ram_mb"])
        gpu3 = "{}%".format(snap.get("gpu")) if snap.get("gpu") is not None else "N/D"
        fps3 = "{:.1f}fps".format(snap["fps"])
        lat3 = "{:.0f}ms".format(snap.get("lat_avg_ms",0)) if snap.get("lat_avg_ms") else "N/D"
        return "CPU:{:>7}  RAM:{:>8}  GPU:{:>6}  FPS:{:>8}  Lat:{:>7}".format(
            cpu3, ram3, gpu3, fps3, lat3)

    def _apply_adv_frame(frame, mode, level):
        f = frame.copy()
        if mode in ("baja_luz","ambos"):
            alpha = {"leve":0.5,"media":0.25,"severa":0.08}.get(level,0.25)
            f = cv2.convertScaleAbs(f, alpha=alpha, beta=0)
        if mode in ("baja_cal","ambos"):
            h3, w3 = f.shape[:2]
            small = cv2.resize(f, (max(1,w3//3), max(1,h3//3)))
            f = cv2.resize(small, (w3, h3))
            _, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 30])
            f = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return f

    def _scenario_worker(sc_key, duration_s, done_cb):
        snaps3 = []; _benchmark_cancel_flag[0] = False
        start3 = time.time()
        while not _benchmark_cancel_flag[0]:
            elapsed3 = time.time() - start3
            if elapsed3 >= duration_s: break
            snap3 = _bm_collect_snapshot(_proc, _gpu_handle)
            snaps3.append(snap3)
            pct3 = int(min(elapsed3 / duration_s * 100, 99))
            def _ui3(p=pct3, s=snap3):
                _upd_prog(sc_key, p)
                if _live_metrics.winfo_exists(): _live_metrics.configure(text=_live_line(s))
            main.after(0, _ui3)
            time.sleep(1.0)
        main.after(0, lambda: _upd_prog(sc_key, 100))
        _current_snaps[sc_key] = snaps3
        agg3 = _bm_aggregate(snaps3) if snaps3 else {"n_samples": 0}
        main.after(0, lambda a3=agg3: done_cb(sc_key, a3))

    def _adversarial_worker(sc_key, duration_s, adv_mode, adv_level, done_cb):
        snaps4 = []; lats4 = []; _benchmark_cancel_flag[0] = False
        start4 = time.time()
        mdl4 = (_model_general or
                (list(_track_models_gen.values())[0] if _track_models_gen else None))
        while not _benchmark_cancel_flag[0]:
            elapsed4 = time.time() - start4
            if elapsed4 >= duration_s: break
            snap4 = _bm_collect_snapshot(_proc, _gpu_handle)
            if mdl4 is not None:
                raw4 = None
                for _ci4 in selected_cameras:
                    _fr4 = _ai_raw_buffer.get(_ci4)
                    if _fr4 is not None: raw4 = _fr4.copy(); break
                if raw4 is not None:
                    try:
                        deg4 = _apply_adv_frame(raw4, adv_mode, adv_level)
                        t04  = time.time()
                        mdl4(deg4, verbose=False, stream=False)
                        lats4.append(time.time() - t04)
                        if lats4:
                            snap4["lat_avg_ms"] = sum(lats4[-20:])/len(lats4[-20:])*1000
                    except Exception: pass
            snaps4.append(snap4)
            pct4 = int(min(elapsed4 / duration_s * 100, 99))
            def _ui4(p=pct4, s=snap4):
                _upd_prog(sc_key, p)
                if _live_metrics.winfo_exists(): _live_metrics.configure(text=_live_line(s))
            main.after(0, _ui4)
            time.sleep(1.0)
        main.after(0, lambda: _upd_prog(sc_key, 100))
        _current_snaps[sc_key] = snaps4
        agg4 = _bm_aggregate(snaps4) if snaps4 else {"n_samples": 0}
        if lats4:
            lms4 = [l*1000 for l in lats4]
            for _lk4, _lp4 in [("lat_avg",50),("lat_p50",50),("lat_p95",95),("lat_p99",99)]:
                agg4[_lk4] = {"avg":_bm_percentile(lms4,_lp4),"min":min(lms4),
                               "max":max(lms4),"p95":_bm_percentile(lms4,95)}
        main.after(0, lambda a4=agg4: done_cb(sc_key, a4))

    # ── helpers benchmark con captura headless ────────────────────────────────
    def _get_bm_cameras():
        """Retorna IDs de cámaras para el benchmark: primero selected_cameras,
        luego cámaras con config IA no seleccionadas. USB (0-19) antes que RTSP (≥100)."""
        seen = set()
        usb, rtsp = [], []
        for cid in list(selected_cameras) + sorted(int(k) for k in ai_config if k.isdigit()):
            if not isinstance(cid, int) or cid in seen:
                continue
            seen.add(cid)
            if 0 <= cid < 20:
                usb.append(cid)
            elif cid >= 100:
                rtsp.append(cid)
        return sorted(usb) + sorted(rtsp)

    def _bm_start_cameras(cam_ids):
        """Inicia captura USB headless + IA para benchmark (sin renderizar en UI)."""
        handles = {}
        for cid in cam_ids:
            cid_str = str(cid)
            if cid >= 100:
                # RTSP: si ya hay frames en buffer (ffmpeg externo) solo activa IA
                if _cam_preview_frame.get(cid_str) is not None:
                    _init_ai_for_cam(cid)
                continue
            running = [True]
            fc = [0]
            def _cap(c=cid, cs=cid_str, r=running, counter=fc):
                try:
                    cap2 = cv2.VideoCapture(c, cv2.CAP_DSHOW)
                    if not cap2.isOpened():
                        cap2 = cv2.VideoCapture(c)
                    if not cap2.isOpened():
                        return
                    cap2.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    while r[0]:
                        ret, frm = cap2.read()
                        if ret and frm is not None:
                            counter[0] += 1
                            _cam_last_frame[cs] = time.time()
                            _cam_preview_frame[cs] = frm
                            if _ai_started.get(cs) and counter[0] % 2 == 0:
                                if _ai_raw_buffer.get(cs) is None:
                                    _ai_raw_buffer[cs] = frm.copy()
                        else:
                            time.sleep(0.01)
                    cap2.release()
                except Exception:
                    pass
            _threading.Thread(target=_cap, daemon=True).start()
            _init_ai_for_cam(cid)
            handles[cid] = running
        return handles

    def _bm_stop_cameras(handles):
        """Detiene capturas headless y limpia estado IA de esas cámaras."""
        for cid, running in list(handles.items()):
            running[0] = False
            cid_str = str(cid)
            _ai_thread_active[cid_str] = False
            _ai_started.pop(cid_str, None)
        handles.clear()

    def _start_quick():
        if _benchmark_running[0]:
            show_notification("BENCHMARK","Ya hay un benchmark en ejecucion.","#f59e0b"); return
        _benchmark_running[0] = True; _benchmark_cancel_flag[0] = False
        _current_snaps.clear(); _bm_run_results[0] = None
        _bm_cam_handles[0] = {}
        _hide_extras()
        for _k2 in _prog_vars: _prog_vars[_k2].set("--")
        try:
            if _prog_status.winfo_exists():
                _prog_status.configure(text="Benchmark Rapido - preparando camara...")
        except Exception: pass
        _prog_vars["rapido"].set("0%"); _show_prog_panel()
        _configured_cams_q = _get_bm_cameras()
        _qd = [False]
        def _on_q(sk, ag):
            if not _qd[0]:
                _qd[0] = True
                _bm_stop_cameras(_bm_cam_handles[0])
                _finish_benchmark({"rapido": ag}, "rapido")
        def _launch_quick():
            if _configured_cams_q:
                _bm_cam_handles[0] = _bm_start_cameras(_configured_cams_q[:1])
                time.sleep(5)
            try:
                if _prog_status.winfo_exists():
                    _prog_status.configure(text="Benchmark Rapido - monitoreando 5 minutos...")
            except Exception: pass
            _scenario_worker("rapido", 300, _on_q)
        t_q = _threading.Thread(target=_launch_quick, daemon=True)
        _benchmark_thread[0] = t_q
        t_q.start()

    def _start_official():
        if _benchmark_running[0]:
            show_notification("BENCHMARK","Ya hay un benchmark en ejecucion.","#f59e0b"); return
        _configured_cams = _get_bm_cameras()
        if not _configured_cams:
            show_notification("BENCHMARK","No hay camaras configuradas con IA.","#ef4444"); return
        _benchmark_running[0] = True; _benchmark_cancel_flag[0] = False
        _current_snaps.clear(); _bm_run_results[0] = None
        _bm_cam_handles[0] = {}
        _hide_extras()
        for _k2 in _prog_vars: _prog_vars[_k2].set("--")
        try:
            if _prog_status.winfo_exists():
                _prog_status.configure(text="Benchmark Oficial - preparando camaras...")
        except Exception: pass
        _show_prog_panel()

        _scs5   = {}
        _advm   = _adv_mode.get()
        _advl   = _adv_level.get()
        _order5  = ["base","sostenido","pico","adverso"]
        _dur5    = {"base":300,"sostenido":300,"pico":600,"adverso":120}
        _n_cams5 = {"base":1, "sostenido":2,
                    "pico":len(_configured_cams), "adverso":len(_configured_cams)}
        _idx5   = [0]

        def _on5(sk5, ag5):
            _scs5[sk5] = ag5; _idx5[0] += 1
            if _benchmark_cancel_flag[0] or _idx5[0] >= len(_order5):
                _bm_stop_cameras(_bm_cam_handles[0])
                _finish_benchmark(_scs5, "oficial"); return
            nxt5 = _order5[_idx5[0]]; dur5 = _dur5[nxt5]
            try:
                if _prog_status.winfo_exists():
                    _prog_status.configure(
                        text="Ejecutando {}... ({} min)".format(nxt5.title(), dur5//60))
            except Exception: pass
            def _transition():
                if _benchmark_cancel_flag[0]:
                    return
                _bm_stop_cameras(_bm_cam_handles[0])
                n = _n_cams5[nxt5]
                _bm_cam_handles[0] = _bm_start_cameras(_configured_cams[:n])
                time.sleep(5)
                if _benchmark_cancel_flag[0]:
                    _bm_stop_cameras(_bm_cam_handles[0]); return
                if nxt5 == "adverso":
                    _adversarial_worker(nxt5, dur5, _advm, _advl, _on5)
                else:
                    _scenario_worker(nxt5, dur5, _on5)
            _threading.Thread(target=_transition, daemon=True).start()

        def _launch_base():
            if _benchmark_cancel_flag[0]:
                _benchmark_running[0] = False; return
            _bm_cam_handles[0] = _bm_start_cameras(_configured_cams[:1])
            try:
                if _prog_status.winfo_exists():
                    _prog_status.configure(
                        text="Benchmark Oficial - Escenario Base (5 min)...")
            except Exception: pass
            time.sleep(5)
            if _benchmark_cancel_flag[0]:
                _bm_stop_cameras(_bm_cam_handles[0]); _benchmark_running[0] = False; return
            _scenario_worker("base", _dur5["base"], _on5)

        t5 = _threading.Thread(target=_launch_base, daemon=True)
        _benchmark_thread[0] = t5
        t5.start()

    def _cancel_bm():
        _benchmark_cancel_flag[0] = True; _benchmark_running[0] = False
        _bm_stop_cameras(_bm_cam_handles[0])
        try:
            if _prog_status.winfo_exists():
                _prog_status.configure(text="Benchmark cancelado.")
        except Exception: pass
        show_notification("BENCHMARK","Cancelado.","#f59e0b")

    def _finish_benchmark(scs6, btype):
        _benchmark_running[0] = False
        cap6 = _bm_capacity_estimate(scs6)
        passed6 = bool(scs6) and all(
            (sc6.get("cpu",{}).get("max") or 0) < 95 and
            (sc6.get("fps",{}).get("avg") or 0) >= 5
            for sc6 in scs6.values() if sc6 and sc6.get("n_samples",0) > 0)
        rec6 = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type":      btype,
            "user":      current_user or "--",
            "hardware":  _hw,
            "config":    _cfg_summary,
            "scenarios": scs6,
            "capacity":  cap6,
            "passed":    passed6,
        }
        _bm_run_results[0] = rec6; _benchmark_results[0] = rec6
        _bm_save_to_history(rec6)
        try:
            if _prog_status.winfo_exists():
                _prog_status.configure(text="Benchmark completado.")
        except Exception: pass
        try:
            _build_results(rec6); _build_history_list()
        except Exception: pass
        show_notification("BENCHMARK","Completado. Revisa los resultados.","#22c55e")

    def _fmt_sd(sd, unit="", hi=None, lo=None):
        if not sd or sd.get("avg") is None: return "NO DISPONIBLE", "#475569"
        avg = sd["avg"]; mx = sd.get("max", avg)
        txt = "{:.1f}{}  (max: {:.1f}{})".format(avg, unit, mx, unit)
        if hi is not None and avg > hi: return txt, "#ef4444"
        if lo is not None and avg < lo: return txt, "#ef4444"
        return txt, "#22c55e"

    def _build_results(rec7):
        for w in _res_content.winfo_children():
            try: w.destroy()
            except: pass
        scs7 = rec7.get("scenarios",{}); cap7 = rec7.get("capacity",{})
        ok7  = rec7.get("passed", False)
        hf = tk.Frame(_res_content, bg="#0b1220"); hf.pack(fill="x", padx=14, pady=10)
        tk.Label(hf, text="APROBADO" if ok7 else "REQUIERE REVISION",
                 fg="#22c55e" if ok7 else "#f59e0b",
                 bg="#0b1220", font=("Segoe UI", 15,"bold")).pack(side="left")
        tk.Label(hf, text="  {}  |  {}".format(
            rec7.get("timestamp",""), rec7.get("type","").title()),
                 fg="#64748b", bg="#0b1220", font=("Segoe UI", 9)).pack(side="left")
        SN = {"base":"Escenario Base","sostenido":"Esc. Sostenido",
              "pico":"Escenario Pico","adverso":"Esc. Adverso","rapido":"Bm. Rapido"}
        for sk7, snm7 in SN.items():
            sc7 = scs7.get(sk7)
            if not sc7 or sc7.get("n_samples",0) == 0: continue
            fsc7 = tk.Frame(_res_content, bg="#0d1b2e"); fsc7.pack(fill="x", padx=14, pady=4)
            tk.Label(fsc7, text=snm7, fg="#3b82f6", bg="#0d1b2e",
                     font=("Segoe UI", 10,"bold")).pack(anchor="w", padx=8, pady=(6,2))
            tk.Frame(fsc7, bg="#132030", height=1).pack(fill="x", padx=8, pady=2)
            for lbl7, key7, unit7, hi7, lo7 in [
                ("CPU prom:","cpu","%",80,None),("RAM prom:","ram"," MB",None,None),
                ("GPU prom:","gpu","%",85,None),("FPS prom:","fps"," fps",None,10),
                ("Throughput:","throughput"," fps",None,None),
                ("Lat. prom:","lat_avg"," ms",500,None),
                ("Lat. p95:","lat_p95"," ms",1000,None),
            ]:
                txt7, col7 = _fmt_sd(sc7.get(key7), unit7, hi7, lo7)
                r7 = tk.Frame(fsc7, bg="#0d1b2e"); r7.pack(fill="x", padx=8, pady=1)
                tk.Label(r7, text=lbl7, fg="#64748b", bg="#0d1b2e",
                         font=("Segoe UI", 9), width=18, anchor="w").pack(side="left")
                tk.Label(r7, text=txt7, fg=col7, bg="#0d1b2e",
                         font=("Segoe UI", 9,"bold"), anchor="w").pack(side="left")
            for lbl8, vk8 in [("Detecciones:",str(sc7.get("det_gen",0))),
                               ("Eventos:",   str(sc7.get("events_gen",0)))]:
                r8 = tk.Frame(fsc7, bg="#0d1b2e"); r8.pack(fill="x", padx=8, pady=1)
                tk.Label(r8, text=lbl8, fg="#64748b", bg="#0d1b2e",
                         font=("Segoe UI", 9), width=18, anchor="w").pack(side="left")
                tk.Label(r8, text=vk8, fg="#94a3b8", bg="#0d1b2e",
                         font=("Segoe UI", 9,"bold")).pack(side="left")
            tk.Frame(fsc7, bg="#0d1b2e", height=4).pack()
        if cap7.get("status") == "ok":
            fca7 = tk.Frame(_res_content, bg="#052e16"); fca7.pack(fill="x", padx=14, pady=4)
            tk.Label(fca7, text="CAPACIDAD RECOMENDADA", fg="#22c55e", bg="#052e16",
                     font=("Segoe UI", 10,"bold")).pack(anchor="w", padx=8, pady=(6,2))
            tk.Frame(fca7, bg="#166534", height=1).pack(fill="x", padx=8, pady=2)
            sat7 = cap7.get("saturated",False)
            for lb9, vl9, cl9 in [
                ("Camaras recom.:", "{} Full HD".format(cap7.get("max_cams",0)), "#22c55e"),
                ("CPU/camara (max):", "{:.1f}%".format(cap7.get("cpu_per_cam",0)), "#3b82f6"),
                ("Utilizacion max:", "{:.0f}%".format(cap7.get("utilization_pct",0)),
                 "#ef4444" if sat7 else "#22c55e"),
                ("FPS saludable:", "Si" if cap7.get("fps_ok") else "No",
                 "#22c55e" if cap7.get("fps_ok") else "#ef4444"),
                ("Estado:", "SATURADO" if sat7 else "NORMAL",
                 "#ef4444" if sat7 else "#22c55e"),
            ]:
                r9 = tk.Frame(fca7, bg="#052e16"); r9.pack(fill="x", padx=8, pady=2)
                tk.Label(r9, text=lb9, fg="#64748b", bg="#052e16",
                         font=("Segoe UI", 9), width=22, anchor="w").pack(side="left")
                tk.Label(r9, text=vl9, fg=cl9, bg="#052e16",
                         font=("Segoe UI", 9,"bold")).pack(side="left")
            tk.Frame(fca7, bg="#052e16", height=4).pack()
        tk.Frame(_res_content, bg="#0b1220", height=8).pack()
        _card_res.pack(fill="x", pady=(0,6), before=_card_hist)

    _lbl_pdf = tk.Label(padded, text="", fg="#22c55e", bg="#020617", font=("Segoe UI", 9))

    def _do_pdf():
        rec_pdf = _bm_run_results[0] or _benchmark_results[0]
        if not rec_pdf:
            show_notification("PDF","Ejecuta un benchmark primero.","#f59e0b"); return
        ts_pdf = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn_pdf = "benchmark_vigilant_{}.pdf".format(ts_pdf)
        _lbl_pdf.configure(text="Generando PDF...", fg="#f59e0b"); _lbl_pdf.pack(anchor="w")
        def _gen():
            ok_pdf, msg_pdf = _bm_generate_pdf(rec_pdf, fn_pdf)
            main.after(0, lambda: _lbl_pdf.configure(
                text="PDF: {}".format(msg_pdf),
                fg="#22c55e" if ok_pdf else "#ef4444"))
            if ok_pdf:
                main.after(0, lambda: show_notification(
                    "PDF EXPORTADO","Archivo: {}".format(fn_pdf),"#22c55e"))
        _threading.Thread(target=_gen, daemon=True).start()

    _bm_btn(_btn_row1, "Benchmark Rapido (5 min)",
             _start_quick, "#1e3a5f","#93c5fd").pack(side="left", padx=(0,8))
    _bm_btn(_btn_row1, "Benchmark Oficial Concurso",
             _start_official, "#3b1f6b","#d8b4fe").pack(side="left", padx=(0,8))
    _bm_btn(_btn_row1, "Cancelar", _cancel_bm, "#7f1d1d","#fca5a5").pack(side="left")
    _bm_btn(_btn_row2, "Exportar PDF", _do_pdf, "#166534","#86efac").pack(side="left", padx=(0,8))
    _bm_btn(_btn_row2, "Actualizar Historial",
             _build_history_list, "#0c4a6e","#7dd3fc").pack(side="left", padx=(0,8))
    _av_vis = [False]
    def _tog_adv():
        if _av_vis[0]: _adv_panel.pack_forget(); _av_vis[0] = False
        else: _adv_panel.pack(fill="x", pady=(0,6), before=_card_hist); _av_vis[0] = True
    _bm_btn(_btn_row2, "Opciones Adverso", _tog_adv, "#1c1917","#d6d3d1").pack(side="left")

    if _benchmark_results[0]:
        _build_results(_benchmark_results[0])

    try:
        _proc_epoch = _psutil.Process(os.getpid()).create_time()
    except Exception:
        _proc_epoch = time.time()

    def _fmt_dur(secs):
        h = int(secs//3600); m = int((secs%3600)//60); s2 = int(secs%60)
        return "{:02d}:{:02d}:{:02d}".format(h, m, s2)

    def _refresh():
        if not padded.winfo_exists(): return
        now = time.time()
        _vv["uptime"].set(_fmt_dur(now - _proc_epoch))
        if _proc:
            try:
                _vv["cpu"].set("{:.1f} %".format(_proc.cpu_percent(interval=None)))
                mi2 = _proc.memory_info()
                _vv["ram"].set("{:.1f} MB  ({:.2f} GB)".format(
                    mi2.rss/1_048_576, mi2.rss/1_073_741_824))
            except Exception: pass
        if _gpu_handle is not None and _pynvml is not None:
            try:
                _ut2 = _pynvml.nvmlDeviceGetUtilizationRates(_gpu_handle)
                _gm2 = _pynvml.nvmlDeviceGetMemoryInfo(_gpu_handle)
                _vv["gpu_use"].set("{} %".format(_ut2.gpu))
                _vv["gpu_vram"].set("{:.0f} MB / {:.0f} MB".format(
                    _gm2.used/1_048_576, _gm2.total/1_048_576))
                try:
                    _vv["gpu_temp"].set("{} C".format(
                        _pynvml.nvmlDeviceGetTemperature(
                            _gpu_handle, _pynvml.NVML_TEMPERATURE_GPU)))
                except Exception: _vv["gpu_temp"].set("NO DISPONIBLE")
            except Exception: pass
        else:
            _vv["gpu_use"].set("NO DISPONIBLE")
            _vv["gpu_vram"].set("NO DISPONIBLE")
            _vv["gpu_temp"].set("NO DISPONIBLE")
        fps_v2 = []
        for _ci2 in selected_cameras:
            tsq2 = _ai_fps_ts.get(str(_ci2))
            if tsq2 and len(tsq2) >= 2:
                span2 = tsq2[-1] - tsq2[0]
                if span2 > 0: fps_v2.append((len(tsq2)-1)/span2)
        _vv["fps"].set("{:.1f} fps".format(sum(fps_v2)/len(fps_v2))
                       if fps_v2 else "NO DISPONIBLE")
        lats2 = list(_diag_latency_all)
        if lats2:
            lms2 = [l*1000 for l in lats2]
            _vv["lat_avg"].set("{:.1f} ms".format(sum(lms2)/len(lms2)))
            p95v = _bm_percentile(lms2, 95)
            _vv["lat_p95"].set("{:.1f} ms".format(p95v) if p95v is not None else "N/D")
        else:
            _vv["lat_avg"].set("NO DISPONIBLE"); _vv["lat_p95"].set("NO DISPONIBLE")
        _vv["cams_ai"].set(str(sum(1 for _ci2 in selected_cameras
                                   if _ai_started.get(str(_ci2)))))
        _vv["events"].set(str(len(_events_list)))
        _vv["det"].set(str(sum(_diag_cam_det_count.values())))
        _dp2 = os.path.join(SNAPSHOTS_DIR, "analitica")
        try:
            sn2 = len(os.listdir(_dp2)) if os.path.isdir(_dp2) else 0
        except Exception:
            sn2 = 0
        _vv["snaps"].set(str(sn2))
        _rp2 = "recordings"
        try:
            if os.path.exists(RECORDINGS_DIR_FILE):
                with open(RECORDINGS_DIR_FILE,"r",encoding="utf-8") as _rf2:
                    _rp2 = json.load(_rf2).get("path","recordings")
        except Exception: pass
        try:
            _vv["vids"].set(str(
                sum(1 for _fn2 in os.listdir(_rp2)
                    if os.path.splitext(_fn2)[1].lower() in {".avi",".mp4",".mkv"})
                if os.path.isdir(_rp2) else 0))
        except Exception: _vv["vids"].set("0")
        _diag_panel_after[0] = main.after(2000, _refresh)

    _refresh()



current = None

def create_button(text, command=None):
    frame = tk.Frame(sidebar, bg="#020617")
    frame.pack(fill="x", padx=8, pady=2)

    line = tk.Frame(frame, bg="#020617", width=4)
    line.pack(side="left", fill="y")

    container = tk.Frame(frame, bg="#020617", height=44)
    container.pack(side="left", fill="x", expand=True)
    container.pack_propagate(False)

    label = tk.Label(container, text=text,
                     fg="#cbd5f5", bg="#020617",
                     font=("Segoe UI", 10), anchor="w")
    label.pack(fill="both", expand=True, padx=12)

    def activate(e=None):
        global current
        
        if current:

            try:

                if current[0].winfo_exists():
                    current[0].config(bg="#020617")

                if current[1].winfo_exists():
                    current[1].config(bg="#020617")

                if current[2].winfo_exists():
                    current[2].config(
                        bg="#020617",
                        fg="#cbd5f5"
                    )

            except:
                pass


        line.config(bg="#3b82f6")
        container.config(bg="#0b1220")
        label.config(bg="#0b1220", fg="#ffffff")

        current = (line, container, label)

        if command:
            command()

    for w in (container, label):
        w.bind("<Button-1>", activate)
        w.config(cursor="hand2")

    return activate, label

# =========================
# VISTA EN VIVO (multi-cámara)
# =========================
def show_live_view():
    global _live_cameras, _live_after_ids, ai_config

    clear_main()
    ai_config = load_ai_config()   # always use the latest saved config
    main.update()

    container = tk.Frame(main, bg="#020617")
    container.pack(fill="both", expand=True)

    # --- barra de grabación ---
    rec_bar = tk.Frame(main, bg="#0f172a", height=36)
    rec_bar.pack(fill="x", side="bottom")
    rec_bar.pack_propagate(False)

    def _toggle_rec(cam_id, btn):
        cid = str(cam_id)
        if _recording_enabled.get(cid):
            _stop_writer(cid)
            _recording_enabled[cid] = False
            btn.configure(text="⏺ Grabar", bg="#1e293b", fg="white")
        else:
            _recording_enabled[cid] = True
            btn.configure(text="⏹ Detener", bg="#ef4444", fg="white")

    def _take_snapshot(cam_id):
        cid_str = str(cam_id)
        clean   = _cam_preview_frame.get(cid_str)
        if clean is None:
            show_notification("SNAPSHOT", "Sin frame disponible.", "#f59e0b")
            return
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        _ensure_snapshots_dir()
        c_name = _cam_name_for(cam_id)
        ai_frm = _ai_frame_buffer.get(cid_str)
        src    = ai_frm if (_ai_started.get(cid_str) and ai_frm is not None) else clean
        bname  = f"snap_c{cam_id}_manual_{ts}_ia.jpg"
        ai_path = os.path.join(SNAPSHOTS_DIR, "analitica", bname)
        cv2.imwrite(ai_path, _burn_timestamp(src, c_name), [cv2.IMWRITE_JPEG_QUALITY, 90])
        show_notification("SNAPSHOT", f"Guardado: {bname}", "#22c55e")
        log_event(cam_id, "Snapshot Manual", "INFO", f"Archivo: {bname}")

    tk.Label(rec_bar, text="CONTROLES:", fg="#64748b", bg="#0f172a",
             font=("Segoe UI", 8, "bold")).pack(side="left", padx=(10, 4))

    for cid in selected_cameras:
        cam_sep = tk.Frame(rec_bar, bg="#1e293b", width=1)
        cam_sep.pack(side="left", fill="y", padx=4)
        tk.Label(rec_bar, text=f"Cam {cid}", fg="#64748b", bg="#0f172a",
                 font=("Segoe UI", 8)).pack(side="left", padx=(0, 4))
        btn = tk.Button(rec_bar, text="⏺ Grabar",
                        bg="#1e293b", fg="white", relief="flat",
                        font=("Segoe UI", 8, "bold"), padx=8, pady=2,
                        cursor="hand2")
        btn.configure(command=lambda c=cid, b=btn: _toggle_rec(c, b))
        btn.pack(side="left", padx=2)
        tk.Button(rec_bar, text="📷 Snap",
                  bg="#1e293b", fg="#22d3ee", relief="flat",
                  font=("Segoe UI", 8, "bold"), padx=8, pady=2,
                  cursor="hand2",
                  command=lambda c=cid: _take_snapshot(c)).pack(side="left", padx=2)

    # --- panel de estado IA (se actualiza automáticamente) ---
    status_bar = tk.Frame(main, bg="#050d1a", height=26)
    status_bar.pack(fill="x", side="bottom")
    status_bar.pack_propagate(False)

    def _make_status_lbl(parent, text, fg):
        return tk.Label(parent, text=text, fg=fg, bg="#050d1a",
                        font=("Segoe UI", 8))

    lbl_yolo  = _make_status_lbl(status_bar, "YOLO: —",       "#64748b")
    lbl_track = _make_status_lbl(status_bar, "ByteTrack: —",  "#64748b")
    lbl_rec   = _make_status_lbl(status_bar, "Grab: —",       "#64748b")
    lbl_fps   = _make_status_lbl(status_bar, "FPS IA: —",     "#64748b")
    lbl_watch = _make_status_lbl(status_bar, "Watchdog: —",   "#64748b")

    for lbl in [lbl_yolo, lbl_track, lbl_rec, lbl_fps, lbl_watch]:
        lbl.pack(side="left", padx=10)

    def _refresh_status_bar():
        if not status_bar.winfo_exists():
            return
        ai_on = any(_ai_started.get(str(c)) for c in selected_cameras)
        lbl_yolo.configure(
            text="YOLO: ACTIVO" if ai_on else "YOLO: INACTIVO",
            fg="#22c55e" if ai_on else "#ef4444")
        lbl_track.configure(
            text="ByteTrack: ACTIVO" if ai_on else "ByteTrack: INACTIVO",
            fg="#22c55e" if ai_on else "#ef4444")
        any_rec = any(_ai_writers.get(str(c)) for c in selected_cameras)
        lbl_rec.configure(
            text=f"Grab: ACTIVA ({_record_mode})" if any_rec else "Grab: INACTIVA",
            fg="#22c55e" if any_rec else "#64748b")
        # Compute average FPS across active cameras
        fps_vals = []
        for c in selected_cameras:
            ts_q = _ai_fps_ts.get(str(c))
            if ts_q and len(ts_q) >= 2:
                span = ts_q[-1] - ts_q[0]
                if span > 0:
                    fps_vals.append((len(ts_q) - 1) / span)
        fps_str = f"{sum(fps_vals)/len(fps_vals):.1f}" if fps_vals else "—"
        lbl_fps.configure(text=f"FPS IA: {fps_str}")
        wd_on = _watchdog_active[0]
        lbl_watch.configure(
            text="Watchdog: ON" if wd_on else "Watchdog: OFF",
            fg="#22c55e" if wd_on else "#64748b")
        status_bar.after(2000, _refresh_status_bar)

    _refresh_status_bar()

    # --- sin cámaras ---
    if not selected_cameras:
        msg_frame = tk.Frame(container, bg="#020617")
        msg_frame.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(msg_frame,
                 text="No hay cámaras seleccionadas.",
                 fg="#94a3b8", bg="#020617",
                 font=("Segoe UI", 16, "bold")).pack(pady=(0, 6))
        tk.Label(msg_frame,
                 text="Ve a Cámaras y selecciona al menos una.",
                 fg="#64748b", bg="#020617",
                 font=("Segoe UI", 11)).pack(pady=(0, 20))
        tk.Button(msg_frame, text="Ir a Cámaras",
                  bg="#2563eb", fg="white", relief="flat",
                  font=("Segoe UI", 11, "bold"), padx=20, pady=10,
                  cursor="hand2", command=show_cameras).pack()
        return

    # Construir mapa de cámaras seleccionadas (USB + RTSP)
    rtsp_map = {c['id']: c for c in rtsp_cameras}
    all_cameras_known = []
    for cid in selected_cameras:
        if isinstance(cid, int) and 0 <= cid < 20:
            all_cameras_known.append({
                'id': cid,
                'name': camera_names.get(str(cid), f'USB Camera {cid}'),
                'alias': camera_names.get(str(cid), ''),
                'status': 'Funcional'
            })
        elif isinstance(cid, int) and cid >= 100 and cid in rtsp_map:
            all_cameras_known.append(rtsp_map[cid])
    all_cameras = all_cameras_known
    cam_data_map = {cam['id']: cam for cam in all_cameras}

    # ── Barra de control IA ──────────────────────────────────────────────────
    ai_bar = tk.Frame(container, bg="#0b1220", height=46)
    ai_bar.pack(fill="x", side="top")
    ai_bar.pack_propagate(False)

    ai_status_lbl = tk.Label(
        ai_bar, text="IA: INACTIVA",
        fg="#ef4444", bg="#0b1220",
        font=("Segoe UI", 9, "bold")
    )
    ai_status_lbl.pack(side="left", padx=(12, 6), pady=12)

    def _init_ai_all():
        global ai_config
        ai_config = load_ai_config()
        ai_start_btn.configure(text="Cargando...", bg="#64748b", state="disabled")
        ai_status_lbl.configure(text="IA: INICIANDO...", fg="#facc15")

        def _do_init():
            ok = 0
            for cid in selected_cameras:
                if _init_ai_for_cam(cid):
                    ok += 1

            def _update_ui():
                if not ai_status_lbl.winfo_exists():
                    return
                if ok:
                    start_watchdog()
                    ai_status_lbl.configure(text=f"IA: ACTIVA  ({ok} cám.)", fg="#22c55e")
                    ai_start_btn.configure(text="Reiniciar IA", bg="#16a34a",
                                           state="normal")
                    show_notification("IA ACTIVADA",
                                      f"{ok} cámara(s) — YOLO + ByteTrack activos",
                                      "#22c55e")
                else:
                    ai_status_lbl.configure(text="IA: INACTIVA", fg="#ef4444")
                    ai_start_btn.configure(text="Iniciar IA", bg="#2563eb",
                                           state="normal")
                    show_notification("SIN CONFIGURACIÓN",
                                      "Configura la IA en Cámaras → Configurar IA.",
                                      "#f59e0b")

            ai_status_lbl.after(0, _update_ui)

        threading.Thread(target=_do_init, daemon=True).start()

    def _stop_ai_all():
        for cid in selected_cameras:
            cid_str = str(cid)
            _ai_thread_active[cid_str] = False
            _ai_started[cid_str]       = False
            log_event(cid_str, "IA Detenida", "INFO", "Procesamiento IA desactivado")
        stop_watchdog()
        _stop_all_writers()
        ai_status_lbl.configure(text="IA: INACTIVA", fg="#ef4444")
        ai_start_btn.configure(text="Iniciar IA", bg="#2563eb")
        show_notification("IA DETENIDA", "Procesamiento IA desactivado.", "#f59e0b")

    ai_start_btn = tk.Button(
        ai_bar, text="Iniciar IA",
        bg="#2563eb", fg="white", relief="flat",
        font=("Segoe UI", 9, "bold"), padx=14, pady=6,
        cursor="hand2", command=_init_ai_all
    )
    ai_start_btn.pack(side="left", padx=4)

    tk.Button(
        ai_bar, text="Detener IA",
        bg="#7f1d1d", fg="white", relief="flat",
        font=("Segoe UI", 9, "bold"), padx=14, pady=6,
        cursor="hand2", command=_stop_ai_all
    ).pack(side="left", padx=4)

    # Indicadores de config por cámara (solo las que están en cam_data_map)
    _preview_cameras_to_show = [cid for cid in selected_cameras if cid in cam_data_map]
    for cid in _preview_cameras_to_show:
        cid_str = str(cid)
        cfg     = ai_config.get(cid_str, {})
        n_rules = sum(1 for v in cfg.values() if v is True)
        col     = "#22c55e" if n_rules > 0 else "#64748b"
        alias   = cam_data_map[cid].get("alias") or cam_data_map[cid].get("name", f"Cam {cid}")
        tk.Label(
            ai_bar,
            text=f"{alias[:12]}: {n_rules} reglas",
            fg=col, bg="#0b1220",
            font=("Segoe UI", 8)
        ).pack(side="left", padx=8)

    # ── Selector de visualización ─────────────────────────────────────────────
    tk.Frame(ai_bar, bg="#1e293b", width=1).pack(
        side="left", fill="y", pady=6, padx=8)
    tk.Label(ai_bar, text="VISTA:", fg="#475569", bg="#0b1220",
             font=("Segoe UI", 8, "bold")).pack(side="left", padx=(0, 3))

    _is_normal = _display_mode[0] == "normal"
    _btn_normal = tk.Button(
        ai_bar, text="Normal",
        bg="#1e3a5f" if _is_normal else "#0d1a2d",
        fg="#e2e8f0" if _is_normal else "#475569",
        activebackground="#1e3a5f", activeforeground="#e2e8f0",
        relief="flat", font=("Segoe UI", 8, "bold"),
        padx=9, pady=4, cursor="hand2"
    )
    _btn_ia = tk.Button(
        ai_bar, text="IA",
        bg="#1e3a5f" if not _is_normal else "#0d1a2d",
        fg="#3b82f6" if not _is_normal else "#475569",
        activebackground="#1e3a5f", activeforeground="#3b82f6",
        relief="flat", font=("Segoe UI", 8, "bold"),
        padx=9, pady=4, cursor="hand2"
    )

    def _set_display_normal():
        _display_mode[0] = "normal"
        _btn_normal.configure(bg="#1e3a5f", fg="#e2e8f0")
        _btn_ia.configure(bg="#0d1a2d", fg="#475569")

    def _set_display_ia():
        _display_mode[0] = "ia"
        _btn_ia.configure(bg="#1e3a5f", fg="#3b82f6")
        _btn_normal.configure(bg="#0d1a2d", fg="#475569")

    _btn_normal.configure(command=_set_display_normal)
    _btn_ia.configure(command=_set_display_ia)
    _btn_normal.pack(side="left", padx=1)
    _btn_ia.pack(side="left", padx=1)

    # Controles de monitoreo — derecha del ai_bar
    if _monitoring_mode[0]:
        tk.Button(
            ai_bar, text="🔓 Salir Monitoreo",
            bg="#7f1d1d", fg="white", relief="flat",
            font=("Segoe UI", 9, "bold"), padx=12, pady=4,
            cursor="hand2",
            command=lambda: _show_pin_unlock(_exit_monitoring_mode)
        ).pack(side="right", padx=8)
        tk.Label(ai_bar, text="MODO MONITOREO ACTIVO",
                 fg="#ef4444", bg="#0b1220",
                 font=("Segoe UI", 8, "bold")).pack(side="right", padx=4)
    else:
        _role_lv = next(
            (u.get("role", "") for u in users_data if u["user"] == current_user), ""
        )
        _is_op_lv = (
            current_user in ("admin", "ROOT_USB")
            or _role_lv in ("Administrador", "Operador")
        )
        if _is_op_lv:
            mon_fs_var = tk.BooleanVar(value=False)

            def _launch_mon():
                _start_monitoring_mode(fullscreen=mon_fs_var.get())

            tk.Checkbutton(
                ai_bar, text="Pantalla completa",
                variable=mon_fs_var,
                fg="#64748b", bg="#0b1220", selectcolor="#1e3a5f",
                activebackground="#0b1220", activeforeground="#94a3b8",
                font=("Segoe UI", 8)
            ).pack(side="right", padx=(4, 10))

            tk.Button(
                ai_bar, text="🔒 Iniciar Monitoreo",
                bg="#1e3a5f", fg="#3b82f6",
                activebackground="#1e3a5f", activeforeground="#60a5fa",
                relief="flat", font=("Segoe UI", 9, "bold"),
                padx=14, pady=6, cursor="hand2",
                command=_launch_mon
            ).pack(side="right", padx=4)

            tk.Frame(ai_bar, bg="#1e293b", width=1).pack(
                side="right", fill="y", pady=6, padx=4
            )

    # ── Grid de cámaras ───────────────────────────────────────────────────────
    # Solo mostrar las cámaras que están tanto en selected_cameras como en cam_data_map.
    # Esto evita que IDs huérfanos o inconsistentes abran feeds inesperados.
    cameras_to_show = [cid for cid in selected_cameras if cid in cam_data_map][:4]

    n = len(cameras_to_show)

    if n == 0:
        tk.Label(container,
                 text="Las cámaras seleccionadas no están disponibles.",
                 fg="#94a3b8", bg="#020617",
                 font=("Segoe UI", 14, "bold")).pack(expand=True)
        return

    # grid layout: 1→1×1  2→1×2  3-4→2×2
    if n == 1:
        cols, rows = 1, 1
    elif n == 2:
        cols, rows = 2, 1
    else:
        cols, rows = 2, 2

    grid = tk.Frame(container, bg="#020617")
    grid.pack(fill="both", expand=True)
    grid.grid_propagate(False)
    for c in range(cols):
        grid.columnconfigure(c, weight=1, uniform="cam_col")
    for r in range(rows):
        grid.rowconfigure(r, weight=1, uniform="cam_row")

    for idx, cam_id in enumerate(cameras_to_show):
        row_idx = idx // cols
        col_idx = idx % cols

        cell = tk.Frame(grid, bg="#000000",
                        highlightbackground="#1e293b",
                        highlightthickness=1)
        cell.grid(row=row_idx, column=col_idx, sticky="nsew", padx=3, pady=3)

        cam_data = cam_data_map.get(cam_id, {})
        cam_label_text = cam_data.get("name", f"Cámara {cam_id}")
        alias = cam_data.get("alias", "")
        if alias:
            cam_label_text = f"{cam_label_text} ({alias})"

        feed = tk.Label(cell, bg="black", width=1, height=1)
        feed.pack(fill="both", expand=True)

        # inicia el feed correcto
        if "rtsp" in cam_data:
            _start_live_rtsp(cam_id, cam_data["rtsp"], feed,
                             cam_data.get("transport", "tcp"),
                             cam_name=cam_label_text)
        else:
            _start_live_usb(cam_id, feed, cam_name=cam_label_text)

def _start_live_usb(cam_id, label, cam_name=""):
    global _live_cameras
    if not cam_name:
        cam_name = _cam_name_for(cam_id)

    label.configure(text="Conectando...", fg='#64748b', bg='black',
                    font=('Segoe UI', 10))

    def _open_cam():
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  _USB_CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, _USB_CAP_HEIGHT)

        if not cap.isOpened():
            label.after(0, lambda: label.configure(
                text='Sin señal', fg='#ef4444', bg='black',
                font=('Segoe UI', 12)))
            log_event(cam_id, "Cámara Sin Señal", "ALERTA",
                      f"No se pudo abrir cam {cam_id}")
            return

        _live_cameras.append(cap)
        log_event(cam_id, "Cámara Conectada", "INFO", f"Feed USB cam {cam_id} iniciado")
        cid_str       = str(cam_id)
        running       = [True]
        frame_counter = [0]
        _latest_frame = [None]  # buffer compartido entre hilo lector y hilo UI

        # ── Hilo lector: cap.read() NUNCA en el hilo principal de Tkinter ────
        def _reader():
            while running[0]:
                ret, frm = cap.read()
                if ret and frm is not None:
                    frame_counter[0] += 1
                    _t_now = time.time()
                    _cam_last_frame[cid_str] = _t_now
                    _perf_cap_ts[cid_str].append(_t_now)
                    _latest_frame[0] = frm
                    _cam_preview_frame[cid_str] = frm  # dashboard preview
                    # ── Grabación en hilo lector (no bloquea la UI) ──────────
                    _t_rec_usb = time.time()
                    if _recording_enabled.get(cid_str):
                        try:
                            if _throttle(cid_str, "grab_status_usb", 15):
                                print(f"[GRAB] Estado USB        | cam={cam_id} | enabled=True | modo={_cam_record_mode.get(cid_str, _record_mode)} | writer={'OK' if _ai_writers.get(cid_str) else 'NONE'}")
                            _ts_clean = _burn_timestamp(frm, cam_name)
                            if _cam_record_mode.get(cid_str, _record_mode) == 'continuous':
                                if _ai_started.get(cid_str) and _ai_writers.get(cid_str) is None:
                                    _start_ai_writer(cid_str, frm)
                            _write_frame(cid_str, _ts_clean)
                            _ai_lck = _ai_lock.get(cid_str)
                            _ai_f = None
                            if _ai_lck:
                                with _ai_lck:
                                    _ai_f = _ai_frame_buffer.get(cid_str)
                            if _ai_started.get(cid_str) and _ai_f is not None:
                                _write_ai_frame(cid_str, _burn_timestamp(_ai_f, cam_name))
                            if _event_active.get(cid_str):
                                _event_active[cid_str] = False
                        except Exception as _rec_exc:
                            print(f"[GRAB] ERROR USB         | cam={cam_id} | {_rec_exc}")
                    _perf_t_record[cid_str].append(time.time() - _t_rec_usb)
                else:
                    time.sleep(0.01)

        threading.Thread(target=_reader, daemon=True).start()

        def update():
            if not running[0]:
                return
            if not label.winfo_exists():
                running[0] = False
                return

            frame = _latest_frame[0]
            if frame is not None:
                raw_frame = frame  # clean frame — reference before AI

                # ── IA activa ──────────────────────────────────────────────────
                if _ai_started.get(cid_str):

                    if not _ai_thread_active.get(cid_str):
                        _ai_thread_active[cid_str] = True
                        threading.Thread(
                            target=_ai_worker, args=(cid_str,), daemon=True).start()

                    if _ai_raw_buffer.get(cid_str) is None and frame_counter[0] % 2 == 0:
                        _ai_raw_buffer[cid_str] = frame.copy()
                    elif frame_counter[0] % 2 == 0:
                        _perf_drops[cid_str] += 1  # buffer ocupado — frame de IA descartado

                    lock = _ai_lock.get(cid_str)
                    if lock:
                        with lock:
                            processed = _ai_frame_buffer.get(cid_str)
                        if processed is not None:
                            try:
                                _ai_age = time.time() - _ai_frame_buffer_ts.get(cid_str, 0)
                                if _ai_age < 2.0 and processed.shape == raw_frame.shape:
                                    frame = processed
                            except Exception:
                                pass

                # ── Selección de frame para display ───────────────────────────
                display_frame = raw_frame
                if _display_mode[0] == "ia" and _ai_started.get(cid_str):
                    display_frame = frame

                # ── Overlay + Renderizado ─────────────────────────────────────
                # Resize primero (frame pequeño), luego overlay y cvtColor — evita
                # copias a resolución completa en el hilo UI cada 40ms.
                try:
                    _t_ren = time.time()
                    online = (time.time() - _cam_last_frame.get(cid_str, 0)) < 5.0
                    pw = label.winfo_width()  or 640
                    ph = label.winfo_height() or 480
                    display_small = cv2.resize(display_frame, (pw, ph)) if pw > 4 and ph > 4 else display_frame
                    display_small = _draw_cam_overlay(display_small, cam_name, online, cam_id=cid_str)
                    frame_rgb = cv2.cvtColor(display_small, cv2.COLOR_BGR2RGB)
                    img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                    label.imgtk = img
                    label.configure(image=img)
                    _perf_t_render[cid_str].append(time.time() - _t_ren)
                except Exception:
                    pass

            if running[0] and label.winfo_exists():
                label.after(40, update)

        _live_after_ids.append((running, cap))
        label.after(100, update)  # 100 ms para que el lector capture el primer frame

    threading.Thread(target=_open_cam, daemon=True).start()

def _start_live_rtsp(cam_id, rtsp_url, label, transport='tcp', cam_name=""):
    """Feed RTSP multi-cámara usando ffmpeg como backend (nunca bloquea el hilo UI)."""
    if not cam_name:
        cam_name = _cam_name_for(cam_id)
    cid_str       = str(cam_id)
    running       = [True]
    frame_counter = [0]
    _latest_frame = [None]

    # Resolución fija para decodificar; el render loop reescala al tamaño real del panel
    _FW, _FH = 640, 360

    label.configure(text="Conectando RTSP...", fg='#64748b', bg='black',
                    font=('Segoe UI', 10))
    log_event(cam_id, "RTSP Iniciando", "INFO",
              f"URL: {rtsp_url[:40]}  transport={transport}")

    # ── Hilo lector: ffmpeg → buffer de frames ────────────────────────────────
    def _reader():
        proc       = None
        frame_size = _FW * _FH * 3

        while running[0]:
            # Abrir / re-abrir ffmpeg
            if proc is None or proc.poll() is not None:
                if proc is not None:
                    try:
                        proc.terminate(); proc.wait(timeout=2)
                    except Exception:
                        pass
                    proc = None
                if not running[0]:
                    break

                cmd = [
                    FFMPEG_BIN,
                    '-rtsp_transport', transport,
                    '-fflags', 'nobuffer',
                    '-flags', 'low_delay',
                    '-probesize', '500000',
                    '-analyzeduration', '500000',
                    '-i', rtsp_url,
                    '-an', '-sn', '-dn',
                    '-vf', f'scale={_FW}:{_FH}',
                    '-f', 'image2pipe',
                    '-pix_fmt', 'bgr24',
                    '-vcodec', 'rawvideo',
                    '-r', '15',
                    '-'
                ]
                print(f"[CAM-{cam_id}] URL       : {rtsp_url}")
                print(f"[CAM-{cam_id}] TRANSPORTE: {transport}")
                print(f"[CAM-{cam_id}] COMANDO   : {' '.join(cmd)}")
                _t_launch = time.time()
                try:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        bufsize=10**7
                    )
                    log_event(cam_id, "RTSP Conectada", "INFO",
                              f"ffmpeg lanzado (transport={transport})")

                    def _print_stderr_live(p=proc, cid=cam_id, t0=_t_launch):
                        try:
                            for line in p.stderr:
                                decoded = line.decode('utf-8', errors='replace').rstrip()
                                if decoded:
                                    print(f"[CAM-{cid}-STDERR] {decoded}")
                        except Exception:
                            pass
                        elapsed = time.time() - t0
                        rc = p.returncode
                        print(f"[CAM-{cid}] ffmpeg terminó: rc={rc}  elapsed={elapsed:.2f}s")

                    threading.Thread(target=_print_stderr_live, daemon=True).start()

                except Exception as exc:
                    print(f"[CAM-{cam_id}] ERROR al lanzar ffmpeg: {exc}")
                    log_event(cam_id, "FFMPEG Error RTSP", "ALERTA", str(exc))
                    time.sleep(5)
                    continue

            # Leer frame crudo del pipe de ffmpeg
            try:
                raw = proc.stdout.read(frame_size)
                if len(raw) != frame_size:
                    # conexión perdida → reconectar
                    log_event(cam_id, "RTSP Señal Perdida", "WARNING",
                              "ffmpeg cerró el pipe — reconectando en 3 s")
                    proc = None
                    time.sleep(3)
                    continue
                frm = np.frombuffer(raw, np.uint8).reshape((_FH, _FW, 3))
                frame_counter[0] += 1
                _t_now = time.time()
                _cam_last_frame[cid_str] = _t_now
                _perf_cap_ts[cid_str].append(_t_now)
                _latest_frame[0] = frm.copy()
                _cam_preview_frame[cid_str] = frm  # dashboard preview
                # ── Grabación en hilo lector (no bloquea la UI) ──────────
                _t_rec_rtsp = time.time()
                if _recording_enabled.get(cid_str):
                    if _throttle(cid_str, "grab_status_rtsp", 15):
                        print(f"[GRAB] Estado RTSP       | cam={cam_id} | enabled=True | modo={_cam_record_mode.get(cid_str, _record_mode)} | writer={'OK' if _ai_writers.get(cid_str) else 'NONE'}")
                    _frm_rec = _latest_frame[0]  # usar copia ya hecha
                    _ts_clean = _burn_timestamp(_frm_rec, cam_name)
                    if _cam_record_mode.get(cid_str, _record_mode) == 'continuous':
                        if _ai_started.get(cid_str) and _ai_writers.get(cid_str) is None:
                            _start_ai_writer(cid_str, _frm_rec)
                    _write_frame(cid_str, _ts_clean)
                    _ai_lck = _ai_lock.get(cid_str)
                    _ai_f = None
                    if _ai_lck:
                        with _ai_lck:
                            _ai_f = _ai_frame_buffer.get(cid_str)
                    if _ai_started.get(cid_str) and _ai_f is not None:
                        _write_ai_frame(cid_str, _burn_timestamp(_ai_f, cam_name))
                    if _event_active.get(cid_str):
                        _event_active[cid_str] = False
                _perf_t_record[cid_str].append(time.time() - _t_rec_rtsp)
            except Exception:
                proc = None
                time.sleep(1)

        # Limpieza al detener
        if proc is not None:
            try:
                proc.terminate(); proc.wait(timeout=2)
            except Exception:
                pass

    threading.Thread(target=_reader, daemon=True).start()

    # Para stop_all_live: registrar running; cap=None porque ffmpeg no usa VideoCapture
    _live_after_ids.append((running, None))

    # ── Loop de render en el hilo UI (sin bloqueos) ───────────────────────────
    def update():
        if not running[0]:
            return
        if not label.winfo_exists():
            running[0] = False
            return

        frame = _latest_frame[0]
        if frame is not None:
            raw_frame = frame  # clean frame — reference before AI

            # ── IA activa ──────────────────────────────────────────────────
            if _ai_started.get(cid_str):

                if not _ai_thread_active.get(cid_str):
                    _ai_thread_active[cid_str] = True
                    threading.Thread(
                        target=_ai_worker, args=(cid_str,), daemon=True).start()

                if _ai_raw_buffer.get(cid_str) is None and frame_counter[0] % 2 == 0:
                    _ai_raw_buffer[cid_str] = frame.copy()
                elif frame_counter[0] % 2 == 0:
                    _perf_drops[cid_str] += 1  # buffer ocupado — frame de IA descartado

                lock = _ai_lock.get(cid_str)
                if lock:
                    with lock:
                        processed = _ai_frame_buffer.get(cid_str)
                    if processed is not None:
                        try:
                            _ai_age = time.time() - _ai_frame_buffer_ts.get(cid_str, 0)
                            if _ai_age < 2.0 and processed.shape == raw_frame.shape:
                                frame = processed
                        except Exception:
                            pass

            # ── Selección de frame para display ───────────────────────────
            display_frame = raw_frame
            if _display_mode[0] == "ia" and _ai_started.get(cid_str):
                display_frame = frame

            # ── Overlay + Renderizado ─────────────────────────────────────
            # Resize primero (frame pequeño), luego overlay y cvtColor.
            try:
                _t_ren = time.time()
                online = (time.time() - _cam_last_frame.get(cid_str, 0)) < 5.0
                pw = label.winfo_width()  or _FW
                ph = label.winfo_height() or _FH
                display_small = cv2.resize(display_frame, (pw, ph)) if pw > 4 and ph > 4 else display_frame
                display_small = _draw_cam_overlay(display_small, cam_name, online, cam_id=cid_str)
                frame_rgb = cv2.cvtColor(display_small, cv2.COLOR_BGR2RGB)
                img = ImageTk.PhotoImage(Image.fromarray(frame_rgb))
                label.imgtk = img
                label.configure(image=img)
                _perf_t_render[cid_str].append(time.time() - _t_ren)
            except Exception:
                pass

        if running[0] and label.winfo_exists():
            label.after(40, update)

    label.after(300, update)  # 300 ms para que ffmpeg inicie y produzca el primer frame

def _update_events_badge():
    lbl = _events_badge_lbl[0]
    if lbl is None:
        return
    try:
        if not lbl.winfo_exists():
            return
    except Exception:
        return
    count = sum(
        1 for e in _events_list
        if any(kw in e.get("type", "") for kw in INCIDENT_KEYWORDS)
    )
    if count > 0:
        lbl.config(text=f"Eventos  ({count})", fg="#f97316")
    else:
        lbl.config(text="Eventos", fg="#cbd5f5")


# ── MONITOREO PROFESIONAL ────────────────────────────────────────────────────

def _get_user_pin(username):
    """Devuelve el hash de PIN almacenado; si no existe retorna hash('1234')."""
    for u in users_data:
        if u["user"] == username:
            return u.get("pin", hash_password("1234"))
    return hash_password("1234")

def _show_pin_unlock(on_success):
    """Modal de ingreso de PIN. Llama on_success() si el PIN es correcto."""
    popup = tk.Toplevel()
    popup.title("MODO MONITOREO — Desbloqueo")
    popup.configure(bg="#020617")
    popup.geometry("360x300")
    popup.resizable(False, False)
    popup.grab_set()
    popup.focus_force()

    tk.Label(popup, text="MODO MONITOREO",
             fg="#3b82f6", bg="#020617",
             font=("Segoe UI", 14, "bold")).pack(pady=(28, 4))
    tk.Label(popup, text="Ingrese PIN para desbloquear",
             fg="#9DB2D4", bg="#020617",
             font=("Segoe UI", 10)).pack(pady=(0, 20))

    pin_var = tk.StringVar()
    pin_entry = tk.Entry(popup, textvariable=pin_var, show="*",
                         bg="#0b1220", fg="white", insertbackground="white",
                         relief="flat", font=("Segoe UI", 18), width=10,
                         justify="center")
    pin_entry.pack(ipady=10, pady=(0, 6))
    pin_entry.focus()

    err_lbl = tk.Label(popup, text="", fg="#ef4444", bg="#020617",
                       font=("Segoe UI", 9))
    err_lbl.pack(pady=(0, 14))

    _attempts = [0]

    def _check():
        entered = pin_var.get().strip()
        if not entered:
            return
        hashed = hash_password(entered)
        valid = False
        if hashed == _get_user_pin(current_user or ""):
            valid = True
        if not valid:
            for u in users_data:
                if u.get("role") == "Administrador":
                    if hashed == u.get("pin", hash_password("1234")):
                        valid = True
                        break
        if valid:
            popup.destroy()
            on_success()
        else:
            _attempts[0] += 1
            err_lbl.configure(text=f"PIN incorrecto — intento {_attempts[0]}")
            register_event(current_user or "?", "PIN_FALLIDO", "FAILED",
                           f"Intento de desbloqueo de monitoreo #{_attempts[0]}")
            pin_var.set("")

    pin_entry.bind("<Return>", lambda _e: _check())

    btns = tk.Frame(popup, bg="#020617")
    btns.pack()
    tk.Button(btns, text="Desbloquear", bg="#3b82f6", fg="white",
              relief="flat", font=("Segoe UI", 10, "bold"),
              padx=26, pady=8, cursor="hand2",
              command=_check).pack(side="left", padx=(0, 8))
    tk.Button(btns, text="Cancelar", bg="#1e293b", fg="#9DB2D4",
              relief="flat", font=("Segoe UI", 9),
              padx=16, pady=8, cursor="hand2",
              command=popup.destroy).pack(side="left")


def _start_monitoring_mode(fullscreen=False):
    """Activa modo monitoreo: oculta sidebar y abre Vista en Vivo."""
    _monitoring_mode[0] = True
    sidebar.pack_forget()
    if fullscreen:
        try:
            root.attributes("-fullscreen", True)
        except Exception:
            pass
    show_live_view()


def _exit_monitoring_mode():
    """Restaura la UI normal tras desbloqueo por PIN."""
    _monitoring_mode[0] = False
    try:
        root.attributes("-fullscreen", False)
    except Exception:
        pass
    sidebar.pack(side="left", fill="y", before=main)
    build_sidebar()
    show_live_view()


def build_sidebar():
    # LIMPIAR SIDEBAR
    for widget in sidebar.winfo_children():
        if widget != header:
            widget.destroy()

    # En modo monitoreo solo mostramos el botón de salida (el sidebar queda oculto,
    # pero build_sidebar() puede llamarse igualmente — no construir nada extra)
    if _monitoring_mode[0]:
        return

    # Detectar rol del usuario actual
    _role = ""
    for u in users_data:
        if u["user"] == current_user:
            _role = u.get("role", "")
            break
    is_admin    = current_user in ("admin", "ROOT_USB") or _role == "Administrador"
    is_operator = _role in ("Operador",) or is_admin
    is_viewer   = _role == "Visualizador"

    def _sep():
        tk.Frame(sidebar, height=1, bg="#0f172a").pack(fill="x", padx=10, pady=4)

    # ── Operación (todos los roles) ───────────────────────────────────────────
    create_button("Inicio", show_inicio)
    create_button("Vista en Vivo", show_live_view)

    _sep()

    # ── Cámaras + Configuración IA (admin y operador) ─────────────────────────
    if is_operator:
        create_button("Cámaras", show_cameras)

    # ── Registros y análisis ──────────────────────────────────────────────────
    _, _ev_lbl = create_button("Eventos", show_events)
    _events_badge_lbl[0] = _ev_lbl
    _update_events_badge()

    create_button("Reproducción", show_biblioteca)

    if is_operator:
        create_button("Registro", show_registro)

    if is_operator:
        create_button("Reportes", show_reportes)

    if not is_viewer:
        create_button("Auditoría", show_audit)

    # ── Administración (solo admin) ───────────────────────────────────────────
    if is_admin:
        _sep()
        create_button("Usuarios", show_users)
        create_button("Turnos", show_turnos)
        create_button("Configuración", show_settings)
        create_button("Diagnóstico", show_diagnostics)

    _sep()

    create_button("Salir", logout)

# =========================
# SIDEBAR
# =========================
build_sidebar()

# =========================
# LOCK SCREEN
# =========================
lock_screen = tk.Frame(root, bg="#020617")
lock_screen.place(relwidth=1, relheight=1)

center = tk.Frame(lock_screen, bg="#020617")
center.place(relx=0.5, rely=0.5, anchor="center")

tk.Label(center, image=big_logo, bg="#020617").pack(pady=(0, 20))

tk.Label(center, text="VIGILANT PRO",
         fg="#e5e7eb", bg="#020617",
         font=("Segoe UI", 20, "bold")).pack()

tk.Label(center, text="Sistema de vigilancia",
         fg="#64748b", bg="#020617",
         font=("Segoe UI", 10)).pack(pady=(0, 20))

tk.Label(center, text="🔒",
         fg="#3b82f6", bg="#020617",
         font=("Segoe UI", 22)).pack()

tk.Label(center, text="Aplicación bloqueada",
         fg="#e5e7eb", bg="#020617",
         font=("Segoe UI", 14, "bold")).pack(pady=(10, 5))

tk.Label(center, text="Ingrese sus credenciales para continuar",
         fg="#64748b", bg="#020617",
         font=("Segoe UI", 10)).pack(pady=(0, 25))

def set_placeholder(entry, text, is_password=False):
    entry.insert(0, text)
    entry.config(fg="#64748b")

    def on_focus_in(e):
        if entry.get() == text:
            entry.delete(0, "end")
            entry.config(fg="#e5e7eb")
            if is_password:
                entry.config(show="*")

    def on_focus_out(e):
        if entry.get() == "":
            entry.insert(0, text)
            entry.config(fg="#64748b")
            if is_password:
                entry.config(show="")

    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)

user_entry = tk.Entry(center,
                      font=("Segoe UI", 12),
                      bg="#0b1220", fg="#e5e7eb",
                      insertbackground="white",
                      relief="flat",
                      width=30)
user_entry.pack(ipady=10, pady=6)
set_placeholder(user_entry, "Usuario")

password_entry = tk.Entry(center,
                          font=("Segoe UI", 12),
                          bg="#0b1220", fg="#e5e7eb",
                          insertbackground="white",
                          relief="flat",
                          width=30)
password_entry.pack(ipady=10, pady=6)
set_placeholder(password_entry, "Contraseña", is_password=True)

def unlock():
    global current_user
    user = user_entry.get()
    password = password_entry.get()

    if user in ["", "Usuario"] or password in ["", "Contraseña"]:
        return

    if authenticate(user, password):
        current_user = user
        build_sidebar()
        lock_screen.place_forget()

        register_event(
            user,
            "LOGIN",
            "OK",
            "Inicio de sesión exitoso"
        )
        global session_start_time
        session_start_time = datetime.now()
        show_inicio()
    else:

        register_event(
            user,
            "LOGIN",
            "FAILED",
            "Contraseña incorrecta"
        )

        show_notification(
            "LOGIN ERROR",
            "Credenciales incorrectas",
            "#ef4444"
        )

tk.Button(center, text="Desbloquear",
          command=unlock,
          bg="#3b82f6", fg="white",
          font=("Segoe UI", 11, "bold"),
          padx=50, pady=12,
          relief="flat",
          cursor="hand2").pack(pady=25)
tk.Button(
    center,
    text="USB Root Recovery",
    command=login_with_root_usb,
    bg="#050816",
    fg="#ef4444",
    activebackground="#050816",
    activeforeground="#ff6666",
    font=("Segoe UI", 9),
    relief="flat",
    borderwidth=0,
    cursor="hand2"
).pack(pady=(5, 0))

tk.Label(center, text="🔒  Acceso no autorizado será registrado",
         fg="#ef4444", bg="#020617",
         font=("Segoe UI", 9)).pack(pady=(10, 0))

# =========================
# RUN
# =========================
def _process_notification_queue():
    try:
        while _notification_queue:
            title, message, color = _notification_queue.popleft()
            show_notification(title, message, color)
    except Exception:
        pass
    root.after(200, _process_notification_queue)

root.after(200, _process_notification_queue)

# ── F12: atajo de emergencia — abre desbloqueo PIN en cualquier estado ────────
def _f12_handler(event=None):
    if _monitoring_mode[0]:
        _show_pin_unlock(_exit_monitoring_mode)
    elif current_user:
        # Fuera de monitoreo, F12 muestra el dashboard ejecutivo
        show_inicio()

root.bind("<F12>", _f12_handler)

# Start shift monitor (fires every 60 s)
root.after(60_000, _update_shift_display)

root.mainloop()