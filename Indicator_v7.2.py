# -*- coding: utf-8 -*-
# Forzar encoding UTF-8 - FIX para Windows + PyInstaller
import sys
import io
if sys.stdout and hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

"""
GENERADOR DE INDICES ESPECTRALES v7.2
- Lector TIFF universal (ImageJ, DJI pixel-interleaved, multibanda normal)
- Procesamiento por bloques (memoria optimizada)
- Selector manual de bandas opcional
- Correcciones matematicas (MSAVI2, GEMI)
- Modulo de edicion multiespectral (Crop + Rotacion)
- Calibracion radiometrica opcional
- Fuentes cientificas de indices
- Thread-safe GUI
- PestaÃ±a ParÃ¡metros con ROIs: copiar/pegar continuo, matriz lineal
- Herramienta de mediciÃ³n de pÃ­xeles con soporte metros (georef)
"""
import sys
import os
import threading
import time
import tempfile
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.windows import Window
from rasterio.transform import Affine

try:
    from scipy.ndimage import rotate as scipy_rotate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ==================== LECTOR TIFF UNIVERSAL ====================

def load_multispectral(path, manual_bands=None):
    """
    Lector TIFF universal que maneja:
    - Caso A: ImageJ multi-directorio
    - Caso B: Pixel-interleaved (DJI)
    - Caso C: Multibanda normal
    - Caso D: Bandas manuales (archivos separados)
    
    Args:
        path: Ruta al archivo TIFF (o None si manual_bands)
        manual_bands: Dict con rutas {"Green": path, "Red": path, ...}
    
    Returns:
        tuple: (numpy array [bands, H, W], profile dict, temp_file_path o None)
    """
    temp_path = None
    
    # Caso D: Bandas manuales desde archivos separados
    if manual_bands is not None:
        bands_data = []
        profile = None
        band_order = ["Green", "Red", "RedEdge", "NIR"]
        
        for band_name in band_order:
            band_path = manual_bands.get(band_name)
            if not band_path or not os.path.exists(band_path):
                raise ValueError(f"Banda {band_name} no especificada o archivo no existe")
            
            with rasterio.open(band_path) as src:
                bands_data.append(src.read(1))
                if profile is None:
                    profile = src.profile.copy()
        
        return np.stack(bands_data), profile, None
    
    with rasterio.open(path) as src:
        tags = src.tags()
        desc = tags.get("TIFFTAG_IMAGEDESCRIPTION", "")
        
        print(f"\n[LOAD] Analizando: {os.path.basename(path)}")
        print(f"   Bandas detectadas: {src.count}")
        print(f"   Interleave: {src.profile.get('interleave', 'N/A')}")
        print(f"   Dtype: {src.profile.get('dtype', 'N/A')}")
        
        # Caso A: ImageJ multi-directorio
        if "ImageJ" in desc and "images=" in desc:
            print(f"   [TIPO] ImageJ multi-directorio detectado")
            try:
                # Extraer nÃºmero de imÃ¡genes del tag
                num_images = int(desc.split("images=")[1].split()[0])
                print(f"   Subarrays: {num_images}")
                
                bands = []
                last_profile = None
                
                # GDAL usa Ã­ndice 1-based para GTIFF_DIR
                for i in range(1, num_images + 1):
                    gdal_path = f"GTIFF_DIR:{i}:{path}"
                    with rasterio.open(gdal_path) as ds:
                        bands.append(ds.read(1))
                        last_profile = ds.profile.copy()
                        print(f"     [OK] DIR:{i} - Shape: {bands[-1].shape}")
                
                return np.stack(bands), last_profile, None
                
            except Exception as e:
                print(f"   [WARNING] Fallo ImageJ, intentando mÃ©todo alternativo: {e}")
        
        # Caso B: Pixel-interleaved (DJI y otros)
        if src.count == 1 and src.profile.get("interleave") == "pixel":
            print(f"   [TIPO] Pixel-interleaved detectado, convirtiendo...")
            
            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmp.close()
            temp_path = tmp.name
            
            rio_copy(path, temp_path, interleave="band")
            
            with rasterio.open(temp_path) as ds:
                print(f"   [OK] Convertido a {ds.count} bandas")
                return ds.read(), ds.profile.copy(), temp_path
        
        # Caso C: Multibanda normal
        if src.count >= 4:
            print(f"   [TIPO] Multibanda normal ({src.count} bandas)")
            return src.read(), src.profile.copy(), None
        
        # Si llegamos aquÃ­, intentar conversiÃ³n genÃ©rica
        print(f"   [TIPO] Intentando conversiÃ³n genÃ©rica...")
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.close()
        temp_path = tmp.name
        
        try:
            rio_copy(path, temp_path, interleave="band")
            with rasterio.open(temp_path) as ds:
                if ds.count >= 4:
                    print(f"   [OK] ConversiÃ³n exitosa: {ds.count} bandas")
                    return ds.read(), ds.profile.copy(), temp_path
                else:
                    raise ValueError(f"Archivo tiene {ds.count} bandas, se requieren 4")
        except Exception as e:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
            raise ValueError(f"No se pudo leer el archivo como multibanda: {e}")


def load_any_tiff(path):
    """
    Lector TIFF flexible para edicion: acepta cualquier numero de bandas.
    Soporta ImageJ multi-directorio, pixel-interleaved y multibanda normal.
    
    Args:
        path: Ruta al archivo TIFF
    
    Returns:
        tuple: (numpy array [bands, H, W], profile dict, temp_file_path o None)
    """
    temp_path = None
    
    with rasterio.open(path) as src:
        tags = src.tags()
        desc = tags.get("TIFFTAG_IMAGEDESCRIPTION", "")
        
        print(f"\n[LOAD-EDIT] Analizando: {os.path.basename(path)}")
        print(f"   Bandas detectadas: {src.count}")
        print(f"   Interleave: {src.profile.get('interleave', 'N/A')}")
        print(f"   Dtype: {src.profile.get('dtype', 'N/A')}")
        
        # Caso A: ImageJ multi-directorio
        if "ImageJ" in desc and "images=" in desc:
            print(f"   [TIPO] ImageJ multi-directorio detectado")
            try:
                num_images = int(desc.split("images=")[1].split()[0])
                print(f"   Subarrays: {num_images}")
                
                bands = []
                last_profile = None
                
                for i in range(1, num_images + 1):
                    gdal_path = f"GTIFF_DIR:{i}:{path}"
                    with rasterio.open(gdal_path) as ds:
                        bands.append(ds.read(1))
                        last_profile = ds.profile.copy()
                        print(f"     [OK] DIR:{i} - Shape: {bands[-1].shape}")
                
                return np.stack(bands), last_profile, None
                
            except Exception as e:
                print(f"   [WARNING] Fallo ImageJ, intentando metodo alternativo: {e}")
        
        # Caso B: Pixel-interleaved
        if src.count == 1 and src.profile.get("interleave") == "pixel":
            print(f"   [TIPO] Pixel-interleaved detectado, convirtiendo...")
            
            tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
            tmp.close()
            temp_path = tmp.name
            
            rio_copy(path, temp_path, interleave="band")
            
            with rasterio.open(temp_path) as ds:
                print(f"   [OK] Convertido a {ds.count} bandas")
                return ds.read(), ds.profile.copy(), temp_path
        
        # Caso C: Cualquier numero de bandas (sin restriccion)
        print(f"   [TIPO] Archivo con {src.count} banda(s)")
        return src.read(), src.profile.copy(), None


# ==================== M\u00d3DULO DE EDICI\u00d3N MULTIESPECTRAL ====================

@dataclass
class EditSession:
    """
    Maneja el estado de edici\u00f3n de una imagen multiespectral.
    Nunca sobrescribe el original - siempre trabaja con copias.
    """
    original_data: Optional[np.ndarray] = None      # Shape: (bands, H, W)
    edited_data: Optional[np.ndarray] = None        # Copia de trabajo
    original_profile: Optional[Dict[str, Any]] = None
    edited_profile: Optional[Dict[str, Any]] = None
    source_filepath: Optional[str] = None
    
    # Estado de edici\u00f3n
    is_modified: bool = False
    is_saved: bool = False
    
    # Historial de operaciones
    operations: List[str] = field(default_factory=list)
    
    # ROI de recorte actual
    crop_roi: Optional[Tuple[int, int, int, int]] = None
    
    # Rotaci\u00f3n acumulada
    rotation_angle: float = 0.0
    
    # Cache multinivel para zoom r\u00e1pido
    _normalized_base: Optional[np.ndarray] = field(default=None, repr=False)
    _preview_cache: Dict = field(default_factory=dict, repr=False)
    _cache_max_entries: int = field(default=10, repr=False)  # M\u00e1s entradas LRU para zoom r\u00e1pido
    
    def load(self, filepath: str) -> bool:
        """Carga un archivo GeoTIFF en la sesi\u00f3n de edici\u00f3n (cualquier numero de bandas)."""
        try:
            data, profile, temp_path = load_any_tiff(filepath)
            
            self.original_data = data.copy()
            self.edited_data = data.copy()
            self.original_profile = profile.copy()
            self.edited_profile = profile.copy()
            self.source_filepath = filepath
            
            # Reset estado
            self.is_modified = False
            self.is_saved = False
            self.operations = []
            self.crop_roi = None
            self.rotation_angle = 0.0
            
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            return True
        except Exception as e:
            print(f"[EditSession] Error cargando archivo: {e}")
            return False
    
    def get_dimensions(self) -> Tuple[int, int, int]:
        """Retorna (bands, height, width) de los datos editados."""
        if self.edited_data is None:
            return (0, 0, 0)
        return self.edited_data.shape
    
    def get_original_dimensions(self) -> Tuple[int, int, int]:
        """Retorna (bands, height, width) de los datos originales."""
        if self.original_data is None:
            return (0, 0, 0)
        return self.original_data.shape
    
    def apply_crop(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Aplica recorte a todas las bandas de forma consistente."""
        if self.edited_data is None:
            return False
        
        bands, h, w = self.edited_data.shape
        
        # Clamping a l\u00edmites v\u00e1lidos
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            print(f"[EditSession] ROI demasiado peque\\u00f1o: {x2-x1}x{y2-y1}")
            return False
        
        # Slicing a todas las bandas
        self.edited_data = self.edited_data[:, y1:y2, x1:x2].copy()
        
        # Actualizar profile
        self.edited_profile['height'] = y2 - y1
        self.edited_profile['width'] = x2 - x1
        
        # Actualizar transform (georeferencia)
        if 'transform' in self.edited_profile and self.edited_profile['transform']:
            old_transform = self.edited_profile['transform']
            new_transform = old_transform * Affine.translation(x1, y1)
            self.edited_profile['transform'] = new_transform
        
        self.is_modified = True
        self.crop_roi = (x1, y1, x2, y2)
        self.operations.append(f"Crop: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}px")
        
        print(f"[EditSession] Crop aplicado: {x2-x1}x{y2-y1} p\\u00edxeles")
        return True
    
    def apply_rotation(self, angle: float, discrete_90: bool = False) -> bool:
        """Aplica rotaci\u00f3n a todas las bandas."""
        if self.edited_data is None:
            return False
        
        if angle == 0:
            return True
        
        angle = angle % 360
        
        if discrete_90 and angle in [90, 180, 270]:
            k = int(angle / 90)
            # k negativo para que +angle = horario (intuitivo)
            self.edited_data = np.rot90(self.edited_data, k=-k, axes=(1, 2)).copy()
            
            if angle in [90, 270]:
                h, w = self.edited_profile['height'], self.edited_profile['width']
                self.edited_profile['height'] = w
                self.edited_profile['width'] = h
            
            self.operations.append(f"Rotaci\\u00f3n: {angle}\\u00b0 (discreta)")
        else:
            if not SCIPY_AVAILABLE:
                print("[EditSession] scipy no disponible para rotaci\\u00f3n grado a grado")
                return False
            
            rotated_bands = []
            for band_idx in range(self.edited_data.shape[0]):
                rotated = scipy_rotate(
                    self.edited_data[band_idx],
                    -angle,  # Invertir: +angle = horario (intuitivo para usuario)
                    reshape=True,
                    order=1,
                    mode='constant',
                    cval=0.0
                )
                rotated_bands.append(rotated)
            
            self.edited_data = np.stack(rotated_bands)
            
            _, new_h, new_w = self.edited_data.shape
            self.edited_profile['height'] = new_h
            self.edited_profile['width'] = new_w
            
            self.operations.append(f"Rotaci\\u00f3n: {angle}\\u00b0 (interpolada)")
        
        self.rotation_angle = (self.rotation_angle + angle) % 360
        self.is_modified = True
        
        if angle not in [0, 90, 180, 270]:
            print(f"[EditSession] AVISO: Rotaci\\u00f3n de {angle}\\u00b0 invalida la georeferencia")
            self.edited_profile['transform'] = None
        
        print(f"[EditSession] Rotaci\\u00f3n aplicada: {angle}\\u00b0")
        return True
    
    def reset(self) -> bool:
        """Resetea a los datos originales."""
        if self.original_data is None:
            return False
        
        self.edited_data = self.original_data.copy()
        self.edited_profile = self.original_profile.copy()
        self.is_modified = False
        self.operations = []
        self.crop_roi = None
        self.rotation_angle = 0.0
        
        print("[EditSession] Reset a datos originales")
        return True
    
    def save(self, output_path: str, compress: str = "deflate") -> bool:
        """Guarda los datos editados. NUNCA sobrescribe el original."""
        if self.edited_data is None:
            return False
        
        if output_path == self.source_filepath:
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_editado{ext}"
            print(f"[EditSession] Renombrando para evitar sobrescritura: {output_path}")
        
        try:
            out_profile = self.edited_profile.copy()
            out_profile.update(
                driver='GTiff',
                count=self.edited_data.shape[0],
                height=self.edited_data.shape[1],
                width=self.edited_data.shape[2],
                compress=compress
            )
            
            with rasterio.open(output_path, 'w', **out_profile) as dst:
                dst.write(self.edited_data)
                operations_str = " | ".join(self.operations) if self.operations else "Sin modificaciones"
                dst.update_tags(edit_operations=operations_str)
            
            self.is_saved = True
            print(f"[EditSession] Guardado: {output_path}")
            return True
            
        except Exception as e:
            print(f"[EditSession] Error guardando: {e}")
            return False
    
    def generate_preview(self, max_size: int = 800) -> Optional[Any]:
        """Genera una imagen PIL para preview con cache multinivel."""
        if not PIL_AVAILABLE or self.edited_data is None:
            return None
        
        # Cache nivel 2: preview ya resize-ado para este zoom_level
        if max_size in self._preview_cache:
            return self._preview_cache[max_size].copy()
        
        bands, h, w = self.edited_data.shape
        
        # Cache nivel 1: imagen normalizada a resolución completa
        if self._normalized_base is None:
            nodata = None
            if self.edited_profile and 'nodata' in self.edited_profile:
                nodata = self.edited_profile.get('nodata')
            
            if bands >= 3:
                r_idx = min(2, bands - 1)
                g_idx = min(1, bands - 1)
                b_idx = min(0, bands - 1)
                
                r = self.edited_data[r_idx].astype(np.float32)
                g = self.edited_data[g_idx].astype(np.float32)
                b = self.edited_data[b_idx].astype(np.float32)
            else:
                r = g = b = self.edited_data[0].astype(np.float32)
            
            def normalize(arr):
                if nodata is not None:
                    arr = np.where(arr == nodata, np.nan, arr)
                valid = arr[np.isfinite(arr)]
                if len(valid) == 0:
                    return np.full(arr.shape, 128, dtype=np.uint8)
                p2, p98 = np.nanpercentile(arr, [2, 98])
                if p98 <= p2:
                    return np.full(arr.shape, 128, dtype=np.uint8)
                clipped = np.clip(arr, p2, p98)
                clipped = np.nan_to_num(clipped, nan=p2)
                return ((clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
            
            self._normalized_base = np.stack([normalize(r), normalize(g), normalize(b)], axis=-1)
        
        # Crear PIL Image desde cache nivel 1
        img = Image.fromarray(self._normalized_base, mode='RGB')
        
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Guardar en cache nivel 2 (LRU simple)
        if len(self._preview_cache) >= self._cache_max_entries:
            oldest_key = next(iter(self._preview_cache))
            del self._preview_cache[oldest_key]
        self._preview_cache[max_size] = img.copy()
        
        return img
    
    def invalidate_preview_cache(self):
        """Invalida ambos niveles de cache."""
        self._normalized_base = None
        self._preview_cache.clear()


class EditCanvas(tk.Canvas):
    """Canvas interactivo para edición de imagen con ROI."""
    
    def __init__(self, parent, edit_session: EditSession, on_crop_callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.edit_session = edit_session
        self.on_crop_callback = on_crop_callback
        
        self.photo_image = None
        self.pil_image = None  # PIL image para muestreo de píxeles (cruz guía)
        self.image_id = None
        self.preview_scale = 1.0
        
        self.drawing = False
        self.roi_start = None
        self.roi_rect_id = None
        self.current_roi = None
        
        # IDs de las líneas de la cruz guía
        self._crosshair_h = None
        self._crosshair_v = None
        
        self.bind("<Button-1>", self._on_mouse_down)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.bind("<Button-3>", self._clear_roi)
        self.bind("<Motion>", self._on_mouse_move)
        self.bind("<Leave>", self._on_mouse_leave)
    
    def set_image(self, pil_image, real_width: int, real_height: int):
        """Establece la imagen de preview."""
        if pil_image is None:
            return
        
        self.pil_image = pil_image.convert("RGB")  # Guardar para muestreo de píxeles
        self.photo_image = ImageTk.PhotoImage(pil_image)
        
        preview_w, preview_h = pil_image.size
        self.preview_scale = preview_w / real_width
        
        self.delete("all")
        self.image_id = self.create_image(0, 0, anchor="nw", image=self.photo_image)
        
        # Configurar scrollregion para permitir scroll
        self.configure(scrollregion=(0, 0, preview_w, preview_h))
        
        self.current_roi = None
        self.roi_rect_id = None
        self._crosshair_h = None
        self._crosshair_v = None
    
    def _canvas_to_image_coords(self, cx, cy) -> Tuple[int, int]:
        if self.preview_scale == 0:
            return (0, 0)
        ix = int(cx / self.preview_scale)
        iy = int(cy / self.preview_scale)
        return (ix, iy)
    
    def _on_mouse_down(self, event):
        self.drawing = True
        # Convertir a coordenadas del canvas (considerando scroll)
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)
        self.roi_start = (cx, cy)
        if self.roi_rect_id:
            self.delete(self.roi_rect_id)
            self.roi_rect_id = None
    
    def _on_mouse_drag(self, event):
        if not self.drawing or self.roi_start is None:
            return
        
        x1, y1 = self.roi_start
        # Convertir a coordenadas del canvas (considerando scroll)
        x2 = self.canvasx(event.x)
        y2 = self.canvasy(event.y)
        
        if self.roi_rect_id:
            self.delete(self.roi_rect_id)
        
        self.roi_rect_id = self.create_rectangle(
            x1, y1, x2, y2,
            outline="#00FF00", width=2, dash=(4, 4)
        )
    
    def _on_mouse_up(self, event):
        if not self.drawing:
            return
        
        self.drawing = False
        
        if self.roi_start is None:
            return
        
        x1, y1 = self.roi_start
        # Convertir a coordenadas del canvas (considerando scroll)
        x2 = self.canvasx(event.x)
        y2 = self.canvasy(event.y)
        
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        ix1, iy1 = self._canvas_to_image_coords(x1, y1)
        ix2, iy2 = self._canvas_to_image_coords(x2, y2)
        
        self.current_roi = (ix1, iy1, ix2, iy2)
        
        if self.roi_rect_id:
            self.delete(self.roi_rect_id)
        
        self.roi_rect_id = self.create_rectangle(
            x1, y1, x2, y2,
            outline="#00FF00", width=2
        )
        
        if self.on_crop_callback:
            self.on_crop_callback(self.current_roi)
    
    def _clear_roi(self, event=None):
        if self.roi_rect_id:
            self.delete(self.roi_rect_id)
            self.roi_rect_id = None
        self.current_roi = None
        
        if self.on_crop_callback:
            self.on_crop_callback(None)
    
    def get_current_roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self.current_roi
    
    def _on_mouse_move(self, event):
        """Dibuja cruz guía con color invertido respecto al píxel bajo el cursor."""
        if self.pil_image is None:
            return
        
        # Coordenadas de canvas (considerando scroll)
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)
        
        # Obtener dimensiones del canvas visible y scrollregion
        try:
            sr = self.cget("scrollregion")
            if sr:
                parts = sr.split()
                img_w = int(float(parts[2]))
                img_h = int(float(parts[3]))
            else:
                img_w = self.pil_image.width
                img_h = self.pil_image.height
        except:
            img_w = self.pil_image.width
            img_h = self.pil_image.height
        
        # Obtener color del píxel bajo el cursor y calcular inverso
        try:
            px = int(cx)
            py = int(cy)
            if 0 <= px < self.pil_image.width and 0 <= py < self.pil_image.height:
                r, g, b = self.pil_image.getpixel((px, py))[:3]
            else:
                r, g, b = 128, 128, 128  # Gris si fuera de imagen
            
            # Color invertido
            inv_color = f"#{255-r:02x}{255-g:02x}{255-b:02x}"
        except:
            inv_color = "#FFFFFF"
        
        # Actualizar o crear líneas de la cruz
        if self._crosshair_h is not None:
            self.coords(self._crosshair_h, 0, cy, img_w, cy)
            self.itemconfig(self._crosshair_h, fill=inv_color)
        else:
            self._crosshair_h = self.create_line(0, cy, img_w, cy, fill=inv_color, width=1, tags="crosshair")
        
        if self._crosshair_v is not None:
            self.coords(self._crosshair_v, cx, 0, cx, img_h)
            self.itemconfig(self._crosshair_v, fill=inv_color)
        else:
            self._crosshair_v = self.create_line(cx, 0, cx, img_h, fill=inv_color, width=1, tags="crosshair")
        
        # Asegurar que la cruz esté por encima de la imagen pero debajo del ROI
        if self.roi_rect_id:
            self.tag_raise(self.roi_rect_id)
    
    def _on_mouse_leave(self, event):
        """Oculta la cruz guía cuando el cursor sale del canvas."""
        self.delete("crosshair")
        self._crosshair_h = None
        self._crosshair_v = None


class EdicionTab:
    """Pestana de edicion multiespectral (Crop + Rotacion)."""
    
    def __init__(self, parent: ttk.Frame, main_gui=None):
        self.parent = parent
        self.main_gui = main_gui
        
        self.edit_session = EditSession()
        
        self.rotation_angle = tk.DoubleVar(value=0.0)
        self.roi_info = tk.StringVar(value="Sin ROI definido")
        self.file_info = tk.StringVar(value="Sin archivo cargado")
        self.operations_info = tk.StringVar(value="Sin operaciones")
        self._preview_rotation = 0.0  # Rotacion visual del preview (no aplicada)
        self._zoom_level = 800  # Tamano maximo del preview (zoom)
        self.pending_rotation_info = tk.StringVar(value="")  # Info de rotacion pendiente
        
        # Debounce IDs para evitar renders redundantes
        self._zoom_debounce_id = None
        self._angle_debounce_id = None
        self._last_zoom_text = "100%"
        self._last_angle_value = 0.0
        
        # Cache de preview para evitar regenerar numpy en cada rotacion
        self._cached_preview = None
        self._cached_preview_zoom = None
        
        self._create_ui()
    
    def _create_ui(self):
        main_frame = ttk.Frame(self.parent, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="[Abrir] Cargar TIFF",
                  command=self._load_tiff, width=18).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(toolbar, text="[Reset] Restaurar",
                  command=self._reset_edits, width=15).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(toolbar, text="[Guardar] Exportar",
                  command=self._save_edited, width=18).pack(side=tk.LEFT, padx=5)
        
        # Separador y controles de zoom
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Label(toolbar, text="Zoom:", font=("Arial", 9)).pack(side=tk.LEFT)
        
        ttk.Button(toolbar, text="-", width=3,
                  command=self._zoom_out).pack(side=tk.LEFT, padx=2)
        
        self.zoom_var = tk.StringVar(value="100%")
        self.zoom_entry = ttk.Entry(toolbar, textvariable=self.zoom_var,
                                    width=6, font=("Arial", 9), justify="center")
        self.zoom_entry.pack(side=tk.LEFT)
        self.zoom_entry.bind("<Return>", lambda e: self._on_zoom_entry())
        self.zoom_entry.bind("<FocusIn>", lambda e: self._on_zoom_focus_in())
        self.zoom_entry.bind("<FocusOut>", lambda e: self._on_zoom_focus_out())
        
        ttk.Button(toolbar, text="+", width=3,
                  command=self._zoom_in).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(toolbar, text="Fit", width=4,
                  command=self._zoom_fit).pack(side=tk.LEFT, padx=2)
        
        self.file_label = ttk.Label(toolbar, textvariable=self.file_info,
                                    font=("Arial", 9), foreground="gray")
        self.file_label.pack(side=tk.RIGHT, padx=5)
        
        # Panel principal
        content = ttk.Frame(main_frame)
        content.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo - Controles
        left_panel = ttk.Frame(content, width=250)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Seccion CROP
        crop_frame = ttk.LabelFrame(left_panel, text="Recorte (Crop)", padding=10)
        crop_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(crop_frame, text="Dibuja un rectangulo en la imagen\npara definir el area de recorte.",
                 font=("Arial", 8), foreground="gray").pack(anchor="w")
        
        ttk.Label(crop_frame, textvariable=self.roi_info,
                 font=("Arial", 9, "bold"), foreground="#1f4788").pack(anchor="w", pady=5)
        
        ttk.Button(crop_frame, text="[Aplicar Crop]",
                  command=self._apply_crop).pack(anchor="w", pady=5)
        
        # Seccion ROTACION (tiempo real)
        rot_frame = ttk.LabelFrame(left_panel, text="Rotacion", padding=10)
        rot_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(rot_frame, text="Angulo (grados):").pack(anchor="w")
        
        # Control de angulo con flechas
        angle_frame = ttk.Frame(rot_frame)
        angle_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(angle_frame, text="<", width=3,
                  command=lambda: self._adjust_angle(-1)).pack(side=tk.LEFT)
        ttk.Button(angle_frame, text="<<", width=3,
                  command=lambda: self._adjust_angle(-10)).pack(side=tk.LEFT, padx=2)
        
        self.angle_spinbox = ttk.Spinbox(
            angle_frame, from_=-180, to=180, width=6,
            increment=0.1,
            textvariable=self.rotation_angle,
            command=self._on_angle_change_spinbox
        )
        self.angle_spinbox.pack(side=tk.LEFT, padx=5)
        self.angle_spinbox.bind("<Return>", lambda e: self._on_angle_change_spinbox())
        self.angle_spinbox.bind("<FocusIn>", lambda e: self._on_angle_focus_in())
        self.angle_spinbox.bind("<FocusOut>", lambda e: self._on_angle_focus_out())
        
        ttk.Label(angle_frame, text="grados").pack(side=tk.LEFT)
        
        ttk.Button(angle_frame, text=">", width=3,
                  command=lambda: self._adjust_angle(1)).pack(side=tk.RIGHT)
        ttk.Button(angle_frame, text=">>", width=3,
                  command=lambda: self._adjust_angle(10)).pack(side=tk.RIGHT, padx=2)
        
        # Botones rotacion rapida
        quick_rot = ttk.Frame(rot_frame)
        quick_rot.pack(fill=tk.X, pady=5)
        
        for angle in [-90, 90, 180]:
            label = f"{angle}" if angle < 0 else f"+{angle}"
            ttk.Button(quick_rot, text=label,
                      command=lambda a=angle: self._quick_rotate(a),
                      width=5).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(quick_rot, text="0 (Reset)",
                  command=lambda: self._quick_rotate(0),
                  width=8).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(rot_frame, text="(-) = izquierda | (+) = derecha",
                 font=("Arial", 7), foreground="gray").pack(anchor="w")
        
        # BotÃ³n para aplicar rotaciÃ³n a los datos
        apply_rot_frame = ttk.Frame(rot_frame)
        apply_rot_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(apply_rot_frame, text="[Aplicar Rotacion]",
                  command=self._apply_pending_rotation,
                  width=18).pack(side=tk.LEFT)
        
        self.pending_rot_label = ttk.Label(apply_rot_frame, textvariable=self.pending_rotation_info,
                                           font=("Arial", 8), foreground="#cc6600")
        self.pending_rot_label.pack(side=tk.LEFT, padx=5)
        
        # Secci\u00f3n INFO
        info_frame = ttk.LabelFrame(left_panel, text="Estado", padding=10)
        info_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(info_frame, textvariable=self.operations_info,
                 font=("Arial", 8), foreground="#666", wraplength=220).pack(anchor="w")
        
        self.georef_warning = ttk.Label(info_frame, text="",
                                        font=("Arial", 8), foreground="red", wraplength=220)
        self.georef_warning.pack(anchor="w", pady=5)
        
        # Panel derecho - Canvas CON SCROLLBARS
        right_panel = ttk.LabelFrame(content, text="Vista Previa - Dibuja para recortar", padding=5)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        canvas_container = ttk.Frame(right_panel)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        h_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        
        self.edit_canvas = EditCanvas(
            canvas_container,
            self.edit_session,
            on_crop_callback=self._on_roi_update,
            bg="#333333",
            highlightthickness=1,
            highlightbackground="#666666",
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set
        )
        
        # Conectar scrollbars
        h_scroll.config(command=self.edit_canvas.xview)
        v_scroll.config(command=self.edit_canvas.yview)
        
        # Grid layout para scrollbars
        self.edit_canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)
        
        # Binding zoom con rueda del raton (Ctrl+Scroll)
        self.edit_canvas.bind("<Control-MouseWheel>", self._on_mouse_wheel_zoom)
        
        self._show_placeholder()
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(status_frame,
                 text="Click izq: dibujar ROI | Click der: limpiar | Rotacion no multiplo 90 invalida georef",
                 font=("Arial", 8), foreground="#999999").pack(side=tk.LEFT)
    
    def _show_placeholder(self):
        self.edit_canvas.configure(width=600, height=400)
        self.edit_canvas.delete("all")
        self.edit_canvas.create_text(
            300, 200,
            text="[Imagen] Carga un archivo GeoTIFF\npara editar (recortar/rotar)",
            fill="#888888", font=("Arial", 14), justify=tk.CENTER
        )
    
    def _load_tiff(self):
        if self.edit_session.is_modified and not self.edit_session.is_saved:
            if not messagebox.askyesno("Cambios sin guardar",
                                       "Hay cambios sin guardar. Continuar y perderlos?"):
                return
        
        filepath = filedialog.askopenfilename(
            title="Seleccionar GeoTIFF para editar",
            filetypes=[("GeoTIFF files", "*.tif *.TIF *.tiff *.TIFF"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        if self.edit_session.load(filepath):
            self._update_preview()
            self._update_info()
        else:
            messagebox.showerror("Error", "No se pudo cargar el archivo")
    
    def _update_info(self):
        if self.edit_session.source_filepath:
            filename = os.path.basename(self.edit_session.source_filepath)
            bands, h, w = self.edit_session.get_dimensions()
            orig_bands, orig_h, orig_w = self.edit_session.get_original_dimensions()
            self.file_info.set(f"{filename} | Original: {orig_w}x{orig_h} | Actual: {w}x{h} | {bands} bandas")
        else:
            self.file_info.set("Sin archivo cargado")
        
        if self.edit_session.operations:
            self.operations_info.set(" > ".join(self.edit_session.operations))
        else:
            self.operations_info.set("Sin operaciones")
        
        if self.edit_session.rotation_angle not in [0, 90, 180, 270, -90, -180, -270]:
            self.georef_warning.config(text="[!] Georeferencia invalidada por rotacion")
        else:
            self.georef_warning.config(text="")
    
    def _on_roi_update(self, roi):
        if roi:
            x1, y1, x2, y2 = roi
            self.roi_info.set(f"ROI: ({x1},{y1})-({x2},{y2}) = {x2-x1}x{y2-y1}px")
        else:
            self.roi_info.set("Sin ROI definido")
    
    def _on_angle_change_spinbox(self):
        """Callback cuando cambia el spinbox - rotacion visual en tiempo real"""
        # Cancelar debounce pendiente si hay uno
        if self._angle_debounce_id is not None:
            try:
                self.parent.after_cancel(self._angle_debounce_id)
            except (ValueError, tk.TclError):
                pass
            self._angle_debounce_id = None
        try:
            angle = float(self.rotation_angle.get())
            angle = max(-180, min(180, angle))
            self.rotation_angle.set(angle)
            self._preview_rotation = angle
            self._last_angle_value = angle
            self._update_preview_rotated()
            self._update_pending_rotation_info()
        except (ValueError, tk.TclError):
            pass
    
    def _on_angle_focus_in(self):
        """Guarda valor actual al entrar al spinbox"""
        try:
            self._last_angle_value = float(self.rotation_angle.get())
        except (ValueError, tk.TclError):
            self._last_angle_value = 0.0
    
    def _on_angle_focus_out(self):
        """Debounce al perder foco: solo actualiza si cambio el valor"""
        if self._angle_debounce_id is not None:
            try:
                self.parent.after_cancel(self._angle_debounce_id)
            except (ValueError, tk.TclError):
                pass
        self._angle_debounce_id = self.parent.after(150, self._apply_angle_debounced)
    
    def _apply_angle_debounced(self):
        """Aplica cambio de angulo solo si realmente cambio"""
        self._angle_debounce_id = None
        try:
            angle = float(self.rotation_angle.get())
            if angle == self._last_angle_value and angle == self._preview_rotation:
                return  # Sin cambios, no re-renderizar
            self._on_angle_change_spinbox()
        except (ValueError, tk.TclError):
            pass
    
    def _adjust_angle(self, delta: int):
        """Ajusta el angulo por delta grados"""
        # Cancelar debounce pendiente de FocusOut
        if self._angle_debounce_id is not None:
            try:
                self.parent.after_cancel(self._angle_debounce_id)
            except (ValueError, tk.TclError):
                pass
            self._angle_debounce_id = None
        try:
            current = float(self.rotation_angle.get())
        except (ValueError, tk.TclError):
            current = 0
        new_angle = max(-180, min(180, current + delta))
        self.rotation_angle.set(new_angle)
        self._preview_rotation = new_angle
        self._last_angle_value = new_angle
        self._update_preview_rotated()
        self._update_pending_rotation_info()
    
    def _quick_rotate(self, angle: int):
        """Rotacion rapida a angulo especifico"""
        self.rotation_angle.set(angle)
        self._preview_rotation = angle
        self._update_preview_rotated()
        self._update_pending_rotation_info()
    
    def _apply_pending_rotation(self):
        """Aplica la rotacion pendiente a los datos reales"""
        if self._preview_rotation == 0:
            messagebox.showinfo("Info", "No hay rotacion pendiente que aplicar")
            return
        
        angle = self._preview_rotation
        discrete = angle in [90, 180, 270, -90, -180, -270]
        
        if not discrete and not SCIPY_AVAILABLE:
            messagebox.showerror("Error",
                "scipy no esta instalado.\n\n"
                "Para rotacion grado a grado, instala scipy:\n"
                "pip install scipy\n\n"
                "O usa rotaciones de 90, 180 o 270 grados")
            return
        
        if self.edit_session.apply_rotation(angle, discrete_90=discrete):
            self._preview_rotation = 0
            self.rotation_angle.set(0)
            self._update_preview()
            self._update_info()
            self._update_pending_rotation_info()
            messagebox.showinfo("Rotacion aplicada", f"Imagen rotada {angle} grados")
        else:
            messagebox.showerror("Error", "No se pudo aplicar la rotacion")
    
    def _update_pending_rotation_info(self):
        """Actualiza el label de rotacion pendiente"""
        if self._preview_rotation != 0:
            # Mostrar decimal solo si no es entero
            val = self._preview_rotation
            if val == int(val):
                self.pending_rotation_info.set(f"Pendiente: {int(val)} grados")
            else:
                self.pending_rotation_info.set(f"Pendiente: {val:.1f} grados")
        else:
            self.pending_rotation_info.set("")
    
    def _invalidate_preview_cache(self):
        """Invalida el cache de preview (llamar cuando cambian los datos)"""
        self._cached_preview = None
        self._cached_preview_zoom = None
        self.edit_session.invalidate_preview_cache()
    
    def _get_cached_preview(self):
        """Obtiene preview del cache o lo genera si es necesario"""
        if (self._cached_preview is not None and 
            self._cached_preview_zoom == self._zoom_level):
            return self._cached_preview
        
        preview = self.edit_session.generate_preview(self._zoom_level)
        if preview is not None:
            self._cached_preview = preview
            self._cached_preview_zoom = self._zoom_level
        return preview
    
    def _update_preview_rotated(self):
        """Actualiza preview con rotacion visual (sin modificar datos)"""
        if self.edit_session.edited_data is None:
            return
        
        bands, h, w = self.edit_session.get_dimensions()  # Dimensiones REALES
        
        preview = self._get_cached_preview()
        if preview is None:
            return
        
        # Rotar el preview visualmente (+angle = horario, intuitivo)
        if self._preview_rotation != 0 and PIL_AVAILABLE:
            preview = preview.copy()  # No modificar el cache
            preview = preview.rotate(
                -self._preview_rotation,  # Angulo positivo = horario (intuitivo)
                expand=True,
                resample=Image.Resampling.BILINEAR
            )
        
        # Pasar dimensiones REALES para escala correcta
        self.edit_canvas.set_image(preview, w, h)
        self._update_zoom_label()
    
    def _update_preview(self):
        """Actualiza preview normal (sin rotacion visual)"""
        self._invalidate_preview_cache()  # Forzar regeneracion
        preview = self._get_cached_preview()
        if preview is None:
            return
        
        bands, h, w = self.edit_session.get_dimensions()
        self.edit_canvas.set_image(preview, w, h)
        self._update_zoom_label()
    
    def _zoom_in(self):
        """Aumentar zoom"""
        if self._zoom_debounce_id is not None:
            try:
                self.parent.after_cancel(self._zoom_debounce_id)
            except (ValueError, tk.TclError):
                pass
            self._zoom_debounce_id = None
        self._zoom_level = min(3200, int(self._zoom_level * 1.5))
        if self._preview_rotation != 0:
            self._update_preview_rotated()
        else:
            preview = self._get_cached_preview()
            if preview is not None:
                bands, h, w = self.edit_session.get_dimensions()
                self.edit_canvas.set_image(preview, w, h)
                self._update_zoom_label()
    
    def _zoom_out(self):
        """Reducir zoom"""
        if self._zoom_debounce_id is not None:
            try:
                self.parent.after_cancel(self._zoom_debounce_id)
            except (ValueError, tk.TclError):
                pass
            self._zoom_debounce_id = None
        self._zoom_level = max(200, int(self._zoom_level / 1.5))
        if self._preview_rotation != 0:
            self._update_preview_rotated()
        else:
            preview = self._get_cached_preview()
            if preview is not None:
                bands, h, w = self.edit_session.get_dimensions()
                self.edit_canvas.set_image(preview, w, h)
                self._update_zoom_label()
    
    def _zoom_fit(self):
        """Ajustar zoom para ver toda la imagen"""
        if self._zoom_debounce_id is not None:
            try:
                self.parent.after_cancel(self._zoom_debounce_id)
            except (ValueError, tk.TclError):
                pass
            self._zoom_debounce_id = None
        self._zoom_level = 800
        if self._preview_rotation != 0:
            self._update_preview_rotated()
        else:
            preview = self._get_cached_preview()
            if preview is not None:
                bands, h, w = self.edit_session.get_dimensions()
                self.edit_canvas.set_image(preview, w, h)
                self._update_zoom_label()
    
    def _on_mouse_wheel_zoom(self, event):
        """Zoom con Ctrl+rueda del raton (con debounce para rendimiento)"""
        if not hasattr(self, '_wheel_zoom_delta'):
            self._wheel_zoom_delta = 0
            self._wheel_zoom_debounce_id = None
        
        if event.delta > 0:
            self._wheel_zoom_delta += 1
        else:
            self._wheel_zoom_delta -= 1
        
        # Cancelar debounce anterior
        if self._wheel_zoom_debounce_id is not None:
            try:
                self.parent.after_cancel(self._wheel_zoom_debounce_id)
            except (ValueError, tk.TclError):
                pass
        
        # Aplicar después de 80ms de inactividad
        self._wheel_zoom_debounce_id = self.parent.after(80, self._apply_wheel_zoom)
    
    def _apply_wheel_zoom(self):
        """Aplica zoom acumulado por debounce de rueda del mouse."""
        self._wheel_zoom_debounce_id = None
        delta = self._wheel_zoom_delta
        self._wheel_zoom_delta = 0
        
        if delta == 0:
            return
        
        # Aplicar zoom acumulado (cada step = 1.5x)
        factor = 1.5 ** abs(delta)
        if delta > 0:
            self._zoom_level = min(3200, int(self._zoom_level * factor))
        else:
            self._zoom_level = max(200, int(self._zoom_level / factor))
        
        if self._preview_rotation != 0:
            self._update_preview_rotated()
        else:
            preview = self._get_cached_preview()
            if preview is not None:
                bands, h, w = self.edit_session.get_dimensions()
                self.edit_canvas.set_image(preview, w, h)
                self._update_zoom_label()
    
    def _on_zoom_focus_in(self):
        """Guarda el texto actual al entrar al Entry de zoom"""
        self._last_zoom_text = self.zoom_var.get()
    
    def _on_zoom_focus_out(self):
        """Debounce al perder foco: solo actualiza si cambio el texto"""
        if self._zoom_debounce_id is not None:
            try:
                self.parent.after_cancel(self._zoom_debounce_id)
            except (ValueError, tk.TclError):
                pass
        self._zoom_debounce_id = self.parent.after(150, self._apply_zoom_debounced)
    
    def _apply_zoom_debounced(self):
        """Aplica zoom solo si el texto cambio respecto al foco"""
        self._zoom_debounce_id = None
        current_text = self.zoom_var.get()
        if current_text == self._last_zoom_text:
            return  # Sin cambios, no re-renderizar
        self._on_zoom_entry()
    
    def _on_zoom_entry(self):
        """Aplica el zoom escrito manualmente por el usuario"""
        # Cancelar debounce pendiente si hay uno
        if self._zoom_debounce_id is not None:
            try:
                self.parent.after_cancel(self._zoom_debounce_id)
            except (ValueError, tk.TclError):
                pass
            self._zoom_debounce_id = None
        try:
            text = self.zoom_var.get().replace("%", "").strip()
            percent = float(text)
            percent = max(1.0, min(400.0, percent))
            if self.edit_session.edited_data is None:
                return
            bands, h, w = self.edit_session.get_dimensions()
            max_dim = max(w, h)
            if max_dim > 0:
                self._zoom_level = max(50, min(3200, int(percent * max_dim / 100)))
                self._last_zoom_text = self.zoom_var.get()
                if self._preview_rotation != 0:
                    self._update_preview_rotated()
                else:
                    self._update_preview()
        except (ValueError, tk.TclError):
            self._update_zoom_label()

    def _update_zoom_label(self):
        """Actualiza el entry de zoom con el porcentaje actual"""
        if self.edit_session.edited_data is None:
            return
        bands, h, w = self.edit_session.get_dimensions()
        max_dim = max(w, h)
        if max_dim > 0:
            percent = (self._zoom_level / max_dim) * 100
            percent = min(percent, 100.0)
            self.zoom_var.set(f"{percent:.1f}%")
    
    def _apply_crop(self):
        roi = self.edit_canvas.get_current_roi()
        if roi is None:
            messagebox.showwarning("Aviso", "Primero dibuja un rectangulo de recorte")
            return
        
        # Validar que no haya rotaciÃ³n pendiente
        if self._preview_rotation != 0:
            messagebox.showwarning("Rotacion pendiente",
                f"Hay una rotacion de {self._preview_rotation:.0f} grados pendiente.\n\n"
                "El crop se aplica a los datos sin rotar, lo cual\n"
                "produciria un resultado incorrecto.\n\n"
                "Primero aplica la rotacion con [Aplicar Rotacion]\n"
                "y luego redibuja el area de recorte.")
            return
        
        x1, y1, x2, y2 = roi
        
        if self.edit_session.apply_crop(x1, y1, x2, y2):
            self._update_preview()
            self._update_info()
            messagebox.showinfo("Crop aplicado", f"Imagen recortada a {x2-x1}x{y2-y1} pixeles")
        else:
            messagebox.showerror("Error", "No se pudo aplicar el recorte")
    
    def _reset_edits(self):
        if not self.edit_session.is_modified and self._preview_rotation == 0:
            messagebox.showinfo("Info", "No hay cambios que restaurar")
            return
        
        if messagebox.askyesno("Confirmar reset", "Restaurar imagen original?\nSe perderan todos los cambios."):
            self.edit_session.reset()
            self._preview_rotation = 0
            self.rotation_angle.set(0)
            self._update_preview()
            self._update_info()
    
    def _save_edited(self):
        if self.edit_session.edited_data is None:
            messagebox.showwarning("Aviso", "No hay imagen cargada")
            return
        
        # Aplicar rotacion pendiente antes de guardar
        if self._preview_rotation != 0:
            angle = self._preview_rotation
            discrete = angle in [90, 180, 270, -90, -180, -270]
            
            if not discrete and not SCIPY_AVAILABLE:
                messagebox.showerror("Error",
                    "scipy no esta instalado.\n\n"
                    "Para rotacion grado a grado, instala scipy:\n"
                    "pip install scipy\n\n"
                    "O usa rotaciones de 90, 180 o 270 grados")
                return
            
            if not self.edit_session.apply_rotation(angle, discrete_90=discrete):
                messagebox.showerror("Error", "No se pudo aplicar la rotacion")
                return
            
            self._preview_rotation = 0
            self.rotation_angle.set(0)
        
        if not self.edit_session.is_modified:
            if not messagebox.askyesno("Sin cambios", "No hay cambios. Guardar copia de todas formas?"):
                return
        
        default_name = "imagen_editada.tif"
        if self.edit_session.source_filepath:
            base = os.path.splitext(os.path.basename(self.edit_session.source_filepath))[0]
            default_name = f"{base}_editado.tif"
        
        filepath = filedialog.asksaveasfilename(
            title="Guardar imagen editada",
            defaultextension=".tif",
            initialfile=default_name,
            filetypes=[("GeoTIFF", "*.tif"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        if self.edit_session.save(filepath):
            ops = " > ".join(self.edit_session.operations) if self.edit_session.operations else "Ninguna"
            messagebox.showinfo("Guardado exitoso", f"Imagen guardada en:\\n{filepath}\\n\\nOperaciones: {ops}")
        else:
            messagebox.showerror("Error", "No se pudo guardar el archivo")
    
    def has_unsaved_changes(self) -> bool:
        # Considerar rotacion pendiente como cambio no guardado
        return (self.edit_session.is_modified and not self.edit_session.is_saved) or self._preview_rotation != 0
    
    def get_edited_data(self) -> Optional[Tuple[np.ndarray, dict]]:
        if self.edit_session.edited_data is not None:
            return (self.edit_session.edited_data, self.edit_session.edited_profile)
        return None


def safe_divide(a, b, nodata=None):
    """DivisiÃ³n segura: maneja div/0 y valores invÃ¡lidos."""
    with np.errstate(divide='ignore', invalid='ignore'):
        out = a / b
    if nodata is not None:
        out = np.where(np.isfinite(out), out, nodata)
    return out


def diagnose_band_values(g, r, re, nir):
    """
    Diagnostica los valores de las bandas y detecta si son DN crudos o reflectancia.
    Retorna: (needs_calibration, suggested_divisor, suggested_multiplier, data_type)
    """
    all_bands = [g, r, re, nir]
    band_names = ['Green', 'Red', 'RedEdge', 'NIR']
    
    max_val = max(np.nanmax(b) for b in all_bands)
    min_val = min(np.nanmin(b) for b in all_bands)
    
    print(f"\n[DIAGNÓSTICO] Valores de bandas:")
    for name, band in zip(band_names, all_bands):
        b_min, b_max = np.nanmin(band), np.nanmax(band)
        print(f"   {name}: min={b_min:.4f}, max={b_max:.4f}")
    
    # Detectar tipo de datos
    if max_val > 100:
        if max_val > 32767:
            # DN 16-bit unsigned (0-65535)
            print(f"\n[WARNING] ⚠️ DATOS CRUDOS DETECTADOS (DN 16-bit: 0-65535)")
            print(f"   Los índices con constantes (BAI, GEMI, PVI, etc.) darán valores INCORRECTOS")
            print(f"   RECOMENDACIÓN: Activar calibración con divisor=65535, multiplicador=1.0")
            return True, 65535.0, 1.0, "DN_16BIT"
        else:
            # DN 15-bit o similar (0-32767)
            print(f"\n[WARNING] ⚠️ DATOS CRUDOS DETECTADOS (DN ~15-bit)")
            print(f"   RECOMENDACIÓN: Activar calibración con divisor=32768, multiplicador=1.0")
            return True, 32768.0, 1.0, "DN_15BIT"
    elif max_val > 1.0:
        # Valores entre 1 y 100 - posiblemente reflectancia * 100 o similar
        print(f"\n[WARNING] ⚠️ Valores fuera de rango [0,1] detectados (max={max_val:.2f})")
        print(f"   Posiblemente reflectancia escalada x100")
        print(f"   RECOMENDACIÓN: Activar calibración con divisor=100, multiplicador=1.0")
        return True, 100.0, 1.0, "SCALED_100"
    else:
        # Valores en [0, 1] - reflectancia correcta
        print(f"\n[OK] ✓ Valores en rango reflectancia [0,1] - No requiere calibración")
        return False, 1.0, 1.0, "REFLECTANCE"


def compute_indices(g, r, re, nir, nodata_value=None, auto_calibrate=False):
    """
    Calcula 40 Ã­ndices espectrales.
    CORREGIDO: MSAVI2 y GEMI con fÃ³rmulas correctas.
    """
    g = g.astype(np.float32)
    r = r.astype(np.float32)
    re = re.astype(np.float32)
    nir = nir.astype(np.float32)

    idx = {}

    # ==================== ÃNDICES NORMALIZADOS (1-10) ====================
    idx["NDVI"] = safe_divide(nir - r, nir + r, nodata=nodata_value)
    idx["GNDVI"] = safe_divide(nir - g, nir + g, nodata=nodata_value)
    idx["NDVI_RE"] = safe_divide(nir - re, nir + re, nodata=nodata_value)
    idx["NDWI"] = safe_divide(g - nir, g + nir, nodata=nodata_value)
    idx["NGRDI"] = safe_divide(g - r, g + r, nodata=nodata_value)
    idx["ND_G_RE"] = safe_divide(g - re, g + re, nodata=nodata_value)
    idx["ND_G_NIR"] = safe_divide(g - nir, g + nir, nodata=nodata_value)
    idx["ND_R_RE"] = safe_divide(r - re, r + re, nodata=nodata_value)
    idx["ND_R_NIR"] = safe_divide(r - nir, r + nir, nodata=nodata_value)
    idx["ND_RE_NIR"] = safe_divide(re - nir, re + nir, nodata=nodata_value)

    # ==================== RATIOS SIMPLES (11-24) ====================
    idx["RVI"] = safe_divide(nir, r, nodata=nodata_value)
    idx["GVI"] = safe_divide(nir, g, nodata=nodata_value)
    idx["SR_G_R"] = safe_divide(g, r, nodata=nodata_value)
    idx["SR_G_RE"] = safe_divide(g, re, nodata=nodata_value)
    idx["SR_G_NIR"] = safe_divide(g, nir, nodata=nodata_value)
    idx["SR_R_G"] = safe_divide(r, g, nodata=nodata_value)
    idx["SR_R_RE"] = safe_divide(r, re, nodata=nodata_value)
    idx["SR_R_NIR"] = safe_divide(r, nir, nodata=nodata_value)
    idx["SR_RE_G"] = safe_divide(re, g, nodata=nodata_value)
    idx["SR_RE_R"] = safe_divide(re, r, nodata=nodata_value)
    idx["SR_RE_NIR"] = safe_divide(re, nir, nodata=nodata_value)
    idx["SR_NIR_G"] = safe_divide(nir, g, nodata=nodata_value)
    idx["SR_NIR_R"] = safe_divide(nir, r, nodata=nodata_value)
    idx["SR_NIR_RE"] = safe_divide(nir, re, nodata=nodata_value)

    # ==================== ÃNDICES ESPECIALIZADOS (25-40) ====================
    idx["DVI"] = (nir - r).astype(np.float32)
    idx["PSRI"] = safe_divide(r - g, re, nodata=nodata_value)
    idx["SAVI"] = safe_divide(nir - r, nir + r + 0.5, nodata=nodata_value) * 1.5
    idx["OSAVI"] = safe_divide(nir - r, nir + r + 0.16, nodata=nodata_value) * 1.16

    # MSAVI2 - CORREGIDO: fÃ³rmula correcta
    # 0.5 * (2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - R)))
    with np.errstate(divide='ignore', invalid='ignore'):
        msavi_inner = np.sqrt(np.maximum((2*nir + 1)**2 - 8*(nir - r), 0))
        idx["MSAVI2"] = 0.5 * ((2*nir + 1) - msavi_inner)
        if nodata_value is not None:
            idx["MSAVI2"] = np.where(np.isfinite(idx["MSAVI2"]), idx["MSAVI2"], nodata_value)

    # PVI
    pvi_denom = np.sqrt(1 + 0.3**2)
    idx["PVI"] = safe_divide(nir - 0.3*r - 0.5, pvi_denom, nodata=nodata_value)

    # TSAVI
    tsavi_num = 0.33 * (nir - 0.33*r - 0.5)
    tsavi_den = 0.5*nir + r + 1.5*(1 + 0.33**2)
    idx["TSAVI"] = safe_divide(tsavi_num, tsavi_den, nodata=nodata_value)

    # GEMI - CORREGIDO: aplica nodata correctamente
    with np.errstate(divide='ignore', invalid='ignore'):
        eta = safe_divide(2*(nir**2 - r**2) + 1.5*nir + 0.5*r, nir + r + 0.5, nodata=nodata_value)
        gemi_part2 = safe_divide(r - 0.125, 1 - r, nodata=nodata_value)
        idx["GEMI"] = eta * (1 - 0.25*eta) - gemi_part2
        if nodata_value is not None:
            idx["GEMI"] = np.where(np.isfinite(idx["GEMI"]), idx["GEMI"], nodata_value)

    # MTVI2
    with np.errstate(divide='ignore', invalid='ignore'):
        mtvi_num = 1.5 * (1.2*(nir - g) - 2.5*(r - g))
        mtvi_inner = (2*nir + 1)**2 - (6*nir - 5*np.sqrt(np.maximum(r, 0))) - 0.5
        mtvi_denom = np.sqrt(np.maximum(mtvi_inner, 0))
        idx["MTVI2"] = safe_divide(mtvi_num, mtvi_denom, nodata=nodata_value)

    # BAI
    bai_denom = (0.1 - r)**2 + (0.06 - nir)**2
    idx["BAI"] = safe_divide(1.0, bai_denom, nodata=nodata_value)

    idx["MCI"] = safe_divide(nir - re, re - r, nodata=nodata_value)
    idx["CI_Green"] = safe_divide(nir, g, nodata=nodata_value) - 1
    idx["CI_RedEdge"] = safe_divide(nir, re, nodata=nodata_value) - 1
    idx["WDVI"] = nir - (1.06 * r)
    idx["R_M"] = safe_divide(nir, re, nodata=nodata_value) - 1
    idx["SIPI"] = safe_divide(nir - g, nir - r, nodata=nodata_value)

    return idx


def write_indices_geotiff(
    input_tif,
    output_path,
    band_order=("Green", "Red", "RedEdge", "NIR"),
    nodata_value=-9999.0,
    compress="deflate",
    selected_indices=None,
    output_mode="single",
    manual_bands=None,
    use_blocks=True,
    block_size=1024,
    apply_calibration=False,
    calib_divisor=32768.0,
    calib_multiplier=0.51,
    auto_calibrate=False
):
    """
    Lee GeoTIFF multibanda, calcula Ã­ndices y guarda resultado.
    
    Args:
        input_tif: Ruta al archivo de entrada (None si manual_bands)
        output_path: Ruta de salida
        manual_bands: Dict con rutas de bandas manuales
        use_blocks: Procesar por bloques para imÃ¡genes grandes
        block_size: TamaÃ±o de bloque en pÃ­xeles
    """
    print("\n" + "="*70)
    print("PROCESANDO - SPECTRAL INDICATOR v7.2")
    print("="*70)
    
    # Lista de Ã­ndices disponibles
    all_index_names = [
        "NDVI", "GNDVI", "NDVI_RE", "NDWI", "NGRDI",
        "ND_G_RE", "ND_G_NIR", "ND_R_RE", "ND_R_NIR", "ND_RE_NIR",
        "RVI", "GVI", "SR_G_R", "SR_G_RE", "SR_G_NIR",
        "SR_R_G", "SR_R_RE", "SR_R_NIR", "SR_RE_G", "SR_RE_R",
        "SR_RE_NIR", "SR_NIR_G", "SR_NIR_R", "SR_NIR_RE",
        "DVI", "PSRI", "SAVI", "OSAVI", "MSAVI2",
        "PVI", "TSAVI", "GEMI", "MTVI2", "BAI",
        "MCI", "CI_Green", "CI_RedEdge", "WDVI", "R_M", "SIPI"
    ]
    
    if selected_indices is None:
        index_names = all_index_names
    else:
        index_names = [n for n in all_index_names if n in selected_indices]
    
    # FÃ³rmulas para metadatos
    formulas = {
        "NDVI": "(NIR - R) / (NIR + R)",
        "GNDVI": "(NIR - G) / (NIR + G)",
        "NDVI_RE": "(NIR - RE) / (NIR + RE)",
        "NDWI": "(G - NIR) / (G + NIR)",
        "NGRDI": "(G - R) / (G + R)",
        "ND_G_RE": "(G - RE) / (G + RE)",
        "ND_G_NIR": "(G - NIR) / (G + NIR)",
        "ND_R_RE": "(R - RE) / (R + RE)",
        "ND_R_NIR": "(R - NIR) / (R + NIR)",
        "ND_RE_NIR": "(RE - NIR) / (RE + NIR)",
        "RVI": "NIR / R",
        "GVI": "NIR / G",
        "SR_G_R": "G / R",
        "SR_G_RE": "G / RE",
        "SR_G_NIR": "G / NIR",
        "SR_R_G": "R / G",
        "SR_R_RE": "R / RE",
        "SR_R_NIR": "R / NIR",
        "SR_RE_G": "RE / G",
        "SR_RE_R": "RE / R",
        "SR_RE_NIR": "RE / NIR",
        "SR_NIR_G": "NIR / G",
        "SR_NIR_R": "NIR / R",
        "SR_NIR_RE": "NIR / RE",
        "DVI": "NIR - R",
        "PSRI": "(R - G) / RE",
        "SAVI": "1.5 * (NIR - R) / (NIR + R + 0.5)",
        "OSAVI": "1.16 * (NIR - R) / (NIR + R + 0.16)",
        "MSAVI2": "0.5 * (2*NIR + 1 - sqrt((2*NIR+1)Â² - 8*(NIR-R)))",
        "PVI": "(NIR - 0.3*R - 0.5) / sqrt(1.09)",
        "TSAVI": "0.33*(NIR - 0.33*R - 0.5) / (0.5*NIR + R + 1.66)",
        "GEMI": "Î·*(1 - 0.25*Î·) - (R - 0.125)/(1 - R)",
        "MTVI2": "1.5*(1.2*(NIR-G) - 2.5*(R-G)) / sqrt(...)",
        "BAI": "1 / ((0.1-R)Â² + (0.06-NIR)Â²)",
        "MCI": "(NIR - RE) / (RE - R)",
        "CI_Green": "(NIR / G) - 1",
        "CI_RedEdge": "(NIR / RE) - 1",
        "WDVI": "NIR - 1.06*R",
        "R_M": "(NIR / RE) - 1",
        "SIPI": "(NIR - G) / (NIR - R)"
    }
    
    # Modo bandas manuales: cargar todo en memoria (archivos separados)
    if manual_bands is not None:
        data, src_profile, temp_path = load_multispectral(input_tif, manual_bands)
        try:
            return _process_in_memory(data, src_profile, output_path, index_names, 
                                     formulas, nodata_value, compress, output_mode,
                                     apply_calibration, calib_divisor, calib_multiplier)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    # Abrir archivo para anÃ¡lisis
    with rasterio.open(input_tif) as src:
        tags = src.tags()
        desc = tags.get("TIFFTAG_IMAGEDESCRIPTION", "")
        rows, cols = src.height, src.width
        count = src.count
        
        print(f"\n[LOAD] Analizando: {os.path.basename(input_tif)}")
        print(f"   Bandas detectadas: {count}")
        print(f"   Dimensiones: {cols}x{rows} pÃ­xeles")
        print(f"   Interleave: {src.profile.get('interleave', 'N/A')}")
        
        # Estimar memoria para decidir modo de procesamiento
        mem_mb = (4 * rows * cols * 4) / (1024**2)  # 4 bandas float32
        output_mem_mb = (len(index_names) * rows * cols * 4) / (1024**2)
        total_mem_mb = mem_mb + output_mem_mb
        
        print(f"   Memoria entrada: {mem_mb:.1f} MB")
        print(f"   Memoria salida: {output_mem_mb:.1f} MB")
        print(f"   Memoria total estimada: {total_mem_mb:.1f} MB")
        
        # Casos especiales que requieren carga completa
        is_imagej = "ImageJ" in desc and "images=" in desc
        is_pixel_interleaved = count == 1 and src.profile.get("interleave") == "pixel"
        
        # Decidir modo de procesamiento
        # Usar bloques si: use_blocks=True, imagen grande (>500MB), y no es caso especial
        use_block_mode = use_blocks and total_mem_mb > 500 and not is_imagej and not is_pixel_interleaved and count >= 4
        
        if use_block_mode:
            print(f"\n[MODE] Procesamiento por BLOQUES ({block_size}x{block_size} px)")
            return _process_by_blocks(input_tif, output_path, index_names, formulas,
                                     nodata_value, compress, output_mode, block_size,
                                     apply_calibration, calib_divisor, calib_multiplier,
                                     auto_calibrate)
        else:
            print(f"\n[MODE] Procesamiento en MEMORIA")
            data, src_profile, temp_path = load_multispectral(input_tif, manual_bands)
            try:
                return _process_in_memory(data, src_profile, output_path, index_names,
                                         formulas, nodata_value, compress, output_mode,
                                         apply_calibration, calib_divisor, calib_multiplier,
                                         auto_calibrate)
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass


def _process_by_blocks(input_tif, output_path, index_names, formulas, 
                       nodata_value, compress, output_mode, block_size,
                       apply_calibration=False, calib_divisor=32768.0, calib_multiplier=0.51,
                       auto_calibrate=False):
    """
    Procesa imagen por bloques para optimizar uso de memoria.
    Lee y escribe bloque por bloque sin cargar toda la imagen.
    
    Args:
        auto_calibrate: Si True, detecta y aplica calibración automáticamente
    """
    with rasterio.open(input_tif) as src:
        rows, cols = src.height, src.width
        src_profile = src.profile.copy()
        
        if src.count < 4:
            raise ValueError(f"Se requieren 4 bandas, archivo tiene {src.count}")
        
        # Configurar perfil de salida
        out_profile = src_profile.copy()
        out_profile.update(
            dtype=rasterio.float32,
            count=len(index_names) if output_mode == "single" else 1,
            nodata=nodata_value,
            compress=compress,
            predictor=2,
            driver='GTiff',
            tiled=True,
            blockxsize=min(block_size, cols),
            blockysize=min(block_size, rows)
        )
        
        # Calcular nÃºmero de bloques
        n_blocks_x = (cols + block_size - 1) // block_size
        n_blocks_y = (rows + block_size - 1) // block_size
        total_blocks = n_blocks_x * n_blocks_y
        
        print(f"   Bloques: {n_blocks_x}x{n_blocks_y} = {total_blocks} total")
        
        # Auto-calibración: diagnosticar con primer bloque para determinar parámetros
        auto_calib_divisor = 1.0
        auto_calib_multiplier = 1.0
        apply_auto = False
        
        if auto_calibrate and not apply_calibration:
            # Leer primer bloque para diagnóstico
            window = Window(0, 0, min(block_size, cols), min(block_size, rows))
            g_sample = src.read(1, window=window).astype(np.float32)
            r_sample = src.read(2, window=window).astype(np.float32)
            re_sample = src.read(3, window=window).astype(np.float32)
            nir_sample = src.read(4, window=window).astype(np.float32)
            
            needs_calib, auto_calib_divisor, auto_calib_multiplier, data_type = diagnose_band_values(
                g_sample, r_sample, re_sample, nir_sample
            )
            apply_auto = needs_calib
            if apply_auto:
                print(f"\n[AUTO-CAL] Se aplicará calibración automática: (banda / {auto_calib_divisor}) * {auto_calib_multiplier}")
        
        if output_mode == "single":
            # Un archivo multibanda
            with rasterio.open(output_path, "w", **out_profile) as dst:
                # Escribir tags de Ã­ndices
                for i, name in enumerate(index_names, start=1):
                    dst.update_tags(i, index_name=name, index_formula=formulas.get(name, ""))
                
                block_num = 0
                for row_off in range(0, rows, block_size):
                    for col_off in range(0, cols, block_size):
                        block_num += 1
                        
                        # Calcular tamaÃ±o real del bloque (bordes pueden ser menores)
                        win_height = min(block_size, rows - row_off)
                        win_width = min(block_size, cols - col_off)
                        window = Window(col_off, row_off, win_width, win_height)
                        
                        # Leer 4 bandas del bloque
                        g = src.read(1, window=window).astype(np.float32)
                        r = src.read(2, window=window).astype(np.float32)
                        re = src.read(3, window=window).astype(np.float32)
                        nir = src.read(4, window=window).astype(np.float32)
                        
                        # Aplicar calibración manual si está activada
                        if apply_calibration:
                            g = (g / calib_divisor) * calib_multiplier
                            r = (r / calib_divisor) * calib_multiplier
                            re = (re / calib_divisor) * calib_multiplier
                            nir = (nir / calib_divisor) * calib_multiplier
                        # O aplicar auto-calibración si fue detectada
                        elif apply_auto:
                            g = (g / auto_calib_divisor) * auto_calib_multiplier
                            r = (r / auto_calib_divisor) * auto_calib_multiplier
                            re = (re / auto_calib_divisor) * auto_calib_multiplier
                            nir = (nir / auto_calib_divisor) * auto_calib_multiplier
                        
                        # Calcular indices para este bloque
                        indices = compute_indices(g, r, re, nir, nodata_value=nodata_value)
                        
                        # Escribir cada Ã­ndice en su banda correspondiente
                        for i, name in enumerate(index_names, start=1):
                            idx_data = indices[name].astype(np.float32)
                            if nodata_value is not None:
                                idx_data = np.where(np.isfinite(idx_data), idx_data, nodata_value)
                            dst.write(idx_data, i, window=window)
                        
                        # Progreso cada 10 bloques o al final
                        if block_num % 10 == 0 or block_num == total_blocks:
                            pct = (block_num / total_blocks) * 100
                            print(f"\r   Progreso: {block_num}/{total_blocks} bloques ({pct:.1f}%)", end="", flush=True)
                
                print()  # Nueva lÃ­nea despuÃ©s del progreso
            
            print(f"\n[OK] Guardado (SINGLE): {output_path}")
            print(f"[OK] {len(index_names)} Ã­ndices procesados por bloques")
            return 1
        
        else:  # output_mode == "multiple"
            base_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            
            # Para mÃºltiples archivos, procesar cada Ã­ndice por separado
            files_created = 0
            print(f"\nGenerando {len(index_names)} archivos por bloques...")
            
            for idx_num, name in enumerate(index_names, start=1):
                file_name = f"{base_name}_{idx_num:03d}_{name}.tif"
                file_path = os.path.join(base_dir, file_name)
                
                with rasterio.open(file_path, "w", **out_profile) as dst:
                    dst.update_tags(1, index_name=name, index_formula=formulas.get(name, ""))
                    
                    for row_off in range(0, rows, block_size):
                        for col_off in range(0, cols, block_size):
                            win_height = min(block_size, rows - row_off)
                            win_width = min(block_size, cols - col_off)
                            window = Window(col_off, row_off, win_width, win_height)
                            
                            g = src.read(1, window=window).astype(np.float32)
                            r = src.read(2, window=window).astype(np.float32)
                            re = src.read(3, window=window).astype(np.float32)
                            nir = src.read(4, window=window).astype(np.float32)
                            
                            # Aplicar calibración manual si está activada
                            if apply_calibration:
                                g = (g / calib_divisor) * calib_multiplier
                                r = (r / calib_divisor) * calib_multiplier
                                re = (re / calib_divisor) * calib_multiplier
                                nir = (nir / calib_divisor) * calib_multiplier
                            # O aplicar auto-calibración si fue detectada
                            elif apply_auto:
                                g = (g / auto_calib_divisor) * auto_calib_multiplier
                                r = (r / auto_calib_divisor) * auto_calib_multiplier
                                re = (re / auto_calib_divisor) * auto_calib_multiplier
                                nir = (nir / auto_calib_divisor) * auto_calib_multiplier
                            
                            indices = compute_indices(g, r, re, nir, nodata_value=nodata_value)
                            
                            idx_data = indices[name].astype(np.float32)
                            if nodata_value is not None:
                                idx_data = np.where(np.isfinite(idx_data), idx_data, nodata_value)
                            dst.write(idx_data, 1, window=window)
                
                print(f"  [{idx_num:2d}/{len(index_names)}] {file_name}")
                files_created += 1
            
            print(f"\n[OK] {files_created} archivos creados en: {base_dir}")
            return files_created


def _process_in_memory(data, src_profile, output_path, index_names, formulas,
                       nodata_value, compress, output_mode,
                       apply_calibration=False, calib_divisor=32768.0, calib_multiplier=0.51,
                       auto_calibrate=False):
    """
    Procesa imagen completa en memoria (metodo original).
    Usado para imagenes pequenas o casos especiales (ImageJ, pixel-interleaved).
    
    Args:
        apply_calibration: Si True, aplica calibracion radiometrica
        calib_divisor: Divisor para calibracion (default 32768)
        calib_multiplier: Multiplicador para calibracion (default 0.51)
        auto_calibrate: Si True, detecta y aplica calibración automáticamente
    """
    count, rows, cols = data.shape
    print(f"\n[OK] Datos cargados: {count} bandas ({rows}x{cols} pixeles)")
    
    mem_mb = (count * rows * cols * 4) / (1024**2)
    print(f"   Memoria datos: {mem_mb:.1f} MB")
    
    if count < 4:
        raise ValueError(f"Se requieren 4 bandas, archivo tiene {count}")
    
    # Aplicar calibracion radiometrica MANUAL si esta activada
    if apply_calibration:
        print(f"[OK] Aplicando calibracion MANUAL: (pixel / {calib_divisor}) * {calib_multiplier}")
        data = (data.astype(np.float32) / calib_divisor) * calib_multiplier
    
    # Asignar bandas segun orden
    g = data[0]
    r = data[1]
    re = data[2]
    nir = data[3]
    
    # Auto-calibración: detectar si hay valores raw DN y convertir a reflectancia
    use_auto = auto_calibrate and not apply_calibration
    if use_auto:
        needs_calib, divisor, multiplier, data_type = diagnose_band_values(g, r, re, nir)
        if needs_calib:
            print(f"[OK] Auto-calibración detectada: {data_type}")
            print(f"     Aplicando: (pixel / {divisor}) * {multiplier}")
            g = (g.astype(np.float32) / divisor) * multiplier
            r = (r.astype(np.float32) / divisor) * multiplier
            re = (re.astype(np.float32) / divisor) * multiplier
            nir = (nir.astype(np.float32) / divisor) * multiplier
        else:
            print(f"[OK] Datos ya en reflectancia ({data_type}), no se requiere calibración")
    
    print(f"[OK] Calculando indices...")
    indices = compute_indices(g, r, re, nir, nodata_value=nodata_value, auto_calibrate=False)

    if output_mode == "single":
        out_stack = np.stack([indices[n] for n in index_names]).astype(np.float32)
        
        if nodata_value is not None:
            out_stack = np.where(np.isfinite(out_stack), out_stack, nodata_value).astype(np.float32)

        out_profile = src_profile.copy()
        out_profile.update(
            dtype=rasterio.float32,
            count=len(index_names),
            nodata=nodata_value,
            compress=compress,
            predictor=2,
            driver='GTiff'
        )

        with rasterio.open(output_path, "w", **out_profile) as dst:
            for i, name in enumerate(index_names, start=1):
                dst.write(out_stack[i-1], i)
                dst.update_tags(i, index_name=name, index_formula=formulas.get(name, ""))

        print(f"\n[OK] Guardado (SINGLE): {output_path}")
        print(f"[OK] {len(index_names)} Ã­ndices en 1 archivo")
        return 1

    elif output_mode == "multiple":
        out_profile = src_profile.copy()
        out_profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=nodata_value,
            compress=compress,
            predictor=2,
            driver='GTiff'
        )

        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        files_created = 0
        print(f"\nGenerando {len(index_names)} archivos...")
        
        for i, name in enumerate(index_names, start=1):
            file_name = f"{base_name}_{i:03d}_{name}.tif"
            file_path = os.path.join(base_dir, file_name)
            
            index_data = indices[name].astype(np.float32)
            if nodata_value is not None:
                index_data = np.where(np.isfinite(index_data), index_data, nodata_value).astype(np.float32)
            
            with rasterio.open(file_path, "w", **out_profile) as dst:
                dst.write(index_data, 1)
                dst.update_tags(1, index_name=name, index_formula=formulas.get(name, ""))
            
            print(f"  [{i:2d}/{len(index_names)}] {file_name}")
            files_created += 1

        print(f"\n[OK] {files_created} archivos creados en: {base_dir}")
        return files_created


# ==================== SPLASH SCREEN ====================

class SplashScreen:
    """Pantalla de presentaciÃ³n"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Spectral Indicator")
        self.root.geometry("500x600")
        self.root.resizable(False, False)
        self.root.configure(bg="#ffffff")
        
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - 250
        y = (self.root.winfo_screenheight() // 2) - 300
        self.root.geometry(f"+{x}+{y}")
        
        main_frame = tk.Frame(self.root, bg="#ffffff")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Logo
        try:
            if hasattr(sys, '_MEIPASS'):
                base_dir = sys._MEIPASS
            else:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            
            logo_path = os.path.join(base_dir, "logo.png")
            
            if os.path.exists(logo_path):
                original = tk.PhotoImage(file=logo_path)
                self.logo_image = original.subsample(2, 2)
                logo_label = tk.Label(main_frame, image=self.logo_image, bg="#ffffff")
                logo_label.pack(pady=10)
        except:
            pass
        
        # TÃ­tulo
        tk.Label(main_frame, text="SPECTRAL INDICATOR", font=("Arial", 20, "bold"),
                bg="#ffffff", fg="#1f4788").pack(pady=10)
        
        tk.Frame(main_frame, height=2, bg="#1f4788").pack(fill=tk.X, pady=10)
        
        tk.Label(main_frame, text="Generador de Indices Espectrales v7.2",
                font=("Arial", 12), bg="#ffffff", fg="#333333").pack(pady=5)
        
        # Contacto
        contact_frame = tk.Frame(main_frame, bg="#ffffff")
        contact_frame.pack(pady=20)
        
        tk.Label(contact_frame, text="Esteban Caffiero", font=("Arial", 12, "bold"),
                bg="#ffffff", fg="#1f4788").pack()
        tk.Label(contact_frame, text="Email: estebancaffiero7@gmail.com",
                font=("Arial", 10), bg="#ffffff").pack()
        tk.Label(contact_frame, text="Tel: +56 9 4578 9407",
                font=("Arial", 10), bg="#ffffff").pack()
        
        # Progreso
        self.progress = ttk.Progressbar(main_frame, length=300, mode='determinate', maximum=100)
        self.progress.pack(pady=20)
        
        self.status_label = tk.Label(main_frame, text="Cargando...", font=("Arial", 10),
                                     bg="#ffffff", fg="#666666")
        self.status_label.pack(pady=5)
        
        tk.Label(main_frame, text="v7.2 | 2026", font=("Arial", 9),
                bg="#ffffff", fg="#999999").pack(side=tk.BOTTOM, pady=10)
    
    def update_progress(self, value, message=""):
        self.progress["value"] = value
        if message:
            self.status_label.config(text=message)
        self.root.update()
    
    def close(self):
        self.root.destroy()


def show_splash_and_check_deps():
    """Mostrar splash y verificar dependencias"""
    splash_root = tk.Tk()
    splash = SplashScreen(splash_root)
    
    for val, msg in [(20, "Verificando numpy..."), (50, "Verificando rasterio..."),
                     (80, "Inicializando..."), (100, "Listo!")]:
        splash.update_progress(val, msg)
        time.sleep(0.3)
    
    time.sleep(0.5)
    splash.close()
    return True


# ==================== GUI PRINCIPAL ====================

class IndicadorGUI:
    INDICES_COMPLETE = [
        "NDVI", "GNDVI", "NDVI_RE", "NDWI", "NGRDI", "ND_G_RE", "ND_G_NIR", 
        "ND_R_RE", "ND_R_NIR", "ND_RE_NIR", "RVI", "GVI", "SR_G_R", "SR_G_RE", 
        "SR_G_NIR", "SR_R_G", "SR_R_RE", "SR_R_NIR", "SR_RE_G", "SR_RE_R",
        "SR_RE_NIR", "SR_NIR_G", "SR_NIR_R", "SR_NIR_RE", "DVI", "PSRI", 
        "SAVI", "OSAVI", "MSAVI2", "PVI", "TSAVI", "GEMI", "MTVI2", "BAI",
        "MCI", "CI_Green", "CI_RedEdge", "WDVI", "R_M", "SIPI"
    ]
    
    INDICES_DESCRIPTIONS = {
        "NDVI": "Normalized Difference Vegetation Index",
        "GNDVI": "Green NDVI", "NDVI_RE": "NDVI Red Edge",
        "NDWI": "Normalized Difference Water Index",
        "NGRDI": "Normalized Green-Red Difference",
        "ND_G_RE": "ND Green-RedEdge", "ND_G_NIR": "ND Green-NIR",
        "ND_R_RE": "ND Red-RedEdge", "ND_R_NIR": "ND Red-NIR",
        "ND_RE_NIR": "ND RedEdge-NIR", "RVI": "Ratio Vegetation Index",
        "GVI": "Green Vegetation Index", "SR_G_R": "Simple Ratio G/R",
        "SR_G_RE": "SR G/RE", "SR_G_NIR": "SR G/NIR", "SR_R_G": "SR R/G",
        "SR_R_RE": "SR R/RE", "SR_R_NIR": "SR R/NIR", "SR_RE_G": "SR RE/G",
        "SR_RE_R": "SR RE/R", "SR_RE_NIR": "SR RE/NIR", "SR_NIR_G": "SR NIR/G",
        "SR_NIR_R": "SR NIR/R", "SR_NIR_RE": "SR NIR/RE",
        "DVI": "Difference Vegetation Index",
        "PSRI": "Plant Senescence Reflectance Index",
        "SAVI": "Soil Adjusted VI", "OSAVI": "Optimized SAVI",
        "MSAVI2": "Modified SAVI 2", "PVI": "Perpendicular VI",
        "TSAVI": "Thematic Mapper SAVI", "GEMI": "Global Environment Monitoring",
        "MTVI2": "Modified Triangular VI 2", "BAI": "Burn Area Index",
        "MCI": "Moisture/Chlorophyll Index", "CI_Green": "Chlorophyll Index Green",
        "CI_RedEdge": "Chlorophyll Index RE", "WDVI": "Water-adjusted DVI",
        "R_M": "Red-edge Modified", "SIPI": "Structural Pigment Index"
    }
    
    # Fuentes cientificas de los indices espectrales
    INDEX_SOURCES = {
        "NDVI": "Rouse et al. (1974)",
        "GNDVI": "Gitelson et al. (1996)",
        "NDVI_RE": "Gitelson & Merzlyak (1994)",
        "NDWI": "McFeeters (1996)",
        "NGRDI": "Tucker (1979)",
        "ND_G_RE": "Derived from ND formula",
        "ND_G_NIR": "Derived from ND formula",
        "ND_R_RE": "Derived from ND formula",
        "ND_R_NIR": "Derived from ND formula",
        "ND_RE_NIR": "Derived from ND formula",
        "RVI": "Jordan (1969)",
        "GVI": "Kauth & Thomas (1976)",
        "SR_G_R": "Birth & McVey (1968)",
        "SR_G_RE": "Derived from SR formula",
        "SR_G_NIR": "Derived from SR formula",
        "SR_R_G": "Derived from SR formula",
        "SR_R_RE": "Derived from SR formula",
        "SR_R_NIR": "Derived from SR formula",
        "SR_RE_G": "Derived from SR formula",
        "SR_RE_R": "Derived from SR formula",
        "SR_RE_NIR": "Derived from SR formula",
        "SR_NIR_G": "Derived from SR formula",
        "SR_NIR_R": "Derived from SR formula",
        "SR_NIR_RE": "Derived from SR formula",
        "DVI": "Richardson & Wiegand (1977)",
        "PSRI": "Merzlyak et al. (1999)",
        "SAVI": "Huete (1988)",
        "OSAVI": "Rondeaux et al. (1996)",
        "MSAVI2": "Qi et al. (1994)",
        "PVI": "Richardson & Wiegand (1977)",
        "TSAVI": "Baret et al. (1989)",
        "GEMI": "Pinty & Verstraete (1992)",
        "MTVI2": "Haboudane et al. (2004)",
        "BAI": "Martin (1998)",
        "MCI": "Gower et al. (2005)",
        "CI_Green": "Gitelson et al. (2003)",
        "CI_RedEdge": "Gitelson et al. (2003)",
        "WDVI": "Clevers (1988)",
        "R_M": "Sims & Gamon (2002)",
        "SIPI": "Penuelas et al. (1995)"
    }
    
    # Formulas de los indices espectrales (para la guia)
    INDICES_FORMULAS = {
        "NDVI": "(NIR - R) / (NIR + R)",
        "GNDVI": "(NIR - G) / (NIR + G)",
        "NDVI_RE": "(NIR - RE) / (NIR + RE)",
        "NDWI": "(G - NIR) / (G + NIR)",
        "NGRDI": "(G - R) / (G + R)",
        "ND_G_RE": "(G - RE) / (G + RE)",
        "ND_G_NIR": "(G - NIR) / (G + NIR)",
        "ND_R_RE": "(R - RE) / (R + RE)",
        "ND_R_NIR": "(R - NIR) / (R + NIR)",
        "ND_RE_NIR": "(RE - NIR) / (RE + NIR)",
        "RVI": "NIR / R",
        "GVI": "NIR / G",
        "SR_G_R": "G / R",
        "SR_G_RE": "G / RE",
        "SR_G_NIR": "G / NIR",
        "SR_R_G": "R / G",
        "SR_R_RE": "R / RE",
        "SR_R_NIR": "R / NIR",
        "SR_RE_G": "RE / G",
        "SR_RE_R": "RE / R",
        "SR_RE_NIR": "RE / NIR",
        "SR_NIR_G": "NIR / G",
        "SR_NIR_R": "NIR / R",
        "SR_NIR_RE": "NIR / RE",
        "DVI": "NIR - R",
        "PSRI": "(R - G) / RE",
        "SAVI": "1.5*(NIR - R) / (NIR + R + 0.5)",
        "OSAVI": "1.16*(NIR - R) / (NIR + R + 0.16)",
        "MSAVI2": "0.5*(2*NIR + 1 - sqrt((2*NIR+1)^2 - 8*(NIR-R)))",
        "PVI": "(NIR - 0.3*R - 0.5) / sqrt(1.09)",
        "TSAVI": "0.33*(NIR - 0.33*R - 0.5) / (0.5*NIR + R + 1.66)",
        "GEMI": "n*(1 - 0.25*n) - (R - 0.125)/(1 - R)",
        "MTVI2": "1.5*(1.2*(NIR-G) - 2.5*(R-G)) / sqrt(...)",
        "BAI": "1 / ((0.1-R)^2 + (0.06-NIR)^2)",
        "MCI": "(NIR - RE) / (RE - R)",
        "CI_Green": "(NIR / G) - 1",
        "CI_RedEdge": "(NIR / RE) - 1",
        "WDVI": "NIR - 1.06*R",
        "R_M": "(NIR / RE) - 1",
        "SIPI": "(NIR - G) / (NIR - R)"
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("Generador de Indices Espectrales v7.2")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f0f0")
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.output_name = tk.StringVar(value="INDICES_RESULTADO")
        self.nodata_value = tk.StringVar(value="-9999")
        self.output_mode = tk.StringVar(value="single")
        
        # Selector manual de bandas
        self.manual_band_mode = tk.BooleanVar(value=False)
        self.band_files = {
            "Green": tk.StringVar(),
            "Red": tk.StringVar(),
            "RedEdge": tk.StringVar(),
            "NIR": tk.StringVar()
        }
        
        self.selected_indices = {idx: tk.BooleanVar(value=True) for idx in self.INDICES_COMPLETE}
        self.checkbox_refs = {}
        
        # Calibracion radiometrica
        self.auto_calibrate = tk.BooleanVar(value=True)  # Auto-calibración activada por defecto
        self.apply_calibration = tk.BooleanVar(value=False)
        self.calib_divisor = tk.StringVar(value="65535")
        self.calib_multiplier = tk.StringVar(value="1.0")
        
        self.create_widgets()
    
    def create_widgets(self):
        """Crea widgets con pestanas"""
        # Header frame con título y botón Nueva Sesión
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(header_frame, text="Generador de Indices Espectrales v7.2",
                font=("Arial", 16, "bold"), bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        
        # Botón Nueva Sesión (limpiar memoria de TODAS las pestañas)
        reset_frame = ttk.Frame(header_frame)
        reset_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(reset_frame, text="🗑️ Nueva Sesión",
                  command=self._reset_all_sessions,
                  width=18).pack(side=tk.TOP)
        ttk.Label(reset_frame, text="(Limpia TODAS las pestañas)",
                 font=("Arial", 8), foreground="#666").pack(side=tk.TOP)
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 1. Edicion (primero para flujo logico)
        self._create_edicion_tab()
        
        # 2. Procesamiento
        self.tab_process = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_process, text="Procesamiento")
        self.create_process_tab()
        
        # 3. Guia de Indices
        self.tab_guide = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_guide, text="Guia de Indices")
        self.create_guide_tab()
        
        # 4. Parametros (ROI Analysis)
        self._create_parametros_tab()
        
        self.status_var = tk.StringVar(value="Listo")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken",
                 font=("Arial", 9)).pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
    
    def create_process_tab(self):
        """PestaÃ±a de procesamiento con selector manual de bandas"""
        main_canvas = tk.Canvas(self.tab_process, bg="#f0f0f0", highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.tab_process, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind("<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind("<MouseWheel>", _on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", _on_mousewheel)
        
        main_frame = scrollable_frame
        
        # ========== 1. MODO DE ENTRADA (PRIMERO) ==========
        input_mode_frame = ttk.LabelFrame(main_frame, text="1. Modo de Entrada", padding=10)
        input_mode_frame.pack(anchor="w", fill="x", padx=15, pady=(15, 5))
        
        ttk.Radiobutton(input_mode_frame, text="[+] Archivo multibanda (4 bandas en 1 TIFF)",
                       variable=self.manual_band_mode, value=False,
                       command=self.toggle_band_mode).pack(anchor="w")
        
        ttk.Radiobutton(input_mode_frame, text="[-] Seleccionar bandas manualmente (4 archivos separados)",
                       variable=self.manual_band_mode, value=True,
                       command=self.toggle_band_mode).pack(anchor="w")
        
        # ========== 2. ARCHIVO MULTIBANDA (visible por defecto) ==========
        self.multiband_frame = ttk.LabelFrame(main_frame, text="2. Archivo TIF de Entrada", padding=10)
        self.multiband_frame.pack(anchor="w", fill="x", padx=15, pady=5)
        
        entry_frame = ttk.Frame(self.multiband_frame)
        entry_frame.pack(fill="x")
        ttk.Entry(entry_frame, textvariable=self.input_file, width=70).pack(side="left", padx=(0, 5))
        ttk.Button(entry_frame, text="Examinar", command=self.select_input_file).pack(side="left")
        
        # ========== 2. BANDAS MANUALES (oculto por defecto) ==========
        self.manual_bands_frame = ttk.LabelFrame(main_frame, text="2. Selecci\u00f3n Manual de Bandas", padding=10)
        # NO se empaqueta aquÃ­, se controla con toggle_band_mode
        
        band_names = [("Green", "[G] Green (Verde)"), ("Red", "[R] Red (Rojo)"),
                     ("RedEdge", "[RE] RedEdge (Borde Rojo)"), ("NIR", "[NIR] NIR (Infrarrojo Cercano)")]
        
        for band_key, band_label in band_names:
            band_row = ttk.Frame(self.manual_bands_frame)
            band_row.pack(fill="x", pady=2)
            
            ttk.Label(band_row, text=band_label, width=25).pack(side="left")
            ttk.Entry(band_row, textvariable=self.band_files[band_key], width=50).pack(side="left", padx=5)
            ttk.Button(band_row, text="...", width=3,
                      command=lambda k=band_key: self.select_band_file(k)).pack(side="left")
        
        # ========== 3. CARPETA DE SALIDA ==========
        output_frame = ttk.LabelFrame(main_frame, text="3. Carpeta de Resultados", padding=10)
        output_frame.pack(anchor="w", fill="x", padx=15, pady=5)
        
        out_entry_frame = ttk.Frame(output_frame)
        out_entry_frame.pack(fill="x")
        ttk.Entry(out_entry_frame, textvariable=self.output_dir, width=70).pack(side="left", padx=(0, 5))
        ttk.Button(out_entry_frame, text="Examinar", command=self.select_output_dir).pack(side="left")
        
        # ========== 4. OPCIONES DE SALIDA ==========
        options_frame = ttk.LabelFrame(main_frame, text="4. Opciones de Salida", padding=10)
        options_frame.pack(anchor="w", fill="x", padx=15, pady=5)
        
        # Nombre del archivo
        name_row = ttk.Frame(options_frame)
        name_row.pack(fill="x", pady=2)
        ttk.Label(name_row, text="Nombre del archivo:", width=20).pack(side="left")
        ttk.Entry(name_row, textvariable=self.output_name, width=40).pack(side="left", padx=5)
        
        # Valor NoData
        nodata_row = ttk.Frame(options_frame)
        nodata_row.pack(fill="x", pady=2)
        ttk.Label(nodata_row, text="Valor NoData:", width=20).pack(side="left")
        ttk.Entry(nodata_row, textvariable=self.nodata_value, width=15).pack(side="left", padx=5)
        
        # Modo de salida
        ttk.Separator(options_frame, orient="horizontal").pack(fill="x", pady=5)
        ttk.Label(options_frame, text="Formato de salida:", font=("Arial", 9, "bold")).pack(anchor="w")
        ttk.Radiobutton(options_frame, text="[+] Un archivo multibanda (todos los indices en 1 TIFF)",
                       variable=self.output_mode, value="single").pack(anchor="w", padx=10)
        ttk.Radiobutton(options_frame, text="[-] Multiples archivos (un TIFF por cada indice)",
                       variable=self.output_mode, value="multiple").pack(anchor="w", padx=10)
        
        # ========== 4.5 CALIBRACION RADIOMETRICA ==========
        calib_frame = ttk.LabelFrame(main_frame, text="4.5 Calibración Radiométrica", padding=10)
        calib_frame.pack(anchor="w", fill="x", padx=15, pady=5)
        
        # Auto-calibración (RECOMENDADA)
        ttk.Checkbutton(calib_frame, text="🔄 Auto-detectar y calibrar (RECOMENDADO para Mavic 3M / Agisoft)",
                       variable=self.auto_calibrate).pack(anchor="w")
        ttk.Label(calib_frame, text="   Detecta automáticamente si los datos son DN crudos y aplica calibración",
                 font=("Arial", 8), foreground="#0066cc").pack(anchor="w")
        
        ttk.Separator(calib_frame, orient="horizontal").pack(fill="x", pady=5)
        
        # Calibración manual (opcional)
        ttk.Checkbutton(calib_frame, text="Aplicar calibración manual (ignora auto-detección)",
                       variable=self.apply_calibration).pack(anchor="w")
        
        ttk.Label(calib_frame, text="Fórmula: pixel_calibrado = (pixel / divisor) * multiplicador",
                 font=("Arial", 8), foreground="gray").pack(anchor="w", pady=(5,2))
        
        calib_row = ttk.Frame(calib_frame)
        calib_row.pack(fill="x", pady=2)
        ttk.Label(calib_row, text="Divisor:", width=12).pack(side="left")
        ttk.Entry(calib_row, textvariable=self.calib_divisor, width=10).pack(side="left", padx=5)
        ttk.Label(calib_row, text="Multiplicador:", width=12).pack(side="left", padx=(10,0))
        ttk.Entry(calib_row, textvariable=self.calib_multiplier, width=10).pack(side="left", padx=5)
        
        ttk.Label(calib_frame, text="Presets: 16-bit → 65535/1.0 | DJI Terra → 32768/0.51",
                 font=("Arial", 8), foreground="#666").pack(anchor="w")
        
        # ========== 5. SELECTOR DE INDICES ==========
        selector_frame = ttk.LabelFrame(main_frame, text="5. Indices a Calcular", padding=10)
        selector_frame.pack(anchor="w", fill="x", padx=15, pady=5)
        
        btn_frame = ttk.Frame(selector_frame)
        btn_frame.pack(anchor="w", pady=5)
        ttk.Button(btn_frame, text="Seleccionar Todo", 
                  command=self.select_all_indices).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Deseleccionar Todo",
                  command=self.deselect_all_indices).pack(side="left", padx=2)
        
        # Canvas para checkboxes
        cb_frame = ttk.Frame(selector_frame)
        cb_frame.pack(fill=tk.BOTH, expand=True)
        
        cb_canvas = tk.Canvas(cb_frame, bg="white", height=180)
        cb_scroll = ttk.Scrollbar(cb_frame, orient="vertical", command=cb_canvas.yview)
        cb_inner = ttk.Frame(cb_canvas)
        
        cb_inner.bind("<Configure>",
            lambda e: cb_canvas.configure(scrollregion=cb_canvas.bbox("all")))
        cb_canvas.create_window((0, 0), window=cb_inner, anchor="nw")
        cb_canvas.configure(yscrollcommand=cb_scroll.set)
        
        def _cb_wheel(event):
            cb_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        cb_canvas.bind("<MouseWheel>", _cb_wheel)
        
        for idx in self.INDICES_COMPLETE:
            row = ttk.Frame(cb_inner)
            row.pack(anchor="w", fill="x", padx=5, pady=1)
            
            self.checkbox_refs[idx] = {'frame': row, 'number_label': None}
            
            cb = ttk.Checkbutton(row, text=f"{idx} - {self.INDICES_DESCRIPTIONS[idx]}",
                                variable=self.selected_indices[idx],
                                command=self.on_index_changed)  # Cambiar aquÃ­
            cb.pack(side="left", fill="x", expand=True)
            cb.bind("<MouseWheel>", _cb_wheel)
            
            num_lbl = ttk.Label(row, text="", font=("Arial", 10, "bold"),
                               foreground="#1f4788", width=4)
            num_lbl.pack(side="right", padx=5)
            self.checkbox_refs[idx]['number_label'] = num_lbl
        
        self.update_dynamic_numbering()
        
        cb_canvas.pack(side="left", fill=tk.BOTH, expand=True)
        cb_scroll.pack(side="right", fill="y")
        
        # ========== 6. BOTONES DE ACCIÃ“N ==========
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(pady=15)
        
        ttk.Button(action_frame, text="[OK] Procesar", command=self.process,
                  width=20).pack(side="left", padx=10)
        ttk.Button(action_frame, text="Salir", command=self.root.quit,
                  width=15).pack(side="left", padx=10)
        
        ttk.Label(main_frame, text="Esteban Caffiero | estebancaffiero7@gmail.com",
                 font=("Arial", 8), foreground="gray").pack(pady=10)
        
        main_canvas.pack(side="left", fill=tk.BOTH, expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def toggle_band_mode(self):
        """Alterna entre modo multibanda y bandas manuales"""
        if self.manual_band_mode.get():
            # Mostrar bandas manuales, ocultar multibanda
            self.multiband_frame.pack_forget()
            # Insertar despuÃ©s del frame de modo de entrada (primer hijo del main_frame)
            self.manual_bands_frame.pack(anchor="w", fill="x", padx=15, pady=5, 
                                        after=self.multiband_frame.master.winfo_children()[0])
        else:
            # Mostrar multibanda, ocultar bandas manuales
            self.manual_bands_frame.pack_forget()
            self.multiband_frame.pack(anchor="w", fill="x", padx=15, pady=5,
                                     after=self.multiband_frame.master.winfo_children()[0])
    
    def select_band_file(self, band_key):
        """Selecciona archivo para una banda especÃ­fica"""
        file = filedialog.askopenfilename(
            title=f"Seleccionar banda {band_key}",
            filetypes=[("TIFF files", "*.tif *.TIF *.tiff"), ("All files", "*.*")]
        )
        if file:
            self.band_files[band_key].set(file)
    
    def update_dynamic_numbering(self):
        """Actualiza numeraciÃ³n dinÃ¡mica"""
        counter = 1
        for idx in self.INDICES_COMPLETE:
            lbl = self.checkbox_refs[idx]['number_label']
            if self.selected_indices[idx].get():
                lbl.config(text=f"[{counter}]", foreground="#1f4788")
                counter += 1
            else:
                lbl.config(text="", foreground="#999999")
    
    def on_index_changed(self):
        """Callback cuando cambia la selecciÃ³n de un Ã­ndice"""
        self.update_dynamic_numbering()
        self.update_guide_content()
    
    def select_all_indices(self):
        for idx in self.INDICES_COMPLETE:
            self.selected_indices[idx].set(True)
        self.update_dynamic_numbering()
        self.update_guide_content()
    
    def deselect_all_indices(self):
        for idx in self.INDICES_COMPLETE:
            self.selected_indices[idx].set(False)
        self.update_dynamic_numbering()
        self.update_guide_content()
    
    def create_guide_tab(self):
        """PestaÃ±a de guÃ­a de Ã­ndices"""
        container = ttk.Frame(self.tab_guide)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Crear el widget de texto para la guÃ­a
        self.guide_text = tk.Text(container, wrap="word", bg="white", font=("Courier", 9))
        scroll = ttk.Scrollbar(container, orient="vertical", command=self.guide_text.yview)
        self.guide_text.configure(yscrollcommand=scroll.set)
        
        # Scroll con rueda del ratÃ³n
        def _on_mousewheel(event):
            self.guide_text.yview_scroll(int(-1*(event.delta/120)), "units")
        self.guide_text.bind("<MouseWheel>", _on_mousewheel)
        
        self.guide_text.pack(side="left", fill=tk.BOTH, expand=True)
        scroll.pack(side="right", fill="y")
        
        # Contenido inicial
        self.update_guide_content()
    
    def update_guide_content(self):
        """Actualiza el contenido de la guÃ­a segÃºn Ã­ndices seleccionados"""
        if hasattr(self, 'guide_text'):
            selected = [i for i in self.INDICES_COMPLETE if self.selected_indices[i].get()]
            
            content = "=" * 50 + "\n"
            content += f"   \u00cdNDICES SELECCIONADOS ({len(selected)}/40)\n"
            content += "=" * 50 + "\n\n"
            
            if not selected:
                content += "[!] No hay \u00edndices seleccionados.\n\n"
                content += "Selecciona al menos uno en la pesta\u00f1a 'Procesamiento'."
            else:
                for i, idx in enumerate(selected, 1):
                    desc = self.INDICES_DESCRIPTIONS.get(idx, "Sin descripcion")
                    formula = self.INDICES_FORMULAS.get(idx, "N/A")
                    source = self.INDEX_SOURCES.get(idx, "N/A")
                    
                    content += f"[{i:2d}] {idx}\n"
                    content += f"     \u251c\u2500 {desc}\n"
                    content += f"     \u251c\u2500 Formula: {formula}\n"
                    content += f"     \u2514\u2500 Fuente: {source}\n\n"
            
            self.guide_text.configure(state="normal")
            self.guide_text.delete(1.0, tk.END)
            self.guide_text.insert(1.0, content)
            self.guide_text.configure(state="disabled")
    
    def select_input_file(self):
        file = filedialog.askopenfilename(
            title="Seleccionar archivo TIF",
            filetypes=[("TIFF files", "*.tif *.TIF *.tiff"), ("All files", "*.*")]
        )
        if file:
            self.input_file.set(file)
    
    def _create_edicion_tab(self):
        """Crea la pesta\u00f1a Edici\u00f3n (Crop + Rotaci\u00f3n)"""
        self.tab_edicion = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_edicion, text="Edici\u00f3n")
        self.edicion_tab = EdicionTab(self.tab_edicion, self)
    
    def _create_parametros_tab(self):
        """Crea la pestaÃ±a ParÃ¡metros (anÃ¡lisis de ROIs) - mÃ³dulo independiente."""
        try:
            from parametros_tab_v72 import create_parametros_tab
            self.parametros_tab = create_parametros_tab(self.notebook)
        except ImportError as e:
            # Si falla, crear pestaÃ±a con mensaje de error
            tab_error = ttk.Frame(self.notebook)
            self.notebook.add(tab_error, text="Par\u00e1metros")
            ttk.Label(
                tab_error,
                text=f"[!] No se pudo cargar el m\u00f3dulo parametros_tab.py\n\nError: {e}",
                font=("Arial", 11),
                foreground="red"
            ).pack(pady=50)
    
    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Seleccionar carpeta")
        if directory:
            self.output_dir.set(directory)
    
    def process(self):
        """Inicia procesamiento en hilo separado"""
        # Verificar cambios no guardados en edici\u00f3n
        if hasattr(self, 'edicion_tab') and self.edicion_tab.has_unsaved_changes():
            if not messagebox.askyesno(
                "Cambios sin guardar",
                "Hay cambios de edici\u00f3n no guardados.\\n\\u00bfDesea continuar sin guardarlos?"
            ):
                return
        
        # Validaciones
        if not self.manual_band_mode.get() and not self.input_file.get():
            messagebox.showerror("Error", "Selecciona un archivo de entrada")
            return
        
        if self.manual_band_mode.get():
            for band_name, var in self.band_files.items():
                if not var.get():
                    messagebox.showerror("Error", f"Falta seleccionar banda: {band_name}")
                    return
        
        if not self.output_dir.get():
            messagebox.showerror("Error", "Selecciona carpeta de salida")
            return
        
        thread = threading.Thread(target=self._process_thread)
        thread.start()
    
    def _process_thread(self):
        """Procesamiento en hilo - thread-safe"""
        try:
            self.status_var.set("Procesando...")
            self.root.update()
            
            selected = [i for i in self.INDICES_COMPLETE if self.selected_indices[i].get()]
            
            if not selected:
                self.root.after(0, lambda: messagebox.showerror("Error", "Selecciona al menos un Ã­ndice"))
                self.status_var.set("Error")
                return
            
            # Preparar parÃ¡metros
            manual_bands = None
            input_file = None
            
            if self.manual_band_mode.get():
                manual_bands = {k: v.get() for k, v in self.band_files.items()}
            else:
                input_file = self.input_file.get()
            
            output_file = os.path.join(self.output_dir.get(), f"{self.output_name.get()}.tif")
            nodata = float(self.nodata_value.get())
            mode = self.output_mode.get()
            
            # Obtener parametros de calibracion
            apply_calib = self.apply_calibration.get()
            auto_calib = self.auto_calibrate.get()
            calib_div = float(self.calib_divisor.get()) if self.calib_divisor.get() else 65535.0
            calib_mult = float(self.calib_multiplier.get()) if self.calib_multiplier.get() else 1.0
            
            # Procesar
            result = write_indices_geotiff(
                input_file,
                output_file,
                nodata_value=nodata,
                selected_indices=selected,
                output_mode=mode,
                manual_bands=manual_bands,
                apply_calibration=apply_calib,
                calib_divisor=calib_div,
                calib_multiplier=calib_mult,
                auto_calibrate=auto_calib
            )
            
            self.status_var.set("Completado!")
            
            # Mostrar mensaje desde hilo principal
            if mode == "single":
                msg = f"Guardado: {output_file}\n\n{len(selected)} Ã­ndices"
            else:
                msg = f"Creados {result} archivos en:\n{self.output_dir.get()}"
            
            self.root.after(0, lambda: messagebox.showinfo("Ã‰xito", msg))
            
        except Exception as e:
            self.status_var.set("Error")
            error_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error:\n{error_msg}"))
    
    def _reset_all_sessions(self):
        """
        Limpia toda la memoria y resetea el estado de la aplicación.
        Útil después de procesar imágenes grandes para liberar RAM.
        """
        # Confirmar antes de limpiar
        if not messagebox.askyesno("Nueva Sesión",
            "¿Descartar todo y empezar una nueva sesión?\n\n"
            "Esto liberará la memoria y limpiará:\n"
            "• Pestaña Edición (imagen cargada)\n"
            "• Pestaña Procesamiento (rutas de archivos)\n"
            "• Pestaña Parámetros (ROIs y datos)"):
            return
        
        try:
            # === LIMPIAR PESTAÑA EDICIÓN ===
            if hasattr(self, 'edicion_tab') and self.edicion_tab is not None:
                session = self.edicion_tab.edit_session
                # Liberar arrays numpy
                session.original_data = None
                session.edited_data = None
                session.original_profile = None
                session.edited_profile = None
                session.source_filepath = None
                session.is_modified = False
                session.is_saved = False
                session.operations = []
                session.crop_roi = None
                session.rotation_angle = 0.0
                
                # Limpiar canvas y mostrar placeholder
                self.edicion_tab.edit_canvas.delete("all")
                if hasattr(self.edicion_tab.edit_canvas, 'photo_image'):
                    self.edicion_tab.edit_canvas.photo_image = None
                self.edicion_tab._preview_rotation = 0
                self.edicion_tab.rotation_angle.set(0)
                self.edicion_tab._show_placeholder()
                self.edicion_tab._update_info()
            
            # === LIMPIAR PESTAÑA PROCESAMIENTO ===
            self.input_file.set("")
            self.output_dir.set("")
            self.output_name.set("INDICES_RESULTADO")
            for band_var in self.band_files.values():
                band_var.set("")
            
            # === LIMPIAR PESTAÑA PARÁMETROS ===
            if hasattr(self, 'parametros_tab') and self.parametros_tab is not None:
                # El módulo parametros_tab tiene su propia estructura
                if hasattr(self.parametros_tab, 'image_loader'):
                    loader = self.parametros_tab.image_loader
                    if hasattr(loader, 'data'):
                        loader.data = None
                    if hasattr(loader, 'profile'):
                        loader.profile = None
                    if hasattr(loader, '_cleanup_temp'):
                        loader._cleanup_temp()
                    if hasattr(loader, 'filepath'):
                        loader.filepath = None
                
                if hasattr(self.parametros_tab, 'roi_manager'):
                    self.parametros_tab.roi_manager.clear()
                
                if hasattr(self.parametros_tab, 'current_filepath'):
                    self.parametros_tab.current_filepath = None
                
                if hasattr(self.parametros_tab, 'interactive_canvas'):
                    canvas = self.parametros_tab.interactive_canvas
                    if hasattr(canvas, 'clear_image'):
                        canvas.clear_image()
                    if hasattr(canvas, 'photo_image'):
                        canvas.photo_image = None
                
                if hasattr(self.parametros_tab, '_show_placeholder'):
                    self.parametros_tab._show_placeholder()
            
            # === FORZAR GARBAGE COLLECTION ===
            import gc
            gc.collect()
            
            self.status_var.set("✓ Sesión limpiada - Memoria liberada")
            print("[RESET] Sesión limpiada y memoria liberada")
            
        except Exception as e:
            print(f"[RESET] Error durante limpieza: {e}")
            self.status_var.set(f"Error al limpiar: {e}")


if __name__ == "__main__":
    try:
        if not show_splash_and_check_deps():
            sys.exit(1)
        
        root = tk.Tk()
        gui = IndicadorGUI(root)
        root.mainloop()
    except Exception as e:
        import traceback
        # Mostrar error en consola y en messagebox
        error_msg = traceback.format_exc()
        print(f"ERROR CRÃTICO:\n{error_msg}")
        try:
            messagebox.showerror("Error CrÃ­tico", f"La aplicaciÃ³n fallÃ³:\n\n{str(e)}\n\nRevisa la consola para mÃ¡s detalles.")
        except:
            pass
        input("Presiona Enter para cerrar...")
