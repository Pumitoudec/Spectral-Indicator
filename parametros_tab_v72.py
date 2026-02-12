# -*- coding: utf-8 -*-
"""
PESTAÑA PARÁMETROS - Spectral Indicator v5.0
Módulo independiente para análisis de ROIs sobre GeoTIFF multibanda.

Arquitectura:
- ImageLoader: Carga GeoTIFF con rasterio, genera preview
- ROIManager: Almacena ROIs con IDs auto-incrementales
- InteractiveCanvas: Canvas Tkinter con eventos de dibujo
- StatsCalculator: Calcula promedios por banda/ROI
- CSVExporter: Exporta resultados a CSV
- ParametrosTab: Orquesta todo el flujo
"""

import os
import csv
import math
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
import numpy as np

try:
    import rasterio
    from rasterio.shutil import copy as rio_copy
except ImportError:
    rasterio = None

try:
    from PIL import Image, ImageTk, ImageDraw
except ImportError:
    Image = None
    ImageTk = None
    ImageDraw = None


# ==================== ESTRUCTURAS DE DATOS ====================

@dataclass
class ROI:
    """ROI con soporte para nombre y rotación."""
    id: int
    x1: int
    y1: int
    x2: int
    y2: int
    name: str = ""
    angle: float = 0.0

    def with_id(self, new_id: int) -> 'ROI':
        """Retorna copia con nuevo ID, preservando name y angle."""
        return ROI(id=new_id, x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2,
                   name=self.name, angle=self.angle)

    def with_angle(self, new_angle: float) -> 'ROI':
        """Retorna copia con nuevo ángulo."""
        return ROI(id=self.id, x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2,
                   name=self.name, angle=new_angle)

    def center(self) -> Tuple[float, float]:
        """Retorna el centro (cx, cy) del ROI."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def half_size(self) -> Tuple[float, float]:
        """Retorna (half_width, half_height)."""
        return ((self.x2 - self.x1) / 2, (self.y2 - self.y1) / 2)

    def rotated_corners(self) -> List[Tuple[float, float]]:
        """Retorna las 4 esquinas rotadas del ROI."""
        cx, cy = self.center()
        hw, hh = self.half_size()
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        if self.angle == 0:
            return [(cx + dx, cy + dy) for dx, dy in corners]
        rad = math.radians(self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        return [
            (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
            for dx, dy in corners
        ]

    def display_label(self) -> str:
        """Etiqueta para mostrar en canvas."""
        if self.name:
            return f"{self.id}: {self.name}"
        return f"ROI {self.id}"


# ==================== IMAGE LOADER ====================

class ImageLoader:
    """
    Carga GeoTIFF con rasterio, obtiene array numpy (bands, H, W),
    maneja nodata, genera preview RGB o pseudocolor.
    """
    
    def __init__(self):
        self.data: Optional[np.ndarray] = None  # Shape: (bands, H, W)
        self.profile: Optional[dict] = None
        self.nodata_value: Optional[float] = None
        self.filepath: Optional[str] = None
        self.temp_path: Optional[str] = None
        # Cache multinivel para zoom rápido
        self._normalized_base: Optional[np.ndarray] = None  # RGB uint8 full-res
        self._preview_cache: Dict = {}  # {zoom_level: PIL.Image}
        self._cache_max_entries: int = 5
    
    def load(self, filepath: str) -> Tuple[np.ndarray, dict]:
        """
        Carga GeoTIFF universal (ImageJ, pixel-interleaved, multibanda normal).
        
        Returns:
            tuple: (numpy array [bands, H, W], profile dict)
        """
        if rasterio is None:
            raise ImportError("rasterio no está instalado")
        
        self.filepath = filepath
        self._cleanup_temp()
        self.invalidate_cache()  # Nueva carga = invalidar cache
        
        with rasterio.open(filepath) as src:
            tags = src.tags()
            desc = tags.get("TIFFTAG_IMAGEDESCRIPTION", "")
            
            # Caso A: ImageJ multi-directorio
            if "ImageJ" in desc and "images=" in desc:
                return self._load_imagej(filepath, desc)
            
            # Caso B: Pixel-interleaved
            if src.count == 1 and src.profile.get("interleave") == "pixel":
                return self._load_pixel_interleaved(filepath)
            
            # Caso C: Multibanda normal
            self.data = src.read()
            self.profile = src.profile.copy()
            self.nodata_value = src.nodata
            
            return self.data, self.profile
    
    def _load_imagej(self, filepath: str, desc: str) -> Tuple[np.ndarray, dict]:
        """Carga TIFF tipo ImageJ multi-directorio."""
        num_images = int(desc.split("images=")[1].split()[0])
        bands = []
        last_profile = None
        
        for i in range(1, num_images + 1):
            gdal_path = f"GTIFF_DIR:{i}:{filepath}"
            with rasterio.open(gdal_path) as ds:
                bands.append(ds.read(1))
                last_profile = ds.profile.copy()
        
        self.data = np.stack(bands)
        self.profile = last_profile
        self.nodata_value = last_profile.get('nodata')
        return self.data, self.profile
    
    def _load_pixel_interleaved(self, filepath: str) -> Tuple[np.ndarray, dict]:
        """Convierte pixel-interleaved a band-interleaved."""
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
        tmp.close()
        self.temp_path = tmp.name
        
        rio_copy(filepath, self.temp_path, interleave="band")
        
        with rasterio.open(self.temp_path) as ds:
            self.data = ds.read()
            self.profile = ds.profile.copy()
            self.nodata_value = ds.nodata
        
        return self.data, self.profile
    
    def generate_preview(self, max_size: int = 800):
        """
        Genera preview PIL Image normalizado a 8-bit.
        Usa cache multinivel: nivel 1 = imagen normalizada base,
        nivel 2 = previews resized por zoom_level.
        """
        if Image is None:
            raise ImportError("PIL/Pillow no está instalado")
        
        if self.data is None:
            raise ValueError("No hay datos cargados. Usa load() primero.")
        
        # Cache nivel 2: preview ya resize-ado para este zoom_level
        if max_size in self._preview_cache:
            return self._preview_cache[max_size].copy()
        
        bands, height, width = self.data.shape
        
        # Cache nivel 1: imagen normalizada a resolución completa
        if self._normalized_base is None:
            if bands >= 3:
                rgb = np.stack([
                    self._normalize_band(self.data[0]),
                    self._normalize_band(self.data[1]),
                    self._normalize_band(self.data[2])
                ], axis=-1)
                self._normalized_base = rgb
            else:
                gray = self._normalize_band(self.data[0])
                self._normalized_base = gray
        
        # Crear PIL Image desde cache nivel 1
        if self._normalized_base.ndim == 3:
            img = Image.fromarray(self._normalized_base, mode='RGB')
        else:
            img = Image.fromarray(self._normalized_base, mode='L')
        
        # Redimensionar para preview
        scale = min(max_size / width, max_size / height, 1.0)
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Guardar en cache nivel 2 (LRU simple)
        if len(self._preview_cache) >= self._cache_max_entries:
            oldest_key = next(iter(self._preview_cache))
            del self._preview_cache[oldest_key]
        self._preview_cache[max_size] = img.copy()
        
        return img
    
    def invalidate_cache(self):
        """Invalida todos los caches (llamar cuando cambian datos)."""
        self._normalized_base = None
        self._preview_cache.clear()
    
    def _normalize_band(self, band: np.ndarray) -> np.ndarray:
        """Normaliza banda a uint8 (0-255) ignorando nodata."""
        band = band.astype(np.float32)
        
        # Reemplazar nodata con NaN
        if self.nodata_value is not None:
            band = np.where(band == self.nodata_value, np.nan, band)
        
        # Calcular percentiles ignorando NaN
        with np.errstate(invalid='ignore'):
            p2 = np.nanpercentile(band, 2)
            p98 = np.nanpercentile(band, 98)
        
        # Normalizar a 0-255
        if p98 > p2:
            band = (band - p2) / (p98 - p2) * 255
        else:
            band = np.zeros_like(band)
        
        band = np.clip(band, 0, 255)
        band = np.where(np.isnan(band), 0, band)
        
        return band.astype(np.uint8)
    
    def get_dimensions(self) -> Tuple[int, int, int]:
        """Retorna (bands, height, width)."""
        if self.data is None:
            return (0, 0, 0)
        return self.data.shape
    
    def _cleanup_temp(self):
        """Limpia archivo temporal si existe."""
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.unlink(self.temp_path)
            except:
                pass
            self.temp_path = None
    
    def __del__(self):
        self._cleanup_temp()


# ==================== ROI MANAGER ====================

class ROIManager:
    """
    Almacena ROIs como coordenadas de imagen real.
    Asigna IDs auto-incrementales, valida límites.
    Soporta copia, matriz lineal y renumeración espacial.
    """
    
    def __init__(self, image_width: int = 0, image_height: int = 0):
        self.rois: List[ROI] = []
        self._next_id: int = 1
        self.image_width = image_width
        self.image_height = image_height
        # Clipboard para copiar/pegar ROIs
        self._clipboard: List[tuple] = []  # Lista de (width, height) de ROIs copiados
    
    def set_image_bounds(self, width: int, height: int):
        """Establece límites de imagen para validación."""
        self.image_width = width
        self.image_height = height
    
    def add_roi(self, x1: int, y1: int, x2: int, y2: int) -> ROI:
        """
        Añade nuevo ROI con validación y clamping.
        Asegura x1 < x2, y1 < y2.
        
        Returns:
            ROI creado
        """
        # Asegurar orden correcto
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Clamping a límites de imagen
        x1 = max(0, min(x1, self.image_width - 1))
        x2 = max(0, min(x2, self.image_width))
        y1 = max(0, min(y1, self.image_height - 1))
        y2 = max(0, min(y2, self.image_height))
        
        # Validar tamaño mínimo
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValueError("ROI demasiado pequeño (mínimo 2x2 píxeles)")
        
        roi = ROI(id=self._next_id, x1=x1, y1=y1, x2=x2, y2=y2)
        self.rois.append(roi)
        self._next_id += 1
        
        return roi
    
    def add_roi_unclamped(self, x1: int, y1: int, x2: int, y2: int) -> ROI:
        """
        Añade ROI SIN validación de límites (para copias fuera de imagen).
        Permite ROIs que excedan los bounds de la imagen.
        
        Returns:
            ROI creado
        """
        # Asegurar orden correcto
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Validar tamaño mínimo
        if x2 - x1 < 2 or y2 - y1 < 2:
            raise ValueError("ROI demasiado pequeño (mínimo 2x2 píxeles)")
        
        roi = ROI(id=self._next_id, x1=x1, y1=y1, x2=x2, y2=y2)
        self.rois.append(roi)
        self._next_id += 1
        
        return roi
    
    def copy_rois(self, roi_ids: List[int]) -> int:
        """
        Copia ROIs al clipboard interno.
        
        Args:
            roi_ids: Lista de IDs de ROIs a copiar
            
        Returns:
            Número de ROIs copiados
        """
        self._clipboard.clear()
        for roi_id in roi_ids:
            for roi in self.rois:
                if roi.id == roi_id:
                    # Guardar dimensiones y posición relativa
                    self._clipboard.append((roi.x1, roi.y1, roi.x2, roi.y2))
                    break
        return len(self._clipboard)
    
    def paste_rois(self, offset_x: int, offset_y: int) -> List[ROI]:
        """
        Pega ROIs del clipboard con desplazamiento.
        
        Args:
            offset_x: Desplazamiento horizontal
            offset_y: Desplazamiento vertical
            
        Returns:
            Lista de nuevos ROIs creados
        """
        new_rois = []
        for (x1, y1, x2, y2) in self._clipboard:
            new_roi = self.add_roi_unclamped(
                x1 + offset_x, y1 + offset_y,
                x2 + offset_x, y2 + offset_y
            )
            new_rois.append(new_roi)
        return new_rois
    
    def create_roi_array(self, roi_ids: List[int], copies: int, 
                         offset_x: int, offset_y: int) -> List[ROI]:
        """
        Crea matriz lineal de ROIs (similar a array de AutoCAD).
        
        Args:
            roi_ids: IDs de ROIs base para copiar
            copies: Número de copias a crear
            offset_x: Espaciado horizontal entre copias
            offset_y: Espaciado vertical entre copias
            
        Returns:
            Lista de todos los nuevos ROIs creados
        """
        if copies < 1:
            return []
        
        # Obtener ROIs originales
        source_rois = []
        for roi_id in roi_ids:
            for roi in self.rois:
                if roi.id == roi_id:
                    source_rois.append(roi)
                    break
        
        if not source_rois:
            return []
        
        new_rois = []
        for copy_num in range(1, copies + 1):
            for roi in source_rois:
                new_x1 = roi.x1 + (offset_x * copy_num)
                new_y1 = roi.y1 + (offset_y * copy_num)
                new_x2 = roi.x2 + (offset_x * copy_num)
                new_y2 = roi.y2 + (offset_y * copy_num)
                
                new_roi = self.add_roi_unclamped(new_x1, new_y1, new_x2, new_y2)
                new_rois.append(new_roi)
        
        return new_rois
    
    def renumber_by_position(self):
        """
        Renumera ROIs según posición espacial: izq→der, arriba→abajo.
        Ordena por centroide (Y primario, X secundario).
        Los IDs se reasignan consecutivamente desde 1.
        """
        if not self.rois:
            return
        
        # Calcular centroide para cada ROI
        def get_centroid(roi):
            cx = (roi.x1 + roi.x2) / 2
            cy = (roi.y1 + roi.y2) / 2
            return (cy, cx)  # Y primario, X secundario
        
        # Ordenar por posición (primero Y, luego X)
        sorted_rois = sorted(self.rois, key=get_centroid)
        
        # Reconstruir lista con IDs consecutivos
        self.rois.clear()
        self._next_id = 1
        
        for old_roi in sorted_rois:
            new_roi = old_roi.with_id(self._next_id)
            self.rois.append(new_roi)
            self._next_id += 1
    
    def get_roi_by_id(self, roi_id: int) -> Optional[ROI]:
        """Obtiene ROI por ID."""
        for roi in self.rois:
            if roi.id == roi_id:
                return roi
        return None
    
    def remove_roi(self, roi_id: int) -> bool:
        """Elimina ROI por ID."""
        for i, roi in enumerate(self.rois):
            if roi.id == roi_id:
                self.rois.pop(i)
                return True
        return False
    
    def change_roi_id(self, old_id: int, new_id: int) -> bool:
        """
        Cambia el ID de un ROI. Valida unicidad del nuevo ID.
        
        Args:
            old_id: ID actual del ROI
            new_id: Nuevo ID deseado
            
        Returns:
            True si se cambió exitosamente
        """
        if new_id < 1:
            raise ValueError("El ID debe ser >= 1")
        
        # Verificar que el nuevo ID no exista (a menos que sea el mismo)
        if old_id != new_id:
            for roi in self.rois:
                if roi.id == new_id:
                    raise ValueError(f"Ya existe un ROI con ID {new_id}")
        
        # Buscar y reemplazar
        for i, roi in enumerate(self.rois):
            if roi.id == old_id:
                self.rois[i] = roi.with_id(new_id)
                # Actualizar _next_id si es necesario
                self._next_id = max(self._next_id, new_id + 1)
                return True
        return False
    
    def swap_roi_ids(self, id_a: int, id_b: int) -> bool:
        """
        Intercambia los IDs de dos ROIs.
        
        Returns:
            True si se intercambiaron exitosamente
        """
        idx_a = idx_b = None
        for i, roi in enumerate(self.rois):
            if roi.id == id_a:
                idx_a = i
            elif roi.id == id_b:
                idx_b = i
        
        if idx_a is None or idx_b is None:
            return False
        
        roi_a = self.rois[idx_a]
        roi_b = self.rois[idx_b]
        self.rois[idx_a] = roi_a.with_id(id_b)
        self.rois[idx_b] = roi_b.with_id(id_a)
        return True
    
    def reorder_roi_list(self, ordered_ids: List[int]):
        """
        Reordena la lista interna de ROIs según la lista de IDs dada.
        Los IDs NO cambian, solo el orden interno.
        
        Args:
            ordered_ids: Lista de IDs en el orden deseado
        """
        id_to_roi = {roi.id: roi for roi in self.rois}
        new_list = []
        for rid in ordered_ids:
            if rid in id_to_roi:
                new_list.append(id_to_roi[rid])
        # Añadir ROIs no listados al final
        listed = set(ordered_ids)
        for roi in self.rois:
            if roi.id not in listed:
                new_list.append(roi)
        self.rois = new_list
    
    # ========== DETECCIÓN DE GRILLA Y OPERACIONES FLIP ==========
    
    def detect_grid(self, tolerance_ratio: float = 0.3) -> Optional[List[List[ROI]]]:
        """
        Detecta si los ROIs forman una grilla rectangular.
        Agrupa por filas usando proximidad de centroide Y.
        
        Args:
            tolerance_ratio: Fracción de la altura media de ROI usada como tolerancia
            
        Returns:
            Lista de filas (cada fila es lista de ROIs ordenados por X),
            o None si no se detecta grilla válida.
        """
        if len(self.rois) < 2:
            return None
        
        # Calcular centroides
        centroids = []
        for roi in self.rois:
            cx = (roi.x1 + roi.x2) / 2
            cy = (roi.y1 + roi.y2) / 2
            h = roi.y2 - roi.y1
            centroids.append((roi, cx, cy, h))
        
        # Tolerancia basada en altura media
        avg_height = sum(c[3] for c in centroids) / len(centroids)
        tolerance = avg_height * tolerance_ratio
        
        # Agrupar por filas (proximidad en Y)
        centroids.sort(key=lambda c: c[2])  # Ordenar por Y
        
        rows: List[List] = []
        current_row = [centroids[0]]
        current_y = centroids[0][2]
        
        for c in centroids[1:]:
            if abs(c[2] - current_y) <= tolerance:
                current_row.append(c)
            else:
                rows.append(current_row)
                current_row = [c]
                current_y = c[2]
        rows.append(current_row)
        
        # Ordenar cada fila por X y extraer solo ROIs
        grid = []
        for row in rows:
            row.sort(key=lambda c: c[1])  # Ordenar por X
            grid.append([c[0] for c in row])
        
        return grid
    
    def flip_horizontal(self):
        """
        Invierte el orden de numeración horizontalmente.
        Detecta grilla y dentro de cada fila invierte los IDs.
        """
        grid = self.detect_grid()
        if grid is None:
            return False
        
        # Recoger pares (posición_roi, id_actual) por fila
        swaps = []
        for row in grid:
            if len(row) < 2:
                continue
            ids = [roi.id for roi in row]
            reversed_ids = list(reversed(ids))
            for roi, new_id in zip(row, reversed_ids):
                swaps.append((roi, new_id))
        
        # Aplicar cambios usando IDs temporales negativos para evitar colisiones
        temp_id = -1
        temp_map = {}
        for roi, new_id in swaps:
            for i, r in enumerate(self.rois):
                if r.id == roi.id and roi.id not in temp_map.values():
                    self.rois[i] = r.with_id(temp_id)
                    temp_map[temp_id] = new_id
                    temp_id -= 1
                    break
        
        # Reemplazar IDs temporales con definitivos
        for i, r in enumerate(self.rois):
            if r.id in temp_map:
                self.rois[i] = r.with_id(temp_map[r.id])
        
        self._next_id = max(r.id for r in self.rois) + 1
        return True
    
    def flip_vertical(self):
        """
        Invierte el orden de numeración verticalmente.
        Detecta grilla y entre filas invierte los IDs por columna.
        """
        grid = self.detect_grid()
        if grid is None:
            return False
        
        # Obtener IDs en orden de grilla (fila por fila)
        all_ids = []
        for row in grid:
            all_ids.append([roi.id for roi in row])
        
        # Invertir filas
        reversed_ids = list(reversed(all_ids))
        
        # Aplicar cambios
        swaps = []
        for row_rois, new_id_row in zip(grid, reversed_ids):
            for roi, new_id in zip(row_rois, new_id_row):
                swaps.append((roi, new_id))
        
        # Usar IDs temporales para evitar colisiones
        temp_id = -1
        temp_map = {}
        for roi, new_id in swaps:
            for i, r in enumerate(self.rois):
                if r.id == roi.id and r.id > 0:
                    self.rois[i] = r.with_id(temp_id)
                    temp_map[temp_id] = new_id
                    temp_id -= 1
                    break
        
        for i, r in enumerate(self.rois):
            if r.id in temp_map:
                self.rois[i] = r.with_id(temp_map[r.id])
        
        self._next_id = max(r.id for r in self.rois) + 1
        return True
    
    def transpose_grid(self):
        """
        Transpone la numeración de la grilla (filas ↔ columnas).
        ROI(1,1) mantiene pos, ROI(1,2)↔ROI(2,1), etc.
        """
        grid = self.detect_grid()
        if grid is None:
            return False
        
        num_rows = len(grid)
        num_cols = max(len(row) for row in grid)
        
        # Necesita grilla rectangular
        for row in grid:
            if len(row) != num_cols:
                return False
        
        # Generar nuevo orden: leer por columnas en vez de filas
        original_ids = []
        for row in grid:
            original_ids.extend([roi.id for roi in row])
        
        transposed_ids = []
        for col in range(num_cols):
            for row in range(num_rows):
                transposed_ids.append(grid[row][col].id)
        
        # Mapear: posición N en original_ids → ID de transposed_ids[N]
        swaps = []
        flat_rois = [roi for row in grid for roi in row]
        for roi, new_id in zip(flat_rois, transposed_ids):
            swaps.append((roi, new_id))
        
        # Aplicar con IDs temporales
        temp_id = -1
        temp_map = {}
        for roi, new_id in swaps:
            for i, r in enumerate(self.rois):
                if r.id == roi.id and r.id > 0:
                    self.rois[i] = r.with_id(temp_id)
                    temp_map[temp_id] = new_id
                    temp_id -= 1
                    break
        
        for i, r in enumerate(self.rois):
            if r.id in temp_map:
                self.rois[i] = r.with_id(temp_map[r.id])
        
        self._next_id = max(r.id for r in self.rois) + 1
        return True
    
    def clear(self):
        """Limpia todos los ROIs y reinicia contador."""
        self.rois.clear()
        self._next_id = 1
        self._clipboard.clear()
    
    def get_roi_count(self) -> int:
        """Retorna número de ROIs."""
        return len(self.rois)
    
    def get_all_rois(self) -> List[ROI]:
        """Retorna copia de lista de ROIs."""
        return list(self.rois)
    
    def has_clipboard(self) -> bool:
        """Retorna True si hay ROIs en el clipboard."""
        return len(self._clipboard) > 0

    def rename_roi(self, roi_id: int, new_name: str) -> bool:
        """Renombra un ROI (asigna nombre de texto)."""
        for i, roi in enumerate(self.rois):
            if roi.id == roi_id:
                self.rois[i] = ROI(id=roi.id, x1=roi.x1, y1=roi.y1,
                                   x2=roi.x2, y2=roi.y2,
                                   name=new_name, angle=roi.angle)
                return True
        return False

    def rotate_rois(self, roi_ids: List[int], angle_delta: float):
        """
        Rota los ROIs seleccionados. Acumula el ángulo sobre el angle existente.
        
        Args:
            roi_ids: IDs de ROIs a rotar
            angle_delta: Incremento de ángulo en grados (positivo = horario)
        """
        for i, roi in enumerate(self.rois):
            if roi.id in roi_ids:
                new_angle = (roi.angle + angle_delta) % 360
                if new_angle > 180:
                    new_angle -= 360
                self.rois[i] = roi.with_angle(new_angle)

    def set_roi_angle(self, roi_ids: List[int], angle: float):
        """
        Establece ángulo absoluto para los ROIs seleccionados.
        
        Args:
            roi_ids: IDs de ROIs
            angle: Ángulo absoluto en grados
        """
        for i, roi in enumerate(self.rois):
            if roi.id in roi_ids:
                self.rois[i] = roi.with_angle(angle)

    def invert_selected_order(self, selected_ids: List[int]) -> bool:
        """
        Invierte el orden de IDs de los ROIs seleccionados.
        Si seleccionados son [1, 2, 3, 4, 5, 6] quedan [6, 5, 4, 3, 2, 1]
        Las posiciones físicas NO cambian, solo los números de ID.
        
        Args:
            selected_ids: Lista de IDs de ROIs a invertir
            
        Returns:
            True si se invirtió exitosamente
        """
        if len(selected_ids) < 2:
            return False
        
        # Obtener los ROIs seleccionados ordenados por ID actual
        selected_rois = [roi for roi in self.rois if roi.id in selected_ids]
        selected_rois.sort(key=lambda r: r.id)
        
        # IDs originales ordenados
        original_ids = [r.id for r in selected_rois]
        # IDs invertidos: [1,2,3,4,5,6] -> [6,5,4,3,2,1]
        inverted_ids = original_ids[::-1]
        
        # Usar IDs temporales negativos para evitar colisiones
        temp_id = -1
        temp_map = {}  # temp_id → new_id final
        
        for roi, new_id in zip(selected_rois, inverted_ids):
            for i, r in enumerate(self.rois):
                if r.id == roi.id:
                    self.rois[i] = r.with_id(temp_id)
                    temp_map[temp_id] = new_id
                    temp_id -= 1
                    break
        
        # Reemplazar IDs temporales con definitivos
        for i, r in enumerate(self.rois):
            if r.id in temp_map:
                self.rois[i] = r.with_id(temp_map[r.id])
        
        self._next_id = max(r.id for r in self.rois) + 1
        return True


# ==================== INTERACTIVE CANVAS ====================

class InteractiveCanvas:
    """
    Renderiza preview en Tkinter Canvas, maneja eventos mouse
    y dibuja rectángulos ROI. Soporta selección múltiple.
    """
    
    # Colores para ROIs (ciclo)
    ROI_COLORS = [
        "#FF6B6B",  # Rojo
        "#4ECDC4",  # Cyan
        "#FFE66D",  # Amarillo
        "#95E1D3",  # Verde menta
        "#F38181",  # Coral
        "#AA96DA",  # Lavanda
        "#FCBAD3",  # Rosa
        "#A8D8EA",  # Azul claro
    ]
    
    SELECTION_COLOR = "#00FF00"  # Verde brillante para selección
    
    def __init__(self, parent: tk.Widget, roi_manager: ROIManager,
                 on_roi_created: callable = None,
                 on_selection_changed: callable = None,
                 on_roi_double_click: callable = None,
                 xscrollcommand=None, yscrollcommand=None):
        """
        Args:
            parent: Widget padre
            roi_manager: Instancia de ROIManager
            on_roi_created: Callback cuando se crea un ROI
            on_selection_changed: Callback cuando cambia la selección
            on_roi_double_click: Callback cuando se hace doble-click en un ROI (roi_id)
            xscrollcommand: Comando para scrollbar horizontal
            yscrollcommand: Comando para scrollbar vertical
        """
        self.parent = parent
        self.roi_manager = roi_manager
        self.on_roi_created = on_roi_created
        self.on_selection_changed = on_selection_changed
        self.on_roi_double_click = on_roi_double_click
        
        # Estado
        self.photo_image = None  # ImageTk.PhotoImage cuando hay imagen
        self.scale_factor: float = 1.0
        self.image_offset_x: int = 0
        self.image_offset_y: int = 0
        
        # Dibujo temporal
        self._drawing = False
        self._start_x = 0
        self._start_y = 0
        self._temp_rect = None
        
        # Modo de selección y ROIs seleccionados
        self._selection_mode = False
        self._selected_rois: set = set()  # Set de ROI IDs seleccionados
        
        # Modo paste visual (arrastrar con cursor)
        self._paste_mode = False
        self._ghost_items: List[int] = []  # IDs de rectángulos fantasma
        self._paste_anchor: Tuple[float, float] = (0, 0)  # Punto ancla (centroide)
        self._paste_offsets: List[Tuple[int, int, int, int]] = []  # Offsets de ROIs relativos al ancla
        self._last_mouse_pos: Tuple[int, int] = (0, 0)  # Última posición del mouse
        self._paste_callback: callable = None  # Callback al confirmar paste
        self._paste_count: int = 0  # Contador de pastes en sesión actual
        self._on_paste_status: callable = None  # Callback para actualizar status
        
        # Herramienta de medición
        self._ruler_mode = False  # Modo regla dedicado (medir sin crear ROI)
        self._geo_transform = None  # Transform del GeoTIFF para calcular metros
        self._geo_pixel_size: Optional[Tuple[float, float]] = None  # (px_width, px_height) en unidades del CRS
        self._crs_unit: Optional[str] = None  # Unidad del CRS (m, ft, degrees, etc.)
        
        # Crear canvas con soporte scroll
        self.canvas = tk.Canvas(
            parent,
            bg="#2d2d2d",
            highlightthickness=1,
            highlightbackground="#555555",
            cursor="crosshair",
            xscrollcommand=xscrollcommand,
            yscrollcommand=yscrollcommand
        )
        
        # Bindings de mouse
        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Control-Button-1>", self._on_ctrl_click)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
    
    def pack(self, **kwargs):
        """Empaqueta el canvas."""
        self.canvas.pack(**kwargs)
    
    def grid(self, **kwargs):
        """Grid layout para el canvas."""
        self.canvas.grid(**kwargs)
    
    def set_image(self, pil_image, original_width: int, original_height: int):
        """
        Establece imagen en el canvas.
        
        Args:
            pil_image: Imagen PIL (ya redimensionada para preview)
            original_width: Ancho original de la imagen
            original_height: Alto original de la imagen
        """
        if ImageTk is None:
            raise ImportError("PIL/Pillow no está instalado")
        
        self.photo_image = ImageTk.PhotoImage(pil_image)
        preview_width = pil_image.width
        preview_height = pil_image.height
        
        # Calcular factor de escala: coord_imagen = coord_canvas / scale_factor
        self.scale_factor = preview_width / original_width
        
        # Configurar tamaño del canvas y scrollregion
        self.canvas.configure(width=preview_width, height=preview_height)
        self.canvas.configure(scrollregion=(0, 0, preview_width, preview_height))
        
        # Dibujar imagen centrada
        self.canvas.delete("all")
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.canvas.create_image(
            self.image_offset_x, self.image_offset_y,
            anchor="nw", image=self.photo_image, tags="background"
        )
        
        # Redibujar ROIs existentes
        self._redraw_all_rois()
    
    def clear_image(self):
        """Limpia imagen del canvas."""
        self.canvas.delete("all")
        self.photo_image = None
        self.scale_factor = 1.0
    
    def _on_press(self, event):
        """Inicio de dibujo de rectángulo o selección."""
        if self.photo_image is None:
            return
        
        # En modo paste, confirmar posición
        if self._paste_mode:
            self._confirm_paste()
            return
        
        # En modo selección, el click simple selecciona un ROI
        if self._selection_mode:
            self._handle_selection_click(event)
            return
        
        self._drawing = True
        # Convertir coordenadas de pantalla a coordenadas de canvas (considerando scroll)
        self._start_x = self.canvas.canvasx(event.x)
        self._start_y = self.canvas.canvasy(event.y)
        
        # Crear rectángulo temporal
        self._temp_rect = self.canvas.create_rectangle(
            self._start_x, self._start_y, self._start_x, self._start_y,
            outline="#FFFFFF", width=2, dash=(4, 4), tags="temp"
        )
    
    def _on_drag(self, event):
        """Actualiza rectángulo mientras se arrastra y muestra mediciones."""
        if not self._drawing or self._temp_rect is None:
            return
        
        # Convertir coordenadas de pantalla a coordenadas de canvas (considerando scroll)
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        
        self.canvas.coords(
            self._temp_rect,
            self._start_x, self._start_y, cx, cy
        )
        
        # Mostrar overlay de medición en tiempo real
        self._draw_measurement_overlay(self._start_x, self._start_y, cx, cy)
    
    def _on_release(self, event):
        """Finaliza dibujo y crea ROI."""
        if not self._drawing:
            return
        
        self._drawing = False
        
        # Eliminar rectángulo temporal y overlay de medición
        if self._temp_rect:
            self.canvas.delete(self._temp_rect)
            self._temp_rect = None
        self._clear_measurement_overlay()
        
        # En modo regla, no crear ROI
        if self._ruler_mode:
            return
        
        # Convertir coordenadas de pantalla a coordenadas de canvas (considerando scroll)
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Convertir coordenadas canvas -> imagen original
        img_x1 = int((self._start_x - self.image_offset_x) / self.scale_factor)
        img_y1 = int((self._start_y - self.image_offset_y) / self.scale_factor)
        img_x2 = int((end_x - self.image_offset_x) / self.scale_factor)
        img_y2 = int((end_y - self.image_offset_y) / self.scale_factor)
        
        try:
            roi = self.roi_manager.add_roi(img_x1, img_y1, img_x2, img_y2)
            self._draw_roi(roi)
            
            if self.on_roi_created:
                self.on_roi_created(roi)
                
        except ValueError as e:
            # ROI muy pequeño, ignorar
            pass
    
    def _on_right_click(self, event):
        """Click derecho: eliminar ROI bajo cursor, o cancelar paste."""
        if self.photo_image is None:
            return
        
        # En modo paste, cancelar
        if self._paste_mode:
            self._cancel_paste()
            return
        
        # Convertir coordenadas de pantalla a coordenadas de canvas (considerando scroll)
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        
        # Buscar ROI bajo el cursor
        items = self.canvas.find_overlapping(cx - 2, cy - 2, cx + 2, cy + 2)
        
        for item in items:
            tags = self.canvas.gettags(item)
            for tag in tags:
                if tag.startswith("roi_"):
                    try:
                        roi_id = int(tag.split("_")[1])
                        if self.roi_manager.remove_roi(roi_id):
                            self._redraw_all_rois()
                            if self.on_roi_created:
                                self.on_roi_created(None)  # Notificar cambio
                            return
                    except ValueError:
                        pass
    
    def _draw_roi(self, roi: ROI):
        """Dibuja un ROI en el canvas (con soporte para rotación)."""
        # Color según selección
        is_selected = roi.id in self._selected_rois
        base_color = self.ROI_COLORS[(roi.id - 1) % len(self.ROI_COLORS)]
        outline_color = self.SELECTION_COLOR if is_selected else base_color
        outline_width = 4 if is_selected else 2
        
        tag = f"roi_{roi.id}"
        
        if roi.angle != 0:
            # Dibujar como polígono rotado
            corners = roi.rotated_corners()
            canvas_corners = []
            for (ix, iy) in corners:
                cx = ix * self.scale_factor + self.image_offset_x
                cy = iy * self.scale_factor + self.image_offset_y
                canvas_corners.extend([cx, cy])
            
            self.canvas.create_polygon(
                *canvas_corners,
                outline=outline_color, width=outline_width,
                fill='', tags=(tag, "roi")
            )
            # Centro para etiqueta
            center_cx = sum(canvas_corners[::2]) / 4
            center_cy = sum(canvas_corners[1::2]) / 4
            label_x = center_cx - 25
            label_y = center_cy - 12
        else:
            # Rectángulo normal (sin rotación)
            cx1 = roi.x1 * self.scale_factor + self.image_offset_x
            cy1 = roi.y1 * self.scale_factor + self.image_offset_y
            cx2 = roi.x2 * self.scale_factor + self.image_offset_x
            cy2 = roi.y2 * self.scale_factor + self.image_offset_y
            
            self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline=outline_color, width=outline_width, tags=(tag, "roi")
            )
            label_x = cx1 + 3
            label_y = cy1 + 3
        
        # Etiqueta con número y nombre
        label_text = roi.display_label()
        label_color = self.SELECTION_COLOR if is_selected else base_color
        
        # Ajustar ancho del fondo según texto
        text_width = max(45, len(label_text) * 7 + 10)
        
        self.canvas.create_rectangle(
            label_x, label_y, label_x + text_width, label_y + 18,
            fill=label_color, outline="", tags=(tag, "roi")
        )
        
        # Texto
        self.canvas.create_text(
            label_x + text_width // 2, label_y + 9,
            text=label_text, fill="#000000",
            font=("Arial", 9, "bold"), tags=(tag, "roi")
        )
        
        # Indicador de ángulo si está rotado
        if roi.angle != 0:
            angle_text = f"{roi.angle:.0f}°"
            self.canvas.create_text(
                label_x + text_width // 2, label_y + 28,
                text=angle_text, fill="#FFD700",
                font=("Arial", 8), tags=(tag, "roi")
            )
    
    def _redraw_all_rois(self):
        """Redibuja todos los ROIs."""
        self.canvas.delete("roi")
        for roi in self.roi_manager.get_all_rois():
            self._draw_roi(roi)
    
    # ========== MÉTODOS DE SELECCIÓN ==========
    
    def set_selection_mode(self, enabled: bool):
        """Activa/desactiva modo selección."""
        self._selection_mode = enabled
        if enabled:
            self.canvas.configure(cursor="hand2")
        else:
            self.canvas.configure(cursor="crosshair")
    
    def is_selection_mode(self) -> bool:
        """Retorna si está en modo selección."""
        return self._selection_mode
    
    def _handle_selection_click(self, event):
        """Maneja click en modo selección."""
        # Convertir coordenadas de pantalla a coordenadas de canvas (considerando scroll)
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        roi_id = self._find_roi_at(cx, cy)
        if roi_id is not None:
            self._toggle_selection(roi_id)
    
    def _on_ctrl_click(self, event):
        """Ctrl+Click: toggle selección de ROI (funciona siempre)."""
        if self.photo_image is None:
            return
        
        # Convertir coordenadas de pantalla a coordenadas de canvas (considerando scroll)
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        roi_id = self._find_roi_at(cx, cy)
        if roi_id is not None:
            self._toggle_selection(roi_id)
        return "break"  # Evitar propagación al _on_press
    
    def _toggle_selection(self, roi_id: int):
        """Alterna selección de un ROI."""
        if roi_id in self._selected_rois:
            self._selected_rois.discard(roi_id)
        else:
            self._selected_rois.add(roi_id)
        
        self._redraw_all_rois()
        
        if self.on_selection_changed:
            self.on_selection_changed(list(self._selected_rois))
    
    def _find_roi_at(self, canvas_x: int, canvas_y: int) -> Optional[int]:
        """Encuentra el ROI bajo las coordenadas del canvas."""
        items = self.canvas.find_overlapping(
            canvas_x - 2, canvas_y - 2, canvas_x + 2, canvas_y + 2
        )
        
        for item in items:
            tags = self.canvas.gettags(item)
            for tag in tags:
                if tag.startswith("roi_"):
                    try:
                        return int(tag.split("_")[1])
                    except ValueError:
                        pass
        return None
    
    def get_selected_rois(self) -> List[int]:
        """Retorna lista de IDs de ROIs seleccionados."""
        return list(self._selected_rois)
    
    def clear_selection(self):
        """Limpia la selección."""
        self._selected_rois.clear()
        self._redraw_all_rois()
        if self.on_selection_changed:
            self.on_selection_changed([])
    
    def select_all_rois(self):
        """Selecciona todos los ROIs."""
        self._selected_rois = set(roi.id for roi in self.roi_manager.get_all_rois())
        self._redraw_all_rois()
        if self.on_selection_changed:
            self.on_selection_changed(list(self._selected_rois))
    
    def _on_double_click(self, event):
        """Doble-click: renumerar ROI bajo el cursor."""
        if self.photo_image is None:
            return "break"
        
        # No actuar en modo paste
        if self._paste_mode:
            return "break"
        
        # Convertir coordenadas de pantalla a coordenadas de canvas (considerando scroll)
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        roi_id = self._find_roi_at(cx, cy)
        if roi_id is not None:
            # Cancelar cualquier dibujo temporal
            self._drawing = False
            if self._temp_rect:
                self.canvas.delete(self._temp_rect)
                self._temp_rect = None
            
            if self.on_roi_double_click:
                self.on_roi_double_click(roi_id)
        
        return "break"  # Evitar propagación
    
    # ========== MÉTODOS DE PASTE VISUAL ==========
    
    def start_paste_mode(self, on_paste_complete: callable = None, on_status_update: callable = None):
        """
        Inicia modo paste visual continuo: los ROIs del clipboard aparecen
        como fantasmas y siguen el cursor. Cada click crea ROIs y continúa
        hasta que se cancele explícitamente.
        
        Args:
            on_paste_complete: Callback cuando se confirma paste (recibe lista de nuevos ROIs)
            on_status_update: Callback para actualizar status (recibe mensaje string)
        """
        if not self.roi_manager.has_clipboard():
            return False
        
        if self.photo_image is None:
            return False
        
        self._paste_mode = True
        self._paste_callback = on_paste_complete
        self._on_paste_status = on_status_update
        self._paste_count = 0  # Reiniciar contador
        
        # Obtener ROIs del clipboard
        clipboard = self.roi_manager._clipboard
        if not clipboard:
            self._paste_mode = False
            return False
        
        # Calcular centroide del grupo (ancla)
        sum_cx = sum((r[0] + r[2]) / 2 for r in clipboard)
        sum_cy = sum((r[1] + r[3]) / 2 for r in clipboard)
        anchor_img_x = sum_cx / len(clipboard)
        anchor_img_y = sum_cy / len(clipboard)
        
        # Convertir ancla a coordenadas canvas
        self._paste_anchor = (
            anchor_img_x * self.scale_factor + self.image_offset_x,
            anchor_img_y * self.scale_factor + self.image_offset_y
        )
        
        # Calcular offsets de cada ROI respecto al ancla (en coordenadas de imagen)
        self._paste_offsets = []
        for (x1, y1, x2, y2) in clipboard:
            off_x1 = x1 - anchor_img_x
            off_y1 = y1 - anchor_img_y
            off_x2 = x2 - anchor_img_x
            off_y2 = y2 - anchor_img_y
            self._paste_offsets.append((off_x1, off_y1, off_x2, off_y2))
        
        # Cambiar cursor
        self.canvas.configure(cursor="fleur")
        
        # Bind de movimiento y Escape
        self._motion_binding = self.canvas.bind("<Motion>", self._on_paste_motion)
        self.canvas.bind("<Escape>", self._on_escape_key)
        
        # Crear fantasmas en posición inicial (centro del canvas visible)
        canvas_center_x = self.canvas.winfo_width() / 2
        canvas_center_y = self.canvas.winfo_height() / 2
        self._last_mouse_pos = (canvas_center_x, canvas_center_y)
        self._create_ghosts(canvas_center_x, canvas_center_y)
        
        return True
    
    def _create_ghosts(self, mouse_x: float, mouse_y: float):
        """Crea rectángulos fantasma en la posición del cursor."""
        # Eliminar fantasmas anteriores
        self._delete_ghosts()
        
        for (off_x1, off_y1, off_x2, off_y2) in self._paste_offsets:
            # Calcular posición en canvas
            cx1 = mouse_x + (off_x1 * self.scale_factor)
            cy1 = mouse_y + (off_y1 * self.scale_factor)
            cx2 = mouse_x + (off_x2 * self.scale_factor)
            cy2 = mouse_y + (off_y2 * self.scale_factor)
            
            # Crear rectángulo fantasma (dashed, semi-transparente)
            ghost_id = self.canvas.create_rectangle(
                cx1, cy1, cx2, cy2,
                outline="#00FF00",
                width=2,
                dash=(6, 4),
                fill="",
                tags="ghost"
            )
            self._ghost_items.append(ghost_id)
    
    def _delete_ghosts(self):
        """Elimina todos los rectángulos fantasma."""
        self.canvas.delete("ghost")
        self._ghost_items.clear()
    
    def _on_paste_motion(self, event):
        """Mueve los fantasmas siguiendo el cursor."""
        if not self._paste_mode:
            return
        
        # Obtener coordenadas del canvas (considerando scroll)
        mouse_x = self.canvas.canvasx(event.x)
        mouse_y = self.canvas.canvasy(event.y)
        
        # Calcular delta desde última posición
        delta_x = mouse_x - self._last_mouse_pos[0]
        delta_y = mouse_y - self._last_mouse_pos[1]
        
        # Mover todos los fantasmas
        for ghost_id in self._ghost_items:
            self.canvas.move(ghost_id, delta_x, delta_y)
        
        self._last_mouse_pos = (mouse_x, mouse_y)
    
    def _on_escape_key(self, event):
        """Cancela paste con Escape."""
        if self._paste_mode:
            self._cancel_paste()
    
    def _confirm_paste(self):
        """Confirma paste: crea ROIs reales en la posición actual y continúa en modo paste."""
        if not self._paste_mode:
            return
        
        # Calcular offset en coordenadas de imagen
        mouse_x, mouse_y = self._last_mouse_pos
        
        # Convertir posición del mouse a coordenadas de imagen
        img_mouse_x = (mouse_x - self.image_offset_x) / self.scale_factor
        img_mouse_y = (mouse_y - self.image_offset_y) / self.scale_factor
        
        # El ancla original estaba en coordenadas de imagen
        anchor_img_x = (self._paste_anchor[0] - self.image_offset_x) / self.scale_factor
        anchor_img_y = (self._paste_anchor[1] - self.image_offset_y) / self.scale_factor
        
        # Calcular offset total
        offset_x = int(img_mouse_x - anchor_img_x)
        offset_y = int(img_mouse_y - anchor_img_y)
        
        # Pegar ROIs con el offset calculado
        new_rois = self.roi_manager.paste_rois(offset_x, offset_y)
        
        # Incrementar contador
        self._paste_count += 1
        
        # Redibujar ROIs (sin salir del modo paste)
        self._redraw_all_rois()
        
        # Callback de paste
        if self._paste_callback:
            self._paste_callback(new_rois)
        
        if self.on_roi_created:
            self.on_roi_created(None)
        
        # Actualizar status con contador
        if self._on_paste_status:
            clipboard_count = len(self.roi_manager._clipboard)
            total_pasted = self._paste_count * clipboard_count
            self._on_paste_status(f"Pegado #{self._paste_count} ({total_pasted} ROIs) - Click para continuar, Escape/Click derecho para terminar")
        
        # Recrear fantasmas para el siguiente paste
        self._create_ghosts(mouse_x, mouse_y)
    
    def _cancel_paste(self):
        """Cancela paste sin crear ROIs."""
        self._end_paste_mode()
    
    def _end_paste_mode(self):
        """Limpia estado de modo paste."""
        paste_count = self._paste_count  # Guardar antes de limpiar
        
        self._paste_mode = False
        self._delete_ghosts()
        self._paste_offsets.clear()
        self._paste_callback = None
        self._paste_count = 0
        
        # Notificar fin del modo paste
        if self._on_paste_status:
            if paste_count > 0:
                self._on_paste_status(f"Pegado completado: {paste_count} paste(s) realizados")
            else:
                self._on_paste_status("Pegado cancelado")
        self._on_paste_status = None
        
        # Restaurar cursor
        if self._selection_mode:
            self.canvas.configure(cursor="hand2")
        else:
            self.canvas.configure(cursor="crosshair")
        
        # Desvincular Motion (opcional, no afecta si no está en paste mode)
        try:
            self.canvas.unbind("<Escape>")
        except:
            pass
    
    # ========== MÉTODOS DE MEDICIÓN ==========
    
    def set_geo_transform(self, transform, crs=None):
        """
        Configura el transform geográfico para calcular distancias.
        
        Args:
            transform: Affine transform del rasterio (profile['transform'])
            crs: Sistema de coordenadas (CRS object de rasterio)
        """
        self._geo_transform = transform
        self._geo_pixel_size = None
        self._crs_unit = None
        
        if transform is not None and crs is not None:
            # Extraer tamaño de píxel del transform
            # transform.a = ancho de píxel (puede ser negativo)
            # transform.e = alto de píxel (normalmente negativo)
            px_width = abs(transform.a) if hasattr(transform, 'a') else None
            px_height = abs(transform.e) if hasattr(transform, 'e') else None
            
            if px_width and px_height:
                self._geo_pixel_size = (px_width, px_height)
                
                # Extraer unidad del CRS
                try:
                    if hasattr(crs, 'linear_units'):
                        self._crs_unit = crs.linear_units
                    elif hasattr(crs, 'to_dict'):
                        crs_dict = crs.to_dict()
                        # Intentar obtener unidad de varios lugares
                        if 'units' in crs_dict:
                            self._crs_unit = crs_dict['units']
                    
                    # Si es proyectado, asumir metros si no se pudo detectar
                    if self._crs_unit is None and hasattr(crs, 'is_projected') and crs.is_projected:
                        self._crs_unit = 'm'
                except:
                    pass
    
    def set_ruler_mode(self, enabled: bool):
        """Activa/desactiva modo regla (medir sin crear ROI)."""
        self._ruler_mode = enabled
        if enabled:
            self.canvas.configure(cursor="tcross")
        elif self._selection_mode:
            self.canvas.configure(cursor="hand2")
        else:
            self.canvas.configure(cursor="crosshair")
    
    def is_ruler_mode(self) -> bool:
        """Retorna si está en modo regla."""
        return self._ruler_mode
    
    def _draw_measurement_overlay(self, canvas_x1: float, canvas_y1: float, 
                                   canvas_x2: float, canvas_y2: float):
        """
        Dibuja overlay flotante con mediciones en tiempo real.
        Muestra ΔX, ΔY, Distancia en píxeles (y en unidades del CRS si hay georef).
        """
        # Limpiar overlay anterior
        self._clear_measurement_overlay()
        
        # Calcular distancias en coordenadas de IMAGEN (píxeles reales)
        img_x1 = int((canvas_x1 - self.image_offset_x) / self.scale_factor)
        img_y1 = int((canvas_y1 - self.image_offset_y) / self.scale_factor)
        img_x2 = int((canvas_x2 - self.image_offset_x) / self.scale_factor)
        img_y2 = int((canvas_y2 - self.image_offset_y) / self.scale_factor)
        
        dx_px = abs(img_x2 - img_x1)
        dy_px = abs(img_y2 - img_y1)
        dist_px = math.sqrt(dx_px**2 + dy_px**2)
        
        # Construir texto de medición
        lines = [
            f"ΔX: {dx_px} px",
            f"ΔY: {dy_px} px",
            f"Dist: {dist_px:.1f} px"
        ]
        
        # Si hay georeferencia con CRS válido, agregar medición en unidades del CRS
        if self._geo_pixel_size and self._crs_unit:
            px_w, px_h = self._geo_pixel_size
            dx_geo = dx_px * px_w
            dy_geo = dy_px * px_h
            dist_geo = math.sqrt(dx_geo**2 + dy_geo**2)
            
            unit = self._crs_unit
            
            # Formatear según magnitud (para metros)
            if unit == 'm':
                if dist_geo >= 1000:
                    lines.append(f"≈ {dist_geo/1000:.2f} km")
                elif dist_geo >= 1:
                    lines.append(f"≈ {dist_geo:.2f} m")
                else:
                    lines.append(f"≈ {dist_geo*100:.1f} cm")
            elif unit == 'ft':
                # Pies
                if dist_geo >= 5280:  # Millas
                    lines.append(f"≈ {dist_geo/5280:.2f} mi")
                else:
                    lines.append(f"≈ {dist_geo:.1f} ft")
            else:
                # Otras unidades, mostrar directo
                lines.append(f"≈ {dist_geo:.4f} {unit}")
        
        text = "\n".join(lines)
        
        # Posición del tooltip (cerca del cursor pero no encima)
        # Ajustar para que no salga del canvas
        label_x = canvas_x2 + 15
        label_y = canvas_y2 - 10
        
        # Calcular dimensiones del texto aproximadas
        num_lines = len(lines)
        box_width = 95
        box_height = 14 * num_lines + 8
        
        # Fondo del label (semi-transparente simulado con color oscuro)
        bg_rect = self.canvas.create_rectangle(
            label_x - 4, label_y - 4,
            label_x + box_width, label_y + box_height,
            fill="#1a1a1a", outline="#00FF00", width=1,
            tags="measurement_overlay"
        )
        
        # Texto de medición
        text_item = self.canvas.create_text(
            label_x, label_y,
            text=text, anchor="nw",
            fill="#00FF00", font=("Consolas", 9, "bold"),
            tags="measurement_overlay"
        )
        
        # Línea de medición diagonal (punteada)
        line_item = self.canvas.create_line(
            canvas_x1, canvas_y1, canvas_x2, canvas_y2,
            fill="#00FF00", width=1, dash=(5, 3),
            tags="measurement_overlay"
        )
    
    def _clear_measurement_overlay(self):
        """Elimina todos los elementos del overlay de medición."""
        self.canvas.delete("measurement_overlay")
    
    def is_paste_mode(self) -> bool:
        """Retorna si está en modo paste."""
        return self._paste_mode
    
    def get_paste_count(self) -> int:
        """Retorna número de pastes realizados en sesión actual."""
        return self._paste_count
    
    def stop_paste_mode(self):
        """Detiene el modo paste de forma explícita."""
        if self._paste_mode:
            self._end_paste_mode()


# ==================== STATS CALCULATOR ====================

class StatsCalculator:
    """
    Extrae píxeles por ROI desde numpy array y calcula
    promedios por banda usando np.nanmean.
    """
    
    def __init__(self, data: np.ndarray, nodata_value: Optional[float] = None):
        """
        Args:
            data: Array numpy shape (bands, H, W)
            nodata_value: Valor a tratar como NaN
        """
        self.data = data.astype(np.float32)
        self.nodata_value = nodata_value
        
        # Reemplazar nodata con NaN
        if nodata_value is not None:
            self.data = np.where(self.data == nodata_value, np.nan, self.data)
    
    def calculate_roi_stats(self, rois: List[ROI]) -> Dict[int, Dict[int, float]]:
        """
        Calcula promedio por banda para cada ROI.
        Soporta ROIs rotados usando máscara poligonal.
        
        Returns:
            Dict[roi_id, Dict[band_idx, mean_value]]
        """
        results = {}
        num_bands = self.data.shape[0]
        
        for roi in rois:
            roi_stats = {}
            
            if roi.angle != 0 and ImageDraw is not None:
                # ROI rotado: usar máscara poligonal
                corners = roi.rotated_corners()
                
                # Calcular bounding box del polígono rotado
                min_x = max(0, int(min(c[0] for c in corners)))
                max_x = min(self.data.shape[2], int(max(c[0] for c in corners)) + 1)
                min_y = max(0, int(min(c[1] for c in corners)))
                max_y = min(self.data.shape[1], int(max(c[1] for c in corners)) + 1)
                
                if max_x <= min_x or max_y <= min_y:
                    for band_idx in range(num_bands):
                        roi_stats[band_idx] = 0.0
                    results[roi.id] = roi_stats
                    continue
                
                # Crear máscara con PIL
                mask_w = max_x - min_x
                mask_h = max_y - min_y
                mask_img = Image.new('L', (mask_w, mask_h), 0)
                draw = ImageDraw.Draw(mask_img)
                # Esquinas relativas al bounding box
                poly = [(c[0] - min_x, c[1] - min_y) for c in corners]
                draw.polygon(poly, fill=255)
                mask = np.array(mask_img) > 0
                
                for band_idx in range(num_bands):
                    roi_data = self.data[band_idx, min_y:max_y, min_x:max_x]
                    masked_data = np.where(mask, roi_data, np.nan)
                    mean_val = np.nanmean(masked_data)
                    if np.isnan(mean_val):
                        mean_val = 0.0
                    roi_stats[band_idx] = float(mean_val)
            else:
                # ROI sin rotación: slicing directo (rápido)
                for band_idx in range(num_bands):
                    roi_data = self.data[band_idx, roi.y1:roi.y2, roi.x1:roi.x2]
                    mean_val = np.nanmean(roi_data)
                    if np.isnan(mean_val):
                        mean_val = 0.0
                    roi_stats[band_idx] = float(mean_val)
            
            results[roi.id] = roi_stats
        
        return results


# ==================== CSV EXPORTER ====================

class CSVExporter:
    """
    Genera archivo CSV con filas=bandas y columnas=ROIs.
    Formato:
        ,ROI_1,ROI_2,ROI_3,...
        Band_1,val,val,val
        Band_2,val,val,val
        ...
    """
    
    @staticmethod
    def export(filepath: str, stats: Dict[int, Dict[int, float]], 
               num_bands: int, decimal_places: int = 4,
               roi_names: Optional[Dict[int, str]] = None) -> str:
        """
        Exporta estadísticas a CSV.
        
        Args:
            filepath: Ruta de salida
            stats: Dict[roi_id, Dict[band_idx, mean_value]]
            num_bands: Número total de bandas
            decimal_places: Decimales en valores
            roi_names: Dict opcional {roi_id: nombre} para incluir nombres
        
        Returns:
            Ruta del archivo creado
        """
        if not stats:
            raise ValueError("No hay ROIs para exportar")
        
        # Ordenar ROIs por ID
        roi_ids = sorted(stats.keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header con nombres si están disponibles
            header = ['']
            for roi_id in roi_ids:
                name = roi_names.get(roi_id, '') if roi_names else ''
                if name:
                    header.append(f'ROI_{roi_id} ({name})')
                else:
                    header.append(f'ROI_{roi_id}')
            writer.writerow(header)
            
            # Filas por banda
            for band_idx in range(num_bands):
                row = [f'Band_{band_idx + 1}']
                
                for roi_id in roi_ids:
                    val = stats[roi_id].get(band_idx, 0.0)
                    row.append(f'{val:.{decimal_places}f}')
                
                writer.writerow(row)
        
        return filepath


# ==================== PARAMETROS TAB ====================

class ParametrosTab:
    """
    Clase principal de la pestaña Parámetros.
    Orquesta ImageLoader, ROIManager, InteractiveCanvas,
    StatsCalculator y CSVExporter.
    """
    
    MAX_PREVIEW_SIZE = 800  # Tamaño máximo del preview
    
    def __init__(self, parent: ttk.Frame):
        """
        Args:
            parent: Frame padre (ttk.Frame del notebook)
        """
        self.parent = parent
        
        # Componentes
        self.image_loader = ImageLoader()
        self.roi_manager = ROIManager()
        self.stats_calculator: Optional[StatsCalculator] = None
        
        # Estado
        self.current_filepath: Optional[str] = None
        self._zoom_level = 800  # Nivel de zoom actual
        
        # Crear UI
        self._create_ui()
    
    def _create_ui(self):
        """Construye la interfaz de la pestaña."""
        # Frame principal con padding
        main_frame = ttk.Frame(self.parent, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ========== BARRA SUPERIOR ==========
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            toolbar, text="[Abrir] Cargar TIFF", 
            command=self._load_tiff, width=18
        ).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(
            toolbar, text="[X] Limpiar ROIs",
            command=self._clear_rois, width=18
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            toolbar, text="[Guardar] Exportar CSV",
            command=self._export_csv, width=18
        ).pack(side=tk.LEFT, padx=5)
        
        # Separador y controles de zoom
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        ttk.Label(toolbar, text="Zoom:", font=("Arial", 9)).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="-", width=3, command=self._zoom_out).pack(side=tk.LEFT, padx=2)
        self.zoom_var = tk.StringVar(value="100%")
        self.zoom_entry = ttk.Entry(toolbar, textvariable=self.zoom_var,
                                    width=6, font=("Arial", 9), justify="center")
        self.zoom_entry.pack(side=tk.LEFT)
        self.zoom_entry.bind("<Return>", lambda e: self._on_zoom_entry())
        self.zoom_entry.bind("<FocusOut>", lambda e: self._on_zoom_entry())
        ttk.Button(toolbar, text="+", width=3, command=self._zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Fit", width=4, command=self._zoom_fit).pack(side=tk.LEFT, padx=2)
        
        # Separador
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=10, fill=tk.Y)
        
        # Info del archivo
        self.file_label = ttk.Label(
            toolbar, text="Sin archivo cargado",
            font=("Arial", 9), foreground="gray"
        )
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        # ========== BARRA DE HERRAMIENTAS ROI ==========
        roi_toolbar = ttk.Frame(main_frame)
        roi_toolbar.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(roi_toolbar, text="ROIs:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        
        # Botón Selección (toggle)
        self._selection_mode_var = tk.BooleanVar(value=False)
        self.btn_selection = ttk.Checkbutton(
            roi_toolbar, text="Selección",
            variable=self._selection_mode_var,
            command=self._toggle_selection_mode,
            width=10
        )
        self.btn_selection.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            roi_toolbar, text="Sel. Todos", width=10,
            command=self._select_all_rois
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            roi_toolbar, text="Deseleccionar", width=12,
            command=self._deselect_all_rois
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(roi_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)
        
        ttk.Button(
            roi_toolbar, text="Copiar", width=8,
            command=self._copy_selected_rois
        ).pack(side=tk.LEFT, padx=2)
        
        # Botón Pegar dinámico (cambia texto según modo)
        self._paste_btn_text = tk.StringVar(value="Pegar")
        self.btn_paste = ttk.Button(
            roi_toolbar, textvariable=self._paste_btn_text, width=12,
            command=self._toggle_paste_mode
        )
        self.btn_paste.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(roi_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)
        
        ttk.Button(
            roi_toolbar, text="Matriz Lineal", width=12,
            command=self._show_matrix_dialog
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(roi_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)
        
        ttk.Button(
            roi_toolbar, text="Renumerar", width=10,
            command=self._renumber_rois
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(roi_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)
        
        # Botón Invertir Orden (solo seleccionados)
        ttk.Button(
            roi_toolbar, text="↔ Invertir Orden", width=14,
            command=self._invert_selected_order
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(roi_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)
        
        # Botón Medir (toggle) - herramienta de medición
        self._ruler_mode_var = tk.BooleanVar(value=False)
        self.btn_ruler = ttk.Checkbutton(
            roi_toolbar, text="📏 Medir",
            variable=self._ruler_mode_var,
            command=self._toggle_ruler_mode,
            width=10
        )
        self.btn_ruler.pack(side=tk.LEFT, padx=2)
        
        # Label de selección
        self.selection_label = ttk.Label(
            roi_toolbar, text="(0 seleccionados)",
            font=("Arial", 8), foreground="#666666"
        )
        self.selection_label.pack(side=tk.LEFT, padx=10)
        
        # ========== ÁREA CENTRAL (PanedWindow: Canvas + Panel ROI) ==========
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # --- Lado izquierdo: Canvas ---
        canvas_frame = ttk.LabelFrame(paned, text="Vista Previa - Dibuja rectángulos para definir ROIs", padding=5)
        paned.add(canvas_frame, weight=3)
        
        # Canvas con scrollbars
        self.canvas_container = ttk.Frame(canvas_frame)
        self.canvas_container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        h_scroll = ttk.Scrollbar(self.canvas_container, orient=tk.HORIZONTAL)
        v_scroll = ttk.Scrollbar(self.canvas_container, orient=tk.VERTICAL)
        
        # Interactive canvas con soporte scroll
        self.interactive_canvas = InteractiveCanvas(
            self.canvas_container,
            self.roi_manager,
            on_roi_created=self._on_roi_changed,
            on_selection_changed=self._on_selection_changed,
            on_roi_double_click=self._on_roi_double_click,
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set
        )
        
        # Conectar scrollbars
        h_scroll.config(command=self.interactive_canvas.canvas.xview)
        v_scroll.config(command=self.interactive_canvas.canvas.yview)
        
        # Grid layout para scrollbars
        self.interactive_canvas.canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        
        self.canvas_container.grid_rowconfigure(0, weight=1)
        self.canvas_container.grid_columnconfigure(0, weight=1)
        
        # Binding zoom con rueda del raton (Ctrl+Scroll)
        self.interactive_canvas.canvas.bind("<Control-MouseWheel>", self._on_mouse_wheel_zoom)
        
        # --- Lado derecho: Panel de ROIs ---
        roi_panel = ttk.LabelFrame(paned, text="Lista de ROIs", padding=5)
        paned.add(roi_panel, weight=1)
        
        # Treeview con columnas
        tree_frame = ttk.Frame(roi_panel)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        
        self.roi_tree = ttk.Treeview(
            tree_frame,
            columns=("id", "name", "x1", "y1", "x2", "y2", "size", "angle"),
            show="headings",
            selectmode="extended",
            yscrollcommand=tree_scroll.set,
            height=15
        )
        tree_scroll.config(command=self.roi_tree.yview)
        
        # Configurar columnas
        self.roi_tree.heading("id", text="#", anchor=tk.CENTER)
        self.roi_tree.heading("name", text="Nombre", anchor=tk.CENTER)
        self.roi_tree.heading("x1", text="X1", anchor=tk.CENTER)
        self.roi_tree.heading("y1", text="Y1", anchor=tk.CENTER)
        self.roi_tree.heading("x2", text="X2", anchor=tk.CENTER)
        self.roi_tree.heading("y2", text="Y2", anchor=tk.CENTER)
        self.roi_tree.heading("size", text="Tamaño", anchor=tk.CENTER)
        self.roi_tree.heading("angle", text="Áng.", anchor=tk.CENTER)
        
        self.roi_tree.column("id", width=35, minwidth=30, anchor=tk.CENTER)
        self.roi_tree.column("name", width=70, minwidth=50, anchor=tk.CENTER)
        self.roi_tree.column("x1", width=45, minwidth=35, anchor=tk.CENTER)
        self.roi_tree.column("y1", width=45, minwidth=35, anchor=tk.CENTER)
        self.roi_tree.column("x2", width=45, minwidth=35, anchor=tk.CENTER)
        self.roi_tree.column("y2", width=45, minwidth=35, anchor=tk.CENTER)
        self.roi_tree.column("size", width=60, minwidth=45, anchor=tk.CENTER)
        self.roi_tree.column("angle", width=40, minwidth=30, anchor=tk.CENTER)
        
        self.roi_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bindings del Treeview
        self.roi_tree.bind("<Double-1>", self._on_tree_double_click)
        self.roi_tree.bind("<Delete>", self._on_tree_delete)
        
        # Sincronizar selección TreeView → Canvas
        self.roi_tree.bind("<<TreeviewSelect>>", self._on_tree_selection_changed)
        
        # Botones del panel de ROIs
        tree_btn_frame = ttk.Frame(roi_panel)
        tree_btn_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(
            tree_btn_frame, text="▲ Subir", width=8,
            command=self._tree_move_up
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            tree_btn_frame, text="▼ Bajar", width=8,
            command=self._tree_move_down
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            tree_btn_frame, text="🗑 Eliminar", width=10,
            command=self._tree_delete_selected
        ).pack(side=tk.RIGHT, padx=2)
        
        # Placeholder inicial
        self._show_placeholder()
        
        # ========== BARRA INFERIOR ==========
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(
            status_frame,
            text="Carga un archivo GeoTIFF para comenzar",
            font=("Arial", 10),
            foreground="#666666"
        )
        self.status_label.pack(side=tk.LEFT)
        
        # Instrucciones
        ttk.Label(
            status_frame,
            text="Click izq: dibujar ROI | Doble-click: renumerar | Click der: eliminar | Ctrl+Click: seleccionar",
            font=("Arial", 8),
            foreground="#999999"
        ).pack(side=tk.RIGHT)
    
    def _show_placeholder(self):
        """Muestra placeholder cuando no hay imagen."""
        self.interactive_canvas.canvas.configure(width=600, height=400)
        self.interactive_canvas.canvas.create_text(
            300, 200,
            text="[Imagen] Carga un archivo GeoTIFF\npara visualizar y definir ROIs",
            fill="#888888",
            font=("Arial", 14),
            justify=tk.CENTER
        )
    
    def _load_tiff(self):
        """Abre diálogo para cargar GeoTIFF."""
        filepath = filedialog.askopenfilename(
            title="Seleccionar GeoTIFF",
            filetypes=[
                ("GeoTIFF files", "*.tif *.TIF *.tiff *.TIFF"),
                ("All files", "*.*")
            ]
        )
        
        if not filepath:
            return
        
        try:
            # Cargar imagen
            self.image_loader.load(filepath)
            self.current_filepath = filepath
            
            # Configurar ROI manager con dimensiones
            bands, height, width = self.image_loader.get_dimensions()
            self.roi_manager.set_image_bounds(width, height)
            self.roi_manager.clear()
            
            # Generar preview con zoom actual
            preview = self.image_loader.generate_preview(self._zoom_level)
            
            # Mostrar en canvas
            self.interactive_canvas.set_image(preview, width, height)
            self._update_zoom_label()
            
            # Actualizar labels
            filename = os.path.basename(filepath)
            self.file_label.config(
                text=f"{filename} ({width}x{height}, {bands} bandas)",
                foreground="#1f4788"
            )
            self._update_status()
            
            # Crear calculador de stats
            self.stats_calculator = StatsCalculator(
                self.image_loader.data,
                self.image_loader.nodata_value
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo:\n{str(e)}")
    
    def _clear_rois(self):
        """Limpia todos los ROIs."""
        self.roi_manager.clear()
        
        # Redibujar canvas sin ROIs
        if self.image_loader.data is not None:
            bands, height, width = self.image_loader.get_dimensions()
            preview = self.image_loader.generate_preview(self._zoom_level)
            self.interactive_canvas.set_image(preview, width, height)
        
        self._refresh_roi_tree()
        self._update_status()
    
    def _export_csv(self):
        """Exporta estadísticas de ROIs a CSV."""
        if self.image_loader.data is None:
            messagebox.showwarning("Aviso", "Primero carga un archivo GeoTIFF")
            return
        
        rois = self.roi_manager.get_all_rois()
        if not rois:
            messagebox.showwarning("Aviso", "No hay ROIs definidos.\nDibuja al menos un rectángulo sobre la imagen.")
            return
        
        # Diálogo de guardado
        default_name = "roi_statistics.csv"
        if self.current_filepath:
            base = os.path.splitext(os.path.basename(self.current_filepath))[0]
            default_name = f"{base}_roi_stats.csv"
        
        filepath = filedialog.asksaveasfilename(
            title="Guardar CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        try:
            # Calcular estadísticas
            stats = self.stats_calculator.calculate_roi_stats(rois)
            num_bands = self.image_loader.get_dimensions()[0]
            
            # Recopilar nombres de ROIs
            roi_names = {roi.id: roi.name for roi in rois if roi.name}
            
            # Exportar
            CSVExporter.export(filepath, stats, num_bands, roi_names=roi_names)
            
            messagebox.showinfo(
                "Éxito", 
                f"CSV exportado correctamente:\n{filepath}\n\n"
                f"{num_bands} bandas × {len(rois)} ROIs"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al exportar:\n{str(e)}")
    
    def _on_roi_changed(self, roi: Optional[ROI]):
        """Callback cuando se crea o elimina un ROI."""
        self._refresh_roi_tree()
        self._update_status()
    
    def _update_status(self):
        """Actualiza label de estado."""
        if self.image_loader.data is None:
            self.status_label.config(text="Carga un archivo GeoTIFF para comenzar")
            return
        
        bands, _, _ = self.image_loader.get_dimensions()
        roi_count = self.roi_manager.get_roi_count()
        
        self.status_label.config(
            text=f"📊 {bands} bandas | 📐 {roi_count} ROIs definidos",
            foreground="#1f4788" if roi_count > 0 else "#666666"
        )
    
    # ========== MÉTODOS DE ZOOM ==========
    
    def _zoom_in(self):
        """Aumentar zoom"""
        self._zoom_level = min(3200, int(self._zoom_level * 1.5))
        self._update_preview()
    
    def _zoom_out(self):
        """Reducir zoom"""
        self._zoom_level = max(200, int(self._zoom_level / 1.5))
        self._update_preview()
    
    def _zoom_fit(self):
        """Ajustar zoom para ver toda la imagen"""
        self._zoom_level = 800
        self._update_preview()
    
    def _on_mouse_wheel_zoom(self, event):
        """Zoom con Ctrl+rueda del raton (con debounce para rendimiento)"""
        if not hasattr(self, '_zoom_wheel_delta'):
            self._zoom_wheel_delta = 0
            self._zoom_wheel_debounce_id = None
        
        if event.delta > 0:
            self._zoom_wheel_delta += 1
        else:
            self._zoom_wheel_delta -= 1
        
        # Cancelar debounce anterior
        if self._zoom_wheel_debounce_id is not None:
            self.parent.after_cancel(self._zoom_wheel_debounce_id)
        
        # Aplicar después de 80ms de inactividad
        self._zoom_wheel_debounce_id = self.parent.after(80, self._apply_wheel_zoom)
    
    def _apply_wheel_zoom(self):
        """Aplica zoom acumulado por debounce de rueda del mouse."""
        self._zoom_wheel_debounce_id = None
        delta = self._zoom_wheel_delta
        self._zoom_wheel_delta = 0
        
        if delta == 0:
            return
        
        # Aplicar zoom acumulado (cada step = 1.5x)
        factor = 1.5 ** abs(delta)
        if delta > 0:
            self._zoom_level = min(3200, int(self._zoom_level * factor))
        else:
            self._zoom_level = max(200, int(self._zoom_level / factor))
        
        self._update_preview()
    
    def _on_zoom_entry(self):
        """Aplica el zoom escrito manualmente por el usuario"""
        try:
            text = self.zoom_var.get().replace("%", "").strip()
            percent = float(text)
            percent = max(1.0, min(400.0, percent))
            if self.image_loader.data is None:
                return
            _, h, w = self.image_loader.get_dimensions()
            max_dim = max(w, h)
            if max_dim > 0:
                self._zoom_level = max(50, min(3200, int(percent * max_dim / 100)))
                self._update_preview()
        except (ValueError, tk.TclError):
            self._update_zoom_label()

    def _update_zoom_label(self):
        """Actualiza el entry de zoom con el porcentaje actual"""
        if self.image_loader.data is None:
            return
        _, h, w = self.image_loader.get_dimensions()
        max_dim = max(w, h)
        if max_dim > 0:
            percent = min(100.0, (self._zoom_level / max_dim) * 100)
            self.zoom_var.set(f"{percent:.1f}%")
    
    def _update_preview(self):
        """Regenera preview con zoom actual."""
        if self.image_loader.data is None:
            return
        _, height, width = self.image_loader.get_dimensions()
        preview = self.image_loader.generate_preview(self._zoom_level)
        self.interactive_canvas.set_image(preview, width, height)
        
        # Pasar geo_transform para medición en metros
        if self.image_loader.profile:
            transform = self.image_loader.profile.get('transform')
            crs = self.image_loader.profile.get('crs')
            self.interactive_canvas.set_geo_transform(transform, crs)
        
        self._update_zoom_label()
    
    # ========== MÉTODOS DE SELECCIÓN Y COPIA ==========
    
    def _toggle_selection_mode(self):
        """Alterna modo selección en el canvas."""
        mode = self._selection_mode_var.get()
        self.interactive_canvas.set_selection_mode(mode)
        # Desactivar modo regla si se activa selección
        if mode:
            self._ruler_mode_var.set(False)
            self.interactive_canvas.set_ruler_mode(False)
    
    def _toggle_ruler_mode(self):
        """Alterna modo regla (medir sin crear ROI)."""
        mode = self._ruler_mode_var.get()
        self.interactive_canvas.set_ruler_mode(mode)
        # Desactivar modo selección si se activa regla
        if mode:
            self._selection_mode_var.set(False)
            self.interactive_canvas.set_selection_mode(False)
            self.status_label.config(
                text="📏 MODO MEDIR: Arrastra para medir distancias | Las mediciones se muestran en tiempo real",
                foreground="#0066cc"
            )
        else:
            self._update_status()
    
    def _on_selection_changed(self, selected_ids: List[int]):
        """Callback cuando cambia la selección de ROIs."""
        count = len(selected_ids)
        self.selection_label.config(text=f"({count} seleccionados)")
    
    def _select_all_rois(self):
        """Selecciona todos los ROIs."""
        self.interactive_canvas.select_all_rois()
    
    def _deselect_all_rois(self):
        """Deselecciona todos los ROIs."""
        self.interactive_canvas.clear_selection()
    
    def _copy_selected_rois(self):
        """Copia ROIs seleccionados al clipboard."""
        selected = self.interactive_canvas.get_selected_rois()
        if not selected:
            messagebox.showwarning("Aviso", "Selecciona al menos un ROI para copiar.\n\nUsa Ctrl+Click o activa el modo Selección.")
            return
        
        count = self.roi_manager.copy_rois(selected)
        messagebox.showinfo("Copiado", f"{count} ROI(s) copiados al portapapeles.\n\nAhora usa [Pegar] para posicionarlos visualmente.")
    
    def _toggle_paste_mode(self):
        """Toggle del modo paste: inicia o detiene el modo pegado continuo."""
        # Si ya está en modo paste, detener
        if self.interactive_canvas.is_paste_mode():
            self.interactive_canvas.stop_paste_mode()
            self._paste_btn_text.set("Pegar")
            self._update_status()
            return
        
        # Iniciar modo paste
        if not self.roi_manager.has_clipboard():
            messagebox.showwarning("Aviso", "No hay ROIs en el portapapeles.\nPrimero copia algunos ROIs.")
            return
        
        if self.image_loader.data is None:
            messagebox.showwarning("Aviso", "Primero carga una imagen.")
            return
        
        # Iniciar modo paste visual continuo
        if self.interactive_canvas.start_paste_mode(
            on_paste_complete=self._on_paste_complete,
            on_status_update=self._on_paste_status_update
        ):
            # Cambiar texto del botón
            self._paste_btn_text.set("Dejar de Pegar")
            # Mostrar instrucciones
            self.status_label.config(
                text="📋 MODO PEGAR CONTINUO: Click izq para pegar | Click der/Esc para terminar",
                foreground="#cc6600"
            )
        else:
            messagebox.showwarning("Aviso", "No se pudo iniciar el modo pegar.")
    
    def _on_paste_status_update(self, message: str):
        """Callback para actualizar status durante paste continuo."""
        self.status_label.config(text=f"📋 {message}", foreground="#cc6600")
        # Verificar si el modo terminó
        if not self.interactive_canvas.is_paste_mode():
            self._paste_btn_text.set("Pegar")
            self.parent.after(2000, self._update_status)  # Restaurar status después de 2 seg
    
    def _on_paste_complete(self, new_rois: List):
        """Callback cuando se realiza un paste (sin cerrar el modo)."""
        # No mostrar mensaje, el modo continuo sigue activo
        pass
    
    def _show_matrix_dialog(self):
        """Muestra diálogo para crear matriz lineal de ROIs."""
        selected = self.interactive_canvas.get_selected_rois()
        if not selected:
            messagebox.showwarning("Aviso", "Selecciona al menos un ROI para crear matriz.\n\nUsa Ctrl+Click o activa el modo Selección.")
            return
        
        dialog = tk.Toplevel(self.parent)
        dialog.title("Matriz Lineal de ROIs")
        dialog.geometry("350x280")
        dialog.resizable(False, False)
        dialog.transient(self.parent)
        dialog.grab_set()
        
        # Centrar
        dialog.update_idletasks()
        x = self.parent.winfo_rootx() + 100
        y = self.parent.winfo_rooty() + 100
        dialog.geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=f"Crear matriz desde {len(selected)} ROI(s):", 
                  font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        ttk.Label(frame, text="Similar al comando ARRAY de AutoCAD", 
                  font=("Arial", 8), foreground="#666").pack(anchor=tk.W, pady=(0, 10))
        
        # Número de copias
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="Número de copias:", width=20).pack(side=tk.LEFT)
        copies_var = tk.IntVar(value=5)
        ttk.Entry(row1, textvariable=copies_var, width=10).pack(side=tk.LEFT)
        
        # Espaciado X
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=5)
        ttk.Label(row2, text="Espaciado X (píxeles):", width=20).pack(side=tk.LEFT)
        spacing_x_var = tk.IntVar(value=100)
        ttk.Entry(row2, textvariable=spacing_x_var, width=10).pack(side=tk.LEFT)
        
        # Espaciado Y
        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=5)
        ttk.Label(row3, text="Espaciado Y (píxeles):", width=20).pack(side=tk.LEFT)
        spacing_y_var = tk.IntVar(value=0)
        ttk.Entry(row3, textvariable=spacing_y_var, width=10).pack(side=tk.LEFT)
        
        ttk.Label(frame, text="Tip: X positivo = derecha, X negativo = izquierda\n      Y positivo = abajo, Y negativo = arriba\n      Para diagonal usa ambos valores", 
                  font=("Arial", 8), foreground="#666").pack(pady=8)
        
        # Botones
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=10)
        
        def do_matrix():
            try:
                copies = copies_var.get()
                sx = spacing_x_var.get()
                sy = spacing_y_var.get()
                
                if copies < 1:
                    messagebox.showwarning("Aviso", "El número de copias debe ser al menos 1.")
                    return
                
                if sx == 0 and sy == 0:
                    messagebox.showwarning("Aviso", "El espaciado no puede ser 0 en ambas direcciones.")
                    return
                
                new_rois = self.roi_manager.create_roi_array(selected, copies, sx, sy)
                self.interactive_canvas._redraw_all_rois()
                self._update_status()
                dialog.destroy()
                messagebox.showinfo("Matriz Creada", f"{len(new_rois)} ROI(s) creados en matriz lineal.")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        
        ttk.Button(btn_frame, text="Crear Matriz", command=do_matrix, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancelar", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)
    
    def _renumber_rois(self):
        """Renumera todos los ROIs según posición espacial."""
        if self.roi_manager.get_roi_count() == 0:
            messagebox.showwarning("Aviso", "No hay ROIs para renumerar.")
            return
        
        self.roi_manager.renumber_by_position()
        self.interactive_canvas.clear_selection()
        self.interactive_canvas._redraw_all_rois()
        self._refresh_roi_tree()
        self._update_status()
        
        messagebox.showinfo("Renumerado", 
            f"ROIs renumerados según posición espacial.\n"
            f"(izquierda→derecha, arriba→abajo)")
    
    # ========== MÉTODOS DE DOBLE-CLICK PARA RENUMERAR ROI ==========
    
    def _on_roi_double_click(self, roi_id: int):
        """Callback cuando se hace doble-click en un ROI del canvas."""
        roi = self.roi_manager.get_roi_by_id(roi_id)
        if roi is None:
            return
        
        self._show_rename_dialog(roi)
    
    def _show_rename_dialog(self, roi: ROI):
        """Diálogo para renumerar y/o renombrar un ROI."""
        dialog = tk.Toplevel(self.parent)
        dialog.title(f"Editar ROI {roi.id}")
        dialog.geometry("320x220")
        dialog.resizable(False, False)
        dialog.transient(self.parent)
        dialog.grab_set()
        
        # Centrar
        dialog.update_idletasks()
        x = self.parent.winfo_rootx() + 150
        y = self.parent.winfo_rooty() + 150
        dialog.geometry(f"+{x}+{y}")
        
        frame = ttk.Frame(dialog, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=f"ROI {roi.id}",
                  font=("Arial", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(frame, text=f"Posición: ({roi.x1}, {roi.y1}) → ({roi.x2}, {roi.y2})",
                  font=("Arial", 8), foreground="#666").pack(anchor=tk.W, pady=(0, 10))
        
        # Campo: Número de ROI
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=3)
        ttk.Label(row1, text="Número:", width=10).pack(side=tk.LEFT)
        id_var = tk.IntVar(value=roi.id)
        id_entry = ttk.Entry(row1, textvariable=id_var, width=10)
        id_entry.pack(side=tk.LEFT)
        
        # Campo: Nombre
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=3)
        ttk.Label(row2, text="Nombre:", width=10).pack(side=tk.LEFT)
        name_var = tk.StringVar(value=roi.name)
        name_entry = ttk.Entry(row2, textvariable=name_var, width=20)
        name_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Campo: Ángulo
        row3 = ttk.Frame(frame)
        row3.pack(fill=tk.X, pady=3)
        ttk.Label(row3, text="Ángulo (°):", width=10).pack(side=tk.LEFT)
        angle_var = tk.DoubleVar(value=roi.angle)
        angle_entry = ttk.Entry(row3, textvariable=angle_var, width=10)
        angle_entry.pack(side=tk.LEFT)
        
        ttk.Label(frame, text="Tip: el nombre aparece junto al número en el canvas",
                  font=("Arial", 8), foreground="#999").pack(anchor=tk.W, pady=(8, 0))
        
        def do_apply():
            try:
                new_id = id_var.get()
                new_name = name_var.get().strip()
                new_angle = angle_var.get()
                
                if new_id < 1:
                    messagebox.showwarning("Error", "El número debe ser >= 1", parent=dialog)
                    return
                
                # Cambiar ID si es diferente
                if new_id != roi.id:
                    try:
                        self.roi_manager.change_roi_id(roi.id, new_id)
                    except ValueError as e:
                        messagebox.showwarning("Error", str(e), parent=dialog)
                        return
                
                # Establecer nombre
                self.roi_manager.rename_roi(new_id, new_name)
                
                # Establecer ángulo
                self.roi_manager.set_roi_angle([new_id], new_angle)
                
                self.interactive_canvas._redraw_all_rois()
                self._refresh_roi_tree()
                self._update_status()
                dialog.destroy()
                
            except (ValueError, tk.TclError) as e:
                messagebox.showwarning("Error", f"Valor inválido: {e}", parent=dialog)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Aplicar", command=do_apply, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancelar", command=dialog.destroy, width=10).pack(side=tk.LEFT, padx=5)
        
        name_entry.focus_set()
        dialog.bind("<Return>", lambda e: do_apply())
        dialog.bind("<Escape>", lambda e: dialog.destroy())
    
    # ========== MÉTODOS DEL PANEL LATERAL (TREEVIEW) ==========
    
    def _refresh_roi_tree(self):
        """Actualiza el Treeview con los ROIs actuales."""
        # Limpiar
        for item in self.roi_tree.get_children():
            self.roi_tree.delete(item)
        
        # Rellenar
        for roi in self.roi_manager.get_all_rois():
            w = roi.x2 - roi.x1
            h = roi.y2 - roi.y1
            angle_str = f"{roi.angle:.0f}°" if roi.angle != 0 else ""
            self.roi_tree.insert("", tk.END, iid=str(roi.id),
                                 values=(roi.id, roi.name, roi.x1, roi.y1,
                                         roi.x2, roi.y2, f"{w}×{h}", angle_str))
    
    def _on_tree_double_click(self, event):
        """Doble-click en Treeview: editar nombre inline o abrir diálogo completo."""
        selection = self.roi_tree.selection()
        if not selection:
            return
        
        item_id = selection[0]
        try:
            roi_id = int(item_id)
        except ValueError:
            return
        
        # Detectar columna clickeada
        col = self.roi_tree.identify_column(event.x)
        col_idx = int(col.replace('#', '')) - 1  # 0-based
        columns = ("id", "name", "x1", "y1", "x2", "y2", "size", "angle")
        
        if col_idx < len(columns) and columns[col_idx] == "name":
            # Edición inline del nombre
            self._start_inline_edit(item_id, col, roi_id)
        else:
            # Diálogo completo
            roi = self.roi_manager.get_roi_by_id(roi_id)
            if roi:
                self._show_rename_dialog(roi)
    
    def _start_inline_edit(self, item_id: str, column: str, roi_id: int):
        """Inicia edición inline de nombre en el Treeview."""
        roi = self.roi_manager.get_roi_by_id(roi_id)
        if roi is None:
            return
        
        # Obtener bbox de la celda
        try:
            bbox = self.roi_tree.bbox(item_id, column)
        except (tk.TclError, ValueError):
            return
        
        if not bbox:
            return
        
        x, y, w, h = bbox
        
        # Crear Entry superpuesto
        edit_var = tk.StringVar(value=roi.name)
        entry = ttk.Entry(self.roi_tree, textvariable=edit_var, width=10)
        entry.place(x=x, y=y, width=w, height=h)
        entry.focus_set()
        entry.select_range(0, tk.END)
        
        def finish_edit(event=None):
            new_name = edit_var.get().strip()
            self.roi_manager.rename_roi(roi_id, new_name)
            entry.destroy()
            self._refresh_roi_tree()
            self.interactive_canvas._redraw_all_rois()
        
        def cancel_edit(event=None):
            entry.destroy()
        
        entry.bind("<Return>", finish_edit)
        entry.bind("<Escape>", cancel_edit)
        entry.bind("<FocusOut>", finish_edit)
    
    def _on_tree_delete(self, event):
        """Tecla Delete en Treeview: eliminar ROIs seleccionados."""
        self._tree_delete_selected()
    
    def _tree_delete_selected(self):
        """Elimina los ROIs seleccionados en el Treeview."""
        selection = self.roi_tree.selection()
        if not selection:
            return
        
        for item_id in selection:
            try:
                roi_id = int(item_id)
                self.roi_manager.remove_roi(roi_id)
            except ValueError:
                pass
        
        self.interactive_canvas.clear_selection()
        self.interactive_canvas._redraw_all_rois()
        self._refresh_roi_tree()
        self._update_status()
    
    def _tree_move_up(self):
        """Mueve el ROI seleccionado una posición arriba en la lista e intercambia IDs."""
        selection = self.roi_tree.selection()
        if not selection:
            return
        
        item_id = selection[0]
        try:
            roi_id = int(item_id)
        except ValueError:
            return
        
        # Encontrar índice en la lista
        rois = self.roi_manager.get_all_rois()
        idx = None
        for i, roi in enumerate(rois):
            if roi.id == roi_id:
                idx = i
                break
        
        if idx is None or idx == 0:
            return
        
        # Intercambiar IDs con el ROI de arriba
        other_id = rois[idx - 1].id
        self.roi_manager.swap_roi_ids(roi_id, other_id)
        
        # También intercambiar posiciones en la lista interna
        self.roi_manager.rois[idx], self.roi_manager.rois[idx - 1] = \
            self.roi_manager.rois[idx - 1], self.roi_manager.rois[idx]
        
        self.interactive_canvas._redraw_all_rois()
        self._refresh_roi_tree()
        
        # Mantener selección en el ROI movido (ahora tiene el ID del otro)
        new_id = str(other_id)
        if self.roi_tree.exists(new_id):
            self.roi_tree.selection_set(new_id)
            self.roi_tree.see(new_id)
    
    def _tree_move_down(self):
        """Mueve el ROI seleccionado una posición abajo en la lista e intercambia IDs."""
        selection = self.roi_tree.selection()
        if not selection:
            return
        
        item_id = selection[0]
        try:
            roi_id = int(item_id)
        except ValueError:
            return
        
        rois = self.roi_manager.get_all_rois()
        idx = None
        for i, roi in enumerate(rois):
            if roi.id == roi_id:
                idx = i
                break
        
        if idx is None or idx >= len(rois) - 1:
            return
        
        other_id = rois[idx + 1].id
        self.roi_manager.swap_roi_ids(roi_id, other_id)
        
        self.roi_manager.rois[idx], self.roi_manager.rois[idx + 1] = \
            self.roi_manager.rois[idx + 1], self.roi_manager.rois[idx]
        
        self.interactive_canvas._redraw_all_rois()
        self._refresh_roi_tree()
        
        new_id = str(other_id)
        if self.roi_tree.exists(new_id):
            self.roi_tree.selection_set(new_id)
            self.roi_tree.see(new_id)
    
    # --- Sincronización de selección TreeView → Canvas ---
    
    def _on_tree_selection_changed(self, event=None):
        """Sincroniza la selección del TreeView con el Canvas."""
        selected_items = self.roi_tree.selection()
        selected_ids = set()
        for item in selected_items:
            try:
                selected_ids.add(int(item))
            except ValueError:
                pass
        
        # Actualizar selección en el canvas
        self.interactive_canvas._selected_rois = selected_ids
        self.interactive_canvas._redraw_all_rois()
        self._update_selection_label()
    
    # ========== MÉTODO DE INVERSIÓN DE ORDEN ==========
    
    def _invert_selected_order(self):
        """Invierte el orden de IDs de los ROIs seleccionados."""
        selected = self.interactive_canvas.get_selected_rois()
        
        if len(selected) < 2:
            messagebox.showwarning("Aviso",
                "Selecciona al menos 2 ROIs para invertir su orden.\n\n"
                "Ejemplo: Si seleccionas 1-2-3-4-5-6\n"
                "Quedarán: 6-5-4-3-2-1\n\n"
                "Usa Ctrl+Click para selección múltiple o\n"
                "activa el modo Selección y dibuja un rectángulo.")
            return
        
        if self.roi_manager.invert_selected_order(selected):
            self.interactive_canvas._redraw_all_rois()
            self._refresh_roi_tree()
            self._update_status()
            
            # Mostrar resultado
            n = len(selected)
            ids_sorted = sorted(selected)
            messagebox.showinfo("Orden invertido", 
                f"Se invirtió el orden de {n} ROIs.\n\n"
                f"ROIs afectados: {ids_sorted[0]} a {ids_sorted[-1]}\n"
                f"El primero ahora es el último y viceversa.")
        else:
            messagebox.showerror("Error", "No se pudo invertir el orden.")


# ==================== INTEGRACIÓN ====================

def create_parametros_tab(notebook: ttk.Notebook) -> ParametrosTab:
    """
    Función de utilidad para crear e integrar la pestaña Parámetros.
    
    Args:
        notebook: ttk.Notebook donde agregar la pestaña
    
    Returns:
        Instancia de ParametrosTab
    
    Usage en IndicadorGUI.create_widgets():
        from parametros_tab import create_parametros_tab
        self.parametros_tab = create_parametros_tab(self.notebook)
    """
    tab_frame = ttk.Frame(notebook)
    notebook.add(tab_frame, text="Parámetros")
    
    return ParametrosTab(tab_frame)


# ==================== TEST STANDALONE ====================

if __name__ == "__main__":
    """Test standalone de la pestaña."""
    root = tk.Tk()
    root.title("Test - Pestaña Parámetros")
    root.geometry("900x700")
    
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Crear pestaña
    params_tab = create_parametros_tab(notebook)
    
    root.mainloop()
