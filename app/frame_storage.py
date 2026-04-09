from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, ContextManager
import numpy as np
import cv2
import os
import tempfile
import shutil
import threading
from contextlib import contextmanager

try:
    from .frame_perfome.video_stream import BufferedVideoReader
except ImportError:
    from frame_perfome.video_stream import BufferedVideoReader


class RWLock:
    """
    Простой read/write lock для потоков.

    - Несколько читателей могут заходить параллельно.
    - Писатель получает эксклюзивный доступ.
    - Приоритет писателя: если есть ожидающие писатели, новые читатели ждут,
      чтобы запись не голодала при постоянном потоке чтений.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @contextmanager
    def read_lock(self) -> ContextManager[None]:
        with self._cond:
            while self._writer or self._waiting_writers > 0:
                self._cond.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._cond:
                self._readers -= 1
                if self._readers == 0:
                    self._cond.notify_all()

    @contextmanager
    def write_lock(self) -> ContextManager[None]:
        with self._cond:
            self._waiting_writers += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
                self._writer = True
            finally:
                self._waiting_writers -= 1
        try:
            yield
        finally:
            with self._cond:
                self._writer = False
                self._cond.notify_all()


@dataclass
class FrameStorage:
    """
    Импортируйте в других модулях: from frame_contoller import frame_storage
    """
    webcam_frame: Optional[np.ndarray] = None      # Кадр с вебкамеры
    landmarks_frame: Optional[np.ndarray] = None   # Кадр с отображением обнаруженных точек
    tiles_frames: Optional[np.ndarray] = None      # Кадр с наложенным контентом
    preview_frames: Optional[np.ndarray] = None    # Кадр с наложенным контентом над вебкамерой
    mapping_frame: Optional[np.ndarray] = None     # Конечный кадр для вывода на проектор
    _rwlock: RWLock = field(default_factory=RWLock, init=False, repr=False)

    def set_webcam(self, frame: np.ndarray) -> None:
        with self._rwlock.write_lock():
            self.webcam_frame = frame.copy() if frame is not None else None

    def set_landmarks(self, frame: np.ndarray) -> None:
        with self._rwlock.write_lock():
            self.landmarks_frame = frame.copy() if frame is not None else None

    def set_tiles(self, frame: np.ndarray) -> None:
        with self._rwlock.write_lock():
            self.tiles_frames = frame.copy() if frame is not None else None

    def set_preview(self, frame: np.ndarray) -> None:
        with self._rwlock.write_lock():
            self.preview_frames = frame.copy() if frame is not None else None

    def set_mapping(self, frame: np.ndarray) -> None:
        with self._rwlock.write_lock():
            self.mapping_frame = frame.copy() if frame is not None else None

    def get_webcam(self) -> Optional[np.ndarray]:
        with self._rwlock.read_lock():
            return self.webcam_frame.copy() if self.webcam_frame is not None else None

    def get_landmarks(self) -> Optional[np.ndarray]:
        with self._rwlock.read_lock():
            return self.landmarks_frame.copy() if self.landmarks_frame is not None else None

    def get_tiles(self) -> Optional[np.ndarray]:
        with self._rwlock.read_lock():
            return self.tiles_frames.copy() if self.tiles_frames is not None else None

    def get_preview(self) -> Optional[np.ndarray]:
        with self._rwlock.read_lock():
            return self.preview_frames.copy() if self.preview_frames is not None else None

    def get_mapping(self) -> Optional[np.ndarray]:
        with self._rwlock.read_lock():
            return self.mapping_frame.copy() if self.mapping_frame is not None else None

    """Разрешение проектора и вебкамеры"""
    mapping_res : Optional[Tuple[int, int]] = None
    webcam_res : Optional[Tuple[int, int]] = None

    def set_mapping_res(self, width : int, height : int):
        with self._rwlock.write_lock():
            self.mapping_res = (width, height)

    def get_mapping_res(self):
        with self._rwlock.read_lock():
            return self.mapping_res
    
    def set_webcam_res(self, width : int, height : int):
        with self._rwlock.write_lock():
            self.webcam_res = (width, height)

    def get_webcam_res(self):
        with self._rwlock.read_lock():
            return self.webcam_res

@dataclass
class TilesStorage:
    torso: Optional[np.ndarray] = None
    l_arm: Optional[np.ndarray] = None
    r_arm: Optional[np.ndarray] = None
    l_leg: Optional[np.ndarray] = None
    r_leg: Optional[np.ndarray] = None

    texture: Optional[np.ndarray] = None
    # alpha маски (uint8 0..255) по имени файла
    masks_alpha: Optional[Dict[str, np.ndarray]] = None

    # Видео (лениво); те же начальные значения, что в __init__
    _video_path: Optional[str] = None
    _video_reader: Optional[BufferedVideoReader] = None
    _video_select_attempted: bool = False

    # Маски, один раз масштабированные под текущее разрешение видео.
    _masks_alpha_scaled: Dict[str, np.ndarray] = field(default_factory=dict)
    _masks_alpha_scaled_res: Optional[Tuple[int, int]] = None  # (vh, vw)

    # mask_name -> ((vh, vw), (y0, y1, x0, x1), overlay_BGRA)
    _overlay_cache: Dict[str, Tuple[Tuple[int, int], Tuple[int, int, int, int], np.ndarray]] = field(
        default_factory=dict
    )

    @staticmethod
    def _imread_unicode(path: str) -> Optional[np.ndarray]:
        """
        Безопасное чтение изображений, в том числе по путям с кириллицей.
        """
        if not os.path.exists(path):
            return None
        # np.fromfile корректно работает с Unicode-путями на Windows
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        return img

    def change_texure(self, path: str):
        img = self._imread_unicode(path)
        if img is None:
            raise FileNotFoundError(f"Не удалось открыть изображение по пути: {path}")
        self.texture = img

    def __init__(self) -> None:
        """
        Инициализация текстуры:
        - исходный файл берётся из папки модуля (в том числе с кириллицей в пути),
        - копируется во временную директорию,
        - читается через _imread_unicode.
        """
        temp_dir = tempfile.gettempdir()
        module_dir = os.path.dirname(os.path.abspath(__file__))
        tiles_dir = os.path.join(module_dir, "frame_perfome", "tiles")
        source = os.path.join(tiles_dir, "test_body.png")

        name = os.path.basename(source)
        dest = os.path.join(temp_dir, f"body_tile_{name}")

        # Перекопировать, если файла нет или он старее исходного
        if not os.path.exists(dest) or os.path.getmtime(dest) < os.path.getmtime(source):
            shutil.copy2(source, dest)

        img = self._imread_unicode(dest)
        if img is None:
            # Последняя попытка — прочитать напрямую исходный файл
            img = self._imread_unicode(source)
        if img is None:
            raise FileNotFoundError(
                f"Не удалось открыть изображение ни по временном пути: {dest}, ни по исходному: {source}"
            )

        self.texture = img
        self.masks_alpha :Dict = {}
        self._overlay_cache :Dict = {}
        self._video_path :str = None
        self._video_reader :BufferedVideoReader = None
        self._video_select_attempted :bool = False
        self._masks_alpha_scaled = {}
        self._masks_alpha_scaled_res = None

        self._load_default_masks(tiles_dir)

    def _load_default_masks(self, tiles_dir: str) -> None:
        """
        Загружает 5 PNG-масок из app/frame_perfome/tiles/.
        Если какой-то файл отсутствует/битый — просто пропускаем его,
        чтобы не ломать весь пайплайн.
        """
        for name in (
            "mask_torso.png",
            "mask_forearm_r.png",
            "mask_forearm_l.png",
            "mask_thigh_r.png",
            "mask_thigh_l.png",
        ):
            path = os.path.join(tiles_dir, name)
            img = self._imread_unicode(path)
            if img is None:
                continue

            alpha: Optional[np.ndarray] = None
            if img.ndim == 3 and img.shape[2] == 4:
                alpha = img[:, :, 3]
            elif img.ndim == 3 and img.shape[2] >= 3:
                alpha = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
            elif img.ndim == 2:
                alpha = img

            if alpha is None:
                continue

            if alpha.dtype != np.uint8:
                alpha = np.clip(alpha, 0, 255).astype(np.uint8)

            self.masks_alpha[name] = alpha

    def get_torso(self):
        if self.texture is None:
            raise RuntimeError("Текстура торса не инициализирована в TilesStorage.")
        return self.texture

    def get_mask_alpha(self, name: str) -> Optional[np.ndarray]:
        return self.masks_alpha.get(name)

    def set_video_path(self, path: str) -> None:
        self._video_path = path
        # переинициализируем ридер при смене пути
        if self._video_reader is not None:
            try:
                self._video_reader.stop()
            except Exception:
                pass
        self._video_reader = BufferedVideoReader(path)
        # При смене видео сбрасываем подготовленные маски и кеш оверлеев.
        self._masks_alpha_scaled = {}
        self._masks_alpha_scaled_res = None
        self._overlay_cache = {}

    def _scale_masks_to_video_res(self, vh: int, vw: int) -> None:
        """
        Масштабирует все alpha-маски в разрешение видео один раз.
        """
        if vh <= 0 or vw <= 0:
            return
        if self._masks_alpha_scaled_res == (vh, vw) and self._masks_alpha_scaled:
            return

        scaled: Dict[str, np.ndarray] = {}
        for name, alpha in (self.masks_alpha or {}).items():
            if alpha is None:
                continue
            if alpha.shape[:2] == (vh, vw):
                scaled[name] = alpha
            else:
                scaled[name] = cv2.resize(alpha, (vw, vh), interpolation=cv2.INTER_NEAREST)

        self._masks_alpha_scaled = scaled
        self._masks_alpha_scaled_res = (vh, vw)
        # bbox/overlay зависит от размеров => сбрасываем
        self._overlay_cache = {}

    def prepare_masks_for_video(self) -> None:
        """
        Вызывается один раз после успешного открытия видео.
        Берёт разрешение видео (через cv2.VideoCapture по пути) и масштабирует маски.
        """
        path = self._video_path
        if not path:
            return
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                cap.release()
                return
            vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
        except Exception:
            return
        self._scale_masks_to_video_res(vh, vw)

    def stop_videoreader(self) -> None:
        reader = self._video_reader
        if reader is None:
            return
        try:
            reader.stop()
        finally:
            self._video_reader = None

    def ensure_video_selected(self) -> None:
        """
        Запрашивает у пользователя видео-файл через диалог (один раз),
        если путь не задан и ридер не инициализирован.
        """
        if self._video_reader is not None:
            return
        if self._video_select_attempted:
            return
        self._video_select_attempted = True

        # tkinter для получения пути до видео
        try:
            import tkinter as tk
            from tkinter import filedialog
        except Exception:
            return
        print("Открываем tinker")
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Выберите видео для масок",
                filetypes=[
                    ("Video files", "*.mp4;*.mov;*.mkv;*.avi;*.webm"),
                    ("All files", "*.*"),
                ],
            )
            print(path)
            try:
                root.destroy()
            except Exception:
                pass
        except Exception:
            return

        if path:
            try:
                self.set_video_path(path)
                # Подготовка (масштабирование) масок под видео делается один раз.
                self.prepare_masks_for_video()
            except Exception:
                # не валим пайплайн, если видео не открылось
                return

    def get_video_frame(self) -> Optional[np.ndarray]:
        if self._video_reader is None:
            self.ensure_video_selected()
        if self._video_reader is None:
            return None
        return self._video_reader.get_latest_frame()

    def build_overlay_mask(self, mask_name: str) -> Optional[np.ndarray]:
        """
        Собирает BGRA: RGB берём из текущего кадра видео, alpha — из маски.
        Алгоритм:
        - масштабируем маску до размеров видео-кадра, чтобы разрешения совпали,
        - по маске берём область интереса (bbox ненулевой альфы),
        - обрезаем видео и маску до bbox (без пустых полей),
        - собираем BGRA на обрезанной области.

        Кеширует последний результат
        """
        overlays = self.build_overlay_masks_batch([mask_name])
        return overlays.get(mask_name)

    def build_overlay_masks_batch(
        self,
        mask_names: List[str],
        video_bgr: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Пакетная версия сборки overlay-ов для списка mask_names.

        Главная цель: использовать один и тот же кадр видео для всех масок
        (синхронность по времени + меньше накладных расходов на get_video_frame()).
        Возвращает словарь mask_name -> BGRA overlay только для успешно собранных масок.
        """
        if not mask_names:
            return {}

        if video_bgr is None:
            video_bgr = self.get_video_frame()
        if video_bgr is None:
            return {}

        vh, vw = video_bgr.shape[:2]
        out: Dict[str, np.ndarray] = {}

        for mask_name in mask_names:
            alpha = self.get_mask_alpha(mask_name)
            if alpha is None:
                continue

            # Маска должна быть в разрешении видео: предпочитаем заранее подготовленную.
            alpha_scaled = None
            if self._masks_alpha_scaled_res == (vh, vw):
                alpha_scaled = self._masks_alpha_scaled.get(mask_name)
            if alpha_scaled is None:
                # Фоллбэк: если prepare_masks_for_video() не сработал — масштабируем на лету.
                alpha_scaled = (
                    alpha
                    if alpha.shape[:2] == (vh, vw)
                    else cv2.resize(alpha, (vw, vh), interpolation=cv2.INTER_NEAREST)
                )

            # Находим bbox ненулевой альфы.
            ys, xs = np.where(alpha_scaled > 0)
            if ys.size == 0 or xs.size == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1

            key = mask_name
            cached = self._overlay_cache.get(key)
            if cached is not None and cached[0] == (vh, vw) and cached[1] == (y0, y1, x0, x1):
                overlay = cached[2]
            else:
                alpha_crop = alpha_scaled[y0:y1, x0:x1]
                h, w = alpha_crop.shape[:2]
                overlay = np.zeros((h, w, 4), dtype=np.uint8)
                overlay[:, :, 3] = alpha_crop
                self._overlay_cache[key] = ((vh, vw), (y0, y1, x0, x1), overlay)

            overlay[:, :, :3] = video_bgr[y0:y1, x0:x1]
            out[mask_name] = overlay

        return out

"""Единственный экземпляр — создаётся при первом импорте модуля"""
frames: FrameStorage = FrameStorage()
tiles: TilesStorage = TilesStorage()
#tiles.chenge_texure()
