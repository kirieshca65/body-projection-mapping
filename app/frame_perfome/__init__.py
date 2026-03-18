#from .tiles_deform import draw_overlay
from .video_stream import BufferedVideoReader

def draw_overlay(*args, **kwargs):
    # Ленивый импорт, чтобы не создавать цикл:
    # frame_storage -> frame_perfome (init) -> tiles_deform -> frame_storage
    from .tiles_deform import draw_overlay as _draw_overlay

    return _draw_overlay(*args, **kwargs)


__all__ = ["BufferedVideoReader", "draw_overlay"]