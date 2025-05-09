"""インポート時間を改善するための遅延インポートを使用するセグメンテーションツール。"""

# utilsは共通ユーティリティを含むため直接インポート
from .utils import *

# 遅延ロード用の全モジュールクラスマッピングを保存
_MODULES_MAP = {
    "CLIPSeg": "clipseg_module",
    "DINO": "dino_module",
    "DINOSeg": "dinoseg_module",
    "OneFormer": "oneformer_module",
    "DepthAnything": "depthanything_module",
    "DINOv2_depth": "dinov2_module",
    "XMem": "xmem_module",
    "SAM": "sam_module",
    "GRiT": "grit_module",
    "DepthPro": "depthpro_module",
    "InsightFace": "insightface_module",
    "SAM2": "sam2_module",
    "Florence": "florence_module",
}

# インポート済みモジュールを追跡する辞書
_modules_cache = {}

def __getattr__(name):
    """アクセスされた時にのみ遅延インポートする。"""
    # 名前が既知のクラスである場合、そのモジュールをインポートしてクラスを返す
    if name in _MODULES_MAP:
        module_name = _MODULES_MAP[name]
        if module_name not in _modules_cache:
            import importlib
            _modules_cache[module_name] = importlib.import_module(f".{module_name}", package=__name__)
        
        # モジュールからクラスを返す
        return getattr(_modules_cache[module_name], name)
        
    # 既知のクラスでない場合、いずれかのモジュールで属性を見つけようとする
    for module_name in _MODULES_MAP.values():
        if module_name not in _modules_cache:
            import importlib
            try:
                _modules_cache[module_name] = importlib.import_module(f".{module_name}", package=__name__)
            except ImportError:
                continue
            
        # この属性がモジュールに存在するか確認
        if hasattr(_modules_cache[module_name], name):
            return getattr(_modules_cache[module_name], name)
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# 全クラス名を明示的に公開
__all__ = list(_MODULES_MAP.keys()) + ["__getattr__"]