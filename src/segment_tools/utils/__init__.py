# ユーティリティモジュール
import os
import sys
import importlib.util
import importlib.machinery
from typing import Dict, Any, Optional

class DynamicModuleLoader:
    """サブモジュールを動的にロードするユーティリティクラス"""
    
    def __init__(self, base_dir: str, package_name: str):
        """
        初期化
        
        Args:
            base_dir: モジュールのベースディレクトリ
            package_name: パッケージ名
        """
        self.base_dir = base_dir
        self.package_name = package_name
        self.modules: Dict[str, Any] = {}
        
        # パッケージの名前空間を作成
        if package_name not in sys.modules:
            module_spec = importlib.machinery.ModuleSpec(package_name, None)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[package_name] = module
            self.modules[package_name] = module
    
    def ensure_parent_modules(self, module_name: str) -> None:
        """
        親モジュールの名前空間を確保する
        
        Args:
            module_name: モジュール名（ドット区切り）
        """
        parts = module_name.split('.')
        
        # ルートパッケージは既に作成済み
        if len(parts) <= 1:
            return
            
        # 段階的に親モジュールを作成
        current = self.package_name
        for part in parts[1:-1]:  # 最後のモジュール名を除く
            parent = current
            current = f"{parent}.{part}"
            
            if current not in sys.modules:
                module_spec = importlib.machinery.ModuleSpec(current, None)
                module = importlib.util.module_from_spec(module_spec)
                sys.modules[current] = module
                self.modules[current] = module
                
                # 親モジュールに属性として追加
                setattr(sys.modules[parent], part, module)
    
    def load_module(self, module_path: str, module_name: Optional[str] = None) -> Any:
        """
        モジュールをロードする
        
        Args:
            module_path: モジュールファイルのパス
            module_name: モジュール名（指定しない場合はパッケージ名を使用）
            
        Returns:
            ロードされたモジュール
        """
        if module_name is None:
            rel_path = os.path.relpath(module_path, self.base_dir)
            module_name_parts = []
            
            # パス部分をモジュール名に変換
            path_parts = os.path.dirname(rel_path).split(os.path.sep)
            for part in path_parts:
                if part and part != '.':
                    module_name_parts.append(part)
                    
            # ファイル名（拡張子なし）を追加
            basename = os.path.basename(rel_path)
            module_name_parts.append(os.path.splitext(basename)[0])
            
            # パッケージプレフィックスを追加
            full_module_name = f"{self.package_name}.{'.'.join(module_name_parts)}"
        else:
            full_module_name = f"{self.package_name}.{module_name}" if '.' not in module_name else module_name
        
        # 親モジュールの名前空間を確保
        self.ensure_parent_modules(full_module_name)
        
        # モジュールをロード
        module_spec = importlib.util.spec_from_file_location(full_module_name, module_path)
        if not module_spec:
            raise ImportError(f"モジュール '{module_path}' の仕様を取得できませんでした")
            
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[full_module_name] = module
        self.modules[full_module_name] = module
        
        # 親モジュールに属性として追加
        parts = full_module_name.split('.')
        if len(parts) > 1:
            parent_name = '.'.join(parts[:-1])
            attr_name = parts[-1]
            parent_module = sys.modules.get(parent_name)
            if parent_module:
                setattr(parent_module, attr_name, module)
        
        # モジュールを実行
        module_spec.loader.exec_module(module)
        return module

# 共通ユーティリティ関数
def check_image_type(image, return_type=None):
    """
    画像の型を変換する便利関数
    
    Args:
        image: 文字列パス、PILイメージ、またはnumpy配列
        return_type: 返却する型（'pil'または'numpy'）
        
    Returns:
        指定された型の画像
    """
    import numpy as np
    
    # 画像がパスの場合、読み込む
    if isinstance(image, str):
        from PIL import Image
        
        if return_type == 'numpy':
            import cv2
            return cv2.imread(image)
        else:
            return Image.open(image).convert("RGB")
    
    # 画像がnumpy配列の場合
    elif isinstance(image, np.ndarray):
        if return_type == 'pil':
            from PIL import Image
            return Image.fromarray(image).convert("RGB")
        else:
            return image
    
    # それ以外はそのまま返す（PILイメージなど）
    else:
        if return_type == 'numpy' and not isinstance(image, np.ndarray):
            import numpy as np
            return np.array(image)
        elif return_type == 'pil' and isinstance(image, np.ndarray):
            from PIL import Image
            return Image.fromarray(image).convert("RGB")
        else:
            return image

# 他のユーティリティ関数をここに追加