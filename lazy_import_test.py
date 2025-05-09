#!/usr/bin/env python3
"""遅延インポートのテストスクリプト"""

import time
import sys

def test_lazy_import():
    """遅延インポートが正しく機能することをテスト"""
    print("遅延インポートをテスト中...")
    
    # インポート時間を計測
    start_time = time.time()
    import segment_tools
    end_time = time.time()
    
    print(f"インポート時間: {end_time - start_time:.4f} 秒")
    
    # utilsに直接アクセスできることを確認（遅延ロードされない）
    print("\nutilsの確認（直接インポート）...")
    if hasattr(segment_tools, 'check_image_type'):
        print("✅ Utilsは正しくインポートされました")
    else:
        print("❌ Utilsはインポートされていません")
    
    # 遅延ロードされるクラスへのアクセス時間を計測
    print("\nCLIPSegの遅延ロードを確認...")
    print("CLIPSegクラスにアクセス中...")
    start_time = time.time()
    clipseg_class = segment_tools.CLIPSeg
    end_time = time.time()
    print(f"初回の遅延ロードアクセス時間: {end_time - start_time:.4f} 秒")
    
    # 2回目のアクセスは瞬時であるべき
    print("\nCLIPSegクラスに再度アクセス（キャッシュされているはず）...")
    start_time = time.time()
    clipseg_class = segment_tools.CLIPSeg
    end_time = time.time()
    print(f"2回目のアクセス時間: {end_time - start_time:.4f} 秒")
    
    # まだアクセスされていない別のモジュールを確認
    print("\nDINOの遅延ロードを確認...")
    print("DINOクラスにアクセス中...")
    start_time = time.time()
    dino_class = segment_tools.DINO
    end_time = time.time()
    print(f"DINOの遅延ロードアクセス時間: {end_time - start_time:.4f} 秒")
    
    print("\n遅延インポートテストが正常に完了しました！")

if __name__ == "__main__":
    test_lazy_import()