# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['F:/sheepy/00GitHub/aFolder_YOLO/01yolov5s/yolov5-7.0/yolov5-7.0/train_cls_pyqt.py'],
    pathex=[],
    binaries=[],
    datas=[('F:/sheepy/00GitHub/aFolder_YOLO/01yolov5s/yolov5-7.0/yolov5-7.0/classify', 'classify/'), ('F:/sheepy/00GitHub/aFolder_YOLO/01yolov5s/yolov5-7.0/yolov5-7.0/data', 'data/'), ('F:/sheepy/00GitHub/aFolder_YOLO/01yolov5s/yolov5-7.0/yolov5-7.0/models', 'models/'), ('F:/sheepy/00GitHub/aFolder_YOLO/01yolov5s/yolov5-7.0/yolov5-7.0/utils', 'utils/'), ('F:/sheepy/00GitHub/aFolder_YOLO/01yolov5s/yolov5-7.0/yolov5-7.0/requirements.txt', '.'), ('F:/sheepy/00GitHub/aFolder_YOLO/01yolov5s/yolov5-7.0/yolov5-7.0/yolov5s-cls.pt', '.')],
    hiddenimports=['models.yolo', 'subprocess'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='train_cls_pyqt',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='train_cls_pyqt',
)
