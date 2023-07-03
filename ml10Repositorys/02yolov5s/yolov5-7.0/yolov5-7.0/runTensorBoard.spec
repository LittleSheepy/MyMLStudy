# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['F:/sheepy/00GitHub/aFolder_YOLO/01yolov5s/yolov5-7.0/yolov5-7.0/runTensorBoard.py'],
    pathex=[],
    binaries=[],
    datas=[('D:/SDK/Anaconda3/envs/py38torch190/Lib/site-packages/tensorboard/webfiles.zip', 'tensorboard/')],
    hiddenimports=['tensorboard'],
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
    name='runTensorBoard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='runTensorBoard',
)
