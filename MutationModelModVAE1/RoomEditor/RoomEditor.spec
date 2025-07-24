# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Add data files that need to be included
added_files = [
    ('Data', 'Data'),  # Include the entire Data folder
]

a = Analysis(
    ['RoomEditor.py', 'tboi_vae.py', 'tboi_bitmap.py'],  # Include all Python files explicitly
    pathex=['.'],  # Add current directory to Python path
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'torch',
        'torchvision', 
        'torch.nn',
        'torch.nn.functional',
        'numpy',
        'PIL',
        'PIL.Image',
        'json',
        'pathlib',
        'math',
        'random',
        'time',
        'os',
        'tboi_vae',  # Explicitly include our custom modules
        'tboi_bitmap'
    ],
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
    name='RoomEditor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Keep console window open
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RoomEditor',
)
