import sys
print("sys.path:")
for p in sys.path:
    print(f"  {p}")

import os
runtime_dir = os.path.dirname(sys.executable)
lib_dir = os.path.join(runtime_dir, "Lib", "site-packages")
print(f"\nLib/site-packages exists: {os.path.isdir(lib_dir)}")

sp2 = os.path.join(runtime_dir, "lib", "site-packages")
print(f"lib/site-packages exists: {os.path.isdir(sp2)}")

# Check the ._pth file
pth_file = os.path.join(runtime_dir, "python310._pth")
if os.path.exists(pth_file):
    print(f"\npython310._pth contents:")
    with open(pth_file) as f:
        print(f.read())
else:
    print(f"\npython310._pth NOT FOUND")

# Check if pip is available
try:
    import pip
    print(f"pip: {pip.__version__}")
except ImportError:
    print("pip: NOT FOUND")
