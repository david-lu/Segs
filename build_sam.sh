(
  cd sam2 || exit 1
  echo "Rebuilding sam2 native extension..."
  rm -rf build/ sam2/_C.*.so
  pip install --no-deps --force-reinstall --no-cache-dir .
)