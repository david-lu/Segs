(
  cd sam2 || exit 1
  echo "Rebuilding sam2 native extension..."
  pip uninstall -y SAM-2
  rm -rf build/ sam2/*.so
  python setup.py build_ext --inplace
#  SAM2_BUILD_ALLOW_ERRORS=0 pip install -v -e ".[notebooks]"
)