# dfd_gc
Depth from Defocus using Graph Cuts


# Windows build for projects

From the project directory type the following on the command line:

  mkdir build
  cd build
  cmake -G "Visual Studio 14 2015 Win64" -T host=x64 -DUSE_AVX_INSTRUCTIONS=ON ..
  cmake --build . --config Release

All build files will be created.

