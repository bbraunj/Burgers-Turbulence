# Burgers Turbulence 1D C++ Implementation

## Dependencies:
- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page)
- [matplotlib-cpp](https://github.com/lava/matplotlib-cpp)
- [FFTW3](http://www.fftw.org)
- [CMake](https://cmake.org)

## Usage
```bash
$ cmake --no-warn-unused-cli \
  -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE \
  -DCMAKE_BUILD_TYPE:STRING=Release \
  -DCMAKE_C_COMPILER:FILEPATH=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/clang++ \
  -B ./build \
  -G "Unix Makefiles"
$ cmake --build ./build --config Release --target all --

# Run with NX=256,512,1024
$ ./build/main 256 512 1024

# Inspect output
$ open E_k_nx=256_512_1024.pdf
```
