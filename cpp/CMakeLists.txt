cmake_minimum_required(VERSION 3.14)
project(burgers_1d)

set(CMAKE_CXX_STANDARD 20)
set(FFTW_USE_DOUBLE ON)
# include_directories(/opt/homebrew/Cellar/fftw/3.3.10/include)
# link_directories(/opt/homebrew/Cellar/fftw/3.3.10/lib)

# include(/opt/homebrew/Cellar/fftw/3.3.10/lib/cmake/fftw3/FFTW3Config.cmake)
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xtensor-fftw REQUIRED)
# find_package(FFTW3 REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_search_module(FFTW3 REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW3)
link_libraries(PkgConfig::FFTW3)

# add_executable(main main.cpp)
# target_link_libraries(main xtensor xtensor::optimize)

add_executable(test test.cpp)
target_link_libraries(test xtensor xtensor::optimize PkgConfig::FFTW3)