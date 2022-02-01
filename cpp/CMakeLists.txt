cmake_minimum_required(VERSION 3.14)
project(burgers_1d)

set(CMAKE_CXX_STANDARD 20)
set(FFTW_USE_DOUBLE ON)

find_package(PkgConfig REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

pkg_search_module(FFTW3 REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW3)
link_libraries(PkgConfig::FFTW3)

# pkg_search_module(gperftools REQUIRED libprofiler IMPORTED_TARGET)
# link_libraries(PkgConfig::gperftools)

# For matplotlib-cpp
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy)

add_executable(main main.cpp)
target_include_directories(main PUBLIC /usr/local/include)
target_link_libraries(main PkgConfig::FFTW3 Eigen3::Eigen Python3::Python Python3::Module Python3::NumPy)
# target_link_libraries(main PkgConfig::FFTW3 Eigen3::Eigen PkgConfig::gperftools Python3::Python Python3::Module Python3::NumPy)

# add_executable(test test.cpp)
# target_include_directories(test PUBLIC /usr/local/include)
# target_link_libraries(test PkgConfig::FFTW3 Eigen3::Eigen Python3::Python Python3::Module Python3::NumPy)