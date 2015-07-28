PYTHON_ADD_MODULE(error_measures pybindings.cpp)
FILE(COPY test_bindings.py DESTINATION .)
ADD_TEST(c++ test_bindings.py)
