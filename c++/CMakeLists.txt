PYTHON_ADD_MODULE(error_measures pybindings/error_measures.cpp)
PYTHON_ADD_MODULE(silence pybindings/silence.cpp)
FILE(COPY test_bindings.py DESTINATION .)
ADD_TEST(c++ test_bindings.py)
