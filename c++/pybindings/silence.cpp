#include <boost/python.hpp>

#include "../silence.hpp"

void export_silence()
{
    using namespace boost::python;
    
    def("restore_output", restore_output);
    def("test_output",    test_output);
    def("redirect_output_to_file", redirect_output_to_file);
}


BOOST_PYTHON_MODULE(silence) {
    export_silence();
}
