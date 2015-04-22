#include <boost/python.hpp>
#include "utilities.hxx"

using namespace boost::python;

BOOST_PYTHON_MODULE(utilities)
{
	def("variation_of_information", variation_of_information)
}
