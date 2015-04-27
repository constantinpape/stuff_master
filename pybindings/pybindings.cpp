#include <boost/python.hpp>
#include "utilities.hpp"
#include <iostream>

using namespace boost::python;

void blob()
{
	std::cout << "BLOB" << std::endl;
} 

BOOST_PYTHON_MODULE(mypybindings)
{
	def("blob", blob);
}

//BOOST_PYTHON_MODULE(utilities)
//{
//	def("variation_of_information", variation_of_information);
//}
