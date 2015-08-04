#include <boost/python.hpp>

#include <numpy/arrayobject.h>
#include <numpy/noprefix.h>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include "f_score.hpp"

template <class T>
vigra::NumpyArray<1, double> py_fscore(
    vigra::NumpyArray<1, T> segA,
    vigra::NumpyArray<1, T> segB)
{
    vigra::NumpyArray<1, double> ret( vigra::NumpyArray<1, double>::difference_type(2) );
    std::pair<double, double> res = fscore( segA.begin(), segA.end(), segB.begin(), segB.end() );
    ret[0] = res.first;
    ret[1] = res.second;
    return ret;
}

template<class T>
void export_fscore_t()
{
    using namespace boost::python;
    
    def("compute_fscore",
        vigra::registerConverters(&py_fscore<T>),
        ( arg("segA"), arg("segB") )
        );
}

void export_fscore()
{
    export_fscore_t<vigra::UInt32>();
    export_fscore_t<vigra::UInt16>();
}


BOOST_PYTHON_MODULE(error_measures) {

    // Do not change next 4 lines
    import_array(); 
    vigra::import_vigranumpy();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    boost::python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above
    
    export_fscore();
}
