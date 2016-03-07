#include <boost/python.hpp>

#include <numpy/arrayobject.h>
#include <numpy/noprefix.h>
#include <vigra/numpy_array.hxx>
#include <vigra/numpy_array_converters.hxx>

#include "../fuzzy_gt_projection.hpp"

void export_fuzzy_gt_projection()
{
    using namespace boost::python;
    
    def("fuzzy_gt_projection",vigra::registerConverters(&candidateSegToRagSeg),
        (
            arg("ragLabels"),
            arg("candidateLabels"),
            arg("uvIds"),
            arg("out") = object()
        )
    )
    ;
}

BOOST_PYTHON_MODULE(gt_projections) {

    // Do not change next 4 lines
    import_array(); 
    vigra::import_vigranumpy();
    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
    boost::python::docstring_options docstringOptions(true,true,false);
    // No not change 4 line above
    
    export_fuzzy_gt_projection();
}
