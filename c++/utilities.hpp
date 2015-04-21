#pragma once

#include <vector>

// variation of information - non templated
double variation_of_information(std::vector<bool> set1, std::vector<bool> set2);

// variation of information - templated
// dummy 
template<class ITERATOR_0, class ITERATOR_1>
double variatio_of_information( ITERATOR_0 begin_0, ITERATOR_0 end_0, ITERATOR_1 begin_1)
{
	return 0.;
}



