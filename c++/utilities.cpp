#include "utilities.hpp"
#include <array>
#include <algorithm>
#include <functional>
#include <iostream>
#include <cmath>

// calculate the variation of information for binary labeling
double variation_of_information(std::vector<bool> set1, std::vector<bool> set2)
{
	if( set1.size() != set2.size() )
	{
		throw std::runtime_error("Trying to compute variation of information for sets of different size!");
	}
	size_t N = set1.size();
	std::array<double, 2> P_array0;
	std::array<double, 2> P_array1;
	std::array<double, 4> P_array01;
	// count occurences
	for( size_t i = 0; i < set1.size(); i++)
	{
		if( set1[i] == 0 and set2[i] == 0)
		{
			P_array0[0]++;
			P_array1[0]++;
			P_array01[0]++;
		}	
		else if( set1[i] == 0  and set2[i] == 1)
		{
			P_array0[0]++;
			P_array1[1]++;
			P_array01[1]++;
		}
		else if( set1[i] == 1  and set2[i] == 0)
		{
			P_array0[1]++;
			P_array1[0]++;
			P_array01[2]++;
		}
		else if( set1[i] == 1  and set2[i] == 1)
		{
			P_array0[1]++;
			P_array1[1]++;
			P_array01[3]++;
		}
	}
	// normalize
	std::transform(P_array0.begin(), P_array0.end(), P_array0.begin(), std::bind2nd( std::divides<double>(), N ) );
	std::transform(P_array1.begin(), P_array1.end(), P_array1.begin(), std::bind2nd( std::divides<double>(), N ) );
	std::transform(P_array01.begin(), P_array01.end(), P_array01.begin(), std::bind2nd( std::divides<double>(), N ) );
	// compute the seperate entropies
	double H0 = - ( P_array0[0] * std::log2(P_array0[0]) + P_array0[1] * std::log2(P_array0[1]) );
	double H1 = - ( P_array1[0] * std::log2(P_array1[0]) + P_array1[1] * std::log2(P_array1[1]) );
	// compute the mutual information
	double I  = P_array01[0] * log2( P_array01[0] / ( P_array0[0] * P_array1[0] ) ) +
		    P_array01[1] * log2( P_array01[1] / ( P_array0[0] * P_array1[1] ) ) +
		    P_array01[2] * log2( P_array01[2] / ( P_array0[1] * P_array1[0] ) ) +
                    P_array01[3] * log2( P_array01[3] / ( P_array0[1] * P_array1[1] ) );
	return H0 + H1 - 2.*I;
}

