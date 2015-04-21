#include "utilities.hpp"
#include <iostream>

int main()
{
	std::vector<bool> set1{ {0,1,0,1,1,0,0,1,0,1,0,0,1} };
	std::vector<bool> set2{ {1,1,0,0,1,0,0,0,1,0,1,0,1} };
	
	std::cout << variation_of_information( set1, set2) << std::endl;

	return 0;
}
