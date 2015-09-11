#pragma once

#include <iostream>
#include <cstdio>

void redirect_output_to_file(const char * file)
{
    freopen(file, "w", stdout);
    freopen(file, "w", stderr);
}

// this is a little dubious....
void restore_output()
{
    freopen("/dev/tty", "a", stdout);
    freopen("/dev/tty", "a", stderr);
}


void test_output()
{
    std::cout << "This is a testing test of tested tests!" << std::endl;
}

