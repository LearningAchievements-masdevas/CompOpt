#ifndef VERBOSE_HPP
#define VERBOSE_HPP

#include <cstdlib>
#include <iostream>

bool check_verbosity();

template <typename... Args>
inline void verbose_print(bool verbosity, Args&&... args) {
	if (verbosity) {
		(std::cout << ... << args) << std::endl;
	}
}

#endif
