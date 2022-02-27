#include "verbose.hpp"

bool check_verbosity() {
	if(std::getenv("COMP_OPT_DEBUG")) {
		return true;
	}
	return false;
}
