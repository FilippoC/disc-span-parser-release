#pragma once

#include <boost/regex.hpp>

namespace dytools
{

extern const boost::regex regex_num;
extern const boost::regex regex_punct;

std::string exec(const char* cmd);

bool is_num(const std::string& s);
bool is_punct(const std::string& s);
std::string to_lower(const std::string& s);


}