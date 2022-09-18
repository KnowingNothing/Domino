#ifndef DOMINO_PRINTER_H
#define DOMINO_PRINTER_H

#include <expr.h>
#include <fmt/ostream.h>

#include <iostream>
#include <string>

namespace domino {

std::ostream& operator<<(std::ostream& os, Expr expr);
std::string to_string(Expr expr);

}  // namespace domino

#endif
