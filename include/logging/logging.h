#ifndef DOMINO_DEBUG_H
#define DOMINO_DEBUG_H

#include <chrono>
#include <iostream>
#include <sstream>

namespace domino {

namespace logging {

int get_evn_value(std::string name);

std::chrono::milliseconds current_time();

enum class LogLevel { tINFO, tWARNING, tERROR };

class LazyLogging {
 private:
  LogLevel log_level;
  bool do_print;
  std::string file_;
  int lineno_;
  std::ostringstream oss;

 public:
  LazyLogging() = default;
  LazyLogging(const LazyLogging&& other) : log_level(other.log_level), do_print(other.do_print) {}
  LazyLogging(LogLevel level, bool do_print = true, std::string file = __FILE__,
              int lineno = __LINE__)
      : log_level(level), do_print(do_print), file_(file), lineno_(lineno) {}
  ~LazyLogging() {
    std::chrono::milliseconds ms = current_time();
    if (do_print) {
      switch (log_level) {
        case LogLevel::tINFO:
          std::cerr << "Domino: [Info] "
                    << "[time=" << ms.count() << "] " << oss.str() << std::flush;
          break;
        case LogLevel::tWARNING:
          std::cerr << "Domino: [Warning] "
                    << "[time=" << ms.count() << "] file:" << file_ << " line:" << lineno_ << " "
                    << oss.str() << std::flush;
          break;
        case LogLevel::tERROR: {
          std::cerr << "Domino: [Error] "
                    << "[time=" << ms.count() << "] " << file_ << " line:" << lineno_ << " "
                    << oss.str() << std::flush;
          abort();
        } break;
        default:
          break;
      }
    }
  }

  template <typename T>
  LazyLogging& operator<<(T& other) {
    oss << other;
    return *this;
  }

  template <typename T>
  LazyLogging& operator<<(T&& other) {
    oss << other;
    return *this;
  }
};

#define WARN(cond)                                                      \
  ([&]() -> LazyLogging {                                               \
    if (!(cond)) {                                                      \
      return LazyLogging(LogLevel::tWARNING, true, __FILE__, __LINE__); \
    } else {                                                            \
      return LazyLogging(LogLevel::tINFO, false, __FILE__, __LINE__);   \
    }                                                                   \
  }())

#define ASSERT(cond)                                                  \
  ([&]() -> LazyLogging {                                             \
    if (!(cond)) {                                                    \
      return LazyLogging(LogLevel::tERROR, true, __FILE__, __LINE__); \
    } else {                                                          \
      return LazyLogging(LogLevel::tINFO, false, __FILE__, __LINE__); \
    }                                                                 \
  }())

#define ERROR (ASSERT(false))

}  // namespace logging

}  // namespace domino

#endif // DOMINO_DEBUG_H