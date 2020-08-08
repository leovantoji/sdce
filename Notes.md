# C++ Prerequisite
- According to Bjarne Stroustrup, the creator of C++, C++ is a programming language that is **primarily for applications** that are very demanding on **performance, energy consumption** and **speed**. If there are serious concerns about **reliability, performance** and **response time**, C++ becomes a good choice.
- Each C++ program consists of two parts: the **preprocessor directives** and the **main function**.
  ```C++
  # include <iostream>

  int main() {
    std::cout << "Hello world, I am ready for C++";
    return 0;
  }
  ```
  - Any line that has a hash sign at the start is a preprocessor directive.
  - The brackets in `#include <iostream>` say "Look for this file in the directory where all the standard libraries are stored". C++ allows us to specify the library name using double quotes as well. `#include "main.hpp"`. The double quotes say "Look in the current directory, if the file is not there, then look in the directory where the standard libraries are stored".
