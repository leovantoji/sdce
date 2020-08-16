# Udacity C++ For Programmers
## Basics
- According to Bjarne Stroustrup, the creator of C++, C++ is a programming language that is **primarily for applications** that are very demanding on **performance, energy consumption** and **speed**. If there are serious concerns about **reliability, performance** and **response time**, C++ becomes a good choice.
- Each C++ program consists of two parts: the **preprocessor directives** and the **main function**.chủ yếu bác Lê ạ. Nên cái vụ dịch bệnh covid này nhất là đợt 2 tái xuất ở ĐN 
  ```C++
  #include <iostream>

  int main() {
    std::cout << "Hello world, I am ready for C++";
    return 0;
  }
  ```
  - Any line that has a hash sign at the start is a preprocessor directive.
  - The brackets in `#include <iostream>` say "Look for this file in the directory where all the standard libraries are stored". C++ allows us to specify the library name using double quotes as well. `#include "main.hpp"`. The double quotes say "Look in the current directory, if the file is not there, then look in the directory where the standard libraries are stored".
- Writing `std::` can be a pain, so C++ actually offers a shortcut for writing cout. Before the start of the main function, put in the command `using namespace std;`. This tells the compiler to assume we are using the standard library, so we don't have to write `std::`. Nonetheless, when the commands are not explicitly defined, there is a possibility that when your code is added to a large project, your code might reference a command from a different library.
  ```C++
  #include <iostream>
  
  using namespace std;
  int main() {
    cout << "Hey, writing std:: is a pain, ";
    cout << "change the program so I don't have to write it.";
    return 0;
  }
  ```
- Write to the console:
  ```C++
  int number = 4543;
  std::cout << "The value of the integer is " << number << "\n";
  ```
- As with other programming languages, the size a variable is allocated in memory is dependent upon its type. To determine how many bytes each variable type uses, C++ provides the function `sizeof(variableType)`.
  ```C++
  #include <iostream>

  using namespace std;
  int main() {
      cout << "int size = " << sizeof(int) << endl;
      cout << "short size = " << sizeof(short) << endl;
      cout << "long size = " << sizeof(long) << endl;
      cout << "char size = " << sizeof(char) << endl;
      cout << "float size = " << sizeof(float) << endl;
      cout << "double size = " << sizeof(double) << endl;
      cout << "bool size = " << sizeof(bool) << endl;
      return 0;
  }
  ```
- We use keyword `const` to define a constant: `const int weightGoal = 100;`.
- C++ allows for **enumerated constants**. This means the programmer can create a new variable type and then assign a finite number of values to it. For example:
  ```C++
  #include <iostream>
  
  using namespace std;
  int main() {
    //define MONTHS as having 12 possible values
    enum MONTH {Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec};
    
    //define bestMonth as a variable type MONTHS
    MONTH bestMonth;
    
    //assign bestMonth one of the values of MONTHS
    bestMonth = Jan;
    
    //check the value of bestMonth just like any other variable
    if (bestMonth == Jan) {
      cout << "I'm not so sure if January is the best month\n";
    }
    
    return 0;
  }
  ```
- Format Output:
  ```C++
  #include <iomanip>

  std::cout<<"\n\nThe text without any formating\n";
  std::cout<<"Ints"<<"Floats"<<"Doubles"<<"\n";
  std::cout<<"\nThe text with setw(15)\n";
  std::cout<<"Ints"<<std::setw(15)<<"Floats"<<std::setw(15)<<"Doubles"<<"\n";
  std::cout<<"\n\nThe text with tabs\n";
  std::cout<<"Ints\t"<<"Floats\t"<<"Doubles"<<"\n";
  ```
- File IO steps:
  - Include the `<fstream>` library.
  - Create a stream (input, output, both):
    - Writing to a file: `ofstream myfile;`.
    - Reading a file: `ifstream myfile;`.
    - Reading and Writing a file: `fstream myfile;`.
  - Open the file: `myfile.open("filename");`.
  - Close the file: `myfile.close();`.
  ```C++
  #include <iostream>
  #include <fstream>
  #include <string>
  using namespace std;
  
  int main() {
    string line;
    //create an output stream to write to the file
    //append the new lines to the end of the file
    ofstream myfileI ("input.txt", ios::app);
    if (myfileI.is_open()) {
      myfileI<<"\n I am adding a line.\n";
      myfileI<<"I am adding another line.\n";
      myfileI.close();
    } else cout<<"Unable to open file for writing";
    
    //create an input stream to read the file
    ifstream myfileO ("input.txt");
    //during the creation of ifstream, the file is opened.
    //so we don't have to explicitly open the file.
    if (myfileO.is_open()) {
      while (getline(myfileO, line)) {
        cout<<line<<"\n";
      }
    } else cout<<"Unable to open file for reading";
    
    return 0;
  }
  ```
- User-created libraries can also be included in C++. Traditionally, these files are called **header files**, and they have an `.hpp` extension. Nonetheless, any extension will work.
  - Header files contain information about how to do a task.
  - The main program contains information about what to do.
  - For example:
    - `main.hpp`
    ```C++
    #include <iostream>
    using namespace std;
    ```
    - `main.cpp`
    ```C++
    #include "main.hpp"

    int main() {
      cout<<"Hello, I use header files!";
      return 0;
    }
    ```
- User Input. Note that `std::cin` doesn't retrieve strings that have a space in them. It treats space as the end of the input. To read string input, we can use `std::getline` command, which retrieves characters from `std::cin` source and stores them in a variable. `std::getline` will retrieve all characters until the newline or "\n" is detected.
  ```C++
  #include <iostream>
  #include <string>
  
  int main() {
    int year = 0;
    //print a message to the user
    std::out<<"What year is your favourite? ";
    
    //get the user response and assign it the variable year
    std::cin>>year;
    
    //output response to the user
    std::cout<<"How interesting, your favourite year is "<<year<<"!\n";
    
    std::string userName;
    std::cout<<"Tell me your nickname?: ";
    std::getline(std::cin, userName);
    std::cout<<"Hello "<<userName<"!\n";
    
    return 0;
  }
  ```
- Stringstream:
  ```C++
   #include <iostream>
   #include <string>
   #include <sstream>

   int main() {
     std::string stringLength, stringWidth;
     float length = 0;
     float width = 0;
     float area = 0;

     std::cout<<"Enter the length of the room: ";
     //get the length as a string
     std::getline(std::cin, stringLength);
     //convert to a float
     std::stringstream(stringLength)>>length;
     //get the width as a string
     std::cout<<"Enter width: ";
     std::getline(std::cin, stringWidth);
     //convert to a float
     std::stringstream(stringWidth)>>width;
     area = length * width;
     std::cout<<"\nThe area of the room is: "<<area<<std::endl;
     return 0;
   }
  ```

## Compilation and Execution
- **Compiling** is the **process of translating the code** that you have written **into machine code** that processors understand. Every program, regardless of the source language, needs to be compiled in order to execute. This is true even for scripting languages like Python or JavaScript. In these cases, the interpreter (or a similar system) is responsible for **compiling code on the fly** in a process known as **just-in-time compiling**. Unlike scripted languages, compiled languages treat compilation and execution as two distinct steps. Compiling a program leaves you with an executable (often called a "binary"), a non-human readable file with machine code that the processor can run.
- Compiling source code, like a single `.cpp` file, results in something called an **object files**. An object file **contains machine code** but may **not be executable** in and of itself. Among other things, object files describe their own public APIs (usually called symbols) as well as references that need to be resolved from other object files. Depended upon object files might come from other source files within the same project or from external or system libraries. In order to be executable, object files need to be linked together.
- **Linking** is the process of creating an executable by effectively combining object files. During the linking process, the linker resolves symbolic references between object files and outputs a self-contained binary with all the machine code needed to execute.
- As an aside, linking is not required for all programs. Most operating systems allow **dynamic linking**, in which symbolic references point to libraries that are not compiled into the resulting binary. With dynamic linking, these references are resolved at runtime. An example of this is a program that depends on a system library. At runtime, the symbolic references of the program resolve to the symbols of the system library.
- Pros and Cons of dynamic linking:
  - That dynamically linked libraries are not linked within binaries keeps the overall file size down.
  - If the libraries change at any point, the code will automatically link to the new version. This may lead to difficult and surprising bugs.
- To compile in a terminal:
  - Open a terminal window.
  - Change the working directory to the directory of the program.
  - Make sure names of folders and files do not have spaces in them.
  - To compile the program: `g++ filename.cpp -o executableName`.
  - To execute the program: `./executatbleName`.
 
## Arithmetic Operations
- When doing math operations, you may need to include `cmath` library, it contains a number of useful functions.
- Some practice:
  ```C++
  #include <iostream>
  #include <cmath>
  
  int main() {
    float cubeSide = 5.4;
    float sphereRadius = 2.33;
    float coneRadius = 7.65;
    float coneHeight = 14;
    
    float volCube, volSphere, volCone = 0;
    
    volCube = std::pow(cubeSide, 3)
    volSphere = (4/3.0)*M_PI*std::pow(sphereRadius, 3)
    volCone = M_PI*std::pow(coneRadius, 2)*coneHeight/3.0
    
    std::cout<<"Volume of the cube: "<<volCube<<"\n";
    std::cout<<"Volume of the sphere: "<<volSphere<<"\n";
    std::cout<<"Volume of the cone: "<<volCone<<"\n";
    
    return 0;
  }
  ```
- C++ is a language that requires variable types to be known at compile time. However, C++ does allow some implicit conversions. For example, an `int` can be assigned to a `float`, or an `int` can be treated as a `char`.
- Prefix and Postfix examples are `++a` and `a++` respectively. The **difference** between prefix and postfix is subtle but crucial. 
  - **Prefix operators** increment the value of the variable, then return the reference to the variable.
  - **Postfix operators** create a copy of the variable, increment the value of the variable, then
return a copy from **BEFORE** the increment.

## Intro to Control Flow
- C++ has several control flow options:
  - `if-else`, `jump`, `switch` statements.
  - `for`, `while` loops.


