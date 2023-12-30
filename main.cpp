#include <cstdlib>
#include <iostream>

int main()
{
    std::string activateCommand = "source ~/miniconda3/bin/activate cpl_pipeline";

    std::string pythonCommand = "python ~/repos/cpl_pipeline";

    std::string fullCommand = activateCommand + " && " + pythonCommand;

    int result = system(fullCommand.c_str());

    if (result == 0)
    {
        std::cout << "Env Activation Successful" << std::endl;
    }
    else
    {
        std::cout << "Error in Env Activation" << std::endl;
    }

    return 0;
}
