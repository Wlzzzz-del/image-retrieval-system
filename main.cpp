#include "extract.h"
#include <string>
#include "rank.h"

int main()
{
    std::string path(".\\data");
    Base_Extract* exptr = new Base_Extract(path,5);
    exptr->_extract();
    RANK* rank = new RANK(path, exptr);

    rank->full_features();
    rank->full_dis("./data/image_0006.jpg");
    rank->sort_dis();

    system("pause");
    return 0;
}