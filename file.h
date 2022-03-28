#ifndef FILE_H
#define FILE_H

#include <dirent.h>
#include <string.h>
#include <vector>
#include <stdio.h>
#include <iostream>

using namespace std;

vector<string> getFileNames (string dir)
{
    vector<string> file_lists;
 
    DIR *dp;
    struct dirent *ep;
    dp = opendir (dir.c_str());
 
    if (dp != NULL)
    {
        while ((ep = readdir (dp))){
            if (strcmp(ep->d_name, ".") && strcmp(ep->d_name, ".."))
                file_lists.push_back(dir + "/"+ ep->d_name);
        }
        (void) closedir (dp);
    }
    else
        perror ("Couldn't open the directory");
 
    return file_lists;
}

#endif