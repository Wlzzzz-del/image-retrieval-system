/*****************************************************************************
*  OpenST Basic tool library                                                 *
*  Copyright (C) 2014 Henry.Wen  renhuabest@163.com.                         *
*                                                                            *
*  This file is part of OST.                                                 *
*                                                                            *
*  This program is free software; you can redistribute it and/or modify      *
*  it under the terms of the GNU General Public License version 3 as         *
*  published by the Free Software Foundation.                                *
*                                                                            *
*  You should have received a copy of the GNU General Public License         *
*  along with OST. If not, see <http://www.gnu.org/licenses/>.               *
*                                                                            *
*  Unless required by applicable law or agreed to in writing, software       *
*  distributed under the License is distributed on an "AS IS" BASIS,         *
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  *
*  See the License for the specific language governing permissions and       *
*  limitations under the License.                                            *
*                                                                            *
*  @file     rank.h                                                       *
*  @brief    feature extractor based SIFT                                                    *
*  Details.                                                                  *
*                                                                            *
*  @author   wu_lizhao                                                       *
*  @email    wu_lizhao@yeah.com                                              *
*  @version  1.0.0.1                                                         *
*  @date     2022/3/27                                                       *
*  @license  GNU General Public License (GPL)                                *
*                                                                            *
*----------------------------------------------------------------------------*
*  Remark         : Description                                              *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2022/03/27 | 1.0.0.1   | wu_lizhao      | realize function                *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/

#ifndef RANK_H
#define RANK_H
#include <iostream>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d.hpp>
#include <vector>
#include <string>
#include <map>
#include "file.h"
#include "extract.h"

using namespace std;
using namespace cv;

/**
 * @brief rank class: receive and count the distance of two img.
 * then rank top20 .
 * include count distance,sort by distance. 
 */
class RANK{
    typedef string         _Path;
    typedef Mat            _Feature;
    typedef double         _Dis;
    typedef vector<string> _File_List;
    typedef int            _Len;
    typedef pair<_Path, _Dis> _pair;

    map<_Path, _Feature>   features;
    map<_Path, _Dis>       dis;
    _Path _data_path;
    Base_Extract* _ext;
    _File_List _files;
    _Len _len;

public:
    RANK(_Path p, Base_Extract* ext):_data_path(p), _ext(ext),
    _files(get_file()),_len(_files.size()){}
    ~RANK(){}
private:
    RANK(const RANK&){}

    /** 
     * @brief calculate
     * @param input1    first img.
     * @param input2    second img. 
     *
     * calculate the distance of two img.
     * 
     * @return distance
     */
    _Dis calculate_dis(Mat input1, Mat input2)
    {
        return norm(input1, input2);
    }    

    /** 
     * @brief set distance.
     * @param path      path of img.
     * @param input     distance from target img to source img.
     *
     * set the distance into map.
     * 
     * @return void.
     */
    void set_dis(_Path path, _Dis d)
    {
        dis[path] = d;
    }

    /** 
     * @brief set features.
     * @param path      path of img.
     * @param input     distance from target img to source img.
     *
     * set the distance into map.
     * 
     * @return void.
     */
    void set_feature(_Path path, _Feature f)
    {
        features[path] = f;
    }

//-------------------------------------------------------------------
//  get files of data dictionary.
//  return: file_list
//-------------------------------------------------------------------
    _File_List get_file()
    {
        _File_List _files = getFileNames(_data_path);
        return _files;
    }

public:

//-------------------------------------------------------------------
//  set features of images from the file.
//-------------------------------------------------------------------
    void full_features()
    {
        for(int i=0; i<_len; i++)
        {
            set_feature(_files[i],_ext->get_one(_files[i]));
        }
    }

//-------------------------------------------------------------------
//  set distance between source image to 
//  target image from the file.
//-------------------------------------------------------------------
    void full_dis(_Path path)
    {
        _Feature one = _ext->get_one(path);
        map<_Path, _Feature>::iterator ite= features.begin();
        while(ite!= features.end())
        {
            double dis = calculate_dis(ite->second,one);
            set_dis(ite->first,dis);
            ite++;
        }
    }

//-------------------------------------------------------------------
//  sort images by distance.
//-------------------------------------------------------------------
    static bool cmp(_pair a, _pair b)
    {
        return a.second < b.second;
    }
    void sort_dis()
    {
        vector<_pair> vec;
        map<_Path, _Dis>::iterator ite= dis.begin();
        while(ite!= dis.end())
        {
            vec.push_back(_pair(ite->first,ite->second));
            ite++;
        }
        sort(vec.begin(),vec.end(),cmp);
        vector<_pair>::iterator it = vec.begin();
        while(it!=vec.end()){
            cout<<it->first<<endl;
            it++;
        }
    }

};


#endif