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
*  @file     hog_extract.h                                                       *
*  @brief    feature extractor based SIFT                                                    *
*  Details.                                                                  *
*                                                                            *
*  @author   wu_lizhao                                                       *
*  @email    wu_lizhao@yeah.com                                              *
*  @version  1.0.0.1                                                         *
*  @date     2022/3/28                                                       *
*  @license  GNU General Public License (GPL)                                *
*                                                                            *
*----------------------------------------------------------------------------*
*  Remark         : Description                                              *
*----------------------------------------------------------------------------*
*  Change History :                                                          *
*  <Date>     | <Version> | <Author>       | <Description>                   *
*----------------------------------------------------------------------------*
*  2022/03/28 | 1.0.0.1   | wu_lizhao      | realize function                *
*----------------------------------------------------------------------------*
*                                                                            *
*****************************************************************************/

#ifndef HOG_EXTRACT_H
#define HOG_EXTRACT_H

#include <io.h>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\objdetect.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/gapi/imgproc.hpp"
#include <vector>
#include "extract.h"
#include <string>
#include "file.h"
using namespace std;
using namespace cv;

/**
 * @brief based features extractor based on HOG
 * 包含文件读取方法、提取路径下的所有图像特征、训练Kmeans聚类模型
 * Include func reading file's name under directionary,
 * extractor features of img under directionary,
 * traning a module of Kmeans cluster
 */
class HOG_Extract:public Base_Extract
{
    private:
    Ptr<HOGDescriptor> detector;
    public:
    // call based constructor
    HOG_Extract(_Path p, int cluster_num):
    Base_Extract(p,cluster_num),
    detector(new HOGDescriptor()){}

    void _extract()
    {
        BOWTrainer trainer(cluster_num);
        
        _File_List _file = get_file();
        int len = _file.size();
        for(int i=0; i<len; i++)
        {
            // read HOG
            vector<float> ders;
            vector<Point> locs;
            _Feature img = imread(_file[i]);
            cvtColor(img,img,COLOR_BGR2GRAY);// into gray

            detector->compute(img,ders,Size(32,32),Size(0,0),locs);
            _Feature output(ders.size(), 1, CV_32FC1);

            // make into matrix
            for(int j=0;j<ders.size();j++)
            {
                output.at<float>(j,0)=ders.at(j);
            }
            trainer.add(output);
            std::cout<<"get"<<_file[i]<<std::endl;
        }
        _Feature dictionary = trainer.cluster();
        bowde->setVocabulary(dictionary);
    }
};

#endif