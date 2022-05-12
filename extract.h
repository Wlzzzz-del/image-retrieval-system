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
*  @file     extract.h                                                       *
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

#ifndef EXTRACT_H
#define EXTRACT_H

#include <io.h>
#include <iostream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d.hpp>
#include <vector>
#include <string>
#include "file.h"
using namespace std;
using namespace cv;

/**
 * @brief based features extractor based on SIFT
 * 包含文件读取方法、提取路径下的所有图像特征、训练Kmeans聚类模型
 * Include func reading file's name under directionary,
 * extractor features of img under directionary,
 * traning a module of Kmeans cluster
 */
class Base_Extract
{
public:
    /** 
     * @brief some declarations of types
     */    
    typedef int              _Sizeof_Imge;
    typedef int              _Feature_Num;
    typedef string           _Path;
    typedef Mat              _Feature;
    typedef const Mat        _Feature_const;
    typedef vector<string>   _File_List;
    typedef vector<KeyPoint> _Point_List;
    typedef BOWKMeansTrainer BOWTrainer;

public:
    _Path  _data_path;
    int    cluster_num;
private:
    Ptr<SiftDescriptorExtractor> detector;
public:
    Ptr<DescriptorMatcher> matcher;
    Ptr<BOWImgDescriptorExtractor> bowde;// bag of words



public:
    /** 
     * @brief constructor
     * @param p              path of data file.
     * @param cluster_num    number of clustering center.
     *
     * initialize detector,matcher,bowde
     * 
     * @return void
     */
    Base_Extract(_Path p, int cluster_num):
    _data_path(p),cluster_num(cluster_num),
    detector(SiftDescriptorExtractor::create(20)),
    matcher(DescriptorMatcher::create("BruteForce")),
    bowde(new BOWImgDescriptorExtractor(detector, matcher))
    {}
protected:
    Base_Extract(){}
    

private:
    Base_Extract(const Base_Extract&){}
public:
    virtual ~Base_Extract(){}


protected:
//-------------------------------------------------------------------
//  get files of data dictionary
//  return: file_list
//-------------------------------------------------------------------
    _File_List get_file()
    {
        _File_List _files = getFileNames(_data_path);
        return _files;
    }

public:
//-------------------------------------------------------------------
//  extract the keypoints of files
//  return: void 
//-------------------------------------------------------------------
    virtual void _extract()
    {
        BOWTrainer trainer(cluster_num);
        
        _File_List _file = get_file();
        int len = _file.size();
        for(int i=0; i<len; i++)
        {
            _Feature img = imread(_file[i]);
            _Feature output;
            _Point_List KeyPoints;

            detector->detect(img, KeyPoints);
            detector->compute(img, KeyPoints, output);
            trainer.add(output);

            std::cout<<"get"<<_file[i]<<std::endl;
        }
        _Feature dictionary = trainer.cluster();// train module
        bowde->setVocabulary(dictionary);
    }

//-------------------------------------------------------------------
//  compute a graph in BOWImgDescriptorExtractor and normalize
//  return: void 
//-------------------------------------------------------------------
    virtual _Feature get_one(_Path path)
    {
        _Feature img = imread(path);
        _Feature output;
        _Point_List KeyPoints;
        detector->detect(img,KeyPoints);// 检测SIFT关键点
        bowde->compute(img, KeyPoints, output);// 计算特征向量
        normalize(output, output, 1.0, 0.0 ,NORM_MINMAX);// 特征向量标准化
        return output;
    }

};

#endif