#include <cmath>
#include <mutex>
#include <chrono>
#include <vector>
#include <thread>
#include <iostream>
#include <chrono>
#include <tbb/tbb.h>
#include <tbb/mutex.h>
#include <tbb/parallel_do.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range2d.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/partitioner.h>

#include <cmath>
#include <cfenv>

#include <osg/Math>
#include <osg/Vec3ub>
#include <osg/Matrix>


#include <osg/Image>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>


std::ostream& operator<<(std::ostream &os, const osg::Matrix3Template<signed char> &p)
{
    return os << "Matrix = | " <<static_cast<int>(p(0,0)) <<", " <<static_cast<int>(p(0,1)) <<", " <<static_cast<int>(p(0,2))<<" |" << std::endl
              << "         | " <<static_cast<int>(p(1,0)) <<", " <<static_cast<int>(p(1,1)) <<", " <<static_cast<int>(p(1,2))<<" |" << std::endl
              << "         | " <<static_cast<int>(p(2,0)) <<", " <<static_cast<int>(p(2,1)) <<", " <<static_cast<int>(p(2,2))<<" |" << std::endl;
}

std::ostream& operator<<(std::ostream &os, const osg::Matrix3Template<unsigned char> &p)
{
    return os << "Matrix = | " <<static_cast<unsigned int>(p(0,0)) <<", " <<static_cast<unsigned int>(p(0,1)) <<", " <<static_cast<unsigned int>(p(0,2))<<" |" << std::endl
              << "         | " <<static_cast<unsigned int>(p(1,0)) <<", " <<static_cast<unsigned int>(p(1,1)) <<", " <<static_cast<unsigned int>(p(1,2))<<" |" << std::endl
              << "         | " <<static_cast<unsigned int>(p(2,0)) <<", " <<static_cast<unsigned int>(p(2,1)) <<", " <<static_cast<unsigned int>(p(2,2))<<" |" << std::endl;
}

std::ostream& operator<<(std::ostream &os, const osg::Matrix3Template<signed short> &p)
{
    return os << "Matrix = | " <<static_cast<int16_t>(p(0,0)) <<", " <<static_cast<int16_t>(p(0,1)) <<", " <<static_cast<int16_t>(p(0,2))<<" |" << std::endl
              << "         | " <<static_cast<int16_t>(p(1,0)) <<", " <<static_cast<int16_t>(p(1,1)) <<", " <<static_cast<int16_t>(p(1,2))<<" |" << std::endl
              << "         | " <<static_cast<int16_t>(p(2,0)) <<", " <<static_cast<int16_t>(p(2,1)) <<", " <<static_cast<int16_t>(p(2,2))<<" |" << std::endl;
}

int getIndexFromCurrentColomnAndRow(std::size_t ccol, std::size_t crow, osg::Image* img){

    int idx = ccol + crow * img->s();
    return idx;
}

unsigned char getRedValue(osg::Image* img, int idx)
{
    unsigned char* imgData = img->data();
    int index = idx * 3;
    return  imgData[index ];
}

unsigned char getGreenValue(osg::Image* img, int idx)
{
    unsigned char* imgData = img->data();
    int index = idx * 3;
    return  imgData[index + 1];
}

unsigned char getBlueValue(osg::Image* img, int idx)
{
    unsigned char* imgData = img->data();
    int index = idx * 3;
    return  imgData[index + 2];
}



// get part of image corresponding to the kernel created
template <class T, unsigned int RowN, unsigned int ColN>
osg::MatrixTemplate<T, RowN, ColN> getImageRedMatrixByKernelDim(osg::Image* img, int curr_index){
    osg::MatrixTemplate<T, RowN, ColN> mat;

    std::div_t result =  std::div( curr_index, img->s());
    int ccol = result.rem;
    int crow = result.quot;



    int lower_row =  -((int)RowN/2);
    int lower_col = -((int)ColN/2);
    int upper_row = RowN/2;
    int uppter_col = ColN/2;

    for (int r = lower_row; r <= upper_row; ++r)
        for (int c = lower_col; c <= uppter_col; ++c)
        {
            int i = ccol + c;
            int j = crow + r;

            int newIndx = getIndexFromCurrentColomnAndRow(i, j, img);
            T val = static_cast<T>(getRedValue(img, newIndx));
            mat(r+RowN/2, c+ColN/2) = val;

        }
    return mat;
}

template <class T, unsigned int RowN, unsigned int ColN>
osg::MatrixTemplate<T, RowN, ColN> getImageGreenMatrixByKernelDim(osg::Image* img, int curr_index){
    osg::MatrixTemplate<T, RowN, ColN> mat;

    std::div_t result =  std::div( curr_index, img->s());
    int ccol = result.rem;
    int crow = result.quot;



    int lower_row =  -((int)RowN/2);
    int lower_col = -((int)ColN/2);
    int upper_row = RowN/2;
    int uppter_col = ColN/2;

    for (int r = lower_row; r <= upper_row; ++r)
        for (int c = lower_col; c <= uppter_col; ++c)
        {
            int i = ccol + c;
            int j = crow + r;

            int newIndx = getIndexFromCurrentColomnAndRow(i, j, img);
            T val = static_cast<T>(getGreenValue(img, newIndx));
            mat(r+RowN/2, c+ColN/2) = val;

        }
    return mat;
}

template <class T, unsigned int RowN, unsigned int ColN>
osg::MatrixTemplate<T, RowN, ColN> getImageBlueMatrixByKernelDim(osg::Image* img, int curr_index){
    osg::MatrixTemplate<T, RowN, ColN> mat;

    std::div_t result =  std::div( curr_index, img->s());
    int ccol = result.rem;
    int crow = result.quot;



    int lower_row =  -((int)RowN/2);
    int lower_col = -((int)ColN/2);
    int upper_row = RowN/2;
    int uppter_col = ColN/2;

    for (int r = lower_row; r <= upper_row; ++r)
        for (int c = lower_col; c <= uppter_col; ++c)
        {
            int i = ccol + c;
            int j = crow + r;

            int newIndx = getIndexFromCurrentColomnAndRow(i, j, img);
            T val = static_cast<T>(getBlueValue(img, newIndx));
            mat(r+RowN/2, c+ColN/2) = val;

        }
    return mat;
}




template <class T, unsigned int RowN, unsigned int ColN>
osg::MatrixTemplate<T, RowN, ColN> createMeanKernel(){
    osg::MatrixTemplate<T, RowN, ColN> mat;

    for (std::size_t r = 0; r < RowN; ++r)
        for (std::size_t c = 0; c < ColN; ++c)
        {
            mat(r,c) = static_cast<T>(1.f/(RowN*ColN));
        }

    return mat;
}

template <class T>
osg::MatrixTemplate<T, 3, 3> createShapening3x3Kernel(){
    osg::MatrixTemplate<T, 3, 3> mat;


    mat(0, 0) =  0;
    mat(0, 1) = -1;
    mat(0, 2) =  0;

    mat(1, 0) = -1;
    mat(1, 1) =  5;
    mat(1, 2) = -1;

    mat(2 , 0) =  0;
    mat(2 , 1) = -1;
    mat(2 , 2) =  0;
    return mat;
}

template <class T>
osg::MatrixTemplate<T, 5, 5> createShapening5x5Kernel(){
    osg::MatrixTemplate<T, 5, 5> mat;

    mat(0, 0) =  -1;
    mat(0, 1) =  -1;
    mat(0, 2) =  -1;
    mat(0, 3) =  -1;
    mat(0, 4) =  -1;


    mat(1, 0) = -1;
    mat(1, 1) =  1;
    mat(1, 2) =  1;
    mat(1, 3) =  1;
    mat(1, 4) = -1;


    mat(2, 0) = -1;
    mat(2, 1) =  1;
    mat(2, 2) =  8;
    mat(2, 3) =  1;
    mat(2, 4) = -1;


    mat(3, 0) = -1;
    mat(3, 1) =  1;
    mat(3, 2) =  1;
    mat(3, 3) =  1;
    mat(3, 4) = -1;

    mat(4, 0) = -1;
    mat(4, 1) = -1;
    mat(4, 2) = -1;
    mat(4, 3) = -1;
    mat(4, 4) = -1;


    return mat;
}

template <class T, unsigned int RowN, unsigned int ColN>
osg::MatrixTemplate<T, RowN, ColN> multiplyKernel(osg::MatrixTemplate<T, RowN, ColN> k1, osg::MatrixTemplate<T, RowN, ColN> k2){
    osg::MatrixTemplate<T, RowN, ColN> mat;

    for (std::size_t r = 0; r < RowN; ++r)
        for (std::size_t c = 0; c < ColN; ++c)
        {
            mat(r,c) = k1(r, c)*k2(r,c);
        }

    return mat;
}

template <class T, unsigned int RowN, unsigned int ColN>
osg::MatrixTemplate<T, RowN, ColN> multiplyKernel(osg::MatrixTemplate<T, RowN, ColN> k, T val){
    osg::MatrixTemplate<T, RowN, ColN> mat;

    for (std::size_t r = 0; r < RowN; ++r)
        for (std::size_t c = 0; c < ColN; ++c)
        {
            mat(r,c) = k(r, c)*val;
        }

    return mat;
}


template <class T>
T adjustPixelValueToUnSignedByte(T val, float divisor = 1.0f){

    T retVal = 0;

            if (val < 0)
                retVal = 0;
            else if (val > 255)
                retVal = 255;
            else
                retVal = val;

    return retVal/divisor;
}

template <class T, unsigned int RowN, unsigned int ColN>
T sumKernelElements(osg::MatrixTemplate<T, RowN, ColN> mat){
    T sumValues = 0;
    for (std::size_t r = 0; r < RowN; ++r)
        for (std::size_t c = 0; c < ColN; ++c)
        {
            sumValues += mat(r,c);
        }

    return sumValues;
}

template <class T, unsigned int RowN, unsigned int ColN>
osg::MatrixTemplate<T, RowN, ColN> createGaussianKernel(){
    osg::MatrixTemplate<T, RowN, ColN> mat;

    int lower_row =  -((int)RowN/2);
    int lower_col = -((int)ColN/2);
    int upper_row = RowN/2;
    int uppter_col = ColN/2;
    int rad_2 = RowN/2;
    int rad = RowN;
    float  stdv = 5.0;
    float  s    = 2.0 * stdv * stdv;  // Assigning standard deviation to 1.0

    for (int j = lower_row; j <= upper_row; ++j)
        for (int i = lower_col; i <= uppter_col; ++i)
        {

            float radius = std::sqrt(i*i + j*j);
            float expval = (std::exp(-(radius*radius)/s))/(M_PI * s);
            mat(j+rad_2, i+rad_2) = static_cast<T>(expval);
            //int radius = std::sqrt((std::abs(i)-rad_2)*(std::abs(i)-rad_2) + (std::abs(j)-rad_2)*(std::abs(j)-rad_2)  );
            //mat(j+rad_2, i+rad_2) = static_cast<T>(radius);

            //if  ( (j==0) && (i==0) )
            //    mat(j+rad_2, i+rad_2) = mat(j+rad_2, i+rad_2)+1;//static_cast<T>(rad_2+1);
        }

    mat = multiplyKernel<T, RowN, ColN>(mat, 1/sumKernelElements<T, RowN, ColN>(mat));
    return mat;

 }


const std::string imgsPath("/home/mselim/Development/Projects/images/");
const std::string srcImagePath(imgsPath+"input.jpg");
const std::string outImagepath(imgsPath+"processed.jpg");

int main() {
    // init the tbb task scheduler, very important
    tbb::task_scheduler_init init();

    // read the source image
    osg::ref_ptr<osg::Image> img =  osgDB::readImageFile(srcImagePath);
    // creating the output image
    // with source image exact dimension
    osg::ref_ptr<osg::Image> imgNew = new osg::Image();
    imgNew->allocateImage(img->s(), img->t(), 1, GL_RGB, GL_UNSIGNED_BYTE);

    std::memset(imgNew->data(), 0, imgNew->s() * imgNew->t()*3);


    auto apply_parallel_kernel_sharpen = [](osg::ref_ptr<osg::Image>& srcImage, osg::ref_ptr<osg::Image>& newImage) {
        //
        const auto start = std::chrono::high_resolution_clock::now();

        const static std::size_t RowN = 3;
        const static std::size_t ColN = 3;
        osg::MatrixTemplate<float, RowN, ColN> kernel;

        unsigned char* imgData = srcImage->data();
        unsigned char* imgNewData = newImage->data();
        std::size_t rows = srcImage->t();
        std::size_t cols = srcImage->s();
        std::size_t r = RowN;
        std::size_t r_2 = RowN/2;

        kernel = createShapening3x3Kernel<float>();

        tbb::parallel_for(tbb::blocked_range2d<std::size_t, std::size_t>(0, srcImage->t(), 0, srcImage->s()),
                          [&](const tbb::blocked_range2d<std::size_t, std::size_t>& r) {
            for( std::size_t crow=r.rows().begin(); crow!=r.rows().end(); ++crow )
            {
                for( std::size_t ccol=r.cols().begin(); ccol!=r.cols().end(); ++ccol )
                {
                    int idx = ccol + crow*cols;

                    int lower_col_check = (int)(ccol - r_2);
                    int lower_row_check = (int)(crow - r_2);

                    int upper_col_check = (int)(ccol + r_2);
                    int upper_row_check = (int)(crow + r_2);

                    if ((lower_col_check >= 0) && (lower_row_check >= 0) && (upper_col_check < cols ) && (upper_row_check < rows )  ){

                        osg::MatrixTemplate<std::float_t, RowN, ColN> cur_red_mat, cur_green_mat, cur_blue_mat;
                        osg::MatrixTemplate<std::float_t, RowN, ColN> res_red_mat, res_green_mat, res_blue_mat;

                        cur_red_mat   = getImageRedMatrixByKernelDim  <float, RowN, ColN>(srcImage, idx);
                        cur_green_mat = getImageGreenMatrixByKernelDim<float, RowN, ColN>(srcImage, idx);
                        cur_blue_mat  = getImageBlueMatrixByKernelDim <float, RowN, ColN>(srcImage, idx);

                        res_red_mat   = multiplyKernel<float, RowN, ColN>(kernel, cur_red_mat);
                        res_green_mat = multiplyKernel<float, RowN, ColN>(kernel, cur_green_mat);
                        res_blue_mat  = multiplyKernel<float, RowN, ColN>(kernel, cur_blue_mat);

                        float red_sum   = sumKernelElements<float, RowN, ColN>(res_red_mat);
                        float green_sum = sumKernelElements<float, RowN, ColN>(res_green_mat);
                        float blue_sum  = sumKernelElements<float, RowN, ColN>(res_blue_mat);

                        int index = idx*3;

                        imgNewData[index]   = static_cast<unsigned char>(adjustPixelValueToUnSignedByte<float>(red_sum));
                        imgNewData[index+1] = static_cast<unsigned char>(adjustPixelValueToUnSignedByte<float>(green_sum));
                        imgNewData[index+2] = static_cast<unsigned char>(adjustPixelValueToUnSignedByte<float>(blue_sum));
                    }
                }
            }
        }, tbb::auto_partitioner());

        const auto finish = std::chrono::high_resolution_clock::now();
        std::cout << "Parallel Sharpen kernel took = " <<
                     std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
                  << " micro seconds" << std::endl;
    };




    auto apply_kernel_sharpen = [](osg::ref_ptr<osg::Image>& srcImage, osg::ref_ptr<osg::Image>& newImage) {
        //
        const auto start = std::chrono::high_resolution_clock::now();

        const static std::size_t RowN = 3;
        const static std::size_t ColN = 3;
        osg::MatrixTemplate<float, RowN, ColN> kernel;
        osg::MatrixTemplate<std::float_t, RowN, ColN> cur_red_mat, cur_green_mat, cur_blue_mat;
        osg::MatrixTemplate<std::float_t, RowN, ColN> res_red_mat, res_green_mat, res_blue_mat;

        unsigned char* imgData = srcImage->data();
        unsigned char* imgNewData = newImage->data();
        std::size_t rows = srcImage->t();
        std::size_t cols = srcImage->s();
        std::size_t r = RowN;
        std::size_t r_2 = RowN/2;

        kernel = createShapening3x3Kernel<float>();
        //kernel = createShapening5x5Kernel<float>();

        for( std::size_t crow = 0; crow < rows; ++crow ){
            for( std::size_t ccol = 0; ccol < cols; ++ccol ) {
                int idx = ccol + crow*cols;

                int lower_col_check = (int)(ccol - r_2);
                int lower_row_check = (int)(crow - r_2);

                int upper_col_check = (int)(ccol + r_2);
                int upper_row_check = (int)(crow + r_2);

                if ((lower_col_check >= 0) && (lower_row_check >= 0) && (upper_col_check < cols ) && (upper_row_check < rows )  ){
                    cur_red_mat   = getImageRedMatrixByKernelDim  <float, RowN, ColN>(srcImage, idx);
                    cur_green_mat = getImageGreenMatrixByKernelDim<float, RowN, ColN>(srcImage, idx);
                    cur_blue_mat  = getImageBlueMatrixByKernelDim <float, RowN, ColN>(srcImage, idx);

                    res_red_mat   = multiplyKernel<float, RowN, ColN>(kernel, cur_red_mat);
                    res_green_mat = multiplyKernel<float, RowN, ColN>(kernel, cur_green_mat);
                    res_blue_mat  = multiplyKernel<float, RowN, ColN>(kernel, cur_blue_mat);

                    float red_sum   = sumKernelElements<float, RowN, ColN>(res_red_mat);
                    float green_sum = sumKernelElements<float, RowN, ColN>(res_green_mat);
                    float blue_sum  = sumKernelElements<float, RowN, ColN>(res_blue_mat);

                    int index = idx*3;

                    imgNewData[index]   = static_cast<unsigned char>(adjustPixelValueToUnSignedByte<float>(red_sum));
                    imgNewData[index+1] = static_cast<unsigned char>(adjustPixelValueToUnSignedByte<float>(green_sum));
                    imgNewData[index+2] = static_cast<unsigned char>(adjustPixelValueToUnSignedByte<float>(blue_sum));


                }
            }
        }

        const auto finish = std::chrono::high_resolution_clock::now();
        std::cout << "Serial sharpening kernel took = " <<
                     std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
                  << " micro seconds" << std::endl;
    };


    auto apply_parallel_kernel_blur = [](osg::ref_ptr<osg::Image>& srcImage, osg::ref_ptr<osg::Image>& newImage) {
        //
        const auto start = std::chrono::high_resolution_clock::now();

        const static std::size_t RowN = 3;
        const static std::size_t ColN = 3;
        osg::MatrixTemplate<float, RowN, ColN> kernel;


        unsigned char* imgData    = srcImage->data();
        unsigned char* imgNewData = newImage->data();
        std::size_t rows = srcImage->t();
        std::size_t cols = srcImage->s();
        std::size_t r = RowN;
        std::size_t r_2 = RowN/2;
        //kernel = createMeanKernel<float, RowN, ColN>();
        kernel = createGaussianKernel<float, RowN, ColN>();

        tbb::parallel_for(tbb::blocked_range2d<std::size_t, std::size_t>(0, srcImage->t(), 0, srcImage->s()),
                          [&](const tbb::blocked_range2d<std::size_t, std::size_t>& r) {
            for( std::size_t crow=r.rows().begin(); crow!=r.rows().end(); ++crow )
            {
                for( std::size_t ccol=r.cols().begin(); ccol!=r.cols().end(); ++ccol )
                {
                    osg::MatrixTemplate<std::float_t, RowN, ColN> cur_red_mat, cur_green_mat, cur_blue_mat;
                    osg::MatrixTemplate<std::float_t, RowN, ColN> res_red_mat, res_green_mat, res_blue_mat;

                    int idx = ccol + crow*cols;

                    int lower_col_check = (int)(ccol - r_2);
                    int lower_row_check = (int)(crow - r_2);

                    int upper_col_check = (int)(ccol + r_2);
                    int upper_row_check = (int)(crow + r_2);

                    if ((lower_col_check >= 0) && (lower_row_check >= 0) && (upper_col_check < cols ) && (upper_row_check < rows ) ){
                        cur_red_mat   = getImageRedMatrixByKernelDim  <float, RowN, ColN>(srcImage, idx);
                        cur_green_mat = getImageGreenMatrixByKernelDim<float, RowN, ColN>(srcImage, idx);
                        cur_blue_mat  = getImageBlueMatrixByKernelDim <float, RowN, ColN>(srcImage, idx);

                        res_red_mat   = multiplyKernel<float, RowN, ColN>(kernel, cur_red_mat);
                        res_green_mat = multiplyKernel<float, RowN, ColN>(kernel, cur_green_mat);
                        res_blue_mat  = multiplyKernel<float, RowN, ColN>(kernel, cur_blue_mat);

                        float red_sum   = sumKernelElements<float, RowN, ColN>(res_red_mat);
                        float green_sum = sumKernelElements<float, RowN, ColN>(res_green_mat);
                        float blue_sum  = sumKernelElements<float, RowN, ColN>(res_blue_mat);

                        int index = idx*3;

                        imgNewData[index]   = static_cast<unsigned char>((red_sum));
                        imgNewData[index+1] = static_cast<unsigned char>((green_sum));
                        imgNewData[index+2] = static_cast<unsigned char>((blue_sum));

                    }
                }
            }
        }, tbb::auto_partitioner());

        const auto finish = std::chrono::high_resolution_clock::now();
        std::cout << "Parallel blur kernel took = "
                  << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
                  << " micro seconds" << std::endl;
    };



    auto apply_kernel_blur = [](osg::ref_ptr<osg::Image>& srcImage, osg::ref_ptr<osg::Image>& newImage) {
        //
        const auto start = std::chrono::high_resolution_clock::now();

        const static std::size_t RowN = 7;
        const static std::size_t ColN = 7;
        osg::MatrixTemplate<float, RowN, ColN> kernel;
        osg::MatrixTemplate<std::float_t, RowN, ColN> cur_red_mat, cur_green_mat, cur_blue_mat;
        osg::MatrixTemplate<std::float_t, RowN, ColN> res_red_mat, res_green_mat, res_blue_mat;

        unsigned char* imgData = srcImage->data();
        unsigned char* imgNewData = newImage->data();
        std::size_t rows = srcImage->t();
        std::size_t cols = srcImage->s();
        std::size_t r = RowN;
        std::size_t r_2 = RowN/2;
        //kernel = createMeanKernel<float, RowN, ColN>();
        kernel = createGaussianKernel<float, RowN, ColN>();

        for( std::size_t crow = 0; crow < rows; ++crow ){
            for( std::size_t ccol = 0; ccol < cols; ++ccol ) {
                int idx = ccol + crow*cols;

                int lower_col_check = (int)(ccol - r_2);
                int lower_row_check = (int)(crow - r_2);

                int upper_col_check = (int)(ccol + r_2);
                int upper_row_check = (int)(crow + r_2);


                if ((lower_col_check >= 0) && (lower_row_check >= 0) && (upper_col_check < cols ) && (upper_row_check < rows )  ){
                    cur_red_mat   = getImageRedMatrixByKernelDim  <float, RowN, ColN>(srcImage, idx);
                    cur_green_mat = getImageGreenMatrixByKernelDim<float, RowN, ColN>(srcImage, idx);
                    cur_blue_mat  = getImageBlueMatrixByKernelDim <float, RowN, ColN>(srcImage, idx);

                    res_red_mat   = multiplyKernel<float, RowN, ColN>(kernel, cur_red_mat);
                    res_green_mat = multiplyKernel<float, RowN, ColN>(kernel, cur_green_mat);
                    res_blue_mat  = multiplyKernel<float, RowN, ColN>(kernel, cur_blue_mat);

                    float red_sum   = sumKernelElements<float, RowN, ColN>(res_red_mat);
                    float green_sum = sumKernelElements<float, RowN, ColN>(res_green_mat);
                    float blue_sum  = sumKernelElements<float, RowN, ColN>(res_blue_mat);

                    int index = idx*3;

                    imgNewData[index]   = static_cast<unsigned char>(red_sum);
                    imgNewData[index+1] = static_cast<unsigned char>(green_sum);
                    imgNewData[index+2] = static_cast<unsigned char>(blue_sum);


                }
            }
        }

        const auto finish = std::chrono::high_resolution_clock::now();
        std::cout << "Serial blur  kernel took = " <<
                     std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
                  << " micro seconds" << std::endl;
    };



    //apply_kernel_blur(img, imgNew);
    //apply_kernel_sharpen(img, imgNew);
    apply_parallel_kernel_blur(img, imgNew);
    //apply_parallel_kernel_sharpen(img, imgNew);



    auto serial_convert_image_grayscale = [](osg::ref_ptr<osg::Image>& srcImg, osg::ref_ptr<osg::Image>& newImg) {
        //
        const auto start = std::chrono::high_resolution_clock::now();

        unsigned char* imgData = srcImg->data();
        unsigned char* imgNewData = newImg->data();
        std::size_t rows = srcImg->t();
        std::size_t cols = srcImg->s();
        for( std::size_t crow = 0; crow < rows; ++crow ){
            for( std::size_t ccol = 0; ccol < cols; ++ccol ) {
                int idx = ccol+crow*cols;
                int index = idx * 3;

                unsigned char R = imgData[index ];
                unsigned char G = imgData[index + 1];
                unsigned char B = imgData[index + 2];

                //gray scale value = 0.21 * R + 0.72 * G + 0.07 * B
                unsigned char gray =(unsigned char) (0.21f*R + 0.72f*G + 0.07f*B);

                imgNewData[index]   = gray;
                imgNewData[index+1] = gray;
                imgNewData[index+2] = gray;
            }
        }

        const auto finish = std::chrono::high_resolution_clock::now();
        std::cout << "Serial conversion took = " <<
                     std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
                  << " micro seconds" << std::endl;
    };


    // parallel code that converts the image using parallel code
    auto parallel_convert_image_grayscale = [](osg::ref_ptr<osg::Image>& srcImage, osg::ref_ptr<osg::Image>& newImage) {
        {
            const auto start = std::chrono::high_resolution_clock::now();
            unsigned char* imgData = srcImage->data();
            unsigned char* imgNewData = newImage->data();

            tbb::parallel_for(tbb::blocked_range2d<std::size_t, std::size_t>(0, srcImage->t(), 0, srcImage->s()),
                              [&](const tbb::blocked_range2d<std::size_t, std::size_t>& r) {
                for( std::size_t crow=r.rows().begin(); crow!=r.rows().end(); ++crow )
                {
                    for( std::size_t ccol=r.cols().begin(); ccol!=r.cols().end(); ++ccol )
                    {
                        int idx = ccol + crow* srcImage->s();
                        int index = idx * 3;

                        unsigned char R = imgData[index ];
                        unsigned char G = imgData[index + 1];
                        unsigned char B = imgData[index + 2];

                        unsigned char gray = (unsigned char) (0.21f*R + 0.72f*G + 0.07f*B);

                        imgNewData[index]   = gray;
                        imgNewData[index+1] = gray;
                        imgNewData[index+2] = gray;

                    }
                }

            }, tbb::auto_partitioner());

            const auto finish = std::chrono::high_resolution_clock::now();
            std::cout << "Parallel conversion took = "
                      << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
                      << " micro seconds" << std::endl;
        }
    };


    // parallel code that converts the image using parallel code
    auto parallel_convert_image_blur = [](osg::ref_ptr<osg::Image>& srcImage, osg::ref_ptr<osg::Image>& newImage) {
        {
            const auto start = std::chrono::high_resolution_clock::now();
            unsigned char* imgData = srcImage->data();
            unsigned char* imgNewData = newImage->data();
            using matkernel = osg::Matrix3Template<uint8_t>;
            using smatkernel = osg::Matrix3Template<int16_t>;
            smatkernel kernel;
            //smatkernel kernel_r, kernel_g, kernel_b;
            kernel.reset();
            kernel(0,0) = static_cast< int16_t >(1); //1
            kernel(0,1) = static_cast< int16_t >(0); //2
            kernel(0,2) = static_cast< int16_t >(0); //1
            kernel(1,0) = static_cast< int16_t >(0); //2
            kernel(1,1) = static_cast< int16_t >(1); //4
            kernel(1,2) = static_cast< int16_t >(0); //2
            kernel(2,0) = static_cast< int16_t >(0); //1
            kernel(2,1) = static_cast< int16_t >(0); //2
            kernel(2,2) = static_cast< int16_t >(1); //1

            int sumK = kernel(0,0) + kernel(0,1)+ kernel(0,2)+ kernel(1,0)+ kernel(1,1)+ kernel(1,2)+ kernel(2,0)+ kernel(2,1)+ kernel(2,2);
            auto run_kernel_by_index = [&kernel, imgData, &sumK](int cur_idx, std::size_t currCol, std::size_t currRow, std::size_t cols, std::size_t rows) ->osg::Vec3ub
            {
              osg::Vec3ub res;
              osg::Vec3s  res2;
              smatkernel kernel_r, kernel_g, kernel_b;
              int idx     = cur_idx; //idx_11
              int idx_red = (idx*3);
              int idx_grn = (idx*3) + 1;
              int idx_blu = (idx*3) + 2;

              if ( (currCol < (cols -1)) && (currRow < (rows -1)) && (currCol > 0) && (currRow > 0) /*false*/)
              {
                  int idx_row_m_1 = cur_idx - cols;//currCol + (currRow - 1) * rows;
                  int idx_row_p_1 = cur_idx + cols;//currCol + (currRow + 1) * rows;


                  int idx_00 = idx_row_m_1 - 1;
                  int idx_01 = idx_row_m_1;
                  int idx_02 = idx_row_m_1 + 1;

                  int idx_10 = idx - 1;
                  int idx_11 = idx;
                  int idx_12 = idx + 1;

                  int idx_20 = idx_row_p_1 - 1;
                  int idx_21 = idx_row_p_1;
                  int idx_22 = idx_row_p_1 + 1;


                  kernel_r(0,0) = imgData[3 * idx_00];
                  kernel_r(0,1) = imgData[3 * idx_01];
                  kernel_r(0,2) = imgData[3 * idx_02];

                  kernel_r(1,0) = imgData[3 * idx_10];
                  kernel_r(1,1) = imgData[3 * idx_11];
                  kernel_r(1,2) = imgData[3 * idx_12];

                  kernel_r(2,0) = imgData[3 * idx_20];
                  kernel_r(2,1) = imgData[3 * idx_21];
                  kernel_r(2,2) = imgData[3 * idx_22];


                  kernel_g(0,0) = imgData[3 * idx_00+1];
                  kernel_g(0,1) = imgData[3 * idx_01+1];
                  kernel_g(0,2) = imgData[3 * idx_02+1];

                  kernel_g(1,0) = imgData[3 * idx_10+1];
                  kernel_g(1,1) = imgData[3 * idx_11+1];
                  kernel_g(1,2) = imgData[3 * idx_12+1];

                  kernel_g(2,0) = imgData[3 * idx_20+1];
                  kernel_g(2,1) = imgData[3 * idx_21+1];
                  kernel_g(2,2) = imgData[3 * idx_22+1];



                  kernel_b(0,0) = imgData[3 * idx_00+2];
                  kernel_b(0,1) = imgData[3 * idx_01+2];
                  kernel_b(0,2) = imgData[3 * idx_02+2];

                  kernel_b(1,0) = imgData[3 * idx_10+2];
                  kernel_b(1,1) = imgData[3 * idx_11+2];
                  kernel_b(1,2) = imgData[3 * idx_12+2];

                  kernel_b(2,0) = imgData[3 * idx_20+2];
                  kernel_b(2,1) = imgData[3 * idx_21+2];
                  kernel_b(2,2) = imgData[3 * idx_22+2];

                  res2[0] =(int16_t)( (kernel_r(0,0)*kernel(0,0) +
                                       kernel_r(0,1)*kernel(0,1) +
                                       kernel_r(0,2)*kernel(0,2) +

                                       kernel_r(1,0)*kernel(1,0) +
                                       kernel_r(1,1)*kernel(1,1) +
                                       kernel_r(1,2)*kernel(1,2) +

                                       kernel_r(2,0)*kernel(2,0) +
                                       kernel_r(2,1)*kernel(2,1) +
                                       kernel_r(2,2)*kernel(2,2) ) / sumK);


                  res2[1] =(int16_t)( (kernel_g(0,0)*kernel(0,0) +
                                       kernel_g(0,1)*kernel(0,1) +
                                       kernel_g(0,2)*kernel(0,2) +

                                       kernel_g(1,0)*kernel(1,0) +
                                       kernel_g(1,1)*kernel(1,1) +
                                       kernel_g(1,2)*kernel(1,2) +

                                       kernel_g(2,0)*kernel(2,0) +
                                       kernel_g(2,1)*kernel(2,1) +
                                       kernel_g(2,2)*kernel(2,2) ) / sumK);


                  res2[2] =(int16_t)( (kernel_b(0,0)*kernel(0,0) +
                                       kernel_b(0,1)*kernel(0,1) +
                                       kernel_b(0,2)*kernel(0,2) +

                                       kernel_b(1,0)*kernel(1,0) +
                                       kernel_b(1,1)*kernel(1,1) +
                                       kernel_b(1,2)*kernel(1,2) +

                                       kernel_b(2,0)*kernel(2,0) +
                                       kernel_b(2,1)*kernel(2,1) +
                                       kernel_b(2,2)*kernel(2,2) ) / sumK);


                  res[0] =  (uint8_t)(res2[0]);
                  res[1] =  (uint8_t)(res2[1]);
                  res[2] =  (uint8_t)(res2[2]);

              }
              else
              {
                  res = osg::Vec3ub(imgData[idx_red], imgData[idx_grn], imgData[idx_blu]);
              }
              return res;
            };


            std::cout << kernel;
            const std::size_t totalpixels = srcImage->t()*srcImage->s();
            tbb::parallel_for(tbb::blocked_range2d<std::size_t, std::size_t>(0, srcImage->t(), 0, srcImage->s()),
                              [&](const tbb::blocked_range2d<std::size_t, std::size_t>& r) {
                for( std::size_t crow=r.rows().begin(); crow!=r.rows().end(); ++crow )
                {
                    for( std::size_t ccol=r.cols().begin(); ccol!=r.cols().end(); ++ccol )
                    {
                        int idx   = ccol +  crow * srcImage->s();

                        int index = idx * 3;

                        osg::Vec3ub median_val = run_kernel_by_index(idx, ccol, crow, srcImage->s(), srcImage->t());

                        imgNewData[index]   = median_val[0];
                        imgNewData[index+1] = median_val[1];
                        imgNewData[index+2] = median_val[2];
                    }
                }

            }, tbb::auto_partitioner());

            const auto finish = std::chrono::high_resolution_clock::now();
            std::cout << "Parallel conversion took = "
                      << std::chrono::duration_cast<std::chrono::microseconds>(finish - start).count()
                      << " micro seconds" << std::endl;
        }
    };


    //serial_convert_image_grayscale(img, imgNew);
    //parallel_convert_image_grayscale(img, imgNew);
    //parallel_convert_image_blur(img, imgNew);
    // setting file path before writing to the file system
    imgNew->setFileName(outImagepath);
    // writing the new image file to file system
    bool res = osgDB::writeImageFile(*imgNew, outImagepath);
    // print to the console the new image attributes
    std::cout  << std::boolalpha << "result  : " << res << std::endl;
    std::cout  << std::boolalpha << "valid  :  " << imgNew->valid() << std::endl;
    std::cout << "image width  : " << imgNew->s() << std::endl;
    std::cout << "image height : " << imgNew->t() << std::endl;
    std::cout << "image name   : " << imgNew->getFileName() << std::endl;
    std::cout << "image size   : " << imgNew->getTotalSizeInBytes() << std::endl;

    return 0;
}
