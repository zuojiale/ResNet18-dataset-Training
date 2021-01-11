#include "HrgTofApi.h"
#include <iostream>
#include <stdio.h>
#include <helper.h>

//#define PCL_VISUALIZER
#define OPENCV_VISUALIZER

#ifdef OPENCV_VISUALIZER
#include <opencv/cv.hpp>
#include <opencv2/opencv.hpp>
#endif

#ifdef PCL_VISUALIZER
#include <pcl/common/transforms.h>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#endif

#ifdef OPENCV_VISUALIZER
using namespace std;

void onMouse_depth(int event, int x, int y, int flags, void *param)
{
    cv::Mat *im = reinterpret_cast<cv::Mat*>(param);
    std::cout << "depth(" << x << "," << y << ") :" << im->at<float>(cv::Point(x, y))<< "m" <<std::endl;
}

void onMouse_amplitude(int event, int x, int y, int flags, void *param)
{
    if(event == cv::EVENT_RBUTTONDOWN)
    {
        char filename[200]; 
        cv::Mat *im = reinterpret_cast<cv::Mat*>(param);
        cv::Size ResImgSiz = cv::Size(32, 32);
        cv::resize(*im,*im, ResImgSiz);
        static int hand =0;
        static int ss=0;
        static int pic_num =1;
        sprintf(filename, "/home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic/Test_Pics/%d_0000%d.jpeg",hand ,ss);
        imwrite(filename, *im);
        cout << hand <<" is saved with " <<pic_num<< endl; 
        ss++;
        pic_num++;
        if(ss % 100 == 0)
        {
            sprintf(filename, "/home/zjl/handgesture/1_OpencvTof/samples/openCV_savePic/Test_Pics/%d_0000%d.jpeg",hand,ss);
            imwrite(filename, *im);
            cout << hand <<" is saved with " <<pic_num<< endl; 
            hand++;
            ss++;
            pic_num = 1;
        }
    }
	
}
#endif


//定义输出格式为一个array数组，维护的是一个有10个类型为AlgorithmOutput_F32的类对象
//该类对象里面存的是depth和振幅alplitim
typedef std::array<AlgorithmOutput_F32, 10> AlgorithmOutputArray;

int main()
{
    //1.连接相机
    Hrg_LogConfig(HRG_LOG_LEVEL_INFO);

    //2.新建一个hrg相机的类对象，并给定连接参数
    Hrg_Dev_Info dev;
    dev.type = Dev_Eth;
    dev.Info.eth.addr = "192.168.1.6";//ip，需要根据连接模式更改
    dev.Info.eth.port = 8567; //端口
    dev.frameReady = NULL; //callback function

    //3.hrg相机的句柄
    Hrg_Dev_Handle handle;

    //4.传入device，输出handle指针，准备打开相机
        //0：成功  -1：失败
    if(0 != Hrg_OpenDevice(&dev, &handle))
    {
        printf("open device failed!\n");
        return -1;
    }

        //Hrg_SetRangeMode(&handle, Mode_Range_S);

        //Hrg_SetDepthRange(&handle, 0, 5000);

    //5，连接相机之后，获取数据流stream
    Hrg_StartStream(&handle);

    //6.定义输出数据流数组output_data
    AlgorithmOutputArray output_data;
    int output_idx = 0;
    Hrg_Frame frame;  
        //深度相机分辨率是240×288，所以存到一个这样的数组里面
    uint8_t* depth_rgb = new uint8_t[IMAGE_HEIGHT*IMAGE_WIDTH*3];
    uint8_t* dst_ir = new uint8_t[IMAGE_HEIGHT*IMAGE_WIDTH];
    float* pcl = new float[IMAGE_HEIGHT*IMAGE_WIDTH*3];

#ifdef PCL_VISUALIZER
    /****** pcl viewer******/
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addCoordinateSystem (0.3, 0.3, 0.3, 0.3);
    viewer->initCameraParameters();
    float theta = M_PI; // The angle of rotation in radians
    transform (0,0) = cos (theta);
    transform (0,1) = -sin(theta);
    transform (1,0) = sin (theta);
    transform (1,1) = cos (theta);
    pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
#endif

    //7.准备显示，死循环
    while(true)
    {
        //8.获取一阵数据getframe，存到了frame指针中
        if(0 == Hrg_GetFrame(&handle, &frame))
        {
            //printf("frame index:%d\n", frame.index);
            //9.该函数获取这frame里面的深度数据和幅度数据，从output_data里面找
            //这里的F32是float32类型的，表示更高精度的数据
            //还提供流Int16类型的，为“低精度”
            Hrg_GetDepthF32andAmplitudeData(&handle,
                                            &frame,
                                            output_data[output_idx].depth.get(),
                                            output_data[output_idx].amplitude.get());

            /* then you can process depth data and amplitude data */
#ifdef OPENCV_VISUALIZER
            //10.可以用openCV显示出来，生成Mat格式

            cv::Mat tof_depth = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32F, output_data[output_idx].depth.get()); //原始深度图
            cv::Mat tof_amplitude= cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_16S, output_data[output_idx].amplitude.get()); //原始幅度图

            /*** In order to obtain higher quality images, bilateral filtering is recommended. ***/
            
            //11.双边滤波，滤波后的图像放置在img_tof_depth_filter里面
            cv::Mat img_tof_depth_filter;
             // 参数解释：void bilateralFilter(InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace, intborderType=BORDER_DEFAULT )
            // src：源必须是8位或者浮点数，1或者3通道图片。
            // dst：输出图片，和输入图片相同大小和深度。
            // d：在滤波过程中使用的各像素邻域直径，如果这是一个非整数，则这个值由sigmaSpace决定。
            // sigmaColor：颜色空间的标准方差。数值越大，意味着越远的的颜色会被混进邻域内，从而使更大的颜色段获得相同的颜色。
            // sigmaSpace：坐标空间的标注方差。 数值越大，以为着越远的像素会相互影响，从而使更大的区域足够相似的颜色获取相同的颜色。当d>0，d指定了邻域大小且与sigmaSpace无关。否则，d正比于sigmaSpace。
            cv::bilateralFilter(tof_depth*1000, img_tof_depth_filter, 20, 40, 10);
            cv::Mat tof_depth_f =img_tof_depth_filter/1000;
            
            //将处理后的tof_depth_f的数据拷贝到我们设定的Tof_depth上
            tof_depth_f.copyTo(tof_depth);

#endif
            /* decode one depth_f32 data to rgb */
            
            //12.根据depth信息渲染到彩色图
            Hrg_DepthF32ToRGB(&handle, depth_rgb, IMAGE_HEIGHT*IMAGE_WIDTH*3, output_data[output_idx].depth.get(), IMAGE_HEIGHT*IMAGE_WIDTH, 0, 3.747);

            /* decode one amplitude data to gray */
            
            //13.根据amplitude信息渲染到灰度图
            Hrg_AmplitudeToIR(&handle, dst_ir, IMAGE_HEIGHT*IMAGE_WIDTH, output_data[output_idx].amplitude.get(), IMAGE_HEIGHT*IMAGE_WIDTH, 1200);

#ifdef OPENCV_VISUALIZER
            cv::Mat tof_depth_RGB = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, depth_rgb); //渲染后的深度图
            cv::Mat tof_amplitude_IR = cv::Mat(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1, dst_ir);//渲染后的幅度图

            cv::namedWindow("depth", 0);
            cv::setMouseCallback("depth", onMouse_depth, reinterpret_cast<void *>(&tof_depth));
            cv::imshow("depth", tof_depth_RGB);
            cv::waitKey(1);

            cv::namedWindow("amplitude", 0);
            cv::setMouseCallback("amplitude", onMouse_amplitude, reinterpret_cast<void *>(&tof_amplitude));
            cv::imshow("amplitude", tof_amplitude_IR);
            cv::waitKey(1);
#endif
            /*** Get point cloud data from distance data. ***/
            Hrg_GetXYZDataF32_f(&handle, output_data[output_idx].depth.get(), pcl, IMAGE_HEIGHT*IMAGE_WIDTH);

#ifdef PCL_VISUALIZER
            viewer->removeAllPointClouds();
            point_cloud_ptr->clear();

            for(int i=0;i<IMAGE_HEIGHT;i++)
            {
                for(int j=0;j<IMAGE_WIDTH;j++)
                {
                    int index = i*IMAGE_WIDTH+j;
                    pcl::PointXYZ point;
                    point.x = pcl[index*3+0];
                    point.y = pcl[index*3+1];
                    point.z = pcl[index*3+2];

                    if(point.z > 0)
                    {
                        point_cloud_ptr->points.push_back(point);
                    }
                }
            }
            point_cloud_ptr->width = point_cloud_ptr->points.size();
            point_cloud_ptr->height = 1;

            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::transformPointCloud(*point_cloud_ptr, *transformed_cloud, transform);
            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(point_cloud_ptr, "z");//按照z字段进行渲染
            viewer->addPointCloud<pcl::PointXYZ>(transformed_cloud, fildColor);//显示点云，其中fildColor为颜色显示
            viewer->spinOnce(1);
            boost::this_thread::sleep(boost::posix_time::microseconds(1));
#endif

            Hrg_FreeFrame(&handle, &frame);
            output_idx = output_idx < 9 ? output_idx + 1 : 0;
        }

    }

    Hrg_StopStream(&handle);
    Hrg_CloseDevice(&handle);

    delete[] depth_rgb;
    delete[] dst_ir;
    delete[] pcl;

    return 0;
}



