#ifndef EVENTSPHERE_H
#define EVENTSPHERE_H
// MIS Laboratory, University of Picardy Jules Verne
// Author: Maxime Robic maxime.robic@u-picardie.fr
#include "metavision/sdk/base/events/event2d.h"
#include "event3d.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <pcl/common/angles.h> // for pcl::deg2rad
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>

namespace Metavision {

/// @brief Class to represent event in a unit sphere
class EventSphere{
public:
    /// @brief List of 3D event in the sphere
    std::vector<Event3d> ListEvent;

    /// @brief Intrinsic parameters cam 1
    cv::Mat K1;

    /// @brief Intrinsic parameters cam 2
    cv::Mat K2;

    /// @brief Parameter xi of Mei's unified model cam 1
    double xi1;

    /// @brief Parameter xi of Mei's unified model cam 2
    double xi2;

    /// @brief Lookup table for cam 1
    cv::Mat Table1;

    /// @brief Lookup table for cam 2
    cv::Mat Table2;

    /// @brief Default constructor
    EventSphere() = default;

    /// @brief Whole constructor
    inline EventSphere(std::vector<Event3d> eventList, cv::Mat K1, double xi1,  cv::Mat K2, double xi2){
        this->ListEvent=eventList;
        this->K1=K1;
        this->xi1=xi1;
        this->K2=K2;
        this->xi2=xi2;
    }

    /// @brief Minimal constructor if only one cam is available
    inline EventSphere(cv::Mat K, double xi){
        this->K1=K;
        this->K2=K;
        this->xi1=xi;
        this->xi2=xi;
    }

    /// @brief Classic constructor, mostly used
    inline EventSphere(cv::Mat K1, double xi1, cv::Mat K2, double xi2){
        this->K1=K1;
        this->K2=K2;
        this->xi1=xi1;
        this->xi2=xi2;
    }

    inline void resetEvent(){
        this->ListEvent.clear();
    }

    inline void generateTable(int width, int height){
        cv::Mat M1(cv::Size(width,height), CV_64FC3,0.);
        cv::Mat M2(cv::Size(width,height), CV_64FC3,0.);
        int w = M1.size().width; int h = M1.size().height;
        for(int i = 0; i < w ; i++){
        for(int j = 0; j < h ; j++){
        M1.at<cv::Vec3f>(j,i) = reverseProjection(i,j,0);
        M2.at<cv::Vec3f>(j,i) = reverseProjection(i,j,1);
        }
        }
        this->Table1=M1;
        this->Table2=M2;
    }

    /// @brief Add an event to the semi sphere 1 by default, if only one cam is available
    inline void addToTheSemiSphere(Event2d ev2){
        cv::Vec3f coord3D = reverseProjection(ev2.x,ev2.y,0);
        Event3d ev3 (ev2, coord3D[0], coord3D[1], coord3D[2]);
        this->ListEvent.push_back(ev3);
    }


    /// @brief Add an event to the whole sphere
    inline void addToTheSphere(Event2d ev2, int id){
        cv::Vec3f coord3D = reverseProjection(ev2.x,ev2.y,id);
        Event3d ev3 (ev2, coord3D[0], coord3D[1], coord3D[2]);
        this->ListEvent.push_back(ev3);
    }
    
   
    /// @brief Function to reproject uv coordinates into unit sphere 3D coordinates
    inline cv::Vec3f reverseProjection(double u, double v, int id){
        double ku, kv, u0, v0, l, m;
        if(id==0){
            ku = this->K1.at<double>(0,0);
            kv = this->K1.at<double>(1,1);
            u0 = this->K1.at<double>(0,2);
            v0 = this->K1.at<double>(1,2);
            l = this->xi1;    
        }
        if(id==1){
            ku = this->K2.at<double>(0,0);
            kv = this->K2.at<double>(1,1);
            u0 = this->K2.at<double>(0,2);
            v0 = this->K2.at<double>(1,2);
            l = this->xi2;
        }

        //change of frame to image frame
        double x = (u - u0)/ku;
        double y = (v - v0)/kv;

        double coeff = (l + sqrt(1.0 + (1 - pow(l,2))*(pow(x,2) + pow(y,2))))/(pow(x,2) + pow(y,2) + 1.0);

        //projection to unit sphere
        double xs = coeff*x;
        
        double ys = coeff*y;

        double zs = coeff - l;


        if(id==1){
            zs=-zs;
            xs=-xs;
        }

        cv::Vec3f coord(xs,ys,zs);
        return coord;
    }

    /// @brief Function to reproject uv coordinates into unit sphere 3D coordinates
    inline cv::Point3d cam2sphere(cv::Point2d pt, int id){
        cv::Vec3f vec3;
        if(id==0){
            vec3 = Table1.at<cv::Vec3f>(pt.y,pt.x);
            //cv::Vec3f coord3D2 = Table2.at<cv::Vec3f>(j,i);
        }
        if(id==1){
            vec3 = Table2.at<cv::Vec3f>(pt.y,pt.x);
        }
        cv::Point3d pt3(vec3);
        return pt3;
    }
    
   
    /// @brief Function to generate an RGB pointcloud using PCL libraries for the visualization
    inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr generatePcl(){
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ( new pcl::PointCloud<pcl::PointXYZRGB> );
        for(int i=0;i<ListEvent.size();i++){
            Metavision::Event3d ev =ListEvent.at(i);
            pcl::PointXYZRGB pt;
            pt.x=ev.X;
            pt.y=ev.Y;
            pt.z=ev.Z;
            if((int)ev.p==1){
            pt.r=pt.g=pt.b=255;
            }
            if((int)ev.p==0){
            pt.r=pt.g=0;
            pt.b=255;
            }
            cloud->points.push_back(pt);
        }
        return cloud;
    }


    /// @brief Construct an accumulated sphere cloud from an accumulated map
    inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr SphereFromAccMapFast(cv::Mat &frame1, cv::Mat &frame2){

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ( new pcl::PointCloud<pcl::PointXYZRGB> );
        int w = frame1.size().width; int h = frame1.size().height;//we suppose frame have same dimensions
        for(int i = 0; i < w ; i++){
        for(int j = 0; j < h ; j++){
        if(frame1.at<cv::Vec3b>(j,i)[0]>150 && frame2.at<cv::Vec3b>(j,i)[0]>150){//event detected on both frames
        
            cv::Vec3f coord3D1 = Table1.at<cv::Vec3f>(j,i);
            cv::Vec3f coord3D2 = Table2.at<cv::Vec3f>(j,i);
  
            pcl::PointXYZRGB pt1; pcl::PointXYZRGB pt2;
            pt1.x=coord3D1[0]; pt1.y=coord3D1[1]; pt1.z=coord3D1[2];
            pt2.x=coord3D2[0]; pt2.y=coord3D2[1]; pt2.z=coord3D2[2];

            pt1.b = frame1.at<cv::Vec3b>(j,i)[0]; pt1.g = frame1.at<cv::Vec3b>(j,i)[1]; pt1.r = frame1.at<cv::Vec3b>(j,i)[2];
            pt2.b = frame2.at<cv::Vec3b>(j,i)[0]; pt2.g = frame2.at<cv::Vec3b>(j,i)[1]; pt2.r = frame2.at<cv::Vec3b>(j,i)[2];
 
            cloud->points.push_back(pt1); cloud->points.push_back(pt2);

        }
        else if(frame1.at<cv::Vec3b>(j,i)[0]>150 && frame2.at<cv::Vec3b>(j,i)[0]<=150){

            cv::Vec3f coord3D1 = Table1.at<cv::Vec3f>(j,i);
            pcl::PointXYZRGB pt1;
            pt1.x=coord3D1[0]; pt1.y=coord3D1[1]; pt1.z=coord3D1[2];

            pt1.b = frame1.at<cv::Vec3b>(j,i)[0]; pt1.g = frame1.at<cv::Vec3b>(j,i)[1]; pt1.r = frame1.at<cv::Vec3b>(j,i)[2];
        
            cloud->points.push_back(pt1); 
        }
        else if(frame1.at<cv::Vec3b>(j,i)[0]<=150 && frame2.at<cv::Vec3b>(j,i)[0]>150){

            cv::Vec3f coord3D2 = Table2.at<cv::Vec3f>(j,i);
            pcl::PointXYZRGB pt2;
            pt2.x=coord3D2[0]; pt2.y=coord3D2[1]; pt2.z=coord3D2[2];

            pt2.b = frame2.at<cv::Vec3b>(j,i)[0]; pt2.g = frame2.at<cv::Vec3b>(j,i)[1]; pt2.r = frame2.at<cv::Vec3b>(j,i)[2];

            cloud->points.push_back(pt2); 
        }
        }
        }
    return cloud;
    }

     /// @brief Construct an accumulated sphere cloud + register event 3D on the sphere, from an accumulated map
    inline pcl::PointCloud<pcl::PointXYZRGB>::Ptr SphereFromAccMap(cv::Mat &frame1, cv::Mat &frame2){

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud ( new pcl::PointCloud<pcl::PointXYZRGB> );
        int w = frame1.size().width; int h = frame1.size().height;//we suppose frame have same dimensions
        for(int i = 0; i < w ; i++){
        for(int j = 0; j < h ; j++){
        if(frame1.at<cv::Vec3b>(j,i)[0]>150 && frame2.at<cv::Vec3b>(j,i)[0]>150){//event detected on both frames
            Metavision::Event3d ev1; Metavision::Event3d ev2; 
            cv::Vec3f coord3D1 = Table1.at<cv::Vec3f>(j,i);
            cv::Vec3f coord3D2 = Table2.at<cv::Vec3f>(j,i);
            ev1.x = ev2.x = i; ev1.y = ev2.y = j; ev1.t = ev2.t = 0;
            ev1.X = coord3D1[0]; ev1.Y = coord3D1[1]; ev1.Z = coord3D1[2];
            ev2.X = coord3D2[0]; ev2.Y = coord3D2[1]; ev2.Z = coord3D2[2];
            pcl::PointXYZRGB pt1; pcl::PointXYZRGB pt2;
            pt1.x=coord3D1[0]; pt1.y=coord3D1[1]; pt1.z=coord3D1[2];
            pt2.x=coord3D2[0]; pt2.y=coord3D2[1]; pt2.z=coord3D2[2];

            pt1.b = frame1.at<cv::Vec3b>(j,i)[0]; pt1.g = frame1.at<cv::Vec3b>(j,i)[1]; pt1.r = frame1.at<cv::Vec3b>(j,i)[2];
            pt2.b = frame2.at<cv::Vec3b>(j,i)[0]; pt2.g = frame2.at<cv::Vec3b>(j,i)[1]; pt2.r = frame2.at<cv::Vec3b>(j,i)[2];

            if(pt1.b == 255){
            ev1.p = 1;
            }
            else {
            ev1.p = 0;
            }

            if(pt2.b == 255){
            ev2.p = 1;
            }
            else {
            ev2.p = 0;
            }
            this->ListEvent.push_back(ev1); this->ListEvent.push_back(ev2);
            cloud->points.push_back(pt1); cloud->points.push_back(pt2);

        }
        else if(frame1.at<cv::Vec3b>(j,i)[0]>150 && frame2.at<cv::Vec3b>(j,i)[0]<=150){
            Metavision::Event3d ev1;
            cv::Vec3f coord3D1 = Table1.at<cv::Vec3f>(j,i);
            ev1.x = i; ev1.y = j; ev1.t = 0;
            ev1.X = coord3D1[0]; ev1.Y = coord3D1[1]; ev1.Z = coord3D1[2];
            pcl::PointXYZRGB pt1;
            pt1.x=coord3D1[0]; pt1.y=coord3D1[1]; pt1.z=coord3D1[2];

            pt1.b = frame1.at<cv::Vec3b>(j,i)[0]; pt1.g = frame1.at<cv::Vec3b>(j,i)[1]; pt1.r = frame1.at<cv::Vec3b>(j,i)[2];

            if(pt1.b == 255){
            ev1.p = 1;
            }
            else {
            ev1.p = 0;
            }

            ListEvent.push_back(ev1);
        
            cloud->points.push_back(pt1); 
        }
        else if(frame1.at<cv::Vec3b>(j,i)[0]<=150 && frame2.at<cv::Vec3b>(j,i)[0]>150){
            Metavision::Event3d ev2;
            cv::Vec3f coord3D2 = Table2.at<cv::Vec3f>(j,i);
            ev2.x = i; ev2.y = j; ev2.t = 0;
            ev2.X = coord3D2[0]; ev2.Y = coord3D2[1]; ev2.Z = coord3D2[2];
            pcl::PointXYZRGB pt2;
            pt2.x=coord3D2[0]; pt2.y=coord3D2[1]; pt2.z=coord3D2[2];

            pt2.b = frame2.at<cv::Vec3b>(j,i)[0]; pt2.g = frame2.at<cv::Vec3b>(j,i)[1]; pt2.r = frame2.at<cv::Vec3b>(j,i)[2];

            if(pt2.b == 255){
            ev2.p = 1;
            }
            else {
            ev2.p = 0;
            }

            ListEvent.push_back(ev2);
            cloud->points.push_back(pt2); 
        }
        }
        }
    return cloud;
    }




};

} // namespace Metavision

//METAVISION_DEFINE_EVENT_TRAIT(Metavision::EventCD, 12, "CD")

#endif // METAVISION_SDK_BASE_EVENT_CD_H

