// MIS Laboratory, University of Picardy Jules Verne
// Author: Maxime Robic maxime.robic@u-picardie.fr
//
// KentTracker: event-based spherical tracking of a moving target with a Kent (FB5) filter.
// Two synchronized event cameras (master/slave fisheye pair) are replayed from .raw files,
// sparse optical flow is computed on each, flows are lifted to the unit sphere through the
// unified camera model, and a Kent filter tracks the target bearing (mu) and heading (xi1).

#include <exception>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <filesystem>

#include <metavision/sdk/cv/algorithms/sparse_optical_flow_algorithm.h>
#include <metavision/sdk/cv/events/event_optical_flow.h>
#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/log.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/sdk/cv/algorithms/anti_flicker_algorithm.h>
#include <metavision/sdk/cv/configs/frequency_estimation_config.h>
#include <metavision/sdk/cv/algorithms/spatio_temporal_contrast_algorithm.h>
#include <metavision/hal/facilities/i_geometry.h>
#include <metavision/hal/facilities/i_events_stream_decoder.h>
#include <metavision/hal/facilities/i_event_decoder.h>
#include <metavision/hal/device/device.h>
#include <metavision/hal/facilities/i_events_stream.h>
#include <metavision/sdk/core/algorithms/periodic_frame_generation_algorithm.h>
#include <metavision/sdk/ui/utils/window.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <metavision/sdk/driver/camera.h>

#include <vtkObject.h>
#include <vtkOutputWindow.h>
#include <vtkCamera.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>

#include <pcl/common/angles.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include "eventSphere.hpp"
#include "kentEstimator.hpp"

using namespace std::chrono_literals;

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

    return buf;
}

auto dateTime = currentDateTime();

std::ofstream flog;    // tracker state, one row per main-loop iteration
std::ofstream faerr;   // alog mode: A_sum vs A_opt approximation-error metrics, one row per Kent update

// Optional binary masks (255 = rejected pixel), loaded only when use_mask=1
cv::Mat imgMaskMaster;
cv::Mat imgMaskSlave;


/// Removes the flows that fall on a masked pixel (id: 0 = master, 1 = slave)
std::vector<Metavision::EventOpticalFlow> filter_mask(const std::vector<Metavision::EventOpticalFlow>& flows, int id) {
    std::vector<Metavision::EventOpticalFlow> filtered;
    filtered.reserve(flows.size());
    const cv::Mat &mask = (id == 0) ? imgMaskMaster : imgMaskSlave;

    for (auto it = flows.rbegin(); it != flows.rend(); ++it) {
        if (mask.at<uchar>((it->center_y), (it->center_x)) != 255) {
            filtered.push_back(*it);
        }
    }

    return filtered;
}


/// Keeps only the most recent flow of each cluster id
std::vector<Metavision::EventOpticalFlow> keep_one_flow_per_id(const std::vector<Metavision::EventOpticalFlow>& flows) {
    std::unordered_set<uint32_t> seen_ids;
    std::vector<Metavision::EventOpticalFlow> filtered;
    filtered.reserve(flows.size());
    for (auto it = flows.rbegin(); it != flows.rend(); ++it) {
        if (seen_ids.insert(it->id).second) {
            filtered.push_back(*it);
        }
    }
    std::reverse(filtered.begin(), filtered.end());
    return filtered;
}


/// Groups flows that are close in position and direction into their average flow
std::vector<Metavision::EventOpticalFlow> group_similar_flows(
    const std::vector<Metavision::EventOpticalFlow>& flows,
    double max_dist = 5.0,      // max distance between origins (pixels)
    double max_angle_deg = 20.0 // max angle between directions (degrees)
) {
    std::vector<Metavision::EventOpticalFlow> result;
    std::vector<bool> used(flows.size(), false);

    for (size_t i = 0; i < flows.size(); ++i) {
        if (used[i]) continue;

        double sum_x = flows[i].center_x, sum_y = flows[i].center_y;
        double sum_vx = flows[i].vx, sum_vy = flows[i].vy;
        double sum_t = flows[i].t;
        int count = 1;
        used[i] = true;

        for (size_t j = i + 1; j < flows.size(); ++j) {
            if (used[j]) continue;

            double dx = flows[j].center_x - flows[i].center_x;
            double dy = flows[j].center_y - flows[i].center_y;
            double dist = std::sqrt(dx*dx + dy*dy);

            double norm1 = std::sqrt(flows[i].vx*flows[i].vx + flows[i].vy*flows[i].vy);
            double norm2 = std::sqrt(flows[j].vx*flows[j].vx + flows[j].vy*flows[j].vy);
            if (norm1 < 1e-6 || norm2 < 1e-6) continue; // skip null vectors
            double dot = (flows[i].vx*flows[j].vx + flows[i].vy*flows[j].vy) / (norm1 * norm2);
            dot = std::max(-1.0, std::min(1.0, dot));
            double angle = std::acos(dot) * 180.0 / M_PI;

            if (dist < max_dist && angle < max_angle_deg) {
                sum_x += flows[j].center_x;
                sum_y += flows[j].center_y;
                sum_vx += flows[j].vx;
                sum_vy += flows[j].vy;
                sum_t += flows[j].t;
                count++;
                used[j] = true;
            }
        }

        Metavision::EventOpticalFlow avg = flows[i];
        avg.center_x = sum_x / count;
        avg.center_y = sum_y / count;
        avg.vx = sum_vx / count;
        avg.vy = sum_vy / count;
        avg.t = sum_t / count;
        result.push_back(avg);
    }
    return result;
}

/// Removes the flows slower than min_speed (pixels/s)
std::vector<Metavision::EventOpticalFlow> filter_by_speed(const std::vector<Metavision::EventOpticalFlow>& flows, double min_speed = 1.0) {
    std::vector<Metavision::EventOpticalFlow> filtered;
    for (const auto &evt : flows) {
        double speed = std::sqrt(evt.vx * evt.vx + evt.vy * evt.vy);
        if (speed >= min_speed) {
            filtered.push_back(evt);
        }
    }
    return filtered;
}

/// Draws the origin of each flow as a small filled circle
template<typename InputIt>
void draw_flow_origin(InputIt begin, InputIt end, cv::Mat &output) {
    const cv::Scalar color(175, 0, 255);
    for (auto it = begin; it != end; ++it)
        cv::circle(output, cv::Point(it->x, it->y), 2, color, -1);
}


/// Holds the latest accumulated frame of one camera, used to build the sphere point cloud
class EventAnalyzer {
public:
    bool is_master;

    cv::Mat frame;

    cv::Vec3b color_bg  = cv::Vec3b(52, 37, 30);

    void setup_display(const int width, const int height) {
        frame = cv::Mat(height, width, CV_8UC3);
        frame.setTo(color_bg);
    }

    cv::Mat get_frame() {
        return frame;
    }

    void set_frame(cv::Mat &display) {
        this->frame = display;
    }

    void setMasterMode(bool is_master){
        this->is_master = is_master;
    }
};

class CloudManager{
public:
    std::mutex m;

    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud;

    pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr get_cloud(){
        std::unique_lock<std::mutex> lock(m);
        return this->cloud;
    }

    void set_cloud(pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_in){
        std::unique_lock<std::mutex> lock(m);
        this->cloud=cloud_in;
    }

};

int main(int argc, char *argv[]) {
    vtkObject::GlobalWarningDisplayOff();
    if(argc < 3)
    {
        std::cerr << "Missing informations" << std::endl;
        std::cerr << "Using : " << argv[0] << " {master.raw} {slave.raw} [key=value ...] [--adaptive]" << std::endl;
        std::cerr << "  keys: bearing_gate heading_gate heading_weight beta_activate reinit_kappa_min reinit_miss_limit"
                  << " pred_k_noise pred_beta_noise min_speed use_mask mask_master mask_slave use_heading adaptive"
                  << " log_dir tag alog dump_dir cam_cx cam_cy" << std::endl;
        return -1;
    }

    // ---- Tunable parameters (optional key=value args after the two .raw paths) ----
    double p_bearing_gate = 15.0, p_heading_gate = 60.0, p_heading_weight = 1, p_beta_activate = 5.0;
    double p_reinit_kappa = 50.0;  int p_reinit_miss = 100;
    double p_pred_k = 500.0, p_pred_beta = 200.0;
    double p_min_speed = 2.0;  int p_use_mask = 0;  int p_use_heading = 1;
    std::string p_mask_master = "mask_master.png", p_mask_slave = "mask_slave.png";
    std::string p_log_dir = ".";
    std::string p_tag = "";
    int p_alog = 0;   // 1 -> log A_sum vs A_opt approximation-error metrics to aerror_<tag>.dat
    std::string p_dump_dir = "";   // if set, save PCL sphere screenshots here (frame_%05d.png)
    double p_cam_cx = 0.1, p_cam_cy = 0.4;  // image-shift of the sphere view (VTK camera WindowCenter)
    int p_adaptive = 0;                      // adaptive association cost (weights = kappa/kappa_max, beta/beta_max)
    for (int a = 3; a < argc; ++a) {
        std::string tok(argv[a]);
        if (tok == "--adaptive") { p_adaptive = 1; continue; }   // bare flag form
        auto eq = tok.find('=');
        if (eq == std::string::npos) continue;
        std::string key = tok.substr(0, eq), val = tok.substr(eq + 1);
        if      (key == "bearing_gate")      p_bearing_gate   = std::stod(val);
        else if (key == "heading_gate")      p_heading_gate   = std::stod(val);
        else if (key == "heading_weight")    p_heading_weight = std::stod(val);
        else if (key == "beta_activate")     p_beta_activate  = std::stod(val);
        else if (key == "reinit_kappa_min")  p_reinit_kappa   = std::stod(val);
        else if (key == "reinit_miss_limit") p_reinit_miss    = std::stoi(val);
        else if (key == "pred_k_noise")      p_pred_k         = std::stod(val);
        else if (key == "pred_beta_noise")   p_pred_beta      = std::stod(val);
        else if (key == "min_speed")         p_min_speed      = std::stod(val);
        else if (key == "use_mask")          p_use_mask       = std::stoi(val);
        else if (key == "mask_master")       p_mask_master    = val;
        else if (key == "mask_slave")        p_mask_slave     = val;
        else if (key == "use_heading")       p_use_heading    = std::stoi(val);
        else if (key == "adaptive")          p_adaptive       = std::stoi(val);
        else if (key == "log_dir")           p_log_dir        = val;
        else if (key == "dump_dir")          p_dump_dir       = val;
        else if (key == "cam_cx")            p_cam_cx         = std::stod(val);
        else if (key == "cam_cy")            p_cam_cy         = std::stod(val);
        else if (key == "tag")               p_tag            = val;
        else if (key == "alog")              p_alog           = std::stoi(val);
        else std::cerr << "[warn] unknown parameter '" << key << "' ignored" << std::endl;
    }
    // Log files are named after the tag (or the date) so that successive runs do not collide
    {
        std::filesystem::create_directories(p_log_dir);
        std::string tagged = p_log_dir + "/trackerKent_" + (p_tag.empty() ? dateTime : p_tag) + ".dat";
        flog.open(tagged.c_str());
        if (p_alog) {
            std::string apath = p_log_dir + "/aerror_" + (p_tag.empty() ? dateTime : p_tag) + ".dat";
            faerr.open(apath.c_str());
            faerr << "t errFrob errFrobRel tangResid betaSum betaOpt nInit\n";
            std::cout << "[config] ALOG (A_sum vs A_opt) -> " << apath << std::endl;
        }
        if (p_use_mask) {
            imgMaskMaster = cv::imread(p_mask_master, cv::IMREAD_GRAYSCALE);
            imgMaskSlave  = cv::imread(p_mask_slave,  cv::IMREAD_GRAYSCALE);
            if (imgMaskMaster.empty() || imgMaskSlave.empty()) {
                std::cerr << "Could not read the masks " << p_mask_master << " / " << p_mask_slave << std::endl;
                return -1;
            }
        }
        std::cout << "[config] tag=" << (p_tag.empty() ? dateTime : p_tag)
                  << " bearing_gate=" << p_bearing_gate << " heading_gate=" << p_heading_gate
                  << " heading_weight=" << p_heading_weight << " beta_activate=" << p_beta_activate
                  << " reinit_kappa_min=" << p_reinit_kappa << " reinit_miss_limit=" << p_reinit_miss
                  << " pred_k_noise=" << p_pred_k << " pred_beta_noise=" << p_pred_beta
                  << " min_speed=" << p_min_speed << " use_mask=" << p_use_mask
                  << " use_heading=" << p_use_heading << " adaptive=" << p_adaptive << std::endl;
        std::cout << "[config] log -> " << tagged << std::endl;
        if (!p_dump_dir.empty()) {
            std::filesystem::create_directories(p_dump_dir);
            std::cout << "[config] PCL sphere frames -> " << p_dump_dir << "/frame_%05d.png" << std::endl;
        }
    }

    // Open the slave camera first
    std::cout << "Opening camera slave..." << std::endl;
    Metavision::Camera slaveCamera;
    try {
        Metavision::FileConfigHints hints;
        hints.real_time_playback(true);
        slaveCamera = Metavision::Camera::from_file(argv[2], hints);
    } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }

    std::cout << "Camera slave open." << std::endl;

    Metavision::I_LL_Biases *i_ll_biases_slave = slaveCamera.get_device().get_facility<Metavision::I_LL_Biases>();
    if (i_ll_biases_slave) {
        i_ll_biases_slave->set("bias_diff",-5);
        i_ll_biases_slave->set("bias_diff_off",190);
        i_ll_biases_slave->set("bias_diff_on",28);
    }

    Metavision::I_EventsStream *i_eventsstream_slave = slaveCamera.get_device().get_facility<Metavision::I_EventsStream>();
    if (!i_eventsstream_slave) {
        std::cerr << "Could not initialize events stream of camera slave." << std::endl;
        return 3;
    }

    // Open the master camera
    std::cout << "Opening camera master..." << std::endl;
    Metavision::Camera masterCamera;
    try {
        Metavision::FileConfigHints hints2;
        hints2.real_time_playback(true);
        masterCamera = Metavision::Camera::from_file(argv[1], hints2);
    } catch (Metavision::CameraException &e) { MV_LOG_ERROR() << e.what(); }

    std::cout << "Camera master open." << std::endl;

    Metavision::I_LL_Biases *i_ll_biases_master = masterCamera.get_device().get_facility<Metavision::I_LL_Biases>();
    if (i_ll_biases_master) {
        i_ll_biases_master->set("bias_diff",-5);
        i_ll_biases_master->set("bias_diff_off",190);
        i_ll_biases_master->set("bias_diff_on",28);
    }

    Metavision::I_EventsStream *i_eventsstream_master = masterCamera.get_device().get_facility<Metavision::I_EventsStream>();
    if (!i_eventsstream_master) {
        std::cerr << "Could not initialize events stream of camera master." << std::endl;
        return 3;
    }

    // get camera slave geometry
    Metavision::I_Geometry *i_geometry_slave = slaveCamera.get_device().get_facility<Metavision::I_Geometry>();
    if (!i_geometry_slave) {
        std::cerr << "Could not retrieve camera slave geometry." << std::endl;
        return 4;
    }

    // get camera master geometry
    Metavision::I_Geometry *i_geometry_master = masterCamera.get_device().get_facility<Metavision::I_Geometry>();
    if (!i_geometry_master) {
        std::cerr << "Could not retrieve camera master geometry." << std::endl;
        return 4;
    }

    // Accumulation time & frame rate
    double fps              = 50;
    const std::uint32_t acc = 30000;
    const std::uint32_t threshold_stc = 10000;

    std::unique_ptr<Metavision::SparseOpticalFlowAlgorithm> flow_algo_slave;
    std::unique_ptr<Metavision::SparseOpticalFlowAlgorithm> flow_algo_master;
    std::vector<Metavision::EventOpticalFlow> flow_output_slave;
    std::vector<Metavision::EventOpticalFlow> flow_output_master;
    std::vector<Metavision::EventOpticalFlow> grouped_flow_s;
    std::vector<Metavision::EventOpticalFlow> grouped_flow_m;


    KentEstimator ke;
    // ---- Kent-aware data association (robustness to dynamic distractors) ----
    ke.use_heading_assoc_    = (p_use_heading != 0);
    ke.assoc_bearing_gate_   = p_bearing_gate;
    ke.assoc_heading_gate_   = p_heading_gate;
    ke.assoc_heading_weight_ = p_heading_weight;
    ke.assoc_beta_activate_  = p_beta_activate;
    ke.adaptive_assoc_       = (p_adaptive != 0);
    ke.reinit_kappa_min_     = p_reinit_kappa;
    ke.reinit_miss_limit_    = p_reinit_miss;
    bool FirstInit = false;
    bool clockStarted = false;

    Metavision::SparseOpticalFlowConfig config;
    config.distance_update_rate = 0.05f;
    config.damping              = 0.707f;
    config.omega_cutoff         = 10.f;
    config.min_cluster_size     = 5;
    config.max_link_time        = 25000;
    config.match_polarity       = true;
    config.use_simple_match     = true;
    config.full_square          = false;
    config.last_event_only      = false;
    config.size_threshold       = 100000000;

    flow_algo_slave = std::make_unique<Metavision::SparseOpticalFlowAlgorithm>(i_geometry_slave->get_width(), i_geometry_slave->get_height(),  config);
    flow_algo_master = std::make_unique<Metavision::SparseOpticalFlowAlgorithm>(i_geometry_master->get_width(), i_geometry_master->get_height(),  config);

    Metavision::FrequencyEstimationConfig config_flickering1(7, 50, 165,3000,false);


    // Slave pipeline: anti-flicker -> spatio-temporal contrast -> frame generation + sparse optical flow
    EventAnalyzer event_analyzer_slave;
    event_analyzer_slave.setMasterMode(false);
    event_analyzer_slave.setup_display(i_geometry_slave->get_width(), i_geometry_slave->get_height());
    auto frame_gen_slave = Metavision::PeriodicFrameGenerationAlgorithm(i_geometry_slave->get_width(), i_geometry_slave->get_height(), acc, fps);
    auto stc_filter_slave = Metavision::SpatioTemporalContrastAlgorithm(i_geometry_slave->get_width(), i_geometry_slave->get_height(), threshold_stc);
    auto anti_flickering_slave1 = Metavision::AntiFlickerAlgorithm(i_geometry_slave->get_width(), i_geometry_slave->get_height(), config_flickering1);

    Metavision::I_EventDecoder<Metavision::EventCD> *i_cddecoder_slave =
        slaveCamera.get_device().get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();

    if (i_cddecoder_slave) {
        i_cddecoder_slave->add_event_buffer_callback(
            [&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
                std::vector<Metavision::EventCD> output,output_temp1;
                anti_flickering_slave1.process_events(begin,end,std::back_inserter(output_temp1));
                stc_filter_slave.process_events(output_temp1.begin(),output_temp1.end(),std::back_inserter(output));
                frame_gen_slave.process_events(output.begin(), output.end());
                flow_output_slave.clear();
                flow_algo_slave->process_events(output.begin(), output.end(), std::back_inserter(flow_output_slave));
                output.clear();output_temp1.clear();
            });
    }

    std::cout << "Camera slave started." << std::endl;


    Metavision::I_Decoder *i_decoder_slave = slaveCamera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();
    bool stop_decoding_slave               = false;
    bool stop_application_slave            = false;
    std::thread decoding_loop_slave([&]() {
        bool got_data = false;   // latch: only trust EOF after data has started flowing
        while (!stop_decoding_slave) {
            short ret = i_eventsstream_slave->poll_buffer();

            // Here we polled data, so we can launch decoding
            auto raw_data = i_eventsstream_slave->get_latest_raw_data();

            // This will trigger the callbacks set on the decoder
            if (raw_data) {
                got_data = true;
                i_decoder_slave->decode(raw_data->data(), raw_data->data() + raw_data->size());
            }
            if (got_data && ret < 0) {   // end of stream (file playback finished) -> auto-exit
                stop_decoding_slave = true;  stop_application_slave = true;
            }
        }
    });


    // Master pipeline: same processing as the slave
    EventAnalyzer event_analyzer_master;
    event_analyzer_master.setMasterMode(true);
    event_analyzer_master.setup_display(i_geometry_master->get_width(), i_geometry_master->get_height());
    auto frame_gen_master = Metavision::PeriodicFrameGenerationAlgorithm(i_geometry_master->get_width(), i_geometry_master->get_height(), acc, fps);
    auto stc_filter_master = Metavision::SpatioTemporalContrastAlgorithm(i_geometry_master->get_width(), i_geometry_master->get_height(), threshold_stc);
    auto anti_flickering_master1 = Metavision::AntiFlickerAlgorithm(i_geometry_master->get_width(), i_geometry_master->get_height(), config_flickering1);

    Metavision::I_EventDecoder<Metavision::EventCD> *i_cddecoder_master =
        masterCamera.get_device().get_facility<Metavision::I_EventDecoder<Metavision::EventCD>>();

    if (i_cddecoder_master) {
        i_cddecoder_master->add_event_buffer_callback(
            [&](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
                std::vector<Metavision::EventCD> output,output_temp1;
                anti_flickering_master1.process_events(begin,end,std::back_inserter(output_temp1));
                stc_filter_master.process_events(output_temp1.begin(),output_temp1.end(),std::back_inserter(output));
                frame_gen_master.process_events(output.begin(), output.end());
                flow_output_master.clear();
                flow_algo_master->process_events(output.begin(), output.end(), std::back_inserter(flow_output_master));
                output.clear();output_temp1.clear();
            });
    }

    std::cout << "Camera master started." << std::endl;

    Metavision::I_Decoder *i_decoder_master = masterCamera.get_device().get_facility<Metavision::I_EventsStreamDecoder>();
    bool stop_decoding_master               = false;
    bool stop_application_master            = false;
    std::thread decoding_loop_master([&]() {
        bool got_data = false;   // latch: only trust EOF after data has started flowing
        while (!stop_decoding_master) {
            short ret = i_eventsstream_master->poll_buffer();

            // Here we polled data, so we can launch decoding
            auto raw_data = i_eventsstream_master->get_latest_raw_data();

            // This will trigger the callbacks set on the decoder
            if (raw_data) {
                got_data = true;
                i_decoder_master->decode(raw_data->data(), raw_data->data() + raw_data->size());
            }
            if (got_data && ret < 0) {   // end of stream (file playback finished) -> auto-exit
                stop_decoding_master = true;  stop_application_master = true;
            }
        }
    });

    // Intrinsics (K) and mirror parameter (xi) of the unified camera model, master (1) and slave (2)
    cv::Mat K1(cv::Size(3,3),CV_64F,0.);
    K1.at<double>(0,0)= 657.51393777325984;K1.at<double>(0,1)=0;K1.at<double>(0,2)=619.35230435997846;
    K1.at<double>(1,0)=0;K1.at<double>(1,1)=657.79979934329992;K1.at<double>(1,2)=345.56186521908671;
    K1.at<double>(2,0)=0;K1.at<double>(2,1)=0;K1.at<double>(2,2)=1;
    double xi1 =1.8132336110293052;

    cv::Mat K2(cv::Size(3,3),CV_64F,0.);
    K2.at<double>(0,0)=647.94561076311186;K2.at<double>(0,1)=0;K2.at<double>(0,2)=646.79204727762703;
    K2.at<double>(1,0)=0;K2.at<double>(1,1)= 646.59722021762173;K2.at<double>(1,2)=358.34918000582428;
    K2.at<double>(2,0)=0;K2.at<double>(2,1)=0;K2.at<double>(2,2)=1;
    double xi2 = 1.7727467951283842;


    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("Global unit sphere"));
    viewer->initCameraParameters ();
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addCoordinateSystem (0.2);
    viewer->setCameraPosition(-0.15, -0.59564, 2.26217,  0.0850799, -0.81015, -0.580016);
    viewer->setCameraFieldOfView(0.8575);
    viewer->setCameraClipDistances(0.00589725 , 6.2);
    viewer->setShowFPS(false);
    // Optional image-shift of the sphere view. cam_cx: +right, cam_cy: +up.
    auto applyCamShift = [&]() {
        if (p_cam_cx == 0.0 && p_cam_cy == 0.0) return;
        auto rc = viewer->getRendererCollection();
        rc->InitTraversal();
        if (vtkRenderer *r = rc->GetNextItem()) r->GetActiveCamera()->SetWindowCenter(p_cam_cx, p_cam_cy);
    };
    applyCamShift();

    Metavision::EventSphere sphere(K1,xi1,K2,xi2);
    sphere.generateTable(i_geometry_slave->get_width(), i_geometry_slave->get_height());

    /******Thread Kent tracking on the unit sphere********/
    double iter = 0.0;
    double dtPredictMoy = 0.0, dtUpdateMoy = 0.0;
    std::thread kent_tracking([&]() {
    int width = i_geometry_master->get_width();
    int height = i_geometry_master->get_height();
    std::vector<cv::Point3d> Zk; // bearing measurements
    std::vector<cv::Point3d> Fk; // flow points on the sphere (heading measurements)
    std::vector<Metavision::EventOpticalFlow> flow_s_copy;
    std::vector<Metavision::EventOpticalFlow> flow_m_copy;
    double max_tmstp = 0.0, last_tmstp = 0.0;
    int nb_it = 0;
    while (!stop_decoding_slave && !stop_application_slave){
    nb_it++;
    Zk.clear();
    Fk.clear();
    flow_s_copy.clear();flow_s_copy=flow_output_slave;
    flow_m_copy.clear();flow_m_copy=flow_output_master;

    if(!flow_s_copy.size()==0 || !flow_m_copy.size()==0){

     auto filtered_flow_m = keep_one_flow_per_id(flow_m_copy);
     if (p_use_mask) filtered_flow_m = filter_mask(filtered_flow_m,0);
     filtered_flow_m = filter_by_speed(filtered_flow_m, p_min_speed);

     grouped_flow_m.clear();
     grouped_flow_m = group_similar_flows(filtered_flow_m, 20.0, 40.0);

     // Lift each flow to the sphere: origin -> bearing Z, origin + 5px along the flow -> flow point F.
     // A flow point leaving the image is wrapped and projected through the other camera.
     for(const auto &e : grouped_flow_m){
        cv::Point2d origin(e.center_x, e.center_y);
        cv::Point3d Z_k_i = sphere.cam2sphere(origin,0);
        Zk.push_back(Z_k_i);
        cv::Vec2d velocity(e.vx, e.vy);
        velocity = velocity / cv::norm(velocity);
        velocity = velocity * 5.0;
        cv::Point2d extremity(e.center_x + velocity(0), e.center_y + velocity(1));
        cv::Point3d F_k_i;

        bool wrapped = false;
        if (extremity.x < 0.0 || extremity.x >= width) {
            extremity.x = std::fmod(extremity.x + width, width);
            if (extremity.x < 0.0) extremity.x += width;
            wrapped = true;
        }
        if (extremity.y < 0.0 || extremity.y >= height) {
            extremity.y = std::fmod(extremity.y + height, height);
            if (extremity.y < 0.0) extremity.y += height;
            wrapped = true;
        }

        if (wrapped) {
            F_k_i = sphere.cam2sphere(extremity, 1);
        } else {
            F_k_i = sphere.cam2sphere(extremity, 0);
        }

        Fk.push_back(F_k_i);
        max_tmstp=std::max(max_tmstp,(double)e.t);
    }

    auto filtered_flow_s = keep_one_flow_per_id(flow_s_copy);

    if (p_use_mask) filtered_flow_s = filter_mask(filtered_flow_s,1);
    filtered_flow_s = filter_by_speed(filtered_flow_s, p_min_speed);

    grouped_flow_s.clear();
    grouped_flow_s = group_similar_flows(filtered_flow_s, 20.0, 40.0);
    for(const auto &e : grouped_flow_s){
        cv::Point2d origin(e.center_x, e.center_y);
        cv::Point3d Z_k_i = sphere.cam2sphere(origin,1);
        Zk.push_back(Z_k_i);
        cv::Vec2d velocity(e.vx, e.vy);
        velocity = velocity / cv::norm(velocity);
        velocity = velocity * 5.0;
        cv::Point2d extremity(e.center_x + velocity(0), e.center_y + velocity(1));
        cv::Point3d F_k_i;

        bool wrapped = false;
        if (extremity.x < 0.0 || extremity.x >= width) {
            extremity.x = std::fmod(extremity.x + width, width);
            if (extremity.x < 0.0) extremity.x += width;
            wrapped = true;
        }
        if (extremity.y < 0.0 || extremity.y >= height) {
            extremity.y = std::fmod(extremity.y + height, height);
            if (extremity.y < 0.0) extremity.y += height;
            wrapped = true;
        }

        if (wrapped) {
            F_k_i = sphere.cam2sphere(extremity, 0);
        } else {
            F_k_i = sphere.cam2sphere(extremity, 1);
        }

        Fk.push_back(F_k_i);
        max_tmstp=std::max(max_tmstp,(double)e.t);
   }

   // Initialize the filter once a single unambiguous measurement is available
   if (!Zk.empty() && Zk.size() == 1 && !Fk.empty() && Fk.size() == 1 && !ke.initialized_)
        {
        cv::Mat Gamma_0 = ke.computeGammaFromFlow(Zk[0],Fk[0]);
        cv::Point3d xi_1_0, xi_2_0;
        xi_1_0.x = Gamma_0.at<double>(0,0);
        xi_1_0.y = Gamma_0.at<double>(1,0);
        xi_1_0.z = Gamma_0.at<double>(2,0);
        xi_2_0.x = Gamma_0.at<double>(0,1);
        xi_2_0.y = Gamma_0.at<double>(1,1);
        xi_2_0.z = Gamma_0.at<double>(2,1);
        ke.initialize(Zk[0],xi_1_0,xi_2_0, 100.0, 20.0);
        last_tmstp = max_tmstp;
        FirstInit = true;
        }

    if (ke.initialized_)
        {
        double dt = (max_tmstp - last_tmstp)/1e6; //dt in s

        if(dt>0){
        auto startPredict = std::chrono::high_resolution_clock::now();
        ke.predict(p_pred_k, p_pred_beta, Zk, dt);
        auto dtPredict = std::chrono::high_resolution_clock::now() - startPredict;
        auto startUpdate = std::chrono::high_resolution_clock::now();
        ke.update(300.0, 40.0, Zk, Fk, dt);
        auto dtUpdate = std::chrono::high_resolution_clock::now() - startUpdate;
        iter = iter + 1;
        dtUpdateMoy = dtUpdateMoy +  std::chrono::duration_cast<std::chrono::microseconds>(dtUpdate).count()/1000.0;
        dtPredictMoy = dtPredictMoy + std::chrono::duration_cast<std::chrono::microseconds>(dtPredict).count()/1000.0;
        last_tmstp = max_tmstp;
        }
        }
    std::this_thread::sleep_for(5ms);
    }
    }
    });

    /******Visualization part********/

    int nFrameSlave = 0, nFrameMaster = 0, nCloud = 0;
    std::vector<Metavision::EventOpticalFlow> flow_s_copy;
    std::vector<Metavision::EventOpticalFlow> flow_m_copy;
    Metavision::Window window_slave("Slave Cam", i_geometry_slave->get_width(), i_geometry_slave->get_height(), Metavision::BaseWindow::RenderMode::BGR);
    window_slave.set_keyboard_callback(
        [&window_slave](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
            if (action == Metavision::UIAction::RELEASE &&
                (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                window_slave.set_close_flag();
            }
        });
    frame_gen_slave.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
        flow_s_copy.clear();flow_s_copy=grouped_flow_s;
        cv::Mat frame_sphere;
        frame.copyTo(frame_sphere);
        if(flow_s_copy.size()!=0){
        draw_flow_origin(flow_s_copy.cbegin(), flow_s_copy.cend(), frame_sphere);
        }
        event_analyzer_slave.set_frame(frame_sphere);
        window_slave.show(frame);
        nFrameSlave++;});


    Metavision::Window window_master("Master Cam", i_geometry_slave->get_width(), i_geometry_slave->get_height(), Metavision::BaseWindow::RenderMode::BGR);
    window_master.set_keyboard_callback(
        [&](Metavision::UIKeyEvent key, int scancode, Metavision::UIAction action, int mods) {
            if (action == Metavision::UIAction::RELEASE &&
                (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q)) {
                window_master.set_close_flag();
            }
            // 'H' prints a snapshot of the association monitor
            if (action == Metavision::UIAction::RELEASE && key == Metavision::UIKeyEvent::KEY_H) {
                std::cout << "[assoc] Kent-aware " << (ke.use_heading_assoc_ ? "ON" : "OFF")
                          << " | divergences = " << ke.assocDivergeCount_
                          << " | heading-rejects = " << ke.headingRejectCount_
                          << " | beta = " << ke.est_beta_ << " | k = " << ke.est_k_ << std::endl;
            }
        });

    frame_gen_master.set_output_callback([&](Metavision::timestamp, cv::Mat &frame) {
        flow_m_copy.clear();flow_m_copy=grouped_flow_m;
        cv::Mat frame_sphere;
        frame.copyTo(frame_sphere);
    if(flow_m_copy.size()!=0){
        draw_flow_origin(flow_m_copy.cbegin(), flow_m_copy.cend(), frame_sphere);
        }
    event_analyzer_master.set_frame(frame_sphere);
    window_master.show(frame);
    nFrameMaster++;});

    CloudManager cloud_m;
    // Unit-sphere thread: projects both accumulated frames onto the sphere point cloud
    std::thread sphere_processing([&]() {
        cv::Mat frame_slave, frame_master;
        while (!stop_decoding_slave && !stop_application_slave) {
         if(nFrameMaster > 0 && nFrameSlave > 0) {
            frame_slave = event_analyzer_slave.get_frame();
            frame_master = event_analyzer_master.get_frame();
            cloud_m.cloud = sphere.SphereFromAccMapFast(frame_master,frame_slave);
            nCloud++;
         }
        }
    });

    // Visualization thread: sphere point cloud + Kent estimate (mode, major and minor axes)
    std::thread visualization([&]() {
        bool cloud_initialized = false;
        bool vmf_cloud_initialized = false;
        std::vector<pcl::visualization::Camera> cam;
        int dumpFrameIdx = 0;
        while (!stop_decoding_slave && !stop_application_slave) {
         if(nCloud>0) {
            viewer->getCameras(cam);
            pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_sphere;
            cloud_sphere = cloud_m.cloud;
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud_sphere);
            if(!cloud_initialized){
                viewer->addPointCloud<pcl::PointXYZRGB> (cloud_sphere,rgb,"Unit sphere");
                cloud_initialized=true;
            }
            else{
                viewer->updatePointCloud<pcl::PointXYZRGB>(cloud_sphere,rgb,"Unit sphere");
            }
            if(ke.initialized_){
                pcl::PointXYZ pt_u;
                pt_u.x=ke.est_u_.x;
                pt_u.y=ke.est_u_.y;
                pt_u.z=ke.est_u_.z;

                double axis_length = 0.2;

                pcl::PointXYZ pt_x;
                pt_x.x = pt_u.x + axis_length * ke.est_xi_1.x;
                pt_x.y = pt_u.y + axis_length * ke.est_xi_1.y;
                pt_x.z = pt_u.z + axis_length * ke.est_xi_1.z;

                pcl::PointXYZ pt_y;
                pt_y.x = pt_u.x + axis_length*0.4 * ke.est_xi_2.x;
                pt_y.y = pt_u.y + axis_length*0.4 * ke.est_xi_2.y;
                pt_y.z = pt_u.z + axis_length*0.4 * ke.est_xi_2.z;

                pcl::PointXYZ pt_u_scaled(pt_u.x*(axis_length+0.05), pt_u.y*(axis_length+0.05), pt_u.z*(axis_length+0.05));

                // cyan: mode/bearing, magenta: major axis xi1 (heading), yellow: minor axis xi2
                if(!vmf_cloud_initialized){
                viewer->addSphere(pt_u, 0.01, 0, 255, 255,"u");
                viewer->addArrow(pt_u_scaled, pcl::PointXYZ(0,0,0), 0, 255, 255, false, "Bearing");
                viewer->addArrow(pt_x, pt_u, 255, 0, 255 , false, "axis_x");
                viewer->addLine(pt_y, pt_u, 255 , 255, 0, "line_y");
                vmf_cloud_initialized = true;
                }
                else{
                viewer->removeShape("axis_x");
                viewer->removeShape("axis_y");
                viewer->removeShape("Bearing");
                viewer->removeShape("line_y");
                viewer->updateSphere(pt_u,0.01, 0, 255 ,255,"u");
                viewer->addArrow(pt_u_scaled, pcl::PointXYZ(0,0,0), 0, 255, 255, false, "Bearing");
                viewer->addArrow(pt_x, pt_u, 255, 0, 255 , false, "axis_x");
                viewer->addLine(pt_y, pt_u, 255 , 255, 0, "line_y");
                }
            }
            viewer->spinOnce(20);
            if (!p_dump_dir.empty()) {
                char fn[600];
                snprintf(fn, sizeof(fn), "%s/frame_%05d.png", p_dump_dir.c_str(), dumpFrameIdx++);
                viewer->saveScreenshot(fn);
            }
            std::this_thread::sleep_for(20ms);
         }
        }
    });

    i_eventsstream_slave->start();
    i_eventsstream_master->start();

    /******Main loop: UI events + logging********/
    auto start = std::chrono::high_resolution_clock::now();
    while(!stop_application_master && !stop_application_slave){

        static constexpr std::int64_t kSleepPeriodMs = (1/100.0)*1000;
        Metavision::EventLoop::poll_and_dispatch(kSleepPeriodMs);
        if(FirstInit == true && clockStarted ==false){
            start = std::chrono::high_resolution_clock::now();
            clockStarted = true;
            flog<<"t u_x u_y u_z xi1_x xi1_y xi1_z k beta nInit useHeadingAssoc headingDisc assocDiverged assocDivergeCount headingRejectCount bearingW headingW assocDb assocDh assocCost\n";
        }
        if(clockStarted == true){
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        double t  = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()/1000.0;
        flog<<t<<" "<<ke.est_u_.x<<" "<<ke.est_u_.y<<" "<<ke.est_u_.z<<" "<<ke.est_xi_1.x<<" "<<ke.est_xi_1.y<<" "<<ke.est_xi_1.z<<" "<<ke.est_k_<<" "<<ke.est_beta_<<" "<<ke.nInit<<" "<<(ke.use_heading_assoc_?1:0)<<" "<<ke.lastHeadingDisc_<<" "<<(ke.lastAssocDiverged_?1:0)<<" "<<ke.assocDivergeCount_<<" "<<ke.headingRejectCount_<<" "<<ke.lastBearingW_<<" "<<ke.lastHeadingW_<<" "<<ke.lastAssocDb_<<" "<<ke.lastAssocDh_<<" "<<ke.lastAssocCost_<<"\n";
        flog.flush();
        // A_sum vs A_opt metrics: one row per Kent update, on the same wall-clock t as flog
        if (p_alog && faerr.is_open()) {
            static long lastA = -1;
            if (ke.aUpdateCount_ != lastA) {
                lastA = ke.aUpdateCount_;
                faerr << t << " " << ke.aErrFrob_ << " " << ke.aErrFrobRel_ << " "
                      << ke.aTangResid_ << " " << ke.aBetaSum_ << " " << ke.aBetaOpt_ << " " << ke.nInit << "\n";
                static int afc=0; if (++afc % 200 == 0) faerr.flush();
            }
        }
        // Live monitor: print each association event (a distractor was preferred/rejected), else a heartbeat
        if(ke.lastAssocDiverged_){
            std::cout<<"[assoc] t="<<t<<"s DIVERGED (heading avoided bearing-NN pick) | headingDisc="<<ke.lastHeadingDisc_*180.0/M_PI
                     <<" deg | divergeCount="<<ke.assocDivergeCount_<<" | headingRejects="<<ke.headingRejectCount_
                     <<" | k="<<ke.est_k_<<" beta="<<ke.est_beta_<<std::endl;
        } else {
            std::cout<<"t="<<t<<"s | divergeCount="<<ke.assocDivergeCount_<<" headingRejects="<<ke.headingRejectCount_
                     <<" | k="<<ke.est_k_<<" beta="<<ke.est_beta_<<" nInit="<<ke.nInit<<std::endl;
        }
        }
    }

    decoding_loop_master.join();
    decoding_loop_slave.join();
    sphere_processing.join();
    visualization.join();
    kent_tracking.join();

    return 0;

}
