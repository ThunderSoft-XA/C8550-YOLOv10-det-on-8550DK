#include <YOLOv10s.h>
#include <opencv2/opencv.hpp>
#include <random>

#include <chrono>

long long GetMillisecondTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return millis;
}

int main() {
    cv::Mat img = cv::imread("../imgs/bus.jpg");
    // cv::Mat img = cv::imread("../imgs/frisbee.jpg");
    cv::Mat img2;
    cv::cvtColor(img, img2, cv::COLOR_BGR2RGB);
    ObjectDetection detect;
    ObjectDetectionConfig cfg;
    cfg.model_path = std::string("../models/yolov10s.dlc");
    cfg.runtime = runtime::CPU;
    cfg.inputLayers = {"images"};

    cfg.outputTensors = {"output0"};
    cfg.outputLayers = {"/model.23/Concat_6"};

    detect.Initialize(cfg);
    std::vector<ObjectData> results;
    auto t0 = GetMillisecondTimestamp();
    detect.Detect(img2, results);
    // printf("Dtect cost %lld \n", GetMillisecondTimestamp()-t0);
    
    for (auto i:results) {
        std::cout << "label:" << i.label << "," ;
        std::cout << "x1:" << i.bbox.x << ",";
        std::cout << "y1:" << i.bbox.y << ",";
        std::cout << "width:" << i.bbox.width << ",";
        std::cout << "height:" << i.bbox.height << ",";
        std::cout << "score:" << i.confidence << "," << std::endl;
        cv::putText(img, std::to_string(i.label)+std::string(" : ")+std::to_string(i.confidence), cv::Point(i.bbox.x, i.bbox.y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
        cv::rectangle(img, cv::Rect(i.bbox.x, i.bbox.y, i.bbox.width, i.bbox.height), cv::Scalar(0, 200, 0), 2);
    }
    cv::imwrite("result.jpg", img);
    printf("I Img saved result.jpg\n");

    return 0;
}

 
