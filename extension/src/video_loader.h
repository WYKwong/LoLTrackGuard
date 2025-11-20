#pragma once
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <string>

class AsyncVideoLoader {
public:
    AsyncVideoLoader(std::string path);
    ~AsyncVideoLoader();
    
    // Returns true if a frame was retrieved, false if EOF or error
    bool get_frame(cv::Mat& out_frame);
    void stop();

private:
    void update();
    
    cv::VideoCapture cap;
    std::thread worker;
    std::mutex mtx;
    std::queue<cv::Mat> buffer;
    std::atomic<bool> stopped;
    const size_t max_buffer_size = 30;
};

