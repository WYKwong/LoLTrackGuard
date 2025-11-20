#include "video_loader.h"
#include <chrono>
#include <iostream>

AsyncVideoLoader::AsyncVideoLoader(std::string path) : stopped(false) {
    cap.open(path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file: " << path << std::endl;
        stopped = true;
        return;
    }
    worker = std::thread(&AsyncVideoLoader::update, this);
}

AsyncVideoLoader::~AsyncVideoLoader() {
    stop();
}

void AsyncVideoLoader::stop() {
    if (!stopped) {
        stopped = true;
        if (worker.joinable()) {
            worker.join();
        }
        cap.release();
    }
}

void AsyncVideoLoader::update() {
    while (!stopped) {
        if (buffer.size() < max_buffer_size) {
            cv::Mat frame;
            bool ret = cap.read(frame);
            if (!ret) {
                stopped = true;
                break;
            }
            
            // Lock and push
            {
                std::lock_guard<std::mutex> lock(mtx);
                buffer.push(frame);
            }
        } else {
            // Sleep briefly if buffer is full to save CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

bool AsyncVideoLoader::get_frame(cv::Mat& out_frame) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!buffer.empty()) {
        out_frame = buffer.front();
        buffer.pop();
        return true;
    }
    // If buffer empty and stopped is true, we are done
    return !stopped;
}

