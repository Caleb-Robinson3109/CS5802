#include <chrono>

class Timer{
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point stop_time;
public:
    void start();
    void stop();
    long long get_duration_ms();
};

void Timer::start(){
    start_time = std::chrono::high_resolution_clock::now();
}

void Timer::stop(){
    stop_time = std::chrono::high_resolution_clock::now();
}

long long Timer::get_duration_ms(){
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop_time - start_time).count();
}