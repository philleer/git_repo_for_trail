#include "util_sys.h"
#include <cstdio>

int is_file_exist(const char *file_path) {
	if (file_path == NULL) {
		return -1;
	}
	if (access(file_path, F_OK) == 0) {
		return 0;
	}
	return -1;
}

int is_dir_exist(const char *dir_path) {
	if (dir_path == NULL) {
		return -1;
	}
	if (opendir(dir_path) == NULL) {
		return -1;
	}
	return 0;
}

Timer::Timer(): started_(false), paused_(false) {}

void Timer::Start() {
	started_ = true;
	paused_ = false;
	start_time_ = SystemTime::now();
}

void Timer::Restart() {
	started_ = false;
	Start();
}

void Timer::Pause() {
	paused_ = true;
	pause_time_ = SystemTime::now();
}

void Timer::Resume() {
	paused_ = false;
	start_time_ += SystemTime::now() - pause_time_;
}

void Timer::Reset() {
	started_ = false;
	paused_ = false;
}

double Timer::ElapsedMicroSeconds() const {
	if (!started_) return 0.0;
	if (paused_) {
		return std::chrono::duration_cast<std::chrono::microseconds>(pause_time_ -
					start_time_).count();
	} else {
		return std::chrono::duration_cast<std::chrono::microseconds>(SystemTime::now() -
					start_time_).count();
	}
}

double Timer::ElapsedSeconds() const { return ElapsedMicroSeconds() / 1e6; }

double Timer::ElapsedMinutes() const { return ElapsedSeconds() / 60; }

double Timer::ElapsedHours() const { return ElapsedMinutes() / 60; }

void Timer::PrintSeconds() const {
	printf("Elaspsed time: %.5f [secons]\n", ElapsedSeconds());
}

void Timer::PrintMinutes() const {
	printf("Elaspsed time: %.3f [minutes]\n", ElapsedMinutes());
}

void Timer::PrintHours() const {
	printf("Elaspsed time: %.3f [hours]\n", ElapsedHours());
}