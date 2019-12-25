#ifndef UTIL_SYS_H
#define UTIL_SYS_H

// Linux file/dir operation
#include <unistd.h>	// access
#include <dirent.h>	// opendir
#include <chrono>	// std::chrono, std::milli

typedef std::chrono::high_resolution_clock SystemTime;
typedef std::chrono::time_point<SystemTime> TimePoint;
typedef std::chrono::duration<double, std::milli> Duration;

/*
 * 返回值为-1表示文件或文件夹不存在，为0表示存在
 */
int is_file_exist(const char *file_path);
int is_dir_exist(const char *dir_path);

class Timer {
public:
	Timer();

	void Start();
	void Restart();
	void Pause();
	void Resume();
	void Reset();

	double ElapsedMicroSeconds() const;
	double ElapsedSeconds() const;
	double ElapsedMinutes() const;
	double ElapsedHours() const;

	void PrintSeconds() const;
	void PrintMinutes() const;
	void PrintHours() const;

private:
	bool started_;
	bool paused_;
	SystemTime::time_point start_time_;
	SystemTime::time_point pause_time_;
};

#endif /* UTIL_SYS_H */
