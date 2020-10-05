
#include <boost/asio/io_service.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>
#include <string.h>
#include <iostream>

void myTask(const char* msg)
{
    //sleep(5);
    std::cerr << "MSG: " << msg << "\n";
}

int main() {
/*
 * Create an asio::io_service and a thread_group (through pool in essence)
 */
    boost::asio::io_service ioService;
    boost::thread_group threadpool;

/*
 * This will start the ioService processing loop. All tasks
 * assigned with ioService.post() will start executing.
 */
    boost::asio::io_service::work work(ioService);

/*
 * This will add 2 threads to the thread pool. (You could just put it in a for loop)
 */
    threadpool.create_thread(
            boost::bind(&boost::asio::io_service::run, &ioService)
    );
    threadpool.create_thread(
            boost::bind(&boost::asio::io_service::run, &ioService)
    );

/*
 * This will assign tasks to the thread pool.
 * More about boost::bind: "http://www.boost.org/doc/libs/1_54_0/libs/bind/bind.html#with_functions"
 */
    ioService.post(boost::bind(myTask, "Hello World!"));
    ioService.post(boost::bind(myTask, "./cache"));
    ioService.post(boost::bind(myTask, "twitter,gmail,facebook,tumblr,reddit"));

    while (ioService.stopped())
    {
        ioService.run();
    }
    ioService.stop();
    threadpool.join_all();


    std::cerr << "STOP\n";
    return 0;
}